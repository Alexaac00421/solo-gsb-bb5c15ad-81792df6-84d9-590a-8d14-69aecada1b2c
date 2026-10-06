"""投递判定的单笔 / 批量受理共享逻辑。

把 POST /deliveries 与 POST /deliveries/batch 共用的四件事从路由层抽出：

1. 地址规范化（去空白、转小写）与批量项预校验（无效地址、批内重复编号）；
2. 既有编号冲突检查（同编号绑定了其他规范化地址）；
3. 受理时策略快照：新判定共用同一份策略修订号 / 比例 / 基线候选版本，
   分别稳定分桶，规则集按版本缓存；
4. 路由持久化：成功存完整终点路径，拒收 / 无路由 / 成环同样作为判定保存，
   且新判定全部写入后候选侧熔断只评估一次（重放不计数）。

本模块不依赖 FastAPI：HTTP 状态码与响应字段仍由 main.py 按既有契约映射，
这里只抛出语义化异常，以保持现有状态码、错误码与响应体形状完全兼容。
"""
from __future__ import annotations

import sqlite3

from . import store
from .routing import ResolutionError, RuleSet, is_valid_address, resolve
from .traffic import select_version


# ---------- 受理期语义化异常（路由层映射为既有 HTTP 错误响应） ----------

class InvalidAddresses(Exception):
    """批量中存在一个或多个无效地址（整批拒绝）。invalid_items 附全部无效项。"""

    def __init__(self, invalid_items: list[dict]):
        super().__init__(f"{len(invalid_items)} invalid address(es) in batch")
        self.invalid_items = invalid_items


class DuplicateDeliveryIds(Exception):
    """批内出现重复投递编号（即使地址相同也整批拒绝）。"""

    def __init__(self, delivery_ids: list[str]):
        super().__init__(f"duplicate delivery id(s) in batch: {', '.join(delivery_ids)}")
        self.delivery_ids = delivery_ids


class DeliveryIdConflicts(Exception):
    """既有编号已绑定其他规范化地址。conflicts 附全部冲突项（单笔至多一项）。"""

    def __init__(self, conflicts: list[dict]):
        super().__init__(f"{len(conflicts)} delivery id(s) already submitted with a different address")
        self.conflicts = conflicts


class PolicyNotConfigured(Exception):
    """存在新编号但尚未配置默认投递策略；重放既有判定不需要策略，不会抛此异常。"""


class PublishedVersionMissing(Exception):
    """防御：策略引用的版本在库中不存在（写入时外键保证不会发生）。"""

    def __init__(self, version: int):
        super().__init__(f"published version not found: {version}")
        self.version = version


# ---------- 1. 地址规范化与批量预校验 ----------

def normalize_address(raw: str) -> str | None:
    """地址规范化：去首尾空白、转小写；非法地址返回 None（便于批量收集全部无效项）。"""
    address = raw.strip().lower()
    return address if is_valid_address(address) else None


def prepare_batch(raw_items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """批量预校验，保持输入顺序返回 [(delivery_id, 规范化地址), ...]。

    - 任一地址无效 → InvalidAddresses（附 {index, delivery_id, address} 全部无效项）；
    - 规范化后编号在批内重复 → DuplicateDeliveryIds（附去重且排序后的编号）。
    两类拒绝都发生在进入写事务之前，整批不写入、不占用任何编号。
    """
    normalized: list[tuple[str, str | None]] = []
    invalid_items: list[dict] = []
    for index, (delivery_id, raw_address) in enumerate(raw_items):
        address = normalize_address(raw_address)
        if address is None:
            invalid_items.append({"index": index, "delivery_id": delivery_id, "address": raw_address})
        normalized.append((delivery_id, address))
    if invalid_items:
        raise InvalidAddresses(invalid_items)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for delivery_id, _ in normalized:
        if delivery_id in seen:
            duplicates.add(delivery_id)
        seen.add(delivery_id)
    if duplicates:
        raise DuplicateDeliveryIds(sorted(duplicates))

    # 到此处所有地址都已规范化成功（非 None）
    return [(delivery_id, address) for delivery_id, address in normalized]  # type: ignore[misc]


# ---------- 3/4. 策略快照 + 分桶解析 + 判定持久化（单笔与批量共用） ----------

def endpoints_doc(endpoints: dict[str, list[tuple[str, ...]]]) -> list[dict]:
    """把解析结果整理为持久化 / 响应共用的终点文档（终点与路径均排序，形状同 /resolve）。"""
    return [
        {"mailbox": mailbox, "paths": sorted(paths)}
        for mailbox, paths in sorted(endpoints.items())
    ]


def persist_new_decision(
    conn: sqlite3.Connection,
    *,
    delivery_id: str,
    address: str,
    policy: dict,
    rulesets: dict[int, RuleSet],
) -> dict:
    """按受理时策略快照为一个新编号做分桶、解析并持久化判定。须在写事务内调用。

    同批多笔共用同一份 policy（同修订号、同比例）；规则集按选中版本缓存，
    命中同一版本时只加载一次。拒收 / 无路由 / 成环整次失败且不切换版本，
    失败详情（含 delivery_id、策略快照）随判定一并持久化，供原样重放。
    """
    chosen = select_version(address, policy["candidate_percent"])
    version = policy["baseline_version"] if chosen == "baseline" else policy["candidate_version"]
    ruleset = rulesets.get(version)
    if ruleset is None:
        ruleset = store.load_ruleset(conn, version)
        if ruleset is None:  # 防御：策略引用的版本必然已发布且不可变
            raise PublishedVersionMissing(version)
        rulesets[version] = ruleset
    try:
        endpoints = resolve(address, ruleset)
    except ResolutionError as exc:  # 选中版本失败：整次失败并持久化，不切换版本
        detail = exc.detail()
        detail.update({
            "delivery_id": delivery_id,
            "policy_revision": policy["revision"],
            "version": version,
            "selection": chosen,
        })
        return store.save_decision(
            conn, delivery_id=delivery_id, address=address, policy=policy,
            selection=chosen, version=version, status="failed", endpoints=None, error=detail,
        )
    return store.save_decision(
        conn, delivery_id=delivery_id, address=address, policy=policy,
        selection=chosen, version=version, status="ok",
        endpoints=endpoints_doc(endpoints), error=None,
    )


# ---------- 单笔受理 ----------

def accept_one(conn: sqlite3.Connection, delivery_id: str, address: str) -> dict:
    """在写事务内受理单笔投递判定。

    已有同编号判定：地址一致 → 原样重放（含原失败状态，不评估熔断）；
    地址不一致 → DeliveryIdConflicts。新编号取受理时策略快照（未配置 →
    PolicyNotConfigured，不占用编号），分桶解析并持久化后评估一次熔断。
    """
    decision = store.get_decision(conn, delivery_id)
    if decision is not None:
        if decision["address"] != address:
            raise DeliveryIdConflicts([{
                "delivery_id": delivery_id,
                "submitted_address": address,
                "stored_address": decision["address"],
            }])
        return decision
    policy = store.get_policy(conn)
    if policy is None:  # 回滚：未配置策略不占用编号
        raise PolicyNotConfigured()
    decision = persist_new_decision(
        conn, delivery_id=delivery_id, address=address, policy=policy, rulesets={}
    )
    # 首次判定已持久化：评估一次受理时修订号的候选侧熔断；
    # 触发也只影响后续投递，本次返回上面已持久化的原判定，历史判定不改写。
    store.maybe_trigger_circuit_breaker(conn, policy)
    return decision


# ---------- 批量受理 ----------

def accept_batch(conn: sqlite3.Connection, items: list[tuple[str, str]]) -> list[dict]:
    """在同一写事务内受理一批已通过预校验（地址有效、批内无重复编号）的投递项。

    既有同编号同地址判定原样重放（保持首次策略修订号与结果，不进入熔断统计）；
    同编号绑定其他规范化地址 → DeliveryIdConflicts（附全部冲突项，整批不写入）。
    只要存在新编号就必须有默认策略，否则 PolicyNotConfigured（整批回滚）；
    全部为重放时不需要策略。新判定共用受理时同一份策略快照、按版本缓存规则集，
    全部持久化完成后熔断仅评估一次。返回与输入顺序一致的判定列表。
    """
    existing = store.get_decisions(conn, [delivery_id for delivery_id, _ in items])
    conflicts = [
        {
            "delivery_id": delivery_id,
            "submitted_address": address,
            "stored_address": existing[delivery_id]["address"],
        }
        for delivery_id, address in items
        if delivery_id in existing and existing[delivery_id]["address"] != address
    ]
    if conflicts:  # 既有编号绑定其他规范化地址：整批拒绝，不写入
        raise DeliveryIdConflicts(conflicts)

    decisions: dict[str, dict] = dict(existing)
    new_items = [(delivery_id, address) for delivery_id, address in items if delivery_id not in existing]
    if new_items:
        policy = store.get_policy(conn)
        if policy is None:  # 回滚：尚无默认策略且存在新编号，整批拒绝
            raise PolicyNotConfigured()
        rulesets: dict[int, RuleSet] = {}  # 同批共用受理时策略：规则集按版本缓存
        for delivery_id, address in new_items:
            decisions[delivery_id] = persist_new_decision(
                conn, delivery_id=delivery_id, address=address, policy=policy, rulesets=rulesets
            )
        # 新判定全部写入后，仅评估一次受理时修订号的候选侧熔断：
        # 触发只影响后续请求；本批响应仍为已持久化的原判定，重放不进入统计。
        store.maybe_trigger_circuit_breaker(conn, policy)
    return [decisions[delivery_id] for delivery_id, _ in items]
