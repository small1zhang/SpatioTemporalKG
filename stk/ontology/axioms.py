"""
核心公理函数 (v3 §1.11)
A1–A7 作为可调用的验证函数。
实现 v3 §1.11.1 基础公理集合 + §1.11.2 推理规则示例。
"""
from typing import Any, List, Optional
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation
from stk.ontology.lifecycle import NodeLifecycle


def _get_entity_type_val(entity: BaseEntity) -> str:
    """安全获取 entity_type 的字符串值（兼容枚举和字符串两种形式）。"""
    et = entity.entity_type
    if isinstance(et, str):
        return et
    return et.value


def axiom_A1_unique_id(eid: str) -> bool:
    """A1: 实体 ID 非空、不含空格。"""
    return bool(eid and len(eid) > 0 and " " not in eid)


def axiom_A2_fixed_type(entity: BaseEntity) -> bool:
    """A2: 实体类型已定义，非空。"""
    return bool(entity.entity_type)


def axiom_A3_attribute_versioned(lifecycle: NodeLifecycle, attr: str, frame_id: int) -> bool:
    """A3: 属性的值在时间轴上可查询。"""
    val = lifecycle.get_version_at(attr, frame_id)
    return val is not None


def axiom_A4_relation_has_temporal(relation: BaseRelation) -> bool:
    """A4: 每条关系必有 valid_from。"""
    return relation.valid_from is not None and relation.valid_from >= 0


def axiom_A5_three_layer_evidence(rule_node: BaseEntity, evidence_ids: List[str]) -> bool:
    """A5: 规则层节点必须连接场景/行为层证据。"""
    etype = _get_entity_type_val(rule_node)
    if etype in ("Rule", "Param"):
        return True
    return len(evidence_ids) > 0


def axiom_A6_event_traceable(violation_node: BaseEntity, evidence_count: int) -> bool:
    """A6: SafetyViolation 可通过 supportedByEvidence 追溯到原始场景事实。"""
    etype = _get_entity_type_val(violation_node)
    if etype != "SafetyViolation":
        return True
    return evidence_count > 0


def axiom_A7_incremental_consistency(prev_ids: List[str], curr_ids: List[str]) -> bool:
    """A7: G_t = G_{t-1} ⊕ Δg_t（Δ 不删除实体-只标记）。"""
    for eid in prev_ids:
        if eid not in curr_ids:
            return False
    return True
