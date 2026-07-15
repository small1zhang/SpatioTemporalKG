"""
规则层关系定义与生成 (v3 sec 4.5 + sec 4.18)

覆盖 7 种 RuleRelationType:
  definedBy           - SafetyViolation -> RuleDefinition (v3 sec 4.3.3)
  usesParam           - RuleDefinition -> RuleParameter (v3 sec 4.9.1)
  supportedByEvidence - SafetyViolation -> 场景层/行为层关系/实体 (v3 sec 4.3.3)
  violates            - 违规实体 -> 另一个实体 (v3 sec 4.3.2)
  triggers             - SafetyViolation -> SafetyViolation (v3 sec 4.18.2 因果链)
  responsibleFor      - ResponsibilityAssignment -> SafetyViolation (v3 sec 4.18.1)
  causedBy            - SafetyViolation -> SafetyViolation (v3 sec 4.18.2)

每条关系通过 build_relation() 创建 BaseRelation 实例。
"""
from typing import Any, Dict, List, Optional

from stk.ontology.relation import BaseRelation
from stk.ontology.types import RuleRelationType


def build_relation(src_id, dst_id, rel_type, frame_id, valid_from,
                   valid_to=None, extra_attrs=None, confidence=1.0):
    """构建一条规则层关系."""
    attrs = {"frame_id": frame_id, "confidence": confidence}
    if extra_attrs:
        attrs.update(extra_attrs)
    return BaseRelation(
        src_id=src_id, dst_id=dst_id,
        relation_type=rel_type.value,
        frame_id=frame_id,
        valid_from=valid_from, valid_to=valid_to,
        attrs=attrs, confidence=confidence,
    )


def defined_by(sv_id, rule_id, frame_id, valid_from, valid_to=None):
    """SafetyViolation -[definedBy]-> RuleDefinition

    v3 sec 4.3.3: 违规节点归属到规则定义。
    """
    return build_relation(
        src_id=sv_id, dst_id=rule_id,
        rel_type=RuleRelationType.DEFINED_BY,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    )


def uses_param(rule_id, param_id, frame_id, valid_from, valid_to=None):
    """RuleDefinition -[usesParam]-> RuleParameter

    v3 sec 4.9.1: 规则使用参数。
    """
    return build_relation(
        src_id=rule_id, dst_id=param_id,
        rel_type=RuleRelationType.USES_PARAM,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
    )


def supported_by_evidence(sv_id, evidence_id, frame_id, valid_from,
                           evidence_idx=0, valid_to=None):
    """SafetyViolation -[supportedByEvidence]-> 证据节点

    v3 sec 4.3.3: 违规证据链。
    边属性: evidence_idx (证据序号, 从 0 开始)
    """
    return build_relation(
        src_id=sv_id, dst_id=evidence_id,
        rel_type=RuleRelationType.SUPPORTED_BY_EVIDENCE,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
        extra_attrs={"evidence_idx": evidence_idx},
    )


def violates(src_entity_id, dst_entity_id, frame_id, valid_from,
             rule_code="", predicate="", sv_id="", severity=0.0, valid_to=None):
    """EntityA -[violates]-> EntityB

    v3 sec 4.3.2: 违规边。
    边属性: rule_code, predicate, sv_id, frame_id, severity
    """
    return build_relation(
        src_id=src_entity_id, dst_id=dst_entity_id,
        rel_type=RuleRelationType.VIOLATES,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
        extra_attrs={
            "rule_code": rule_code,
            "predicate": predicate,
            "sv_id": sv_id,
            "severity": min(1.0, max(0.0, severity)),
        },
    )


def triggers(src_sv_id, dst_sv_id, frame_id, valid_from,
             reason="", valid_to=None):
    """SafetyViolation -[triggers]-> SafetyViolation

    因果链: sv_A 触发了 sv_B.
    v3 sec 4.18.2.
    """
    return build_relation(
        src_id=src_sv_id, dst_id=dst_sv_id,
        rel_type=RuleRelationType.TRIGGERS,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
        extra_attrs={"reason": reason} if reason else None,
    )


def responsible_for(resp_id, sv_id, frame_id, valid_from,
                     reason="", valid_to=None):
    """ResponsibilityAssignment -[responsibleFor]-> SafetyViolation

    v3 sec 4.18.1: 责任归因边。
    """
    return build_relation(
        src_id=resp_id, dst_id=sv_id,
        rel_type=RuleRelationType.RESPONSIBLE_FOR,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
        extra_attrs={"reason": reason} if reason else None,
    )


def caused_by(sv_a_id, sv_b_id, frame_id, valid_from,
              reason="", valid_to=None):
    """SafetyViolation -[causedBy]-> SafetyViolation

    因果链: sv_A 被 sv_B 导致.
    v3 sec 4.18.2.
    """
    return build_relation(
        src_id=sv_a_id, dst_id=sv_b_id,
        rel_type=RuleRelationType.CAUSED_BY,
        frame_id=frame_id, valid_from=valid_from, valid_to=valid_to,
        extra_attrs={"reason": reason} if reason else None,
    )
