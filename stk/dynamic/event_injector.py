# -*- coding: utf-8 -*-
"""规则事件反向插入 (v3 §5.4.4)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def inject_violation(graph: dict, violation: dict) -> dict:
    """将 SafetyViolation 反向插入内存图谱.

    Args:
        graph: dict, 含 violations/violation_rels 等列表
        violation: SafetyViolation attrs dict (含 sv_id/rule_code/predicate_str 等)

    Returns:
        graph (直接修改后返回)
    """
    graph.setdefault("violations", []).append(violation)
    # evidence 链
    ev_path = violation.get("evidence_path", [])
    for idx, ev_id in enumerate(ev_path):
        graph.setdefault("evidence_rels", []).append({
            "src": violation.get("sv_id", ""), "dst": ev_id,
            "relation_type": "supportedByEvidence",
            "frame_id": violation.get("frame_id", 0),
            "attrs": {"evidence_idx": idx},
        })
    # violates 边
    graph.setdefault("violation_rels", []).append({
        "src": violation.get("src_id", ""), "dst": violation.get("dst_id", ""),
        "relation_type": "violates",
        "frame_id": violation.get("frame_id", 0),
        "attrs": {
            "rule_code": violation.get("rule_code", ""),
            "sv_id": violation.get("sv_id", ""),
            "severity": violation.get("severity", 0.0),
        },
    })
    # definedBy 边
    graph.setdefault("defined_by_rels", []).append({
        "src": violation.get("sv_id", ""),
        "dst": violation.get("rule_code", ""),
        "relation_type": "definedBy",
        "frame_id": violation.get("frame_id", 0),
        "attrs": {},
    })
    return graph