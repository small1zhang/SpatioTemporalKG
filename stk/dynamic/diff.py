# -*- coding: utf-8 -*-
"""
Δg_t = (delta_entities, delta_attrs, delta_relations, rule_events)
(v3 §5.3)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DiffSet:
    """通用的增/删/保持 三集合."""
    added: Set[Any] = field(default_factory=set)
    removed: Set[Any] = field(default_factory=set)
    unchanged: Set[Any] = field(default_factory=set)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.removed) + len(self.unchanged)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed)

    def __repr__(self) -> str:
        return f"DiffSet(+{len(self.added)}/-{len(self.removed)})"


@dataclass
class DeltaGraph:
    delta_entities: DiffSet = field(default_factory=DiffSet)
    delta_attrs: Dict[str, Dict[str, Tuple[Any, Any]]] = field(default_factory=dict)
    delta_relations: DiffSet = field(default_factory=DiffSet)
    rule_events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (not self.delta_entities and not self.delta_attrs and
                not self.delta_relations and not self.rule_events)

    def __repr__(self):
        return (f"Δg_t(entities={self.delta_entities}, "
                f"attrs={len(self.delta_attrs)}eids, "
                f"relations={self.delta_relations}, "
                f"events={len(self.rule_events)})")


def _entity_key(e: dict) -> Tuple[str, str]:
    return (e.get("entity_id", ""), e.get("entity_type", ""))


def _relation_key(r: dict) -> Tuple[str, str, str, int]:
    rtype = r.get("relation_type") or r.get("type", "")
    return (r.get("src_id", ""), r.get("dst_id", ""), rtype, r.get("frame_id", 0))


def compute_delta_entities(curr: List[dict], prev: List[dict]) -> DiffSet:
    ck = {_entity_key(e) for e in curr}
    pk = {_entity_key(e) for e in prev}
    return DiffSet(added=ck - pk, removed=pk - ck, unchanged=ck & pk)


def compute_delta_relations(curr: List[dict], prev: List[dict]) -> DiffSet:
    ck = {_relation_key(r) for r in curr}
    pk = {_relation_key(r) for r in prev}
    return DiffSet(added=ck - pk, removed=pk - ck, unchanged=ck & pk)


def compute_delta(
    curr_snapshot: dict, prev_snapshot: Optional[dict] = None,
    threshold: float = 0.01,
) -> DeltaGraph:
    """计算两帧之间的 Δg_t.

    Args:
        curr_snapshot: dict, should contain keys 'vehicles','pedestrians','traffic_lights',
                       'scene_rels','behavior_rels' each as list of dict.
        prev_snapshot: 上一帧 dict (None 表示首帧, 全部 added).
        threshold: 浮点属性防抖(属性变化<阈值不算变化).

    Returns: DeltaGraph
    """
    if prev_snapshot is None:
        all_curr = _collect_entities(curr_snapshot)
        all_rel = curr_snapshot.get("scene_rels", []) + curr_snapshot.get("behavior_rels", [])
        return DeltaGraph(
            delta_entities=DiffSet(added={_entity_key(e) for e in all_curr}),
            delta_relations=DiffSet(added={_relation_key(r) for r in all_rel}),
            rule_events=curr_snapshot.get("rule_events", []),
        )

    dg = DeltaGraph()
    all_curr = _collect_entities(curr_snapshot)
    all_prev = _collect_entities(prev_snapshot)
    dg.delta_entities = compute_delta_entities(all_curr, all_prev)
    dg.delta_attrs = _compute_attrs(all_curr, all_prev, threshold)
    cur_rels = curr_snapshot.get("scene_rels", []) + curr_snapshot.get("behavior_rels", [])
    pre_rels = prev_snapshot.get("scene_rels", []) + prev_snapshot.get("behavior_rels", [])
    dg.delta_relations = compute_delta_relations(cur_rels, pre_rels)
    dg.rule_events = curr_snapshot.get("rule_events", [])
    return dg


def _collect_entities(s: dict) -> List[dict]:
    rv = []
    rv.extend(s.get("vehicles", []))
    rv.extend(s.get("pedestrians", []))
    rv.extend(s.get("traffic_lights", []))
    rv.extend(s.get("road_elements", []))
    return rv


def _compute_attrs(curr: List[dict], prev: List[dict],
                   threshold: float) -> Dict[str, Dict[str, Tuple]]:
    cm = {_entity_key(e): e for e in curr}
    pm = {_entity_key(e): e for e in prev}
    result: Dict[str, Dict[str, Tuple]] = {}
    skip = {"entity_id", "entity_type", "valid_from", "valid_to", "labels", "confidence"}
    for key, ce in cm.items():
        eid = key[0]
        pe = pm.get(key)
        if pe is None:
            continue
        delta = {}
        for k in set(list(ce.keys()) + list(pe.keys())):
            if k in skip:
                continue
            cv = ce.get(k)
            pv = pe.get(k)
            if cv == pv:
                continue
            if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                if abs(float(cv) - float(pv)) < threshold:
                    continue
            delta[k] = (pv, cv)
        if delta:
            result[eid] = delta
    return result