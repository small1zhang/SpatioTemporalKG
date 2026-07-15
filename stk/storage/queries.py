# -*- coding: utf-8 -*-
"""子图查询 Cypher 模板 (v3 §6.3)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def time_slice_query(frame_id: int) -> str:
    return f"MATCH (v:Vehicle)-[r:in_lane]->(road:RoadElementEntity) WHERE r.frame_id = {frame_id} RETURN v, r, road"


def lifecycle_query(vehicle_id: str) -> str:
    return f"""MATCH (v:Vehicle {{vehicle_id: '{vehicle_id}'}})
MATCH (v)-[r:in_lane]->(road)
MATCH (v)-[f:following]->(w)
WHERE r.frame_id >= v.valid_from AND coalesce(v.valid_to, 999999999) >= r.frame_id
  AND f.frame_id >= v.valid_from AND coalesce(v.valid_to, 999999999) >= f.frame_id
RETURN v, r, road, f, w ORDER BY r.frame_id"""


def anomaly_trace_query(sv_id: str) -> str:
    return f"""MATCH (sv:SafetyViolation {{sv_id: '{sv_id}'}})
MATCH (sv)-[:definedBy]->(rule:Rule)
OPTIONAL MATCH (sv)-[:supportedByEvidence]->(evidence)
OPTIONAL MATCH (sv)<-[:responsibleFor]-(ra:Responsibility)
RETURN sv, rule, evidence, ra"""


def spatiotemporal_aggregate_query(frame_start: int, frame_end: int) -> str:
    return f"""MATCH (sv:SafetyViolation)
WHERE sv.frame_id >= {frame_start} AND sv.frame_id <= {frame_end}
  AND sv.rule_layer IN ['RSS', 'TrafficLaw']
RETURN sv.rule_id, count(*) as count, avg(sv.severity) as avg_sev ORDER BY avg_sev DESC"""


def spatiotemporal_subgraph_query(frame_start: int, frame_end: int, road_id: int = 5) -> str:
    return f"""MATCH (center:RoadElementEntity {{road_id: {road_id}}})
MATCH (center)<-[r:in_lane]-(v:Vehicle)
WHERE r.frame_id >= {frame_start} AND r.frame_id <= {frame_end}
WITH v, r, center
OPTIONAL MATCH (v)-[b:following]->(w:Vehicle) WHERE b.frame_id >= {frame_start} AND b.frame_id <= {frame_end}
OPTIONAL MATCH (v)-[sv:violates]->(w2) WHERE sv.frame_id >= {frame_start} AND sv.frame_id <= {frame_end}
RETURN v, b, w, sv, w2, r, center"""


def export_for_gnn_cypher(frame_start: int, frame_end: int, road_id: int = 5) -> str:
    return f"""MATCH (center:RoadElementEntity {{road_id: {road_id}}})
MATCH (center)<-[r:in_lane]-(v:Vehicle)
WHERE r.frame_id >= {frame_start} AND r.frame_id <= {frame_end}
OPTIONAL MATCH (v)-[b:following]->(w:Vehicle) WHERE b.frame_id >= {frame_start} AND b.frame_id <= {frame_end}
OPTIONAL MATCH (v)-[sv:violates]->() WHERE sv.frame_id >= {frame_start} AND sv.frame_id <= {frame_end}
RETURN v, b, w, sv, r, center"""


def temporal_attr_query(entity_id: str, t_start: int, t_end: int) -> str:
    return f"""MATCH (e {{entity_id: '{entity_id}'}})-[:hasVersion]->(av:AttrVersion)
WHERE av.valid_from_frame <= {t_end} AND coalesce(av.valid_to_frame, 999999999) >= {t_start}
RETURN av ORDER BY av.valid_from_frame"""