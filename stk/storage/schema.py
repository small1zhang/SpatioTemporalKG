# -*- coding: utf-8 -*-
"""Neo4j Schema — 节点标签、关系类型、索引与约束 (v3 §6.2)."""
from __future__ import annotations
from typing import List


NODE_LABELS = {
    "Vehicle": "Vehicle", "Pedestrian": "Pedestrian",
    "TrafficLight": "TrafficLight", "RoadElement": "RoadElementEntity",
    "EnvSnapshot": "EnvSnapshot", "SceneSnapshot": "SceneSnapshot",
    "Maneuver": "Maneuver", "Interaction": "Interaction",
    "Rule": "Rule", "Param": "Param",
    "SafetyViolation": "SafetyViolation",
    "Responsibility": "Responsibility", "AttrVersion": "AttrVersion",
}

RELATION_TYPES: List[str] = [
    "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
    "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
    "containsVehicle", "containsPedestrian", "containsTrafficLight",
    "containsRoad", "hasEnvironment", "weather_context",
    "standing_still", "changing_lane", "following", "approaching",
    "yielding_to", "overtaking", "wrong_side_meeting",
    "opposite_direction", "same_direction", "blocked_view",
    "approaching_pedestrian", "approaching_intersection", "crossing",
    "definedBy", "usesParam", "supportedByEvidence",
    "violates", "triggers", "responsibleFor", "causedBy",
    "hasVersion",
]


def get_schema_cypher() -> str:
    c = ""
    constraints = [
        ("vehicle_id_unique","Vehicle","vehicle_id"),
        ("pedestrian_id_unique","Pedestrian","pedestrian_id"),
        ("sv_id_unique","SafetyViolation","sv_id"),
        ("rule_id_unique","Rule","rule_id"),
    ]
    for name, label, prop in constraints:
        c += f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE;\n"
    indexes = [
        ("frame_id_idx","SceneSnapshot","frame_id"),
        ("sv_frame_idx","SafetyViolation","frame_id"),
        ("sv_rule_idx","SafetyViolation","rule_id"),
        ("vehicle_id_idx","Vehicle","vehicle_id"),
    ]
    for name, label, prop in indexes:
        c += f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop});\n"
    return c