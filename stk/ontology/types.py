"""
本模块枚举 (v3 §1.3, §1.9, §2.2, §3.2, §4.4)

所有上游模块的类型基础设施。
"""
from enum import Enum


class EntityType(str, Enum):
    """实体类型 (14类)"""
    VEHICLE = "Vehicle"
    PEDESTRIAN = "Pedestrian"
    TRAFFIC_LIGHT = "TrafficLight"
    LANE = "Lane"
    ROAD = "Road"
    JUNCTION = "Junction"
    ENV_SNAPSHOT = "EnvSnapshot"
    SCENE_SNAPSHOT = "SceneSnapshot"
    MANEUVER = "Maneuver"
    INTERACTION_EVENT = "Interaction"
    RULE_DEFINITION = "Rule"
    RULE_PARAMETER = "Param"
    SAFETY_VIOLATION = "SafetyViolation"
    RESPONSIBILITY_ASSIGNMENT = "Responsibility"


class SceneRelationType(str, Enum):
    """场景层关系 (15种, §2.8)"""
    IN_LANE = "in_lane"; ON_ROAD = "on_road"; IN_JUNCTION = "in_junction"
    ADJACENT_LANE = "adjacent_lane"; LANE_CONNECTS = "lane_connects"
    AHEAD_OF = "ahead_of"; BESIDE = "beside"; NEARBY_PEDESTRIAN = "nearby_pedestrian"
    CONTROLLED_BY = "controlled_by"
    CONTAINS_VEHICLE = "containsVehicle"; CONTAINS_PEDESTRIAN = "containsPedestrian"
    CONTAINS_TRAFFIC_LIGHT = "containsTrafficLight"; CONTAINS_ROAD = "containsRoad"
    HAS_ENVIRONMENT = "hasEnvironment"; WEATHER_CONTEXT = "weather_context"


class BehaviorRelationType(str, Enum):
    """行为层关系 (13种, §3.3)"""
    STANDING_STILL = "standing_still"; CHANGING_LANE = "changing_lane"
    FOLLOWING = "following"; APPROACHING = "approaching"; YIELDING_TO = "yielding_to"
    OVERTAKING = "overtaking"; WRONG_SIDE_MEETING = "wrong_side_meeting"
    OPPOSITE_DIRECTION = "opposite_direction"; SAME_DIRECTION = "same_direction"
    BLOCKED_VIEW = "blocked_view"; APPROACHING_PEDESTRIAN = "approaching_pedestrian"
    APPROACHING_INTERSECTION = "approaching_intersection"; CROSSING = "crossing"


class RuleRelationType(str, Enum):
    """规则层关系 (7种, §4.5)"""
    DEFINED_BY = "definedBy"; USES_PARAM = "usesParam"
    SUPPORTED_BY_EVIDENCE = "supportedByEvidence"; VIOLATES = "violates"
    TRIGGERS = "triggers"; RESPONSIBLE_FOR = "responsibleFor"; CAUSED_BY = "causedBy"


class CrossLayerRelationType(str, Enum):
    """跨层桥接 (7种, §3.5)"""
    MANIFESTS_AS = "manifestsAs"; ACTOR = "actor"; SRC = "src"
    DST = "dst"; HAS_VERSION = "hasVersion"
    HAS_MANEUVER = "has_maneuver"; HAS_INTERACTION = "has_interaction"


NODE_LABELS: dict[EntityType, str] = {
    EntityType.VEHICLE: "Vehicle", EntityType.PEDESTRIAN: "Pedestrian",
    EntityType.TRAFFIC_LIGHT: "TrafficLight", EntityType.LANE: "Lane",
    EntityType.ROAD: "Road", EntityType.JUNCTION: "Junction",
    EntityType.ENV_SNAPSHOT: "EnvSnapshot", EntityType.SCENE_SNAPSHOT: "SceneSnapshot",
    EntityType.MANEUVER: "Maneuver", EntityType.INTERACTION_EVENT: "Interaction",
    EntityType.RULE_DEFINITION: "Rule", EntityType.RULE_PARAMETER: "Param",
    EntityType.SAFETY_VIOLATION: "SafetyViolation",
    EntityType.RESPONSIBILITY_ASSIGNMENT: "Responsibility",
}


def entity_type_from_label(label: str) -> EntityType:
    for et, lbl in NODE_LABELS.items():
        if lbl == label:
            return et
    raise ValueError(f"Unknown label: {label}")


def relation_type_from_value(value: str):
    for cls in [SceneRelationType, BehaviorRelationType, RuleRelationType, CrossLayerRelationType]:
        try:
            return cls(value)
        except ValueError:
            continue
    raise ValueError(f"Unknown relation: {value}")
