"""
命名空间与 ID 生成器 (v3 §1.9)
前缀: veh_, ped_, tl_, road_, man_, sv_, resp_
"""
from typing import Dict, Optional, Set


NAMESPACE_PREFIXES: Dict[str, str] = {
    "veh_": "VehicleEntity", "ped_": "PedestrianEntity",
    "tl_": "TrafficLightEntity", "road_": "RoadElementEntity",
    "man_": "ManeuverNode", "sv_": "SafetyViolation",
    "resp_": "ResponsibilityAssignment",
}
ENTITY_TYPE_TO_PREFIX: Dict[str, str] = {v: k for k, v in NAMESPACE_PREFIXES.items()}


class IDGenerator:
    """实体 ID 生成器 (公理 A1: 全局唯一)。"""

    def __init__(self):
        self._allocated: Set[str] = set()

    def generate(self, entity_type: str, **kwargs) -> str:
        prefix = ENTITY_TYPE_TO_PREFIX.get(entity_type)
        if prefix is None:
            raise ValueError(f"Unknown type: {entity_type}")
        if entity_type == "VehicleEntity":
            eid = f"{prefix}{kwargs['actor_id']}"
        elif entity_type == "PedestrianEntity":
            eid = f"{prefix}{kwargs['actor_id']}"
        elif entity_type == "TrafficLightEntity":
            eid = f"{prefix}{kwargs['actor_id']}"
        elif entity_type == "RoadElementEntity":
            eid = f"{prefix}{kwargs['road_id']}_lane_{kwargs['lane_id']}"
        elif entity_type == "ManeuverNode":
            eid = f"{prefix}{kwargs['veh_id']}_{kwargs['frame_start']}"
        elif entity_type == "SafetyViolation":
            eid = f"{prefix}{kwargs['rule_code']}_{kwargs['frame']}"
        elif entity_type == "ResponsibilityAssignment":
            eid = f"{prefix}{kwargs['sv_id']}_{kwargs['actor_id']}"
        else:
            raise ValueError(f"No ID format for: {entity_type}")
        if eid in self._allocated:
            raise ValueError(f"ID 重复 (公理 A1): {eid}")
        self._allocated.add(eid)
        return eid

    def parse_type(self, eid: str) -> Optional[str]:
        for prefix, etype in NAMESPACE_PREFIXES.items():
            if eid.startswith(prefix):
                return etype
        return None

    def reset(self):
        self._allocated.clear()


GLOBAL_ID_GENERATOR = IDGenerator()
