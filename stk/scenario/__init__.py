"""场景层: 每帧静态语义快照的建模 (§2)。"""

from .nodes import (
    VehicleEntity, PedestrianEntity, TrafficLightEntity,
    RoadElementEntity, EnvironmentSnapshot, ScenarioSnapshot,
)
from .relations import (
    build_relation, in_lane, on_road, in_junction,
    adjacent_lane, lane_connects, ahead_of, beside,
    nearby_pedestrian, controlled_by,
    contains_vehicle, contains_pedestrian, contains_traffic_light,
    contains_road, has_environment, weather_context,
)
from .spatial import (
    compute_in_lane, compute_ahead_of, compute_beside,
    compute_nearby_pedestrian, compute_adjacent_lanes,
)
from .snapshot_builder import build_snapshot, build_sample_frame, FrameData
from .lifecycle_manager import LifecycleManager

__all__ = [
    "VehicleEntity", "PedestrianEntity", "TrafficLightEntity",
    "RoadElementEntity", "EnvironmentSnapshot", "ScenarioSnapshot",
    "build_relation",
    "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
    "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
    "contains_vehicle", "contains_pedestrian", "contains_traffic_light",
    "contains_road", "has_environment", "weather_context",
    "compute_in_lane", "compute_ahead_of", "compute_beside",
    "compute_nearby_pedestrian", "compute_adjacent_lanes",
    "build_snapshot", "build_sample_frame", "FrameData",
    "LifecycleManager",
]
