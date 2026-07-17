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
# 场景库 (v3 §2 附录 A2): 14 个预置复杂场景
from .scenario_library import (
    list_scenarios, get_scenario, get_scenario_meta, all_scenarios,
    total_frames, SCENARIO_REGISTRY, SCENARIO_FACTORIES,
    make_S00_baseline_following,
    make_S01_normal_signalized_intersection,
    make_S02_pedestrian_far_avoidance,
    make_S10_pedestrian_sudden_crossing,
    make_S11_unprotected_left_turn_conflict,
    make_S12_red_light_running,
    make_S13_too_close_following,
    make_S20_merging_conflict,
    make_S21_three_way_unsignalized,
    make_S22_emergency_vehicle_yielding,
    make_S30_night_pedestrian_sudden,
    make_S31_rainy_lane_change_blind,
    make_S32_construction_detour,
    make_S33_glare_multi_pedestrian,
)

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
    # 场景库
    "list_scenarios", "get_scenario", "get_scenario_meta", "all_scenarios",
    "total_frames", "SCENARIO_REGISTRY", "SCENARIO_FACTORIES",
    "make_S00_baseline_following",
    "make_S01_normal_signalized_intersection",
    "make_S02_pedestrian_far_avoidance",
    "make_S10_pedestrian_sudden_crossing",
    "make_S11_unprotected_left_turn_conflict",
    "make_S12_red_light_running",
    "make_S13_too_close_following",
    "make_S20_merging_conflict",
    "make_S21_three_way_unsignalized",
    "make_S22_emergency_vehicle_yielding",
    "make_S30_night_pedestrian_sudden",
    "make_S31_rainy_lane_change_blind",
    "make_S32_construction_detour",
    "make_S33_glare_multi_pedestrian",
]
