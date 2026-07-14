"""
场景关系定义与生成 (v3 §2.7-§2.11)

覆盖 5 类共 15 种 SceneRelationType:
  拓扑: in_lane, on_road, in_junction, adjacent_lane, lane_connects
  空间: ahead_of, beside, nearby_pedestrian
  控制: controlled_by
  帧聚合: containsVehicle, containsPedestrian, containsTrafficLight, containsRoad, hasEnvironment
  语境: weather_context

每条关系通过 build_relation() 函数创建 BaseRelation 实例。
空间/控制关系计算函数在 spatial.py 中实现。
"""

from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.relation import BaseRelation
from stk.ontology.types import SceneRelationType

# 关系通用属性 (v3 §2.10)
COMMON_ATTRS = {
    "confidence": 1.0,
    "source": "carla_ground_truth",
    "valid": True,
}


def build_relation(
    src_id: str,
    dst_id: str,
    rel_type: SceneRelationType,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
) -> BaseRelation:
    """构建一条场景层关系。

    Args:
        src_id: 源实体 ID
        dst_id: 目标实体 ID
        rel_type: 关系类型 (SceneRelationType 枚举)
        frame_id: 帧号
        valid_from: 有效起始帧
        valid_to: 有效截止帧（None=持续有效）
        extra_attrs: 关系特定属性（如 distance_to_lane_center）
        confidence: 置信度（CARLA 真值默认 1.0）

    Returns:
        BaseRelation 实例
    """
    attrs: Dict[str, Any] = {
        "frame_id": frame_id,
        "confidence": confidence,
        "valid": True,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return BaseRelation(
        src_id=src_id, dst_id=dst_id,
        relation_type=rel_type.value,
        frame_id=frame_id,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs=attrs,
        confidence=confidence,
    )


# ========== 拓扑关系工厂函数 (v3 §2.9.1) ==========


def in_lane(entity_id: str, lane_id: str, frame_id: int, valid_from: int,
            distance_to_lane_center: float = 0.0) -> BaseRelation:
    """InLane: ActorEntity → RoadElementEntity, 实体在当前车道内。"""
    return build_relation(
        src_id=entity_id, dst_id=lane_id,
        rel_type=SceneRelationType.IN_LANE,
        frame_id=frame_id, valid_from=valid_from,
        extra_attrs={"distance_to_lane_center": distance_to_lane_center},
    )


def on_road(entity_id: str, road_id: str, frame_id: int, valid_from: int) -> BaseRelation:
    """OnRoad: VehicleEntity → RoadElementEntity, 车辆在路段上。"""
    return build_relation(
        src_id=entity_id, dst_id=road_id,
        rel_type=SceneRelationType.ON_ROAD,
        frame_id=frame_id, valid_from=valid_from,
    )


def in_junction(entity_id: str, junction_id: str, frame_id: int, valid_from: int) -> BaseRelation:
    """InJunction: VehicleEntity → RoadElementEntity, 车辆在路口内。"""
    return build_relation(
        src_id=entity_id, dst_id=junction_id,
        rel_type=SceneRelationType.IN_JUNCTION,
        frame_id=frame_id, valid_from=valid_from,
    )


def adjacent_lane(lane_a_id: str, lane_b_id: str, frame_id: int) -> BaseRelation:
    """AdjacentLane: Lane → Lane, 相邻车道。"""
    return build_relation(
        src_id=lane_a_id, dst_id=lane_b_id,
        rel_type=SceneRelationType.ADJACENT_LANE,
        frame_id=frame_id, valid_from=frame_id,
    )


def lane_connects(from_lane_id: str, to_lane_id: str, frame_id: int) -> BaseRelation:
    """LaneConnects: Lane → Lane, 道路连通（前后续）。"""
    return build_relation(
        src_id=from_lane_id, dst_id=to_lane_id,
        rel_type=SceneRelationType.LANE_CONNECTS,
        frame_id=frame_id, valid_from=frame_id,
    )


# ========== 空间关系工厂函数 (v3 §2.9.2) ==========


def ahead_of(vehicle_id: str, target_id: str, frame_id: int,
             longitudinal_distance: float, lateral_distance: float = 0.0) -> BaseRelation:
    """AheadOf: 后车 → 前车（同车道或邻车道）。"""
    return build_relation(
        src_id=vehicle_id, dst_id=target_id,
        rel_type=SceneRelationType.AHEAD_OF,
        frame_id=frame_id, valid_from=frame_id,
        extra_attrs={"longitudinal_distance": longitudinal_distance,
                     "lateral_distance": lateral_distance},
    )


def beside(vehicle_id: str, target_id: str, frame_id: int,
           lateral_distance: float, longitudinal_distance: float = 0.0) -> BaseRelation:
    """Beside: 并排。"""
    return build_relation(
        src_id=vehicle_id, dst_id=target_id,
        rel_type=SceneRelationType.BESIDE,
        frame_id=frame_id, valid_from=frame_id,
        extra_attrs={"lateral_distance": lateral_distance,
                     "longitudinal_distance": longitudinal_distance},
    )


def nearby_pedestrian(vehicle_id: str, pedestrian_id: str, frame_id: int,
                      distance: float) -> BaseRelation:
    """NearbyPedestrian: 车辆附近的行人（relative_distance < 20m）。"""
    return build_relation(
        src_id=vehicle_id, dst_id=pedestrian_id,
        rel_type=SceneRelationType.NEARBY_PEDESTRIAN,
        frame_id=frame_id, valid_from=frame_id,
        extra_attrs={"distance": distance},
    )


# ========== 控制关系工厂函数 (v3 §2.9.3) ==========


def controlled_by(lane_id: str, tl_id: str, frame_id: int) -> BaseRelation:
    """ControlledBy: 车道受信号灯控制。"""
    return build_relation(
        src_id=lane_id, dst_id=tl_id,
        rel_type=SceneRelationType.CONTROLLED_BY,
        frame_id=frame_id, valid_from=frame_id,
    )


# ========== 帧聚合关系工厂函数 (v3 §2.9.4) ==========


def contains_vehicle(snapshot_id: str, vehicle_id: str, frame_id: int) -> BaseRelation:
    """ContainsVehicle: 帧包含车辆。"""
    return build_relation(
        src_id=snapshot_id, dst_id=vehicle_id,
        rel_type=SceneRelationType.CONTAINS_VEHICLE,
        frame_id=frame_id, valid_from=frame_id,
    )


def contains_pedestrian(snapshot_id: str, pedestrian_id: str, frame_id: int) -> BaseRelation:
    """ContainsPedestrian: 帧包含行人。"""
    return build_relation(
        src_id=snapshot_id, dst_id=pedestrian_id,
        rel_type=SceneRelationType.CONTAINS_PEDESTRIAN,
        frame_id=frame_id, valid_from=frame_id,
    )


def contains_traffic_light(snapshot_id: str, tl_id: str, frame_id: int) -> BaseRelation:
    """ContainsTrafficLight: 帧包含信号灯。"""
    return build_relation(
        src_id=snapshot_id, dst_id=tl_id,
        rel_type=SceneRelationType.CONTAINS_TRAFFIC_LIGHT,
        frame_id=frame_id, valid_from=frame_id,
    )


def contains_road(snapshot_id: str, road_id: str, frame_id: int) -> BaseRelation:
    """ContainsRoad: 帧包含道路元素。"""
    return build_relation(
        src_id=snapshot_id, dst_id=road_id,
        rel_type=SceneRelationType.CONTAINS_ROAD,
        frame_id=frame_id, valid_from=frame_id,
    )


def has_environment(snapshot_id: str, env_id: str, frame_id: int) -> BaseRelation:
    """HasEnvironment: 帧关联环境快照。"""
    return build_relation(
        src_id=snapshot_id, dst_id=env_id,
        rel_type=SceneRelationType.HAS_ENVIRONMENT,
        frame_id=frame_id, valid_from=frame_id,
    )


# ========== 全局语境关系工厂函数 (v3 §2.9.5) ==========


def weather_context(env_id: str, snapshot_id: str, frame_id: int) -> BaseRelation:
    """WeatherContext: 环境作为帧的全局语境。"""
    return build_relation(
        src_id=env_id, dst_id=snapshot_id,
        rel_type=SceneRelationType.WEATHER_CONTEXT,
        frame_id=frame_id, valid_from=frame_id,
    )
