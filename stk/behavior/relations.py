"""
行为层关系定义与生成 (v3 sec 3.3)

覆盖 13 种 BehaviorRelationType:
  个体: standing_still (self-loop), changing_lane
  交互: following, approaching, yielding_to, overtaking,
        wrong_side_meeting, opposite_direction, same_direction,
        blocked_view, approaching_pedestrian
  演化: approaching_intersection, crossing

每条关系通过 build_relation() 创建 BaseRelation 实例。
所有行为关系携带 source_relations 属性, 记录其所依赖的场景层关系 ID 链 (v3 sec 3.6.2)。
防抖机制 (v3 sec 3.4) 在 debouncer.py 中实现, 与关系本身的创建正交。
"""
from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.relation import BaseRelation
from stk.ontology.types import BehaviorRelationType


def build_relation(
    src_id: str,
    dst_id: str,
    rel_type: BehaviorRelationType,
    frame_id: int,
    valid_from: int,
    valid_to: Optional[int] = None,
    source_relations: Optional[List[str]] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
) -> BaseRelation:
    """构建一条行为层关系.

    Args:
        src_id: 源实体 ID (Vehicle/Pedestrian)
        dst_id: 目标实体 ID (Vehicle/Pedestrian/TrafficLight/RoadElement)
        rel_type: 行为关系类型 (BehaviorRelationType)
        frame_id: 当前观测帧号 (用于建立最早的触发记录)
        valid_from: 关系有效起始帧
        valid_to:   关系有效截止帧 (None 表示行为进行中)
        source_relations: 依赖的场景层关系 ID 链 (v3 sec 3.6.2)
        extra_attrs: 关系特定属性 (TTC, distance, lateral_distance 等)
        confidence: 置信度 (CARLA 真值默认 1.0)
    Returns:
        BaseRelation 实例
    """
    attrs: Dict[str, Any] = {
        "frame_id": frame_id,
        "confidence": confidence,
        "source_relations": source_relations or [],
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


# ========== 个体行为关系 (v3 sec 3.3.1) ==========


def standing_still(vehicle_id: str, frame_id: int,
                   speed: float = 0.0,
                   source_relations: Optional[List[str]] = None) -> BaseRelation:
    """StandingStill: vehicle -> vehicle (self-loop).

    表示该实体处于静止状态, 通过 ManeuverNode 节点 + 自环边表达 (v3 sec 3.3.1).
    边属性: speed, source_relations
    """
    return build_relation(
        src_id=vehicle_id, dst_id=vehicle_id,
        rel_type=BehaviorRelationType.STANDING_STILL,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"speed": speed},
    )


def changing_lane(vehicle_id: str, target_lane_id: str, frame_id: int,
                  lateral_speed: float = 0.0,
                  source_relations: Optional[List[str]] = None) -> BaseRelation:
    """ChangingLane: vehicle -> RoadElement (target lane).

    变道行为识别基础 (v3 sec 3.3.1).
    边属性: lateral_speed, target_lane_id
    """
    return build_relation(
        src_id=vehicle_id, dst_id=target_lane_id,
        rel_type=BehaviorRelationType.CHANGING_LANE,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"lateral_speed": lateral_speed,
                     "target_lane_id": target_lane_id},
    )


# ========== 交互行为关系 (v3 sec 3.3.2) ==========


def following(vehicle_id: str, leader_id: str, frame_id: int,
              distance: float, relative_speed: float = 0.0,
              ttc: Optional[float] = None,
              source_relations: Optional[List[str]] = None) -> BaseRelation:
    """Following: vehicle -> leader vehicle.

    v3 sec 3.3.2.
    边属性: distance, relative_speed, ttc
    典型 source_relations: [in_lane_id, ahead_of_id]
    """
    return build_relation(
        src_id=vehicle_id, dst_id=leader_id,
        rel_type=BehaviorRelationType.FOLLOWING,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"distance": distance,
                     "relative_speed": relative_speed,
                     "ttc": ttc},
    )


def approaching(vehicle_id: str, target_id: str, frame_id: int,
                distance: float, ttc: Optional[float] = None,
                target_type: str = "traffic_light",
                source_relations: Optional[List[str]] = None) -> BaseRelation:
    """Approaching: vehicle -> TrafficLight (或车辆 -> 目标对象).

    v3 sec 3.3.2 + sec 3.3.3 演化关系.
    边属性: distance, ttc, target_type
    """
    return build_relation(
        src_id=vehicle_id, dst_id=target_id,
        rel_type=BehaviorRelationType.APPROACHING,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"distance": distance, "ttc": ttc,
                     "target_type": target_type},
    )


def yielding_to(vehicle_id: str, pedestrian_id: str, frame_id: int,
                distance: float, ped_action: str = "Walking",
                ego_speed: float = 0.0,
                source_relations: Optional[List[str]] = None) -> BaseRelation:
    """YieldingTo: vehicle -> pedestrian.

    v3 sec 3.3.2 / sec 3.7.2 谓词 YieldingTo(A, P, Frame_t).
    边属性: distance, ped_action, ego_speed
    """
    return build_relation(
        src_id=vehicle_id, dst_id=pedestrian_id,
        rel_type=BehaviorRelationType.YIELDING_TO,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"distance": distance,
                     "ped_action": ped_action,
                     "ego_speed": ego_speed},
    )


def overtaking(vehicle_id: str, target_id: str, frame_id: int,
               lateral_distance: float, longitudinal_speed_diff: float = 0.0,
               source_relations: Optional[List[str]] = None) -> BaseRelation:
    """Overtaking: vehicle -> vehicle.

    v3 sec 3.3.2.
    边属性: lateral_distance, longitudinal_speed_diff
    复杂行为 = following + beside + overtaking 的复合 (v3 sec 3.1.1 文末段)
    """
    return build_relation(
        src_id=vehicle_id, dst_id=target_id,
        rel_type=BehaviorRelationType.OVERTAKING,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"lateral_distance": lateral_distance,
                     "longitudinal_speed_diff": longitudinal_speed_diff},
    )


def wrong_side_meeting(vehicle_a_id: str, vehicle_b_id: str, frame_id: int,
                       lateral_distance: float = 0.0,
                       source_relations: Optional[List[str]] = None) -> BaseRelation:
    """WrongSideMeeting: vehicle -> vehicle.

    v3 sec 3.3.2 (反向会车 / 错并行).
    边属性: lateral_distance
    """
    return build_relation(
        src_id=vehicle_a_id, dst_id=vehicle_b_id,
        rel_type=BehaviorRelationType.WRONG_SIDE_MEETING,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"lateral_distance": lateral_distance},
    )


def opposite_direction(vehicle_a_id: str, vehicle_b_id: str, frame_id: int,
                       closing_speed: float = 0.0,
                       source_relations: Optional[List[str]] = None) -> BaseRelation:
    """OppositeDirection: vehicle -> vehicle.

    v3 sec 3.3.2.
    边属性: closing_speed (相向接近的速度)
    """
    return build_relation(
        src_id=vehicle_a_id, dst_id=vehicle_b_id,
        rel_type=BehaviorRelationType.OPPOSITE_DIRECTION,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"closing_speed": closing_speed},
    )


def same_direction(vehicle_a_id: str, vehicle_b_id: str, frame_id: int,
                   speed_diff: float = 0.0,
                   source_relations: Optional[List[str]] = None) -> BaseRelation:
    """SameDirection: vehicle -> vehicle.

    v3 sec 3.3.2.
    边属性: speed_diff
    """
    return build_relation(
        src_id=vehicle_a_id, dst_id=vehicle_b_id,
        rel_type=BehaviorRelationType.SAME_DIRECTION,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"speed_diff": speed_diff},
    )


def blocked_view(observer_id: str, target_id: str, frame_id: int,
                 occlusion_ratio: float = 1.0,
                 source_relations: Optional[List[str]] = None) -> BaseRelation:
    """BlockedView: observer vehicle -> target vehicle (or scene object).

    v3 sec 3.3.2.
    边属性: occlusion_ratio (0.0 全遮挡 .. 1.0 完全可见)
    """
    return build_relation(
        src_id=observer_id, dst_id=target_id,
        rel_type=BehaviorRelationType.BLOCKED_VIEW,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"occlusion_ratio": occlusion_ratio},
    )


def approaching_pedestrian(vehicle_id: str, pedestrian_id: str, frame_id: int,
                            distance: float, ttc: Optional[float] = None,
                            source_relations: Optional[List[str]] = None) -> BaseRelation:
    """ApproachingPedestrian: vehicle -> pedestrian.

    区别于 yielding_to: 强调"正在接近的瞬时态", 还未发生让行.
    v3 sec 3.3.2.
    边属性: distance, ttc
    """
    return build_relation(
        src_id=vehicle_id, dst_id=pedestrian_id,
        rel_type=BehaviorRelationType.APPROACHING_PEDESTRIAN,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"distance": distance, "ttc": ttc},
    )


# ========== 演化行为关系 (v3 sec 3.3.3) ==========


def approaching_intersection(vehicle_id: str, junction_id: str, frame_id: int,
                              distance_to_junction: float,
                              source_relations: Optional[List[str]] = None) -> BaseRelation:
    """ApproachingIntersection: vehicle -> RoadElement (junction).

    v3 sec 3.3.3 演化行为.
    边属性: distance_to_junction
    source_relations 通常包含 [in_junction_id] 反向触发或 [in_lane_id] 链
    """
    return build_relation(
        src_id=vehicle_id, dst_id=junction_id,
        rel_type=BehaviorRelationType.APPROACHING_INTERSECTION,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"distance_to_junction": distance_to_junction},
    )


def crossing(pedestrian_id: str, crosswalk_id: str, frame_id: int,
             crossing_speed: float = 1.2,
             source_relations: Optional[List[str]] = None) -> BaseRelation:
    """Crossing: pedestrian -> RoadElement (crosswalk).

    v3 sec 3.3.3 演化行为.
    边属性: crossing_speed
    """
    return build_relation(
        src_id=pedestrian_id, dst_id=crosswalk_id,
        rel_type=BehaviorRelationType.CROSSING,
        frame_id=frame_id, valid_from=frame_id,
        source_relations=source_relations,
        extra_attrs={"crossing_speed": crossing_speed},
    )
