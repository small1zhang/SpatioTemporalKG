"""
行为关系检测器 (v3 sec 3.3 + sec 7.3.2)

本模块实现从 CARLA 场景层数据 (实体属性 + 场景关系) 生成行为层关系的检测逻辑。

每个 detect_* 函数:
  - 输入: 一帧内的场景层上下文 (实体 dict + 场景关系列表)
  - 输出: 行为关系候选列表 (含 condition_met 标志 + 附件属性)
  - 防抖处理: 由 BehaviorRelationGenerator 在 debouncer.py 中完成

与 v8.3 方案映射 (v3 sec 3.8):
  BehaviorRelationGenerator.generate() -> {behavior_rels} (含防抖)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.types import SceneRelationType, BehaviorRelationType
from stk.scenario.nodes import VehicleEntity


# ============================================================
# 检测阈值 (可根据场景进行调整)
# ============================================================

# 跟驰 TTC 阈值 (秒)：TTC < 3s 认为高风险跟随
TTC_CRITICAL = 3.0
# 跟驰距离阈值 (米)
FOLLOWING_MAX_DISTANCE = 100.0
# 静止速度阈值 (m/s)
STANDING_SPEED_THRESHOLD = 0.5
# 变道横向速度阈值 (m/s)
LANE_CHANGE_LATERAL_SPEED = 0.3
# 行人距离阈值 (米) — 初次激活让行/接近检测
PEDESTRIAN_ACTIVATION_DISTANCE = 50.0
# 路口距离阈值 (米)
JUNCTION_ACTIVATION_DISTANCE = 30.0
# 对向相向判定：航向差阈值 (rad)
OPPOSITE_HEADING_DIFF = 2.5  # ~143 deg
# 并排: 横向距离阈值 (米)
BESIDE_LATERAL_THRESHOLD = 2.5


# ============================================================
# 1. 个体行为检测
# ============================================================


def detect_standing_still(vehicle: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测车辆是否静止 (v3 sec 3.3.1).
    条件: speed < STANDING_SPEED_THRESHOLD
    """
    speed = vehicle.get("speed", 0.0)
    condition_met = speed < STANDING_SPEED_THRESHOLD
    extra = {"speed": speed}
    return condition_met, extra


def detect_changing_lane(vehicle: Dict[str, Any],
                         scene_relations: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """检测车辆是否正在变道 (v3 sec 3.3.1).
    条件: 横向速度 > LANE_CHANGE_LATERAL_SPEED
          且相邻车道存在
          且 in_junction == False (交叉口内不算变道)
    简化实现: 使用 scenario.relations.adjacent_lane + lateral_speed
    """
    vx = vehicle.get("velocity_x", 0.0)
    vy = vehicle.get("velocity_y", 0.0)
    # 估计横向速度: 以 heading 将 vx,vy 旋转到车道坐标系
    # 简化: 用 vy 绝对值 > 0.3 近似
    lateral_speed = abs(vy)
    condition_met = lateral_speed > LANE_CHANGE_LATERAL_SPEED
    extra = {"lateral_speed": lateral_speed, "velocity_x": vx, "velocity_y": vy}
    return condition_met, extra


# ============================================================
# 2. 交互行为检测
# ============================================================


def detect_following(vehicle: Dict[str, Any],
                     leader: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测跟驰行为 (v3 sec 3.3.2).

    条件:
      - 两车在同车道 (v -[in_lane]-> lane <-[in_lane]- w)
      - 车间距 < FOLLOWING_MAX_DISTANCE
      - 前车在前方 (v -[ahead_of]-> w 或 if v.speed < w.speed 后车逼近前车)

    Args:
        vehicle: ego 车辆属性 dict
        leader: 前车属性 dict

    Returns:
        (condition_met, extra_attrs)
        extra_attrs: distance, relative_speed, ttc
    """
    # 简化实现: 根据位置和速度推定
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    l_x = leader.get("location_x", 0.0)
    l_y = leader.get("location_y", 0.0)
    dx = l_x - v_x
    dy = l_y - v_y
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if distance > FOLLOWING_MAX_DISTANCE or distance < 1.0:
        return (False, {"distance": distance, "relative_speed": 0.0, "ttc": None})

    v_speed = vehicle.get("speed", 0.0)
    l_speed = leader.get("speed", 0.0)
    relative_speed = v_speed - l_speed  # 正值 = 后车更快

    # TTC = 距离 / closing_speed (正向相对速度)
    closing_speed = max(relative_speed, 0.1)
    ttc = distance / closing_speed if closing_speed > 0 else None

    condition_met = distance < FOLLOWING_MAX_DISTANCE
    return (condition_met, {
        "distance": round(distance, 2),
        "relative_speed": round(relative_speed, 2),
        "ttc": round(ttc, 2) if ttc is not None else None,
    })


def detect_approaching(vehicle: Dict[str, Any],
                       target: Dict[str, Any],
                       target_type: str = "traffic_light") -> Tuple[bool, Dict[str, Any]]:
    """检测接近信号灯或目标对象 (v3 sec 3.3.2).
    条件: 距离 < JUNCTION_ACTIVATION_DISTANCE
          且速度 > 0
    """
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    t_x = target.get("location_x", 0.0)
    t_y = target.get("location_y", 0.0)
    distance = ((v_x - t_x) ** 2 + (v_y - t_y) ** 2) ** 0.5
    speed = vehicle.get("speed", 0.0)

    condition_met = distance < JUNCTION_ACTIVATION_DISTANCE and speed > 0.5
    ttc = (distance / speed) if speed > 0.5 else None
    return (condition_met, {
        "distance": round(distance, 2),
        "ttc": round(ttc, 2) if ttc is not None else None,
        "target_type": target_type,
    })


def detect_yielding_to(vehicle: Dict[str, Any],
                       pedestrian: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测车辆是否对行人在让行 (v3 sec 3.3.2).
    条件: 行人距离 < PEDESTRIAN_ACTIVATION_DISTANCE
          且车辆正在减速 (brake > 0.0 或 speed 明显下降)
    简化实现: 行人距离 + 车辆速度 (低速视为让行)
    """
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    p_x = pedestrian.get("location_x", 0.0)
    p_y = pedestrian.get("location_y", 0.0)
    distance = ((v_x - p_x) ** 2 + (v_y - p_y) ** 2) ** 0.5
    speed = vehicle.get("speed", 0.0)
    ped_action = pedestrian.get("action", "Idle")

    condition_met = (distance < PEDESTRIAN_ACTIVATION_DISTANCE and speed < 1.5)
    return (condition_met, {
        "distance": round(distance, 2),
        "ped_action": ped_action,
        "ego_speed": round(speed, 2),
    })


def detect_approaching_pedestrian(vehicle: Dict[str, Any],
                                   pedestrian: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测车辆正在接近行人 (v3 sec 3.3.2).
    条件: 距离 < PEDESTRIAN_ACTIVATION_DISTANCE 且速度 > 0.5
    """
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    p_x = pedestrian.get("location_x", 0.0)
    p_y = pedestrian.get("location_y", 0.0)
    distance = ((v_x - p_x) ** 2 + (v_y - p_y) ** 2) ** 0.5
    speed = vehicle.get("speed", 0.0)
    closing_speed = max(speed - pedestrian.get("speed", 0.0), 0.1)
    ttc = distance / closing_speed if closing_speed > 0 else None

    condition_met = (distance < PEDESTRIAN_ACTIVATION_DISTANCE and speed > 0.5
                     and distance > 1.0)
    return (condition_met, {
        "distance": round(distance, 2),
        "ttc": round(ttc, 2) if ttc is not None else None,
    })


def detect_overtaking(vehicle: Dict[str, Any],
                      target: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测超车行为 (v3 sec 3.3.2).
    条件: 横向并排 + 车速高于被超车
    """
    v_speed = vehicle.get("speed", 0.0)
    t_speed = target.get("speed", 0.0)
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    t_x = target.get("location_x", 0.0)
    t_y = target.get("location_y", 0.0)
    lateral_distance = abs(v_y - t_y)
    longitudinal_diff = v_speed - t_speed

    condition_met = (lateral_distance < BESIDE_LATERAL_THRESHOLD
                     and longitudinal_diff > 0.5
                     and v_speed > 1.0)
    return (condition_met, {
        "lateral_distance": round(lateral_distance, 2),
        "longitudinal_speed_diff": round(longitudinal_diff, 2),
    })


def detect_opposite_direction(vehicle_a: Dict[str, Any],
                               vehicle_b: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测对向行驶 (v3 sec 3.3.2).
    条件: 航向差 > OPPOSITE_HEADING_DIFF (约 143 deg)
    """
    heading_a = vehicle_a.get("heading_rad", 0.0)
    heading_b = vehicle_b.get("heading_rad", 0.0)
    heading_diff = abs(heading_a - heading_b) % (2 * 3.14159)
    if heading_diff > 3.14159:
        heading_diff = 2 * 3.14159 - heading_diff
    closing_speed = vehicle_a.get("speed", 0.0) + vehicle_b.get("speed", 0.0)
    condition_met = heading_diff > OPPOSITE_HEADING_DIFF
    return (condition_met, {
        "heading_diff": round(heading_diff, 2),
        "closing_speed": round(closing_speed, 2),
    })


def detect_blocked_view(observer: Dict[str, Any],
                         target: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测视野遮挡 (v3 sec 3.3.2).
    简化: 两车之间距离较近且横向有重叠 (前后车结构)
    """
    o_x = observer.get("location_x", 0.0)
    o_y = observer.get("location_y", 0.0)
    t_x = target.get("location_x", 0.0)
    t_y = target.get("location_y", 0.0)
    longitudinal_diff = abs(o_x - t_x)
    lateral_diff = abs(o_y - t_y)
    # 前后距离 < 30m, 横向距离 < 3m 视为可能遮挡
    condition_met = longitudinal_diff < 30.0 and lateral_diff < 3.0
    return (condition_met, {
        "occlusion_ratio": round(1.0 - longitudinal_diff / 30.0, 2),
    })


# ============================================================
# 3. 演化行为检测
# ============================================================


def detect_approaching_intersection(vehicle: Dict[str, Any],
                                     junction: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测车辆接近路口 (v3 sec 3.3.3).
    条件: 距路口中心 < JUNCTION_ACTIVATION_DISTANCE
    """
    v_x = vehicle.get("location_x", 0.0)
    v_y = vehicle.get("location_y", 0.0)
    j_x = junction.get("center_x", 0.0)
    j_y = junction.get("center_y", 0.0)
    distance = ((v_x - j_x) ** 2 + (v_y - j_y) ** 2) ** 0.5
    condition_met = distance < JUNCTION_ACTIVATION_DISTANCE
    return (condition_met, {
        "distance_to_junction": round(distance, 2),
    })


def detect_crossing(pedestrian: Dict[str, Any],
                     crosswalk: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """检测行人横穿人行道 (v3 sec 3.3.3).
    条件: 行人在斑马线上且速度 > 0
    """
    is_on_crosswalk = pedestrian.get("is_on_crosswalk", False)
    speed = pedestrian.get("speed", 0.0)
    condition_met = is_on_crosswalk and speed > 0.2
    return (condition_met, {
        "crossing_speed": round(speed, 2),
    })


# ============================================================
# 4. 综合检测器 — 对一帧运行所有检测
# ============================================================


def run_all_detectors(
    vehicles: List[Dict[str, Any]],
    pedestrians: List[Dict[str, Any]],
    traffic_lights: List[Dict[str, Any]],
    junctions: List[Dict[str, Any]],
    crosswalks: List[Dict[str, Any]],
    scene_relations: List[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, str, bool, Dict[str, Any]]]]:
    """在一帧内运行所有行为检测器.

    对输入的场景层数据 (实体 dict + 场景关系列表) 运行所有 detect_* 函数。
    输出格式按关系类型分组:
      {
        rel_type_str: [(src_id, dst_id, condition_met, extra_attrs), ...],
        ...
      }

    此输出可直接喂给 BehaviorRelationGenerator 做防抖 + 节点创建。

    这是 v3 sec 7.3.2 BehaviorRelationGenerator.generate() 的核心驱动步骤。
    """
    results: Dict[str, List[Tuple[str, str, bool, Dict[str, Any]]]] = {}

    def add(rel_type: str, src: str, dst: str, condition: bool, extra: Dict):
        results.setdefault(rel_type, []).append((src, dst, condition, extra))

    # 个体行为
    for v in vehicles:
        eid = v.get("entity_id", "")
        if not eid:
            continue
        cond, extra = detect_standing_still(v)
        add("standing_still", eid, eid, cond, extra)
        cond, extra = detect_changing_lane(v, scene_relations)
        add("changing_lane", eid, eid, cond, extra)

    # 车辆 - 车辆交互
    for i, v_a in enumerate(vehicles):
        eid_a = v_a.get("entity_id", "")
        if not eid_a:
            continue
        for j, v_b in enumerate(vehicles):
            if i >= j:
                continue
            eid_b = v_b.get("entity_id", "")
            if not eid_b:
                continue
            # Following: A 跟在 B 后面
            cond, extra = detect_following(v_a, v_b)
            add("following", eid_a, eid_b, cond, extra)
            # Overtaking
            cond, extra = detect_overtaking(v_a, v_b)
            add("overtaking", eid_a, eid_b, cond, extra)
            # Opposite direction
            cond, extra = detect_opposite_direction(v_a, v_b)
            add("opposite_direction", eid_a, eid_b, cond, extra)
            # Blocked view
            cond, extra = detect_blocked_view(v_a, v_b)
            add("blocked_view", eid_a, eid_b, cond, extra)

    # 车辆 - 行人交互
    for v in vehicles:
        eid_v = v.get("entity_id", "")
        if not eid_v:
            continue
        for p in pedestrians:
            eid_p = p.get("entity_id", "")
            if not eid_p:
                continue
            cond, extra = detect_yielding_to(v, p)
            add("yielding_to", eid_v, eid_p, cond, extra)
            cond, extra = detect_approaching_pedestrian(v, p)
            add("approaching_pedestrian", eid_v, eid_p, cond, extra)

    # 车辆 - 信号灯交互
    for v in vehicles:
        eid_v = v.get("entity_id", "")
        if not eid_v:
            continue
        for tl in traffic_lights:
            eid_tl = tl.get("entity_id", "")
            if not eid_tl:
                continue
            cond, extra = detect_approaching(v, tl, target_type="traffic_light")
            add("approaching", eid_v, eid_tl, cond, extra)

    # 演化行为: 车辆 - 路口
    for v in vehicles:
        eid_v = v.get("entity_id", "")
        if not eid_v:
            continue
        for jnc in junctions:
            eid_j = jnc.get("entity_id", "")
            if not eid_j:
                continue
            cond, extra = detect_approaching_intersection(v, jnc)
            add("approaching_intersection", eid_v, eid_j, cond, extra)

    # 演化行为: 行人 - 人行道
    for p in pedestrians:
        eid_p = p.get("entity_id", "")
        if not eid_p:
            continue
        for cw in crosswalks:
            eid_cw = cw.get("entity_id", "")
            if not eid_cw:
                continue
            cond, extra = detect_crossing(p, cw)
            add("crossing", eid_p, eid_cw, cond, extra)

    return results
