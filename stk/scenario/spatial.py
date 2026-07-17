"""
空间关系计算函数 (v3 §2.9, §7.1.2)

本模块实现场景层空间关系的纯函数计算逻辑。
输入是实体对象的字典/列表（不依赖 CARLA 运行时），
输出是 BaseRelation 列表。

所有函数为纯函数：
  compute_*(entities, lanes, ...) -> List[BaseRelation]

设计原则 (v3 §2.7):
  - 同帧性：所有关系成立的时态戳为某一特定帧 t
  - 几何确定性：每条关系可由 CARLA 真值直接计算
  - 零防抖：场景层空间关系不等抖（区别于行为关系）
  - 唯一性：每对 (src, dst, type, t) 唯一
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.relation import BaseRelation
from stk.scenario.relations import (
    adjacent_lane, ahead_of, beside, in_junction, in_lane, lane_connects,
    nearby_pedestrian, on_road,
)


def _location(entity: "BaseEntity") -> Optional[Tuple[float, float, float]]:
    """从实体 attrs 中安全获取位置。"""
    attrs = entity.attrs if hasattr(entity, "attrs") else (entity.get("attrs", entity) if isinstance(entity, dict) else entity)
    x = attrs.get("location_x")
    y = attrs.get("location_y")
    if x is None or y is None:
        return None
    return (x, y, attrs.get("location_z", 0.0))


def _heading(entity: "BaseEntity") -> float:
    """获取朝向角（弧度）。"""
    attrs = entity.attrs if hasattr(entity, "attrs") else entity.get("attrs", {})
    return attrs.get("heading_rad", 0.0)


def compute_in_lane(vehicles: List, lanes: List[Dict], frame_id: int) -> List[BaseRelation]:
    """InLane: 计算每个车辆所在的车道 (v3 §2.9.1)。

    通过 waypoint.get_waypoint(location) 比较 lane_id。
    离线模式使用最近车道匹配（按横向距离）。

    Args:
        vehicles: VehicleEntity 列表
        lanes: lane 属性字典列表，每条有 road_id, lane_id, center_x/y, heading_rad
        frame_id: 当前帧号

    Returns:
        BaseRelation 列表
    """
    results: List[BaseRelation] = []
    for v in vehicles:
        loc = _location(v)
        if loc is None:
            continue
        # 找到最近的 lane（按横向距离）
        best_lane = None
        best_dist = float("inf")
        for lane in lanes:
            lx = lane.get("center_x", 0.0) if isinstance(lane, dict) else _location(lane)
            if lx is None:
                continue
            if isinstance(lane, dict):
                lx, ly, lz = lane.get("center_x", 0.0), lane.get("center_y", 0.0), 0.0
            else:
                lx, ly, lz = _location(lane) or (0.0, 0.0, 0.0)
            # 横向距离 = 点到点距离的横向分量
            dx = loc[0] - lx
            dy = loc[1] - ly
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
        if best_lane is not None and best_dist < 10.0:  # 阈值 10m
            lane_id = best_lane.get("entity_id","") if isinstance(best_lane, dict) else best_lane.entity_id
            results.append(in_lane(v.entity_id, lane_id, frame_id, frame_id, best_dist))
    return results


def compute_ahead_of(vehicles: List, frame_id: int) -> List[BaseRelation]:
    """AheadOf: 同车道后车→前车 (v3 §2.9.2)。

    判断条件：is_in_same_lane(v, w) 且 longitudinal_distance(v, w) > 0。
    """
    results: List[BaseRelation] = []
    for v in vehicles:
        for w in vehicles:
            if v.entity_id == w.entity_id:
                continue
            loc_v = _location(v)
            loc_w = _location(w)
            if loc_v is None or loc_w is None:
                continue
            # 纵向投影差异
            heading_v = _heading(v)
            dx = loc_w[0] - loc_v[0]
            dy = loc_w[1] - loc_v[1]
            long_dist = dx * math.cos(heading_v) + dy * math.sin(heading_v)
            lat_dist = -dx * math.sin(heading_v) + dy * math.cos(heading_v)
            if long_dist > 0 and abs(lat_dist) < 3.5:  # 同车道或邻车道
                results.append(ahead_of(w.entity_id, v.entity_id, frame_id, long_dist, lat_dist))
    return results


def compute_beside(vehicles: List, frame_id: int) -> List[BaseRelation]:
    """Beside: 并排 (v3 §2.9.2) |lateral| < 3m, |longitudinal| < 5m。"""
    results: List[BaseRelation] = []
    for v in vehicles:
        for w in vehicles:
            if v.entity_id >= w.entity_id:
                continue
            loc_v = _location(v)
            loc_w = _location(w)
            if loc_v is None or loc_w is None:
                continue
            heading_v = _heading(v)
            dx = loc_w[0] - loc_v[0]
            dy = loc_w[1] - loc_v[1]
            long_dist = dx * math.cos(heading_v) + dy * math.sin(heading_v)
            lat_dist = -dx * math.sin(heading_v) + dy * math.cos(heading_v)
            # v3 §2.9.2: |lateral| ≤ 3m（含边界）、|longitudinal| < 5m 视为并排
            if abs(lat_dist) <= 3.0 and abs(long_dist) < 5.0:
                results.append(beside(v.entity_id, w.entity_id, frame_id, lat_dist, long_dist))
    return results


def compute_nearby_pedestrian(vehicles: List, pedestrians: List, frame_id: int, threshold: float = 20.0) -> List[BaseRelation]:
    """NearbyPedestrian: 车辆附近行人 (v3 §2.9.2) distance < 20m。"""
    results: List[BaseRelation] = []
    for v in vehicles:
        loc_v = _location(v)
        if loc_v is None:
            continue
        for p in pedestrians:
            loc_p = _location(p)
            if loc_p is None:
                continue
            dist = math.sqrt((loc_v[0] - loc_p[0])**2 + (loc_v[1] - loc_p[1])**2)
            if dist < threshold:
                results.append(nearby_pedestrian(v.entity_id, p.entity_id, frame_id, dist))
    return results


def compute_adjacent_lanes(lanes: List[Dict], frame_id: int) -> List[BaseRelation]:
    """AdjacentLane: 相邻车道 (v3 §2.9.1)。

    通过 left_lane_id / right_lane_id 字段建立。
    """
    results: List[BaseRelation] = []
    for lane in lanes:
        # 如果 lane 是实体对象
        lid = lane.entity_id if hasattr(lane, "entity_id") else lane.get("entity_id", "")
        attrs = lane.attrs if hasattr(lane, "attrs") else lane
        left = attrs.get("left_lane_id")
        right = attrs.get("right_lane_id")
        if left is not None:
            results.append(adjacent_lane(lid, f"road_{attrs.get('road_id',0)}_lane_{left}", frame_id))
        if right is not None:
            results.append(adjacent_lane(lid, f"road_{attrs.get('road_id',0)}_lane_{right}", frame_id))
    return results


def compute_in_junction(vehicles: List, frame_id: int, lanes: Optional[List[Dict]] = None) -> List[BaseRelation]:
    """InJunction: 车辆在路口内 (v3 §2.9.1)。

    通过 vehicle 当前所在 lane 的 junction_id 判定: 若 vehicle.attrs["current_lane_id"]
    对应的 lane attrs["junction_id"] != -1, 则创建
    in_junction(vehicle_id, junction_id, frame_id, frame_id) 关系.

    Args:
        vehicles: VehicleEntity 列表, attrs 中需含 current_lane_id
        frame_id: 当前帧号
        lanes: 可选, lane 属性字典列表 (含 entity_id 与 junction_id)

    Returns:
        BaseRelation 列表
    """
    results: List[BaseRelation] = []
    # 构建 lane_id -> junction_id 索引
    lane_to_junction: Dict[str, int] = {}
    if lanes is not None:
        for ln in lanes:
            if hasattr(ln, "entity_id"):
                eid = ln.entity_id
                jid = (ln.attrs or {}).get("junction_id", -1) if hasattr(ln, "attrs") else getattr(ln, "junction_id", -1)
            else:
                eid = ln.get("entity_id", "")
                jid = ln.get("junction_id", -1)
            if eid and jid != -1:
                lane_to_junction[eid] = int(jid)

    for v in vehicles:
        attrs = v.attrs if hasattr(v, "attrs") else (v.get("attrs", v) if isinstance(v, dict) else v)
        cur_lane_id = attrs.get("current_lane_id")
        if not cur_lane_id:
            continue
        jid = lane_to_junction.get(str(cur_lane_id), -1)
        if jid != -1:
            results.append(in_junction(v.entity_id, f"junction_{jid}", frame_id, frame_id))
    return results

