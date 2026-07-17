# -*- coding: utf-8 -*-
"""路网拓扑提取: Waypoint 遍历生成 RoadElementEntity (v3 §7.2)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def extract_waypoints(waypoint_data: List[dict]) -> List[Dict[str, Any]]:
    """从 waypoint 列表提取道路元素."""
    roads = []
    for wp in waypoint_data:
        road = {
            "entity_id": f"road_{wp.get('road_id',0)}_lane_{wp.get('lane_id',0)}",
            "entity_type": "RoadElement",
            "road_id": wp.get("road_id", 0),
            "lane_id": wp.get("lane_id", 0),
            "junction_id": wp.get("junction_id", -1),
            "lane_type": wp.get("lane_type", "Driving"),
            "lane_width": wp.get("lane_width", 3.5),
            "speed_limit": wp.get("speed_limit", 60.0),
            "center_x": wp.get("x", 0.0),
            "center_y": wp.get("y", 0.0),
            "center_z": wp.get("z", 0.0),
            "heading_rad": wp.get("heading_rad", 0.0),
            "left_lane_id": wp.get("left_lane_id"),
            "right_lane_id": wp.get("right_lane_id"),
            "has_traffic_light": wp.get("has_traffic_light", False),
        }
        roads.append(road)
    return roads


def build_lane_topology(waypoint_data: List[dict]) -> List[Dict[str, Any]]:
    """构建车道拓扑关系 (adjacent / connects / controlled_by)."""
    rels = []
    road_groups = {}
    for wp in waypoint_data:
        rid = wp.get("road_id", 0)
        road_groups.setdefault(rid, []).append(wp)
    for rid, wps in road_groups.items():
        lane_ids = sorted(set(wp["lane_id"] for wp in wps))
        for i in range(len(lane_ids) - 1):
            nid_a = "road_" + str(rid) + "_lane_" + str(lane_ids[i])
            nid_b = "road_" + str(rid) + "_lane_" + str(lane_ids[i+1])
            rels.append({
                "src_id": nid_a, "dst_id": nid_b,
                "relation_type": "lane_connects", "frame_id": 0,
            })
    for wp in waypoint_data:
        nid = "road_" + str(wp["road_id"]) + "_lane_" + str(wp["lane_id"])
        left = wp.get("left_lane_id")
        right = wp.get("right_lane_id")
        if left is not None:
            rels.append({
                "src_id": nid, "dst_id": "road_" + str(wp["road_id"]) + "_lane_" + str(left),
                "relation_type": "adjacent_lane", "frame_id": 0,
            })
        if right is not None:
            rels.append({
                "src_id": nid, "dst_id": "road_" + str(wp["road_id"]) + "_lane_" + str(right),
                "relation_type": "adjacent_lane", "frame_id": 0,
            })
        if wp.get("has_traffic_light", False):
            rels.append({
                "src_id": nid, "dst_id": "tl_on_" + nid,
                "relation_type": "controlled_by", "frame_id": 0,
            })
    return rels


def build_junction_connections(waypoint_data: List[dict]) -> List[Dict[str, Any]]:
    """从 waypoint 列表构建 junction 内 lane→lane 连接关系.

    每个在 junction 内的 lane 节点 (junction_id != -1) 会下一代 carla.Waypoint.next(.) 的
    next lane 视为后继; 这里只基于已采集到的 waypoint_data 推断: 若某条 lane (junction_id != -1)
    没有外部 successor lane 信息, 退回到空的连接列表 (保持向后兼容).

    当前面向 phase1 已经送进来的字段: road_id, lane_id, junction_id, left_lane_id,
    right_lane_id. junction 内两个 lane 视为相邻当且仅当其中一个的 left/right lane_id 等于
    另一个的 lane_id 且二者都属于同一 junction (junction_id 相同且 != -1).

    返回:
        [{"src_id": "road_X_lane_Y", "dst_id": "road_X_lane_Z", "relation_type": "connects",
          "frame_id": 0}]
    """
    rels = []
    by_lane_id = {}
    for wp in waypoint_data:
        if wp.get("junction_id", -1) != -1:
            by_lane_id[(wp["road_id"], wp["lane_id"])] = wp

    for (jid, wp) in by_lane_id.items():
        nid = f"road_{wp['road_id']}_lane_{wp['lane_id']}"
        for side_key in ("left_lane_id", "right_lane_id"):
            other_lane = wp.get(side_key)
            if other_lane is None:
                continue
            other_wp = by_lane_id.get((wp["road_id"], other_lane))
            if other_wp is None:
                continue
            other_jid = other_wp.get("junction_id", -1)
            if other_jid != wp.get("junction_id", -1):
                continue
            other_nid = f"road_{other_wp['road_id']}_lane_{other_wp['lane_id']}"
            rels.append({
                "src_id": nid, "dst_id": other_nid,
                "relation_type": "connects", "frame_id": 0,
            })
    return rels
