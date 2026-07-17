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
    """构建车道拓扑关系 (adjacent / connects)."""
    rels = []
    for wp in waypoint_data:
        nid = f"road_{wp['road_id']}_lane_{wp['lane_id']}"
        left = wp.get("left_lane_id")
        right = wp.get("right_lane_id")
        if left is not None:
            rels.append({"src_id": nid, "dst_id": f"road_{wp['road_id']}_lane_{left}",
                         "relation_type": "adjacent_lane", "frame_id": 0})
        if right is not None:
            rels.append({"src_id": nid, "dst_id": f"road_{wp['road_id']}_lane_{right}",
                         "relation_type": "adjacent_lane", "frame_id": 0})
    return rels