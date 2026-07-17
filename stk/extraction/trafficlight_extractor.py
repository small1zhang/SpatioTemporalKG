# -*- coding: utf-8 -*-
"""信号灯提取 (v3 §7.3)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def extract_traffic_light(tl_data: dict) -> Dict[str, Any]:
    loc = tl_data.get("location", {})
    return {
        "entity_id": f"tl_{tl_data.get('id',0)}",
        "entity_type": "TrafficLight",
        "state": tl_data.get("state", "Green"),
        "elapsed_time": tl_data.get("elapsed_time", 0.0),
        "location_x": loc.get("x", 0.0),
        "location_y": loc.get("y", 0.0),
        "location_z": loc.get("z", 0.0),
        "rotation_yaw": tl_data.get("rotation_yaw", 0.0),
        "affected_lane_ids": tl_data.get("affected_lane_ids", []),
    }


def extract_all_traffic_lights(tl_list: List[dict]) -> List[Dict]:
    return [extract_traffic_light(tl) for tl in tl_list]