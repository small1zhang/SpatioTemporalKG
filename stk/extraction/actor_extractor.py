# -*- coding: utf-8 -*-
"""Actor 提取: 从 CARLA world 快照生成车辆/行人 dict (v3 §7.1)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def extract_vehicle(actor_data: dict) -> Dict[str, Any]:
    """从 CARLA Vehicle actor dict 提取."""
    loc = actor_data.get("location", {})
    vel = actor_data.get("velocity", {})
    return {
        "entity_id": actor_data.get("id", ""),
        "entity_type": "Vehicle",
        "vehicle_type": actor_data.get("type_id", "vehicle.*"),
        "is_ego": actor_data.get("is_ego", False),
        "location_x": loc.get("x", 0.0),
        "location_y": loc.get("y", 0.0),
        "location_z": loc.get("z", 0.0),
        "velocity_x": vel.get("x", 0.0),
        "velocity_y": vel.get("y", 0.0),
        "velocity_z": vel.get("z", 0.0),
        "speed": actor_data.get("speed", 0.0),
        "speed_kmh": actor_data.get("speed", 0.0) * 3.6,
        "heading_rad": actor_data.get("heading_rad", 0.0),
        "brake": actor_data.get("brake", 0.0),
        "throttle": actor_data.get("throttle", 0.0),
        "steer": actor_data.get("steer", 0.0),
    }


def extract_pedestrian(actor_data: dict) -> Dict[str, Any]:
    """从 CARLA Walker/Pedestrian actor dict 提取."""
    loc = actor_data.get("location", {})
    vel = actor_data.get("velocity", {})
    return {
        "entity_id": actor_data.get("id", ""),
        "entity_type": "Pedestrian",
        "location_x": loc.get("x", 0.0),
        "location_y": loc.get("y", 0.0),
        "location_z": loc.get("z", 0.0),
        "velocity_x": vel.get("x", 0.0),
        "velocity_y": vel.get("y", 0.0),
        "speed": actor_data.get("speed", 0.0),
        "heading_rad": actor_data.get("heading_rad", 0.0),
        "is_on_crosswalk": actor_data.get("is_on_crosswalk", False),
        "action": actor_data.get("action", "Idle"),
    }


def extract_all_actors(frame_data: dict) -> Dict[str, List[Dict]]:
    """从帧数据提取所有 actors，返回 {vehicles, pedestrians}."""
    return {
        "vehicles": [extract_vehicle(a) for a in frame_data.get("actors", [])
                     if a.get("type", "").startswith("vehicle")],
        "pedestrians": [extract_pedestrian(a) for a in frame_data.get("actors", [])
                        if a.get("type", "").startswith("walker")],
    }