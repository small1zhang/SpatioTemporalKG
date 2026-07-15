# -*- coding: utf-8 -*-
"""传感器事件提取: Collision + LaneInvasion (v3 §7.5)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def extract_collision_event(event_data: dict) -> Dict[str, Any]:
    return {
        "event_type": "Collision",
        "frame_id": event_data.get("frame_id", 0),
        "ego_actor_id": event_data.get("ego_id", ""),
        "other_actor_id": event_data.get("other_id", ""),
        "impulse": event_data.get("impulse", 0.0),
        "location_x": event_data.get("location_x", 0.0),
        "location_y": event_data.get("location_y", 0.0),
    }


def extract_lane_invasion_event(event_data: dict) -> Dict[str, Any]:
    return {
        "event_type": "LaneInvasion",
        "frame_id": event_data.get("frame_id", 0),
        "actor_id": event_data.get("actor_id", ""),
        "crossed_lane_markings": event_data.get("crossed_lane_markings", []),
    }


def extract_sensor_events(events: List[dict]) -> Dict[str, List]:
    result = {"collisions": [], "lane_invasions": []}
    for ev in events:
        if ev.get("event_type") == "Collision":
            result["collisions"].append(extract_collision_event(ev))
        elif ev.get("event_type") == "LaneInvasion":
            result["lane_invasions"].append(extract_lane_invasion_event(ev))
    return result