# -*- coding: utf-8 -*-
"""提取流水线: 串联 7.3-7.7 产生帧快照 (v3 §7.8)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from stk.extraction.actor_extractor import extract_all_actors
from stk.extraction.waypoint_extractor import extract_waypoints, build_lane_topology
from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
from stk.extraction.sensor_extractor import extract_sensor_events
from stk.extraction.weather_extractor import build_environment_snapshot


def process_frame(frame_data: dict) -> Dict[str, Any]:
    """处理单帧 CARLA 数据，产出 FrameData 兼容字典.

    Args:
        frame_data: dict, 含 actors, waypoints, traffic_lights, events, weather 等

    Returns:
        dict, 含 vehicles/pedestrians/lanes/traffic_lights/scene_rels/weather
    """
    actors = extract_all_actors(frame_data)
    lanes = extract_waypoints(frame_data.get("waypoints", []))
    tl = extract_all_traffic_lights(frame_data.get("traffic_lights", []))
    sensors = extract_sensor_events(frame_data.get("events", []))
    weather = build_environment_snapshot(frame_data.get("weather", {}),
                                         frame_data.get("frame_id", 0))
    topo = build_lane_topology(frame_data.get("waypoints", []))
    return {
        "frame_id": frame_data.get("frame_id", 0),
        "elapsed_seconds": frame_data.get("elapsed_seconds", 0.0),
        "delta_seconds": frame_data.get("delta_seconds", 0.05),
        "vehicles": actors.get("vehicles", []),
        "pedestrians": actors.get("pedestrians", []),
        "traffic_lights": tl,
        "lanes": lanes,
        "scene_rels": topo,
        "weather": weather,
        "sensor_events": sensors,
    }