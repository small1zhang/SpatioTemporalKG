# -*- coding: utf-8 -*-
"""API 映射速查表 (v3 §7.7)."""
from __future__ import annotations
from typing import Dict, Callable


API_MAPPING: Dict[str, str] = {
    "get_world": "carla.Client.get_world()",
    "get_snapshot": "world.get_snapshot()",
    "get_actors": "world.get_actors().filter(*filter_type)",
    "vehicle_transform": "actor.get_transform()",
    "vehicle_location": "actor.get_location()",
    "vehicle_velocity": "actor.get_velocity()",
    "vehicle_control": "actor.get_control()",
    "vehicle_bounding_box": "actor.get_bounding_box()",
    "waypoint_next": "map.get_waypoint(location).next(distance)",
    "traffic_light_state": "traffic_light.get_state()",
    "collision_sensor": "blueprint.sensor.other.collision",
    "lane_invasion_sensor": "blueprint.sensor.other.lane_detector",
    "weather": "world.get_weather()",
}

FUNCTION_MAP: Dict[str, Callable] = {}