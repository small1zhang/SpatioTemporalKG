# -*- coding: utf-8 -*-
"""天气快照提取 (v3 §7.6)."""
from __future__ import annotations
from typing import Any, Dict, Optional


def extract_weather(weather_data: dict) -> Dict[str, Any]:
    return {
        "fog_density": weather_data.get("fog_density", 0.0),
        "cloudiness": weather_data.get("cloudiness", 0.0),
        "precipitation": weather_data.get("precipitation", 0.0),
        "wetness": weather_data.get("wetness", 0.0),
        "sun_altitude_angle": weather_data.get("sun_altitude_angle", 90.0),
        "wind_intensity": weather_data.get("wind_intensity", 0.0),
    }


def build_environment_snapshot(weather_data: dict, frame_id: int) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "entity_type": "EnvSnapshot",
        **extract_weather(weather_data),
    }