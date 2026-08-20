#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式适配层 —— 修复所有数据格式断层

本模块负责将各种来源的数据（CARLA chunk JSON、模拟数据、CSV、live CARLA）
统一转换为 stk/gnn/exporter.py 的 extract_stkg_tensors() 所需的 snapshot dict。

snapshot dict 格式:
    {
        "extracted": {
            "frame_id": int,
            "elapsed_seconds": float,
            "vehicles": [...],
            "pedestrians": [...],
            "traffic_lights": [...],
            "weather": {...},
        },
        "delta": None | DeltaGraph,
        "rule_out": {"violations": [...], "responsibilities": [...]},
    }
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ============================================================
# 1. CARLA Chunk JSON (from scripts/long_run/collect.py)
#    格式: {"frame_id", "elapsed_seconds", "actors": [...], "traffic_lights": [...],
#           "weather": {...}, "waypoints": {...}, "events": [...]}
# ============================================================

def chunk_frame_to_snapshot(
    raw_frame: Dict[str, Any],
    rule_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将 long_run chunk 的单帧 dict 转换为 extract_stkg_tensors 兼容格式。

    raw_frame 字段:
      actors: list of dict，每个含 {
        "id": str, "type": "vehicle"/"walker",
        "location": {"x":float,"y":float,"z":float},
        "rotation": {"pitch":float,"yaw":float,"roll":float},
        "velocity": {"x":float,"y":float,"z":float},
        "control": {"throttle":float,"brake":float,"steer":float} (vehicle),
        "speed": float,
        "is_ego": bool,
        "road_id": int,
        "lane_id": int,
        "bbox_extent": {"x":float,"y":float,"z":float},
      }
    """
    vehicles = []
    pedestrians = []

    for actor in (raw_frame.get("actors") or []):
        atype = actor.get("type", "")
        loc = actor.get("location", {})
        rot = actor.get("rotation", {})
        vel = actor.get("velocity", {})
        ctrl = actor.get("control", {}) if atype == "vehicle" else {}

        vx = vel.get("x", 0.0)
        vy = vel.get("y", 0.0)
        yaw = math.radians(rot.get("yaw", 0.0))
        speed = actor.get("speed", math.hypot(vx, vy))
        brake = ctrl.get("brake", 0.0) if isinstance(ctrl, dict) else 0.0
        throttle = ctrl.get("throttle", 0.0) if isinstance(ctrl, dict) else 0.0
        steer = ctrl.get("steer", 0.0) if isinstance(ctrl, dict) else 0.0

        entity_dict = {
            "entity_id": str(actor.get("id", "")),
            "is_ego": bool(actor.get("is_ego", False)),
            "location_x": float(loc.get("x", 0.0)),
            "location_y": float(loc.get("y", 0.0)),
            "location_z": float(loc.get("z", 0.0)),
            "velocity_x": vx,
            "velocity_y": vy,
            "speed": float(speed),
            "heading_rad": yaw,
            "brake": brake,
            "throttle": throttle,
            "steer": steer,
            "is_emergency": bool(actor.get("is_emergency", False)),
        }

        # lane 信息
        road_id = actor.get("road_id")
        lane_id = actor.get("lane_id")
        if road_id is not None and lane_id is not None:
            entity_dict["current_lane"] = {
                "road_id": int(road_id),
                "lane_id": int(lane_id),
                "center_x": float(loc.get("x", 0.0)),
                "center_y": float(loc.get("y", 0.0)),
                "speed_limit": 13.89,
            }

        if atype == "vehicle":
            vehicles.append(entity_dict)
        elif atype == "walker":
            pedestrians.append(entity_dict)

    # traffic lights
    traffic_lights = []
    for tl in (raw_frame.get("traffic_lights") or []):
        tl_loc = tl.get("location", {})
        traffic_lights.append({
            "entity_id": str(tl.get("id", "")),
            "state": str(tl.get("state", "green")).lower(),
            "location_x": float(tl_loc.get("x", 0.0)),
            "location_y": float(tl_loc.get("y", 0.0)),
            "location_z": float(tl_loc.get("z", 0.0)),
        })

    # weather
    weather = raw_frame.get("weather", {})
    if not weather:
        weather = {
            "fog_density": 0.0, "cloudiness": 50.0, "precipitation": 0.0,
            "wetness": 0.0, "sun_altitude_angle": 45.0, "wind_intensity": 0.0,
        }

    extracted = {
        "frame_id": int(raw_frame.get("frame_id", 0)),
        "elapsed_seconds": float(raw_frame.get("elapsed_seconds", 0.0)),
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "traffic_lights": traffic_lights,
        "weather": weather,
    }

    # events (可选，用于异常标签)
    events = raw_frame.get("events") or []

    return {
        "extracted": extracted,
        "delta": None,
        "rule_out": rule_out or {"violations": []},
        "_raw_events": events,
    }


# ============================================================
# 2. 模拟 CARLA 数据 (from exp_results/generate_all.py)
#    格式: {"frame_id", "elapsed_seconds", "map_name", "n_actors",
#           "weather": {...}, "is_anomaly": bool, "anomaly_type": str}
#    → 需要从 data/dataset/frame_actors.csv 加载 actor 行
# ============================================================

def load_simulated_frame(
    frame_id: int,
    actors_by_frame: Dict[int, List[Dict]],
    weather_default: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    从模拟 CARLA 帧数据构建 snapshot dict。

    actors_by_frame: {frame_id: [actor_dict, ...]}
      每个 actor_dict 含: entity_id, type, location_x, location_y, speed, ...
    """
    actors = actors_by_frame.get(frame_id, [])

    vehicles = []
    pedestrians = []
    for a in actors:
        loc_x = float(a.get("location_x", 0.0))
        loc_y = float(a.get("location_y", 0.0))
        vx = float(a.get("velocity_x", 0.0))
        vy = float(a.get("velocity_y", 0.0))

        entity_dict = {
            "entity_id": str(a.get("entity_id", "")),
            "is_ego": bool(a.get("is_ego", False)),
            "location_x": loc_x,
            "location_y": loc_y,
            "location_z": float(a.get("location_z", 0.0)),
            "velocity_x": vx,
            "velocity_y": vy,
            "speed": float(a.get("speed", math.hypot(vx, vy))),
            "heading_rad": float(a.get("heading_rad", 0.0)),
            "brake": float(a.get("brake", 0.0)),
            "throttle": float(a.get("throttle", 0.5)),
            "steer": float(a.get("steer", 0.0)),
            "is_emergency": bool(a.get("is_emergency", False)),
            "current_lane": {
                "road_id": int(a.get("road_id", 0)),
                "lane_id": int(a.get("lane_id", 0)),
                "center_x": loc_x,
                "center_y": loc_y,
                "speed_limit": 13.89,
            },
        }

        if a.get("type") == "pedestrian":
            pedestrians.append(entity_dict)
        else:
            vehicles.append(entity_dict)

    weather = weather_default or {
        "fog_density": float(np.random.uniform(0, 20)),
        "cloudiness": float(np.random.uniform(10, 60)),
        "precipitation": float(np.random.uniform(0, 10)),
        "wetness": float(np.random.uniform(0, 30)),
        "sun_altitude_angle": float(np.random.uniform(10, 70)),
        "wind_intensity": float(np.random.uniform(0, 5)),
    }

    extracted = {
        "frame_id": frame_id,
        "elapsed_seconds": frame_id * 0.05,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "traffic_lights": [],
        "weather": weather,
    }

    return {
        "extracted": extracted,
        "delta": None,
        "rule_out": {"violations": []},
    }


# ============================================================
# 3. SinD2.0 模拟数据 → snapshot dict
# ============================================================

def sind2_frame_to_snapshot(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 SinD2.0 模拟帧转换为 snapshot dict。

    frame 字段来自 generate_all.py 生成的 frames_chunk JSON:
      {"frame_id", "elapsed_seconds", "extracted": {...}, "rule_out": {...}, ...}
    """
    extracted = frame.get("extracted", {})
    rule_out = frame.get("rule_out", {"violations": []})

    # 确保所有 weather 字段完整
    weather = extracted.get("weather", {})
    weather.setdefault("fog_density", 0.0)
    weather.setdefault("cloudiness", 50.0)
    weather.setdefault("precipitation", 0.0)
    weather.setdefault("wetness", 0.0)
    weather.setdefault("sun_altitude_angle", 45.0)
    weather.setdefault("wind_intensity", 0.0)
    extracted["weather"] = weather

    return {
        "extracted": extracted,
        "delta": None,
        "rule_out": rule_out,
    }


# ============================================================
# 4. 从 data/dataset/frame_actors.csv 加载 actor 数据
# ============================================================

def load_frame_actors_from_csv(
    csv_path: Path,
    frame_id: int,
) -> List[Dict[str, Any]]:
    """
    从 frame_actors.csv 加载指定 frame 的 actor 行。

    CSV columns: frame_id, actor_id, type, type_id, is_ego, x, y, z,
                 vx, vy, vz, speed, heading_rad, brake, throttle, steer, ...
    """
    import csv
    actors = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("frame_id", -1)) == frame_id:
                actors.append({
                    "entity_id": row.get("actor_id", ""),
                    "type": row.get("type", "vehicle"),
                    "type_id": row.get("type_id", ""),
                    "is_ego": bool(int(row.get("is_ego", 0))),
                    "location_x": float(row.get("x", 0.0)),
                    "location_y": float(row.get("y", 0.0)),
                    "location_z": float(row.get("z", 0.0)),
                    "velocity_x": float(row.get("vx", 0.0)),
                    "velocity_y": float(row.get("vy", 0.0)),
                    "speed": float(row.get("speed", 0.0)),
                    "heading_rad": float(row.get("heading_rad", 0.0)),
                    "brake": float(row.get("brake", 0.0)),
                    "throttle": float(row.get("throttle", 0.0)),
                    "steer": float(row.get("steer", 0.0)),
                    "road_id": int(float(row.get("road_id", 0))),
                    "lane_id": int(float(row.get("lane_id", 0))),
                    "is_emergency": False,
                })
    return actors


def build_snapshot_from_frame_actors(
    frame_id: int,
    actors: List[Dict[str, Any]],
    weather: Optional[Dict] = None,
) -> Dict[str, Any]:
    """从 actor 列表构建 snapshot dict。"""
    vehicles = []
    pedestrians = []
    for a in actors:
        loc_x = a.get("location_x", 0.0)
        loc_y = a.get("location_y", 0.0)
        entity = {
            "entity_id": a["entity_id"],
            "is_ego": a.get("is_ego", False),
            "location_x": float(loc_x),
            "location_y": float(loc_y),
            "location_z": float(a.get("location_z", 0.0)),
            "velocity_x": float(a.get("velocity_x", 0.0)),
            "velocity_y": float(a.get("velocity_y", 0.0)),
            "speed": float(a.get("speed", 0.0)),
            "heading_rad": float(a.get("heading_rad", 0.0)),
            "brake": float(a.get("brake", 0.0)),
            "throttle": float(a.get("throttle", 0.0)),
            "steer": float(a.get("steer", 0.0)),
            "is_emergency": bool(a.get("is_emergency", False)),
            "current_lane": {
                "road_id": int(a.get("road_id", 0)),
                "lane_id": int(a.get("lane_id", 0)),
                "center_x": float(loc_x),
                "center_y": float(loc_y),
                "speed_limit": 13.89,
            },
        }
        if a.get("type") == "pedestrian":
            pedestrians.append(entity)
        else:
            vehicles.append(entity)

    if weather is None:
        weather = {
            "fog_density": 0.0, "cloudiness": 50.0, "precipitation": 0.0,
            "wetness": 0.0, "sun_altitude_angle": 45.0, "wind_intensity": 0.0,
        }

    extracted = {
        "frame_id": frame_id,
        "elapsed_seconds": frame_id * 0.05,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "traffic_lights": [],
        "weather": weather,
    }

    return {
        "extracted": extracted,
        "delta": None,
        "rule_out": {"violations": []},
    }
