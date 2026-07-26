# -*- coding: utf-8 -*-
"""Actor 提取: 从 CARLA world 快照生成车辆/行人 dict (v3 §7.1)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def _derive_vehicle_category(vehicle_type: str) -> str:
    """从 CARLA vehicle type_id 派生 vehicle_category.

    Returns 值: car, bicycle, motorcycle, bus_or_truck, emergency, truck_trailer.
    不匹配时回退 "car".  用于按类别差异化 ROI 半径.
    """
    vt = vehicle_type.lower()
    # Bicycle
    if "gazelle" in vt or "bicycle" in vt or "bikehiker" in vt:
        return "bicycle"
    # Motorcycle
    if any(kw in vt for kw in ("kawasaki", "harley", "yamaha", "vespa", "ninja")):
        return "motorcycle"
    # Bus
    if "bus" in vt or "minivan" in vt or "microbus" in vt:
        return "bus_or_truck"
    # Truck / trailer
    if any(kw in vt for kw in ("truck", "carlamotors", "carlacola", "box", "tractor", "semi", "trailer")):
        return "bus_or_truck"
    # Emergency
    if any(kw in vt for kw in ("police", "ambulance", "fire", "swat")):
        return "emergency"
    # Van
    if "van" in vt or "jeep" in vt or "suv" in vt or "landrover" in vt:
        return "car"
    # Default: car
    return "car"


def extract_vehicle(actor_data: dict) -> Dict[str, Any]:
    """从 CARLA Vehicle actor dict 提取.

    支持两种输入布局：
      - CARLA 原生（嵌套 location/velocity/acceleration/bbox_extent/control 字典）
      - 场景库扁平形式（location_x, location_y, speed, heading_rad 等顶层字段）
    """
    loc = actor_data.get("location", {})
    vel = actor_data.get("velocity", {})
    acc = actor_data.get("acceleration", {})
    bbox = actor_data.get("bbox_extent", {})
    ctrl = actor_data.get("control", {})
    # 扁平字段回退（适配 scenario_library.FrameData.vehicles）
    loc_x = loc.get("x", actor_data.get("location_x", 0.0))
    loc_y = loc.get("y", actor_data.get("location_y", 0.0))
    loc_z = loc.get("z", actor_data.get("location_z", 0.0))
    vel_x = vel.get("x", actor_data.get("velocity_x", 0.0))
    vel_y = vel.get("y", actor_data.get("velocity_y", 0.0))
    vel_z = vel.get("z", actor_data.get("velocity_z", 0.0))
    acc_x = acc.get("x", actor_data.get("acceleration_x", 0.0))
    acc_y = acc.get("y", actor_data.get("acceleration_y", 0.0))
    acc_z = acc.get("z", actor_data.get("acceleration_z", 0.0))
    bbox_x = bbox.get("x", actor_data.get("bbox_extent_x", 0.0))
    bbox_y = bbox.get("y", actor_data.get("bbox_extent_y", 0.0))
    bbox_z = bbox.get("z", actor_data.get("bbox_extent_z", 0.0))
    brk = ctrl.get("brake", actor_data.get("brake", 0.0))
    thr = ctrl.get("throttle", actor_data.get("throttle", 0.0))
    ste = ctrl.get("steer", actor_data.get("steer", 0.0))
    vehicle_type = actor_data.get("type_id") or actor_data.get("vehicle_type") or "vehicle.*"
    # 兼容场景库字段名（'entity_id'）与 CARLA 原始字段（'id'）：优先使用已有 entity_id；
    # 缺失时回退到 'id'，再缺失则使用 GLOBAL_ID_GENERATOR 生成。
    from stk.ontology.namespace import GLOBAL_ID_GENERATOR
    entity_id = actor_data.get("entity_id") or actor_data.get("id") or \
        GLOBAL_ID_GENERATOR.generate("Vehicle")
    return {
        "entity_id": entity_id,
        "entity_type": "Vehicle",
        "vehicle_type": vehicle_type,
        "vehicle_category": _derive_vehicle_category(vehicle_type),
        "is_ego": actor_data.get("is_ego", False),
        "location_x": loc_x,
        "location_y": loc_y,
        "location_z": loc_z,
        "velocity_x": vel_x,
        "velocity_y": vel_y,
        "velocity_z": vel_z,
        "acceleration_x": acc_x,
        "acceleration_y": acc_y,
        "acceleration_z": acc_z,
        "speed": actor_data.get("speed", 0.0),
        "speed_kmh": actor_data.get("speed", 0.0) * 3.6,
        "heading_rad": actor_data.get("heading_rad", 0.0),
        "pitch": actor_data.get("pitch", 0.0),
        "roll": actor_data.get("roll", 0.0),
        "bbox_extent_x": bbox_x,
        "bbox_extent_y": bbox_y,
        "bbox_extent_z": bbox_z,
        "brake": brk,
        "throttle": thr,
        "steer": ste,
        "is_alive": actor_data.get("is_alive", True),
        "is_emergency": actor_data.get("is_emergency", False),
    }


def extract_pedestrian(actor_data: dict) -> Dict[str, Any]:
    """从 CARLA Walker/Pedestrian actor dict 提取.

    支持两种输入布局（同 extract_vehicle 注释）。
    """
    loc = actor_data.get("location", {})
    vel = actor_data.get("velocity", {})
    bbox = actor_data.get("bbox_extent", {})
    loc_x = loc.get("x", actor_data.get("location_x", 0.0))
    loc_y = loc.get("y", actor_data.get("location_y", 0.0))
    loc_z = loc.get("z", actor_data.get("location_z", 0.0))
    vel_x = vel.get("x", actor_data.get("velocity_x", 0.0))
    vel_y = vel.get("y", actor_data.get("velocity_y", 0.0))
    vel_z = vel.get("z", actor_data.get("velocity_z", 0.0))
    bbox_x = bbox.get("x", actor_data.get("bbox_extent_x", 0.0))
    bbox_y = bbox.get("y", actor_data.get("bbox_extent_y", 0.0))
    bbox_z = bbox.get("z", actor_data.get("bbox_extent_z", 0.0))
    # 兼容场景库字段名（'entity_id'）与 CARLA 原始字段（'id'）：
    from stk.ontology.namespace import GLOBAL_ID_GENERATOR
    entity_id = actor_data.get("entity_id") or actor_data.get("id") or \
        GLOBAL_ID_GENERATOR.generate("Pedestrian")
    return {
        "entity_id": entity_id,
        "entity_type": "Pedestrian",
        "location_x": loc_x,
        "location_y": loc_y,
        "location_z": loc_z,
        "velocity_x": vel_x,
        "velocity_y": vel_y,
        "velocity_z": vel_z,
        "speed": actor_data.get("speed", 0.0),
        "heading_rad": actor_data.get("heading_rad", 0.0),
        "bbox_extent_x": bbox.get("x", 0.0),
        "bbox_extent_y": bbox.get("y", 0.0),
        "bbox_extent_z": bbox.get("z", 0.0),
        "is_on_crosswalk": actor_data.get("is_on_crosswalk", False),
        "is_on_sidewalk": actor_data.get("is_on_sidewalk", False),
        "action": actor_data.get("action", "Idle"),
        "is_alive": actor_data.get("is_alive", True),
    }


def extract_all_actors(frame_data: dict) -> Dict[str, List[Dict]]:
    """从帧数据提取所有 actors，返回 {vehicles, pedestrians}."""
    return {
        "vehicles": [extract_vehicle(a) for a in frame_data.get("actors", [])
                     if a.get("type", "").startswith("vehicle")],
        "pedestrians": [extract_pedestrian(a) for a in frame_data.get("actors", [])
                        if a.get("type", "").startswith("walker")],
    }