"""
场景层节点定义 (v3 §2.2-§2.4)

本模块定义场景层 6 类节点：
  - VehicleEntity:      车辆实体，对应 carla.Vehicle, 18+ 个字段
  - PedestrianEntity:   行人实体，对应 carla.Walker
  - TrafficLightEntity: 信号灯实体，对应 carla.TrafficLight
  - RoadElementEntity:  道路元素 (Road/Lane/Junction)，对应 carla.Waypoint
  - EnvironmentSnapshot:环境快照（天气/光照/路面）
  - ScenarioSnapshot:   帧根节点，每帧唯一的聚合根

每个节点继承 ontology.base.BaseEntity，通过 attrs 字典存储各自的属性字段。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stk.ontology.types import EntityType
from stk.ontology.entity import BaseEntity


class VehicleEntity(BaseEntity):
    """车辆实体 (v3 §2.3.1)

    对应 carla.Vehicle，18 个字段全部通过 attrs 存储。
    CARLA 工厂方法: from_carla_actor(actor, is_ego=False) -> VehicleEntity
    """

    def __init__(
        self,
        entity_id: str,
        vehicle_type: str = "",
        is_ego: bool = False,
        location_x: float = 0.0, location_y: float = 0.0, location_z: float = 0.0,
        velocity_x: float = 0.0, velocity_y: float = 0.0, velocity_z: float = 0.0,
        acceleration_x: float = 0.0, acceleration_y: float = 0.0, acceleration_z: float = 0.0,
        speed: float = 0.0, speed_kmh: float = 0.0,
        heading_rad: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
        bbox_extent_x: float = 0.0, bbox_extent_y: float = 0.0, bbox_extent_z: float = 0.0,
        throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0,
        is_alive: bool = True,
        valid_from: int = 0, valid_to: Optional[int] = None,
        confidence: float = 1.0,
    ):
        attrs: Dict[str, Any] = {
            "vehicle_type": vehicle_type,
            "is_ego": is_ego,
            "location_x": location_x, "location_y": location_y, "location_z": location_z,
            "velocity_x": velocity_x, "velocity_y": velocity_y, "velocity_z": velocity_z,
            "acceleration_x": acceleration_x, "acceleration_y": acceleration_y, "acceleration_z": acceleration_z,
            "speed": speed, "speed_kmh": speed_kmh,
            "heading_rad": heading_rad, "pitch": pitch, "roll": roll,
            "bbox_extent_x": bbox_extent_x, "bbox_extent_y": bbox_extent_y, "bbox_extent_z": bbox_extent_z,
            "throttle": throttle, "brake": brake, "steer": steer,
            "is_alive": is_alive,
        }
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.VEHICLE,
            valid_from=valid_from,
            valid_to=valid_to,
            attrs=attrs,
            confidence=confidence,
        )

    @classmethod
    def from_carla_actor(cls, actor, is_ego: bool = False, frame_id: int = 0) -> "VehicleEntity":
        """从 CARLA actor 创建 VehicleEntity（v3 §7.1.1）。"""
        location = actor.get_location()
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        bbox = actor.get_bounding_box()
        control = actor.get_control()
        import math
        heading_rad = math.radians(actor.get_transform().rotation.yaw)
        speed = velocity.magnitude()

        return cls(
            entity_id=f"veh_{actor.id}",
            vehicle_type=actor.type_id,
            is_ego=is_ego,
            location_x=location.x, location_y=location.y, location_z=location.z,
            velocity_x=velocity.x, velocity_y=velocity.y, velocity_z=velocity.z,
            acceleration_x=acceleration.x, acceleration_y=acceleration.y, acceleration_z=acceleration.z,
            speed=speed, speed_kmh=speed * 3.6,
            heading_rad=heading_rad,
            bbox_extent_x=bbox.extent.x, bbox_extent_y=bbox.extent.y, bbox_extent_z=bbox.extent.z,
            throttle=control.throttle, brake=control.brake, steer=control.steer,
            is_alive=actor.is_alive,
            valid_from=frame_id,
        )


class PedestrianEntity(BaseEntity):
    """行人实体 (v3 §2.3.2)

    对应 carla.Walker。
    """

    def __init__(
        self,
        entity_id: str,
        location_x: float = 0.0, location_y: float = 0.0, location_z: float = 0.0,
        velocity_x: float = 0.0, velocity_y: float = 0.0, velocity_z: float = 0.0,
        speed: float = 0.0, heading_rad: float = 0.0,
        bbox_extent_x: float = 0.0, bbox_extent_y: float = 0.0, bbox_extent_z: float = 0.0,
        action: str = "Idle",
        is_on_crosswalk: bool = False,
        is_on_sidewalk: bool = False,
        is_alive: bool = True,
        valid_from: int = 0, valid_to: Optional[int] = None,
    ):
        attrs: Dict[str, Any] = {
            "location_x": location_x, "location_y": location_y, "location_z": location_z,
            "velocity_x": velocity_x, "velocity_y": velocity_y, "velocity_z": velocity_z,
            "speed": speed, "heading_rad": heading_rad,
            "bbox_extent_x": bbox_extent_x, "bbox_extent_y": bbox_extent_y, "bbox_extent_z": bbox_extent_z,
            "action": action,
            "is_on_crosswalk": is_on_crosswalk, "is_on_sidewalk": is_on_sidewalk,
            "is_alive": is_alive,
        }
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.PEDESTRIAN,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )


class TrafficLightEntity(BaseEntity):
    """信号灯实体 (v3 §2.3.3)

    对应 carla.TrafficLight。
    """

    def __init__(
        self,
        entity_id: str,
        state: str = "Green",
        elapsed_time: float = 0.0,
        location_x: float = 0.0, location_y: float = 0.0, location_z: float = 0.0,
        rotation_yaw: float = 0.0,
        affected_lane_ids: Optional[List[int]] = None,
        valid_from: int = 0, valid_to: Optional[int] = None,
    ):
        attrs: Dict[str, Any] = {
            "state": state, "elapsed_time": elapsed_time,
            "location_x": location_x, "location_y": location_y, "location_z": location_z,
            "rotation_yaw": rotation_yaw,
            "affected_lane_ids": affected_lane_ids or [],
        }
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.TRAFFIC_LIGHT,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )


class RoadElementEntity(BaseEntity):
    """道路元素实体 (v3 §2.4.1)

    一条车道一个节点，多标签 (Lane/Road/Junction) 通过 labels 属性支持。
    对应 carla.Waypoint。
    """

    def __init__(
        self,
        entity_id: str,
        road_id: int = 0,
        lane_id: int = 0,
        junction_id: int = -1,
        lane_type: str = "Driving",
        lane_width: float = 3.5,
        speed_limit: float = 60.0,
        center_x: float = 0.0, center_y: float = 0.0, center_z: float = 0.0,
        heading_rad: float = 0.0,
        left_lane_id: Optional[int] = None,
        right_lane_id: Optional[int] = None,
        has_traffic_light: bool = False,
        labels: Optional[List[str]] = None,
        valid_from: int = 0, valid_to: Optional[int] = None,
    ):
        attrs: Dict[str, Any] = {
            "road_id": road_id, "lane_id": lane_id, "junction_id": junction_id,
            "lane_type": lane_type, "lane_width": lane_width, "speed_limit": speed_limit,
            "center_x": center_x, "center_y": center_y, "center_z": center_z,
            "heading_rad": heading_rad,
            "left_lane_id": left_lane_id, "right_lane_id": right_lane_id,
            "has_traffic_light": has_traffic_light,
        }
        super().__init__(
            entity_id=entity_id,
            entity_type=EntityType.LANE,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
            labels=labels or ["Lane"],
        )

    @property
    def in_junction(self) -> bool:
        return self.attrs.get("junction_id", -1) != -1


class EnvironmentSnapshot(BaseEntity):
    """环境快照 (v3 §2.4.2)

    每帧的环境上下文，场景层的全局属性节点。
    """

    def __init__(
        self,
        frame_id: int,
        elapsed_seconds: float = 0.0, delta_seconds: float = 0.0,
        map_name: str = "",
        fog_density: float = 0.0, cloudiness: float = 0.0,
        precipitation: float = 0.0, wetness: float = 0.0,
        sun_altitude_angle: float = 90.0, wind_intensity: float = 0.0,
        random_seed: int = 0, traffic_density: int = 0,
        valid_from: int = 0, valid_to: Optional[int] = None,
    ):
        attrs: Dict[str, Any] = {
            "frame_id": frame_id,
            "elapsed_seconds": elapsed_seconds, "delta_seconds": delta_seconds,
            "map_name": map_name,
            "fog_density": fog_density, "cloudiness": cloudiness,
            "precipitation": precipitation, "wetness": wetness,
            "sun_altitude_angle": sun_altitude_angle, "wind_intensity": wind_intensity,
            "random_seed": random_seed, "traffic_density": traffic_density,
        }
        super().__init__(
            entity_id=f"env_{frame_id}",
            entity_type=EntityType.ENV_SNAPSHOT,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )


class ScenarioSnapshot(BaseEntity):
    """帧根节点 (v3 §2.4.3)

    每帧唯一的聚合根，连接所有同帧实体。
    """

    def __init__(
        self,
        frame_id: int,
        elapsed_seconds: float = 0.0,
        n_vehicles: int = 0, n_pedestrians: int = 0,
        n_active_rules: int = 0,
        valid_from: int = 0, valid_to: Optional[int] = None,
    ):
        attrs: Dict[str, Any] = {
            "frame_id": frame_id,
            "elapsed_seconds": elapsed_seconds,
            "n_vehicles": n_vehicles, "n_pedestrians": n_pedestrians,
            "n_active_rules": n_active_rules,
        }
        super().__init__(
            entity_id=f"frame_{frame_id}",
            entity_type=EntityType.SCENE_SNAPSHOT,
            valid_from=valid_from, valid_to=valid_to,
            attrs=attrs,
        )
