"""
Snapshot 构建器 (v3 §2.4.2-§2.4.3)

从每帧的 raw data（实体列表 + 天气 + 帧信息）构建帧根节点和环境快照。

ScenarioSnapshot: 帧根节点，聚合所有同帧实体
EnvironmentSnapshot: 环境上下文（天气/光照/路面）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.types import EntityType
from stk.scenario.nodes import (
    EnvironmentSnapshot, ScenarioSnapshot, VehicleEntity, PedestrianEntity,
    TrafficLightEntity, RoadElementEntity,
)


class FrameData:
    """单帧原始数据的轻量容器，供 snapshot_builder 处理。

    实际使用中这个数据可以来自 CARLA world.tick() 或录放文件。
    """

    def __init__(
        self,
        frame_id: int,
        elapsed_seconds: float = 0.0,
        delta_seconds: float = 0.05,
        vehicles: Optional[List[Dict[str, Any]]] = None,
        pedestrians: Optional[List[Dict[str, Any]]] = None,
        traffic_lights: Optional[List[Dict[str, Any]]] = None,
        lanes: Optional[List[Dict[str, Any]]] = None,
        weather: Optional[Dict[str, Any]] = None,
        map_name: str = "Town01",
        random_seed: int = 0,
        traffic_density: int = 0,
    ):
        self.frame_id = frame_id
        self.elapsed_seconds = elapsed_seconds
        self.delta_seconds = delta_seconds
        self.vehicles = vehicles or []
        self.pedestrians = pedestrians or []
        self.traffic_lights = traffic_lights or []
        self.lanes = lanes or []
        self.weather = weather or {}
        self.map_name = map_name
        self.random_seed = random_seed
        self.traffic_density = traffic_density


def build_snapshot(frame_data: FrameData) -> Tuple[ScenarioSnapshot, EnvironmentSnapshot]:
    """从 FrameData 构建 ScenarioSnapshot + EnvironmentSnapshot。

    Args:
        frame_data: 单帧原始数据

    Returns:
        (ScenarioSnapshot, EnvironmentSnapshot) 元组
    """
    # EnvironmentSnapshot
    weather = frame_data.weather
    env = EnvironmentSnapshot(
        frame_id=frame_data.frame_id,
        elapsed_seconds=frame_data.elapsed_seconds,
        delta_seconds=frame_data.delta_seconds,
        map_name=frame_data.map_name,
        fog_density=weather.get("fog_density", 0.0),
        cloudiness=weather.get("cloudiness", 0.0),
        precipitation=weather.get("precipitation", 0.0),
        wetness=weather.get("wetness", 0.0),
        sun_altitude_angle=weather.get("sun_altitude_angle", 90.0),
        wind_intensity=weather.get("wind_intensity", 0.0),
        random_seed=frame_data.random_seed,
        traffic_density=frame_data.traffic_density,
        valid_from=frame_data.frame_id,
    )

    # ScenarioSnapshot
    scenario = ScenarioSnapshot(
        frame_id=frame_data.frame_id,
        elapsed_seconds=frame_data.elapsed_seconds,
        n_vehicles=len(frame_data.vehicles),
        n_pedestrians=len(frame_data.pedestrians),
        n_active_rules=0,
        valid_from=frame_data.frame_id,
    )

    return scenario, env


def build_sample_frame(frame_id: int = 0, n_vehicles: int = 3, n_pedestrians: int = 1) -> FrameData:
    """构建一个用于测试的样本帧。

    Args:
        frame_id: 帧号
        n_vehicles: 车辆数
        n_pedestrians: 行人数

    Returns:
        FrameData 实例，可直接输入 build_snapshot
    """
    import random
    random.seed(42)

    vehicles = []
    for i in range(n_vehicles):
        vehicles.append({
            "entity_id": f"veh_{1000 + i}",
            "vehicle_type": "vehicle.audi.tt",
            "location_x": 100.0 + i * 15.0,
            "location_y": 200.0 + i * 2.0,
            "location_z": 0.5,
            "speed": 8.0 + i * 2.0,
            "heading_rad": 0.0,
        })

    pedestrians = []
    for i in range(n_pedestrians):
        pedestrians.append({
            "entity_id": f"ped_{500 + i}",
            "location_x": 110.0,
            "location_y": 210.0 + i * 3.0,
            "location_z": 0.0,
            "speed": 1.2,
            "action": "Walking",
        })

    lanes = [
        {"entity_id": "road_5_lane_1", "road_id": 5, "lane_id": 1, "center_x": 100.0, "center_y": 198.0},
        {"entity_id": "road_5_lane_2", "road_id": 5, "lane_id": 2, "center_x": 100.0, "center_y": 202.0},
    ]

    return FrameData(
        frame_id=frame_id,
        elapsed_seconds=frame_id * 0.05,
        delta_seconds=0.05,
        vehicles=vehicles,
        pedestrians=pedestrians,
        traffic_lights=[],
        lanes=lanes,
        weather={"fog_density": 0, "cloudiness": 10, "precipitation": 0, "wetness": 0,
                 "sun_altitude_angle": 80, "wind_intensity": 0},
        map_name="Town01",
        random_seed=42,
        traffic_density=n_vehicles,
    )
