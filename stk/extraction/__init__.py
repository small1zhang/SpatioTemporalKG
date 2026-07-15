"""CARLA 提取: 真值数据源提取与 API 映射 (§7)."""
from .actor_extractor import extract_all_actors, extract_vehicle, extract_pedestrian
from .waypoint_extractor import extract_waypoints, build_lane_topology
from .trafficlight_extractor import extract_traffic_light, extract_all_traffic_lights
from .sensor_extractor import extract_collision_event, extract_lane_invasion_event, extract_sensor_events
from .weather_extractor import extract_weather, build_environment_snapshot
from .api_mapping import API_MAPPING
from .pipeline import process_frame

__all__ = [
    "extract_all_actors", "extract_vehicle", "extract_pedestrian",
    "extract_waypoints", "build_lane_topology",
    "extract_traffic_light", "extract_all_traffic_lights",
    "extract_collision_event", "extract_lane_invasion_event", "extract_sensor_events",
    "extract_weather", "build_environment_snapshot",
    "API_MAPPING",
    "process_frame",
]