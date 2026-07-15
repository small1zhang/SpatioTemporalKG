# -*- coding: utf-8 -*-
"""阶段七：CARLA 数据提取 — 6 类提取器 + pipeline 测试 (mock 模式)."""
from __future__ import annotations
import pytest

from stk.extraction.actor_extractor import extract_vehicle, extract_pedestrian, extract_all_actors
from stk.extraction.waypoint_extractor import extract_waypoints, build_lane_topology
from stk.extraction.trafficlight_extractor import extract_traffic_light, extract_all_traffic_lights
from stk.extraction.sensor_extractor import extract_collision_event, extract_lane_invasion_event, extract_sensor_events
from stk.extraction.weather_extractor import extract_weather, build_environment_snapshot
from stk.extraction.api_mapping import API_MAPPING
from stk.extraction.pipeline import process_frame


# ——— T7.3 actor_extractor.py ———

class TestActorExtractor:
    def test_extract_vehicle(self):
        actor_data = {"id": "veh_1", "type": "vehicle.tesla.model3",
                      "is_ego": True, "speed": 10.0,
                      "location": {"x": 10.0, "y": 20.0, "z": 0.0},
                      "velocity": {"x": 5.0, "y": 0.0, "z": 0.0},
                      "heading_rad": 0.0, "brake": 0.0, "throttle": 0.5, "steer": 0.0}
        v = extract_vehicle(actor_data)
        assert v["entity_type"] == "Vehicle"
        assert v["speed"] == 10.0
        assert v["location_x"] == 10.0

    def test_extract_pedestrian(self):
        actor_data = {"id": "ped_1", "type": "walker.pedestrian",
                      "speed": 1.0, "is_on_crosswalk": True,
                      "location": {"x": 5.0, "y": 0.0, "z": 0.0},
                      "velocity": {"x": 0.5, "y": 0.0},
                      "heading_rad": 0.0}
        p = extract_pedestrian(actor_data)
        assert p["entity_type"] == "Pedestrian"
        assert p["is_on_crosswalk"] is True

    def test_extract_all_actors(self):
        frame = {"actors": [
            {"id": "veh_1", "type": "vehicle.tesla.model3", "is_ego": True,
             "speed": 10.0, "location": {"x": 1.0}, "velocity": {}},
            {"id": "ped_1", "type": "walker.pedestrian", "speed": 1.0,
             "location": {"x": 2.0}, "velocity": {},
             "is_on_crosswalk": False},
        ]}
        result = extract_all_actors(frame)
        assert len(result["vehicles"]) == 1
        assert len(result["pedestrians"]) == 1


# ——— T7.4 waypoint_extractor.py ———

class TestWaypointExtractor:
    def test_extract_waypoints(self):
        wps = [
            {"road_id": 1, "lane_id": 2, "junction_id": -1, "lane_type": "Driving",
             "x": 0.0, "y": 0.0, "z": 0.0, "lane_width": 3.5, "speed_limit": 60.0},
        ]
        roads = extract_waypoints(wps)
        assert len(roads) == 1
        assert roads[0]["entity_id"] == "road_1_lane_2"

    def test_build_lane_topology(self):
        wps = [{"road_id": 1, "lane_id": 2, "left_lane_id": 1, "right_lane_id": 3}]
        rels = build_lane_topology(wps)
        assert len(rels) == 2
        assert rels[0]["relation_type"] == "adjacent_lane"
        assert rels[0]["src_id"] == "road_1_lane_2"


# ——— T7.5 trafficlight_extractor.py ———

class TestTrafficLightExtractor:
    def test_extract_traffic_light(self):
        tl = {"id": 42, "state": "Red", "elapsed_time": 10.0,
              "location": {"x": 100.0, "y": 200.0, "z": 0.0},
              "affected_lane_ids": [1, 2]}
        t = extract_traffic_light(tl)
        assert t["entity_id"] == "tl_42"
        assert t["state"] == "Red"
        assert t["affected_lane_ids"] == [1, 2]

    def test_extract_all_traffic_lights(self):
        result = extract_all_traffic_lights([{"id": 1, "state": "Green",
                                              "location": {}, "affected_lane_ids": []}])
        assert len(result) == 1
        assert result[0]["state"] == "Green"


# ——— T7.6 sensor_extractor.py ———

class TestSensorExtractor:
    def test_extract_collision(self):
        ev = {"event_type": "Collision", "frame_id": 5, "ego_id": "veh_1",
              "other_id": "veh_2", "impulse": 500.0,
              "location_x": 10.0, "location_y": 0.0}
        c = extract_collision_event(ev)
        assert c["event_type"] == "Collision"
        assert c["impulse"] == 500.0

    def test_extract_lane_invasion(self):
        ev = {"event_type": "LaneInvasion", "frame_id": 5, "actor_id": "veh_1",
              "crossed_lane_markings": ["Solid"]}
        li = extract_lane_invasion_event(ev)
        assert li["event_type"] == "LaneInvasion"

    def test_extract_sensor_events(self):
        events = [
            {"event_type": "Collision", "frame_id": 0, "impulse": 100.0},
            {"event_type": "LaneInvasion", "frame_id": 0},
        ]
        result = extract_sensor_events(events)
        assert len(result["collisions"]) == 1
        assert len(result["lane_invasions"]) == 1


# ——— T7.7 weather_extractor.py ———

class TestWeatherExtractor:
    def test_extract_weather(self):
        w = extract_weather({"fog_density": 50.0, "cloudiness": 0.5,
                             "precipitation": 0.3, "wetness": 0.8,
                             "sun_altitude_angle": 45.0, "wind_intensity": 10.0})
        assert w["fog_density"] == 50.0
        assert w["sun_altitude_angle"] == 45.0

    def test_build_environment_snapshot(self):
        snap = build_environment_snapshot({"fog_density": 10.0}, frame_id=3)
        assert snap["frame_id"] == 3
        assert snap["fog_density"] == 10.0


# ——— T7.8 api_mapping.py ———

class TestApiMapping:
    def test_has_mappings(self):
        assert len(API_MAPPING) >= 10


# ——— T7.9 pipeline.py ———

class TestPipeline:
    def test_process_frame(self):
        frame = {
            "frame_id": 0,
            "elapsed_seconds": 0.0,
            "actors": [
                {"id": "veh_1", "type": "vehicle.tesla.model3", "is_ego": True,
                 "speed": 10.0, "location": {"x": 1.0}, "velocity": {}},
            ],
            "waypoints": [{"road_id": 1, "lane_id": 2}],
            "traffic_lights": [{"id": 42, "state": "Green", "location": {}, "affected_lane_ids": []}],
            "events": [],
            "weather": {"fog_density": 0.0},
        }
        result = process_frame(frame)
        assert "frame_id" in result
        assert result["frame_id"] == 0
        assert len(result["vehicles"]) == 1
        assert len(result["lanes"]) == 1
        assert len(result["traffic_lights"]) == 1