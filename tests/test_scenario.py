"""阶段 2：场景层单元测试。覆盖 nodes、relations、spatial、snapshot_builder、lifecycle_manager 五大模块。"""

import pytest
from stk.ontology.types import EntityType, SceneRelationType
from stk.scenario.nodes import (
    VehicleEntity, PedestrianEntity, TrafficLightEntity,
    RoadElementEntity, EnvironmentSnapshot, ScenarioSnapshot,
)
from stk.scenario.relations import (
    build_relation, in_lane, ahead_of, contains_vehicle, controlled_by,
    weather_context, nearby_pedestrian, beside,
)
from stk.scenario.spatial import (
    compute_in_lane, compute_ahead_of, compute_beside,
    compute_nearby_pedestrian, compute_adjacent_lanes,
)
from stk.scenario.snapshot_builder import build_sample_frame, build_snapshot, FrameData
from stk.scenario.lifecycle_manager import LifecycleManager


class TestNodes:
    """6 类场景节点 (v3 §2.2-§2.4)"""

    def test_vehicle_entity(self):
        v = VehicleEntity(entity_id="veh_001", vehicle_type="vehicle.audi.tt",
                          location_x=10.0, location_y=20.0, speed=15.0, heading_rad=0.5)
        assert v.entity_id == "veh_001"
        assert v.entity_type == EntityType.VEHICLE
        assert v.attrs["speed"] == 15.0
        assert v.attrs["location_x"] == 10.0

    def test_pedestrian_entity(self):
        p = PedestrianEntity(entity_id="ped_001", location_x=5.0, speed=1.2, action="Walking")
        assert p.entity_id == "ped_001"
        assert p.entity_type == EntityType.PEDESTRIAN
        assert p.attrs["action"] == "Walking"

    def test_traffic_light_entity(self):
        tl = TrafficLightEntity(entity_id="tl_001", state="Red", elapsed_time=3.0,
                                affected_lane_ids=[1, 2], location_x=50.0)
        assert tl.entity_id == "tl_001"
        assert tl.attrs["state"] == "Red"
        assert tl.attrs["affected_lane_ids"] == [1, 2]

    def test_road_element_entity(self):
        road = RoadElementEntity(entity_id="road_5_lane_1", road_id=5, lane_id=1,
                                  lane_type="Driving", speed_limit=60.0,
                                  labels=["Lane", "Road"])
        assert road.entity_id == "road_5_lane_1"
        assert road.entity_type == EntityType.LANE
        assert road.in_junction is False  # junction_id 默认 -1
        assert "Lane" in road.labels

    def test_road_element_in_junction(self):
        road = RoadElementEntity(entity_id="road_5_junc", road_id=5, lane_id=1,
                                  junction_id=3)
        assert road.in_junction is True

    def test_environment_snapshot(self):
        env = EnvironmentSnapshot(frame_id=100, elapsed_seconds=5.0, fog_density=20.0)
        assert env.entity_id == "env_100"
        assert env.attrs["fog_density"] == 20.0

    def test_scenario_snapshot(self):
        snap = ScenarioSnapshot(frame_id=100, elapsed_seconds=5.0,
                                 n_vehicles=3, n_pedestrians=1)
        assert snap.entity_id == "frame_100"
        assert snap.attrs["n_vehicles"] == 3


class TestRelations:
    """15 种场景关系工厂函数 (v3 §2.8-§2.10)"""

    def test_in_lane(self):
        r = in_lane("veh_001", "road_5_lane_1", frame_id=100, valid_from=100,
                     distance_to_lane_center=0.3)
        assert r.relation_type == "in_lane"
        assert r.src_id == "veh_001"
        assert r.attrs["distance_to_lane_center"] == 0.3

    def test_ahead_of(self):
        r = ahead_of("veh_001", "veh_002", frame_id=100,
                      longitudinal_distance=12.5, lateral_distance=0.5)
        assert r.relation_type == "ahead_of"
        assert r.attrs["longitudinal_distance"] == 12.5
        assert r.predicate_str() == "AheadOf(veh_001, veh_002, Frame_100)"

    def test_contains_vehicle(self):
        r = contains_vehicle("frame_100", "veh_001", frame_id=100)
        assert r.relation_type == "containsVehicle"

    def test_controlled_by(self):
        r = controlled_by("road_5_lane_1", "tl_001", frame_id=0)
        assert r.relation_type == "controlled_by"

    def test_build_relation_with_extra(self):
        r = build_relation("veh_001", "veh_002", SceneRelationType.AHEAD_OF,
                            frame_id=100, valid_from=100,
                            extra_attrs={"distance": 15.0})
        assert r.relation_type == "ahead_of"
        assert r.attrs["distance"] == 15.0


class TestSpatial:
    """空间关系计算函数 (v3 §2.9)"""

    def test_compute_in_lane(self):
        vehicles = [
            VehicleEntity(entity_id="veh_001", location_x=100.0, location_y=200.0),
            VehicleEntity(entity_id="veh_002", location_x=115.0, location_y=202.0),
        ]
        lanes = [
            {"entity_id": "road_5_lane_1", "road_id": 5, "lane_id": 1,
             "center_x": 100.0, "center_y": 198.0},
            {"entity_id": "road_5_lane_2", "road_id": 5, "lane_id": 2,
             "center_x": 100.0, "center_y": 202.0},
        ]
        rels = compute_in_lane(vehicles, lanes, frame_id=100)
        assert len(rels) > 0
        assert rels[0].relation_type == "in_lane"

    def test_compute_ahead_of(self):
        vehicles = [
            VehicleEntity(entity_id="veh_001", location_x=100.0, location_y=200.0, heading_rad=0.0),
            VehicleEntity(entity_id="veh_002", location_x=120.0, location_y=200.0, heading_rad=0.0),
        ]
        rels = compute_ahead_of(vehicles, frame_id=100)
        # veh_002 在 veh_001 前方20m，方向一致
        assert len(rels) > 0
        assert rels[0].relation_type == "ahead_of"

    def test_compute_beside(self):
        vehicles = [
            VehicleEntity(entity_id="veh_001", location_x=0.0, location_y=0.0, heading_rad=0.0),
            VehicleEntity(entity_id="veh_002", location_x=2.0, location_y=3.0, heading_rad=0.0),
        ]
        rels = compute_beside(vehicles, frame_id=100)
        # veh_002 在 veh_001 右侧2m, 前方3m (lateral=2, longitudinal=3)
        # 阈值: |lateral|<3m AND |longitudinal|<5m -> 应命中
        assert len(rels) > 0, f"beside 应产出关系，实际结果: {len(rels)}"

    def test_compute_nearby_pedestrian(self):
        vehicles = [VehicleEntity(entity_id="veh_001", location_x=0.0, location_y=0.0)]
        peds = [PedestrianEntity(entity_id="ped_001", location_x=10.0, location_y=0.0)]
        rels = compute_nearby_pedestrian(vehicles, peds, frame_id=100, threshold=20.0)
        assert len(rels) == 1
        assert rels[0].relation_type == "nearby_pedestrian"

    def test_compute_nearby_pedestrian_outside(self):
        vehicles = [VehicleEntity(entity_id="veh_001", location_x=0.0, location_y=0.0)]
        peds = [PedestrianEntity(entity_id="ped_001", location_x=50.0, location_y=0.0)]
        rels = compute_nearby_pedestrian(vehicles, peds, frame_id=100, threshold=20.0)
        assert len(rels) == 0  # 距离50m > 20m


class TestSnapshotBuilder:
    """帧快照构建器 (v3 §2.4)"""

    def test_build_sample_frame(self):
        fd = build_sample_frame(frame_id=10, n_vehicles=3, n_pedestrians=1)
        assert fd.frame_id == 10
        assert len(fd.vehicles) == 3
        assert len(fd.pedestrians) == 1

    def test_build_snapshot(self):
        fd = build_sample_frame(frame_id=10)
        scenario, env = build_snapshot(fd)
        assert scenario.entity_id == "frame_10"
        assert scenario.attrs["n_vehicles"] == 3
        assert env.entity_id == "env_10"
        assert env.attrs["map_name"] == "Town01"


class TestLifecycleManager:
    """生命周期管理器 (v3 §2.5)"""

    def test_step_activate(self):
        mgr = LifecycleManager()
        changes = mgr.step(["veh_001", "veh_002"], frame_id=100)
        assert changes["veh_001"] == "activated"
        assert changes["veh_002"] == "activated"
        assert len(mgr.all_active_ids()) == 2

    def test_step_deactivate(self):
        mgr = LifecycleManager()
        mgr.step(["veh_001", "veh_002"], frame_id=100)
        changes = mgr.step(["veh_001"], frame_id=110)
        assert changes["veh_001"] == "stable"
        assert changes["veh_002"] == "deactivated"
        assert mgr.all_active_ids() == ["veh_001"]

    def test_step_stable(self):
        mgr = LifecycleManager()
        mgr.step(["veh_001"], frame_id=100)
        changes = mgr.step(["veh_001"], frame_id=110)
        assert changes["veh_001"] == "stable"

    def test_get_lifecycle(self):
        mgr = LifecycleManager()
        mgr.step(["veh_001"], frame_id=100)
        lc = mgr.get("veh_001")
        assert lc is not None
        assert lc.is_active_at(100) is True

    def test_to_dict(self):
        mgr = LifecycleManager()
        mgr.step(["veh_001"], frame_id=100, entity_type_map={"veh_001": "Vehicle"})
        d = mgr.to_dict()
        assert "veh_001" in d
        assert d["veh_001"]["entity_id"] == "veh_001"

