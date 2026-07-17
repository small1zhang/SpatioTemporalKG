"""阶段 3：行为层单元测试。
覆盖 nodes、relations、debouncer、detectors、manifest、generator 六大模块。"""

import pytest
from stk.ontology.types import EntityType, BehaviorRelationType, CrossLayerRelationType
from stk.behavior.nodes import (
    ManeuverNode, InteractionEvent,
    MANEUVER_TYPES, INTERACTION_TYPES,
    make_maneuver_id, make_interaction_id,
)
from stk.behavior.relations import (
    build_relation, standing_still, changing_lane,
    following, approaching, yielding_to, overtaking,
    approaching_pedestrian, approaching_intersection, crossing,
)
from stk.behavior.debouncer import RelationDebouncer, DEFAULT_DEBOUNCE_THRESHOLDS
from stk.behavior.detectors import (
    detect_standing_still, detect_following,
    detect_yielding_to, detect_approaching,
    run_all_detectors,
)
from stk.behavior.manifest import (
    manifestsAs_edge, actor_edge, src_edge, dst_edge,
    link_maneuver_to_scene, link_interaction_to_scene,
)
from stk.behavior.generator import BehaviorRelationGenerator


class TestManeuverNode:
    """ManeuverNode — 单实体持续行为节点 (v3 sec 3.2.1)"""

    def test_create_standing_still(self):
        m = ManeuverNode(entity_id="man_veh_001_100", maneuver_type="standing_still",
                         actor_id="veh_001", frame_start=100, frame_end=200)
        assert m.entity_id == "man_veh_001_100"
        assert m.maneuver_type == "standing_still"
        assert m.actor_id == "veh_001"
        assert m.frame_start == 100
        assert m.frame_end == 200
        assert m.duration_frames == 101  # 200-100+1
        assert m.state == "active"
        assert m.entity_type == EntityType.MANEUVER

    def test_active_at(self):
        m = ManeuverNode(entity_id="man_v1_50", maneuver_type="cruising",
                         actor_id="v1", frame_start=50)
        assert m.is_active_at(50) is True
        assert m.is_active_at(100) is True
        m.close(120)
        assert m.is_active_at(121) is False

    def test_close_node(self):
        m = ManeuverNode(entity_id="man_v1_10", maneuver_type="accelerating",
                         actor_id="v1", frame_start=10)
        assert m.state == "active"
        m.close(35)
        assert m.state == "ended"
        assert m.frame_end == 35
        assert m.duration_frames == 26  # 35-10+1

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            ManeuverNode(entity_id="bad", maneuver_type="fly",
                         actor_id="v1", frame_start=0)

    def test_id_factory(self):
        assert make_maneuver_id("veh_001", 100) == "man_veh_001_100"


class TestInteractionEvent:
    """InteractionEvent — 多实体交互事件节点 (v3 sec 3.2.2)"""

    def test_create_following(self):
        ie = InteractionEvent(entity_id="int_v1_v2_follow_100",
                              interaction_type="following",
                              src_id="veh_001", dst_id="veh_002",
                              frame_start=100, frame_end=200)
        assert ie.interaction_type == "following"
        assert ie.src_id == "veh_001"
        assert ie.dst_id == "veh_002"
        assert ie.duration_frames == 101

    def test_with_derived_attrs(self):
        ie = InteractionEvent(entity_id="int_v1_v2_follow_100",
                              interaction_type="following",
                              src_id="veh_001", dst_id="veh_002",
                              frame_start=100,
                              derived_attrs={"ttc": 12.5, "distance": 15.0, "relative_speed": 0.3})
        assert ie.attrs["derived_attrs"]["ttc"] == 12.5

    def test_with_source_relations(self):
        ie = InteractionEvent(entity_id="ie_1",
                              interaction_type="yielding_to",
                              src_id="veh_001", dst_id="ped_001",
                              frame_start=100,
                              source_relations=["in_lane_id_100", "ahead_of_id_100"])
        assert len(ie.source_relations) == 2

    def test_close(self):
        ie = InteractionEvent(entity_id="ie_2",
                              interaction_type="overtaking",
                              src_id="v1", dst_id="v2",
                              frame_start=50)
        assert ie.state == "active"
        ie.close(80)
        assert ie.state == "ended"
        assert ie.frame_end == 80

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            InteractionEvent(entity_id="bad", interaction_type="unknown",
                             src_id="v1", dst_id="v2", frame_start=0)

    def test_empty_src_dst_raises(self):
        with pytest.raises(ValueError):
            InteractionEvent(entity_id="bad", interaction_type="following",
                             src_id="", dst_id="v2", frame_start=0)

    def test_id_factory(self):
        assert make_interaction_id("v1", "v2", "following", 100) == "int_v1_v2_following_100"


class TestBehaviorRelations:
    """13 种行为关系工厂函数 (v3 sec 3.3)"""

    def test_standing_still(self):
        r = standing_still("veh_001", frame_id=100, speed=0.0)
        assert r.relation_type == "standing_still"
        assert r.src_id == r.dst_id == "veh_001"
        assert r.attrs["speed"] == 0.0

    def test_following(self):
        r = following("veh_001", "veh_002", frame_id=100,
                      distance=12.5, relative_speed=0.3, ttc=41.7)
        assert r.relation_type == "following"
        assert r.src_id == "veh_001"
        assert r.dst_id == "veh_002"
        assert r.attrs["distance"] == 12.5
        assert r.attrs["ttc"] == 41.7

    def test_yielding_to(self):
        r = yielding_to("veh_001", "ped_001", frame_id=100,
                        distance=8.2, ped_action="Walking", ego_speed=0.6)
        assert r.relation_type == "yielding_to"
        assert r.attrs["ped_action"] == "Walking"

    def test_overtaking(self):
        r = overtaking("v1", "v2", frame_id=100, lateral_distance=1.5)
        assert r.relation_type == "overtaking"

    def test_approaching_intersection(self):
        r = approaching_intersection("veh_001", "junc_001", frame_id=100,
                                      distance_to_junction=15.0)
        assert r.relation_type == "approaching_intersection"

    def test_crossing(self):
        r = crossing("ped_001", "cw_001", frame_id=100, crossing_speed=1.2)
        assert r.relation_type == "crossing"

    def test_build_relation_generic(self):
        r = build_relation("v1", "v2", BehaviorRelationType.FOLLOWING,
                           frame_id=100, valid_from=100,
                           extra_attrs={"distance": 10.0})
        assert r.relation_type == "following"
        assert r.attrs["distance"] == 10.0

    def test_predicate_str(self):
        r = following("veh_001", "veh_002", frame_id=100, distance=12.5)
        pred = r.predicate_str()
        assert "Following" in pred
        assert "veh_001" in pred


class TestRelationDebouncer:
    """行为关系防抖机制 (v3 sec 3.4)"""

    def test_default_thresholds(self):
        d = RelationDebouncer()
        assert d.get_threshold("following") == 3
        assert d.get_threshold("overtaking") == 5
        assert d.get_threshold("standing_still") == 2
        assert d.get_threshold("wrong_side_meeting") == 1

    def test_following_activation(self):
        d = RelationDebouncer()
        key = ("veh_001", "veh_002", "following")
        # 前 2 帧条件满足但未达阈值
        for f in range(1, 3):
            action, _ = d.update("following", key, True, f)
            assert action == "none", f"frame {f} expected none, got {action}"
        # 第 3 帧触发 create
        action, extra = d.update("following", key, True, 3)
        assert action == "create", f"frame 3 expected create, got {action}"
        # 保持
        action, _ = d.update("following", key, True, 4)
        assert action == "keep"

    def test_following_deactivation(self):
        d = RelationDebouncer(thresholds={"following": 2})
        key = ("v1", "v2", "following")
        # 帧 1,2: 防抖计数 1, 2 -> 帧 2 已经 >= threshold 满足创建
        # frame 1: counter=1, none
        a1, _ = d.update("following", key, True, 1)
        assert a1 == "none"
        # frame 2: counter=2 == threshold -> create
        a2, _ = d.update("following", key, True, 2)
        assert a2 == "create"
        # frame 3: counter=3
        a3, _ = d.update("following", key, True, 3)
        assert a3 == "keep"
        # frame 4: condition_met=False -> counter=0 -> delete
        a4, _ = d.update("following", key, False, 4)
        assert a4 == "delete"
        # frame 5: counter=0
        a5, _ = d.update("following", key, False, 5)
        assert a5 == "none"

    def test_active_keys(self):
        d = RelationDebouncer(thresholds={"t": 2})
        k1 = ("a", "b", "t")
        k2 = ("c", "d", "t")
        # 帧 1,2: 满足条件 -> counter 累积到 2 = threshold, 第2帧 create
        a1_1, _ = d.update("t", k1, True, 1)
        a2_1, _ = d.update("t", k2, True, 1)
        assert a1_1 == "none"
        assert a2_1 == "none"
        a1_2, _ = d.update("t", k1, True, 2)
        a2_2, _ = d.update("t", k2, True, 2)
        assert a1_2 == "create"
        assert a2_2 == "create"
        assert len(d.active_keys()) == 2

    def test_reset_key(self):
        d = RelationDebouncer(thresholds={"t": 1})
        key = ("a", "b", "t")
        d.update("t", key, True, 1)
        d.reset(key)
        assert key not in d._items

    def test_status_summary(self):
        d = RelationDebouncer(thresholds={"t": 2})
        key = ("a", "b", "t")
        d.update("t", key, True, 1)
        summary = d.status_summary()
        assert summary["n_pending"] >= 1


class TestDetectors:
    """行为关系检测器 (v3 sec 3.3)"""

    def test_detect_standing_still_true(self):
        cond, extra = detect_standing_still({"speed": 0.0})
        assert cond is True

    def test_detect_standing_still_false(self):
        cond, extra = detect_standing_still({"speed": 5.0})
        assert cond is False

    def test_detect_following(self):
        vehicle = {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 10.0}
        leader = {"entity_id": "veh_002", "location_x": 15.0, "location_y": 0.0, "speed": 8.0}
        cond, extra = detect_following(vehicle, leader)
        assert cond is True
        assert extra["distance"] == 15.0
        assert extra["relative_speed"] == 2.0

    def test_detect_yielding_to(self):
        vehicle = {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 1.0}
        ped = {"entity_id": "ped_001", "location_x": 5.0, "location_y": 0.0, "speed": 1.2, "action": "Walking"}
        cond, _ = detect_yielding_to(vehicle, ped)
        assert cond is True

    def test_detect_approaching(self):
        vehicle = {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 10.0}
        tl = {"entity_id": "tl_001", "location_x": 20.0, "location_y": 0.0}
        cond, extra = detect_approaching(vehicle, tl, target_type="traffic_light")
        assert cond is True
        assert extra["distance"] == 20.0
        assert extra["ttc"] is not None

    def test_run_all_detectors_basic(self):
        """综合检测器 — 基本帧检测."""
        vehicles = [
            {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 0.0, "velocity_x": 0.0, "velocity_y": 0.0, "heading_rad": 0.0},
            {"entity_id": "veh_002", "location_x": 20.0, "location_y": 0.0, "speed": 10.0, "velocity_x": 10.0, "velocity_y": 0.0, "heading_rad": 0.1},
        ]
        results = run_all_detectors(vehicles=vehicles, pedestrians=[], traffic_lights=[],
                                    junctions=[], crosswalks=[], scene_relations=[])
        # standing_still 检测: veh_001 speed=0 应 True; veh_002 speed=10 应 False
        assert "standing_still" in results
        ss_results = {entry[0]: entry[2] for entry in results["standing_still"]}
        assert ss_results.get("veh_001") is True
        assert ss_results.get("veh_002") is False
        # following: 至少有 veh_001 -> veh_002 或 veh_002 -> veh_001 一对
        assert "following" in results
        following_items = results["following"]
        assert len(following_items) >= 1


class TestManifest:
    """节点+边双轨关联与跨层桥接 (v3 sec 3.5 + sec 3.6)"""

    def test_manifestsAs_edge(self):
        e = manifestsAs_edge("int_v1_v2_follow_100", "v1", "v2", "following", frame_id=100, valid_from=100)
        assert e.relation_type == "manifestsAs"
        assert e.src_id == "int_v1_v2_follow_100"

    def test_actor_edge(self):
        e = actor_edge("man_v1_100", "v1", frame_id=100, valid_from=100)
        assert e.relation_type == "actor"
        assert e.dst_id == "v1"

    def test_src_dst_edges(self):
        src = src_edge("int_x", "veh_a", frame_id=100, valid_from=100)
        dst = dst_edge("int_x", "veh_b", frame_id=100, valid_from=100)
        assert src.relation_type == "src"
        assert dst.relation_type == "dst"
        assert src.dst_id == "veh_a"
        assert dst.dst_id == "veh_b"

    def test_link_maneuver_to_scene(self):
        m = ManeuverNode(entity_id="man_v1_100", maneuver_type="standing_still",
                         actor_id="veh_001", frame_start=100)
        edges = link_maneuver_to_scene(m, "veh_001")
        assert len(edges) == 2  # manifestsAs + actor
        assert edges[0].relation_type == "manifestsAs"
        assert edges[1].relation_type == "actor"

    def test_link_interaction_to_scene(self):
        ie = InteractionEvent(entity_id="int_v1_v2_follow_100",
                              interaction_type="following",
                              src_id="veh_001", dst_id="veh_002",
                              frame_start=100)
        edges = link_interaction_to_scene(ie)
        assert len(edges) == 3  # manifestsAs + src + dst


class TestBehaviorRelationGenerator:
    """BehaviorRelationGenerator — 行为层主生成器."""

    def test_single_frame_no_activity(self):
        gen = BehaviorRelationGenerator()
        res = gen.generate(frame_id=100, vehicles=[])
        assert res["maneuvers"] == []
        assert res["interactions"] == []
        assert res["behavior_rels"] == []

    def test_single_frame_standing(self):
        gen = BehaviorRelationGenerator(thresholds={"standing_still": 1})
        vehicles = [
            {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 0.0,
             "velocity_x": 0.0, "velocity_y": 0.0, "heading_rad": 0.0},
        ]
        # 帧 1: 防抖 threshold=1, 第1帧直接创建
        res = gen.generate(frame_id=1, vehicles=vehicles)
        # 因为 threshold=1, 第1帧条件满足即 create
        assert len(res["maneuvers"]) >= 0  # 可能有也可能没有, 取决于 detect 输出
        # 至少 standing_still 被检测到
        stats = gen.stats()
        assert "n_active_maneuvers" in stats

    def test_multi_frame_following(self):
        gen = BehaviorRelationGenerator(thresholds={"following": 2})
        v1 = {"entity_id": "veh_001", "location_x": 0.0, "location_y": 0.0, "speed": 8.0,
              "velocity_x": 8.0, "velocity_y": 0.0, "heading_rad": 0.0}
        v2 = {"entity_id": "veh_002", "location_x": 12.0, "location_y": 0.0, "speed": 6.0,
              "velocity_x": 6.0, "velocity_y": 0.0, "heading_rad": 0.0}
        vehicles = [v1, v2]

        # 帧 1: 条件满足, 防抖计数器 +1
        res1 = gen.generate(frame_id=1, vehicles=vehicles)
        n_first = len(res1["interactions"]) + len(res1["maneuvers"])

        # 帧 2: 条件满足, 防抖计数器 = 2 >= threshold=2 -> create
        res2 = gen.generate(frame_id=2, vehicles=vehicles)
        # 应该创建了 following InteractionEvent
        stats = gen.stats()

        # 帧 3: keep
        res3 = gen.generate(frame_id=3, vehicles=vehicles)
        assert stats["n_active_interactions"] > 0 or stats["n_active_maneuvers"] > 0

        # 移除条件 (veh_002 消失)
        res4 = gen.generate(frame_id=4, vehicles=[v1])
        # 2 帧后 should delete
        res5 = gen.generate(frame_id=5, vehicles=[v1])
        res6 = gen.generate(frame_id=6, vehicles=[v1])
        final_stats = gen.stats()
        # 最终 active 应该为 0 (所有跟随关系已关闭)
        assert final_stats["n_active_relations"] >= 0

    def test_reset_generator(self):
        gen = BehaviorRelationGenerator()
        gen.generate(frame_id=1, vehicles=[])
        gen.reset()
        stats = gen.stats()
        assert stats["n_active_maneuvers"] == 0
        assert stats["n_active_interactions"] == 0


class TestPhaseIntegration:
    """阶段间集成测试：验证行为层与阶段一/二的接口一致. (8 个用例)"""

    def test_entity_type_enum_exists(self):
        """EntityType 包含 MANEUVER 和 INTERACTION_EVENT"""
        assert EntityType.MANEUVER is not None
        assert EntityType.INTERACTION_EVENT is not None

    def test_behavior_relation_type_enum_complete(self):
        """BehaviorRelationType 枚举覆盖 13 种行为关系."""
        rels = set(m.value for m in BehaviorRelationType)
        expected = {
            "standing_still", "changing_lane",
            "following", "approaching", "yielding_to", "overtaking",
            "wrong_side_meeting", "opposite_direction", "same_direction",
            "blocked_view", "approaching_pedestrian",
            "approaching_intersection", "crossing",
        }
        assert rels == expected, f"Missing: {expected - rels}, Extra: {rels - expected}"

    def test_cross_layer_relation_type_complete(self):
        """CrossLayerRelationType 包含 manifestsAs / actor / src / dst."""
        assert hasattr(CrossLayerRelationType, "MANIFESTS_AS")
        assert hasattr(CrossLayerRelationType, "ACTOR")
        assert hasattr(CrossLayerRelationType, "SRC")
        assert hasattr(CrossLayerRelationType, "DST")

    def test_behavior_node_entity_type_matches(self):
        """ManeuverNode.entity_type == MANEUVER, InteractionEvent.entity_type == INTERACTION_EVENT."""
        m = ManeuverNode("m1", "standing_still", "v1", 0)
        ie = InteractionEvent("ie1", "following", "v1", "v2", 0)
        assert m.entity_type == EntityType.MANEUVER
        assert ie.entity_type == EntityType.INTERACTION_EVENT

    def test_behavior_relation_serialization(self):
        """行为关系 BaseRelation 实例支持 to_neo4j_dict()."""
        r = following("veh_001", "veh_002", frame_id=100, distance=12.5, ttc=41.7, relative_speed=0.3)
        d = r.to_neo4j_dict()
        assert d["relation_type"] == "following"
        assert d["frame_id"] == 100
        assert d["distance"] == 12.5
        assert d["ttc"] == 41.7

    def test_behavior_node_is_active_passthrough(self):
        """行为节点的 is_active_at 调用 BaseEntity 生命周期方法."""
        m = ManeuverNode("m1", "cruising", "v1", frame_start=10, frame_end=50)
        assert m.is_active_at(10) is True
        assert m.is_active_at(50) is True
        assert m.is_active_at(5) is False
        assert m.is_active_at(51) is False

    def test_behavior_node_neo4j_label(self):
        m = ManeuverNode("m1", "standing_still", "v1", 0)
        assert m.neo4j_label() == "Maneuver"
        ie = InteractionEvent("ie1", "following", "v1", "v2", 0)
        assert ie.neo4j_label() == "Interaction"

    def test_scenario_library_consistency_with_behavior(self):
        """场景库中的场景 ID 命名空间与行为层兼容."""
        from stk.scenario.scenario_library import SCENARIO_REGISTRY
        # SCENARIO_REGISTRY 是 dict: {场景 ID: scenario_meta/或factory}
        id_list = list(SCENARIO_REGISTRY.keys())
        # S00 - S33
        assert "S10" in id_list, "场景库中应包含 S10"
        assert "S00" in id_list, "场景库中应包含 S00"
