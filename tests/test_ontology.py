"""
阶段 1：本体层单元测试。
覆盖全部 7 个子模块：types、entity、relation、temporal_triple、namespace、lifecycle、axioms。
"""
import pytest
from stk.ontology.types import (
    EntityType, SceneRelationType, BehaviorRelationType,
    RuleRelationType, CrossLayerRelationType, NODE_LABELS,
    entity_type_from_label, relation_type_from_value,
)
from stk.ontology.entity import BaseEntity
from stk.ontology.relation import BaseRelation
from stk.ontology.temporal_triple import TemporalTriple
from stk.ontology.namespace import IDGenerator, NAMESPACE_PREFIXES
from stk.ontology.lifecycle import NodeLifecycle, NodeLifecycleStatus
from stk.ontology.axioms import (
    axiom_A1_unique_id, axiom_A2_fixed_type, axiom_A3_attribute_versioned,
    axiom_A4_relation_has_temporal, axiom_A5_three_layer_evidence,
    axiom_A6_event_traceable, axiom_A7_incremental_consistency,
)


class TestTypes:
    """枚举类型 (1.2)"""

    def test_entity_type_count(self):
        assert len(EntityType) == 14

    def test_entity_type_values(self):
        assert EntityType.VEHICLE.value == "Vehicle"
        assert EntityType.PEDESTRIAN.value == "Pedestrian"

    def test_scene_relation_count(self):
        assert len(SceneRelationType) == 15

    def test_behavior_relation_count(self):
        assert len(BehaviorRelationType) == 13

    def test_rule_relation_count(self):
        assert len(RuleRelationType) == 7

    def test_cross_layer_count(self):
        assert len(CrossLayerRelationType) == 7

    def test_node_labels_count(self):
        assert len(NODE_LABELS) == 14

    def test_entity_type_from_label(self):
        assert entity_type_from_label("Vehicle") == EntityType.VEHICLE
        assert entity_type_from_label("TrafficLight") == EntityType.TRAFFIC_LIGHT
        with pytest.raises(ValueError):
            entity_type_from_label("Unknown")

    def test_relation_type_from_value(self):
        assert relation_type_from_value("in_lane") == SceneRelationType.IN_LANE
        assert relation_type_from_value("violates") == RuleRelationType.VIOLATES
        assert relation_type_from_value("manifestsAs") == CrossLayerRelationType.MANIFESTS_AS
        with pytest.raises(ValueError):
            relation_type_from_value("not_exists")


class TestEntity:
    """Entity 基类 (1.3)"""

    def test_entity_create(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE)
        assert e.entity_id == "veh_001"
        assert e.confidence == 1.0

    def test_entity_with_enum(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE)
        assert e.neo4j_label() == "Vehicle"

    def test_entity_with_string(self):
        e = BaseEntity(entity_id="ped_001", entity_type="Pedestrian")
        assert e.neo4j_label() == "Pedestrian"

    def test_entity_is_active_default(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE)
        assert e.is_active(0) is True
        assert e.is_active(999) is True

    def test_entity_is_active_bounded(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE,
                       valid_from=100, valid_to=200)
        assert e.is_active(50) is False
        assert e.is_active(100) is True
        assert e.is_active(200) is True
        assert e.is_active(250) is False

    def test_entity_to_neo4j_dict(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE,
                       attrs={"speed": 10.5, "heading": 1.2})
        d = e.to_neo4j_dict()
        assert d["entity_id"] == "veh_001"
        assert d["speed"] == 10.5
        assert d["confidence"] == 1.0

    def test_entity_attrs_expanded(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE,
                       attrs={"speed": 10.5, "location_x": 100.0})
        assert e.attrs["speed"] == 10.5


class TestRelation:
    """Relation 基类 (1.4)"""

    def test_relation_create(self):
        r = BaseRelation(src_id="veh_001", dst_id="veh_002",
                         relation_type="following", frame_id=2048, valid_from=2048)
        assert r.src_id == "veh_001"
        assert r.dst_id == "veh_002"
        assert r.frame_id == 2048

    def test_predicate_str(self):
        r = BaseRelation(src_id="veh_001", dst_id="veh_002",
                         relation_type="following", frame_id=2048, valid_from=2048)
        assert r.predicate_str() == "Following(veh_001, veh_002, Frame_2048)"

    def test_predicate_str_snake(self):
        r = BaseRelation(src_id="veh_001", dst_id="veh_002",
                         relation_type="red_light_violation", frame_id=100, valid_from=100)
        assert r.predicate_str() == "RedLightViolation(veh_001, veh_002, Frame_100)"

    def test_to_neo4j_dict(self):
        r = BaseRelation(src_id="veh_001", dst_id="veh_002",
                         relation_type="following", frame_id=2048, valid_from=2048,
                         attrs={"distance": 12.5})
        d = r.to_neo4j_dict()
        assert d["relation_type"] == "following"
        assert d["distance"] == 12.5
        assert d["frame_id"] == 2048


class TestTemporalTriple:
    """时态三元组 (1.5)"""

    def test_triple_create(self):
        tt = TemporalTriple(subject="veh_001", predicate="in_lane",
                            object="road_5", frame_id=2048)
        assert tt.subject == "veh_001"
        assert tt.frame_id == 2048

    def test_to_triple_string(self):
        tt = TemporalTriple(subject="veh_001", predicate="following",
                            object="veh_002", frame_id=2048)
        s = tt.to_triple_string()
        assert "veh_001" in s
        assert "following" in s
        assert "2048" in s

    def test_to_dict(self):
        tt = TemporalTriple(subject="veh_001", predicate="in_lane",
                            object="road_5", frame_id=2048)
        d = tt.to_dict()
        assert d["subject"] == "veh_001"
        assert d["predicate"] == "in_lane"


class TestNamespace:
    """命名空间与 ID 生成器 (1.6)"""

    def test_prefix_count(self):
        assert len(NAMESPACE_PREFIXES) == 7

    def test_generate_vehicle(self):
        g = IDGenerator()
        eid = g.generate("VehicleEntity", actor_id=123)
        assert eid == "veh_123"

    def test_generate_safety_violation(self):
        g = IDGenerator()
        eid = g.generate("SafetyViolation", rule_code="R13", frame=2048)
        assert eid == "sv_R13_2048"

    def test_generate_road(self):
        g = IDGenerator()
        eid = g.generate("RoadElementEntity", road_id=5, lane_id=2)
        assert eid == "road_5_lane_2"

    def test_parse_type(self):
        g = IDGenerator()
        assert g.parse_type("veh_123") == "VehicleEntity"
        assert g.parse_type("ped_45") == "PedestrianEntity"
        assert g.parse_type("tl_7") == "TrafficLightEntity"
        assert g.parse_type("unknown") is None

    def test_unique_id_violation(self):
        g = IDGenerator()
        g.generate("VehicleEntity", actor_id=123)
        with pytest.raises(ValueError, match="重复"):
            g.generate("VehicleEntity", actor_id=123)

    def test_reset(self):
        g = IDGenerator()
        g.generate("VehicleEntity", actor_id=123)
        g.reset()
        # 重置后可再次生成
        eid = g.generate("VehicleEntity", actor_id=123)
        assert eid == "veh_123"


class TestLifecycle:
    """节点生命周期状态机 (1.7)"""

    def test_initial_state(self):
        lc = NodeLifecycle("veh_123")
        assert lc.status == NodeLifecycleStatus.CREATED
        assert lc.frame_start is None

    def test_activate(self):
        lc = NodeLifecycle("veh_123")
        lc.activate(frame_id=100)
        assert lc.status == NodeLifecycleStatus.ACTIVE
        assert lc.frame_start == 100

    def test_is_active_at(self):
        lc = NodeLifecycle("veh_123")
        lc.activate(frame_id=100)
        assert lc.is_active_at(150) is True
        assert lc.is_active_at(50) is False

    def test_deactivate(self):
        lc = NodeLifecycle("veh_123")
        lc.activate(frame_id=100)
        lc.deactivate(frame_id=200)
        assert lc.status == NodeLifecycleStatus.STALE
        assert lc.frame_end == 200
        assert lc.is_active_at(250) is False

    def test_version_chain(self):
        lc = NodeLifecycle("veh_123")
        lc.activate(frame_id=100)
        lc.add_version("speed", 10.5, valid_from=100)
        lc.add_version("speed", 15.0, valid_from=110)
        assert lc.get_version_at("speed", 105) == 10.5
        assert lc.get_version_at("speed", 115) == 15.0

    def test_version_not_found(self):
        lc = NodeLifecycle("veh_123")
        assert lc.get_version_at("speed", 0) is None

    def test_transitions(self):
        lc = NodeLifecycle("veh_123")
        lc.activate(frame_id=100)
        lc.deactivate(frame_id=200)
        assert len(lc.transitions) == 2
        assert lc.transitions[0].from_status == NodeLifecycleStatus.CREATED
        assert lc.transitions[0].to_status == NodeLifecycleStatus.ACTIVE

    def test_to_dict(self):
        lc = NodeLifecycle("veh_123", "Vehicle")
        lc.activate(frame_id=100)
        lc.add_version("speed", 10, 100)
        d = lc.to_dict()
        assert d["entity_id"] == "veh_123"
        assert d["n_transitions"] == 1
        assert d["n_versions"] == 1


class TestAxioms:
    """公理函数 (1.8)"""

    def test_A1_valid_id(self):
        assert axiom_A1_unique_id("veh_001") is True
        assert axiom_A1_unique_id("") is False
        assert axiom_A1_unique_id(" ") is False

    def test_A2_fixed_type(self):
        e = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE)
        assert axiom_A2_fixed_type(e) is True

    def test_A3_attribute_versioned(self):
        lc = NodeLifecycle("veh_001")
        lc.add_version("speed", 10.5, 100)
        assert axiom_A3_attribute_versioned(lc, "speed", 105) is True
        assert axiom_A3_attribute_versioned(lc, "speed", 50) is False

    def test_A4_relation_temporal(self):
        r = BaseRelation(src_id="a", dst_id="b", relation_type="f",
                         frame_id=1, valid_from=1)
        assert axiom_A4_relation_has_temporal(r) is True

    def test_A5_rule_node_evidence(self):
        rule = BaseEntity(entity_id="rule_R1", entity_type=EntityType.RULE_DEFINITION)
        assert axiom_A5_three_layer_evidence(rule, []) is True  # 规则定义不需证据
        sv = BaseEntity(entity_id="sv_001", entity_type=EntityType.SAFETY_VIOLATION)
        assert axiom_A5_three_layer_evidence(sv, ["e1"]) is True
        assert axiom_A5_three_layer_evidence(sv, []) is False

    def test_A6_event_traceable(self):
        sv = BaseEntity(entity_id="sv_001", entity_type=EntityType.SAFETY_VIOLATION)
        assert axiom_A6_event_traceable(sv, 1) is True
        assert axiom_A6_event_traceable(sv, 0) is False
        vehicle = BaseEntity(entity_id="veh_001", entity_type=EntityType.VEHICLE)
        assert axiom_A6_event_traceable(vehicle, 0) is True  # 非违规节点不强制

    def test_A7_incremental_consistency(self):
        assert axiom_A7_incremental_consistency(["a", "b"], ["a", "b", "c"]) is True
        assert axiom_A7_incremental_consistency(["a", "b"], ["a"]) is False
