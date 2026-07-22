# -*- coding: utf-8 -*-
"""T4: SafetyViolation 双表达 + supportedByEvidence 证据链 + RuleEnforcer 集成 (v3 sec 4.3 / 4.9 / 4.11)."""
from __future__ import annotations
import math
import pytest

from stk.ontology.relation import BaseRelation
from stk.rules.generator import RuleEnforcer
from stk.rules.nodes import (
    ResponsibilityAssignment,
    RuleDefinition,
    RuleParameter,
    SafetyViolation,
)
from stk.rules.relations import RuleRelationType


# ---------- 工厂函数 ----------

def _vehicle(vid, speed=10.0, vx=10.0, vy=0.0, loc_x=0.0, loc_y=0.0, brake=0.8):
    return {
        "entity_id": vid, "speed": speed, "speed_kmh": speed * 3.6,
        "velocity_x": vx, "velocity_y": vy,
        "location_x": loc_x, "location_y": loc_y, "brake": brake,
    }


def _pedestrian(pid, loc_x=10.0, loc_y=0.0, on_cw=True):
    return {"entity_id": pid, "location_x": loc_x, "location_y": loc_y,
            "is_on_crosswalk": on_cw, "speed": 0.0}


# ---------- SafetyViolation 节点字段完整性 ----------

class TestSafetyViolationNode:
    def test_node_has_required_attrs(self):
        sv = SafetyViolation(
            entity_id="SV_R13a_F1_V1_V2",
            rule_code="R13a", rule_name="SafeDistanceViolation",
            rule_layer="RSS", frame_id=1, severity=0.7,
            src_id="V1", dst_id="V2",
            predicate_str="DangerousState(V1, V2, Frame_1)",
        )
        a = sv.attrs
        assert a["sv_id"] == "SV_R13a_F1_V1_V2"
        assert a["rule_code"] == "R13a"
        assert a["rule_layer"] == "RSS"
        assert a["severity"] == 0.7
        assert a["src_id"] == "V1"
        assert a["dst_id"] == "V2"

    def test_default_severity_is_positive(self):
        sv = SafetyViolation(
            entity_id="SV1", rule_code="R1", rule_name="X",
            rule_layer="TrafficLaw", frame_id=1,
            src_id="A", dst_id="B", predicate_str="P",
        )
        assert sv.attrs["severity"] > 0

    def test_evidence_path_list(self):
        sv = SafetyViolation(
            entity_id="SV1", rule_code="R1", rule_name="X",
            rule_layer="TrafficLaw", frame_id=1,
            src_id="A", dst_id="B", predicate_str="P",
            evidence_path=["scene_rel_A_B_1", "maneuver_A_1"],
        )
        assert isinstance(sv.attrs["evidence_path"], list)
        assert len(sv.attrs["evidence_path"]) == 2


# ---------- 双表达 (节点 + violates 边) 一致性 ----------

class TestDualRepresentation:
    """v3 §4.3: 节点 + violates 边双表达."""

    def test_rss_violation_produces_node_and_matching_violates_edge(self):
        enforcer = RuleEnforcer()
        # 两车间距 1m, 纵向 (dy!=0 避免误触发横向)
        v_a = _vehicle("V1", speed=20.0, vx=20.0, vy=0.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  vx=0.0,  vy=0.0, loc_x=1.0, loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])

        violations = out["violations"]
        assert len(violations) >= 1

        # 找出 R13a (纵向) 的违规
        r13a = [sv for sv in violations if sv.attrs.get("rule_code") == "R13a"]
        assert r13a, "should produce R13a longitudinal violation"
        sv = r13a[0]
        assert sv.attrs["rule_layer"] == "RSS"
        assert sv.attrs["predicate_str"].startswith("SafeDistanceViolation")
        assert sv.attrs["src_id"] == "V1"
        assert sv.attrs["dst_id"] == "V2"
        assert sv.attrs["frame_id"] == 1

        # 对应的 violates 边
        v_rels = [r for r in out["violation_rels"]
                  if r.attrs.get("rule_code") == "R13a"
                  and r.attrs.get("sv_id") == sv.attrs["sv_id"]]
        assert len(v_rels) >= 1
        r = v_rels[0]
        # 边的端点 (src_id/dst_id) 在 BaseRelation 模型字段
        assert {r.src_id, r.dst_id} == {sv.attrs["src_id"], sv.attrs["dst_id"]}
        # 边携带的属性
        assert r.attrs.get("predicate") == "SafeDistanceViolation"
        assert r.attrs.get("sv_id") == sv.attrs["sv_id"]
        assert r.attrs.get("severity") == sv.attrs["severity"]

    def test_violates_edge_carries_rule_code_and_predicate(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])
        v_rels = out["violation_rels"]
        assert len(v_rels) >= 1
        for r in v_rels:
            assert "rule_code" in r.attrs
            assert "predicate" in r.attrs
            assert "sv_id" in r.attrs


# ---------- definedBy edge ----------

class TestDefinedByEdge:
    def test_rss_violation_has_definedby_edge(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])
        db_rels = out["defined_by_rels"]
        assert len(db_rels) >= 1
        for r in db_rels:
            # relation_type 字段
            assert r.relation_type == RuleRelationType.DEFINED_BY.value


# ---------- supportedByEvidence ----------

class TestSupportedByEvidence:
    def test_evidence_rels_key_present(self):
        enforcer = RuleEnforcer()
        out = enforcer.enforce(frame_id=1, vehicles=[])
        assert "evidence_rels" in out
        assert isinstance(out["evidence_rels"], list)

    def test_evidence_rels_use_supported_type(self):
        enforcer = RuleEnforcer()
        # 即使无 scene_rels 被传入, 也应是一个 list (允许为空)
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])
        for r in out["evidence_rels"]:
            assert r.relation_type == RuleRelationType.SUPPORTED_BY_EVIDENCE.value


# ---------- 责任归因 ----------

class TestResponsibility:
    def test_output_has_responsibilities_keys(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        for f in range(3):
            out = enforcer.enforce(frame_id=f, vehicles=[v_a, v_b])
        assert "responsibilities" in out
        assert "resp_rels" in out
        assert isinstance(out["responsibilities"], list)
        assert isinstance(out["resp_rels"], list)

    def test_resp_rels_use_responsible_for_type(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        for f in range(3):
            out = enforcer.enforce(frame_id=f, vehicles=[v_a, v_b])
        for r in out["resp_rels"]:
            assert r.relation_type == RuleRelationType.RESPONSIBLE_FOR.value


# ---------- RuleEnforcer 集成: R1 行人优先 ----------

class TestRuleEnforcerPedestrianR1:
    def test_r1_violation_triggered(self):
        enforcer = RuleEnforcer()
        v = _vehicle("V1", speed=8.0, vx=8.0, loc_x=0.0, loc_y=0.0, brake=0.5)
        p = _pedestrian("P1", loc_x=5.0, loc_y=0.0, on_cw=True)
        out = enforcer.enforce(frame_id=1, vehicles=[v], pedestrians=[p])
        r1 = [sv for sv in out["violations"] if sv.attrs.get("rule_code") == "R1"]
        assert len(r1) >= 1
        sv = r1[0]
        assert sv.attrs["rule_layer"] == "TrafficLaw"
        assert sv.attrs["rule_name"] == "YieldingToPedestrianViolation"
        assert sv.attrs["src_id"] == "V1"
        assert sv.attrs["dst_id"] == "P1"

    def test_r1_not_triggered_when_far(self):
        enforcer = RuleEnforcer()
        v = _vehicle("V1", speed=8.0, loc_x=0.0, loc_y=0.0, brake=0.5)
        p = _pedestrian("P1", loc_x=80.0, loc_y=0.0, on_cw=True)
        out = enforcer.enforce(frame_id=1, vehicles=[v], pedestrians=[p])
        r1 = [sv for sv in out["violations"] if sv.attrs.get("rule_code") == "R1"]
        assert len(r1) == 0

    def test_r1_not_triggered_when_not_on_crosswalk(self):
        enforcer = RuleEnforcer()
        v = _vehicle("V1", speed=8.0, loc_x=0.0, loc_y=0.0, brake=0.5)
        p = _pedestrian("P1", loc_x=5.0, loc_y=0.0, on_cw=False)
        out = enforcer.enforce(frame_id=1, vehicles=[v], pedestrians=[p])
        r1 = [sv for sv in out["violations"] if sv.attrs.get("rule_code") == "R1"]
        assert len(r1) == 0


# ---------- RuleEnforcer 集成: RSS 纵向 ----------

class TestRuleEnforcerRSSLongitudinal:
    def test_too_close_triggers_r13a_with_frame_id(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0,  loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0,  loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=42, vehicles=[v_a, v_b])
        rss = [sv for sv in out["violations"] if sv.attrs.get("rule_layer") == "RSS"]
        assert len(rss) >= 1
        sv = rss[0]
        assert sv.attrs["frame_id"] == 42
        assert "V1" in sv.attrs["predicate_str"]
        assert "V2" in sv.attrs["predicate_str"]
        assert "Frame_42" in sv.attrs["predicate_str"]

    def test_safe_distance_no_longitudinal_violation(self):
        enforcer = RuleEnforcer()
        # 同 y 轴 -> d_lat=0 会触发 R14a; 让两车在 y 也拉开, 且速度差不大
        v_a = _vehicle("V1", speed=10.0, vx=10.0, vy=0.3, loc_x=0.0,   loc_y=0.0,  brake=0.0)
        v_b = _vehicle("V2", speed=10.0, vx=10.0, vy=0.3, loc_x=200.0, loc_y=200.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])
        # 不应触发 R13a 纵向 (距离 200m)
        r13a = [sv for sv in out["violations"] if sv.attrs.get("rule_code") == "R13a"]
        assert len(r13a) == 0

    def test_dmin_recorded_in_attrs(self):
        enforcer = RuleEnforcer()
        v_a = _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0)
        v_b = _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)
        out = enforcer.enforce(frame_id=1, vehicles=[v_a, v_b])
        r13a = [sv for sv in out["violations"] if sv.attrs.get("rule_code") == "R13a"]
        assert r13a
        sv = r13a[0]
        # d_min_long 应直接在 attrs 里 (实现把 extra_attrs 合并在一起)
        assert "d_min_long" in sv.attrs or "d_min_long" in sv.attrs.get("extra_attrs", {})
        # or 任何 key 含 d_min
        keys = list(sv.attrs.keys())
        assert any("d_min" in k for k in keys)


# ---------- 输出结构契约 ----------

class TestRuleEnforcerOutputContract:
    def test_enforce_returns_all_required_keys(self):
        enforcer = RuleEnforcer()
        out = enforcer.enforce(frame_id=1)
        for k in ("violations", "violation_rels", "defined_by_rels",
                  "evidence_rels", "responsibilities", "resp_rels"):
            assert k in out
            assert isinstance(out[k], list)

    def test_empty_input_yields_empty_output(self):
        enforcer = RuleEnforcer()
        out = enforcer.enforce(frame_id=1)
        for key in ("violations", "violation_rels", "defined_by_rels",
                    "evidence_rels", "responsibilities", "resp_rels"):
            assert out[key] == []


# ---------- 跨帧状态 ----------

class TestCrossFrameState:
    def test_brake_history_accumulates(self):
        enforcer = RuleEnforcer()
        v = _vehicle("V1", speed=5.0, brake=0.1)
        for f in range(5):
            enforcer.enforce(frame_id=f, vehicles=[v])
        assert enforcer.stats()["n_brake_history"] == 1
        assert len(enforcer._brake_history["V1"]) == 5

    def test_reset_clears_state(self):
        enforcer = RuleEnforcer()
        v = _vehicle("V1", speed=5.0, brake=0.1)
        enforcer.enforce(frame_id=0, vehicles=[v])
        assert enforcer.stats()["n_brake_history"] == 1
        enforcer.reset()
        assert enforcer.stats()["n_brake_history"] == 0


# ---------- RuleDefinition 与 RuleParameter ----------

class TestRuleDefinitionAndParameter:
    def test_rule_definition_attrs(self):
        rd = RuleDefinition(
            entity_id="R1",
            rule_id="R1",
            rule_name="YieldingToPedestrian",
            rule_layer="TrafficLaw",
            predicate_name="YieldingToPedestrianViolation",
            formula_str="YieldingToPedestrianViolation(A,P,t) <- ...",
        )
        assert rd.entity_id == "R1"
        a = rd.attrs
        assert a["rule_name"] == "YieldingToPedestrian"
        assert a["rule_layer"] == "TrafficLaw"
        assert a["predicate_name"] == "YieldingToPedestrianViolation"
        assert a["formula_str"].startswith("YieldingToPedestrianViolation")

    def test_rule_parameter_attrs(self):
        rp = RuleParameter(
            entity_id="P_rho",
            param_id="P_rho",
            name="rho",
            value=0.1,
            unit="s",
        )
        a = rp.attrs
        assert a["name"] == "rho"
        assert a["value"] == 0.1
        assert a["unit"] == "s"

    def test_sv_id_unique_per_frame(self):
        enforcer = RuleEnforcer()
        out1 = enforcer.enforce(frame_id=1,
            vehicles=[_vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0),
                      _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)])
        out2 = enforcer.enforce(frame_id=2,
            vehicles=[_vehicle("V1", speed=20.0, loc_x=0.0, loc_y=10.0, brake=0.0),
                      _vehicle("V2", speed=0.0,  loc_x=1.0, loc_y=10.0, brake=0.0)])
        ids1 = {sv.attrs["sv_id"] for sv in out1["violations"]}
        ids2 = {sv.attrs["sv_id"] for sv in out2["violations"]}
        # 不同 frame_id -> 不同 sv_id
        assert ids1.isdisjoint(ids2)


# ---------- EgoCentric RSS 对子 集成测试 (阶段1) ----------

def _ego_vehicle(vid, speed=10.0, vx=10.0, vy=0.0, loc_x=0.0, loc_y=0.0,
                 brake=0.8, heading=0.0, is_ego=True):
    """带有 is_ego / heading_rad 的车辆工厂 (用于 EgoCentric 测试)."""
    return {
        "entity_id": vid, "speed": speed, "speed_kmh": speed * 3.6,
        "velocity_x": vx, "velocity_y": vy,
        "location_x": loc_x, "location_y": loc_y, "brake": brake,
        "heading_rad": heading, "is_ego": is_ego,
    }


class TestEgoCentricRSSPairs:
    """默认 EgoCentric (legacy_full_pairing=False) 时 RSS 只评 ego×ROI 内他车."""

    def test_egocentric_skips_far_ahead_vehicle(self):
        """前方 40m 车在 ROI 内 → 产出 R13a + R14a 两条 SafetyViolation.
           前方 100m 车在 ROI 外 → 不产出其违规.

           配置: ego 在原点朝 +x.
        """
        enforcer = RuleEnforcer()
        vehicles = [
            _ego_vehicle("ego",   speed=20.0, loc_x=0.0,  loc_y=0.0, brake=0.0),
            _vehicle("V_close", speed=5.0, loc_x=40.0, loc_y=0.0, brake=0.0),  # ROI 内
            _vehicle("V_far",   speed=5.0, loc_x=100.0, loc_y=0.0, brake=0.0),  # ROI 外
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        violations = out["violations"]
        # ego vs V_close 产出纵向(R13a) + 横向(R14a) 两条
        dst_ids = {sv.attrs["dst_id"] for sv in violations}
        assert "V_far" not in dst_ids, "ROI 外的车不应出现在违规 dst_id"
        assert "V_close" in dst_ids, "ROI 内的车应出现在违规 dst_id"
        # 全部 violations 都是 ego×V_close 对子
        for sv in violations:
            assert sv.attrs["src_id"] == "ego"
            assert sv.attrs["dst_id"] == "V_close"
        # 应包含 R13a + R14a 两条
        rule_codes = {sv.attrs["rule_code"] for sv in violations}
        assert rule_codes == {"R13a", "R14a"}

    def test_egocentric_includes_rear_vehicle(self):
        """后方 20m 车在 ROI 内 (R_rear=30) → 产出违规."""
        enforcer = RuleEnforcer()
        vehicles = [
            _ego_vehicle("ego", speed=20.0, loc_x=0.0, loc_y=0.0, brake=0.0),
            _vehicle("V_rear", speed=10.0, loc_x=0.0, loc_y=-20.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        assert len(out["violations"]) >= 1
        assert out["violations"][0].attrs["dst_id"] == "V_rear"

    def test_egocentric_excludes_rear_out_of_roi(self):
        """后方 80m 车超出后向 ROI (R_rear=30) → 不产出违规.
           注: ego 朝 +x, (0,-80) 在车体坐标系下 lateral=-80, lateral/50=1.6, ROI 外.
        """
        enforcer = RuleEnforcer()
        vehicles = [
            _ego_vehicle("ego", speed=20.0, loc_x=0.0, loc_y=0.0, brake=0.0),
            _vehicle("V_rear_far", speed=10.0, loc_x=0.0, loc_y=-80.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        dst_ids = {sv.attrs["dst_id"] for sv in out["violations"]}
        assert "V_rear_far" not in dst_ids

    def test_egocentric_excludes_side_out_of_roi(self):
        """侧向 80m 车超出侧向 ROI (R_side=50) → 不产出违规."""
        enforcer = RuleEnforcer()
        vehicles = [
            _ego_vehicle("ego", speed=20.0, loc_x=0.0, loc_y=0.0, brake=0.0),
            _vehicle("V_side", speed=5.0, loc_x=0.0, loc_y=80.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        dst_ids = {sv.attrs["dst_id"] for sv in out["violations"]}
        assert "V_side" not in dst_ids

    def test_legacy_full_pairing_still_checks_all(self):
        """legacy_full_pairing=True 时仍产出所有全对子 RSS 违规."""
        from stk.config import EgoCentricConfig
        cfg = EgoCentricConfig(legacy_full_pairing=True)
        enforcer = RuleEnforcer(ego_config=cfg)
        vehicles = [
            _ego_vehicle("ego",   speed=20.0, loc_x=0.0,  loc_y=0.0, brake=0.0),
            _vehicle("V_close", speed=5.0, loc_x=40.0, loc_y=0.0, brake=0.0),
            _vehicle("V_far",   speed=5.0, loc_x=100.0, loc_y=0.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        dst_ids = {sv.attrs["dst_id"] for sv in out["violations"]}
        # legacy 模式应包含 V_close 和 V_far 的违规
        assert "V_close" in dst_ids
        assert "V_far" in dst_ids

    def test_egocentric_no_ego_fallback_first_vehicle(self):
        """所有 vehicle 都没有 is_ego → fallback vehicles[0] 为 ego.

           V1 被 fallback 认定为 ego, V2 在 ROI 内 → 产出 R13a + R14a 两条.
        """
        enforcer = RuleEnforcer()
        vehicles = [
            _vehicle("V1", speed=20.0, loc_x=0.0, loc_y=0.0, brake=0.0),
            _vehicle("V2", speed=5.0, loc_x=40.0, loc_y=0.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        # V1 被 fallback 认定为 ego, V2 在 ROI 内 → 产出违规
        assert len(out["violations"]) >= 1
        for sv in out["violations"]:
            assert sv.attrs["src_id"] == "V1"
            assert sv.attrs["dst_id"] == "V2"

    def test_egocentric_violation_has_responsibility_or_no_proper_response(self):
        """ego×ROI 的违规应产出责任归因 OR 不触发 NoProperResponse 的判定.

           brake=0.0 触发 NoProperResponse 时, 应有 1 个 ResponsibilityAssignment.
           (本测试仅验证责任归因机制存在且关联 ego, 不强求每帧都触发.)
        """
        enforcer = RuleEnforcer()
        vehicles = [
            _ego_vehicle("ego", speed=20.0, loc_x=0.0, loc_y=0.0, brake=0.0),
            _vehicle("V_close", speed=5.0, loc_x=40.0, loc_y=0.0, brake=0.0),
        ]
        out = enforcer.enforce(frame_id=1, vehicles=vehicles)
        # 至少有 ego×V_close 的违规
        assert len(out["violations"]) >= 1
        # 责任归因: 若 NoProperResponse 触发, responsible_actor_id 应为 ego
        for ra in out["responsibilities"]:
            assert ra.attrs.get("responsible_actor_id") == "ego"