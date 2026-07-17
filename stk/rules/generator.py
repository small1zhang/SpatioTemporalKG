"""
RuleEnforcer — 规则层主生成器 (v3 sec 4 / sec 7.3.3)

本模块是规则层的核心驱动:
  1. 从场景层+行为层的数据 (实体 dict + 关系列表) 通过 RSS 和交通法规检测器
     生成所有"规则违规候选"
  2. 为每个违规创建:
       - SafetyViolation 节点
       - violates 违规边 (节点+边双轨, v3 sec 4.3)
       - definedBy 边 (指向 RuleDefinition)
       - supportedByEvidence 边 (指向证据)
       - ResponsibilityAssignment 节点 + responsibleFor 边 (RSS 责任归因)
  3. 支持复杂场景的因果链 (v3 sec 4.18.2)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from stk.ontology.relation import BaseRelation
from stk.ontology.types import EntityType
from stk.rules.nodes import (
    SafetyViolation, ResponsibilityAssignment,
    make_sv_id, make_resp_id,
)
from stk.rules.relations import (
    defined_by, supported_by_evidence, violates,
    responsible_for, caused_by,
)
from stk.rules.rss.model import run_rss_check, DEFAULT_RSS_PARAMS
from stk.rules.traffic.rules import (
    check_R1_pedestrian_priority,
    check_R2_red_light,
    check_R7_junction_no_yield,
    check_R8_vulnerable_protection,
    check_R9_school_zone_speed,
    check_R10_highway_speed,
    check_R11_weather_speed,
    check_R13_illegal_stop,
    check_R16_amber_jumping,
    check_R17_wrong_lane,
    check_R18_wrong_direction_lane,
)


class RuleEnforcer:
    """规则层主生成器.

    跨帧状态:
      _sv_counter: dict, 记录已发出的 SafetyViolation 计数
      _brake_history: {veh_id: [brake_values]}, 制动历史用于 NoProperResponse 判定
      _stop_duration: {veh_id: frames}, 静止持续帧数
    """

    def __init__(self, rss_params=None):
        self._rss_params = rss_params or dict(DEFAULT_RSS_PARAMS)
        self._sv_counter: int = 0
        self._brake_history: Dict[str, List[float]] = {}
        self._stop_duration: Dict[str, int] = {}

    def enforce(self, frame_id,
                vehicles=None, pedestrians=None, traffic_lights=None,
                junctions=None, crosswalks=None,
                scene_rels=None, behavior_rels=None) -> Dict[str, List]:
        """对单帧运行规则层生成.

        Args: 场景层/行为层输入

        Returns:
            {
              "violations": [SafetyViolation, ...],
              "violation_rels": [BaseRelation(violates), ...],
              "defined_by_rels": [definedBy edges, ...],
              "evidence_rels": [supportedByEvidence edges, ...],
              "responsibilities": [ResponsibilityAssignment, ...],
              "resp_rels": [responsibleFor edges, ...],
            }
        """
        vehicles = vehicles or []
        pedestrians = pedestrians or []
        traffic_lights = traffic_lights or []
        scene_rels = scene_rels or []
        behavior_rels = behavior_rels or []

        violations: List[SafetyViolation] = []
        violation_rels: List[BaseRelation] = []
        defined_by_rels: List[BaseRelation] = []
        evidence_rels: List[BaseRelation] = []
        responsibilities: List[ResponsibilityAssignment] = []
        resp_rels: List[BaseRelation] = []

        # 更新跨帧状态
        self._update_state(vehicles, frame_id)

        # --- RSS 检查 ---
        for i, v_a in enumerate(vehicles):
            eid_a = v_a.get("entity_id", "")
            speed_a = v_a.get("speed", 0.0)
            for j, v_b in enumerate(vehicles):
                if i >= j:
                    continue
                eid_b = v_b.get("entity_id", "")
                speed_b = v_b.get("speed", 0.0)
                # 简化: 获取纵向/横向距离
                dx = v_a.get("location_x", 0) - v_b.get("location_x", 0)
                dy = v_a.get("location_y", 0) - v_b.get("location_y", 0)
                d_long = abs(dx)
                d_lat = abs(dy)
                v_lat_a = abs(v_a.get("velocity_y", 0.0))
                v_lat_b = abs(v_b.get("velocity_y", 0.0))
                brake_vals = self._brake_history.get(eid_a, [0])

                rss = run_rss_check(
                    d_long=d_long, d_lat=d_lat,
                    v_A=speed_a, v_B=speed_b,
                    v_lat_A=v_lat_a, v_lat_B=v_lat_b,
                    brake_values=brake_vals,
                    params=self._rss_params,
                )

                if rss["is_dangerous"]:
                    for rss_name in ["is_long_violation", "is_lat_violation"]:
                        if not rss[rss_name]:
                            continue
                        rule_code = "R13a" if rss_name == "is_long_violation" else "R14a"
                        pred_name = "SafeDistanceViolation" if rule_code == "R13a" else "LateralDangerousState"
                        rule_id = rule_code

                        sv_id = make_sv_id(rule_code, frame_id, eid_a, eid_b)
                        sev = min(rss["is_dangerous"] * 0.8, 0.95)

                        sv = SafetyViolation(
                            entity_id=sv_id,
                            rule_code=rule_code, rule_name=pred_name,
                            rule_layer="RSS", frame_id=frame_id, severity=sev,
                            src_id=eid_a, dst_id=eid_b,
                            predicate_str=f"{pred_name}({eid_a}, {eid_b}, Frame_{frame_id})",
                            evidence_path=[f"scene_rel_{eid_a}_{eid_b}_{frame_id}"],
                            extra_attrs={
                                "d_min_long": rss["d_min_long"],
                                "d_min_lat": rss["d_min_lat"],
                            },
                        )
                        violations.append(sv)
                        violation_rels.append(violates(
                            src_entity_id=eid_a, dst_entity_id=eid_b,
                            frame_id=frame_id, valid_from=frame_id,
                            rule_code=rule_code, predicate=pred_name,
                            sv_id=sv_id, severity=sev,
                        ))
                        defined_by_rels.append(defined_by(
                            sv_id=sv_id, rule_id=rule_id,
                            frame_id=frame_id, valid_from=frame_id,
                        ))
                        # RSS 责任归因
                        if rss["is_no_proper_response"]:
                            resp_id = make_resp_id(sv_id, eid_a)
                            ra = ResponsibilityAssignment(
                                entity_id=resp_id,
                                sv_id=sv_id,
                                responsible_actor_id=eid_a,
                                reason="no_proper_response",
                            )
                            responsibilities.append(ra)
                            resp_rels.append(responsible_for(
                                resp_id=resp_id, sv_id=sv_id,
                                frame_id=frame_id, valid_from=frame_id,
                                reason="no_proper_response",
                            ))

        # --- 交通法规 R1-R18 检查 ---
        for v in vehicles:
            eid = v.get("entity_id", "")
            speed = v.get("speed", 0.0)
            speed_kmh = v.get("speed_kmh", speed * 3.6)

            # R9 学区限速
            is_v, sev, extra = check_R9_school_zone_speed(v, in_school_zone=False)
            if is_v and eid:
                _add_violation("R9", "SchoolZoneSpeedViolation", "TrafficLaw",
                               eid, "", eid + "_road", frame_id, sev, violations,
                               violation_rels, defined_by_rels, evidence_rels)

            # R10 高速公路限速
            is_v, sev, _ = check_R10_highway_speed(v)
            if is_v and eid:
                _add_violation("R10", "HighwaySpeedViolation", "TrafficLaw",
                               eid, "", eid + "_road", frame_id, sev, violations,
                               violation_rels, defined_by_rels, evidence_rels)

            # R11 恶劣天气限速
            is_v, sev, _ = check_R11_weather_speed(v, precipitation=0)
            if is_v and eid:
                _add_violation("R11", "WeatherSpeedViolation", "TrafficLaw",
                               eid, "", eid + "_env", frame_id, sev, violations,
                               violation_rels, defined_by_rels, evidence_rels)

            # R13 禁停
            stop_dur = self._stop_duration.get(eid, 0)
            is_v, sev, _ = check_R13_illegal_stop(v, in_no_stop_zone=False, duration_frames=stop_dur)
            if is_v and eid:
                _add_violation("R13", "IllegalStopViolation", "TrafficLaw",
                               eid, "", eid + "_road", frame_id, sev, violations,
                               violation_rels, defined_by_rels, evidence_rels)

            # R17 不按规定车道
            is_v, sev, _ = check_R17_wrong_lane(v, lane_type="Driving")
            if is_v and eid:
                _add_violation("R17", "WrongLaneViolation", "TrafficLaw",
                               eid, "", eid + "_road", frame_id, sev, violations,
                               violation_rels, defined_by_rels, evidence_rels)

        # 行人-车辆交互
        for v in vehicles:
            eid_v = v.get("entity_id", "")
            for p in pedestrians:
                eid_p = p.get("entity_id", "")
                dx = v.get("location_x", 0) - p.get("location_x", 0)
                dy = v.get("location_y", 0) - p.get("location_y", 0)
                dist = (dx ** 2 + dy ** 2) ** 0.5
                is_on_cw = p.get("is_on_crosswalk", False)

                # R1 行人优先
                is_v, sev, _ = check_R1_pedestrian_priority(v, p, dist, is_on_crosswalk=is_on_cw)
                if is_v and eid_v and eid_p:
                    _add_violation("R1", "YieldingToPedestrianViolation", "TrafficLaw",
                                   eid_v, eid_p, eid_p, frame_id, sev, violations,
                                   violation_rels, defined_by_rels, evidence_rels)

                # R8 弱势参与者保护
                is_v, sev, _ = check_R8_vulnerable_protection(v, p, dist)
                if is_v and eid_v and eid_p:
                    _add_violation("R8", "VulnerableUserProtectionViolation", "TrafficLaw",
                                   eid_v, eid_p, eid_p, frame_id, sev, violations,
                                   violation_rels, defined_by_rels, evidence_rels)

        # 交通灯交互
        for v in vehicles:
            eid_v = v.get("entity_id", "")
            for tl in traffic_lights:
                eid_tl = tl.get("entity_id", "")
                tl_state = tl.get("state", "Green")
                is_v, sev, _ = check_R2_red_light(v, tl, in_junction=False, signal_state=tl_state)
                if is_v and eid_v:
                    _add_violation("R2", "RedLightViolation", "TrafficLaw",
                                   eid_v, eid_tl, eid_tl, frame_id, sev, violations,
                                   violation_rels, defined_by_rels, evidence_rels)

                # R16 黄灯抢行
                is_v, sev, _ = check_R16_amber_jumping(v, tl_state_before=tl_state,
                                                         tl_state_now=tl_state)
                if is_v and eid_v:
                    _add_violation("R16", "AmberLightJumpingViolation", "TrafficLaw",
                                   eid_v, eid_tl, eid_tl, frame_id, sev, violations,
                                   violation_rels, defined_by_rels, evidence_rels)

        return {
            "violations": violations,
            "violation_rels": violation_rels,
            "defined_by_rels": defined_by_rels,
            "evidence_rels": evidence_rels,
            "responsibilities": responsibilities,
            "resp_rels": resp_rels,
        }

    def _update_state(self, vehicles, frame_id):
        """更新跨帧状态 (制动历史、静止时长)."""
        for v in vehicles:
            eid = v.get("entity_id", "")
            if not eid:
                continue
            brake = v.get("brake", 0.0)
            history = self._brake_history.setdefault(eid, [])
            history.append(brake)
            if len(history) > 30:
                history.pop(0)

            speed = v.get("speed", 0.0)
            if speed < 0.3:
                self._stop_duration[eid] = self._stop_duration.get(eid, 0) + 1
            else:
                self._stop_duration[eid] = 0

    def reset(self):
        self._sv_counter = 0
        self._brake_history.clear()
        self._stop_duration.clear()

    def stats(self):
        return {
            "n_brake_history": len(self._brake_history),
            "n_stop_tracked": len(self._stop_duration),
        }


def _add_violation(rule_code, rule_name, rule_layer,
                   src_id, dst_id, evidence_id, frame_id, severity,
                   violations, violation_rels, defined_by_rels, evidence_rels):
    """辅助函数: 创建 SafetyViolation + 边."""
    sv_id = make_sv_id(rule_code, frame_id, src_id, dst_id)
    pred_name = rule_name
    sv = SafetyViolation(
        entity_id=sv_id,
        rule_code=rule_code, rule_name=rule_name,
        rule_layer=rule_layer, frame_id=frame_id, severity=severity,
        src_id=src_id, dst_id=dst_id,
        predicate_str=f"{pred_name}({src_id}, {dst_id}, Frame_{frame_id})",
        evidence_path=[evidence_id],
    )
    violations.append(sv)
    violation_rels.append(violates(
        src_entity_id=src_id, dst_entity_id=dst_id,
        frame_id=frame_id, valid_from=frame_id,
        rule_code=rule_code, predicate=pred_name,
        sv_id=sv_id, severity=severity,
    ))
    defined_by_rels.append(defined_by(
        sv_id=sv_id, rule_id=rule_code,
        frame_id=frame_id, valid_from=frame_id,
    ))
    evidence_rels.append(supported_by_evidence(
        sv_id=sv_id, evidence_id=evidence_id,
        frame_id=frame_id, valid_from=frame_id,
        evidence_idx=0,
    ))
