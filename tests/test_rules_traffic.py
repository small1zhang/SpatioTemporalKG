# -*- coding: utf-8 -*-
"""T3: traffic rules R1-R18 eval tests (v3 sec 4.15).

每条规则覆盖正例(应触发违规, sev>0) 与 反例(不触发违规).
返回 Tuple[is_violation: bool, severity: float, evidence: Dict[str, Any]].

注: vehicle["speed"] 视为 m/s, 由 evaluator 内部 *3.6 转 km/h.
"""
from __future__ import annotations
import math
import pytest
from stk.rules.traffic.rules import (
    check_R10_highway_speed,
    check_R11_weather_speed,
    check_R13_illegal_stop,
    check_R16_amber_jumping,
    check_R17_wrong_lane,
    check_R18_wrong_direction_lane,
    check_R1_pedestrian_priority,
    check_R2_red_light,
    check_R3_solid_line_change,
    check_R4_opposite_meeting,
    check_R5_reversing,
    check_R7_junction_no_yield,
    check_R8_vulnerable_protection,
    check_R9_school_zone_speed,
)


def _veh(speed=10.0, x=0.0, y=0.0, vid="V001"):
    return {"id": vid, "speed": speed, "x": x, "y": y}


def _ped(x=20.0, y=0.0, pid="P001"):
    return {"id": pid, "x": x, "y": y}


# ---------- R1 行人优先 (v3 sec 4.15.1) ----------

class TestR1PedestrianPriority:
    def test_violation_close_ped_on_crosswalk(self):
        veh = _veh(speed=8.0); ped = _ped(x=10.0)
        is_v, sev, ev = check_R1_pedestrian_priority(veh, ped, distance=10.0, is_on_crosswalk=True)
        assert is_v is True and sev > 0
        assert ev["distance"] == 10.0

    def test_no_violation_ped_not_on_crosswalk(self):
        veh = _veh(speed=8.0); ped = _ped()
        is_v, sev, _ = check_R1_pedestrian_priority(veh, ped, distance=10.0, is_on_crosswalk=False)
        assert is_v is False and sev == 0.0

    def test_no_violation_too_far(self):
        veh = _veh(speed=8.0); ped = _ped(x=50.0)
        is_v, sev, _ = check_R1_pedestrian_priority(veh, ped, distance=20.0, is_on_crosswalk=True)
        assert is_v is False

    def test_no_violation_almost_stopped(self):
        veh = _veh(speed=0.3); ped = _ped(x=5.0)
        is_v, sev, _ = check_R1_pedestrian_priority(veh, ped, distance=5.0, is_on_crosswalk=True)
        assert is_v is False


# ---------- R2 闯红灯 (v3 sec 4.15.2) ----------

class TestR2RedLight:
    def test_violation_red_light_in_junction(self):
        veh = _veh(speed=8.0); tl = {"id": "TL001", "color": "Red"}
        is_v, sev, _ = check_R2_red_light(veh, tl, in_junction=True, signal_state="Red")
        assert is_v is True and sev > 0

    def test_no_violation_green_light(self):
        veh = _veh(speed=8.0); tl = {"id": "TL001", "color": "Green"}
        is_v, sev, _ = check_R2_red_light(veh, tl, in_junction=True, signal_state="Green")
        assert is_v is False

    def test_no_violation_red_but_stopped(self):
        veh = _veh(speed=0.2); tl = {"id": "TL001", "color": "Red"}
        is_v, sev, _ = check_R2_red_light(veh, tl, in_junction=True, signal_state="Red")
        assert is_v is False

    def test_no_violation_outside_junction(self):
        veh = _veh(speed=8.0); tl = {"id": "TL001", "color": "Red"}
        is_v, sev, _ = check_R2_red_light(veh, tl, in_junction=False, signal_state="Red")
        assert is_v is False


# ---------- R3 实线变道 (v3 sec 4.15.3) ----------

class TestR3SolidLineChange:
    def test_violation(self):
        is_v, sev, _ = check_R3_solid_line_change(_veh(), crossed_solid=True, is_changing_lane=True)
        assert is_v is True and sev > 0

    def test_no_violation_no_solid(self):
        is_v, sev, _ = check_R3_solid_line_change(_veh(), crossed_solid=False, is_changing_lane=True)
        assert is_v is False and sev == 0.0

    def test_no_violation_not_changing(self):
        is_v, sev, _ = check_R3_solid_line_change(_veh(), crossed_solid=True, is_changing_lane=False)
        assert is_v is False


# ---------- R4 对向会车 (v3 sec 4.15.4) ----------

class TestR4OppositeMeeting:
    def test_violation(self):
        a = _veh(speed=15.0); b = _veh(speed=12.0, vid="V002")
        is_v, sev, _ = check_R4_opposite_meeting(a, b, distance=5.0, is_opposite_lane=True)
        assert is_v is True and sev > 0

    def test_no_violation_same_lane(self):
        a = _veh(); b = _veh(vid="V002")
        is_v, sev, _ = check_R4_opposite_meeting(a, b, distance=5.0, is_opposite_lane=False)
        assert is_v is False

    def test_no_violation_too_far(self):
        a = _veh(); b = _veh(vid="V002")
        is_v, sev, _ = check_R4_opposite_meeting(a, b, distance=20.0, is_opposite_lane=True)
        assert is_v is False


# ---------- R5 逆行 (v3 sec 4.15.5) ----------

class TestR5Reversing:
    def test_violation(self):
        is_v, sev, _ = check_R5_reversing(_veh(), angle_diff=180.0, duration_frames=5)
        assert is_v is True and sev > 0

    def test_no_violation_small_angle(self):
        is_v, sev, _ = check_R5_reversing(_veh(), angle_diff=10.0, duration_frames=10)
        assert is_v is False

    def test_no_violation_short_duration(self):
        is_v, sev, _ = check_R5_reversing(_veh(), angle_diff=180.0, duration_frames=1)
        assert is_v is False


# ---------- R7 路口未让行 (v3 sec 4.15.7) ----------

class TestR7JunctionNoYield:
    def test_violation(self):
        veh = _veh(); other = _veh(vid="V002")
        is_v, sev, _ = check_R7_junction_no_yield(
            veh, other, distance=8.0, in_junction=True, other_has_priority=True, is_yielding=False)
        assert is_v is True and sev > 0

    def test_no_violation_yielding(self):
        veh = _veh(); other = _veh(vid="V002")
        is_v, sev, _ = check_R7_junction_no_yield(
            veh, other, distance=8.0, in_junction=True, other_has_priority=True, is_yielding=True)
        assert is_v is False

    def test_no_violation_other_no_priority(self):
        veh = _veh(); other = _veh(vid="V002")
        is_v, sev, _ = check_R7_junction_no_yield(
            veh, other, distance=8.0, in_junction=True, other_has_priority=False, is_yielding=False)
        assert is_v is False

    def test_no_violation_outside_junction(self):
        veh = _veh(); other = _veh(vid="V002")
        is_v, sev, _ = check_R7_junction_no_yield(
            veh, other, distance=8.0, in_junction=False, other_has_priority=True, is_yielding=False)
        assert is_v is False


# ---------- R8 弱势参与者保护 (v3 sec 4.15.8) ----------
# VulnerableUserProtectionViolation <- weather_severity in {poor, severely_reduced}
#                                       AND d<20 AND speed_kmh > school_zone_limit

class TestR8VulnerableProtection:
    def test_violation_poor_weather_close_distance(self):
        # vehicle speed=10 m/s = 36 km/h > 30 limit, weather poor, d<20
        veh = _veh(speed=10.0); ped = _ped(x=5.0)
        is_v, sev, _ = check_R8_vulnerable_protection(
            veh, ped, distance=5.0, weather_severity="poor", school_zone_speed_limit=30.0)
        assert is_v is True and sev > 0

    def test_no_violation_clear_weather(self):
        veh = _veh(speed=10.0); ped = _ped(x=5.0)
        is_v, sev, _ = check_R8_vulnerable_protection(
            veh, ped, distance=5.0, weather_severity="clear")
        assert is_v is False

    def test_no_violation_far_distance(self):
        veh = _veh(speed=10.0); ped = _ped(x=30.0)
        is_v, sev, _ = check_R8_vulnerable_protection(
            veh, ped, distance=30.0, weather_severity="poor")
        assert is_v is False

    def test_no_violation_slow_speed(self):
        veh = _veh(speed=2.0); ped = _ped(x=5.0)
        is_v, sev, _ = check_R8_vulnerable_protection(
            veh, ped, distance=5.0, weather_severity="poor", school_zone_speed_limit=30.0)
        assert is_v is False  # 2 m/s = 7.2 km/h <= 30

    def test_worse_in_severely_reduced(self):
        veh = _veh(speed=10.0); ped = _ped(x=10.0)
        _, sev_poor, _ = check_R8_vulnerable_protection(veh, ped, distance=10.0, weather_severity="poor")
        _, sev_sev, _ = check_R8_vulnerable_protection(veh, ped, distance=10.0, weather_severity="severely_reduced")
        # both violations triggered; severity proportional to ttec(distance)
        assert sev_poor > 0 and sev_sev > 0


# ---------- R9 学区限速 (v3 sec 4.15.9) ----------
# SchoolZoneSpeedViolation <- in_school_zone AND speed_kmh > limit

class TestR9SchoolZoneSpeed:
    def test_violation_over_limit(self):
        # 40 m/s = 144 km/h > 30
        veh = _veh(speed=40.0)
        is_v, sev, _ = check_R9_school_zone_speed(veh, in_school_zone=True, speed_limit=30.0)
        assert is_v is True and sev > 0

    def test_no_violation_under_limit(self):
        # 5 m/s = 18 km/h < 30
        veh = _veh(speed=5.0)
        is_v, sev, _ = check_R9_school_zone_speed(veh, in_school_zone=True, speed_limit=30.0)
        assert is_v is False

    def test_no_violation_outside_school_zone(self):
        veh = _veh(speed=40.0)
        is_v, sev, _ = check_R9_school_zone_speed(veh, in_school_zone=False, speed_limit=30.0)
        assert is_v is False


# ---------- R10 高速公路限速 (v3 sec 4.15.10) ----------
# HighwaySpeedViolation <- road_type == "Highway" AND (speed_kmh > 120 OR speed_kmh < 60)

class TestR10HighwaySpeed:
    def test_violation_too_fast(self):
        # 40 m/s = 144 km/h > 120
        veh = _veh(speed=40.0)
        is_v, sev, _ = check_R10_highway_speed(veh, road_type="Highway", highway_speed_max=120.0)
        assert is_v is True and sev > 0

    def test_violation_too_slow(self):
        # 10 m/s = 36 km/h < 60
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R10_highway_speed(veh, road_type="Highway", highway_speed_min=60.0)
        assert is_v is True and sev > 0

    def test_no_violation_normal_speed(self):
        # 25 m/s = 90 km/h, in [60,120]
        veh = _veh(speed=25.0)
        is_v, sev, _ = check_R10_highway_speed(veh, road_type="Highway")
        assert is_v is False

    def test_no_violation_non_highway(self):
        veh = _veh(speed=40.0)
        is_v, sev, _ = check_R10_highway_speed(veh, road_type="urban")
        assert is_v is False


# ---------- R11 恶劣天气限速 (v3 sec 4.15.11) ----------
# WeatherSpeedViolation <- precipitation > 50 AND speed_kmh > max_allowed

class TestR11WeatherSpeed:
    def test_violation_heavy_rain(self):
        # 30 m/s = 108 km/h > 60 (max_speed when 50 < precipitation <= 80)
        veh = _veh(speed=30.0)
        is_v, sev, _ = check_R11_weather_speed(veh, precipitation=70.0)
        assert is_v is True and sev > 0

    def test_no_violation_clear(self):
        veh = _veh(speed=30.0)
        is_v, sev, _ = check_R11_weather_speed(veh, precipitation=0.0)
        assert is_v is False

    def test_no_violation_under_limit(self):
        # 10 m/s = 36 km/h < 60 even in rain
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R11_weather_speed(veh, precipitation=70.0)
        assert is_v is False

    def test_no_violation_light_rain(self):
        veh = _veh(speed=30.0)
        is_v, sev, _ = check_R11_weather_speed(veh, precipitation=30.0)
        assert is_v is False  # only triggered when precipitation > 50


# ---------- R13 禁停 (v3 sec 4.15.12) ----------
# IllegalStopViolation <- speed<0.3 AND in_no_stop_zone AND duration>=30 frames

class TestR13IllegalStop:
    def test_violation(self):
        veh = _veh(speed=0.0)
        is_v, sev, _ = check_R13_illegal_stop(veh, in_no_stop_zone=True, duration_frames=30)
        assert is_v is True and sev > 0

    def test_no_violation_outside_zone(self):
        veh = _veh(speed=0.0)
        is_v, sev, _ = check_R13_illegal_stop(veh, in_no_stop_zone=False, duration_frames=30)
        assert is_v is False

    def test_no_violation_short_stop(self):
        veh = _veh(speed=0.0)
        is_v, sev, _ = check_R13_illegal_stop(veh, in_no_stop_zone=True, duration_frames=10)
        assert is_v is False


# ---------- R16 黄灯抢行 (v3 sec 4.15.15) ----------
# AmberLightJumpingViolation <- tl_state_before=Yellow AND tl_state_now=Yellow
#                                AND NOT is_at_stop_line AND in_junction

class TestR16AmberJumping:
    def test_violation_yellow_to_yellow_in_junction_not_at_line(self):
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R16_amber_jumping(
            veh, tl_state_before="Yellow", tl_state_now="Yellow",
            is_at_stop_line=False, in_junction=True, speed=10.0)
        assert is_v is True and sev > 0

    def test_no_violation_green_to_yellow(self):
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R16_amber_jumping(
            veh, tl_state_before="Green", tl_state_now="Yellow",
            is_at_stop_line=False, in_junction=True, speed=10.0)
        assert is_v is False

    def test_no_violation_at_stop_line(self):
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R16_amber_jumping(
            veh, tl_state_before="Yellow", tl_state_now="Yellow",
            is_at_stop_line=True, in_junction=True, speed=10.0)
        assert is_v is False

    def test_no_violation_outside_junction(self):
        veh = _veh(speed=10.0)
        is_v, sev, _ = check_R16_amber_jumping(
            veh, tl_state_before="Yellow", tl_state_now="Yellow",
            is_at_stop_line=False, in_junction=False, speed=10.0)
        assert is_v is False


# ---------- R17 不按规定车道 (v3 sec 4.15.16) ----------

class TestR17WrongLane:
    def test_violation_bus_lane(self):
        is_v, sev, _ = check_R17_wrong_lane(_veh(), lane_type="BusOnly", duration_on_lane=10)
        assert is_v is True and sev > 0

    def test_no_violation_driving_lane(self):
        is_v, sev, _ = check_R17_wrong_lane(_veh(), lane_type="Driving", duration_on_lane=10)
        assert is_v is False

    def test_no_violation_short_duration(self):
        is_v, sev, _ = check_R17_wrong_lane(_veh(), lane_type="BusOnly", duration_on_lane=2)
        assert is_v is False


# ---------- R18 不按导向车道 (v3 sec 4.15.17) ----------

class TestR18WrongDirectionLane:
    def test_violation(self):
        is_v, sev, _ = check_R18_wrong_direction_lane(
            _veh(), in_junction=True, maneuver_action="straight", designated_direction="left")
        assert is_v is True and sev > 0

    def test_no_violation_correct_direction(self):
        is_v, sev, _ = check_R18_wrong_direction_lane(
            _veh(), in_junction=True, maneuver_action="left", designated_direction="left")
        assert is_v is False

    def test_no_violation_outside_junction(self):
        is_v, sev, _ = check_R18_wrong_direction_lane(
            _veh(), in_junction=False, maneuver_action="straight", designated_direction="left")
        assert is_v is False


# ---------- 返回元组结构契约 ----------

class TestReturnValueContract:
    """所有评估器返回 (is_violation: bool, severity: float, evidence: dict)."""

    def _check_contract(self, result):
        assert isinstance(result, tuple) and len(result) == 3
        is_v, sev, ev = result
        assert isinstance(is_v, bool)
        assert isinstance(sev, float)
        assert isinstance(ev, dict)
        assert sev >= 0.0
        if is_v:
            assert sev > 0.0

    def test_all_rules_return_proper_triple(self):
        veh = _veh(speed=10.0)
        ped = _ped(x=10.0)
        results = [
            check_R1_pedestrian_priority(veh, ped, distance=10.0, is_on_crosswalk=True),
            check_R2_red_light(veh, {"id": "TL", "color": "Red"}, in_junction=True, signal_state="Red"),
            check_R3_solid_line_change(veh, crossed_solid=True, is_changing_lane=True),
            check_R4_opposite_meeting(veh, _veh(vid="V2", speed=10.0), distance=5.0, is_opposite_lane=True),
            check_R5_reversing(veh, angle_diff=180.0, duration_frames=5),
            check_R7_junction_no_yield(veh, _veh(vid="V2"), distance=5.0,
                                       in_junction=True, other_has_priority=True, is_yielding=False),
            check_R8_vulnerable_protection(veh, ped, distance=10.0, weather_severity="poor"),
            check_R9_school_zone_speed(_veh(speed=40.0), in_school_zone=True, speed_limit=30.0),
            check_R10_highway_speed(_veh(speed=40.0), road_type="Highway"),
            check_R11_weather_speed(_veh(speed=30.0), precipitation=70.0),
            check_R13_illegal_stop(_veh(speed=0.0), in_no_stop_zone=True, duration_frames=30),
            check_R16_amber_jumping(veh, tl_state_before="Yellow", tl_state_now="Yellow",
                                    is_at_stop_line=False, in_junction=True, speed=10.0),
            check_R17_wrong_lane(veh, lane_type="BusOnly", duration_on_lane=10),
            check_R18_wrong_direction_lane(veh, in_junction=True,
                                           maneuver_action="straight", designated_direction="left"),
        ]
        for r in results:
            self._check_contract(r)