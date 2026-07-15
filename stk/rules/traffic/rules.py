"""
交通法规 R1-R18 检测器 (v3 sec 4.14 - 4.16)

每个 check_Rx() 函数:
  - 输入: 场景层/行为层实体 dict + 关系列表
  - 输出: (is_violation, severity, extra_attrs) 元组

与 v3 sec 4.15 形式化判别式一一对应。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


def severity_from_ttec(distance, threshold_near, threshold_far, max_sev=0.9, min_sev=0.2):
    """基于距离的严重度插值。"""
    if distance >= threshold_far:
        return min_sev
    if distance <= threshold_near:
        return max_sev
    ratio = (threshold_far - distance) / (threshold_far - threshold_near)
    return round(min_sev + ratio * (max_sev - min_sev), 2)


def check_R1_pedestrian_priority(
    vehicle: Dict[str, Any],
    pedestrian: Dict[str, Any],
    distance: float,
    is_on_crosswalk: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R1 行人优先(人行横道)(v3 sec 4.15.1)

    # YieldingToPedestrianViolation(A,P,t) <=>
    #   is_on_crosswalk(P,t) AND distance(A,P,t) < 15 AND speed(A,t) > 0.5

    Returns: (is_violation, severity, extra)
    """
    speed = vehicle.get("speed", 0.0)
    is_violation = is_on_crosswalk and distance < 15.0 and speed > 0.5
    sev = severity_from_ttec(distance, 3.0, 15.0) if is_violation else 0.0
    return is_violation, sev, {"distance": distance, "ped_on_crosswalk": is_on_crosswalk, "speed": speed}


def check_R2_red_light(
    vehicle: Dict[str, Any],
    traffic_light: Dict[str, Any],
    in_junction: bool = False,
    signal_state: str = "Green",
) -> Tuple[bool, float, Dict[str, Any]]:
    """R2 闯红灯 (v3 sec 4.15.2)

    # RedLightViolation(A,R,t) <=>
    #   in_junction(A,R,t) AND signal_color(A,t-delta)=Red AND speed(A,t) > 0.3
    """
    speed = vehicle.get("speed", 0.0)
    is_violation = in_junction and signal_state == "Red" and speed > 0.3
    sev = 0.95 if is_violation else 0.0
    return is_violation, sev, {"signal_state": signal_state, "speed": speed, "in_junction": in_junction}


def check_R3_solid_line_change(
    vehicle: Dict[str, Any],
    crossed_solid: bool = False,
    is_changing_lane: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R3 实线变道 (v3 sec 4.15.3)

    IllegalLaneCrossing(A,R,t) <=>
      crossed_lane_markings 包含 Solid AND changing_lane(A,t)
    """
    is_violation = crossed_solid and is_changing_lane
    sev = 0.75 if is_violation else 0.0
    return is_violation, sev, {"crossed_solid": crossed_solid, "changing_lane": is_changing_lane}


def check_R4_opposite_meeting(
    vehicle_a: Dict[str, Any],
    vehicle_b: Dict[str, Any],
    distance: float,
    is_opposite_lane: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R4 对向会车违规 (v3 sec 4.15.4)

    WrongSideMeetingViolation(A,B,t) <=>
      opposite_direction(A,B,t) AND is_in_opposite_lane(A,B,t) AND distance < 10
    """
    is_violation = is_opposite_lane and distance < 10.0
    sev = severity_from_ttec(distance, 2.0, 10.0) if is_violation else 0.0
    return is_violation, sev, {"distance": distance, "is_opposite_lane": is_opposite_lane}


def check_R5_reversing(
    vehicle: Dict[str, Any],
    angle_diff: float = 0.0,
    duration_frames: int = 0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R5 逆行 (v3 sec 4.15.5)

    IllegalReversing(A,R,t) <=>
      angle(velocity, heading) > 135 deg AND duration >= 5 frames
    """
    is_violation = angle_diff > 135.0 and duration_frames >= 5
    sev = 0.9 if is_violation else 0.0
    return is_violation, sev, {"angle_diff_deg": angle_diff, "duration_frames": duration_frames}


def check_R7_junction_no_yield(
    vehicle: Dict[str, Any],
    other: Dict[str, Any],
    distance: float,
    in_junction: bool = False,
    other_has_priority: bool = False,
    is_yielding: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R7 路口未让行 (v3 sec 4.15.7)

    # JunctionNoYieldViolation(A,B,t) <=>
    #   in_junction(A,R,t) AND other_has_priority(B,R,t)
    #   AND NOT yielding(A,B,t) AND speed(A,t) > 0.3
    """
    speed = vehicle.get("speed", 0.0)
    is_violation = in_junction and other_has_priority and not is_yielding and speed > 0.3
    sev = severity_from_ttec(distance, 5.0, 30.0) if is_violation else 0.0
    return is_violation, sev, {
        "in_junction": in_junction, "other_has_priority": other_has_priority,
        "is_yielding": is_yielding, "speed": speed, "distance": distance,
    }


def check_R8_vulnerable_protection(
    vehicle: Dict[str, Any],
    pedestrian: Dict[str, Any],
    distance: float,
    weather_severity: str = "clear",
    school_zone_speed_limit: float = 30.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R8 弱势参与者保护 (v3 sec 4.15.8)

    VulnerableUserProtectionViolation(A,P,t) <=>
      weather_severity in {poor, severely_reduced}
      AND distance < 20 AND speed > school_zone_speed_limit
    """
    speed = vehicle.get("speed_kmh", vehicle.get("speed", 0.0) * 3.6)
    bad_weather = weather_severity in ("poor", "severely_reduced")
    is_violation = bad_weather and distance < 20.0 and speed > school_zone_speed_limit
    sev = severity_from_ttec(distance, 5.0, 20.0) if is_violation else 0.0
    return is_violation, sev, {
        "distance": distance, "weather": weather_severity,
        "speed_kmh": speed, "limit": school_zone_speed_limit,
    }


def check_R9_school_zone_speed(
    vehicle: Dict[str, Any],
    in_school_zone: bool = False,
    speed_limit: float = 30.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R9 学区限速 (v3 sec 4.15.9)

    SchoolZoneSpeedViolation(A,E,t) <=>
      road_id in SchoolZoneSet AND speed > 30 km/h
    """
    speed_kmh = vehicle.get("speed_kmh", vehicle.get("speed", 0.0) * 3.6)
    is_violation = in_school_zone and speed_kmh > speed_limit
    sev = min(0.8, (speed_kmh - speed_limit) / speed_limit * 0.5) if is_violation else 0.0
    return is_violation, round(sev, 2), {"speed_kmh": speed_kmh, "limit": speed_limit}


def check_R10_highway_speed(
    vehicle: Dict[str, Any],
    road_type: str = "",
    highway_speed_max: float = 120.0,
    highway_speed_min: float = 60.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R10 高速公路限速 (v3 sec 4.15.10)

    HighwaySpeedViolation(A,R,t) <=>
      road_type == Highway AND (speed > 120 OR speed < 60) km/h
    """
    speed_kmh = vehicle.get("speed_kmh", vehicle.get("speed", 0.0) * 3.6)
    is_violation = (road_type == "Highway") and (speed_kmh > highway_speed_max or speed_kmh < highway_speed_min)
    sev = 0.7 if is_violation else 0.0
    return is_violation, sev, {"speed_kmh": speed_kmh, "road_type": road_type,
                                "max": highway_speed_max, "min": highway_speed_min}


def check_R11_weather_speed(
    vehicle: Dict[str, Any],
    precipitation: float = 0.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R11 恶劣天气限速 (v3 sec 4.15.11)

    WeatherSpeedViolation(A,E,t) <=>
      precipitation > 50 AND speed > max_allowed(weather_severity)
    """
    speed_kmh = vehicle.get("speed_kmh", vehicle.get("speed", 0.0) * 3.6)
    if precipitation > 50:
        max_speed = 60.0 if precipitation <= 80 else 40.0
        is_violation = speed_kmh > max_speed
        sev = 0.6 if is_violation else 0.0
    else:
        is_violation = False
        sev = 0.0
        max_speed = 120.0
    return is_violation, sev, {"speed_kmh": speed_kmh, "precipitation": precipitation, "max_speed": max_speed}


def check_R13_illegal_stop(
    vehicle: Dict[str, Any],
    in_no_stop_zone: bool = False,
    duration_frames: int = 0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R13 禁停 (v3 sec 4.15.12)

    IllegalStopViolation(A,R,t) <=>
      speed < 0.3 AND A in NoStopZone AND duration >= 30 frames
    """
    speed = vehicle.get("speed", 0.0)
    is_violation = speed < 0.3 and in_no_stop_zone and duration_frames >= 30
    sev = 0.5 if is_violation else 0.0
    return is_violation, sev, {"speed": speed, "in_no_stop_zone": in_no_stop_zone, "duration": duration_frames}


def check_R16_amber_jumping(
    vehicle: Dict[str, Any],
    tl_state_before: str = "Green",
    tl_state_now: str = "Yellow",
    is_at_stop_line: bool = False,
    in_junction: bool = False,
    speed: Optional[float] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R16 黄灯抢行 (v3 sec 4.15.15)

    AmberLightJumpingViolation(A,TL,t) <=>
      TL.state(t-delta)=Yellow AND TL.state(t)=Yellow
      AND NOT is_at_stop_line(A,R,t-delta) AND in_junction(A,R,t)
    """
    is_violation = (tl_state_before == "Yellow" and tl_state_now == "Yellow"
                    and not is_at_stop_line and in_junction)
    sev = 0.7 if is_violation else 0.0
    return is_violation, sev, {
        "tl_state_before": tl_state_before, "tl_state_now": tl_state_now,
        "is_at_stop_line": is_at_stop_line, "in_junction": in_junction,
    }


def check_R17_wrong_lane(
    vehicle: Dict[str, Any],
    lane_type: str = "Driving",
    duration_on_lane: int = 0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """R17 不按规定车道 (v3 sec 4.15.16)

    WrongLaneViolation(A,R,t) <=>
      lane_type(R) != Driving AND duration(A on R) >= 5 frames
    """
    is_violation = lane_type != "Driving" and duration_on_lane >= 5
    sev = 0.6 if is_violation else 0.0
    return is_violation, sev, {"lane_type": lane_type, "duration": duration_on_lane}


def check_R18_wrong_direction_lane(
    vehicle: Dict[str, Any],
    in_junction: bool = False,
    maneuver_action: str = "",
    designated_direction: str = "",
) -> Tuple[bool, float, Dict[str, Any]]:
    """R18 不按导向车道 (v3 sec 4.15.17)

    WrongDirectionLaneViolation(A,R,t) <=>
      in_junction(A,R,t) AND maneuver_action != designated_direction
    """
    is_violation = in_junction and maneuver_action != "" and designated_direction != "" and maneuver_action != designated_direction
    sev = 0.6 if is_violation else 0.0
    return is_violation, sev, {"maneuver": maneuver_action, "designated": designated_direction}
