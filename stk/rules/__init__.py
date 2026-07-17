"""规则层: RSS 安全距离 + 交通法规 R1-R18 推理 (v3 sec 4)."""

from .nodes import (
    RuleDefinition, RuleParameter, SafetyViolation, ResponsibilityAssignment,
    make_sv_id, make_resp_id,
)
from .relations import (
    build_relation,
    defined_by, uses_param, supported_by_evidence,
    violates, triggers, responsible_for, caused_by,
)
from .rss.model import (
    DEFAULT_RSS_PARAMS,
    compute_dmin_long, check_safe_distance_longitudinal,
    compute_dmin_lat, check_lateral_dangerous_state,
    is_dangerous_state, check_no_proper_response,
    check_responsible_agent, run_rss_check,
)
from .traffic.rules import (
    check_R1_pedestrian_priority, check_R2_red_light,
    check_R3_solid_line_change, check_R4_opposite_meeting,
    check_R5_reversing, check_R7_junction_no_yield,
    check_R8_vulnerable_protection, check_R9_school_zone_speed,
    check_R10_highway_speed, check_R11_weather_speed,
    check_R13_illegal_stop, check_R16_amber_jumping,
    check_R17_wrong_lane, check_R18_wrong_direction_lane,
)
from .generator import RuleEnforcer

__all__ = [
    # nodes
    "RuleDefinition", "RuleParameter", "SafetyViolation",
    "ResponsibilityAssignment", "make_sv_id", "make_resp_id",
    # relations
    "build_relation",
    "defined_by", "uses_param", "supported_by_evidence",
    "violates", "triggers", "responsible_for", "caused_by",
    # rss
    "DEFAULT_RSS_PARAMS",
    "compute_dmin_long", "check_safe_distance_longitudinal",
    "compute_dmin_lat", "check_lateral_dangerous_state",
    "is_dangerous_state", "check_no_proper_response",
    "check_responsible_agent", "run_rss_check",
    # traffic
    "check_R1_pedestrian_priority", "check_R2_red_light",
    "check_R3_solid_line_change", "check_R4_opposite_meeting",
    "check_R5_reversing", "check_R7_junction_no_yield",
    "check_R8_vulnerable_protection", "check_R9_school_zone_speed",
    "check_R10_highway_speed", "check_R11_weather_speed",
    "check_R13_illegal_stop", "check_R16_amber_jumping",
    "check_R17_wrong_lane", "check_R18_wrong_direction_lane",
    # generator
    "RuleEnforcer",
]
