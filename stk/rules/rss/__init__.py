"""RSS 子层: 车辆间纵横向安全距离的实时校验 (v3 sec 4.7-4.11) + 扩充场景规则 (§8.5.2/8.5.3/9.2/10)."""
from .model import (
    DEFAULT_RSS_PARAMS,
    compute_dmin_long, check_safe_distance_longitudinal,
    compute_dmin_lat, check_lateral_dangerous_state,
    is_dangerous_state, check_no_proper_response,
    check_responsible_agent, run_rss_check,
)
from .extended import (
    # 规则码常量
    RSS_CUTIN_RULE_CODE, RSS_CUTOUT_RULE_CODE,
    RSS_NPR_ENH_RULE_CODE, RSS_CZ_ADAPT_RULE_CODE,
    EXTENDED_RSS_RULE_CODES, EXTENDED_PRED_NAMES,
    # 参数/阈值
    CUTIN_DISTANCE_MULTIPLIER, CUTOUT_BUFFER_FRAMES,
    NPR_ENH_BRAKE_LOW, NPR_ENH_BRAKE_HIGH, NPR_ENH_TRANSITION_FRAMES,
    CONSTRUCTION_ZONE_PARAMS,
    # 检测函数
    check_cutin_violation,
    check_cutout_violation,
    check_no_proper_response_enhanced,
    check_construction_zone_adaptive,
    get_construction_zone_params,
    run_rss_extended_check,
)

__all__ = [
    # 基本 RSS (model.py)
    "DEFAULT_RSS_PARAMS",
    "compute_dmin_long", "check_safe_distance_longitudinal",
    "compute_dmin_lat", "check_lateral_dangerous_state",
    "is_dangerous_state", "check_no_proper_response",
    "check_responsible_agent", "run_rss_check",
    # 扩充场景规则 (extended.py)
    "RSS_CUTIN_RULE_CODE", "RSS_CUTOUT_RULE_CODE",
    "RSS_NPR_ENH_RULE_CODE", "RSS_CZ_ADAPT_RULE_CODE",
    "EXTENDED_RSS_RULE_CODES", "EXTENDED_PRED_NAMES",
    "CUTIN_DISTANCE_MULTIPLIER", "CUTOUT_BUFFER_FRAMES",
    "NPR_ENH_BRAKE_LOW", "NPR_ENH_BRAKE_HIGH", "NPR_ENH_TRANSITION_FRAMES",
    "CONSTRUCTION_ZONE_PARAMS",
    "check_cutin_violation",
    "check_cutout_violation",
    "check_no_proper_response_enhanced",
    "check_construction_zone_adaptive",
    "get_construction_zone_params",
    "run_rss_extended_check",
]
