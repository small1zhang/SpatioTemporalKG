"""RSS 子层: 车辆间纵横向安全距离的实时校验 (v3 sec 4.7-4.11)."""
from .model import (
    DEFAULT_RSS_PARAMS,
    compute_dmin_long, check_safe_distance_longitudinal,
    compute_dmin_lat, check_lateral_dangerous_state,
    is_dangerous_state, check_no_proper_response,
    check_responsible_agent, run_rss_check,
)
__all__ = [
    "DEFAULT_RSS_PARAMS",
    "compute_dmin_long", "check_safe_distance_longitudinal",
    "compute_dmin_lat", "check_lateral_dangerous_state",
    "is_dangerous_state", "check_no_proper_response",
    "check_responsible_agent", "run_rss_check",
]
