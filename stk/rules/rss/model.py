"""
RSS 形式化模型 (v3 sec 4.7 - 4.8 + sec 4.11)

实现 RSS（Responsibility-Sensitive Safety）的三个核心算子:
  1. 纵向安全距离 (v3 sec 4.8.1)
  2. 横向安全距离 (v3 sec 4.8.2)
  3. 复合状态与责任归因 (v3 sec 4.8.3)

所有公式直接翻译 v3 文档中的 LaTeX 形式化定义。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# RSS 默认参数 (v3 sec 4.8.1 + sec 4.8.2 参数表)
# ============================================================

DEFAULT_RSS_PARAMS: Dict[str, float] = {
    "rho": 0.1,                # 反应时间 (s)
    "a_max_accel": 1.5,        # 最大加速 (m/s^2)
    "a_min_brake_long": 4.0,   # 最小纵向减速 (m/s^2)
    "a_brake_long": 8.0,       # 前车最大减速 (m/s^2)
    "mu": 0.5,                 # 横向安全裕度 (m)
    "a_min_brake_lat": 1.0,    # 最小横向减速 (m/s^2)
    "a_brake_lat": 3.0,        # 目标横向减速 (m/s^2)
}


# ============================================================
# 1. 纵向安全距离 (v3 sec 4.8.1)
# ============================================================


def compute_dmin_long(
    v_A: float,      # 后车 A 当前速度 (m/s)
    v_B: float,      # 前车 B 当前速度 (m/s)
    params: Optional[Dict[str, float]] = None,
) -> float:
    """计算纵向最小安全距离 d_min^long(A,B,t) (v3 sec 4.8.1).

    公式:
    d_min = max(0, v_A * rho + 0.5 * a_max_accel * rho^2
           + (v_A + a_max_accel * rho)^2 / (2 * a_min_brake)
           - v_B^2 / (2 * a_brake))
    """
    p = params or DEFAULT_RSS_PARAMS
    rho = p["rho"]
    a_max = p["a_max_accel"]
    a_min_b = p["a_min_brake_long"]
    a_brake = p["a_brake_long"]

    term1 = v_A * rho
    term2 = 0.5 * a_max * rho * rho
    term3 = (v_A + a_max * rho) ** 2 / (2.0 * a_min_b)
    term4 = v_B * v_B / (2.0 * a_brake)

    d_min = max(0.0, term1 + term2 + term3 - term4)
    return d_min


def check_safe_distance_longitudinal(
    d_actual: float,   # 实际纵向距离 (m)
    v_A: float,        # 后车速度 (m/s)
    v_B: float,        # 前车速度 (m/s)
    params: Optional[Dict[str, float]] = None,
) -> Tuple[bool, float, float]:
    """纵向危险判定 (v3 sec 4.8.1).

    SafeDistanceViolation(A,B,t) <=> d < d_min

    Returns:
        (is_violation, d_actual, d_min)
    """
    d_min = compute_dmin_long(v_A, v_B, params)
    is_violation = d_actual < d_min
    return is_violation, d_actual, d_min


# ============================================================
# 2. 横向安全距离 (v3 sec 4.8.2)
# ============================================================


def compute_dmin_lat(
    v_lat_A: float,   # A 的横向速度 (m/s)
    v_lat_B: float,   # B 的横向速度 (m/s)
    params: Optional[Dict[str, float]] = None,
) -> float:
    """计算横向最小安全距离 d_min^lat(A,B,t) (v3 sec 4.8.2).

    公式:
    d_min = mu + v_lat_A^2 / (2 * a_min_brake_lat) + rho * v_lat_A
            - v_lat_B^2 / (2 * a_min_brake_lat_B)
    """
    p = params or DEFAULT_RSS_PARAMS
    mu = p["mu"]
    rho = p["rho"]
    a_min_lat = p["a_min_brake_lat"]

    # 简化: 假设 a_min_lat_A == a_min_lat_B
    term_A = v_lat_A ** 2 / (2.0 * a_min_lat)
    term_rho = rho * v_lat_A
    term_B = v_lat_B ** 2 / (2.0 * a_min_lat)

    d_min = mu + term_A + term_rho - term_B
    return max(0.0, d_min)


def check_lateral_dangerous_state(
    d_lat_actual: float,
    v_lat_A: float,
    v_lat_B: float,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[bool, float, float]:
    """横向危险判定 (v3 sec 4.8.2).

    LateralDangerousState(A,B,t) <=> d_lat < d_min_lat

    Returns:
        (is_violation, d_lat_actual, d_min_lat)
    """
    d_min = compute_dmin_lat(v_lat_A, v_lat_B, params)
    is_violation = d_lat_actual < d_min
    return is_violation, d_lat_actual, d_min


# ============================================================
# 3. 复合状态与责任归因 (v3 sec 4.8.3)
# ============================================================


def is_dangerous_state(
    is_long_violation: bool,
    is_lat_violation: bool,
) -> bool:
    """DangerousState = SafeDistanceViolation OR LateralDangerousState

    v3 sec 4.8.3.
    """
    return is_long_violation or is_lat_violation


def check_no_proper_response(
    brake_values: List[float],
    threshold: float = 0.3,
    required_consecutive: int = 3,
) -> bool:
    """反应不当判定 (v3 sec 4.8.3).

    NoProperResponse(A,t) <=> DangerousState(A,B,t)
        AND brake(A, t+k) < threshold for k=0..required_consecutive-1

    Args:
        brake_values: 连续 required_consecutive 帧的 brake 值列表
        threshold: brake 阈值 (0.3)
        required_consecutive: 需要连续满足的帧数

    Returns:
        True if NoProperResponse
    """
    if len(brake_values) < required_consecutive:
        return False
    return all(b < threshold for b in brake_values[-required_consecutive:])


def check_responsible_agent(
    ego_has_no_proper_response: bool,
    other_is_compliant: bool,
) -> bool:
    """责任归因 (v3 sec 4.8.3).

    ResponsibleAgent(A, event_k) <=>
        NoProperResponse(A, t) AND B(t) is compliant
    """
    return ego_has_no_proper_response and other_is_compliant


# ============================================================
# 4. 全量 RSS 检查 — 一键运行
# ============================================================


def run_rss_check(
    d_long: float,
    d_lat: float,
    v_A: float,
    v_B: float,
    v_lat_A: float,
    v_lat_B: float,
    brake_values: List[float],
    other_is_compliant: bool = True,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """对一对车辆(A,B)运行完整的 RSS 检查。

    Args:
        d_long: 纵向实际距离 (m)
        d_lat: 横向实际距离 (m)
        v_A: 后车速度 (m/s)
        v_B: 前车速度 (m/s)
        v_lat_A: A 的横向速度 (m/s)
        v_lat_B: B 的横向速度 (m/s)
        brake_values: 最近连续帧的 brake 值列表
        other_is_compliant: 对方行为是否合规
        params: RSS 参数

    Returns:
        dict with keys:
          is_long_violation, d_min_long,
          is_lat_violation, d_min_lat,
          is_dangerous, is_no_proper_response, is_responsible
    """
    is_long_v, _, d_min_long = check_safe_distance_longitudinal(d_long, v_A, v_B, params)
    is_lat_v, _, d_min_lat = check_lateral_dangerous_state(d_lat, v_lat_A, v_lat_B, params)
    dangerous = is_dangerous_state(is_long_v, is_lat_v)
    no_proper = check_no_proper_response(brake_values) if dangerous else False
    responsible = check_responsible_agent(no_proper, other_is_compliant) if dangerous else False

    return {
        "is_long_violation": is_long_v,
        "d_min_long": round(d_min_long, 2),
        "is_lat_violation": is_lat_v,
        "d_min_lat": round(d_min_lat, 2),
        "is_dangerous": dangerous,
        "is_no_proper_response": no_proper,
        "is_responsible": responsible,
    }
