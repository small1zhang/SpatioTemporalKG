"""
RSS 扩充场景规则 (Mobileye RSS v3.0 §8.5.2 / §8.5.3 / §9.2 / §10)

实现 4 条基本 RSS 之外的扩充场景规则:
  1. Cut-in 切入安全距离      (v3.0 §8.5.2) — RSS_CUTIN
  2. Cut-out 驶离缓冲          (v3.0 §8.5.3) — RSS_CUTOUT
  3. 反应不当判定增强           (v3.0 §9.2)   — RSS_NPR_ENH
  4. 施工路段参数自适应          (v3.0 §10)    — RSS_CZ_ADAPT

与 model.py 的基本 RSS 公式 (3.11)/(3.13)/(3.16) 互补, 不修改基本算子.
所有规则均以纯函数形式提供, 跨帧状态由 caller (RuleEnforcer) 维护.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .model import (
    DEFAULT_RSS_PARAMS,
    compute_dmin_long,
    compute_dmin_lat,
    check_safe_distance_longitudinal,
    check_lateral_dangerous_state,
)


# ============================================================
# 0. 扩充规则码常量与参数表
# ============================================================

# 规则码 (与论文 §3.3.3.1a 表 3-17a 严格对齐)
RSS_CUTIN_RULE_CODE = "RSS_CUTIN"
RSS_CUTOUT_RULE_CODE = "RSS_CUTOUT"
RSS_NPR_ENH_RULE_CODE = "RSS_NPR_ENH"
RSS_CZ_ADAPT_RULE_CODE = "RSS_CZ_ADAPT"

# 扩充规则码集合 (用于 RuleEnforcer 统一识别)
EXTENDED_RSS_RULE_CODES = frozenset({
    RSS_CUTIN_RULE_CODE,
    RSS_CUTOUT_RULE_CODE,
    RSS_NPR_ENH_RULE_CODE,
    RSS_CZ_ADAPT_RULE_CODE,
})

# 扩充规则的中文谓词名 (与生成器的 pred_name 对齐)
EXTENDED_PRED_NAMES: Dict[str, str] = {
    RSS_CUTIN_RULE_CODE: "CutInSafeDistanceViolation",
    RSS_CUTOUT_RULE_CODE: "CutOutSafeDistanceViolation",
    RSS_NPR_ENH_RULE_CODE: "NoProperResponseEnhanced",
    RSS_CZ_ADAPT_RULE_CODE: "ConstructionZoneAdaptive",
}

# 切入场景的安全距离倍数 (Mobileye v3.0 §8.5.2 推荐 1.5×)
CUTIN_DISTANCE_MULTIPLIER = 1.5

# Cut-out 缓冲窗口 (以帧数计; 假设 20 Hz → 60 帧 = 3 s)
CUTOUT_BUFFER_FRAMES = 60

# 反应不当增强参数
NPR_ENH_BRAKE_LOW = 0.3       # 与基本 NPR 一致的低阈值
NPR_ENH_BRAKE_HIGH = 0.7      # 紧急制动时应迅速达到的高阈值
NPR_ENH_TRANSITION_FRAMES = 3 # low→high 允许的最大过渡帧数

# 施工路段参数集 Θ_RSS^cz (论文 §3.3.3.1a 公式 3.17d)
# 在基本参数上: ρ + 0.1s, 2×a_max_accel, 0.75×a_min_brake_long
CONSTRUCTION_ZONE_PARAMS: Dict[str, float] = {
    "rho": DEFAULT_RSS_PARAMS["rho"] + 0.1,           # 0.4 s
    "a_max_accel": DEFAULT_RSS_PARAMS["a_max_accel"] * 2.0,  # 1.0 m/s²
    "a_min_brake_long": DEFAULT_RSS_PARAMS["a_min_brake_long"] * 0.75,  # 2.25 m/s²
    "a_brake_long": DEFAULT_RSS_PARAMS["a_brake_long"],
    "mu": DEFAULT_RSS_PARAMS["mu"],
    "a_min_brake_lat": DEFAULT_RSS_PARAMS["a_min_brake_lat"],
    "a_brake_lat": DEFAULT_RSS_PARAMS["a_brake_lat"],
}


# ============================================================
# 1. Cut-in 切入安全距离 (v3.0 §8.5.2)
# ============================================================

def check_cutin_violation(
    d_long: float,
    v_A: float,
    v_B: float,
    is_changing_lane: bool,
    ahead_of: bool,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[bool, float, float]:
    """Cut-in 切入安全距离违规判定 (论文式 3.17a).

    触发条件:
      - B 车正在变道 (changing_lane)
      - B 已切入 A 的前方 (ahead_of(B, A))
      - 纵向距离低于 1.5 × d_min_long

    Args:
        d_long: A 与 B 之间的纵向距离
        v_A, v_B: A/B 速度
        is_changing_lane: B 是否正在变道 (来自行为层 changing_lane 关系)
        ahead_of: B 是否在 A 的前方
        params: RSS 参数 (None 表示默认)

    Returns:
        (is_violation, severity, d_min_cutin)
        d_min_cutin = 1.5 * d_min_long (切入点完成时应保持的安全距离)
    """
    # 取基本纵向 d_min
    _, _, d_min_long = check_safe_distance_longitudinal(d_long, v_A, v_B, params)
    # 论文式 (3.17a): min(d_min_long, 1.5 * d_min_long) = d_min_long 但 Mobileye v3.0 §8.5.2
    # 实际定义为 1.5 × d_min_long 作为切入场景的保守上界
    d_min_cutin = CUTIN_DISTANCE_MULTIPLIER * d_min_long

    if not is_changing_lane or not ahead_of:
        return False, 0.0, round(d_min_cutin, 2)

    is_viol = d_long < d_min_cutin
    # severity 按距离差与安全距离的比例计算
    sev = 0.0
    if is_viol and d_min_cutin > 1e-3:
        sev = min(1.0, (d_min_cutin - d_long) / max(d_min_cutin, 1e-3))
    return is_viol, round(sev, 3), round(d_min_cutin, 2)


# ============================================================
# 2. Cut-out 驶离缓冲 (v3.0 §8.5.3)
# ============================================================

def check_cutout_violation(
    d_long_new: float,
    v_A: float,
    v_C: float,
    exit_lane_observed: bool,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[bool, float, float]:
    """Cut-out 驶离缓冲违规判定 (论文式 3.17b).

    触发条件:
      - B 车驶离原车道 (exit_lane_observed=True 由跨帧 _cutout_buffer 追踪)
      - K 帧缓冲窗口内, A 与新前车 C 的纵向距离必须维持原 RSS 要求

    Args:
        d_long_new: A 与新前车 C 的纵向距离
        v_A: A 速度
        v_C: C 速度
        exit_lane_observed: 是否处于 cut-out 缓冲窗口内
        params: RSS 参数

    Returns:
        (is_violation, severity, d_min_long_new)
    """
    _, _, d_min_long = check_safe_distance_longitudinal(d_long_new, v_A, v_C, params)
    if not exit_lane_observed:
        return False, 0.0, round(d_min_long, 2)

    is_viol = d_long_new < d_min_long
    sev = 0.0
    if is_viol and d_min_long > 1e-3:
        sev = min(1.0, (d_min_long - d_long_new) / max(d_min_long, 1e-3))
    return is_viol, round(sev, 3), round(d_min_long, 2)


# ============================================================
# 3. 反应不当判定增强 (v3.0 §9.2)
# ============================================================

def check_no_proper_response_enhanced(
    brake_values: List[float],
    speed_values: Optional[List[float]] = None,
    rho: float = DEFAULT_RSS_PARAMS["rho"],
    a_min_brake: float = DEFAULT_RSS_PARAMS["a_min_brake_long"],
    required_consecutive: int = 3,
) -> Tuple[bool, float, str]:
    """NoProperResponse 增强判定 (论文式 3.17c).

    在基本 NPR (brake < 0.3 持续 3 帧) 基础上, 增加:
      - 制动速率学约束: brake_jerk 不达标时也记违规
      - 速度因变阈值: speed > v_safe = v - a_min_brake * ρ

    Args:
        brake_values: 最近 required_consecutive+ 帧的 brake 值
        speed_values: 对应帧的速度值 (None 表示不参与速度约束)
        rho: 反应时间
        a_min_brake: 最小合理制动减速度
        required_consecutive: 需要连续满足的帧数

    Returns:
        (is_violation, severity, reason)
        reason ∈ {"basic_low_brake", "enhanced_jerk", "enhanced_speed_safe"}
    """
    if len(brake_values) < required_consecutive:
        return False, 0.0, "insufficient_history"

    recent_brakes = brake_values[-required_consecutive:]

    # 基本条件: 3 帧内持续低制动
    basic_viol = all(b < NPR_ENH_BRAKE_LOW for b in recent_brakes)

    # 速度-时间因变阈值: v_safe = v - a_min_brake * rho
    speed_viol = False
    if speed_values is not None and len(speed_values) >= required_consecutive:
        recent_speeds = speed_values[-required_consecutive:]
        # 取首帧速度作为参考
        v_ref = recent_speeds[0]
        v_safe = v_ref - a_min_brake * rho
        # 若 3 帧内速度仍高于 v_safe, 说明制动未有效减速
        speed_viol = all(s > v_safe for s in recent_speeds)

    # 制动速率学约束: 从低制动到高制动应 ≤ NPR_ENH_TRANSITION_FRAMES 帧
    jerk_viol = False
    if len(brake_values) >= required_consecutive + NPR_ENH_TRANSITION_FRAMES:
        # 看是否在 NPR_ENH_TRANSITION_FRAMES 帧内出现从低到高的过渡
        window = brake_values[-(required_consecutive + NPR_ENH_TRANSITION_FRAMES):]
        low_to_high = False
        for i in range(len(window) - 1):
            if window[i] < NPR_ENH_BRAKE_LOW and window[i + 1] >= NPR_ENH_BRAKE_HIGH:
                low_to_high = True
                break
        # 若 6 帧内无 low→high 过渡且最近 3 帧持续低制动, 则 jerk_viol=True
        if basic_viol and not low_to_high:
            jerk_viol = True

    is_viol = basic_viol or jerk_viol or speed_viol
    if not is_viol:
        return False, 0.0, "ok"

    # severity
    if jerk_viol:
        sev, reason = 0.9, "enhanced_jerk"
    elif speed_viol:
        sev, reason = 0.8, "enhanced_speed_safe"
    else:
        sev, reason = 0.6, "basic_low_brake"

    return is_viol, round(sev, 3), reason


# ============================================================
# 4. 施工路段参数自适应 (v3.0 §10)
# ============================================================

def get_construction_zone_params() -> Dict[str, float]:
    """返回施工路段专用参数集 Θ_RSS^cz (论文式 3.17d).

    替换规则:
      ρ → ρ + 0.1
      a_max_accel → 2 × a_max_accel
      a_min_brake_long → 0.75 × a_min_brake_long
      其他参数不变
    """
    return dict(CONSTRUCTION_ZONE_PARAMS)


def check_construction_zone_adaptive(
    is_construction_zone: bool,
    d_long: float,
    v_A: float,
    v_B: float,
    d_lat: float = 0.0,
    v_lat_A: float = 0.0,
    v_lat_B: float = 0.0,
) -> Tuple[bool, float, str, float, float]:
    """施工路段参数自适应违规判定 (论文式 3.17d).

    当车辆进入施工路段 (is_construction_zone=True), 使用 CONSTRUCTION_ZONE_PARAMS
    替代 DEFAULT_RSS_PARAMS 重新计算 RSS 安全距离.

    Args:
        is_construction_zone: 是否在施工路段 (由 RoadElementEntity.lane_type 决定)
        d_long, d_lat: A/B 之间纵/横向距离
        v_A, v_B: A/B 速度
        v_lat_A, v_lat_B: A/B 横向速度

    Returns:
        (is_violation, severity, reason, d_min_long_cz, d_min_lat_cz)
    """
    if not is_construction_zone:
        return False, 0.0, "not_in_cz", 0.0, 0.0

    cz_params = get_construction_zone_params()

    # 重新计算 RSS 安全距离
    _, _, d_min_long_cz = check_safe_distance_longitudinal(d_long, v_A, v_B, cz_params)
    _, _, d_min_lat_cz = check_lateral_dangerous_state(d_lat, v_lat_A, v_lat_B, cz_params)

    is_viol_long = d_long < d_min_long_cz
    is_viol_lat = d_lat < d_min_lat_cz
    is_viol = is_viol_long or is_viol_lat

    if not is_viol:
        return False, 0.0, "cz_safe", round(d_min_long_cz, 2), round(d_min_lat_cz, 2)

    sev = 0.0
    if is_viol_long and d_min_long_cz > 1e-3:
        sev = max(sev, min(1.0, (d_min_long_cz - d_long) / max(d_min_long_cz, 1e-3)))
    if is_viol_lat and d_min_lat_cz > 1e-3:
        sev = max(sev, min(1.0, (d_min_lat_cz - d_lat) / max(d_min_lat_cz, 1e-3)))

    reason = "cz_long_violation" if is_viol_long else "cz_lat_violation"
    return True, round(sev, 3), reason, round(d_min_long_cz, 2), round(d_min_lat_cz, 2)


# ============================================================
# 5. 扩充规则一键运行 (与 run_rss_check 对偶)
# ============================================================

def run_rss_extended_check(
    # 帧实体状态
    d_long: float,
    d_lat: float,
    v_A: float,
    v_B: float,
    v_lat_A: float,
    v_lat_B: float,
    brake_values: List[float],
    speed_values: Optional[List[float]] = None,
    # 扩充场景输入
    is_changing_lane: bool = False,
    ahead_of: bool = True,
    is_exit_lane_buffer: bool = False,
    d_long_new_front: Optional[float] = None,
    v_new_front: Optional[float] = None,
    is_construction_zone: bool = False,
    # 参数
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """对一对车辆 (A, B) 运行扩充 RSS 检查.

    与 model.py:run_rss_check 互补 — 仅返回扩充规则判定.

    Args:
        d_long, d_lat, v_A, v_B, v_lat_A, v_lat_B, brake_values: 与 run_rss_check 一致
        speed_values: 速度历史 (用于 NPR 增强)
        is_changing_lane: B 是否正在变道
        ahead_of: B 是否在 A 前方
        is_exit_lane_buffer: 是否处于 cut-out 缓冲窗口
        d_long_new_front, v_new_front: 新前车 C 的距离/速度 (用于 cut-out)
        is_construction_zone: 是否在施工路段

    Returns:
        dict with keys:
          is_cutin_violation, cutin_severity, d_min_cutin,
          is_cutout_violation, cutout_severity, d_min_long_new,
          is_npr_enhanced, npr_enh_severity, npr_enh_reason,
          is_cz_adapt_violation, cz_adapt_severity, cz_adapt_reason,
          cz_d_min_long, cz_d_min_lat
    """
    # 1. Cut-in
    cutin_viol, cutin_sev, d_min_cutin = check_cutin_violation(
        d_long=d_long, v_A=v_A, v_B=v_B,
        is_changing_lane=is_changing_lane,
        ahead_of=ahead_of,
        params=params,
    )

    # 2. Cut-out (仅在缓冲窗口内 + 提供新前车数据时判定)
    if is_exit_lane_buffer and d_long_new_front is not None and v_new_front is not None:
        cutout_viol, cutout_sev, d_min_long_new = check_cutout_violation(
            d_long_new=d_long_new_front,
            v_A=v_A, v_C=v_new_front,
            exit_lane_observed=True,
            params=params,
        )
    else:
        cutout_viol, cutout_sev, d_min_long_new = False, 0.0, 0.0

    # 3. NPR 增强
    npr_viol, npr_sev, npr_reason = check_no_proper_response_enhanced(
        brake_values=brake_values,
        speed_values=speed_values,
    )

    # 4. Construction zone adaptive
    cz_viol, cz_sev, cz_reason, cz_d_long, cz_d_lat = check_construction_zone_adaptive(
        is_construction_zone=is_construction_zone,
        d_long=d_long, d_lat=d_lat,
        v_A=v_A, v_B=v_B,
        v_lat_A=v_lat_A, v_lat_B=v_lat_B,
    )

    return {
        # Cut-in
        "is_cutin_violation": cutin_viol,
        "cutin_severity": cutin_sev,
        "d_min_cutin": d_min_cutin,
        # Cut-out
        "is_cutout_violation": cutout_viol,
        "cutout_severity": cutout_sev,
        "d_min_long_new": d_min_long_new,
        # NPR enhanced
        "is_npr_enhanced": npr_viol,
        "npr_enh_severity": npr_sev,
        "npr_enh_reason": npr_reason,
        # Construction zone adaptive
        "is_cz_adapt_violation": cz_viol,
        "cz_adapt_severity": cz_sev,
        "cz_adapt_reason": cz_reason,
        "cz_d_min_long": cz_d_long,
        "cz_d_min_lat": cz_d_lat,
    }
