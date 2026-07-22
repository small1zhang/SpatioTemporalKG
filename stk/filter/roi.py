# -*- coding: utf-8 -*-
"""ROI 椭圆判定: 以自车为中心的前后差异化笛卡尔椭圆 (阶段1)."""
from __future__ import annotations

import math
from typing import Any, Dict


def in_ego_ellipse(
    ego_x: float,
    ego_y: float,
    ego_heading_rad: float,
    tgt_x: float,
    tgt_y: float,
    radius_front: float = 70.0,
    radius_rear: float = 30.0,
    radius_side: float = 50.0,
) -> bool:
    """笛卡尔椭圆判定: 纵向按前后取不同 R, 横向统一 R.

    将目标 (tgt_x, tgt_y) 投影到自车 (ego) 车体坐标系,
    按椭圆方程 (lon/R_long)² + (lat/R_lat)² ≤ 1 判断.

    Args:
        ego_x, ego_y:         自车全局坐标 (m)
        ego_heading_rad:      自车航向角 (rad)
        tgt_x, tgt_y:         目标的全局坐标 (m)
        radius_front:         前方半径 (m, 默认 70)
        radius_rear:          后方半径 (m, 默认 30)
        radius_side:          侧向半径 (m, 默认 50)

    Returns:
        True 表示目标在自车 ROI 椭圆内.
    """
    dx = tgt_x - ego_x
    dy = tgt_y - ego_y
    cos_h = math.cos(ego_heading_rad)
    sin_h = math.sin(ego_heading_rad)

    # 投影到车体坐标系: longitudinal 沿车头方向, lateral 沿车侧方向
    longitudinal = dx * cos_h + dy * sin_h
    lateral = -dx * sin_h + dy * cos_h

    R_long = radius_front if longitudinal >= 0 else radius_rear
    # 防止除零
    if R_long <= 0 or radius_side <= 0:
        return False
    return (longitudinal / R_long) ** 2 + (lateral / radius_side) ** 2 <= 1.0
