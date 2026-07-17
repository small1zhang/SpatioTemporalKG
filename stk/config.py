#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置加载：从 config/ 目录加载 YAML 配置文件 + 阈值配置."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(name: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """加载 config/ 下的 YAML 配置文件.

    Args:
        name: 配置文件名 (如 'ontology.yaml')
        base_dir: 项目根目录, 默认为当前文件向上找 2 级

    Returns:
        解析后的配置字典
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ThresholdConfig:
    """可配置的阈值集合, 用于 spatial / behavior / rules 层.

    使用 dataclass, 所有字段类型均带默认值, 可直接跨地图复用。
    如需运行时调整, 创建实例后修改字段即可。

    分组说明:
      - lane:       车道宽度、车道匹配距离 (spatial.py)
      - vehicle:    跟驰、变道、静止 (detectors.py)
      - pedestrian: 行人检测范围 (detectors.py, spatial.py)
      - junction:   路口检测范围 (detectors.py)
      - occlusion:  遮挡判定 (detectors.py)
      - traffic_rule: R1-R18 硬编码阈值 (rules.py)
      - rss:        RSS 安全模型阈值
    """

    # ========== 车道 (spatial.py) ==========
    max_lane_match_distance: float = 10.0  # 离线模式最近车道匹配距离 [m]
    lane_width_max: float = 3.5  # 同车道横向距离上界 [m]
    beside_lateral_max: float = 3.0  # 并排横向距离上界 [m]
    beside_longitudinal_max: float = 5.0  # 并排纵向距离上界 [m]
    nearby_pedestrian_max: float = 20.0  # 行人最近检测距离 [m]

    # ========== 车辆行为 (behavior/detectors.py) ==========
    ttc_critical: float = 3.0  # 跟驰 TTC 高风险阈值 [s]
    following_max_distance: float = 100.0  # 跟驰最大检测距离 [m]
    standing_speed_threshold: float = 0.5  # 静止速度阈值 [m/s]
    lane_change_lateral_speed: float = 0.3  # 变道横向速度阈值 [m/s]

    # ========== 行人 (detectors.py, spatial.py) ==========
    pedestrian_activation_distance: float = 50.0  # 行人检测激活距离 [m]
    pedestrian_approach_speed_ceiling: float = 1.5  # 接近行人时自车速度上限 [m/s]

    # ========== 路口 (detectors.py) ==========
    junction_activation_distance: float = 30.0  # 路口激活距离 [m]

    # ========== 遮挡 (detectors.py) ==========
    occlusion_longitudinal_max: float = 30.0  # 遮挡前后距离上限 [m]
    occlusion_lateral_max: float = 3.0  # 遮挡横向距离上限 [m]

    # ========== 交规 (rules/traffic/rules.py) ==========
    opposite_lane_violation_distance: float = 10.0  # 对向车道逆行违规距离 [m]
    bad_weather_distance: float = 20.0  # 恶劣天气下危险距离 [m]
    school_zone_speed_limit: float = 8.3  # ~30 km/h 学区限速 [m/s]
    no_stop_speed_threshold: float = 0.3  # 禁止停区内速度判定阈值 [m/s]
    no_stop_duration_frames: int = 30  # 禁止停区内持续帧数 [帧]
    intersection_stop_box_distance: float = 5.0  # 路口停止线盒 [m]

    # ========== RSS 安全模型 (rules/rss/model.py) ==========
    brake_threshold: float = 0.3  # 刹车加速度阈值 [m/s^2]

    @classmethod
    def default(cls) -> ThresholdConfig:
        return cls()

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def update_from(self, d: dict) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> ThresholdConfig:
        cfg = cls()
        cfg.update_from(d)
        return cfg
