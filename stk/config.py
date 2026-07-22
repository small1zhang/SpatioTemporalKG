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


@dataclass
class EgoCentricConfig:
    """以自车为中心的 ROI 过滤配置 (v3 §4.5, 阶段2).

    控制自车周围 ROI 椭圆的前/后/侧向半径，支持按类别差异化.

    保留 radius_front/rear/side 作为 fallback (向后兼容阶段1),
    但优先从 radii_by_category[category] 读取对应类别的半径.
    """

    # 显式 ego_id (可选, 为空时自动从 is_ego / vehicles[0] 识别)
    ego_id_opt: Optional[str] = None

    # ── 笛卡尔椭圆半径 (通用 fallback) ──
    radius_front: float = 70.0   # 前方半径 (m)
    radius_rear: float = 30.0    # 后方半径 (m)
    radius_side: float = 50.0    # 侧向半径 (m)

    # ── 按类别差异化半径 (阶段2) ──
    radii_by_category: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "car":          {"front": 70.0, "rear": 30.0, "side": 50.0},
        "bus_or_truck": {"front": 70.0, "rear": 30.0, "side": 50.0},
        "motorcycle":   {"front": 50.0, "rear": 25.0, "side": 40.0},
        "bicycle":      {"front": 50.0, "rear": 25.0, "side": 40.0},
        "emergency":    {"front": 60.0, "rear": 30.0, "side": 50.0},
        "pedestrian":   {"front": 40.0, "rear": 20.0, "side": 40.0},
    })
    pedestrian_radius_front: float = 40.0  # 行人前方半径快捷字段
    pedestrian_radius_rear: float = 20.0   # 行人的后方半径
    pedestrian_radius_side: float = 40.0   # 行人的侧向半径

    # ── 过滤行为 ──
    hysteresis_frames: int = 3   # ENTER/EXIT 滞回帧数
    forget_frames: int = 30      # FORGET 过期帧数 (阶段2, 预留)
    legacy_full_pairing: bool = False
    """True: 退回到老的全对子 O(N²) 枚举 (向后兼容).
       False: 按 Ego x ROI 内他车做 RSS 对子."""

    # ── 行为层/场景层协同过滤开关 (阶段 2/3 启用, 预留) ──
    filter_behavior_detectors: bool = False
    filter_scene_spatial: bool = False

    # ── 实体重要性打分 (阶段 3: E1-E5) ──
    importance_threshold: float = 0.30
    """重要性得分阈值, <threshold 的实体和边在序列化时被剔除. -1 表示不过滤."""
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "ego": 0.40,
        "distance": 0.20,
        "visibility": 0.15,
        "interaction": 0.15,
        "anomaly": 0.10,
    })
    """E1-E5 加权权重, auto-normalize (总和应≈1.0)."""

    # ── 静态背景外移 (阶段 3) ──
    exclude_lanes: bool = True
    """True= lane 节点/边不进 KG, 信息平铺到 VehicleEntity.attrs.lane_id."""
    exclude_road_elements: bool = False
    """True= RoadElement 节点不进 KG (暂缺静态提取器, 默认关)."""

    def _radii_for(self, entity: dict) -> tuple:
        """返回给定 entity 的 (radius_front, radius_rear, radius_side).

        优先级:
          1. radii_by_category 匹配 (按 entity.get("vehicle_category") 或 entity_type)
          2. pedestrian_radius_* (若 entity_type == "Pedestrian")
          3. 通用 radius_front/rear/side (阶段1 fallback)
        """
        if entity.get("entity_type") == "Pedestrian":
            return (self.pedestrian_radius_front,
                    self.pedestrian_radius_rear,
                    self.pedestrian_radius_side)
        cat = entity.get("vehicle_category", "car")
        radii = self.radii_by_category.get(cat)
        if radii:
            return (radii.get("front", self.radius_front),
                    radii.get("rear", self.radius_rear),
                    radii.get("side", self.radius_side))
        return (self.radius_front, self.radius_rear, self.radius_side)

    @classmethod
    def default(cls) -> EgoCentricConfig:
        return cls()

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def update_from(self, d: dict) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> EgoCentricConfig:
        cfg = cls()
        cfg.update_from(d)
        return cfg
