# -*- coding: utf-8 -*-
"""EgoCentricFilter: 以自车为中心的 ROI 过滤生成器 (阶段1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from stk.config import EgoCentricConfig
from stk.filter.roi import in_ego_ellipse


@dataclass
class EgoRoiDecision:
    """一次 ROI 过滤的结果."""

    ego: Optional[Dict[str, Any]]
    """当前帧识别到的自车实体 (None 表示无车)."""

    roi_targets: List[Dict[str, Any]] = field(default_factory=list)
    """ROI 椭圆内的他车列表 (不含 ego)."""

    dropped: List[Dict[str, Any]] = field(default_factory=list)
    """被 ROI 椭圆过滤掉的他车列表."""

    frame_id: int = 0
    """对应帧 id."""


class EgoCentricFilter:
    """以自车为中心的 ROI 过滤生成器.

    负责从一帧的 vehicles 列表中识别自车,
    并对其他车辆做笛卡尔椭圆 ROI 判定.

    用法::

        cfg = EgoCentricConfig.default()
        ef = EgoCentricFilter(cfg)
        decision = ef.select(vehicles, frame_id)
        for v in decision.roi_targets:
            # 只处理 ROI 内的车辆
    """

    def __init__(self, cfg: Optional[EgoCentricConfig] = None):
        self.cfg = cfg or EgoCentricConfig.default()

    def select(
        self,
        vehicles: List[Dict[str, Any]],
        frame_id: int,
    ) -> EgoRoiDecision:
        """对一帧 vehicles 执行 ROI 过滤.

        Args:
            vehicles: 当前帧的车辆字典列表.
                每个字典应含 entity_id, location_x, location_y, heading_rad, is_ego.
            frame_id: 帧 id.

        Returns:
            EgoRoiDecision.
        """
        if not vehicles:
            return EgoRoiDecision(ego=None, roi_targets=[], dropped=[], frame_id=frame_id)

        ego = self._pick_ego(vehicles)
        if ego is None:
            return EgoRoiDecision(ego=None, roi_targets=[], dropped=list(vehicles), frame_id=frame_id)

        ego_id = ego.get("entity_id", "")
        roi: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []

        for v in vehicles:
            if v.get("entity_id", "") == ego_id:
                continue
            if in_ego_ellipse(
                ego_x=ego.get("location_x", 0.0),
                ego_y=ego.get("location_y", 0.0),
                ego_heading_rad=ego.get("heading_rad", 0.0),
                tgt_x=v.get("location_x", 0.0),
                tgt_y=v.get("location_y", 0.0),
                radius_front=self.cfg.radius_front,
                radius_rear=self.cfg.radius_rear,
                radius_side=self.cfg.radius_side,
            ):
                roi.append(v)
            else:
                dropped.append(v)

        return EgoRoiDecision(ego=ego, roi_targets=roi, dropped=dropped, frame_id=frame_id)

    def _pick_ego(self, vehicles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从车辆列表中识别自车.

        优先级:
          1. 显式 self.cfg.ego_id_opt (若不为空且匹配)
          2. is_ego==True
          3. vehicles[0] (fallback, 向后兼容)
        """
        if not vehicles:
            return None

        # 1) 显式 ego_id_opt
        if self.cfg.ego_id_opt:
            for v in vehicles:
                if v.get("entity_id", "") == self.cfg.ego_id_opt:
                    return v

        # 2) is_ego 字段
        for v in vehicles:
            if v.get("is_ego", False):
                return v

        # 3) fallback: 第一个车辆
        return vehicles[0]
