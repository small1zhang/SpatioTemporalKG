# -*- coding: utf-8 -*-
"""EgoCentricFilter: 以自车为中心的 ROI 过滤生成器 (阶段2)."""
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

    roi_pedestrians: List[Dict[str, Any]] = field(default_factory=list)
    """ROI 椭圆内的行人列表 (阶段2, 预留)."""

    dropped_pedestrians: List[Dict[str, Any]] = field(default_factory=list)
    """被 ROI 椭圆过滤掉的行人列表 (阶段2, 预留)."""

    frame_id: int = 0
    """对应帧 id."""


class EgoCentricFilter:
    """以自车为中心的 ROI 过滤生成器.

    负责从一帧的 vehicles 列表中识别自车,
    并对其他车辆 (按 vehicle_category 差异化半径) 做笛卡尔椭圆 ROI 判定.
    行人作为一个独立集合处理 (按 pedestrian 半径).
    """

    def __init__(self, cfg: Optional[EgoCentricConfig] = None):
        self.cfg = cfg or EgoCentricConfig.default()

    def select(
        self,
        vehicles: List[Dict[str, Any]],
        pedestrians: Optional[List[Dict[str, Any]]] = None,
        frame_id: int = 0,
    ) -> EgoRoiDecision:
        """对一帧 entities 执行 ROI 过滤.

        Args:
            vehicles:   当前帧的车辆字典列表 (必选).
            pedestrians: 当前帧的行人字典列表 (可选, 预留).
            frame_id:   帧 id.

        Returns:
            EgoRoiDecision.
        """
        # ── 处理 vehicles ──
        if not vehicles:
            return EgoRoiDecision(ego=None, roi_targets=[], dropped=[],
                                  frame_id=frame_id)

        ego = self._pick_ego(vehicles)
        if ego is None:
            return EgoRoiDecision(ego=None,
                                  roi_targets=[],
                                  dropped=list(vehicles) + (pedestrians or []),
                                  frame_id=frame_id)

        ego_id = ego.get("entity_id", "")
        roi: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []

        for v in vehicles:
            if v.get("entity_id", "") == ego_id:
                continue
            rf, rr, rs = self.cfg._radii_for(v)
            if in_ego_ellipse(
                ego_x=ego.get("location_x", 0.0),
                ego_y=ego.get("location_y", 0.0),
                ego_heading_rad=ego.get("heading_rad", 0.0),
                tgt_x=v.get("location_x", 0.0),
                tgt_y=v.get("location_y", 0.0),
                radius_front=rf,
                radius_rear=rr,
                radius_side=rs,
            ):
                roi.append(v)
            else:
                dropped.append(v)

        # ── 处理 pedestrians (预留) ──
        roi_peds: List[Dict[str, Any]] = []
        dropped_peds: List[Dict[str, Any]] = []
        if pedestrians:
            for p in pedestrians:
                rf, rr, rs = self.cfg._radii_for(p)
                if in_ego_ellipse(
                    ego_x=ego.get("location_x", 0.0),
                    ego_y=ego.get("location_y", 0.0),
                    ego_heading_rad=ego.get("heading_rad", 0.0),
                    tgt_x=p.get("location_x", 0.0),
                    tgt_y=p.get("location_y", 0.0),
                    radius_front=rf,
                    radius_rear=rr,
                    radius_side=rs,
                ):
                    roi_peds.append(p)
                else:
                    dropped_peds.append(p)

        return EgoRoiDecision(
            ego=ego, roi_targets=roi, dropped=dropped,
            roi_pedestrians=roi_peds, dropped_pedestrians=dropped_peds,
            frame_id=frame_id,
        )

    def _pick_ego(self, vehicles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从车辆列表中识别自车.

        优先级:
          1. 显式 self.cfg.ego_id_opt (若不为空且匹配)
          2. is_ego==True
          3. vehicles[0] (fallback, 向后兼容)
        """
        if not vehicles:
            return None

        if self.cfg.ego_id_opt:
            for v in vehicles:
                if v.get("entity_id", "") == self.cfg.ego_id_opt:
                    return v

        for v in vehicles:
            if v.get("is_ego", False):
                return v

        return vehicles[0]
