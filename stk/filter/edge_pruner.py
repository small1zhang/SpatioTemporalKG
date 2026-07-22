# -*- coding: utf-8 -*-
"""EdgePruner: 边稀疏化策略 (阶段 3 §4.6.2).

输入: 边列表 (dict with src_id, dst_id, type/relation_type)
输出: 过滤后的边列表

规则:
  - 桥接/时序边一律保留:
    actor/dst/has_maneuver/has_interaction/defined_by/supported_by/
    violates/responsible_for/manifests_as/next_frame/has_environment/
    weather_context/containsXXX (FE-11 处理 background 才 drop)
  - ego 参与的 spatia/behavior 边: 保留
  - ROI 内两车间 spatia/behavior 边: 保留
  - ROI 外对子间的 spatia/behavior 边: 丢弃
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

from stk.config import EgoCentricConfig


# 桥接/时序边类型, 一律保留
_BRIDGE_TYPES: Set[str] = {
    "actor", "dst",
    "has_maneuver", "has_interaction",
    "defined_by", "supported_by",
    "violates", "responsible_for",
    "manifests_as",
    "next_frame", "has_environment", "weather_context",
}

# spatia/behavior 主体边类型, 需做 ego/ROI 检查
_SpatialBehavior_TYPES: Set[str] = {
    "in_lane", "ahead_of", "beside", "nearby_pedestrian",
    "in_junction", "adjacent_lane", "lane_connects", "on_road",
    # 行为层
    "following", "approaching", "yielding_to", "overtaking",
    "wrong_side_meeting", "opposite_direction", "same_direction",
    "blocked_view", "approaching_pedestrian",
    "approaching_intersection", "crossing",
    "standing_still", "changing_lane",
}


class EdgePruner:
    """边稀疏化.

    用法::

        pruner = EdgePruner(ego_cfg)
        kept = pruner.prune_edges(edges, ego_id="ego",
                                  roi_ids={"v1","v2"},
                                  importance_scores={"ego":1.0,"v1":0.6})
    """

    def __init__(
        self,
        ego_cfg: Optional[EgoCentricConfig] = None,
        importance_threshold: Optional[float] = None,
    ):
        self._ego_cfg = ego_cfg or EgoCentricConfig.default()
        self._threshold = (
            importance_threshold if importance_threshold is not None
            else self._ego_cfg.importance_threshold
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    def prune_edges(
        self,
        edges: Iterable[Dict],
        ego_id: Optional[str] = None,
        roi_ids: Optional[Set[str]] = None,
        importance_scores: Optional[Dict[str, float]] = None,
    ) -> list:
        """过滤边列表.

        Args:
            edges: 边 dict 列表, 每条含 src_id/dst_id/type 或 relation_type.
            ego_id: 当前 ego entity_id.
            roi_ids: 当前帧 ROI 内实体 id 集合.
            importance_scores: {eid: score} 当实体的 score < threshold 时,
                              该实体参与的非桥接边一律 drop.

        Returns:
            过滤后的边 list (新 list, 输入不变).
        """
        roi_ids = roi_ids or set()
        scores = importance_scores or {}
        kept = []
        for e in edges:
            rtype = e.get("type") or e.get("relation_type") or ""
            src = str(e.get("src_id", ""))
            dst = str(e.get("dst_id", ""))
            if self._keep_edge(rtype, src, dst, ego_id, roi_ids, scores):
                kept.append(e)
        return kept

    def _keep_edge(
        self,
        rtype: str,
        src: str,
        dst: str,
        ego_id: Optional[str],
        roi_ids: Set[str],
        scores: Dict[str, float],
    ) -> bool:
        # 1. 桥接/时序边一律保留
        if rtype in _BRIDGE_TYPES:
            return True
        # 2. ego 参与的边: 保留
        if ego_id and (src == ego_id or dst == ego_id):
            return True
        # 3. ROI 内两车间: 保留
        if src in roi_ids and dst in roi_ids:
            return True
        # 4. 双端点都已打过分的 → 任一达标即保留 (在 ROI 检查之前)
        s_score = scores.get(src)
        d_score = scores.get(dst)
        if s_score is not None and d_score is not None:
            return s_score >= self._threshold or d_score >= self._threshold
        # 5. 未打分场景: 已知 spatia/behavior 类型 → drop
        if rtype in _SpatialBehavior_TYPES:
            return False
        # 6. 未知类型 endpoints 无 scores → 保守保留
        return True
