# -*- coding: utf-8 -*-
"""ImportanceScorer: 实体重要性打分 E1-E5 (阶段 3 §4.6).

五条规则各产出 [0,1] 子分:
  E1 ego:        ego 自身 → 1.0, 其他 → 0
  E2 distance:   距 ego 越近分越高, 用 ROI radius_front 归一
  E3 visibility: 与 ego 是否存在 spatia/behavior 关系, 每条 +0.25 (上限 1.0)
  E4 anomaly:    是否为 anomaly_target (build_anomaly_dataset 标的) → 1.0
  E5 category:  类型静态偏好 (车 0.6 / 行人 0.5 / 灯 0.4 / lane 0.2 / 其它 0.1)

总分 = ∑ weight_i * score_i,  >importance_threshold 保留.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from stk.config import EgoCentricConfig


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ego": 0.40,
    "distance": 0.20,
    "visibility": 0.15,
    "interaction": 0.15,
    "anomaly": 0.10,
}

# 类型静态偏好 (E5)
_TYPE_PRIOR: Dict[str, float] = {
    "Vehicle": 0.6,
    "Pedestrian": 0.5,
    "TrafficLight": 0.4,
    "RoadElement": 0.2,    # lane / road
    "Maneuver": 0.7,
    "InteractionEvent": 0.7,
    "SafetyViolation": 1.0,
}


class ImportanceScorer:
    """E1-E5 加权重要性打分.

    用法::

        scorer = ImportanceScorer(ego_cfg)
        scores = scorer.score_frame(snap, ego_id="ego",
                                    scene_rels=[...],
                                    anomaly_ids={"v1"})
        # scores: {entity_id: float ∈ [0,1]}
        keep = {eid for eid, s in scores.items() if s >= scorer.threshold}
    """

    def __init__(
        self,
        ego_cfg: Optional[EgoCentricConfig] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ):
        self._ego_cfg = ego_cfg or EgoCentricConfig.default()
        self._weights = dict(weights) if weights is not None \
                        else dict(self._ego_cfg.importance_weights or _DEFAULT_WEIGHTS)
        # 归一化权重 (总和为 1)
        s = sum(self._weights.values()) or 1.0
        self._weights = {k: v / s for k, v in self._weights.items()}
        self._threshold = (
            threshold if threshold is not None
            else self._ego_cfg.importance_threshold
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    # ---------------- 公共 API ----------------

    def score_frame(
        self,
        snap: Dict[str, Any],
        ego_id: Optional[str] = None,
        scene_rels: Optional[List[Any]] = None,
        anomaly_ids: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """对单帧所有实体打分.

        Args:
            snap: 帧 dict, 含 vehicles / pedestrians / traffic_lights / lanes.
            ego_id: 显式 ego id (若空则自动从 is_ego 或 vehicles[0] 识别).
            scene_rels: 当帧场景层关系列表 (用于 E3 visibility 推断).
            anomaly_ids: 当帧异常目标的 entity_id 集合 (build_anomaly_dataset 标签).

        Returns:
            {entity_id: score ∈ [0,1]}  (含 ego 自身).
        """
        vehicles = snap.get("vehicles", [])
        pedestrians = snap.get("pedestrians", [])
        traffic_lights = snap.get("traffic_lights", [])
        lanes = snap.get("lanes", [])

        # 1. 解析 ego
        ego = self._resolve_ego(vehicles, ego_id)
        ego_eid = ego.get("entity_id") if ego else None

        # 2. 索引 ego 参与的对子 (E3 visibility)
        vis_partners: Set[str] = set()
        if scene_rels and ego_eid:
            for r in scene_rels:
                s = self._rel_field(r, "src_id")
                d = self._rel_field(r, "dst_id")
                if s == ego_eid and d:
                    vis_partners.add(d)
                elif d == ego_eid and s:
                    vis_partners.add(s)

        # 3. 逐实体打分
        scores: Dict[str, float] = {}
        anomaly_ids = anomaly_ids or set()

        for v in vehicles:
            eid = str(v.get("entity_id") or v.get("id", ""))
            if not eid:
                continue
            scores[eid] = self._score_entity(
                v, ego, ego_eid, vis_partners, anomaly_ids)

        for p in pedestrians:
            eid = str(p.get("entity_id") or p.get("id", ""))
            if not eid:
                continue
            scores[eid] = self._score_entity(
                p, ego, ego_eid, vis_partners, anomaly_ids)

        for tl in traffic_lights:
            eid = str(tl.get("entity_id") or tl.get("id", ""))
            if not eid:
                continue
            scores[eid] = self._score_entity(
                tl, ego, ego_eid, vis_partners, anomaly_ids)

        for ln in lanes:
            eid = str(ln.get("entity_id") or ln.get("id", ""))
            if not eid:
                continue
            scores[eid] = self._score_entity(
                ln, ego, ego_eid, vis_partners, anomaly_ids)

        return scores

    # ---------------- 内部 ----------------

    @staticmethod
    def _rel_field(r: Any, name: str) -> str:
        if isinstance(r, dict):
            return str(r.get(name, ""))
        return str(getattr(r, name, ""))

    def _resolve_ego(
        self, vehicles: List[Dict], ego_id: Optional[str]
    ) -> Optional[Dict]:
        if ego_id:
            for v in vehicles:
                if str(v.get("entity_id") or v.get("id", "")) == ego_id:
                    return v
        # 自动识别 is_ego
        for v in vehicles:
            if v.get("is_ego"):
                return v
        # fallback: 第一个车
        return vehicles[0] if vehicles else None

    def _score_entity(
        self,
        entity: Dict[str, Any],
        ego: Optional[Dict],
        ego_eid: Optional[str],
        vis_partners: Set[str],
        anomaly_ids: Set[str],
    ) -> float:
        eid = str(entity.get("entity_id") or entity.get("id", ""))

        # E1: ego 自身 → 1.0
        s1 = 1.0 if ego is not None and eid == ego_eid else 0.0

        # E2: 距离分数 (距 ego 越近越高)
        s2 = self._score_distance(entity, ego)

        # E3: 是否与 ego 有 spatia/behavior 关系
        # visibility 中, ego 参与的对子得 1.0
        s3 = 1.0 if eid in vis_partners else 0.0

        # E4: anomaly_target
        s4 = 1.0 if eid in anomaly_ids else 0.0

        # E5: 类型静态偏好
        etype = entity.get("entity_type", "Vehicle")
        s5 = _TYPE_PRIOR.get(etype, 0.1)

        total = (
            self._weights.get("ego", 0.0) * s1
            + self._weights.get("distance", 0.0) * s2
            + self._weights.get("visibility", 0.0) * s3
            + self._weights.get("interaction", 0.0) * s3  # 与 vis 共用, 见下注
            + self._weights.get("anomaly", 0.0) * s4
            + 0.05 * s5  # 类型偏好当作 tiebreaker, 不入主权重
        )
        # 注: E3 visibility 与 E4 interaction 在当前实现共用 vis_partners 集合,
        # 若需独立区分 spatia vs behavior 重要性, 可后续扩展接口分别传入两套 set.
        return max(0.0, min(1.0, total))

    def _score_distance(
        self, entity: Dict, ego: Optional[Dict]
    ) -> float:
        if ego is None:
            return 0.0
        ex = entity.get("location_x")
        ey = entity.get("location_y")
        gx = ego.get("location_x")
        gy = ego.get("location_y")
        if ex is None or ey is None or gx is None or gy is None:
            return 0.0
        dist = math.sqrt((ex - gx) ** 2 + (ey - gy) ** 2)
        # 用 ego 的 ROI radius_front 归一化
        rf, _, rs = self._ego_cfg._radii_for(entity)
        # 取 front 与 side 中较大者作为归一化基准 (保证 ROI 内可达 1.0)
        max_r = max(rf, rs)
        if max_r <= 0:
            return 0.0
        # dist=0 → 1.0, dist=max_r → 0.5, dist=2*max_r → 0
        # 线性衰减: clip(1 - dist / (2*max_r), 0, 1)
        return max(0.0, min(1.0, 1.0 - dist / (2.0 * max_r)))

    # ---------------- Checkpoint ----------------

    def to_dict(self) -> dict:
        return {
            "weights": dict(self._weights),
            "threshold": self._threshold,
        }

    @classmethod
    def from_dict(
        cls, data: dict, ego_cfg: Optional[EgoCentricConfig] = None
    ) -> "ImportanceScorer":
        return cls(
            ego_cfg=ego_cfg,
            weights=data.get("weights"),
            threshold=data.get("threshold"),
        )
