# -*- coding: utf-8 -*-
"""FE-12: serialize_graph 集成 ImportanceScorer/EdgePruner/BackgroundFilter 三道 pass.

集成测试验证:
  - 不传 cfg → 默认行为 (所有节点/边进图)
  - 传 importance_cfg → 低分实体被剔除 (attrs.importance 字段已注入)
  - 传 background_cfg → lane 节点不进图, in_lane 边被裁
  - 传 edge_pruner_cfg → ROI 外 spatia/behavior 边被裁
  - 三道全开 → 端到端可工作
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from stk.config import EgoCentricConfig
from stk.filter.importance import ImportanceScorer
from stk.filter.edge_pruner import EdgePruner
from stk.filter.background_filter import BackgroundFilter
from stk.storage.serializer import serialize_graph


def _snap(vids: List[tuple], lanes=None, scene_rels=None, fid=0):
    """构造一帧快照: vids = [(eid, x, y, is_ego, etype), ...]."""
    vehicles = []
    pedestrians = []
    traffic_lights = []
    for eid, x, y, is_ego, etype in vids:
        e = {
            "entity_id": eid,
            "entity_type": etype,
            "vehicle_category": "car",
            "location_x": x, "location_y": y, "location_z": 0.0,
            "heading_rad": 0.0,
            "is_ego": is_ego,
            "current_lane_id": "road_1_lane_1",
            "lane_id": 1, "road_id": 1,
        }
        if etype == "Vehicle":
            vehicles.append(e)
        elif etype == "Pedestrian":
            pedestrians.append(e)
        elif etype == "TrafficLight":
            traffic_lights.append(e)
    return {
        "frame_id": fid,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "traffic_lights": traffic_lights,
        "lanes": lanes or [],
        "scene_rels": scene_rels or [],
        "weather": {},
    }


class TestBackwardCompatibleNoCfg:
    """不传任何 cfg → 默认行为不变."""

    def test_no_cfg_all_nodes_kept(self):
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v1", 30, 0, False, "Vehicle"),
            ("v_far", 200, 0, False, "Vehicle"),
        ])
        out = serialize_graph(snap)
        ids = {n["id"] for n in out["nodes"]}
        assert "ego" in ids and "v1" in ids and "v_far" in ids

    def test_no_cfg_no_importance_attr(self):
        snap = _snap([("ego", 0, 0, True, "Vehicle")])
        out = serialize_graph(snap)
        for n in out["nodes"]:
            assert "importance" not in n["attrs"]


class TestImportanceScorerIntegration:
    """传 importance_cfg → 低分实体被剔除, attrs.importance 已注入."""

    def test_low_score_entity_dropped(self):
        # ego 在原点, 一台近车 v_near (10m) 加上 spatia 关系提分,
        # 一台远车 v_far (200m) → 应被裁
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v_near", 10, 0, False, "Vehicle"),
            ("v_far", 200, 0, False, "Vehicle"),
        ], scene_rels=[
            {"src_id": "v_near", "dst_id": "ego",
             "relation_type": "ahead_of", "frame_id": 0},
        ])
        cfg = EgoCentricConfig.default()
        cfg.importance_threshold = 0.30
        scorer = ImportanceScorer(cfg)
        out = serialize_graph(snap, importance_cfg=scorer, ego_id="ego")
        ids = {n["id"] for n in out["nodes"]}
        assert "ego" in ids
        assert "v_near" in ids  # 有 spatia 关系提分, 应 >0.30
        assert "v_far" not in ids  # 远车应该被裁

    def test_importance_attr_injected(self):
        snap = _snap([("ego", 0, 0, True, "Vehicle")])
        cfg = EgoCentricConfig.default()
        scorer = ImportanceScorer(cfg)
        out = serialize_graph(snap, importance_cfg=scorer, ego_id="ego")
        for n in out["nodes"]:
            if n["id"] == "ego":
                assert "importance" in n["attrs"]
                assert 0.0 <= n["attrs"]["importance"] <= 1.0


class TestBackgroundFilterIntegration:
    """传 background_cfg → lane 节点不进图."""

    def test_lane_node_filtered_out(self):
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v1", 30, 0, False, "Vehicle"),
        ], lanes=[
            {"entity_id": "road_1_lane_1", "entity_type": "RoadElement",
             "road_id": 1, "lane_id": 1, "junction_id": -1},
        ])
        cfg = EgoCentricConfig.default()
        bg = BackgroundFilter(cfg)
        out = serialize_graph(snap, background_cfg=bg)
        ids = {n["id"] for n in out["nodes"]}
        assert "road_1_lane_1" not in ids

    def test_in_lane_edge_filtered(self):
        """lane 节点不建, in_lane 边也无 dst, 应被裁."""
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
        ], lanes=[])
        cfg = EgoCentricConfig.default()
        bg = BackgroundFilter(cfg)
        out = serialize_graph(snap, background_cfg=bg,
                              with_relations=True)
        edge_types = {e.get("type") or e.get("relation_type")
                      for e in out["edges"] if isinstance(e, dict)}
        # in_lane 边应被裁掉 (lane 节点不存在)
        assert "in_lane" not in edge_types


class TestEdgePrunerIntegration:
    """传 edge_pruner_cfg → ROI 外 spatia/behavior 边被裁."""

    def test_far_pair_edge_dropped(self):
        # 两台远车 v_far1/v_far2 都不在 ego ROI, 它们之间的 ahead_of 边应被裁
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v_far1", 200, 0, False, "Vehicle"),
            ("v_far2", 210, 0, False, "Vehicle"),
        ], scene_rels=[
            {"src_id": "v_far2", "dst_id": "v_far1",
             "relation_type": "ahead_of", "frame_id": 0},
        ])
        cfg = EgoCentricConfig.default()
        cfg.importance_threshold = 0.30
        scorer = ImportanceScorer(cfg)
        pruner = EdgePruner(cfg)
        out = serialize_graph(snap, importance_cfg=scorer,
                              edge_pruner_cfg=pruner, ego_id="ego",
                              with_relations=True)
        edge_types_pairs = {
            (e.get("type") or e.get("relation_type"),
             e.get("src_id"), e.get("dst_id"))
            for e in out["edges"] if isinstance(e, dict)
        }
        # v_far1↔v_far2 之间的 ahead_of 边应被裁
        far_pairs = {(t, s, d) for (t, s, d) in edge_types_pairs
                     if t == "ahead_of" and ("v_far" in s or "v_far" in d)}
        assert len(far_pairs) == 0


class TestFullPipeline:
    """三道全开 → 端到端."""

    def test_three_passes_compatible(self):
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v_near", 15, 0, False, "Vehicle"),
            ("v_far", 200, 0, False, "Vehicle"),
        ], lanes=[
            {"entity_id": "road_1_lane_1", "entity_type": "RoadElement",
             "road_id": 1, "lane_id": 1},
        ], scene_rels=[
            {"src_id": "v_near", "dst_id": "ego",
             "relation_type": "ahead_of", "frame_id": 0},
        ])
        cfg = EgoCentricConfig.default()
        cfg.importance_threshold = 0.30
        scorer = ImportanceScorer(cfg)
        bg = BackgroundFilter(cfg)
        pruner = EdgePruner(cfg)
        out = serialize_graph(
            snap,
            importance_cfg=scorer,
            background_cfg=bg,
            edge_pruner_cfg=pruner,
            ego_id="ego",
            with_relations=True,
        )
        ids = {n["id"] for n in out["nodes"]}
        assert "ego" in ids
        assert "v_near" in ids  # 与 ego 有 spatia 关系, 分够
        assert "v_far" not in ids
        assert "road_1_lane_1" not in ids


class TestAnomalyIntegration:
    """anomaly_ids 字段: anomaly_target 实体被保留 (即便距离远时 boost 应超过 threshold)."""

    def test_anomaly_target_kept(self):
        # 使用低 threshold 确保 anomaly boost 起作用
        snap = _snap([
            ("ego", 0, 0, True, "Vehicle"),
            ("v_far", 100, 0, False, "Vehicle"),
        ])
        cfg = EgoCentricConfig.default()
        cfg.importance_threshold = 0.15
        cfg.importance_weights = {"ego": 0.40, "distance": 0.20,
                                   "visibility": 0.15, "interaction": 0.15,
                                   "anomaly": 0.10}
        scorer = ImportanceScorer(cfg)
        scores = scorer.score_frame(
            snap, ego_id="ego", anomaly_ids={"v_far"},
        )
        assert scores["v_far"] > 0.15, f"anomaly boosted v_far score {scores['v_far']} should exceed threshold"
        out = serialize_graph(
            snap, importance_cfg=scorer, ego_id="ego",
            anomaly_ids={0: {"v_far"}},
        )
        ids = {n["id"] for n in out["nodes"]}
        assert "v_far" in ids
