# -*- coding: utf-8 -*-
"""FE-10: EdgePruner 边稀疏化单元测试."""
from __future__ import annotations

import pytest

from stk.config import EgoCentricConfig
from stk.filter.edge_pruner import EdgePruner


def _edge(src, dst, rtype, **extra):
    e = {"src_id": src, "dst_id": dst, "type": rtype}
    e.update(extra)
    return e


class TestBridgeTypesAlwaysKept:
    """桥接/时序边一律保留, 不论 endpoints."""

    @pytest.mark.parametrize("rtype", [
        "actor", "dst", "has_maneuver", "has_interaction",
        "defined_by", "supported_by", "violates", "responsible_for",
        "manifests_as", "next_frame", "has_environment", "weather_context",
    ])
    def test_bridge_type_kept_without_ego(self, rtype):
        pruner = EdgePruner()
        kept = pruner.prune_edges([_edge("v1", "v2", rtype)])
        assert len(kept) == 1

    def test_bridge_type_kept_when_endpoints_low_score(self):
        pruner = EdgePruner(EgoCentricConfig.default(), importance_threshold=0.5)
        kept = pruner.prune_edges(
            [_edge("v_low1", "v_low2", "actor")],
            importance_scores={"v_low1": 0.1, "v_low2": 0.2},
        )
        assert len(kept) == 1


class TestEgoParticipatedEdges:
    def test_ego_src_kept(self):
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("ego", "v1", "following")],
            ego_id="ego",
        )
        assert len(kept) == 1

    def test_ego_dst_kept(self):
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v1", "ego", "following")],
            ego_id="ego",
        )
        assert len(kept) == 1

    def test_non_ego_outside_roi_dropped_spatial(self):
        """非 ego, ROI 外的两车之间 spatia 边 → drop."""
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v1", "v2", "ahead_of")],
            ego_id="ego", roi_ids=set(),
        )
        assert len(kept) == 0


class TestROIInterior:
    def test_two_roi_vehicles_kept(self):
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v1", "v2", "beside")],
            ego_id="ego", roi_ids={"v1", "v2"},
        )
        assert len(kept) == 1

    def test_one_in_roi_one_out_dropped_spatial(self):
        """spatia 边: 一端在 ROI, 一端在外 → drop (非 ego)."""
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v_in", "v_out", "beside")],
            ego_id="ego", roi_ids={"v_in"},
        )
        assert len(kept) == 0


class TestImportanceScores:
    def test_low_score_endpoint_dropped(self):
        """非桥接 spatia 边, 两端点 score 都低于 threshold → drop."""
        pruner = EdgePruner(EgoCentricConfig.default(), importance_threshold=0.5)
        kept = pruner.prune_edges(
            [_edge("v1", "v2", "following")],
            ego_id="ego", roi_ids=set(),
            importance_scores={"v1": 0.1, "v2": 0.2},
        )
        assert len(kept) == 0

    def test_one_high_score_endpoint_kept(self):
        pruner = EdgePruner(EgoCentricConfig.default(), importance_threshold=0.5)
        kept = pruner.prune_edges(
            [_edge("v_high", "v_low", "following")],
            ego_id="ego", roi_ids=set(),
            importance_scores={"v_high": 0.9, "v_low": 0.1},
        )
        assert len(kept) == 1

    def test_no_scores_kept(self):
        """无 scores 信息时, 非桥接 spatia 边按 ROI 检查."""
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v1", "v2", "beside")],
            ego_id="ego", roi_ids=set(), importance_scores=None,
        )
        assert len(kept) == 0


class TestEdgeCases:
    def test_empty_input(self):
        pruner = EdgePruner()
        assert pruner.prune_edges([]) == []

    def test_unknown_type_kept_when_unscored(self):
        """未知边类型, endpoints 无 scores → 保留 (保守)."""
        pruner = EdgePruner()
        kept = pruner.prune_edges([_edge("v1", "v2", "weird_type")])
        assert len(kept) == 1

    def test_in_lane_ego_kept(self):
        """in_lane ego×lane → 保留 (ego 参与)."""
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("ego", "lane1", "in_lane")],
            ego_id="ego",
        )
        assert len(kept) == 1

    def test_in_lane_non_ego_outside_roi_dropped(self):
        """in_lane v1×lane1, v1 在 ROI 外 → drop."""
        pruner = EdgePruner()
        kept = pruner.prune_edges(
            [_edge("v_far", "lane1", "in_lane")],
            ego_id="ego", roi_ids=set(),  # v_far 不在 ROI
        )
        assert len(kept) == 0


class TestThreshold:
    def test_threshold_property(self):
        pruner = EdgePruner(EgoCentricConfig.default(), importance_threshold=0.5)
        assert pruner.threshold == 0.5

    def test_default_threshold_uses_config(self):
        pruner = EdgePruner(EgoCentricConfig.default())
        assert pruner.threshold == pytest.approx(0.30)
