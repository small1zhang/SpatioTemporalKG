# -*- coding: utf-8 -*-
"""FE-9: ImportanceScorer E1-E5 打分单元测试."""
from __future__ import annotations

import pytest

from stk.config import EgoCentricConfig
from stk.filter.importance import ImportanceScorer


def _veh(vid, x=0.0, y=0.0, is_ego=False, etype="Vehicle",
         vehicle_category="car"):
    return {
        "entity_id": vid,
        "entity_type": etype,
        "vehicle_category": vehicle_category,
        "location_x": x, "location_y": y, "location_z": 0.0,
        "is_ego": is_ego,
    }


def _ped(pid, x=0.0, y=0.0):
    return {
        "entity_id": pid,
        "entity_type": "Pedestrian",
        "location_x": x, "location_y": y,
    }


class TestE1Ego:
    def test_ego_self_score_one(self):
        ego = _veh("ego", is_ego=True)
        cfg = EgoCentricConfig.default()
        scorer = ImportanceScorer(cfg)
        scores = scorer.score_frame({"vehicles": [ego]}, ego_id="ego")
        # ego 自身: E1=1.0(0.4w) + E2=distance(0.2w)*1.0 + E5 tiebreaker=0.05*0.6
        # = 0.4 + 0.2 + 0.03 = 0.63
        assert scores["ego"] == pytest.approx(0.63, abs=1e-4)

    def test_non_ego_zero_e1(self):
        ego = _veh("ego", is_ego=True)
        other = _veh("v2", x=10, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": [ego, other]}, ego_id="ego")
        # v2 不是 ego, 主要靠 distance/visibility 等项打分; 此处 v2 处于 ROI 内
        # 应有 pos score 但应小于 ego
        assert scores["v2"] < scores["ego"]
        assert 0.0 <= scores["v2"] <= 1.0


class TestE2Distance:
    def test_distance_close_high(self):
        ego = _veh("ego", is_ego=True)
        close = _veh("close", x=10, y=0)
        far = _veh("far", x=200, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": [ego, close, far]}, ego_id="ego")
        assert scores["close"] > scores["far"]
        # far 距 200m 超过 2*radius_front=140m → distance 分 0
        assert scores["far"] < 0.1   # 仅类型偏好 0.05*0.6 贡献

    def test_distance_decay_monotonic(self):
        ego = _veh("ego", is_ego=True)
        vs = [ego] + [_veh(f"v{i}", x=10 * i, y=0) for i in range(1, 12)]
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": vs}, ego_id="ego")
        # 单调递减 (大体一致)
        score_seq = [scores[f"v{i}"] for i in range(1, 12)]
        for i in range(len(score_seq) - 1):
            assert score_seq[i] >= score_seq[i + 1] - 1e-6


class TestE3Visibility:
    def test_visibility_partner_boosted(self):
        ego = _veh("ego", is_ego=True)
        partner = _veh("v_partner", x=30, y=0)  # 几乎同样距离
        other = _veh("v_no_rel", x=30, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        # 用 scene_rels 让 v_partner 与 ego 有关系
        scene_rels = [
            {"src_id": "ego", "dst_id": "v_partner", "relation_type": "ahead_of"},
        ]
        scores = scorer.score_frame(
            {"vehicles": [ego, partner, other]},
            ego_id="ego", scene_rels=scene_rels,
        )
        assert scores["v_partner"] > scores["v_no_rel"]

    def test_no_scene_rels_no_visibility_boost(self):
        ego = _veh("ego", is_ego=True)
        v = _veh("v", x=30, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame(
            {"vehicles": [ego, v]}, ego_id="ego", scene_rels=[],
        )
        scores_no_rel = scorer.score_frame(
            {"vehicles": [ego, v]}, ego_id="ego", scene_rels=None,
        )
        assert scores["v"] == pytest.approx(scores_no_rel["v"], abs=1e-6)


class TestE4Anomaly:
    def test_anomaly_target_boosted(self):
        ego = _veh("ego", is_ego=True)
        a = _veh("v_anomaly", x=50, y=0)
        b = _veh("v_normal", x=50, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame(
            {"vehicles": [ego, a, b]}, ego_id="ego",
            anomaly_ids={"v_anomaly"},
        )
        assert scores["v_anomaly"] > scores["v_normal"]

    def test_anomaly_empty_safe(self):
        ego = _veh("ego")
        v = _veh("v1", x=10, y=0)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": [ego, v]}, ego_id="ego")
        # anomaly_ids 默认空集, 不应抛错
        assert 0 <= scores["v1"] <= 1.0


class TestE5TypePrior:
    def test_vehicle_higher_than_lane(self):
        ego = _veh("ego", is_ego=True)
        v = _veh("v", x=20, y=0)
        lane = {"entity_id": "lane1", "entity_type": "RoadElement"}
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame(
            {"vehicles": [ego, v], "lanes": [lane]}, ego_id="ego",
        )
        assert scores["v"] > scores["lane1"]


class TestThreshold:
    def test_threshold_filter_set(self):
        cfg = EgoCentricConfig.default()
        cfg.importance_threshold = 0.5
        scorer = ImportanceScorer(cfg)
        assert scorer.threshold == 0.5

    def test_default_threshold_30(self):
        scorer = ImportanceScorer(EgoCentricConfig.default())
        assert scorer.threshold == pytest.approx(0.30)


class TestFrameMultiEntity:
    def test_score_frame_returns_all_entities(self):
        snap = {
            "vehicles": [_veh("ego", is_ego=True), _veh("v1", x=10)],
            "pedestrians": [_ped("p1", x=5, y=2)],
            "traffic_lights": [
                {"entity_id": "tl1", "entity_type": "TrafficLight",
                 "location_x": 0, "location_y": 30},
            ],
        }
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame(snap, ego_id="ego")
        assert set(scores.keys()) == {"ego", "v1", "p1", "tl1"}
        # ego 应是最高分
        assert scores["ego"] == max(scores.values())

class TestCheckpoint:
    def test_round_trip(self):
        scorer = ImportanceScorer(
            EgoCentricConfig.default(),
            weights={"ego": 0.5, "distance": 0.3, "visibility": 0.0,
                     "interaction": 0.0, "anomaly": 0.2},
            threshold=0.45,
        )
        d = scorer.to_dict()
        scorer2 = ImportanceScorer.from_dict(d)
        assert scorer2.threshold == pytest.approx(0.45)
        # 权重已 normalize 到 1.0
        assert sum(scorer2._weights.values()) == pytest.approx(1.0)


class TestNoEgo:
    def test_empty_frame_safe(self):
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": []}, ego_id=None)
        assert scores == {}

    def test_auto_ego_first_vehicle(self):
        v1 = _veh("v1", x=0)
        v2 = _veh("v2", x=10)
        scorer = ImportanceScorer(EgoCentricConfig.default())
        scores = scorer.score_frame({"vehicles": [v1, v2]}, ego_id=None)
        # 没 is_ego 字段时退化用第一个车作 ego, 得 0.63
        assert scores["v1"] == pytest.approx(0.63, abs=1e-4)
