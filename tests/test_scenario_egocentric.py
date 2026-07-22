# -*- coding: utf-8 -*-
"""FE-7: 场景层 spatial 关系 ego 化集成测试.

compute_ahead_of / compute_beside / compute_nearby_pedestrian 可选 ego_id 参数.
不传 ego_id → 全枚举 (向后兼容, 原行为)
传 ego_id   → 只生成 ego 参与的车辆-车辆对子.
"""
from __future__ import annotations

import math

import pytest

from stk.scenario.spatial import (
    compute_ahead_of, compute_beside, compute_nearby_pedestrian,
)
from stk.scenario.relations import BaseRelation


def _make_vehicle(eid, x, y, heading=0.0):
    imports = {"SimpleNamespace": __import__("types").SimpleNamespace}
    v = __import__("types").SimpleNamespace()
    v.entity_id = eid
    v.attrs = {
        "location_x": x,
        "location_y": y,
        "location_z": 0.0,
        "heading_rad": heading,
    }
    return v


def _make_ped(eid, x, y):
    p = __import__("types").SimpleNamespace()
    p.entity_id = eid
    p.attrs = {
        "location_x": x,
        "location_y": y,
        "location_z": 0.0,
    }
    return p


class TestAheadOfEgo:
    """compute_ahead_of 的 ego 模式 (带 ego_id 参数)."""

    def test_legacy_no_ego_full_pairs(self):
        """不传 ego_id → 全枚举 (原行为)."""
        v1 = _make_vehicle("V1", 0, 0)
        v2 = _make_vehicle("V2", 10, 0)  # V2 在 V1 前方
        v3 = _make_vehicle("V3", 20, 0)  # V3 在 V2 前方
        rels = compute_ahead_of([v1, v2, v3], frame_id=0)
        # 应有 V2→V1 (V2 in front of V1), V3→V2, V3→V1
        pairs = {(r.relation_type, r.src_id, r.dst_id) for r in rels}
        assert ("ahead_of", "V2", "V1") in pairs
        assert ("ahead_of", "V3", "V2") in pairs
        assert ("ahead_of", "V3", "V1") in pairs

    def test_ego_side_only_ego_relations(self):
        """传 ego_id="ego" → 只生成 ego 参与的 ahead_of."""
        vs = [
            _make_vehicle("ego", 0, 0),
            _make_vehicle("V_front", 15, 0),   # ego 前方 15m
            _make_vehicle("V_left", 0, 5),      # ego 侧向 (同性线不带入 ahead_of)
            _make_vehicle("V_far", 50, 0),      # 超远
        ]
        rels = compute_ahead_of(vs, frame_id=0, ego_id="ego")
        # 所有 ahead_of 关系的 src/dst 必须包含 "ego"
        for r in rels:
            assert "ego" in (r.src_id, r.dst_id)
        # V_front 在 ego 前方 (15m > 0, lateral < 3.5)
        ahead_ids = {(r.src_id, r.dst_id) for r in rels
                     if r.relation_type == "ahead_of"}
        assert ("V_front", "ego") in ahead_ids  # V_front ahead_of ego

    def test_ego_no_relations_when_no_vehicle_ahead(self):
        """传 ego_id 但没有车在前方 → 空."""
        vs = [_make_vehicle("ego", 0, 0)]
        rels = compute_ahead_of(vs, frame_id=0, ego_id="ego")
        assert len(rels) == 0


class TestBesideEgo:
    """compute_beside 的 ego 模式."""

    def test_legacy_side_full_pairs(self):
        """不传 ego_id → 全枚举."""
        v1 = _make_vehicle("V1", 0, 0)
        v2 = _make_vehicle("V2", 0, 2)  # 并排
        rels = compute_beside([v1, v2], frame_id=0)
        pairs = {(r.relation_type, r.src_id, r.dst_id) for r in rels}
        assert ("beside", "V1", "V2") in pairs

    def test_ego_side_only_ego_relations(self):
        """传 ego_id → 只生成 ego 参与的 beside."""
        vs = [
            _make_vehicle("ego", 0, 0),
            _make_vehicle("V_side", 0, 2),     # 侧向 2m, 在 beside 阈值内
            _make_vehicle("V_far", 0, 20),      # 侧向 20m → outside.
        ]
        rels = compute_beside(vs, frame_id=0, ego_id="ego")
        for r in rels:
            assert "ego" in (r.src_id, r.dst_id)
        # 应只有 ego × V_side
        beside_ids = {(r.src_id, r.dst_id) for r in rels
                      if r.relation_type == "beside"}
        assert beside_ids.issubset({("ego", "V_side"), ("V_side", "ego")})

    def test_ego_beside_empty(self):
        """无并排车辆 → 空."""
        vs = [_make_vehicle("ego", 0, 0),
              _make_vehicle("V_far", 0, 20)]
        rels = compute_beside(vs, frame_id=0, ego_id="ego")
        assert len(rels) == 0


class TestNearbyPedestrianEgo:
    """compute_nearby_pedestrian 的 ego 模式."""

    def test_legacy_all_vehicles(self):
        """不传 ego_id → 所有车辆计算附近行人."""
        v1 = _make_vehicle("V1", 0, 0)
        v2 = _make_vehicle("V2", 10, 10)
        p = _make_ped("P1", 1, 0)
        rels = compute_nearby_pedestrian([v1, v2], [p], frame_id=0)
        # V1 离 1m < 20 → 应产生 relation
        # V2 离 √(200) ≈ 14m < 20 → 也应产生
        assert len(rels) == 2

    def test_ego_only_ego_pedestrian(self):
        """传 ego_id → 只计算 ego 附近行人."""
        v1 = _make_vehicle("ego", 0, 0)
        v2 = _make_vehicle("V2", 50, 0)  # 该车不应产生 pedestrian 关系
        p_near = _make_ped("P_near", 5, 0)
        p_far = _make_ped("P_far", 30, 0)  # > threshold
        rels = compute_nearby_pedestrian([v1, v2], [p_near, p_far],
                                         frame_id=0, ego_id="ego")
        # 所有 relations 的 src 必须是 "ego"
        for r in rels:
            assert r.src_id == "ego"
        # 只有 P_near (<20m)
        ped_ids = {r.dst_id for r in rels}
        assert "P_near" in ped_ids
        assert "P_far" not in ped_ids
