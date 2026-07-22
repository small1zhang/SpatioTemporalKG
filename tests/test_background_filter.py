# -*- coding: utf-8 -*-
"""FE-11: BackgroundFilter 静态背景外移单元测试."""
from __future__ import annotations

import pytest

from stk.config import EgoCentricConfig
from stk.filter.background_filter import BackgroundFilter


def _node(nid, ntype):
    return {"id": nid, "type": ntype, "attrs": {}}


def _edge(src, dst, rtype):
    return {"src_id": src, "dst_id": dst, "type": rtype}


class TestDropEntity:
    def test_lane_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_entity(_node("lane1", "RoadElement")) is True

    def test_vehicle_not_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_entity(_node("v1", "Vehicle")) is False

    def test_pedestrian_not_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_entity(_node("p1", "Pedestrian")) is False

    def test_traffic_light_not_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_entity(_node("tl1", "TrafficLight")) is False

    def test_disable_exclude_lanes_keeps_lane(self):
        cfg = EgoCentricConfig.default()
        cfg.exclude_lanes = False
        bg = BackgroundFilter(cfg)
        assert bg.should_drop_entity(_node("lane1", "RoadElement")) is False


class TestDropEdge:
    def test_lane_connects_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("lane1", "lane2", "lane_connects")) is True

    def test_adjacent_lane_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("lane1", "lane2", "adjacent_lane")) is True

    def test_in_lane_vehicle_to_lane_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("v1", "lane1", "in_lane")) is True

    def test_on_road_dropped(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("v1", "road1", "on_road")) is True

    def test_ahead_of_kept(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("v1", "v2", "ahead_of")) is False

    def test_ego_participated_following_kept(self):
        bg = BackgroundFilter()
        assert bg.should_drop_edge(_edge("ego", "v1", "following")) is False


class TestBatch:
    def test_filter_nodes(self):
        bg = BackgroundFilter()
        nodes = [
            _node("v1", "Vehicle"),
            _node("lane1", "RoadElement"),
            _node("lane2", "RoadElement"),
            _node("p1", "Pedestrian"),
        ]
        kept = bg.filter_nodes(nodes)
        ids = {n["id"] for n in kept}
        assert ids == {"v1", "p1"}

    def test_filter_edges(self):
        bg = BackgroundFilter()
        edges = [
            _edge("v1", "lane1", "in_lane"),
            _edge("v1", "v2", "ahead_of"),
            _edge("lane1", "lane2", "lane_connects"),
            _edge("ego", "v1", "following"),
        ]
        kept = bg.filter_edges(edges)
        rtypes = {e["type"] for e in kept}
        assert rtypes == {"ahead_of", "following"}

    def test_stats(self):
        bg = BackgroundFilter()
        nodes = [_node("v1", "Vehicle"), _node("lane1", "RoadElement")]
        edges = [_edge("lane1", "lane2", "lane_connects")]
        s = bg.stats(nodes, edges)
        assert s["n_nodes_in"] == 2
        assert s["n_nodes_dropped"] == 1
        assert s["n_edges_in"] == 1
        assert s["n_edges_dropped"] == 1
        assert s["exclude_lanes"] is True


class TestDisableSwitch:
    def test_disable_keeps_all_lanes(self):
        cfg = EgoCentricConfig.default()
        cfg.exclude_lanes = False
        bg = BackgroundFilter(cfg)
        nodes = [_node("lane1", "RoadElement"), _node("v1", "Vehicle")]
        kept = bg.filter_nodes(nodes)
        assert len(kept) == 2
        edges = [
            _edge("v1", "lane1", "in_lane"),
            _edge("lane1", "lane2", "lane_connects"),
        ]
        kept_e = bg.filter_edges(edges)
        assert len(kept_e) == 2
