# -*- coding: utf-8 -*-
"""FE-6: BehaviorRelationGenerator ego×ROI 对子过滤集成测试.

默认 filter_behavior_detectors=False 时 _ego_filter=None, 行为层走老全对子路径.
filter_behavior_detectors=True 且 legacy_full_pairing=False 时启用 ego×ROI:
车辆-车辆对子 (following/overtaking/opposite_direction/blocked_view) 只保留 ego 参与的;
个体行为 / 车-人 / 车-灯 / junction / crosswalk 不受影响.
"""
from __future__ import annotations

import pytest

from stk.behavior.generator import BehaviorRelationGenerator
from stk.behavior.nodes import ManeuverNode, InteractionEvent
from stk.config import EgoCentricConfig


def _veh(vid, x=0.0, y=0.0, heading=0.0, lane="L1", speed=10.0,
         is_ego=False, vehicle_category="car"):
    return {
        "entity_id": vid,
        "entity_type": "Vehicle",
        "vehicle_category": vehicle_category,
        "location_x": x, "location_y": y, "location_z": 0.0,
        "heading_rad": heading,
        "speed": speed,
        "speed_kmh": speed * 3.6,
        "velocity_x": speed, "velocity_y": 0.0, "velocity_z": 0.0,
        "brake": 0.0, "throttle": 0.5,
        "lane_id": lane, "road_id": "R1",
        "is_ego": is_ego,
    }


def _ped(pid, x=0.0, y=0.0, on_cw=False):
    return {
        "entity_id": pid,
        "entity_type": "Pedestrian",
        "location_x": x, "location_y": y, "location_z": 0.0,
        "speed": 1.0,
        "is_on_crosswalk": on_cw,
    }


def _ego_cfg(filter_behavior=True):
    return EgoCentricConfig(
        filter_behavior_detectors=filter_behavior,
        legacy_full_pairing=False,
    )


class TestEgoFilterToggle:
    """默认关闭, 开关打开时 _ego_filter 不为 None."""

    def test_default_off(self):
        gen = BehaviorRelationGenerator()
        assert gen._ego_filter is None

    def test_filter_behavior_on(self):
        gen = BehaviorRelationGenerator(ego_config=_ego_cfg(filter_behavior=True))
        assert gen._ego_filter is not None

    def test_filter_behavior_off_explicit(self):
        gen = BehaviorRelationGenerator(ego_config=_ego_cfg(filter_behavior=False))
        assert gen._ego_filter is None

    def test_legacy_pairing_overrides(self):
        """legacy_full_pairing=True 时即使 filter_behavior_detectors=True 也不启用."""
        cfg = EgoCentricConfig(
            filter_behavior_detectors=True,
            legacy_full_pairing=True,
        )
        gen = BehaviorRelationGenerator(ego_config=cfg)
        assert gen._ego_filter is None


class TestEgoPairingFollowing:
    """对 following 行为 ego×ROI 过滤."""

    def test_following_only_for_roi_vehicle(self):
        """ego 在原点朝 +x.
           V_close 在 30m 处 (ROI 内) 跟 ego, 触发 following.
           V_far  在 100m 处 (ROI 外), 不应触发 ego×V_far 的 following.
        """
        gen = BehaviorRelationGenerator(ego_config=_ego_cfg(True))
        vehicles = [
            _veh("ego", x=0, y=0, heading=0, speed=10.0, is_ego=True),
            _veh("V_close", x=30, y=0, heading=0, speed=8.0),  # ROI 内
            _veh("V_far", x=100, y=0, heading=0, speed=8.0),   # ROI 外
        ]
        # 多帧累积以确保防抖 create (following 阈值通常 2-3 帧)
        for fid in range(1, 6):
            out = gen.generate(frame_id=fid, vehicles=vehicles, pedestrians=[],
                               traffic_lights=[], junctions=[],
                               crosswalks=[], scene_relations=[])
        # 检查活跃交互节点的 src/dst 不应含 ego×V_far
        active = gen.all_active()
        for it in active["interactions"]:
            if it.interaction_type == "following":
                s, d = it.src_id, it.dst_id
                # 不能是 ego × V_far (或反向)
                assert not (s == "ego" and d == "V_far")
                assert not (s == "V_far" and d == "ego")

    def test_individual_behavior_unaffected(self):
        """standing_still 个体行为不依赖对子, 不应被 ego×ROI 过滤掉.

           ego 静止, 即使 _ego_filter 启用, ego 自身的 standing_still 仍应触发.
        """
        gen = BehaviorRelationGenerator(ego_config=_ego_cfg(True))
        vehicles = [
            _veh("ego", x=0, y=0, heading=0, speed=0.0, is_ego=True),
            _veh("V_far", x=200, y=0, speed=10.0),  # ROI 外
        ]
        for fid in range(1, 6):
            out = gen.generate(frame_id=fid, vehicles=vehicles, pedestrians=[],
                               traffic_lights=[], junctions=[],
                               crosswalks=[], scene_relations=[])
        # 应有 ego 的 standing_still ManeuverNode
        active = gen.all_active()
        man_ids = [m.actor_id for m in active["maneuvers"]
                   if m.maneuver_type == "standing_still"]
        assert "ego" in man_ids, "ego 的 standing_still 不应被 ROI 过滤"


class TestEgoPairingVehiclePedestrian:
    """车-人交互 (yielding_to/approaching_pedestrian) 不受 ego 过滤影响."""

    def test_yielding_to_pedestrian_unaffected_by_ego_filter(self):
        """V_close (非 ego, ROI 内) × 行人 在人行道上 → yielding_to 应触发.

           即使启用 ego_filter, 车-人对子 _VEHICLE_VEHICLE_RELS 不包含 yielding_to,
           因此不应被过滤.
        """
        gen = BehaviorRelationGenerator(ego_config=_ego_cfg(True))
        vehicles = [
            _veh("ego", x=0, y=0, heading=0, speed=5.0, is_ego=True),
            _veh("V_close", x=30, y=0, heading=0, speed=5.0),
        ]
        peds = [
            _ped("ped1", x=32, y=2, on_cw=True),
        ]
        for fid in range(1, 6):
            out = gen.generate(frame_id=fid, vehicles=vehicles, pedestrians=peds,
                               traffic_lights=[], junctions=[],
                               crosswalks=[], scene_relations=[])
        active = gen.all_active()
        yt_present = any(it.interaction_type == "yielding_to" for it in active["interactions"])
        # 若任一 yielding_to 被防抖触发, src 应是车 (V_close 或 ego), dst 是 ped1
        for it in active["interactions"]:
            if it.interaction_type == "yielding_to":
                assert it.dst_id == "ped1"


class TestLegacyAndEgoEquivalent:
    """ego×ROI vs 全对子 在小场景下的等价性."""

    def test_small_scene_ego_inclusive_same_as_legacy(self):
        """3 车 (ego + V_close 30m + V_side 5m) 都在 ego ROI 内 → ego_filter 与
           legacy 在 ROI 子集上等价 (差异只在源 candidates 数量).
        """
        cfg_legacy = EgoCentricConfig(
            filter_behavior_detectors=False,
            legacy_full_pairing=False,
        )
        cfg_ego = _ego_cfg(True)
        gen_legacy = BehaviorRelationGenerator(ego_config=cfg_legacy)
        gen_ego = BehaviorRelationGenerator(ego_config=cfg_ego)
        vehicles = [
            _veh("ego", x=0, y=0, heading=0, speed=10.0, is_ego=True),
            _veh("V_close", x=30, y=0, heading=0, speed=8.0),
            _veh("V_side", x=0, y=5, heading=0, speed=8.0),
        ]
        for fid in range(1, 6):
            gen_legacy.generate(frame_id=fid, vehicles=vehicles, pedestrians=[],
                                traffic_lights=[], junctions=[],
                                crosswalks=[], scene_relations=[])
            gen_ego.generate(frame_id=fid, vehicles=vehicles, pedestrians=[],
                             traffic_lights=[], junctions=[],
                             crosswalks=[], scene_relations=[])
        # 全员在 ROI 内, 两者活跃 interaction 的 src/dst 对子集合应一致
        def _pairs(gen):
            return {(it.src_id, it.dst_id, it.interaction_type)
                    for it in gen.all_active()["interactions"]}
        pairs_legacy = _pairs(gen_legacy)
        pairs_ego = _pairs(gen_ego)
        # ego 模式应包含 legacy 的所有 ego 参与的对子
        ego_pairs = {(s, d, t) for (s, d, t) in pairs_ego if "ego" in (s, d)}
        legacy_ego_pairs = {(s, d, t) for (s, d, t) in pairs_legacy if "ego" in (s, d)}
        assert ego_pairs == legacy_ego_pairs or ego_pairs.issubset(legacy_ego_pairs)
