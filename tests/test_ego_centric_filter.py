# -*- coding: utf-8 -*-
"""TestE4: EgoCentric ROI 过滤单元测试 (阶段1)."""
from __future__ import annotations

import math

import pytest

from stk.filter.roi import in_ego_ellipse
from stk.filter.generator import EgoCentricFilter, EgoRoiDecision
from stk.config import EgoCentricConfig

# ════════════════════════════════════════════
# 纯函数测试: in_ego_ellipse
# ════════════════════════════════════════════

_R = 70.0, 30.0, 50.0  # (radius_front, radius_rear, radius_side)


class TestInEgoEllipse:
    """笛卡尔椭圆判定双线性边界."""

    def test_front_inside(self):
        """前方 60m → 在 ROI 内 (前 R=70)."""
        assert in_ego_ellipse(0, 0, 0, 60, 0, *_R)

    def test_front_outside(self):
        """前方 90m → 超出 ROI (前 R=70)."""
        assert not in_ego_ellipse(0, 0, 0, 90, 0, *_R)

    def test_front_boundary_inside(self):
        """前方 70m 正好在椭圆边界上 → 在 ROI 内 (≤1)."""
        assert in_ego_ellipse(0, 0, 0, 70, 0, *_R)

    def test_rear_inside(self):
        """后方 20m → 在 ROI 内 (后 R=30)."""
        assert in_ego_ellipse(0, 0, 0, -20, 0, *_R)

    def test_rear_outside(self):
        """后方 50m → 超出 ROI (后 R=30)."""
        assert not in_ego_ellipse(0, 0, 0, -50, 0, *_R)

    def test_side_inside(self):
        """侧向 40m → 在 ROI 内 (侧 R=50)."""
        assert in_ego_ellipse(0, 0, 0, 0, 40, *_R)

    def test_side_outside(self):
        """侧向 80m → 超出 ROI (侧 R=50)."""
        assert not in_ego_ellipse(0, 0, 0, 0, 80, *_R)

    def test_heading_rotation_east(self):
        """ego 朝向 +x (东), 目标在前方 50m 侧向 20m → 椭圆内.
        (50/70)² + (20/50)² = 0.510 + 0.160 = 0.670 ≤ 1."""
        assert in_ego_ellipse(0, 0, 0, 50, 20, *_R)

    def test_heading_rotation_north(self):
        """ego 朝向 +y (北), 目标在 ego"前方" (沿 +y) 60m → ROI 内."""
        # 此时在全局坐标下, 目标在 (0, 60), 但 ego 朝向北
        # 车体坐标系: y 正方向为 longitudinal → 投影: lon = 60 >= 0
        assert in_ego_ellipse(0, 0, math.pi / 2, 0, 60, *_R)

    def test_heading_rotation_south(self):
        """ego 朝向 -y (南), 目标在全局 (0, -60) 是 ego 前方 → ROI 内."""
        assert in_ego_ellipse(0, 0, -math.pi / 2, 0, -60, *_R)

    def test_heading_far_side_outside(self):
        """ego 朝向 45°, 目标前方 80m 侧向 10m → 超出前向 70m ROI."""
        h = math.pi / 4
        dist = 80.0
        tgt_x = dist * math.cos(h)
        tgt_y = dist * math.sin(h)
        assert not in_ego_ellipse(0, 0, h, tgt_x, tgt_y, *_R)

    def test_zero_radius_returns_false(self):
        """半径 ≤0 时安全返回 False."""
        assert not in_ego_ellipse(0, 0, 0, 10, 0, radius_front=0, radius_rear=0, radius_side=0)


# ════════════════════════════════════════════
# EgoCentricFilter.select 测试
# ════════════════════════════════════════════

def _vehicle(entity_id: str, x: float = 0, y: float = 0,
             heading: float = 0, is_ego: bool = False,
             speed: float = 0.0, brake: float = 0.0) -> dict:
    """构造简化的车辆字典 (匹配 actor_extractor 字段)."""
    return {
        "entity_id": entity_id,
        "location_x": x,
        "location_y": y,
        "heading_rad": heading,
        "is_ego": is_ego,
        "speed": speed,
        "brake": brake,
        "entity_type": "Vehicle",
    }


class TestEgoCentricFilterSelect:
    """EgoCentricFilter.select 输出对错."""

    def _make_filter(self, **overrides) -> EgoCentricFilter:
        cfg = EgoCentricConfig.default()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return EgoCentricFilter(cfg)

    def test_pick_ego_by_is_ego_flag(self):
        """优先按 is_ego==True 识别自车."""
        vs = [
            _vehicle("V0", is_ego=False),
            _vehicle("ego", is_ego=True, x=10, y=0),
            _vehicle("V1", is_ego=False),
        ]
        ef = self._make_filter()
        d = ef.select(vs, frame_id=0)
        assert d.ego is not None
        assert d.ego["entity_id"] == "ego"

    def test_pick_ego_by_explicit_id(self):
        """显式 ego_id_opt 优先于 is_ego."""
        vs = [
            _vehicle("V0", is_ego=True),    # 有 is_ego, 但被显式覆盖
            _vehicle("explicit_ego", x=10, y=0),
        ]
        ef = self._make_filter(ego_id_opt="explicit_ego")
        d = ef.select(vs, frame_id=0)
        assert d.ego["entity_id"] == "explicit_ego"

    def test_pick_ego_fallback_first(self):
        """无 is_ego 无 ego_id_opt → fallback vehicles[0]."""
        vs = [
            _vehicle("first"),
            _vehicle("second"),
        ]
        ef = self._make_filter()
        d = ef.select(vs, frame_id=0)
        assert d.ego["entity_id"] == "first"

    def test_select_roi_vs_dropped(self):
        """10 辆车, 1 ego + 1 近 + 8 远 → 输出正确."""
        vs = [_vehicle("ego", x=0, y=0, is_ego=True)]
        # 前方 30m ROI 内
        vs.append(_vehicle("close", x=30, y=0))
        # 其余超远
        for i in range(8):
            vs.append(_vehicle(f"far_{i}", x=200 + i, y=0))
        ef = self._make_filter()
        d = ef.select(vs, frame_id=0)
        assert d.ego["entity_id"] == "ego"
        assert len(d.roi_targets) == 1
        assert d.roi_targets[0]["entity_id"] == "close"
        assert len(d.dropped) == 8

    def test_select_empty_input(self):
        """空列表 → ego=None, 无 ROI."""
        d = EgoCentricFilter().select([], frame_id=0)
        assert d.ego is None
        assert len(d.roi_targets) == 0
        assert len(d.dropped) == 0

    def test_select_single_vehicle_no_other(self):
        """只有 ego 一车 → ego 识别, roi 为空, dropped 为空."""
        vs = [_vehicle("ego", x=0, y=0, is_ego=True)]
        d = EgoCentricFilter().select(vs, frame_id=0)
        assert d.ego["entity_id"] == "ego"
        assert len(d.roi_targets) == 0
        assert len(d.dropped) == 0

    def test_roi_vehicle_by_rear_inside(self):
        """后方 20m 处的车在 ROI 内 (后向 R=30)."""
        vs = [
            _vehicle("ego", x=0, y=0, is_ego=True),
            _vehicle("behind", x=0, y=-20),
        ]
        d = EgoCentricFilter().select(vs, frame_id=0)
        assert len(d.roi_targets) == 1
        assert d.roi_targets[0]["entity_id"] == "behind"

    def test_roi_vehicle_by_side_inside(self):
        """侧向 49m 处的车在 ROI 内 (侧 R=50)."""
        vs = [
            _vehicle("ego", x=0, y=0, is_ego=True),
            _vehicle("side", x=0, y=49),
        ]
        d = EgoCentricFilter().select(vs, frame_id=0)
        assert len(d.roi_targets) == 1

    def test_ego_not_repeated_in_roi(self):
        """ego 自身不出现在 roi_targets/dropped."""
        vs = [_vehicle("ego", x=0, y=0, is_ego=True),
              _vehicle("other", x=10, y=0)]
        d = EgoCentricFilter().select(vs, frame_id=0)
        ids_roi = [v["entity_id"] for v in d.roi_targets]
        ids_drop = [v["entity_id"] for v in d.dropped]
        assert "ego" not in ids_roi
        assert "ego" not in ids_drop

    def test_legacy_config_no_filter(self):
        """legacy_full_pairing=True 时 select 仍正常工作 (调用方负责切换)."""
        vs = [
            _vehicle("ego", x=0, y=0, is_ego=True),
            _vehicle("far", x=200, y=0),
        ]
        ef = self._make_filter(legacy_full_pairing=True)
        d = ef.select(vs, frame_id=0)
        # legacy 模式只影响 RuleEnforcer 的对子枚举策略, filter.select 仍正常
        assert len(d.roi_targets) == 0
        assert len(d.dropped) == 1


# ════════════════════════════════════════════
# EgoCentricConfig 序列化
# ════════════════════════════════════════════

class TestEgoCentricConfig:
    """配置类 round-trip."""

    def test_default_values(self):
        cfg = EgoCentricConfig.default()
        assert cfg.radius_front == 70.0
        assert cfg.radius_rear == 30.0
        assert cfg.radius_side == 50.0
        assert cfg.legacy_full_pairing is False

    def test_from_dict_round_trip(self):
        d = {"radius_front": 80.0, "radius_rear": 40.0, "legacy_full_pairing": True}
        cfg = EgoCentricConfig.from_dict(d)
        assert cfg.radius_front == 80.0
        assert cfg.radius_rear == 40.0
        assert cfg.legacy_full_pairing is True
        # 未覆盖的字段保持默认
        assert cfg.radius_side == 50.0

    def test_to_dict_round_trip(self):
        cfg = EgoCentricConfig(radius_front=50.0, radius_rear=20.0)
        d = cfg.to_dict()
        assert d["radius_front"] == 50.0
        assert d["radius_rear"] == 20.0

    def test_update_from(self):
        cfg = EgoCentricConfig.default()
        cfg.update_from({"radius_side": 60.0, "hysteresis_frames": 5})
        assert cfg.radius_side == 60.0
        assert cfg.hysteresis_frames == 5
