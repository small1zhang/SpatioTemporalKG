# -*- coding: utf-8 -*-
"""增量引擎回归测试: 跳帧检测 + 字符串污染 + checkpoint 非序列化防守.

验证点:
  1. frame_jump: prev=99, curr=200 时 baseline 重置, delta 按全 added 处理
  2. string_pollution: prev_frame 含字符串数字 (如 location_x="100.0")
     to_dict 应返回 prev=None
  3. non_serializable: prev_frame 含不可 JSON 序列化的自定义对象
     to_dict 应返回 prev=None, 不抛异常
"""
from __future__ import annotations

from stk.dynamic.incremental_updater import IncrementalEngine, _validate_numeric_attrs


class TestValidateNumericAttrs:
    """_validate_numeric_attrs 工具函数单测."""

    def test_normal_dict(self):
        frame = {
            "vehicles": [{"entity_id": "v1", "location_x": 100.0, "speed": 5.0}],
            "pedestrians": [],
        }
        assert _validate_numeric_attrs(frame) is True

    def test_string_pollution(self):
        """location_x 是字符串 -> 应拒绝."""
        frame = {
            "vehicles": [{"entity_id": "v1", "location_x": "100.0", "speed": 5.0}],
            "pedestrians": [],
        }
        assert _validate_numeric_attrs(frame) is False, \
            "string attr should be rejected"

    def test_bool_is_rejected(self):
        """bool 是 int 子类, 但应被拒."""
        frame = {
            "vehicles": [{"entity_id": "v1", "location_x": True, "speed": 5.0}],
            "pedestrians": [],
        }
        assert _validate_numeric_attrs(frame) is False, \
            "bool attr should be rejected"

    def test_none_field_is_ok(self):
        """None 跳过, 不报错."""
        frame = {
            "vehicles": [{"entity_id": "v1", "location_x": None, "speed": 5.0}],
            "pedestrians": [],
        }
        assert _validate_numeric_attrs(frame) is True

    def test_pedestrian_speed_string(self):
        """pedestrian 中 speed 是字符串 -> 拒绝."""
        frame = {
            "vehicles": [],
            "pedestrians": [{"entity_id": "p1", "speed": "2.0"}],
        }
        assert _validate_numeric_attrs(frame) is False


class TestFrameJump:
    """跳帧时 baseline 应重置."""

    def test_jump_resets_prev(self):
        engine = IncrementalEngine()
        f1 = {"frame_id": 99, "vehicles": [{"entity_id": "v1", "speed": 5}],
              "pedestrians": [], "traffic_lights": [], "scene_rels": [],
              "behavior_rels": []}
        f2 = {"frame_id": 200, "vehicles": [{"entity_id": "v1", "speed": 6}],
              "pedestrians": [], "traffic_lights": [], "scene_rels": [],
              "behavior_rels": []}
        dg1 = engine.process_frame(f1)
        # 跳帧: prev=99, curr=200, 应重置 prev=None
        dg2 = engine.process_frame(f2)
        # 重置 baseline 后相当于首帧 -> entities = added
        assert len(dg2.delta_entities.added) > 0, \
            "frame jump should reset baseline, so all entities become 'added'"
        assert len(dg2.delta_entities.removed) == 0

    def test_continuous_frames_normal(self):
        engine = IncrementalEngine()
        f1 = {"frame_id": 0, "vehicles": [{"entity_id": "v1", "speed": 5}],
              "pedestrians": [], "traffic_lights": [], "scene_rels": [],
              "behavior_rels": []}
        f2 = {"frame_id": 1, "vehicles": [{"entity_id": "v1", "speed": 5.1}],
              "pedestrians": [], "traffic_lights": [], "scene_rels": [],
              "behavior_rels": []}
        dg1 = engine.process_frame(f1)
        dg2 = engine.process_frame(f2)
        # 连续帧: 无跳帧, entities 应为 unchanged
        assert len(dg2.delta_entities.unchanged) > 0, \
            "continuous frames should have unchanged entities"


class TestCheckpointSafety:
    """to_dict 防守型清理验证."""

    def test_string_pollution_dropped(self):
        engine = IncrementalEngine()
        engine._prev_frame = {
            "frame_id": 5,
            "vehicles": [{"entity_id": "v1", "location_x": "100.0", "speed": 5.0}],
            "pedestrians": [], "traffic_lights": [], "scene_rels": [],
        }
        buf = engine.to_dict()
        # 字符串污染应被丢弃, prev=None
        assert buf["prev_frame"] is None, \
            f"string pollution should be dropped, got {buf['prev_frame']}"

    def test_non_serializable_dropped(self):
        """含不可序列化对象的 prev -> to_dict 不抛异常, prev=None."""
        class CustomObj:
            pass
        engine = IncrementalEngine()
        engine._prev_frame = {
            "frame_id": 3,
            "vehicles": [{"entity_id": "v1", "location_x": 100.0, "speed": 5.0, "custom": CustomObj()}],
            "pedestrians": [], "traffic_lights": [], "scene_rels": [],
        }
        buf = engine.to_dict()
        assert buf["prev_frame"] is None, \
            f"non-serializable object should be dropped, got {buf['prev_frame']}"

    def test_normal_frame_serialized(self):
        engine = IncrementalEngine()
        engine._prev_frame = {
            "frame_id": 5,
            "vehicles": [{"entity_id": "v1", "location_x": 100.0, "speed": 5.0}],
            "pedestrians": [], "traffic_lights": [], "scene_rels": [],
        }
        buf = engine.to_dict()
        assert buf["prev_frame"] is not None, "normal frame should survive"
        assert buf["prev_frame"]["frame_id"] == 5
        assert buf["n_deltas"] == 0

    def test_load_dict_polluted_prev_resets(self):
        """load_dict 含污染 prev 时, 引擎自动清空 prev."""
        engine = IncrementalEngine()
        data = {
            "prev_frame": {
                "frame_id": 10,
                "vehicles": [{"entity_id": "v1", "location_x": "bad_string", "speed": 5.0}],
                "pedestrians": [], "traffic_lights": [], "scene_rels": [],
            },
        }
        engine.load_dict(data)
        assert engine._prev_frame is None, \
            "polluted load_dict should reset prev to None"
