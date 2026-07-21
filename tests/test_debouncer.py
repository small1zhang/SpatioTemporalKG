# -*- coding: utf-8 -*-
"""防抖状态机回归测试 (v3 sec 3.4.2).

验证点:
  1. off_counter 抑制: 激活后单帧抖动不关节点, 连续缺失 threshold 帧才删除
  2. on_counter 蓄能: 连续满足 threshold 帧才创建
  3. checkpoint round-trip: to_dict → from_dict → update 行为一致
  4. 旧格式兼容: 只有 counter 字段的旧 checkpoint 仍可解析为 on_counter
"""
from __future__ import annotations

from stk.behavior.debouncer import DebounceItem, RelationDebouncer, DEFAULT_DEBOUNCE_THRESHOLDS


class TestDebounceItem:
    """DebounceItem 防抖项单测."""

    def test_on_counter_threshold(self):
        """连续满足 threshold 帧才 create, 不满不激活."""
        item = DebounceItem(threshold=3)
        # 前 2 帧满足: 不激活
        action, _ = item.update(True, 0)
        assert action == "none", f"f0 expected none, got {action}"
        action, _ = item.update(True, 1)
        assert action == "none", f"f1 expected none, got {action}"
        # 第 3 帧满足: create
        action, extra = item.update(True, 2)
        assert action == "create", f"f2 expected create, got {action}"
        assert extra["on_counter"] == 3
        assert item.is_active is True

    def test_off_counter_threshold(self):
        """激活后单帧抖动不关 (keep), 连续缺失 threshold 帧才 delete."""
        item = DebounceItem(threshold=2)
        # 激活
        item.update(True, 0)
        item.update(True, 1)
        # 第 1 帧不满足: off_counter=1, 仍 keep (不是 delete)
        action, extra = item.update(False, 2)
        assert action == "keep", f"f2 expected keep, got {action}"
        assert item.is_active is True, "single miss should NOT deactivate"
        # 第 2 帧不满足: off_counter=2, 达到 threshold, 正式 delete
        action, extra = item.update(False, 3)
        assert action == "delete", f"f3 expected delete, got {action}"
        assert item.is_active is False

    def test_immediate_delete_without_active(self):
        """从未激活的条件下, 条件消失一直 none."""
        item = DebounceItem(threshold=2)
        action, _ = item.update(False, 0)
        assert action == "none", f"expected none, got {action}"
        action, _ = item.update(False, 1)
        assert action == "none", f"expected none, got {action}"


class TestRelationDebouncer:
    """RelationDebouncer 整体测试."""

    def test_create_and_keep_cycle(self):
        d = RelationDebouncer(thresholds={"following": 3})
        key = ("v1", "v2", "following")
        # 前 2 帧: none
        action, _ = d.update("following", key, True, 100)
        assert action == "none"
        action, _ = d.update("following", key, True, 101)
        assert action == "none"
        # 第 3 帧: create
        action, _ = d.update("following", key, True, 102)
        assert action == "create"
        # 后续保持: keep
        action, _ = d.update("following", key, True, 103)
        assert action == "keep"

    def test_delete_after_off_threshold(self):
        d = RelationDebouncer(thresholds={"following": 2})
        key = ("v1", "v2", "following")
        d.update("following", key, True, 100)
        d.update("following", key, True, 101)
        # 激活后 1 帧不满足: keep (抖动抑制)
        action, _ = d.update("following", key, False, 102)
        assert action == "keep", f"expected keep on first miss, got {action}"
        # 再 1 帧不满足: delete
        action, _ = d.update("following", key, False, 103)
        assert action == "delete", f"expected delete on second miss, got {action}"

    def test_checkpoint_round_trip(self):
        """to_dict → from_dict → update 行为一致."""
        d = RelationDebouncer(thresholds={"following": 2})
        key = ("v1", "v2", "following")
        d.update("following", key, True, 100)
        d.update("following", key, True, 101)  # create at 101
        d.update("following", key, True, 102)  # keep

        # 序列化
        buf = d.to_dict()
        d2 = RelationDebouncer.from_dict(buf)

        # 恢复后继续
        key2 = ("v1", "v2", "following")
        action, _ = d2.update("following", key2, True, 103)
        assert action == "keep", f"post-restore expected keep, got {action}"
        # 再 1 帧不满足: 抖动抑制
        action, _ = d2.update("following", key2, False, 104)
        assert action == "keep", f"post-restore miss expected keep, got {action}"
        # 再 1 帧不满足: delete
        action, _ = d2.update("following", key2, False, 105)
        assert action == "delete", f"post-restore second miss expected delete, got {action}"

    def test_old_format_compatibility(self):
        """旧格式 checkpoint (只有 'counter' 字段) 仍能恢复为 on_counter."""
        old_data = {
            "thresholds": {"following": 3},
            "items": {
                "('v1', 'v2', 'following')": {
                    "threshold": 3, "counter": 2,
                    "is_active": False, "active_since": None,
                    "last_condition_met": None,
                },
            },
        }
        d = RelationDebouncer.from_dict(old_data)
        key = ("v1", "v2", "following")
        # counter=2, 再 1 帧满足就 create
        action, _ = d.update("following", key, True, 100)
        assert action == "create", f"old-format resume expected create, got {action}"
