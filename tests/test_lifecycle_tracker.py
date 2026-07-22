# -*- coding: utf-8 -*-
"""FE-8: LifecycleTracker ENTER/UPDATE/EXIT/FORGET 单元测试."""
from __future__ import annotations

import pytest

from stk.filter.lifecycle import LifecycleTracker


class TestLifecycleEnterUpdate:
    def test_first_appearance_enter(self):
        lc = LifecycleTracker()
        events = lc.step({"v1"}, frame_id=0)
        assert events["v1"] == "ENTER"
        assert lc.state_of("v1") == "ENTER"

    def test_continuous_presence_update(self):
        lc = LifecycleTracker()
        # 帧 1: ENTER
        lc.step({"v1"}, frame_id=1)
        # 帧 2: UPDATE
        ev2 = lc.step({"v1"}, frame_id=2)
        assert ev2["v1"] == "UPDATE"
        # 帧 3: UPDATE
        ev3 = lc.step({"v1"}, frame_id=3)
        assert ev3["v1"] == "UPDATE"
        assert lc.state_of("v1") == "UPDATE"

    def test_second_appearance_after_exit_is_enter(self):
        """EXIT 之后再在 ROI 内 → ENTER."""
        lc = LifecycleTracker(hysteresis_frames=2, forget_frames=10)
        lc.step({"v1"}, frame_id=1)     # ENTER
        lc.step({"v1"}, frame_id=2)     # UPDATE
        for fid in range(3, 13):
            lc.step(set(), frame_id=fid)   # 持续不在 ROI
        # hysteresis=2 → frame 4 即 EXIT; forget=10 → frame 14 FORGET
        assert lc.state_of("v1") in ("EXIT", "FORGET")
        # 重新出现
        ev = lc.step({"v1"}, frame_id=14)
        assert ev["v1"] == "ENTER"


class TestLifecycleExit:
    def test_exit_after_hysteresis_frames(self):
        """连续 hysteresis_frames=3 帧不在 ROI → EXIT."""
        lc = LifecycleTracker(hysteresis_frames=3, forget_frames=30)
        lc.step({"v1"}, frame_id=1)    # ENTER
        lc.step({"v1"}, frame_id=2)    # UPDATE
        # 帧 3 不在 ROI, 仍 UPDATE (抖动期间)
        ev3 = lc.step(set(), frame_id=3)
        assert ev3["v1"] == "UPDATE"
        # 帧 4 不在 ROI, 仍 UPDATE
        ev4 = lc.step(set(), frame_id=4)
        assert ev4["v1"] == "UPDATE"
        # 帧 5 不在 ROI, consecutive=3 → EXIT
        ev5 = lc.step(set(), frame_id=5)
        assert ev5["v1"] == "EXIT"
        assert lc.state_of("v1") == "EXIT"

    def test_brief_absence_below_hysteresis_keeps_update(self):
        """离开 1 帧后回来,期间 hysteresis=3 → 不触发 EXIT."""
        lc = LifecycleTracker(hysteresis_frames=3)
        lc.step({"v1"}, frame_id=1)
        lc.step({"v1"}, frame_id=2)
        ev3 = lc.step(set(), frame_id=3)   # 缺一帧
        ev4 = lc.step({"v1"}, frame_id=4)  # 重回 ROI
        assert ev3["v1"] == "UPDATE"
        assert ev4["v1"] == "UPDATE"
        assert lc.state_of("v1") == "UPDATE"


class TestLifecycleForget:
    def test_forget_after_forget_frames(self):
        """EXIT 后连续 forget_frames 不在 ROI → FORGET, 内部清理.

        hysteresis=2 → consecutive>=2 触发 EXIT (frame 4 即 EXIT).
        forget=5   → consecutive>=5 触发 FORGET (frame 7 即 FORGET).
        """
        lc = LifecycleTracker(hysteresis_frames=2, forget_frames=5)
        lc.step({"v1"}, frame_id=1)
        lc.step({"v1"}, frame_id=2)
        # 帧 3: hysteresis=2 未到, still UPDATE
        ev3 = lc.step(set(), frame_id=3)
        assert ev3["v1"] == "UPDATE"
        # 帧 4: consecutive=2 → EXIT
        ev4 = lc.step(set(), frame_id=4)
        assert ev4["v1"] == "EXIT"
        # 帧 5, 6: consecutive=3, 4, still EXIT
        ev5 = lc.step(set(), frame_id=5)
        ev6 = lc.step(set(), frame_id=6)
        assert ev5["v1"] == "EXIT"
        assert ev6["v1"] == "EXIT"
        # 帧 7: consecutive=5 → FORGET
        ev7 = lc.step(set(), frame_id=7)
        assert ev7["v1"] == "FORGET"
        assert lc.state_of("v1") == "FORGET"

    def test_forget_clears_internal_state(self):
        """FORGET 后内部 trackers 中应被清理."""
        lc = LifecycleTracker(hysteresis_frames=1, forget_frames=2)
        lc.step({"v1"}, frame_id=1)
        lc.step(set(), frame_id=2)   # EXIT (consecutive=1)
        lc.step(set(), frame_id=3)   # EXIT
        lc.step(set(), frame_id=4)   # FORGET
        assert "v1" not in lc._trackers


class TestLifecycleMultiEntity:
    def test_multiple_entities_independent(self):
        lc = LifecycleTracker(hysteresis_frames=3, forget_frames=30)
        # 帧 1: 出现 v1, v2
        ev1 = lc.step({"v1", "v2"}, frame_id=1)
        assert ev1["v1"] == "ENTER"
        assert ev1["v2"] == "ENTER"
        # 帧 2: v1 离开, v3 进入
        ev2 = lc.step({"v2", "v3"}, frame_id=2)
        assert ev2["v2"] == "UPDATE"
        assert ev2["v3"] == "ENTER"
        assert ev2["v1"] == "UPDATE"  # 抖动期间


class TestLifecycleCheckpoint:
    def test_round_trip(self):
        lc = LifecycleTracker(hysteresis_frames=3, forget_frames=30)
        lc.step({"v1", "v2"}, frame_id=1)
        lc.step({"v1"}, frame_id=2)
        d = lc.to_dict()
        lc2 = LifecycleTracker.from_dict(d)
        assert lc2._hysteresis == 3
        assert lc2._forget == 30
        assert lc2.state_of("v1") == "UPDATE"
        assert lc2.state_of("v2") == "UPDATE"  # 抖动期 (absent=1)

    def test_reset(self):
        lc = LifecycleTracker()
        lc.step({"v1"}, frame_id=1)
        assert len(lc._trackers) == 1
        lc.reset()
        assert len(lc._trackers) == 0
        assert lc.state_of("v1") == "FORGET"

    def test_stats(self):
        lc = LifecycleTracker()
        lc.step({"v1", "v2"}, frame_id=1)  # 2 ENTER
        lc.step({"v1"}, frame_id=2)         # v1 UPDATE, v2 absent
        stats = lc.stats()
        assert stats["n_tracked"] == 2
        assert stats["state_counts"]["UPDATE"] == 2  # 都还在 UPDATE
        assert stats["hysteresis_frames"] == 3
        assert stats["forget_frames"] == 30
