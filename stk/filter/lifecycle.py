# -*- coding: utf-8 -*-
"""LifecycleTracker: 实体进出 ROI 的 ENTER/UPDATE/EXIT/FORGET 生命周期状态机 (阶段2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class _EntityState:
    """单个实体的生命周期状态."""

    state: str = "FORGET"   # ENTER | UPDATE | EXIT | FORGET
    consecutive_absent: int = 0   # 连续不在 ROI 的帧数 (for EXIT/FORGET debounce)
    enter_frame: int = 0
    last_seen_frame: int = 0
    exit_frame: int = 0


class LifecycleTracker:
    """实体进出 ROI 的生命周期状态机.

    每帧调用 step(current_roi_ids, frame_id) 获取当帧所有实体的生命周期事件.
    内部用 hysteresis_frames 做 EXIT 防抖; forget_frames 做 FORGET 过期清理.

    用法::

        lc = LifecycleTracker(hysteresis_frames=3, forget_frames=30)
        events = lc.step({"v1", "v2"}, frame_id=42)
        # => {"v1": "UPDATE", "v2": "UPDATE"}  # 持续存在
        events = lc.step({"v1"}, frame_id=43)
        # => {"v1": "UPDATE", "v2": "EXIT"}    # v2 离开
    """

    def __init__(
        self,
        hysteresis_frames: int = 3,
        forget_frames: int = 30,
    ):
        self._hysteresis = hysteresis_frames
        self._forget = forget_frames
        self._trackers: Dict[str, _EntityState] = {}

    def step(self, current_ids: Set[str], frame_id: int) -> Dict[str, str]:
        """对当前帧执行生命周期状态推进.

        Args:
            current_ids: 当前帧在 ROI 内的实体 id 集合.
            frame_id: 当前帧号 (用于 ENTER/EXIT/FORGET 记录).

        Returns:
            {entity_id: "ENTER"|"UPDATE"|"EXIT"|"FORGET"}
        """
        events: Dict[str, str] = {}

        # 1. 处理所有已知实体: 在 ROI 中 vs 不在 ROI 中
        for eid, ts in list(self._trackers.items()):
            if eid in current_ids:
                # 在 ROI 内
                if ts.state in ("FORGET", "EXIT"):
                    ts.state = "ENTER"
                    ts.enter_frame = frame_id
                    ts.consecutive_absent = 0
                    events[eid] = "ENTER"
                elif ts.state in ("ENTER", "UPDATE"):
                    ts.state = "UPDATE"
                    ts.last_seen_frame = frame_id
                    events[eid] = "UPDATE"
            else:
                # 不在 ROI 内
                ts.consecutive_absent += 1
                if ts.state in ("ENTER", "UPDATE"):
                    if ts.consecutive_absent >= self._hysteresis:
                        ts.state = "EXIT"
                        ts.exit_frame = frame_id
                        events[eid] = "EXIT"
                    else:
                        # 抖动期间, 状态从 ENTER 推进到 UPDATE,
                        # 仍返回 UPDATE 让调用方不删除
                        if ts.state == "ENTER":
                            ts.state = "UPDATE"
                        events[eid] = "UPDATE"
                elif ts.state == "EXIT":
                    if ts.consecutive_absent >= self._forget:
                        ts.state = "FORGET"
                        events[eid] = "FORGET"
                        self._trackers.pop(eid)  # 清理
                    else:
                        ts.exit_frame = frame_id
                        events[eid] = "EXIT"

        # 2. 处理新出现的实体
        for eid in current_ids:
            if eid not in self._trackers:
                ts = _EntityState(state="ENTER", enter_frame=frame_id)
                self._trackers[eid] = ts
                events[eid] = "ENTER"

        return events

    def state_of(self, entity_id: str) -> str:
        """查询某实体当前生命周期状态."""
        ts = self._trackers.get(entity_id)
        return ts.state if ts else "FORGET"

    def reset(self) -> None:
        """清空所有状态."""
        self._trackers.clear()

    def to_dict(self) -> dict:
        """用于 checkpoint 序列化."""
        return {
            "hysteresis": self._hysteresis,
            "forget": self._forget,
            "trackers": {
                eid: {
                    "state": ts.state,
                    "consecutive_absent": ts.consecutive_absent,
                    "enter_frame": ts.enter_frame,
                    "last_seen_frame": ts.last_seen_frame,
                    "exit_frame": ts.exit_frame,
                }
                for eid, ts in self._trackers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> LifecycleTracker:
        """从 to_dict() 恢复状态."""
        lc = cls(
            hysteresis_frames=data.get("hysteresis", 3),
            forget_frames=data.get("forget", 30),
        )
        for eid, sd in data.get("trackers", {}).items():
            lc._trackers[eid] = _EntityState(
                state=sd.get("state", "FORGET"),
                consecutive_absent=sd.get("consecutive_absent", 0),
                enter_frame=sd.get("enter_frame", 0),
                last_seen_frame=sd.get("last_seen_frame", 0),
                exit_frame=sd.get("exit_frame", 0),
            )
        return lc

    def stats(self) -> dict:
        """返回当前内部状态统计 (用于监控)."""
        counts = {"ENTER": 0, "UPDATE": 0, "EXIT": 0, "FORGET": 0}
        for ts in self._trackers.values():
            counts[ts.state] = counts.get(ts.state, 0) + 1
        return {
            "n_tracked": len(self._trackers),
            "state_counts": counts,
            "hysteresis_frames": self._hysteresis,
            "forget_frames": self._forget,
        }
