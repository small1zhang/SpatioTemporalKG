# -*- coding: utf-8 -*-
"""单帧增量更新主流程 (v3 §5.5.1)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from stk.dynamic.diff import DeltaGraph, compute_delta


class IncrementalEngine:
    """增量引擎 — 5步: recv → diff → patch → eval → writeback.

    维护 G_prev: 保存每帧的原始 dict 快照 (含 vehicles/pedestrians/scene_rels/behavior_rels).
    """

    def __init__(self):
        self._prev_frame: Optional[dict] = None
        self._delta_history: List[DeltaGraph] = []

    def process_frame(self, frame: dict) -> DeltaGraph:
        """输入帧 snapshot, 输出 Δg_t."""
        dg = compute_delta(frame, self._prev_frame)
        self._delta_history.append(dg)
        self._prev_frame = frame
        return dg

    @property
    def n_deltas(self) -> int:
        return len(self._delta_history)

    @property
    def delta_history(self) -> List[DeltaGraph]:
        return list(self._delta_history)

    def reset(self):
        self._prev_frame = None
        self._delta_history.clear()

    # ---------------- 序列化 (用于 checkpoint) ----------------

    def to_dict(self) -> dict:
        """导出增量引擎状态, 用于 checkpoint 持久化.

        序列化内容:
          - _prev_frame: 最新一帧的 dict (可 JSON 序列化)
          - delta_history 的统计摘要 (全量 list 过大, 不落盘)
        """
        # _prev_frame 中可能含不可 JSON 序列化的字段, 做清理
        import json
        prev = self._prev_frame
        if prev is not None:
            try:
                json.dumps(prev, default=str)
            except (TypeError, ValueError):
                prev = None  # 不可序列化则丢弃 (极少出现)
        return {
            "prev_frame": prev,
            "n_deltas": len(self._delta_history),
            "last_processed_frame": prev.get("frame_id", -1) if isinstance(prev, dict) else -1,
        }

    def load_dict(self, data: dict) -> None:
        """从 to_dict() 恢复引擎状态."""
        self._prev_frame = data.get("prev_frame", None)
        if self._prev_frame is not None:
            # 确保 frame_id 正确
            pass
        # delta_history 不恢复 (仅做统计), 从 checkpoint 恢复后视为无历史增量
        self._delta_history.clear()