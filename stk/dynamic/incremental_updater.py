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