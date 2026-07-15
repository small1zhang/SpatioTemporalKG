# -*- coding: utf-8 -*-
"""帧快照存储查询 (v3 §5.5.2)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


class SnapshotStore:
    """按 frame_id 存储/取回帧快照 G_t."""

    def __init__(self):
        self._snapshots: Dict[int, Any] = {}

    def put(self, frame_id: int, snapshot: Any):
        self._snapshots[frame_id] = snapshot

    def get(self, frame_id: int) -> Optional[Any]:
        return self._snapshots.get(frame_id)

    def list_frame_ids(self) -> List[int]:
        return sorted(self._snapshots.keys())

    def count(self) -> int:
        return len(self._snapshots)

    def clear(self):
        self._snapshots.clear()