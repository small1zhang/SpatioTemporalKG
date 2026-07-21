# -*- coding: utf-8 -*-
"""单帧增量更新主流程 (v3 §5.5.1)."""
from __future__ import annotations
import json
import sys
from typing import Any, Dict, List, Optional
from stk.dynamic.diff import DeltaGraph, compute_delta


# 数值属性白名单 (任何不合规即视为字符串污染, 拒绝恢复)
_NUMERIC_ATTRS = ("location_x", "location_y", "location_z",
                  "velocity_x", "velocity_y", "velocity_z",
                  "speed", "speed_kmh", "heading_rad", "heading_deg",
                  "brake", "throttle", "steer")


def _validate_numeric_attrs(frame: dict) -> bool:
    """扫描 prev_frame 中所有实体的数值属性, 任何字符串/None 污染返回 False.

    防御 `json.dumps(prev, default=str)` 把不可序列化对象转字符串后污染 checkpoint,
    导致 resume 后 compute_delta 误判全字段变更 (字符串 vs 数字永远不会相等).
    """
    if not isinstance(frame, dict):
        return False
    for entity in frame.get("vehicles", []) + frame.get("pedestrians", []):
        if not isinstance(entity, dict):
            return False
        for k in _NUMERIC_ATTRS:
            v = entity.get(k)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False
    return True


class IncrementalEngine:
    """增量引擎 — 5步: recv → diff → patch → eval → writeback.

    维护 G_prev: 保存每帧的原始 dict 快照 (含 vehicles/pedestrians/scene_rels/behavior_rels).
    """

    def __init__(self):
        self._prev_frame: Optional[dict] = None
        self._delta_history: List[DeltaGraph] = []

    def process_frame(self, frame: dict) -> DeltaGraph:
        """输入帧 snapshot, 输出 Δg_t.

        修复:
          - 跳帧检测: 当 prev.frame_id 与 curr.frame_id 不连续时, 重置 baseline,
            让 compute_delta 按首帧处理 (全 added), 避免跨 chunk 残差污染.
        """
        if self._prev_frame is not None:
            prev_fid = self._prev_frame.get("frame_id") if isinstance(self._prev_frame, dict) else None
            curr_fid = frame.get("frame_id") if isinstance(frame, dict) else None
            if prev_fid is not None and curr_fid is not None and curr_fid != prev_fid + 1:
                print(f"[dynamic] frame jump detected: prev={prev_fid} curr={curr_fid}, "
                      f"resetting baseline to avoid false deltas", file=sys.stderr)
                self._prev_frame = None
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

        修复:
          - 严格 json.dumps (不使用 default=str), 任何不可序列化字段直接判废;
          - _validate_numeric_attrs 二次校验, 拒绝字符串数字污染.
        """
        prev = self._prev_frame
        if prev is not None:
            try:
                json.dumps(prev)  # 严格模式, 不允许 default=str
            except (TypeError, ValueError):
                prev = None
            else:
                if not _validate_numeric_attrs(prev):
                    prev = None
        return {
            "prev_frame": prev,
            "n_deltas": len(self._delta_history),
            "last_processed_frame": prev.get("frame_id", -1) if isinstance(prev, dict) else -1,
        }

    def load_dict(self, data: dict) -> None:
        """从 to_dict() 恢复引擎状态.

        修复:
          - 数值完整性校验, 防御字符串污染型恢复 (resume 后伪变化);
          - 占位 `pass` 替换为实际校验逻辑.
        """
        self._prev_frame = data.get("prev_frame", None)
        if self._prev_frame is not None:
            if not _validate_numeric_attrs(self._prev_frame):
                print("[dynamic] WARNING: prev_frame numeric attrs corrupted, "
                      "resetting baseline to avoid false deltas", file=sys.stderr)
                self._prev_frame = None
        # delta_history 不恢复 (仅做统计), 从 checkpoint 恢复后视为无历史增量
        self._delta_history.clear()
