# -*- coding: utf-8 -*-
"""时间窗口聚合 (v3 §5.2.2)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SummaryEvent:
    window_start: int = 0
    window_end: int = 0
    violation_count: int = 0
    max_severity: float = 0.0
    rule_codes: Set[str] = field(default_factory=set)
    involved_actors: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {"window_start": self.window_start, "window_end": self.window_end,
                "violation_count": self.violation_count, "max_severity": self.max_severity,
                "rule_codes": list(self.rule_codes),
                "involved_actors": list(self.involved_actors)}


class TimeWindowAggregator:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._violations: List[dict] = []
        self._fids: List[int] = []

    def add(self, frame_id: int, violations: List[dict]):
        self._fids.append(frame_id)
        self._violations.extend(violations)

    def summarize(self, start: Optional[int] = None, end: Optional[int] = None) -> SummaryEvent:
        if start is None:
            start = max(0, max(self._fids) - self.window_size + 1) if self._fids else 0
        if end is None:
            end = max(self._fids) if self._fids else 0
        filtered = [v for v in self._violations if start <= v.get("frame_id", 0) <= end]
        sevs = [v.get("severity", 0.0) for v in filtered]
        return SummaryEvent(
            window_start=start, window_end=end,
            violation_count=len(filtered),
            max_severity=max(sevs) if sevs else 0.0,
            rule_codes={v.get("rule_code", "") for v in filtered},
            involved_actors={v.get("src_id", "") for v in filtered} |
                           {v.get("dst_id", "") for v in filtered},
        )

    def clear(self):
        self._violations.clear()
        self._fids.clear()