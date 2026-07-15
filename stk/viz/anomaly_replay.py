# -*- coding: utf-8 -*-
"""异常事件可视化: 输出 SafetyViolation 证据链 PNG (v3 §6.4)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from stk.storage.replay import replay_violation


def plot_anomaly_trace(sv_id: str, output_path: str = "anomaly_trace.png") -> str:
    """输出异常证据链的可视化 (占位 — 输出 Cypher 和基本信息)."""
    result = replay_violation(sv_id)
    lines = [
        f"SafetyViolation: {result.get('sv_id','')}",
        f"Cypher: {result.get('cypher','')}",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path