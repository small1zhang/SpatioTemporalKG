# -*- coding: utf-8 -*-
"""异常回放 CLI (v3 §6.4.2)."""
from __future__ import annotations
from typing import Any, Dict, Optional
from stk.storage.queries import anomaly_trace_query, temporal_attr_query


def replay_violation(sv_id: str) -> Dict[str, Any]:
    cypher = anomaly_trace_query(sv_id)
    return {"sv_id": sv_id, "cypher": cypher, "nodes": {}, "edges": []}


def format_replay_output(result: Dict[str, Any]) -> str:
    lines = [f"SafetyViolation: {result.get('sv_id','?')}",
             f"Cypher: {result.get('cypher','')}"]
    if result.get("nodes"):
        lines.append(f"Nodes: {len(result['nodes'])}")
    if result.get("edges"):
        lines.append(f"Edges: {len(result['edges'])}")
    return "\n".join(lines)