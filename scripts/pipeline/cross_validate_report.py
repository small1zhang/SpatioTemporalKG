#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cross_validate_report.py -- summary comparing Town01 vs Town10HD
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
ROOT = _REPO / "data" / "runs" / "cross_validation"
RUN_GROUPS = {"Town01":"phases_20260717_103147_15f","Town10HD":"phases_20260717_105106_15f"}

def load_graph(p):
    g = p/"phase5_graph.json"
    return json.loads(g.read_text()) if g.exists() else None

def stat(g):
    return dict(Counter(n["type"] for n in g["nodes"])), dict(Counter(e["type"] for e in g["edges"])), len(g["nodes"]), len(g["edges"])

def main():
    runs = {}
    for t,d in RUN_GROUPS.items():
        g = load_graph(ROOT/d)
        if g:
            nt,et,nn,ne = stat(g)
            runs[t] = {"node_types":nt,"edge_types":et,"total_nodes":nn,"total_edges":ne,"dir":d}
    if not runs:
        print("No runs found."); return
    print()
    print("=" * 70)
    print("CROSS-VALIDATION REPORT " + datetime.now().isoformat())
    print("=" * 70)
    towns = list(runs.keys())
    print("Type".ljust(22) + "".join(x.rjust(12) for x in towns) + "Diff %".rjust(12))
    print("-" * 70)
    print("\n--- Node types ---")
    all_nt = sorted(set().union(*[set(r["node_types"]) for r in runs.values()]))
    for k in all_nt:
        vals = [runs[t]["node_types"].get(k, 0) for t in towns]
        diff = "-"
        if len(vals) == 2 and vals[0] > 0:
            diff = "{:+.0f}%".format((vals[1] - vals[0]) / vals[0] * 100)
        print(k.ljust(22) + "".join(str(v).rjust(12) for v in vals) + diff.rjust(12))
    print("TOTAL NODES".ljust(22) + "".join(str(runs[t]["total_nodes"]).rjust(12) for t in towns))
    print("\n--- Edge types ---")
    all_et = sorted(set().union(*[set(r["edge_types"]) for r in runs.values()]))
    for k in all_et:
        vals = [runs[t]["edge_types"].get(k, 0) for t in towns]
        diff = "-"
        if len(vals) == 2 and vals[0] > 0:
            diff = "{:+.0f}%".format((vals[1] - vals[0]) / vals[0] * 100)
        print(k.ljust(22) + "".join(str(v).rjust(12) for v in vals) + diff.rjust(12))
    print("TOTAL EDGES".ljust(22) + "".join(str(runs[t]["total_edges"]).rjust(12) for t in towns))
    print("\n--- Behavior coverage ---")
    for t in towns:
        nt = runs[t]["node_types"]
        v = nt.get("SafetyViolation", 0)
        i = nt.get("InteractionEvent", 1)
        r = v / i if i > 0 else 0
        print(t.ljust(22) + "violations/interactions = " + str(v) + "/" + str(i) + " = {:.2f}".format(r))
    print("\nKey observations:")
    if len(towns) == 2:
        t1, t2 = towns
        print("- " + t1 + ": " + str(runs[t1]["total_nodes"]) + " nodes / " + str(runs[t1]["total_edges"]) + " edges")
        print("- " + t2 + ": " + str(runs[t2]["total_nodes"]) + " nodes / " + str(runs[t2]["total_edges"]) + " edges")
        tl1 = runs[t1]["node_types"].get("TrafficLight", 0)
        tl2 = runs[t2]["node_types"].get("TrafficLight", 0)
        if tl1 != tl2:
            print("- TrafficLight: " + str(tl1) + " vs " + str(tl2))
        ld1 = runs[t1]["node_types"].get("RoadElement", 0)
        ld2 = runs[t2]["node_types"].get("RoadElement", 0)
        print("- RoadElement (lanes): " + str(ld1) + " vs " + str(ld2))
        sv1 = runs[t1]["node_types"].get("SafetyViolation", 0)
        sv2 = runs[t2]["node_types"].get("SafetyViolation", 0)
        print("- SafetyViolation: " + str(sv1) + " vs " + str(sv2))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = ROOT / ("cross_validate_report_" + timestamp + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Validation Report\n\nGenerated: " + datetime.now().isoformat() + "\n\n")
        f.write("## Setup\n\n- Frames: 15\n- Vehicles: 8\n- Pedestrians: 3\n\n")
        f.write("## Node types\n\n")
        f.write("| Type | " + " | ".join(towns) + " |\n")
        f.write("|---" + "|---" * len(towns) + "|\n")
        for k in all_nt:
            f.write("| " + k + " | " + " | ".join(str(runs[t]["node_types"].get(k, 0)) for t in towns) + " |\n")
        f.write("| **TOTAL NODES** | " + " | ".join(str(runs[t]["total_nodes"]) for t in towns) + " |\n")
        f.write("\n## Edge types\n\n")
        f.write("| Type | " + " | ".join(towns) + " |\n")
        f.write("|---" + "|---" * len(towns) + "|\n")
        for k in all_et:
            f.write("| " + k + " | " + " | ".join(str(runs[t]["edge_types"].get(k, 0)) for t in towns) + " |\n")
        f.write("| **TOTAL EDGES** | " + " | ".join(str(runs[t]["total_edges"]) for t in towns) + " |\n")
    print("\nReport saved: " + str(md_path))

if __name__ == "__main__":
    main()
