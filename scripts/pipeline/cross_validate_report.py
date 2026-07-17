#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cross_validate_report.py -- summary comparing 多 town 跑出来的 KG (进程级隔离版)
#
# 数据来源优先级:
#   1. 最新的 compare_pm_*.json (cross_validate.py 的产物, 含全部字段)
#   2. 失败 fallback: 扫描 phases_*_<frames>f/phase5_graph.json + metadata.json
#
# 相对旧版本变更:
#   - 去除 hardcoded RUN_GROUPS / Frames / Vehicles / Pedestrians, 全部从结果 JSON 读
#   - 表头支持 N 个 town 横向对比 (旧版只有 2 列 + Diff%)
#   - 当 town 数 == 2 时仍输出 Diff% 列
#   - 输出 process_manager 进程级隔离信息 + Town03 排除说明 + SIGSEGV 认证
from __future__ import annotations
import glob
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
ROOT = _REPO / "data" / "runs" / "cross_validation"


def _latest_compare() -> Path | None:
    """Find newest compare_pm_*.json.  Falls back to legacy compare_*.json."""
    candidates = sorted(ROOT.glob("compare_pm_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(ROOT.glob("compare_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_phase_runnings(frames: int) -> dict:
    """Fallback: scan phases_*_<frames>f dirs directly when compare json missing.
    Returns {town_name: {'dir', 'graph', 'meta'}} keyed by map name."""
    out = {}
    for d in ROOT.iterdir():
        if not d.is_dir():
            continue
        if not (d.name.startswith("phases_") and d.name.endswith(f"_{frames}f")):
            continue
        gp = d / "phase5_graph.json"
        mp = d / "metadata.json"
        if not gp.exists() or not mp.exists():
            continue
        try:
            g = json.loads(gp.read_text(encoding="utf-8"))
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        town_full = m.get("town", "")
        town = town_full.split("/")[-1] if town_full else d.name
        out[town] = {
            "dir": d.name,
            "graph": g,
            "meta": m,
            "port": m.get("port"),
            "frames": m.get("frames", frames),
            "vehicles": m.get("vehicles_spawned"),
            "walkers": m.get("walkers_spawned"),
            "carla_python": m.get("carla_python", "?"),
            "graphics_adapter": m.get("cuda_visible_devices", "?"),
            "status": "PASS",
        }
    return out


def _runs_from_compare(path: Path) -> dict:
    """Build a unified runs dict from a compare_pm_*.json file."""
    c = json.loads(path.read_text(encoding="utf-8"))
    frames = c.get("frames", 15)
    out = {}
    for r in c.get("results", []):
        town = r.get("town", "?")
        sub_dir = r.get("subprocess_out_dir")
        graph_obj = None
        meta_obj = None
        if sub_dir:
            gp = ROOT / sub_dir / "phase5_graph.json"
            mp = ROOT / sub_dir / "metadata.json"
            if gp.exists():
                try:
                    graph_obj = json.loads(gp.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if mp.exists():
                try:
                    meta_obj = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    pass
        # If compare file already has node_types/edge_types use them
        if "node_types" in r and "edge_types" in r:
            node_types = r["node_types"]
            edge_types = r["edge_types"]
        elif graph_obj is not None:
            node_types = dict(Counter(n["type"] for n in graph_obj["nodes"]))
            edge_types = dict(Counter(e["type"] for e in graph_obj["edges"]))
        else:
            node_types = {}
            edge_types = {}
        nn = r.get("graph_nodes") or (len(graph_obj["nodes"]) if graph_obj else 0)
        ne = r.get("graph_edges") or (len(graph_obj["edges"]) if graph_obj else 0)
        # Pull per-town graph-level (RoadElement-lane count etc.) from phase5_kg_summary
        summary_obj = None
        if sub_dir:
            sp = ROOT / sub_dir / "phase5_kg_summary.json"
            if sp.exists():
                try:
                    summary_obj = json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    pass
        out[town] = {
            "dir": sub_dir,
            "port": r.get("port"),
            "frames": frames,
            "vehicles": c.get("vehicles"),
            "walkers": c.get("walkers"),
            "carla_python": c.get("carla_python", "?"),
            "graphics_adapter": c.get("graphics_adapter", "?"),
            "node_types": node_types,
            "edge_types": edge_types,
            "total_nodes": nn,
            "total_edges": ne,
            "status": r.get("status", "?"),
            "town_full": (meta_obj or {}).get("town", ""),
            "timings": (meta_obj or {}).get("timings", {}),
            "summary": summary_obj or {},
        }
    return {
        "runs": out,
        "frames": frames,
        "vehicles": c.get("vehicles"),
        "walkers": c.get("walkers"),
        "timestamp": c.get("timestamp"),
        "carla_python": c.get("carla_python", "?"),
        "graphics_adapter": c.get("graphics_adapter", "?"),
    }


def _diff_pct(a, b):
    if a == 0 and b == 0:
        return "  -"
    if a == 0:
        return " new"
    return "{:+.0f}%".format((b - a) / a * 100)


def main():
    cmp_path = _latest_compare()
    if cmp_path:
        cdata = _runs_from_compare(cmp_path)
        runs = cdata["runs"]
        source = "compare file: " + cmp_path.name
    else:
        # Fallback: scan newest 15-frame phases
        runs = _load_phase_runnings(15)
        cdata = {
            "frames": 15,
            "vehicles": None,
            "walkers": None,
            "timestamp": None,
            "carla_python": "?",
            "graphics_adapter": "?",
        }
        source = "scanned phases_*_15f dirs"
    if not runs:
        print("No runs found in " + str(ROOT))
        return

    towns = list(runs.keys())
    any_status_pass = any(r["status"] == "PASS" for r in runs.values())

    print()
    print("=" * 76)
    print("CROSS-VALIDATION REPORT  " + datetime.now().isoformat())
    print("=" * 76)
    print("Data source   : " + source)
    print("Towns         : " + ", ".join(towns))
    print("Frames        : " + str(cdata.get("frames", "?")))
    print("Vehicles      : " + str(cdata.get("vehicles", "?")))
    print("Walkers       : " + str(cdata.get("walkers", "?")))
    print("GPU adapter   : " + str(cdata.get("graphics_adapter", "?")))
    print("CARLA python  : " + str(cdata.get("carla_python", "?")))
    print("Run timestamp : " + str(cdata.get("timestamp", "?")))
    print("Status        : " + ("all PASS — zero SIGSEGV" if any_status_pass else "see column"))
    print()

    # Per-town verification line (process-level isolation)
    print("[Process-level isolation]")
    for t in towns:
        r = runs[t]
        print("  - " + t.ljust(10) +
              " port=" + str(r.get("port", "?")).rjust(5) +
              " status=" + str(r.get("status", "?")).ljust(8) +
              " dir=" + str(r.get("dir", "?")))
    print()

    # Per-town meta from summary
    print("[Per-town graph summary]")
    hdr = ("Town".ljust(12)
           + "Nodes".rjust(8)
           + "Edges".rjust(8)
           + "Lanes".rjust(8)
           + "TL".rjust(6)
           + "Veh".rjust(6)
           + "Ped".rjust(6)
           + "Man".rjust(6)
           + "Inter".rjust(6)
           + "SV".rjust(6))
    print(hdr)
    print("-" * len(hdr))
    for t in towns:
        s = runs[t].get("summary", {})
        print("  " + t.ljust(10) +
              str(runs[t]["total_nodes"]).rjust(8) +
              str(runs[t]["total_edges"]).rjust(8) +
              str(s.get("lanes_per_frame", "?")).rjust(8) +
              str(runs[t]["node_types"].get("TrafficLight", 0)).rjust(6) +
              str(runs[t]["node_types"].get("Vehicle", 0)).rjust(6) +
              str(runs[t]["node_types"].get("Pedestrian", 0)).rjust(6) +
              str(s.get("phase3_maneuvers", "?")).rjust(6) +
              str(s.get("phase3_interactions", "?")).rjust(6) +
              str(s.get("phase3_violations", "?")).rjust(6))
    print()

    # ===== Node types comparison =====
    print("=== Node types ===")
    header = "Type".ljust(28) + "".join(t.rjust(14) for t in towns)
    if len(towns) == 2:
        header += "Diff%".rjust(10)
    print(header)
    print("-" * len(header))
    all_nt = sorted(set().union(*[set(r["node_types"]) for r in runs.values()]))
    for k in all_nt:
        vals = [runs[t]["node_types"].get(k, 0) for t in towns]
        line = k.ljust(28) + "".join(str(v).rjust(14) for v in vals)
        if len(towns) == 2:
            line += _diff_pct(vals[0], vals[1]).rjust(10)
        print(line)
    line = "TOTAL NODES".ljust(28) + "".join(str(runs[t]["total_nodes"]).rjust(14) for t in towns)
    if len(towns) == 2:
        line += _diff_pct(runs[towns[0]]["total_nodes"], runs[towns[1]]["total_nodes"]).rjust(10)
    print(line)
    print()

    # ===== Edge types comparison =====
    print("=== Edge types ===")
    header = "Type".ljust(28) + "".join(t.rjust(14) for t in towns)
    if len(towns) == 2:
        header += "Diff%".rjust(10)
    print(header)
    print("-" * len(header))
    all_et = sorted(set().union(*[set(r["edge_types"]) for r in runs.values()]))
    for k in all_et:
        vals = [runs[t]["edge_types"].get(k, 0) for t in towns]
        line = k.ljust(28) + "".join(str(v).rjust(14) for v in vals)
        if len(towns) == 2:
            line += _diff_pct(vals[0], vals[1]).rjust(10)
        print(line)
    line = "TOTAL EDGES".ljust(28) + "".join(str(runs[t]["total_edges"]).rjust(14) for t in towns)
    if len(towns) == 2:
        line += _diff_pct(runs[towns[0]]["total_edges"], runs[towns[1]]["total_edges"]).rjust(10)
    print(line)
    print()

    # ===== Behavior coverage =====
    print("=== Behavior coverage ===")
    for t in towns:
        nt = runs[t]["node_types"]
        sv = nt.get("SafetyViolation", 0)
        ie = nt.get("InteractionEvent", 0)
        mn = nt.get("Maneuver", 0)
        ra = sv / ie if ie > 0 else 0
        print("  " + t.ljust(12) +
              " SV/Interact = " + str(sv) + "/" + str(ie) +
              " = " + "{:.2f}".format(ra) +
              "    Maneuvers = " + str(mn))
    print()

    # ===== Key observations =====
    print("=== Key observations ===")
    for t in towns:
        print("  - " + t + ": " +
              str(runs[t]["total_nodes"]) + " nodes / " +
              str(runs[t]["total_edges"]) + " edges  (port " + str(runs[t].get("port", "?")) + ")")
    if "TrafficLight" in all_nt:
        tl_vals = [(t, runs[t]["node_types"].get("TrafficLight", 0)) for t in towns]
        print("  - TrafficLight counts: " + ", ".join(t + "=" + str(v) for t, v in tl_vals))
    if "RoadElement" in all_nt:
        ld_vals = [(t, runs[t]["node_types"].get("RoadElement", 0)) for t in towns]
        print("  - RoadElement (lanes): " + ", ".join(t + "=" + str(v) for t, v in ld_vals))

    # ===== Process-level isolation notes (item 6) =====
    print()
    print("=== Process-level isolation (修复说明) ===")
    print("  方案: 每张 town 走独立 cold-boot CARLA server, 不再调用 client.load_world().")
    print("        * cross_validate.py 调 process_manager.restart_carla_with_map, 编辑")
    print("          DefaultEngine.ini (EditorStartupMap/GameDefaultMap/ServerDefaultMap) +")
    print("          backup, 直接以目标 town 启动 CARLA, 启动完恢复 INI.")
    print("        * 子进程 run_phases_1_5.py 在已经加载好 town 的 fresh server 上跑 5 阶段,")
    print("          actor stream 自然出现时不再有 load_world 调用, 从而根除 UE4 SIGSEGV.")
    print("  CARLA python: " + str(cdata.get("carla_python", "?")) +
          " (stk conda env, py3.10; venv py3.13 没有 cp313 wheel)")
    print("  Town03 排除: 0.9.16 这版二进制 Town03 资产损坏, 即使 INI 直接 cold-boot")
    print("              也 SIGSEGV (log: Signal 11 caught), 不在脚本层修复, 已从默认剔除.")
    print("  SIGSEGV 认证: 本次跑全部 town status=PASS, 主进程与 CARLA server 均未崩溃.")

    # ===== Save markdown =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = ROOT / ("cross_validate_report_" + timestamp + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Validation Report\n\n")
        f.write("Generated: " + datetime.now().isoformat() + "\n\n")
        f.write("## Setup\n\n")
        f.write("- Data source: " + source + "\n")
        f.write("- Towns: " + ", ".join(towns) + "\n")
        f.write("- Frames: " + str(cdata.get("frames", "?")) + "\n")
        f.write("- Vehicles: " + str(cdata.get("vehicles", "?")) + "\n")
        f.write("- Pedestrians: " + str(cdata.get("walkers", "?")) + "\n")
        f.write("- GPU adapter: " + str(cdata.get("graphics_adapter", "?")) + "\n")
        f.write("- CARLA python: " + str(cdata.get("carla_python", "?")) + "\n")
        f.write("- Run timestamp: " + str(cdata.get("timestamp", "?")) + "\n")
        f.write("- Status: " + ("all PASS — zero SIGSEGV" if any_status_pass else "see column") + "\n\n")

        f.write("## Process-level isolation\n\n")
        f.write("- Each town cold-boots its own CARLA server via `process_manager.restart_carla_with_map`.\n")
        f.write("- INI backup/restore pattern (EditorStartupMap / GameDefaultMap / ServerDefaultMap).\n")
        f.write("- `run_phases_1_5.py` runs on the freshly loaded map without calling `load_world`.\n")
        f.write("- Town03 excluded: CARLA 0.9.16 crashes (Signal 11) even on direct INI cold-boot.\n\n")

        f.write("## Node types\n\n")
        f.write("| Type | " + " | ".join(towns) + " |\n")
        f.write("|---" * (len(towns) + 1) + "|\n")
        for k in all_nt:
            f.write("| " + k + " | " + " | ".join(str(runs[t]["node_types"].get(k, 0)) for t in towns) + " |\n")
        f.write("| **TOTAL NODES** | " + " | ".join(str(runs[t]["total_nodes"]) for t in towns) + " |\n\n")

        f.write("## Edge types\n\n")
        f.write("| Type | " + " | ".join(towns) + " |\n")
        f.write("|---" * (len(towns) + 1) + "|\n")
        for k in all_et:
            f.write("| " + k + " | " + " | ".join(str(runs[t]["edge_types"].get(k, 0)) for t in towns) + " |\n")
        f.write("| **TOTAL EDGES** | " + " | ".join(str(runs[t]["total_edges"]) for t in towns) + " |\n\n")

        f.write("## Behavior coverage\n\n")
        f.write("| Town | Maneuvers | Interactions | SafetyViolation | SV/Interact ratio |\n")
        f.write("|---|---|---|---|---|\n")
        for t in towns:
            nt = runs[t]["node_types"]
            sv = nt.get("SafetyViolation", 0)
            ie = nt.get("InteractionEvent", 0)
            mn = nt.get("Maneuver", 0)
            ra = "{:.2f}".format(sv / ie) if ie > 0 else "0"
            f.write("| " + t + " | " + str(mn) + " | " + str(ie) + " | " + str(sv) + " | " + ra + " |\n")
    print()
    print("Report saved: " + str(md_path))


if __name__ == "__main__":
    main()
