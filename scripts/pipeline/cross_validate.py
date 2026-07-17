#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cross_validate.py -- 地图交叉验证 (进程级隔离修复版)

修复说明 (诊断根因: CARLA 0.9.16 在 server 内仍有 actor stream 时调
load_world 会触发 UE4 SIGSEGV):

- 严禁"同一 CARLA server 进程内二次 load_world 切图"路径
- 每张 town 走独立端口 + 独立 cold-boot CARLA server:
    1. kill 当前端口上的 CARLA (SIGTERM grace -> SIGKILL)
    2. 等端口真正释放
    3. cold-boot 全新 CARLA server (新 PID, 新 actor stream)
    4. health check (server_version + get_world().get_map() 双关)
    5. 子进程跑 run_phases_1_5.py 在已经加载 town 的 CARLA 上 (无需 load_world)
       -- town 已通过 INI 配置在 cold-boot 时直接加载
    6. run 完该 town 后再 kill 该端口的 CARLA (为下一张 town 让路)

Py3.13 兼容:
  stk 项目用 conda `stk` env (py3.10) 安装; venv 是 py3.13 没有 carla wheel.
  本脚本不再用 sys.executable 跑 run_phases_1_5.py, 而是通过 process_manager
  找到能 import carla 的 python 解释器. 本脚本自己也不需 import carla.

输出:
    data/runs/cross_validation/compare_pm_<timestamp>.json
    data/runs/cross_validation/town_<town>_<timestamp>_<f>f/
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "carla"))
from process_manager import (  # noqa: E402
    restart_carla_on_port, restart_carla_with_map,
    kill_carla_on_port, health_check, find_carla_python,
)

TOWNS = ["Town10HD", "Town01", "Town05"]   # 3 张可加载地图交叉验证
DEFAULT_PORT_BASE = 2300


def run_town(town: str, port: int, frames: int, vehicles: int,
             walkers: int, out_root: Path, graphics_adapter: int,
             carla_python: str) -> dict:
    """一张 town 的完整 cycle: kill -> cold-boot -> health -> 子进程 run_phases -> kill"""
    cycle_start_ts = datetime.now().timestamp()
    print(f"\n{'='*70}")
    print(f"[cycle] {town}: port={port}, frames={frames}, veh={vehicles}, ped={walkers}")
    print(f"        output scan root: {out_root}")
    print(f"        carla_python={carla_python}")

    # 1) 进程级 cold-boot, 直接以目标 town 启动 CARLA (无需 load_world)
    print(f"[*] cold-boot fresh CARLA on port {port} with map {town} ...")
    ok = restart_carla_with_map(
        port,
        town,
        host="localhost",
        cold_boot_timeout=120.0,
        graphics_adapter=graphics_adapter,
        quality_level="Low",
        log_path=f"/home/aisecurity/carla_cv_{port}_{town}.log",
        python=carla_python,
    )
    if not ok:
        print(f"[FATAL] cold-boot failed on {port} for {town}")
        return {"town": town, "status": "COLD_BOOT_FAIL", "port": port}

    if not health_check("localhost", port, timeout=10.0, python=carla_python):
        print(f"[FATAL] post-startup health check failed on {port}")
        kill_carla_on_port(port, grace_seconds=2.0)
        return {"town": town, "status": "HEALTH_FAIL", "port": port}

    # 2) 子进程跑 5 阶段管线. 用 carla_python (stk env py3.10), 不是 sys.executable
    proc = subprocess.run(
        [
            carla_python,
            str(_REPO / "scripts" / "pipeline" / "run_phases_1_5.py"),
            "--host", "localhost",
            "--port", str(port),
            "--carla-port", str(port),
            "--frames", str(frames),
            "--vehicles", str(vehicles),
            "--walkers", str(walkers),
            # town is already loaded via server INI; do not call load_world.
            "--out", "data/runs/cross_validation",
        ],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(graphics_adapter)},
    )

    stdout = (proc.stdout or "") + (proc.stderr or "")
    summary = {"town": town, "status": "UNKNOWN", "port": port}
    for line in stdout.splitlines():
        if "KG summary:" in line:
            try:
                import re
                m = re.search(r"KG summary:\s*(\{.*\})", line)
                if m:
                    summary = json.loads(m.group(1))
                    summary["town"] = town
                    summary["port"] = port
            except Exception:
                pass
        if "All 5 phases passed" in line:
            summary["status"] = "PASS"
    if summary.get("status") != "PASS":
        summary["stdout_tail"] = stdout[-2000:]

    # 3) 不管成败, kill 该端口的 CARLA, 为下一张 town 让路 (彻底隔离)
    print(f"[*] kill CARLA on port {port} to isolate next town ...")
    kill_carla_on_port(port, grace_seconds=3.0)

    # 4) Locate the phases_*_<frames>f dir that the subprocess wrote to.
    #    run_phases_1_5.py creates its own dir with its own timestamp; find the
    #    newest phases_*_<frames>f that was created during this town cycle.
    print(f"    [*] scanning out_root for newest phases_*_{frames}f since cycle start ...")
    sub_dirs = []
    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not (name.startswith("phases_") and name.endswith(f"_{frames}f")):
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        # Only consider dirs created after cycle_start_ts (allow -3s slack)
        if mtime >= cycle_start_ts - 3:
            sub_dirs.append((mtime, d))
    graph_obj = None
    if sub_dirs:
        sub_dirs.sort()
        latest = sub_dirs[-1][1]
        summary["subprocess_out_dir"] = str(latest.name)
        gp = latest / "phase5_graph.json"
        if gp.exists():
            with open(gp) as f:
                graph_obj = json.load(f)
            print(f"    [+] found graph at {gp}")
        else:
            print(f"    [!] found {latest.name} but no phase5_graph.json")
    else:
        print(f"    [!] no phases_*_{frames}f directory created after cycle start")
    if graph_obj:
        summary["graph_nodes"] = len(graph_obj["nodes"])
        summary["graph_edges"] = len(graph_obj["edges"])
        summary["node_types"] = dict(Counter(n["type"] for n in graph_obj["nodes"]))
        summary["edge_types"] = dict(Counter(e["type"] for e in graph_obj["edges"]))
    else:
        print(f"    [!] phase5_graph.json not found anywhere after cycle start")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--vehicles", type=int, default=8)
    p.add_argument("--walkers", type=int, default=3)
    p.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    p.add_argument("--towns", default=",".join(TOWNS))
    p.add_argument("--graphics-adapter", type=int, default=3)
    p.add_argument("--out", default="data/runs/cross_validation")
    args = p.parse_args()

    towns = [t.strip() for t in args.towns.split(",") if t.strip()]
    out_root = _REPO / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # 探测能 import carla 的 python 解释器 (一次就好)
    print(f"[*] locating carla-capable python interpreter ...")
    try:
        carla_python = find_carla_python()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"[meta] CUDA_VISIBLE_DEVICES={args.graphics_adapter}  towns={towns}  port_base={args.port_base}")
    print(f"[meta] carla_python={carla_python}")
    print(f"[meta] output: {out_root}")

    results = []
    for i, town in enumerate(towns):
        port = args.port_base + i
        r = run_town(town, port, args.frames, args.vehicles,
                      args.walkers, out_root, args.graphics_adapter, carla_python)
        results.append(r)

    # ===== 对比表 =====
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    hdrs = ["Town", "Port", "Nodes", "Edges", "SV", "RA", "Maneuver", "Interact", "Status"]
    print(" ".join(f"{h:<10}" for h in hdrs))
    print("-" * 100)
    table = {}
    for r in results:
        t = r.get("town", "?")
        nt = r.get("node_types", {})
        row = {
            "port": r.get("port", "?"),
            "nodes": r.get("graph_nodes", "?"),
            "edges": r.get("graph_edges", "?"),
            "SafetyViolation": nt.get("SafetyViolation", "?"),
            "ResponsibilityAssignment": nt.get("ResponsibilityAssignment", "?"),
            "Maneuver": nt.get("Maneuver", "?"),
            "InteractionEvent": nt.get("InteractionEvent", "?"),
            "status": r.get("status", "?"),
        }
        table[t] = row
        print(f"{t:<10} {row['port']!s:<10} {row['nodes']!s:<8} {row['edges']!s:<8} "
              f"{row['SafetyViolation']!s:<6} {row['ResponsibilityAssignment']!s:<6} "
              f"{row['Maneuver']!s:<10} {row['InteractionEvent']!s:<10} {row['status']!s:<10}")

    compare = {
        "timestamp": datetime.now().isoformat(),
        "frames": args.frames, "vehicles": args.vehicles, "walkers": args.walkers,
        "port_base": args.port_base,
        "graphics_adapter": args.graphics_adapter,
        "carla_python": carla_python,
        "results": results,
        "table": table,
    }
    cmp_path = out_root / f"compare_pm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Compare saved: {cmp_path}")

    # 最后再保险: 把所有用过的端口都收掉
    for i, _ in enumerate(towns):
        kill_carla_on_port(args.port_base + i, grace_seconds=1.0)


if __name__ == "__main__":
    main()
