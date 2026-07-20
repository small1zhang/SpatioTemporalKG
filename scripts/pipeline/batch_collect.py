#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_collect.py -- 批量采集调度器 (5地图 × 14场景 = 70任务)

特性:
    - 多 GPU 并行: 自动分配任务到 4 张 RTX 5090
    - 进程级隔离: 每个任务 cold-boot CARLA (避免 SIGSEGV)
    - 断点续传: 跳过已完成的任务
    - 结构化输出: data/runs/batch/<map>/<scenario>/
    - 实时进度和结果汇总

用法:
    # 跑全部 70 个任务
    python scripts/pipeline/batch_collect.py

    # 只跑特定地图
    python scripts/pipeline/batch_collect.py --maps Town10HD,Town01

    # 只跑特定场景
    python scripts/pipeline/batch_collect.py --scenarios S00,S10,S30

    # 自定义 GPU 列表和并行数
    python scripts/pipeline/batch_collect.py --gpus 1,2,3 --parallel 3

    # 从断点继续 (默认行为)
    python scripts/pipeline/batch_collect.py --resume
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import math
import random
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "carla"))
from process_manager import (
    restart_carla_with_map, kill_carla_on_port, health_check, find_carla_python,
)

# ─── 配置 ───────────────────────────────────────────────────────────
ALL_MAPS = ["Town10HD", "Town01", "Town02", "Town04", "Town05"]
ALL_SCENARIOS = [
    "S00", "S01", "S02",                     # A 基线
    "S10", "S11", "S12", "S13",              # B 风险
    "S20", "S21", "S22",                     # C 复杂
    "S30", "S31", "S32", "S33",              # D 环境
]
PORT_BASE = 3000   # 任务端口 = PORT_BASE + task_index
TASK_TIMEOUT = 600  # 每个任务最长 10 分钟

# 场景 -> 帧数/车辆/行人 配置
SCENARIO_PROFILES = {
    "S00": {"frames": 100, "vehicles": 3,  "walkers": 0},
    "S01": {"frames": 100, "vehicles": 5,  "walkers": 0},
    "S02": {"frames": 100, "vehicles": 3,  "walkers": 8},
    "S10": {"frames": 150, "vehicles": 3,  "walkers": 5},
    "S11": {"frames": 150, "vehicles": 6,  "walkers": 0},
    "S12": {"frames": 150, "vehicles": 5,  "walkers": 2},
    "S13": {"frames": 150, "vehicles": 8,  "walkers": 0},
    "S20": {"frames": 150, "vehicles": 10, "walkers": 0},
    "S21": {"frames": 150, "vehicles": 8,  "walkers": 0},
    "S22": {"frames": 150, "vehicles": 6,  "walkers": 2},
    "S30": {"frames": 150, "vehicles": 3,  "walkers": 5},
    "S31": {"frames": 150, "vehicles": 6,  "walkers": 0},
    "S32": {"frames": 150, "vehicles": 5,  "walkers": 0},
    "S33": {"frames": 150, "vehicles": 5,  "walkers": 6},
}


def load_map_config(map_name: str) -> dict:
    cfg_path = _REPO / "map_configs" / f"{map_name}.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run_single_task(task_id: int, map_name: str, scenario_id: str,
                    port: int, gpu: int, carla_python: str,
                    out_root: Path) -> dict:
    """单个采集任务: cold-boot -> run 5 phases -> kill"""
    task_start = datetime.now().timestamp()
    task_out = out_root / map_name / scenario_id
    task_out.mkdir(parents=True, exist_ok=True)

    map_cfg = load_map_config(map_name)
    yaml_params = map_cfg.get("scenario_parameters", {}).get(scenario_id, {})
    profile = SCENARIO_PROFILES.get(scenario_id, {"frames": 100, "vehicles": 5, "walkers": 2})

    result = {
        "task_id": task_id,
        "map": map_name,
        "scenario": scenario_id,
        "port": port,
        "gpu": gpu,
        "status": "UNKNOWN",
        "yaml_params": yaml_params,
    }

    # 1) Cold-boot
    log_path = task_out / f"carla_{port}.log"
    ok = restart_carla_with_map(
        port, map_name, host="localhost",
        cold_boot_timeout=120.0, graphics_adapter=gpu,
        quality_level="Low",
        log_path=str(log_path),
        python=carla_python,
    )
    if not ok:
        result["status"] = "COLD_BOOT_FAIL"
        result["elapsed_s"] = round(datetime.now().timestamp() - task_start, 1)
        return result

    if not health_check("localhost", port, timeout=10.0, python=carla_python):
        kill_carla_on_port(port, grace_seconds=2.0)
        result["status"] = "HEALTH_FAIL"
        result["elapsed_s"] = round(datetime.now().timestamp() - task_start, 1)
        return result

    # 2) Run pipeline
    frames = yaml_params.get("frames", profile["frames"])
    n_vehicles = yaml_params.get("vehicle_count", profile["vehicles"])
    n_walkers = yaml_params.get("pedestrian_count", profile["walkers"])

    try:
        proc = subprocess.run(
            [
                carla_python,
                str(_REPO / "scripts" / "pipeline" / "run_phases_1_5.py"),
                "--host", "localhost",
                "--port", str(port),
                "--carla-port", str(port),
                "--frames", str(frames),
                "--vehicles", str(n_vehicles),
                "--walkers", str(n_walkers),
                "--seed", "42",
                "--out", f"data/runs/batch/{map_name}/{scenario_id}",
            ],
            capture_output=True, text=True, timeout=TASK_TIMEOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        kill_carla_on_port(port, grace_seconds=3.0)
        result["elapsed_s"] = round(datetime.now().timestamp() - task_start, 1)
        return result

    stdout = (proc.stdout or "") + (proc.stderr or "")

    for line in stdout.splitlines():
        if "KG summary:" in line:
            try:
                import re
                m = re.search(r"KG summary:\s*(\{.*\})", line)
                if m:
                    summary = json.loads(m.group(1))
                    result.update(summary)
            except Exception:
                pass
        if "All 5 phases passed" in line:
            result["status"] = "PASS"

    # Read KG summary from output files
    import glob
    kg_files = sorted(
        out_root.glob(f"{map_name}/{scenario_id}/../../data/runs/batch/{map_name}/{scenario_id}/phases_*_{frames}f/phase5_kg_summary.json")
    )
    # Try the actual output path
    phases_root = _REPO / "data" / "runs" / "batch" / map_name / scenario_id
    kg_files = sorted(
        phases_root.glob("phases_*/phase5_kg_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if kg_files:
        with open(kg_files[-1]) as kf:
            kg = json.load(kf)
        result["graph_nodes"] = kg.get("phase5_graph_nodes")
        result["graph_edges"] = kg.get("phase5_graph_edges")
        result["node_types"] = kg.get("phase5_node_types")
        result["total_entities"] = kg.get("total_unique_entities")

    if result["status"] != "PASS":
        result["status"] = "FAIL"
        result["stdout_tail"] = stdout[-2000:]

    # 3) Kill CARLA
    kill_carla_on_port(port, grace_seconds=3.0)

    result["elapsed_s"] = round(datetime.now().timestamp() - task_start, 1)

    # 保存单任务结果
    with open(task_out / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


def build_task_list(maps: List[str], scenarios: List[str],
                    resume: bool, out_root: Path) -> List[Tuple[int, str, str]]:
    """构建任务列表, 可跳过已完成的"""
    tasks = []
    task_id = 0
    for map_name in maps:
        for scenario_id in scenarios:
            if resume:
                result_path = out_root / map_name / scenario_id / "result.json"
                if result_path.exists():
                    try:
                        with open(result_path) as f:
                            prev = json.load(f)
                        if prev.get("status") == "PASS":
                            task_id += 1
                            continue
                    except Exception:
                        pass
            tasks.append((task_id, map_name, scenario_id))
            task_id += 1
    return tasks


def main():
    p = argparse.ArgumentParser(description="批量采集调度器 (5地图 × 14场景)")
    p.add_argument("--maps", default=",".join(ALL_MAPS), help="逗号分隔的地图列表")
    p.add_argument("--scenarios", default=",".join(ALL_SCENARIOS), help="逗号分隔的场景列表")
    p.add_argument("--gpus", default="1,2,3", help="可用GPU编号 (default: 1,2,3)")
    p.add_argument("--parallel", type=int, default=3, help="并行任务数 (default: 3)")
    p.add_argument("--port-base", type=int, default=PORT_BASE, help="起始端口")
    p.add_argument("--resume", action="store_true", default=True, help="跳过已完成的任务")
    p.add_argument("--out", default="data/runs/batch", help="输出根目录")
    args = p.parse_args()

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    out_root = _REPO / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # 找 carla python
    print("[*] locating carla python ...")
    carla_python = find_carla_python()
    print(f"  [ok] {carla_python}")

    # 构建任务列表
    tasks = build_task_list(maps, scenarios, args.resume, out_root)
    total_tasks = len(maps) * len(scenarios)
    skipped = total_tasks - len(tasks)
    print(f"\n[*] Batch collection plan:")
    print(f"  Maps:      {maps}")
    print(f"  Scenarios: {len(scenarios)} scenarios")
    print(f"  Total:     {total_tasks} tasks")
    print(f"  Skipped:   {skipped} (already PASS)")
    print(f"  Remaining: {len(tasks)} tasks")
    print(f"  GPUs:      {gpus}")
    print(f"  Parallel:  {args.parallel}")
    print(f"  Ports:     {args.port_base}-{args.port_base + args.parallel - 1}")
    print(f"  Output:    {out_root}")

    if not tasks:
        print("\n✅ All tasks already completed!")
        return

    # 执行任务 (串行: 每次只跑一个任务, 因为需要独占 CARLA server)
    # TODO: 如果有多台机器或多个 CARLA 实例, 可以改为并行
    results = []
    start_all = datetime.now().timestamp()

    for idx, (task_id, map_name, scenario_id) in enumerate(tasks):
        gpu = gpus[idx % len(gpus)]
        port = args.port_base + (idx % args.parallel)

        print(f"\n[{idx+1}/{len(tasks)}] {map_name} + {scenario_id} (gpu={gpu}, port={port})")
        result = run_single_task(
            task_id, map_name, scenario_id, port, gpu, carla_python, out_root,
        )
        results.append(result)

        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} {result['status']}  "
              f"nodes={result.get('graph_nodes', '?')} edges={result.get('graph_edges', '?')}  "
              f"time={result.get('elapsed_s', '?')}s")

    # 汇总
    total_elapsed = datetime.now().timestamp() - start_all
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = len(results) - pass_count

    summary = {
        "timestamp": datetime.now().isoformat(),
        "maps": maps,
        "scenarios": scenarios,
        "total_tasks": total_tasks,
        "skipped": skipped,
        "completed": len(results),
        "passed": pass_count,
        "failed": fail_count,
        "total_elapsed_s": round(total_elapsed, 1),
        "results": results,
    }
    summary_path = out_root / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 打印汇总表
    print(f"\n{'='*70}")
    print("BATCH COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {total_tasks}  |  Skipped: {skipped}  |  "
          f"Pass: {pass_count}  |  Fail: {fail_count}")
    print(f"  Time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  Summary: {summary_path}")

    # 按地图分组打印结果
    print(f"\n{'Map':<12} {'Scenario':<8} {'Status':<8} {'Nodes':<8} {'Edges':<8} {'Time':<8}")
    print("-" * 60)
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{r['map']:<12} {r['scenario']:<8} {icon}{r['status']:<7} "
              f"{r.get('graph_nodes', 'N/A')!s:<8} {r.get('graph_edges', 'N/A')!s:<8} "
              f"{r.get('elapsed_s', 'N/A')!s:<8}")

    if fail_count > 0:
        print(f"\n⚠️  {fail_count} tasks failed. Run with --resume to retry.")


if __name__ == "__main__":
    main()
