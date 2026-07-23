#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test.py -- 单地图单场景冒烟测试

用法:
    python scripts/pipeline/smoke_test.py --map Town10HD --scenario S00

流程:
    1. cold-boot CARLA (进程级隔离)
    2. 按场景参数注入交通参与者和天气
    3. 跑 5 阶段提取管线
    4. 输出结构化结果到 data/runs/smoke_test/<map>_<scenario>/
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
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "carla"))
from process_manager import (
    restart_carla_with_map, kill_carla_on_port, health_check, find_carla_python,
)

# 场景参数映射: YAML scenario_parameters -> CARLA traffic/weather 设置
SCENARIO_TRAFFIC_PROFILES = {
    "S00": {"vehicle_count": 3,  "pedestrian_count": 0,  "weather": "clear",    "frames": 100},
    "S01": {"vehicle_count": 5,  "pedestrian_count": 0,  "weather": "clear",    "frames": 100},
    "S02": {"vehicle_count": 3,  "pedestrian_count": 8,  "weather": "clear",    "frames": 100},
    "S10": {"vehicle_count": 3,  "pedestrian_count": 5,  "weather": "clear",    "frames": 150},
    "S11": {"vehicle_count": 6,  "pedestrian_count": 0,  "weather": "clear",    "frames": 150},
    "S12": {"vehicle_count": 5,  "pedestrian_count": 2,  "weather": "clear",    "frames": 150},
    "S13": {"vehicle_count": 8,  "pedestrian_count": 0,  "weather": "clear",    "frames": 150},
    "S20": {"vehicle_count": 10, "pedestrian_count": 0,  "weather": "clear",    "frames": 150},
    "S21": {"vehicle_count": 8,  "pedestrian_count": 0,  "weather": "clear",    "frames": 150},
    "S22": {"vehicle_count": 6,  "pedestrian_count": 2,  "weather": "clear",    "frames": 150},
    "S30": {"vehicle_count": 3,  "pedestrian_count": 5,  "weather": "night",    "frames": 150},
    "S31": {"vehicle_count": 6,  "pedestrian_count": 0,  "weather": "rain",     "frames": 150},
    "S32": {"vehicle_count": 5,  "pedestrian_count": 0,  "weather": "clear",    "frames": 150},
    "S33": {"vehicle_count": 5,  "pedestrian_count": 6,  "weather": "glare",    "frames": 150},
}

WEATHER_PRESETS = {
    "clear": {"cloudiness": 10, "precipitation": 0, "precipitation_deposits": 0,
              "wind_intensity": 0, "sun_altitude_angle": 80, "fog_density": 0, "wetness": 0},
    "night": {"cloudiness": 80, "precipitation": 0, "precipitation_deposits": 0,
              "wind_intensity": 0, "sun_altitude_angle": -30, "fog_density": 0, "wetness": 0},
    "rain":  {"cloudiness": 80, "precipitation": 60, "precipitation_deposits": 40,
              "wind_intensity": 30, "sun_altitude_angle": 40, "fog_density": 0, "wetness": 60},
    "glare": {"cloudiness": 0, "precipitation": 0, "precipitation_deposits": 0,
              "wind_intensity": 0, "sun_altitude_angle": 5, "fog_density": 10, "wetness": 0},
}


def load_map_config(map_name: str) -> dict:
    """加载 map_configs/<map>.yaml"""
    cfg_path = _REPO / "map_configs" / f"{map_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Map config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def apply_weather(world, weather_key: str):
    """设置 CARLA 天气"""
    carla = sys.modules["carla"]
    w = WEATHER_PRESETS.get(weather_key, WEATHER_PRESETS["clear"])
    weather = carla.WeatherParameters(
        cloudiness=w["cloudiness"],
        precipitation=w["precipitation"],
        precipitation_deposits=w["precipitation_deposits"],
        wind_intensity=w["wind_intensity"],
        sun_altitude_angle=w["sun_altitude_angle"],
        fog_density=w["fog_density"],
        wetness=w["wetness"],
    )
    world.set_weather(weather)
    print(f"  [weather] set to '{weather_key}': sun={w['sun_altitude_angle']} rain={w['precipitation']}")


def spawn_scenario_traffic(world, scenario_id: str, yaml_params: dict, seed: int = 42):
    """根据场景参数生成交通参与者"""
    import carla
    random.seed(seed)
    profile = SCENARIO_TRAFFIC_PROFILES.get(scenario_id, {"vehicles": 5, "walkers": 2})
    n_vehicles = yaml_params.get("vehicle_count", profile["vehicles"])
    n_walkers = yaml_params.get("pedestrian_count", profile["walkers"])

    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()
    spawn_points = map_.get_spawn_points()

    spawned_vehicles = []
    spawned_walkers = []

    # 生成车辆
    vehicle_bps = bp_lib.filter("vehicle.*")
    used_indices = set()
    for i in range(n_vehicles):
        for _try in range(10):
            idx = random.randint(0, len(spawn_points) - 1)
            if idx in used_indices:
                continue
            bp = random.choice(vehicle_bps)
            bp.set_attribute("role_name", "autopilot")
            try:
                v = world.spawn_actor(bp, spawn_points[idx])
                v.set_autopilot(True)
                spawned_vehicles.append(v)
                used_indices.add(idx)
                break
            except RuntimeError:
                continue

    # 生成行人
    if n_walkers > 0:
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_controller_bp = bp_lib.find("controller.ai.walker")
        for i in range(n_walkers):
            if not walker_bps:
                break
            bp = random.choice(walker_bps)
            loc = None
            for _try in range(30):
                _loc = world.get_random_location_from_navigation()
                if _loc is not None:
                    loc = _loc
                    break
            if loc is None:
                continue
            try:
                w = world.spawn_actor(bp, carla.Transform(loc))
                try:
                    ctl_tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 1.0))
                    ctl = world.spawn_actor(walker_controller_bp, ctl_tf, attach_to=w)
                    ctl.start()
                    ctl.go_to_location(world.get_random_location_from_navigation())
                    ctl.set_max_speed(1.0 + random.random())
                    spawned_walkers.append((w, ctl))
                except Exception:
                    spawned_walkers.append((w, None))
            except RuntimeError:
                continue

    print(f"  [traffic] spawned {len(spawned_vehicles)} vehicles, {len(spawned_walkers)} walkers")
    return spawned_vehicles, spawned_walkers


def cleanup_actors(world, spawned_vehicles, spawned_walkers):
    """清理生成的 actor"""
    for v in spawned_vehicles:
        try:
            if v.is_alive:
                v.destroy()
        except Exception:
            pass
    for walker, ctl in spawned_walkers:
        try:
            if ctl and ctl.is_alive:
                ctl.stop()
                ctl.destroy()
        except Exception:
            pass
        try:
            if walker.is_alive:
                walker.destroy()
        except Exception:
            pass


def run_smoke_test(map_name: str, scenario_id: str, port: int, gpu: int,
                   carla_python: str, out_root: Path) -> dict:
    """完整冒烟测试: cold-boot -> inject scenario -> run 5 phases -> cleanup"""
    print(f"\n{'='*70}")
    print(f"[smoke] {map_name} + {scenario_id}  port={port}  gpu={gpu}")
    print(f"{'='*70}")

    start_ts = datetime.now().timestamp()

    # 加载 YAML 配置
    map_cfg = load_map_config(map_name)
    yaml_params = map_cfg.get("scenario_parameters", {}).get(scenario_id, {})
    print(f"  [yaml] params: {yaml_params}")

    # 1) Cold-boot CARLA
    print(f"  [boot] cold-boot CARLA on port {port} with map {map_name} ...")
    ok = restart_carla_with_map(
        port, map_name, host="localhost",
        cold_boot_timeout=120.0, graphics_adapter=gpu,
        quality_level="Low",
        log_path=str(out_root / f"carla_{map_name}_{port}.log"),
        python=carla_python,
    )
    if not ok:
        return {"map": map_name, "scenario": scenario_id, "status": "COLD_BOOT_FAIL"}

    if not health_check("localhost", port, timeout=10.0, python=carla_python):
        kill_carla_on_port(port, grace_seconds=2.0)
        return {"map": map_name, "scenario": scenario_id, "status": "HEALTH_FAIL"}

    # 2) 跑 5 阶段管线 (复用 run_phases_1_5.py)
    profile = SCENARIO_TRAFFIC_PROFILES.get(scenario_id, {})
    frames = profile.get("frames", 100)
    n_vehicles = yaml_params.get("vehicle_count", profile.get("vehicle_count", 5))
    n_walkers = yaml_params.get("pedestrian_count", profile.get("pedestrian_count", 2))

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
            "--out", f"data/runs/smoke_test/{map_name}_{scenario_id}",
        ],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
    )

    stdout = (proc.stdout or "") + (proc.stderr or "")

    # 解析结果
    result = {
        "map": map_name,
        "scenario": scenario_id,
        "status": "FAIL",
        "port": port,
        "gpu": gpu,
        "yaml_params": yaml_params,
        "frames": frames,
    }
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
        if "Total:" in line:
            try:
                result["total_time_s"] = float(line.split(":")[-1].strip().rstrip("s"))
            except Exception:
                pass

    # Also try to read KG summary from output files
    import glob
    kg_files = sorted(
        _REPO.glob(f"data/runs/smoke_test/{map_name}_{scenario_id}/phases_*_{frames}f/phase5_kg_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if kg_files:
        with open(kg_files[-1]) as kf:
            kg = json.load(kf)
        result["graph_nodes"] = kg.get("phase5_graph_nodes")
        result["graph_edges"] = kg.get("phase5_graph_edges")
        result["node_types"] = kg.get("phase5_node_types")
        result["edge_types"] = kg.get("phase5_edge_types")
        result["total_entities"] = kg.get("total_unique_entities")
        result["scene_rels_total"] = kg.get("scene_rels_total")

    if result["status"] != "PASS":
        result["status"] = "FAIL"
        result["stdout_tail"] = stdout[-2000:]
        result["stdout_tail"] = stdout[-3000:]

    # Kill CARLA
    kill_carla_on_port(port, grace_seconds=3.0)

    elapsed = datetime.now().timestamp() - start_ts
    result["elapsed_s"] = round(elapsed, 1)
    status_icon = "✅" if result["status"] == "PASS" else "❌"
    print(f"\n  {status_icon} {map_name}+{scenario_id}: {result['status']}  "
          f"nodes={result.get('graph_nodes', '?')} edges={result.get('graph_edges', '?')}  "
          f"time={elapsed:.1f}s")
    return result


def main():
    p = argparse.ArgumentParser(description="单地图单场景冒烟测试")
    p.add_argument("--map", default="Town10HD", help="地图名 (default: Town10HD)")
    p.add_argument("--scenario", default="S00", help="场景ID (default: S00)")
    p.add_argument("--port", type=int, default=2500, help="CARLA 端口 (default: 2500)")
    p.add_argument("--gpu", type=int, default=1, help="GPU 编号 (default: 1)")
    p.add_argument("--out", default="data/runs/smoke_test", help="输出目录")
    args = p.parse_args()

    out_root = _REPO / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # 找 carla python
    print("[*] locating carla python ...")
    carla_python = find_carla_python()
    print(f"  [ok] {carla_python}")

    # 验证场景存在
    map_cfg = load_map_config(args.map)
    if args.scenario not in map_cfg.get("enabled_scenarios", []):
        print(f"[WARN] {args.scenario} not in enabled_scenarios for {args.map}, running anyway")

    # 跑冒烟测试
    result = run_smoke_test(
        args.map, args.scenario, args.port, args.gpu,
        carla_python, out_root,
    )

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = out_root / f"smoke_{args.map}_{args.scenario}_{ts}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[*] result saved: {result_path}")

    # 打印摘要
    print(f"\n{'='*70}")
    print("SMOKE TEST RESULT")
    print(f"{'='*70}")
    print(f"  Map:      {result['map']}")
    print(f"  Scenario: {result['scenario']}")
    print(f"  Status:   {result['status']}")
    print(f"  Nodes:    {result.get('graph_nodes', 'N/A')}")
    print(f"  Edges:    {result.get('graph_edges', 'N/A')}")
    print(f"  Time:     {result.get('elapsed_s', 'N/A')}s")

    if result["status"] == "PASS":
        print("\n✅ Pipeline validated! Ready for batch collection.")
    else:
        print("\n❌ Smoke test failed. Check stdout_tail in result JSON.")
        if "stdout_tail" in result:
            print(f"\n--- stdout tail ---\n{result['stdout_tail'][-1500:]}")


if __name__ == "__main__":
    main()
