#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spawn_traffic.py  --  在 CARLA 里生成车流人流量，便于提取
用法:
    python3 scripts/spawn_traffic.py --vehicles 20 --walkers 10 --frames 50
"""
from __future__ import annotations
import argparse
import sys
import time
import random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--vehicles", type=int, default=20)
    p.add_argument("--walkers", type=int, default=10)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--out", default="data/recordings")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import carla
    except ImportError:
        print("[FATAL] cant import carla"); sys.exit(1)

    random.seed(args.seed)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print("[+] server:", client.get_server_version())

    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()

    # 1) 生成车辆
    print(f"[*] Spawning {args.vehicles} vehicles ...")
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = map_.get_spawn_points()
    if not spawn_points:
        print("[FATAL] no spawn points in this map"); sys.exit(2)

    spawned_vehicles = []
    used_indices = set()
    for i in range(args.vehicles):
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
            except RuntimeError as e:
                continue
    print(f"[+] spawned {len(spawned_vehicles)} vehicles")

    # 2) 生成行人 (要 batch)
    print(f"[*] Spawning {args.walkers} walkers ...")
    spawned_walkers = []
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    if not walker_bps:
        print("[!] no walker blueprints"); walker_bps = []

    batches = []
    for i in range(args.walkers):
        if not walker_bps:
            break
        bp = random.choice(walker_bps)
        loc = None
        for _try in range(10):
            p = world.get_random_location_from_navigation()
            if p is not None:
                loc = p
                break
        if loc is None:
            continue
        tf = carla.Transform(loc)
        walker_batch = carla.command.SpawnActor(bp, tf)
        walker_batch.then(carla.command.SpawnActor(controller_bp, tf))
        batches.append(walker_batch)

    if batches:
        results = client.apply_batch_sync(batches, True)
        ok = 0
        for i, r in enumerate(results):
            if r.error:
                continue
            walker_id = r.actor_id
            if i + 1 < len(results) and not results[i + 1].error:
                controller_id = results[i + 1].actor_id
                spawned_walkers.append((walker_id, controller_id))
                ok += 1
        print(f"[+] spawned {ok} walkers (batched)")

    # 3) 跑 frames 帧
    print(f"[*] Running {args.frames} frames ...")
    settings = world.get_settings()
    tick_s = settings.fixed_delta_seconds or 0.05

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _REPO = Path(__file__).resolve().parent.parent
    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"traffic_{timestamp}_{args.frames}f"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 用提取管道
    sys.path.insert(0, str(_REPO))
    from stk.extraction.actor_extractor import extract_all_actors
    from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
    from stk.extraction.weather_extractor import build_environment_snapshot

    frames_data = []
    for i in range(args.frames):
        world.tick()
        actors = world.get_actors()
        vehicles = actors.filter("vehicle.*")
        walkers = actors.filter("walker.*")
        tls = actors.filter("traffic.traffic_light*")
        w = world.get_weather()

        actors_list = []
        for v in vehicles:
            t = v.get_transform()
            vel = v.get_velocity()
            actors_list.append({
                "type": "vehicle", "id": v.id, "type_id": v.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            })
        for wk in walkers:
            t = wk.get_transform()
            vel = wk.get_velocity()
            actors_list.append({
                "type": "walker", "id": wk.id, "type_id": wk.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            })

        tl_list = []
        for tl in tls:
            t = tl.get_transform()
            tl_list.append({"id": tl.id, "state": str(tl.get_state()),
                            "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z}})

        weather_dict = {"cloudiness": w.cloudiness, "precipitation": w.precipitation,
                        "precipitation_deposits": w.precipitation_deposits,
                        "wind_intensity": w.wind_intensity, "sun_altitude_angle": w.sun_altitude_angle,
                        "fog_density": w.fog_density, "wetness": w.wetness}

        raw_frame = {"frame_id": i, "elapsed_seconds": i * tick_s,
                     "actors": actors_list, "traffic_lights": tl_list, "weather": weather_dict}
        ext_a = extract_all_actors(raw_frame)
        ext_tl = extract_all_traffic_lights(tl_list)
        ext_w = build_environment_snapshot(weather_dict, i)
        extracted = {"vehicles": ext_a.get("vehicles", []),
                     "pedestrians": ext_a.get("pedestrians", []),
                     "traffic_lights": ext_tl, "weather": ext_w}
        frames_data.append({"frame_id": i, "elapsed_seconds": i * tick_s, "extracted": extracted})

        if (i+1) % 10 == 0 or i == 0 or i == args.frames - 1:
            print(f"  [{i+1:>3}/{args.frames}] v={len(extracted['vehicles'])} p={len(extracted['pedestrians'])} tl={len(ext_tl)}")

    import json
    meta = {"host": args.host, "port": args.port, "total_frames": args.frames,
            "tick_s": tick_s, "town": world.get_map().name,
            "spawned_vehicles": len(spawned_vehicles), "spawned_walkers": len(spawned_walkers),
            "start_time": timestamp}
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(run_dir / "frames.json", "w", encoding="utf-8") as f:
        json.dump(frames_data, f, ensure_ascii=False, indent=2)
    sz = (run_dir / "frames.json").stat().st_size
    print()
    print(f"[OK] {len(frames_data)} frames, size {sz/1024:.1f} KB")
    print(f"     out: {run_dir}")
    print(f"     last: v={len(frames_data[-1]['extracted']['vehicles'])} p={len(frames_data[-1]['extracted']['pedestrians'])}")

    # 4) 清理生成的 actor
    print()
    print("[*] Cleaning up spawned actors ...")
    for v in spawned_vehicles:
        try: v.destroy()
        except: pass
    # walkers need to destroy both walker + controller
    actor_map = {a.id: a for a in world.get_actors()}
    for walker_id, controller_id in spawned_walkers:
        if controller_id in actor_map:
            try: actor_map[controller_id].destroy()
            except: pass
        if walker_id in actor_map:
            try: actor_map[walker_id].destroy()
            except: pass
    print("[+] cleanup done")


if __name__ == "__main__":
    main()
