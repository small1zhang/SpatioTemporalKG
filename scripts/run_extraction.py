#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extraction.py  --  CARLA 数据提取主脚本（验证完整链路）
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--out", default="data/recordings")
    p.add_argument("--town", default=None)
    args = p.parse_args()

    try:
        import carla
    except ImportError:
        print("[FATAL] cant import carla")
        sys.exit(1)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    try:
        print("[+] server=", client.get_server_version(), "client=", client.get_client_version())
    except Exception as e:
        print("[FATAL]", e); sys.exit(2)

    world = client.get_world()
    if args.town:
        print("[*] loading map", args.town)
        world = client.load_world(args.town)
        time.sleep(5)

    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{timestamp}_{args.frames}f"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("[*] out_dir:", run_dir)

    from stk.extraction.actor_extractor import extract_all_actors
    from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
    from stk.extraction.weather_extractor import build_environment_snapshot

    settings = world.get_settings()
    world_tick = settings.fixed_delta_seconds or 0.05

    print(f"[*] collecting {args.frames} frames tick={world_tick}s")
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
                "type": "vehicle",
                "id": v.id,
                "type_id": v.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            })
        for wk in walkers:
            t = wk.get_transform()
            vel = wk.get_velocity()
            actors_list.append({
                "type": "walker",
                "id": wk.id,
                "type_id": wk.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            })

        tl_list = []
        for tl in tls:
            t = tl.get_transform()
            tl_list.append({
                "id": tl.id,
                "state": str(tl.get_state()),
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
            })

        weather_dict = {
            "cloudiness": w.cloudiness,
            "precipitation": w.precipitation,
            "precipitation_deposits": w.precipitation_deposits,
            "wind_intensity": w.wind_intensity,
            "sun_altitude_angle": w.sun_altitude_angle,
            "fog_density": w.fog_density,
            "wetness": w.wetness,
        }

        raw_frame = {
            "frame_id": i,
            "elapsed_seconds": i * world_tick,
            "actors": actors_list,
            "traffic_lights": tl_list,
            "weather": weather_dict,
        }

        extracted_actors = extract_all_actors(raw_frame)
        extracted_tl = extract_all_traffic_lights(tl_list)
        extracted_weather = build_environment_snapshot(weather_dict, i)

        extracted = {
            "vehicles": extracted_actors.get("vehicles", []),
            "pedestrians": extracted_actors.get("pedestrians", []),
            "traffic_lights": extracted_tl,
            "weather": extracted_weather,
        }
        frames_data.append({
            "frame_id": i,
            "elapsed_seconds": i * world_tick,
            "extracted": extracted,
        })

        if (i+1) % 10 == 0 or i == 0 or i == args.frames - 1:
            print(f"  [{i+1:>3}/{args.frames}] v={len(extracted['vehicles'])} p={len(extracted['pedestrians'])} tl={len(extracted_tl)}")

    meta = {
        "host": args.host,
        "port": args.port,
        "total_frames": args.frames,
        "tick_s": world_tick,
        "town": world.get_map().name if world.get_map() else "unknown",
        "start_time": timestamp,
        "timestamp_end": datetime.now().isoformat(),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(run_dir / "frames.json", "w", encoding="utf-8") as f:
        json.dump(frames_data, f, ensure_ascii=False, indent=2)

    sz = (run_dir / "frames.json").stat().st_size
    print()
    print(f"[OK] done. {len(frames_data)} frames")
    print(f"     out: {run_dir}")
    print(f"     size: {sz/1024:.1f} KB")
    print(f"     town: {meta['town']}")
    if frames_data:
        last = frames_data[-1]
        print(f"     last: v={len(last['extracted']['vehicles'])} p={len(last['extracted']['pedestrians'])} tl={len(last['extracted']['traffic_lights'])}")
    print()
    print("[*] pipeline ok")


if __name__ == "__main__":
    main()
