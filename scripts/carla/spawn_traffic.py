#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spawn_traffic.py  --  在 CARLA 里生成车流人流量，然后跑提取流水线
修正版 v2: 
  - 填充 acceleration / bbox_extent / control / speed / heading 等全部字段
  - 修正 walker batch spawn 的清理逻辑
  - 支持 --continuous 模式持续跑多轮
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
import random
from pathlib import Path


def _compute_speed(vel_dict):
    """从 velocity dict 计算 speed (m/s)."""
    x = vel_dict.get("x", 0.0)
    y = vel_dict.get("y", 0.0)
    z = vel_dict.get("z", 0.0)
    return math.sqrt(x * x + y * y + z * z)


def build_actor_dict(actor, a_type, prev_loc=None):
    """从 CARLA Actor 构建完整 dict, 补全提取器需要的所有字段."""
    t = actor.get_transform()
    loc = t.location
    rot = t.rotation
    vel = actor.get_velocity()
    acc = actor.get_acceleration()
    bbox = actor.get_bounding_box()

    speed = _compute_speed({"x": vel.x, "y": vel.y, "z": vel.z})
    heading_rad = math.radians(rot.yaw)

    d = {
        "type": a_type,
        "id": actor.id,
        "type_id": actor.type_id,
        "location": {"x": loc.x, "y": loc.y, "z": loc.z},
        "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
        "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
        "acceleration": {"x": acc.x, "y": acc.y, "z": acc.z},
        "bbox_extent": {"x": bbox.extent.x, "y": bbox.extent.y, "z": bbox.extent.z},
        "speed": speed,
        "speed_kmh": speed * 3.6,
        "heading_rad": heading_rad,
        "pitch": rot.pitch,
        "roll": rot.roll,
        "is_alive": actor.is_alive,
        "is_emergency": actor.get_attributes().get("special_type", "") == "emergency",
    }

    if a_type == "vehicle":
        try:
            ctrl = actor.get_control()
            d["control"] = {"throttle": ctrl.throttle, "steer": ctrl.steer, "brake": ctrl.brake}
        except Exception:
            d["control"] = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
        d["is_ego"] = actor.attributes.get("role_name", "") == "hero"
    elif a_type == "walker":
        try:
            d["action"] = actor.get_action()
        except Exception:
            d["action"] = "Idle"
        d["is_on_crosswalk"] = False
        d["is_on_sidewalk"] = False

    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--vehicles", type=int, default=20)
    p.add_argument("--walkers", type=int, default=10)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--rounds", type=int, default=1,
                   help="连续跑几轮（每轮清洗并重新生成）")
    p.add_argument("--out", default="data/recordings")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import carla
    except ImportError:
        print("[FATAL] cant import carla")
        sys.exit(1)

    random.seed(args.seed)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print("[+] server:", client.get_server_version())

    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()

    _REPO = Path(__file__).resolve().parent.parent.parent
    out_dir = _REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载提取模块
    sys.path.insert(0, str(_REPO))
    from stk.extraction.actor_extractor import extract_all_actors
    from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
    from stk.extraction.weather_extractor import build_environment_snapshot

    settings = world.get_settings()
    tick_s = settings.fixed_delta_seconds or 0.05

    for round_idx in range(args.rounds):
        print(f"\n{=*60}")
        print(f"[*] Round {round_idx+1}/{args.rounds}")
        print(f"{=*60}")

        # ---- 1) 生成车辆 ----
        print(f"[*] Spawning {args.vehicles} vehicles ...")
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = map_.get_spawn_points()
        if not spawn_points:
            print("[FATAL] no spawn points"); sys.exit(2)

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
                except RuntimeError:
                    continue
        print(f"[+] spawned {len(spawned_vehicles)} vehicles")

        # ---- 2) 生成行人 (逐个 spawn, 更可靠) ----
        print(f"[*] Spawning {args.walkers} walkers ...")
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        controller_bp = bp_lib.find("controller.ai.walker")
        
        spawned_walkers = []
        for i in range(args.walkers):
            if not walker_bps:
                break
            bp = random.choice(walker_bps)
            loc = None
            for _try in range(10):
                _loc = world.get_random_location_from_navigation()
                if _loc is not None:
                    loc = _loc
                    break
            if loc is None:
                continue
            tf = carla.Transform(loc)
            try:
                walker = world.spawn_actor(bp, tf)
                # try to attach a controller
                try:
                    ctl_tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 1.0))
                    controller = world.spawn_actor(controller_bp, ctl_tf, attach_to=walker)
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    spawned_walkers.append((walker, controller))
                except Exception:
                    spawned_walkers.append((walker, None))
            except RuntimeError:
                continue
        print(f"[+] spawned {len(spawned_walkers)} walkers")

        # ---- 3) 跑帧 ----
        print(f"[*] Running {args.frames} frames (tick={tick_s}s)...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / f"traffic_{timestamp}_R{round_idx+1}_{args.frames}f"
        run_dir.mkdir(parents=True, exist_ok=True)

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
                actors_list.append(build_actor_dict(v, "vehicle"))
            for wk in walkers:
                actors_list.append(build_actor_dict(wk, "walker"))

            tl_list = []
            for tl in tls:
                t = tl.get_transform()
                tl_list.append({
                    "id": tl.id,
                    "state": str(tl.get_state()),
                    "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                    "elapsed_time": tl.get_elapsed_time(),
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
                "elapsed_seconds": i * tick_s,
                "actors": actors_list,
                "traffic_lights": tl_list,
                "weather": weather_dict,
            }
            ext_a = extract_all_actors(raw_frame)
            ext_tl = extract_all_traffic_lights(tl_list)
            ext_w = build_environment_snapshot(weather_dict, i)

            extracted = {
                "vehicles": ext_a.get("vehicles", []),
                "pedestrians": ext_a.get("pedestrians", []),
                "traffic_lights": ext_tl,
                "weather": ext_w,
            }
            frames_data.append({
                "frame_id": i,
                "elapsed_seconds": i * tick_s,
                "extracted": extracted,
            })

            if (i + 1) % 20 == 0 or i == 0 or i == args.frames - 1:
                vs = len(extracted["vehicles"])
                ps = len(extracted["pedestrians"])
                print(f"  [{i+1:>4}/{args.frames}] v={vs} p={ps} tl={len(ext_tl)}")

        # 保存
        meta = {
            "host": args.host, "port": args.port,
            "total_frames": args.frames, "tick_s": tick_s,
            "town": world.get_map().name,
            "spawned_vehicles": len(spawned_vehicles),
            "spawned_walkers": len(spawned_walkers),
            "start_time": timestamp,
            "round": round_idx + 1,
            "total_rounds": args.rounds,
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(run_dir / "frames.json", "w", encoding="utf-8") as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)

        sz = (run_dir / "frames.json").stat().st_size
        last = frames_data[-1]
        print(f"\n[OK] Round {round_idx+1} done: {len(frames_data)} frames, {sz/1024:.1f} KB")
        print(f"     out: {run_dir}")
        print(f"     last: v={len(last[extracted][vehicles])} p={len(last[extracted][pedestrians])}")

        # 验证第一辆车的所有字段
        f0_v = frames_data[0]["extracted"]["vehicles"]
        if f0_v:
            print(f"     sample vehicle keys: {list(f0_v[0].keys())}")
            v0 = f0_v[0]
            print(f"     speed={v0[speed]:.3f} speed_kmh={v0[speed_kmh]:.2f} heading={v0[heading_rad]:.3f}")

        # ---- 4) 清理 ----
        print(f"\n[*] Cleaning up round {round_idx+1} ...")
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
        print(f"[+] cleanup done ({len(spawned_vehicles)} vehicles, {len(spawned_walkers)} walkers)")
        spawned_vehicles.clear()
        spawned_walkers.clear()

        # 等待一帧让世界回到初始
        world.tick()

    print(f"\n{=*60}")
    print(f"[OK] All {args.rounds} rounds completed!")


if __name__ == "__main__":
    main()
PYEOF
echo "write OK, lines=$(wc -l < /home/aisecurity/01_ZHB/SpatioTemporalKG/scripts/carla/spawn_traffic.py)"
