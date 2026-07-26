"""Town02/Town04 专用采集脚本 (绕开 CARLA 0.9.16 sync mode walker SIGSEGV).

========
背景
========
`scripts/long_run/collect.py` 在 synchronous_mode=True 下 spawn AI walker controller
会触发 CARLA 0.9.16 C++ 段错误 (SIGSEGV), 进程整个挂掉, Python 的 try/except 拦不住.

复现规律:
- Town01 / Town03 / Town05 / Town10HD : 同步模式 spawn walker 正常
- Town02 / Town04                    : 同步模式 spawn walker 必崩 (高频)

根因: CARLA 0.9.16 在 sync_mode 下 `attach_to=walker` 的 controller spawn 路径
有 race condition — 服务器不 tick, controller init 期望 walker 已经历某个 tick,
撞到 NULL 指针. 不动 C++ 没法修.

========
本脚本做法 (不动 collect.py)
========
1. 加载地图后保持 **异步模式** spawn 所有 vehicles + walkers (异步下正常);
2. spawn 完成后再切 **同步模式**;
3. 主循环逻辑直接复用 collect.py 的纯函数:
   - build_actor_dict / apply_anomaly / apply_weather_at_frame
   - density_target_at_frame / adjust_traffic_density
   - attach_ego_sensors / EventScheduler / bind_targets
   - update_spectator_follow_ego / collect_waypoints ... 等
4. 输出格式与 collect.py 完全一致 (chunk_XXXX.json / metadata.json / anomaly_log.json);
5. CLI 参数与 collect.py 完全兼容; 只是 spawn 阶段在异步模式下完成.

这样 Town01/03/05/10HD 继续用 `collect.py` (不改), Town02/04 用本脚本.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 复用 collect.py 的所有纯函数 (不调用 main, 只是 import)
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "long_run"))
import collect as _c  # noqa: E402

import carla  # noqa: E402


# ───────────────────────── walker spawn (异步模式版, 复用 collect.py 的逻辑) ─────────────────────────
def _spawn_walkers_async(world: Any, n: int, bp_lib, carla_module, seed: int = 42) -> List[Tuple[Any, Any]]:
    """异步模式下的 walker spawn (与 collect.py spawn_walkers 唯一区别是无需切回异步)."""
    carla = carla_module
    random.seed(seed + 1)
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    spawned: List[Tuple[Any, Any]] = []
    for _ in range(n):
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
        tf = carla.Transform(loc)
        try:
            walker = world.spawn_actor(bp, tf)
            try:
                ctl_tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 1.0))
                controller = world.spawn_actor(controller_bp, ctl_tf, attach_to=walker)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                spawned.append((walker, controller))
            except Exception:
                spawned.append((walker, None))
        except RuntimeError:
            continue
    return spawned


def _spawn_walkers_ego_centric_async(
    world: Any, n: int, bp_lib, carla_module,
    ego_spawn_point, radius_front=70.0, radius_rear=30.0, radius_side=50.0,
    seed=42, ego_yaw_deg: Optional[float] = None,
) -> List[Tuple[Any, Any]]:
    """异步模式下的 ego-centric walker spawn."""
    carla = carla_module
    random.seed(seed + 1)
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    spawned: List[Tuple[Any, Any]] = []
    for _ in range(n):
        if not walker_bps:
            break
        bp = random.choice(walker_bps)
        loc = None
        for _try in range(50):
            _loc = world.get_random_location_from_navigation()
            if _loc is None:
                continue
            if _c._ellipse_distance_to_ego(
                type("_sp", (), {"location": _loc})(),
                ego_spawn_point,
                radius_front, radius_rear, radius_side,
                ego_yaw_deg=ego_yaw_deg,
            ):
                loc = _loc
                break
        if loc is None:
            continue
        tf = carla.Transform(loc)
        try:
            walker = world.spawn_actor(bp, tf)
            try:
                ctl_tf = carla.Transform(carla.Location(loc.x, loc.y, loc.z + 1.0))
                controller = world.spawn_actor(controller_bp, ctl_tf, attach_to=walker)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                spawned.append((walker, controller))
            except Exception:
                spawned.append((walker, None))
        except RuntimeError:
            continue
    return spawned


# ───────────────────────── 主入口 ─────────────────────────
def main() -> None:
    # 复用 collect.py 的 argparse (再追加本脚本独立说明)
    p = _c.argparse.ArgumentParser(description="Town02/Town04 采集 (异步 spawn walker)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", required=True)
    p.add_argument("--total-frames", type=int, default=12000)
    p.add_argument("--chunk-frames", type=int, default=2000)
    p.add_argument("--vehicles", type=int, default=25)
    p.add_argument("--walkers", type=int, default=12)
    p.add_argument("--density", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-interval", type=int, default=200)
    p.add_argument("--no-spectator", action="store_true")
    p.add_argument("--weather-cycle", action="store_true")
    p.add_argument("--density-ramp", action="store_true")
    p.add_argument("--spawn-offset", type=int, default=0)
    p.add_argument("--ego-centric", action="store_true")
    p.add_argument("--npc-radius-front", type=float, default=70.0)
    p.add_argument("--npc-radius-rear", type=float, default=30.0)
    p.add_argument("--npc-radius-side", type=float, default=50.0)
    p.add_argument("--emergency-vehicles", type=int, default=0)
    args = p.parse_args()

    out_path = _REPO / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    current_map = world.get_map().name
    print(f"[+] server={client.get_server_version()} client={client.get_client_version()}")

    if args.town not in current_map:
        print(f"[*] Loading town {args.town} ...")
        client.load_world(args.town)
        time.sleep(5)
        world = client.get_world()
        current_map = world.get_map().name
        print(f"[+] map loaded: {current_map}")

    # 清理既有 actor
    print(f"[*] Cleaning up existing actors on {current_map} ...")
    try:
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
    except Exception:
        pass
    time.sleep(1)
    to_kill = []
    for a in world.get_actors():
        try:
            if a.type_id.startswith("vehicle.") or a.type_id.startswith("walker."):
                to_kill.append(a)
        except Exception:
            pass
    for a in to_kill:
        try: a.destroy()
        except Exception: pass
    print(f"[+] destroyed {len(to_kill)} actors")
    time.sleep(2)

    # === 关键: 先在异步模式下 spawn 所有 actor ===
    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()

    # 静态车道采集
    lane_wps = [] if False else _c.collect_waypoints(world, max_count=2000)
    print(f"[+] lane_wps={len(lane_wps)}")

    # spawn vehicles (复用 collect.py 的 spawn_vehicles - 异步/同步都安全)
    print(f"[*] Spawning {args.vehicles} vehicles (first=ego=hero) ...")
    if args.ego_centric or args.spawn_offset > 0:
        spawn_pts = map_.get_spawn_points()
        ego_offset = args.spawn_offset if 0 <= args.spawn_offset < len(spawn_pts) else 0
        ego_spawn_pt = spawn_pts[ego_offset]
        vehicle_bps = bp_lib.filter("vehicle.*")
        random.seed(args.seed)
        ego_bp = random.choice(vehicle_bps)
        ego_bp.set_attribute("role_name", "hero")
        try:
            ego = world.spawn_actor(ego_bp, ego_spawn_pt)
            ego.set_autopilot(True)
            try:
                ego_yaw_live = ego.get_transform().rotation.yaw
            except Exception:
                ego_yaw_live = ego_spawn_pt.rotation.yaw
            spawned_npcs = _c.spawn_vehicles_ego_centric(
                world, args.vehicles, bp_lib, map_, carla,
                ego_spawn_pt,
                radius_front=args.npc_radius_front,
                radius_rear=args.npc_radius_rear,
                radius_side=args.npc_radius_side,
                seed=args.seed,
                ego_yaw_deg=ego_yaw_live,
            )
            spawned_vehicles = [ego] + spawned_npcs
            print(f"[+] ego at spawn_point[{ego_offset}] npcs={len(spawned_npcs)}")
        except RuntimeError as e:
            print(f"[!] ego spawn failed ({e}), fallback default")
            spawned_vehicles = _c.spawn_vehicles(
                world, args.vehicles, bp_lib, map_, carla,
                seed=args.seed, emergency_count=args.emergency_vehicles)
            ego = spawned_vehicles[0] if spawned_vehicles else None
    else:
        spawned_vehicles = _c.spawn_vehicles(
            world, args.vehicles, bp_lib, map_, carla,
            seed=args.seed, emergency_count=args.emergency_vehicles)
        ego = spawned_vehicles[0] if spawned_vehicles else None
    ego_id = ego.id if ego else None
    print(f"[+] ego_id={ego_id}, vehicles spawned={len(spawned_vehicles)}")

    # spawn walkers (异步模式 — 关键)
    print(f"[*] Spawning {args.walkers} walkers (async mode) ...")
    if args.ego_centric and ego is not None:
        ego_tf = ego.get_transform()
        class _A: pass
        anchor = _A()
        anchor.location = ego_tf.location
        anchor.rotation = ego_tf.rotation
        spawned_walkers = _spawn_walkers_ego_centric_async(
            world, args.walkers, bp_lib, carla, anchor,
            radius_front=args.npc_radius_front,
            radius_rear=args.npc_radius_rear,
            radius_side=args.npc_radius_side,
            seed=args.seed, ego_yaw_deg=ego_tf.rotation.yaw,
        )
    else:
        spawned_walkers = _spawn_walkers_async(
            world, args.walkers, bp_lib, carla, seed=args.seed)
    print(f"[+] walkers spawned={len(spawned_walkers)}")
    if len(spawned_walkers) < args.walkers:
        pct = 100 * len(spawned_walkers) / max(args.walkers, 1)
        print(f"[!] walkers under-spawned: requested={args.walkers}, "
              f"got={len(spawned_walkers)} ({pct:.0f}%)")

    if args.weather_cycle:
        print(f"[+] weather_cycle: ON ({args.total_frames / args.fps / 60:.1f} min)")
    if args.density_ramp:
        v1, w1 = _c.DENSITY_PHASES[0]
        v2, w2 = _c.DENSITY_PHASES[1]
        v3, w3 = _c.DENSITY_PHASES[2]
        print(f"[+] density_ramp: ON ({v1}v{w1}w → {v2}v{w2}w → {v3}v{w3}w)")

    # ego 传感器 (此时还在异步模式, spawn sensor 安全)
    sensor_events: List[Dict[str, Any]] = []
    sensors = []
    if ego is not None:
        sensors = _c.attach_ego_sensors(world, ego, bp_lib, carla, sensor_events)
        print(f"[+] ego sensors attached: {len(sensors)}")

    # === 现在切同步模式 ===
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    print(f"[+] synchronous_mode=True, fixed_delta={1.0/args.fps:.4f}s ({args.fps} fps)")
    # tick 几帧让所有 actor 稳定
    for _ in range(10): world.tick()
    print("[+] tick 10 frames: actors stabilized")

    # 异常调度
    sched = _c.EventScheduler(
        total_frames=args.total_frames, seed=args.seed,
        density_per_minute=args.density, ego_is_target=False,
        frames_per_second=args.fps,
    )
    events = sched.build_schedule()
    print(f"[+] anomaly schedule: {len(events)} events over {args.total_frames} frames")

    bg_vehicles = [v for v in spawned_vehicles if v.id != ego_id]
    ego_tf = None
    vehicle_waypoints = {}
    try:
        if ego is not None:
            ego_tf = ego.get_transform()
            ego_wp = map_.get_waypoint(ego_tf.location, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if ego_wp is not None:
                vehicle_waypoints[str(ego_id)] = (int(ego_wp.road_id), int(ego_wp.lane_id))
            for v in bg_vehicles:
                try:
                    vloc = v.get_location()
                    vwp = map_.get_waypoint(vloc, project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                    if vwp is not None:
                        vehicle_waypoints[str(v.id)] = (int(vwp.road_id), int(vwp.lane_id))
                except Exception: pass
    except Exception as e:
        print(f"[!] waypoint lookup failed: {e}")
    _c.bind_targets(events, bg_vehicles, ego_id, seed=args.seed,
                    ego_transform=ego_tf, vehicle_waypoints=vehicle_waypoints,
                    spawned_walkers=spawned_walkers)
    print(f"[+] bound {sum(1 for e in events if e.target_actor_id)} events to actors")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"run_{timestamp}_{args.total_frames}f"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "host": args.host, "port": args.port, "town": args.town,
        "total_frames": args.total_frames, "chunk_frames": args.chunk_frames,
        "fps": args.fps, "vehicles": args.vehicles, "walkers": args.walkers,
        "density_per_minute": args.density, "seed": args.seed,
        "ego_id": str(ego_id) if ego_id else None,
        "spawn_mode": "town02_04_async_walker" + ("_ego_centric" if args.ego_centric else ""),
        "vehicle_ids": [str(v.id) for v in spawned_vehicles],
        "walker_ids": [str(w.id) for w, _ in spawned_walkers],
        "lane_waypoints_count": len(lane_wps),
        "anomaly_events": sched.to_dict(),
        "start_time": timestamp,
        "collector": "collect_town02_04.py (async walker spawn)",
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[+] metadata saved: {run_dir / 'metadata.json'}")

    # 主循环
    print(f"\n[*] Entering main loop: frames 0~{args.total_frames}, chunk size={args.chunk_frames}")
    print(f"    ego视角跟随: {'OFF' if args.no_spectator else 'ON'}")
    print(f"    写 chunk 到: {run_dir}")

    tick_s = 1.0 / args.fps
    chunk: List[Dict[str, Any]] = []
    chunk_idx = 0
    last_status = time.time()
    last_ckpt_frame = 0
    anomaly_log: List[Dict[str, Any]] = []

    try:
        for i in range(args.total_frames):
            world.tick()

            active_events, completed_events = sched.tick(i)
            for ev in active_events:
                log = _c.apply_anomaly(world, ev, ego, carla,
                                       spawned_walkers=spawned_walkers,
                                       bp_lib=bp_lib, map_=map_)
                if log:
                    anomaly_log.append({
                        "frame_id": i, "event_id": ev.event_id,
                        "anomaly_type": ev.anomaly_type,
                        "target_actor_id": ev.target_actor_id,
                        "log": log,
                    })
            for ev in completed_events:
                if ev.target_actor_id is not None:
                    try:
                        a = world.get_actor(int(ev.target_actor_id))
                        if a is not None and a.is_alive:
                            if ev.anomaly_type == "obs_blk":
                                a.destroy()
                                extra_obs_id = ev.extra.get("obstacle_actor_id")
                                if extra_obs_id and extra_obs_id != ev.target_actor_id:
                                    try:
                                        oa = world.get_actor(int(extra_obs_id))
                                        if oa is not None and oa.is_alive:
                                            oa.destroy()
                                    except Exception: pass
                            else:
                                a.set_autopilot(True)
                    except Exception: pass

            if not args.no_spectator and ego is not None:
                _c.update_spectator_follow_ego(world, ego, carla)

            vehicles = world.get_actors().filter("vehicle.*")
            walkers = world.get_actors().filter("walker.*")
            tls = world.get_actors().filter("traffic.traffic_light*")
            weather = world.get_weather()

            actors_list = []
            for v in vehicles:
                actors_list.append(_c.build_actor_dict(
                    v, "vehicle", map_, carla, is_ego=(v.id == ego_id)))
            for w in walkers:
                actors_list.append(_c.build_actor_dict(w, "walker", map_, carla))
            tl_list = []
            for tl in tls:
                t = tl.get_transform()
                tl_list.append({
                    "id": str(tl.id), "state": str(tl.get_state()),
                    "elapsed_time": tl.get_elapsed_time(),
                    "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                    "rotation_yaw": t.rotation.yaw,
                    "affected_lane_ids": list(tl.get_affected_lane_id_list())
                                       if hasattr(tl, "get_affected_lane_id_list") else [],
                })
            if args.weather_cycle:
                weather_dict = _c.apply_weather_at_frame(world, carla, i, args.total_frames)
            else:
                weather_dict = {
                    "cloudiness": weather.cloudiness, "precipitation": weather.precipitation,
                    "precipitation_deposits": weather.precipitation_deposits,
                    "wind_intensity": weather.wind_intensity,
                    "sun_altitude_angle": weather.sun_altitude_angle,
                    "fog_density": weather.fog_density, "wetness": weather.wetness,
                }
            frame_sensor_events = list(sensor_events)
            sensor_events.clear()
            for ev in frame_sensor_events:
                ev["frame_id"] = i

            frame_dict = {
                "frame_id": i, "elapsed_seconds": i * tick_s,
                "actors": actors_list, "traffic_lights": tl_list,
                "weather": weather_dict, "waypoints": lane_wps,
                "events": frame_sensor_events,
            }
            chunk.append(frame_dict)

            if (i + 1) % args.chunk_frames == 0:
                chunk_idx += 1
                cf = run_dir / f"chunk_{chunk_idx:04d}.json"
                with open(cf, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2, default=str)
                sz = cf.stat().st_size / 1024
                print(f"  [chunk {chunk_idx:04d}] frames {(i+1)-args.chunk_frames}~{i} -> "
                      f"{cf.name} ({sz:.1f} KB)")
                chunk.clear()
                _c._write_checkpoint(run_dir, i, chunk_idx, sched, anomaly_log)
                last_ckpt_frame = i

                if args.density_ramp and i + 1 < args.total_frames:
                    tv, tw = _c.density_target_at_frame(i + 1, args.total_frames)
                    # ⚠️ density ramp 增补 walker 时, 同步模式下 spawn AI controller
                    # 会 SIGSEGV — 调用 adjust_traffic_density 期间临时切异步.
                    if tw > len(spawned_walkers):
                        try:
                            s = world.get_settings()
                            saved_sync = s.synchronous_mode
                            saved_ddt = s.fixed_delta_seconds
                            s.synchronous_mode = False
                            s.fixed_delta_seconds = None
                            world.apply_settings(s)
                            time.sleep(0.2)
                        except Exception:
                            saved_sync = True
                            saved_ddt = 1.0 / args.fps
                    else:
                        saved_sync = None
                    spawned_vehicles, spawned_walkers = _c.adjust_traffic_density(
                        world, carla, bp_lib, map_,
                        spawned_vehicles, spawned_walkers, tv, tw,
                        seed=args.seed,
                    )
                    # 恢复同步
                    if saved_sync is True:
                        try:
                            s = world.get_settings()
                            s.synchronous_mode = True
                            s.fixed_delta_seconds = saved_ddt
                            world.apply_settings(s)
                            # tick 几帧让 actor 状态稳定
                            for _ in range(5): world.tick()
                        except Exception:
                            pass
                    print(f"  [density] frame {i+1}: -> v={len(spawned_vehicles)}/"
                          f"target={tv}, w={len(spawned_walkers)}/target={tw}")

            if (i - last_ckpt_frame) >= args.checkpoint_interval:
                _c._write_checkpoint(run_dir, i, chunk_idx, sched, anomaly_log)
                last_ckpt_frame = i

            if time.time() - last_status > 5.0:
                last_status = time.time()
                active_n = len(sched._active)
                v_n = len(list(vehicles))
                w_n = len(list(walkers))
                print(f"  f={i+1:>5}/{args.total_frames} ({100*(i+1)/args.total_frames:.1f}%) "
                      f"v={v_n} w={w_n} active_anom={active_n} sensor_ev={len(frame_sensor_events)}")

        if chunk:
            chunk_idx += 1
            cf = run_dir / f"chunk_{chunk_idx:04d}.json"
            with open(cf, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2, default=str)
            print(f"  [chunk {chunk_idx:04d}] tail frames -> {cf.name}")
            chunk.clear()

        with open(run_dir / "anomaly_log.json", "w", encoding="utf-8") as f:
            json.dump(anomaly_log, f, ensure_ascii=False, indent=2)
        print(f"\n[+] anomaly_log saved: {run_dir / 'anomaly_log.json'} ({len(anomaly_log)} entries)")

    finally:
        print("\n[*] Cleaning up ...")
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception: pass
        for s in sensors:
            try: s.stop(); s.destroy()
            except Exception: pass
        for v in spawned_vehicles:
            try: v.destroy()
            except Exception: pass
        for w, ctl in spawned_walkers:
            try:
                if ctl: ctl.stop(); ctl.destroy()
            except Exception: pass
            try: w.destroy()
            except Exception: pass
        print("[+] cleanup done")

    print(f"\n[OK] Town02/04 collection finished.")
    print(f"     out_dir: {run_dir}")
    print(f"     chunks:  {chunk_idx}")
    print(f"    下一步: python scripts/long_run/pipeline.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
