#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect.py — 长时间 Phase1 采集 (≥20分钟主循环)

设计要点 (长期连续采集方案 §四):
  • 同步模式 world.tick() 主动驱动, 避免长时间运行中丢帧/时间戳失真.
  • 异常事件调度表 (anomaly_scheduler.EventScheduler) 按时间窗口随机穿插
    异常, 其余时间正常背景车流; 异常通过 apply_control / set_target_velocity
    等 CARLA 原生 API 直接构造, 不依赖 ScenarioRunner.
  • ego 视角跟随: 每帧 spectator.set_transform(ego_transform + offset) 让
    操作者在 CARLA View 里实时看到自车视角.
  • 分块输出: 每 N 帧 (默认 2000) 写一个 chunk_XXXX.json, 防止单文件过大
    或采集中断丢失全部数据. chunk 文件是 Phase1 中间产物, 处理完即可归档.
  • 输出格式与 run_phases_1_5.py 的 phase1_extraction.json 兼容 (每帧 dict
    包含 frame_id, elapsed_seconds, actors, traffic_lights, weather, waypoints,
    events), 后续 pipeline.py 可直接喂入 stk/scenario / stk/behavior / stk/rules
    / stk/dynamic / stk/storage 现有模块.

用法:
    python scripts/long_run/collect.py \\
        --host localhost --port 2000 \\
        --town Town10HD \\
        --total-frames 24000 \\
        --chunk-frames 2000 \\
        --vehicles 20 --walkers 8 \\
        --density 2.0 \\
        --out data/long_run/Town10HD_20260428
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

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# 复用 anomaly_scheduler
from scripts.long_run.anomaly_scheduler import (
    EventScheduler, AnomalyEvent, apply_anomaly, bind_targets,
)


# ============================================================
# 工具函数: 与 run_phases_1_5.py 对齐的 actor dict 构建
# ============================================================

def build_actor_dict(actor, atype: str, map_, carla_module, is_ego: bool = False) -> Dict[str, Any]:
    """与 run_phases_1_5.py 行 292-352 对齐的 actor dict (compat phase1_extraction.json).

    字段集与 run_phases_1_5.py phase1 输出完全一致, 保证下游 stk/extraction 模块
    可直接复用 (不需要修改 actor_extractor.py).
    """
    carla = carla_module
    t = actor.get_transform(); vel = actor.get_velocity()
    acc = actor.get_acceleration() if hasattr(actor, "get_acceleration") else None
    bb = actor.bounding_box
    ctrl = actor.get_control() if atype == "vehicle" else None

    lane_info = None
    if atype == "vehicle":
        try:
            wp = map_.get_waypoint(t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                lane_info = (int(wp.road_id), int(wp.lane_id))
        except Exception:
            pass

    d: Dict[str, Any] = {
        "is_ego": is_ego,
        "type": atype,
        "id": str(actor.id),
        "type_id": actor.type_id,
        "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
        "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
        "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
    }
    if lane_info is not None:
        d["road_id"] = lane_info[0]
        d["lane_id"] = lane_info[1]
    d["acceleration"] = ({"x": acc.x, "y": acc.y, "z": acc.z} if acc is not None else {})
    d["speed"] = vel.length()
    d["heading_rad"] = math.radians(t.rotation.yaw)
    d["pitch"] = t.rotation.pitch
    d["roll"] = t.rotation.roll
    d["bbox_extent"] = {"x": bb.extent.x, "y": bb.extent.y, "z": bb.extent.z}
    d["is_alive"] = actor.is_alive
    d["is_emergency"] = bool(actor.is_emergency_vehicle) if hasattr(actor, "is_emergency_vehicle") else False

    if atype == "vehicle":
        d["control"] = {"throttle": ctrl.throttle, "brake": ctrl.brake, "steer": ctrl.steer}
    elif atype == "walker":
        try:
            d["action"] = actor.get_action()
        except Exception:
            d["action"] = "Idle"
        try:
            wp_walker = map_.get_waypoint(t.location, project_to_road=True, lane_type=carla.LaneType.Any)
            d["is_on_crosswalk"] = (wp_walker.lane_type == carla.LaneType.Crosswalk) if wp_walker else False
            d["is_on_sidewalk"] = (wp_walker.lane_type == carla.LaneType.Sidewalk) if wp_walker else False
        except Exception:
            d["is_on_crosswalk"] = False
            d["is_on_sidewalk"] = False
    return d


# 完全删除下面那个重复的 stub


# ============================================================
# ego 视角跟随
# ============================================================

def update_spectator_follow_ego(world, ego, carla_module,
                                  offset_back: float = -8.0,
                                  offset_up: float = 4.0) -> None:
    """让 CARLA spectator 跟随 ego 车 (第三人称视角).

    每帧采集前调用: 取 ego transform, 加 offset 后 set_transform 给 spectator.
    offset_back > 0 表示在 ego 前方, 这里默认 -8 表示后方 8m, 上方 4m.
    """
    carla = carla_module
    try:
        tf = ego.get_transform()
        # ego 朝向单位向量
        yaw_rad = math.radians(tf.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        # 偏移到 ego 后方 + 上方
        cam_loc = carla.Location(
            x=tf.location.x + forward_x * offset_back,
            y=tf.location.y + forward_y * offset_back,
            z=tf.location.z + offset_up,
        )
        # 朝向 ego 略下俯
        cam_rot = carla.Rotation(
            yaw=tf.rotation.yaw,
            pitch=-8.0,
        )
        world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))
    except Exception:
        pass


# ============================================================
# 主采集逻辑
# ============================================================

def spawn_vehicles(world, n: int, bp_lib, map_, carla_module,
                   seed: int = 42) -> List[Any]:
    """与 run_phases_1_5.py 一致的 spawn 逻辑, 首车设为 hero (ego)."""
    carla = carla_module
    random.seed(seed)
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = map_.get_spawn_points()
    used = set()
    spawned = []
    for i in range(n):
        for _try in range(10):
            idx = random.randint(0, len(spawn_points) - 1)
            if idx in used:
                continue
            bp = random.choice(vehicle_bps)
            bp.set_attribute("role_name", "hero" if i == 0 else "autopilot")
            try:
                v = world.spawn_actor(bp, spawn_points[idx])
                v.set_autopilot(True)
                spawned.append(v)
                used.add(idx)
                break
            except RuntimeError:
                continue
    return spawned


def spawn_walkers(world, n: int, bp_lib, carla_module, seed: int = 42) -> List[Tuple[Any, Any]]:
    """与 run_phases_1_5.py 一致的 walker spawn (含 AI controller)."""
    carla = carla_module
    random.seed(seed + 1)
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    spawned = []
    for _ in range(n):
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


def attach_ego_sensors(world, ego, bp_lib, carla_module,
                      sensor_events: List[Dict[str, Any]]) -> List[Any]:
    """给挂 collision + lane invasion sensor, 复用 run_phases_1_5.py 的 listener.

    sensor_events 列表会被 listener 在每个 tick 周期里 append; collect 主循环每帧
    从该列表取出 events 注入当前帧 dict.
    """
    carla = carla_module
    sensors = []
    try:
        col_bp = bp_lib.find("sensor.other.collision")
        li_bp = bp_lib.find("sensor.other.lane_invasion")
        col_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        for bp in (col_bp, li_bp):
            s = world.spawn_actor(bp, col_tf, attach_to=ego)
            s.listen(lambda event, etype=("collision" if bp == col_bp else "lane_invasion"):
                     sensor_events.append({
                         "ego_id": str(ego.id),
                         "ego_actor_id": str(ego.id),
                         "event_type": etype,
                         "frame_id": getattr(event, "frame", 0),
                         "other_actor_id": str(getattr(event, "other_actor", "").id) if getattr(event, "other_actor", None) else "",
                         "location": {"x": getattr(event, "transform", carla.Transform()).location.x,
                                      "y": getattr(event, "transform", carla.Transform()).location.y,
                                      "z": getattr(event, "transform", carla.Transform()).location.z},
                     }))
            sensors.append(s)
    except Exception as e:
        print(f"[!] sensor attach failed: {e}")
    return sensors


def collect_waypoints(world, max_count: int = 2000) -> List[Dict[str, Any]]:
    """复用 run_phases_1_5.py 的 collect_waypoints, 顶级路网静态采集."""
    sys.path.insert(0, str(_REPO / "scripts" / "pipeline"))
    try:
        from run_phases_1_5 import collect_waypoints as _cw
        return _cw(world, max_count=max_count)
    except Exception as e:
        print(f"[!] cant import collect_waypoints: {e}")
        return []


# ------------- Checkpoint 写入 -------------

def _write_checkpoint(run_dir: Path, frame: int, chunk_idx: int,
                       sched: EventScheduler,
                       anomaly_log: List[Dict[str, Any]]) -> None:
    """将采集进度写入 collect_checkpoint.json.

    包含: 当前帧号, chunk 编号, 异常调度器状态, 异常日志.
    被主循环周期性调用 (默认每 200 帧一次 + 每个 chunk 边界一次).
    """
    ckpt = {
        "last_frame": frame,
        "chunk_idx": chunk_idx,
        "anomaly_events": sched.to_dict(),
        "anomaly_log": anomaly_log[-5000:],  # 最多保留 5000 条
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    ckpt_path = run_dir / "collect_checkpoint.json"
    tmp_path = ckpt_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2, default=str)
        tmp_path.replace(ckpt_path)  # 原子替换, 防写半截崩溃
    except Exception as e:
        print(f"[!] checkpoint write error: {e}")


def main():
    p = argparse.ArgumentParser(description="Long-run Phase1 collector (≥20 min)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town10HD")
    p.add_argument("--total-frames", type=int, default=24000, help="20min @ 20fps = 24000")
    p.add_argument("--chunk-frames", type=int, default=2000, help="每 2000 帧一个 chunk")
    p.add_argument("--vehicles", type=int, default=20)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--fps", type=float, default=20.0, help="同步模式 fixed_delta_seconds=1/fps")
    p.add_argument("--density", type=float, default=2.0, help="每分钟异常事件数 (期望)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/long_run/Town10HD_default")
    p.add_argument("--no-lanes", action="store_true", help="跳过 waypoint 采集 (复用旧 chunk 时)")
    p.add_argument("--no-spectator", action="store_true", help="不跟随 ego, 仅由用户操作视角")
    p.add_argument("--resume", default=None, help="从已有 run_dir 恢复 (含 collect_checkpoint.json)")
    p.add_argument("--checkpoint-interval", type=int, default=200,
                   help="每 N 帧写一次 checkpoint (默认 200 帧, 配合 chunk_frames)")
    args = p.parse_args()

    # ------------- Resume 加载 -------------
    resume_dir: Optional[Path] = None
    resume_ckpt: Optional[dict] = None
    start_frame = 0
    chunk_idx_start = 0
    if args.resume:
        resume_dir = Path(args.resume).resolve()
        ckpt_path = resume_dir / "collect_checkpoint.json"
        if not ckpt_path.exists():
            print(f"[FATAL] --resume given but {ckpt_path} not found")
            sys.exit(1)
        with open(ckpt_path) as f:
            resume_ckpt = json.load(f)
        start_frame = resume_ckpt.get("last_frame", -1) + 1
        chunk_idx_start = resume_ckpt.get("chunk_idx", 0)
        print(f"[*] RESUME from {resume_dir}")
        print(f"    last_frame={resume_ckpt.get('last_frame')} -> start at frame {start_frame}")
        print(f"    chunk_idx={chunk_idx_start} (next chunk = {chunk_idx_start+1:04d})")
        # 复用旧 run_dir, 不再创建新的
        run_dir = resume_dir

    try:
        import carla
    except ImportError:
        print("[FATAL] cant import carla"); sys.exit(1)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    try:
        print(f"[+] server={client.get_server_version()} client={client.get_client_version()}")
    except Exception as e:
        print(f"[FATAL] cant connect: {e}"); sys.exit(2)

    # 加载地图 (若不在目标地图上)
    current_map = world.get_map().name
    target_map = f"Carla/Maps/{args.town}"
    if current_map != target_map:
        print(f"[*] Switching from {current_map} to {target_map} ...")
        for a in world.get_actors():
            try: a.destroy()
            except Exception: pass
        client.load_world(args.town)
        time.sleep(5)
        world = client.get_world()
        for _ in range(10):
            actors = world.get_actors()
            if (len(list(actors.filter("vehicle.*"))) == 0
                and len(list(actors.filter("walker.*"))) == 0):
                break
            time.sleep(1)
        world = client.get_world()
        print(f"[+] Loaded map: {world.get_map().name}")
    else:
        # 地图已匹配, 但需清理旧 actors, 否则 spawn 阶段操作已销毁的 actor 会 crash
        print(f"[*] Cleaning up existing actors on {current_map} ...")
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
        time.sleep(2)  # 等异步销毁

    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    print(f"[+] synchronous_mode=True, fixed_delta={1.0/args.fps:.4f}s ({args.fps} fps)")

    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()

    # 静态车道采集 (resume 时可复用旧 metadata 中的 lane_wps 计数)
    if resume_ckpt is not None:
        lane_wps = []  # resume 时不重新采集 (已在 metadata 里)
        print(f"[+] (resume) skip lane采集, 复用旧 run 数据")
    else:
        lane_wps = [] if args.no_lanes else collect_waypoints(world, max_count=2000)
    print(f"[+] lane_wps={len(lane_wps)}")

    # spawn 交通 (resume 时也要重新 spawn — 因为 CARLA 重启了)
    print(f"[*] Spawning {args.vehicles} vehicles (first=ego=hero) ...")
    spawned_vehicles = spawn_vehicles(world, args.vehicles, bp_lib, map_, carla, seed=args.seed)
    ego = spawned_vehicles[0] if spawned_vehicles else None
    ego_id = ego.id if ego else None
    print(f"[+] ego_id={ego_id}, vehicles spawned={len(spawned_vehicles)}")

    print(f"[*] Spawning {args.walkers} walkers ...")
    spawned_walkers = spawn_walkers(world, args.walkers, bp_lib, carla, seed=args.seed)
    print(f"[+] walkers spawned={len(spawned_walkers)}")

    # ego 传感器
    sensor_events: List[Dict[str, Any]] = []
    sensors = []
    if ego is not None:
        sensors = attach_ego_sensors(world, ego, bp_lib, carla, sensor_events)
        print(f"[+] ego sensors attached: {len(sensors)}")

    # 异常调度 (build_schedule 是确定性的, 跑同样的 seed 会得到同样的事件表)
    sched = EventScheduler(
        total_frames=args.total_frames, seed=args.seed,
        density_per_minute=args.density, ego_is_target=False,
        frames_per_second=args.fps,
    )
    events = sched.build_schedule()
    print(f"[+] anomaly schedule: {len(events)} events over {args.total_frames} frames "
          f"(~{args.total_frames/args.fps/60:.1f} min @ {args.fps} fps)")

    # 绑定 target_actor_id (粗绑定到随机背景车; 实时按位置重绑在 tick 中做)
    bg_vehicles = [v for v in spawned_vehicles if v.id != ego_id]
    bind_targets(events, bg_vehicles, ego_id, seed=args.seed)
    print(f"[+] bound {sum(1 for e in events if e.target_actor_id)} events to background vehicles")

    # RESUME: 用 checkpoint 中的 events 重建事件状态 (含已 applied/completed)
    if resume_ckpt is not None:
        sched.load_from_list(resume_ckpt.get("anomaly_events", []))
        # 同时让 _active 列表恢复
        sched._active = [e for e in sched._events if e.applied and not e.completed]
        print(f"    (resume) restored anomaly schedule: "
              f"applied={sum(1 for e in events if e.applied)}, "
              f"completed={sum(1 for e in events if e.completed)}, "
              f"active={len(sched._active)}")

    # 输出目录 (resume 时已经设过 run_dir 了)
    if resume_ckpt is None:
        out_dir = _REPO / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / f"run_{timestamp}_{args.total_frames}f"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 写 metadata
        meta = {
            "host": args.host, "port": args.port, "town": args.town,
            "total_frames": args.total_frames, "chunk_frames": args.chunk_frames,
            "fps": args.fps, "vehicles": args.vehicles, "walkers": args.walkers,
            "density_per_minute": args.density, "seed": args.seed,
            "ego_id": str(ego_id) if ego_id else None,
            "vehicle_ids": [str(v.id) for v in spawned_vehicles],
            "walker_ids": [str(w.id) for w, _ in spawned_walkers],
            "lane_waypoints_count": len(lane_wps),
            "anomaly_events": sched.to_dict(),
            "start_time": timestamp,
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[+] metadata saved: {run_dir / 'metadata.json'}")
    else:
        # resume 时: 追加 metadata 中的 resume 记录
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta.setdefault("resumes", []).append({
                "resume_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "start_frame": start_frame,
                "chunk_idx_start": chunk_idx_start,
            })
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[+] (resume) appending to existing run_dir: {run_dir}")

    # 主循环
    print(f"\n[*] Entering main loop: frames {start_frame}~{args.total_frames}, chunk size={args.chunk_frames}")
    print(f"    ego视角跟随: {'OFF' if args.no_spectator else 'ON'} (CARLA View 切到 spectator)")
    print(f"    写 chunk 到: {run_dir}")
    print(f"    写 chunk 到: {run_dir}")

    tick_s = 1.0 / args.fps
    chunk: List[Dict[str, Any]] = []
    # resume 时从已有 chunk 的最后一个帧继续 (chunk_idx_start = 已写完的 chunk 编号)
    chunk_idx = chunk_idx_start
    # 若 resume 且 start_frame 不是从 0 开始, 先 tick 几帧到 start_frame (因为 CARLA 重新 spawn 后 actor 状态不同, 让模拟器稳定几帧)
    if start_frame > 0:
        print(f"[*] (resume) fast-forwarding tick from frame 0 to {start_frame} ...")
        for ff in range(start_frame):
            world.tick()
            if not args.no_spectator and ego is not None:
                update_spectator_follow_ego(world, ego, carla)
        print(f"    fast-forward done (ticked {start_frame} frames)")

    anomaly_log: List[Dict[str, Any]] = []
    if resume_ckpt is not None:
        anomaly_log = resume_ckpt.get("anomaly_log", [])
        print(f"[+] (resume) restored anomaly_log: {len(anomaly_log)} entries")
    last_status = time.time()
    last_ckpt_frame = start_frame

    try:
        for i in range(start_frame, args.total_frames):
            world.tick()

            # 应用异常 (返回 active 事件列表)
            active_events = sched.tick(i)
            for ev in active_events:
                log = apply_anomaly(world, ev, ego, carla)
                if log:
                    anomaly_log.append({
                        "frame_id": i, "event_id": ev.event_id,
                        "anomaly_type": ev.anomaly_type,
                        "target_actor_id": ev.target_actor_id,
                        "log": log,
                    })

            # 让 spectator 跟随 ego
            if not args.no_spectator and ego is not None:
                update_spectator_follow_ego(world, ego, carla)

            # 收集本帧 actor
            vehicles = world.get_actors().filter("vehicle.*")
            walkers  = world.get_actors().filter("walker.*")
            tls      = world.get_actors().filter("traffic.traffic_light*")
            weather  = world.get_weather()

            actors_list = []
            for v in vehicles:
                actors_list.append(build_actor_dict(
                    v, "vehicle", map_, carla,
                    is_ego=(v.id == ego_id)))
            for w in walkers:
                actors_list.append(build_actor_dict(w, "walker", map_, carla))
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

            # 每 chunk_frames 帧, 写一个 chunk 文件
            if (i + 1) % args.chunk_frames == 0:
                chunk_idx += 1
                cf = run_dir / f"chunk_{chunk_idx:04d}.json"
                with open(cf, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2, default=str)
                sz = cf.stat().st_size / 1024
                print(f"  [chunk {chunk_idx:04d}] frames {(i+1)-args.chunk_frames}~{i} -> "
                      f"{cf.name} ({sz:.1f} KB)")
                chunk.clear()
                # chunk 边界同步写 checkpoint
                _write_checkpoint(run_dir, i, chunk_idx, sched, anomaly_log)
                last_ckpt_frame = i

            # 周期性 checkpoint (默认 200 帧一次, 比 chunk 边界更细)
            if (i - last_ckpt_frame) >= args.checkpoint_interval:
                _write_checkpoint(run_dir, i, chunk_idx, sched, anomaly_log)
                last_ckpt_frame = i

            # 进度打印 (5s 一次)
            if time.time() - last_status > 5.0:
                last_status = time.time()
                active_n = len(sched._active)
                v_n = len(list(vehicles))
                w_n = len(list(walkers))
                print(f"  f={i+1:>5}/{args.total_frames} ({100*(i+1)/args.total_frames:.1f}%) "
                      f"v={v_n} w={w_n} active_anom={active_n} sensor_ev={len(frame_sensor_events)}")

        # 写最后不满 chunk 的部分
        if chunk:
            chunk_idx += 1
            cf = run_dir / f"chunk_{chunk_idx:04d}.json"
            with open(cf, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2, default=str)
            print(f"  [chunk {chunk_idx:04d}] tail frames -> {cf.name}")
            chunk.clear()

        # 写异常日志
        with open(run_dir / "anomaly_log.json", "w", encoding="utf-8") as f:
            json.dump(anomaly_log, f, ensure_ascii=False, indent=2)
        print(f"\n[+] anomaly_log saved: {run_dir / 'anomaly_log.json'} ({len(anomaly_log)} entries)")

    finally:
        # 清理: 关同步 + 销毁 actor
        print("\n[*] Cleaning up ...")
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass
        for s in sensors:
            try: s.stop(); s.destroy()
            except Exception: pass
        for v in spawned_vehicles:
            try: v.destroy()
            except Exception: pass
        for w, ctl in spawned_walkers:
            try:
                if ctl:
                    ctl.stop(); ctl.destroy()
            except Exception: pass
            try: w.destroy()
            except Exception: pass
        print("[+] cleanup done")

    print(f"\n[OK] Long-run collection finished.")
    print(f"     out_dir: {run_dir}")
    print(f"     chunks:  {chunk_idx}")
    print(f"    下一步: python scripts/long_run/pipeline.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
