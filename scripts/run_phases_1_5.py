#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_phases_1_5.py -- 一键跑通时空动态知识图谱 5 个阶段 (GPU 3)

更新版:
  - 采 lane topology (waypoints) 真实入图
  - 车辆 -> in_lane 关系
  - phase5 用真正的 serialize_graph 导出 KG
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def banner(label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)


def collect_waypoints(world, max_count: int = 2000):
    """遍历 map waypoint, 提取 road/lane 拓扑信息."""
    carla = sys.modules["carla"]
    map_ = world.get_map()
    # generate_waypoints 在 0.9.16 上要传 distance
    try:
        wps = map_.generate_waypoints(2.0)  # 每 2m 一个
    except Exception:
        wps = map_.get_topology_waypoints()
    if not wps:
        return []
    out = []
    seen = set()
    for wp in wps[:max_count]:
        try:
            tid = wp.road_id
            lid = wp.lane_id
            key = (tid, lid)
            if key in seen:
                continue
            seen.add(key)
            tr = wp.transform
            loc = tr.location
            # left / right lane (SiblingLane)
            left_id = None; right_id = None
            try:
                left = wp.get_left_lane()
                right = wp.get_right_lane()
                if left is not None and left.lane_type == carla.LaneType.Driving:
                    left_id = left.lane_id
                if right is not None and right.lane_type == carla.LaneType.Driving:
                    right_id = right.lane_id
            except Exception:
                pass
            out.append({
                "road_id": int(tid),
                "lane_id": int(lid),
                "junction_id": -1 if wp.is_junction else -1,
                "lane_type": str(wp.lane_type).split(".")[-1],
                "lane_width": float(wp.lane_width),
                "x": float(loc.x), "y": float(loc.y), "z": float(loc.z),
                "heading_rad": float(tr.rotation.yaw * 3.141592653589793 / 180.0),
                "left_lane_id": left_id,
                "right_lane_id": right_id,
                "speed_limit": 0.0,  # CARLA 0.9.16 没 speed_limit API, 默认值
            })
        except Exception:
            continue
    return out


def get_lane_of(map_, location, fallback_lane_id=None):
    """给 actor location 返回 (road_id, lane_id) 或 None."""
    try:
        wp = map_.get_waypoint(location, project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        return (int(wp.road_id), int(wp.lane_id))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--vehicles", type=int, default=30)
    p.add_argument("--walkers", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/runs")
    p.add_argument("--no-spawn", action="store_true")
    p.add_argument("--collect-lanes", action="store_true",
                   help="强制采 lane 拓扑 (默认 True)")
    p.add_argument("--no-lanes", action="store_true")
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO / args.out / f"phases_{timestamp}_{args.frames}f"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "(none)")
    print(f"[meta] CUDA_VISIBLE_DEVICES = {gpu_env}")
    print(f"[meta] output: {out_dir}")

    timings = {}

    # ===== Phase 1: CARLA extraction =====
    banner("Phase 1: CARLA extraction")
    t0 = time.time()

    try:
        import carla  # noqa
    except ImportError:
        print("[FATAL] cant import carla"); sys.exit(1)

    import random
    random.seed(args.seed)

    print(f"[*] Connecting CARLA {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"[+] server={client.get_server_version()} client={client.get_client_version()}")
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    map_ = world.get_map()

    # --- 1.1 采集 lane 拓扑 (全局,一次即可) ---
    lane_wps = []
    if not args.no_lanes:
        print("[*] Collecting lane waypoints ...")
        try:
            lane_wps = collect_waypoints(world, max_count=2000)
            print(f"[+] {len(lane_wps)} unique driving lanes")
        except Exception as e:
            print(f"[!] cant collect waypoints: {e}")
            lane_wps = []

    # --- 1.2 spawn 交通 ---
    spawned_vehicles = []
    spawned_walkers = []
    if not args.no_spawn:
        print(f"[*] Spawning {args.vehicles} vehicles ...")
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = map_.get_spawn_points()
        used = set()
        for _ in range(args.vehicles):
            for _try in range(10):
                idx = random.randint(0, len(spawn_points) - 1)
                if idx in used:
                    continue
                bp = random.choice(vehicle_bps)
                bp.set_attribute("role_name", "autopilot")
                try:
                    v = world.spawn_actor(bp, spawn_points[idx])
                    v.set_autopilot(True)
                    spawned_vehicles.append(v)
                    used.add(idx)
                    break
                except RuntimeError:
                    continue
        print(f"[+] spawned {len(spawned_vehicles)} vehicles")

        print(f"[*] Spawning {args.walkers} walkers ...")
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        controller_bp = bp_lib.find("controller.ai.walker")
        batches = []
        for _ in range(args.walkers):
            if not walker_bps:
                break
            bp = random.choice(walker_bps)
            nav_loc = world.get_random_location_from_navigation()
            if nav_loc is None:
                continue
            tf = carla.Transform(nav_loc)
            wc = carla.command.SpawnActor(bp, tf)
            wc.then(carla.command.SpawnActor(controller_bp, tf))
            batches.append(wc)
        if batches:
            results = client.apply_batch_sync(batches, True)
            for i in range(0, len(results) - 1, 2):
                if not results[i].error and not results[i+1].error:
                    spawned_walkers.append((results[i].actor_id, results[i+1].actor_id))
        print(f"[+] spawned {len(spawned_walkers)} walkers")

    settings = world.get_settings()
    tick_s = settings.fixed_delta_seconds or 0.05

    print(f"[*] Sampling {args.frames} frames @ {tick_s}s ...")
    from stk.extraction.actor_extractor import extract_all_actors
    from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
    from stk.extraction.weather_extractor import build_environment_snapshot
    from stk.extraction.waypoint_extractor import extract_waypoints, build_lane_topology

    phase1_frames = []
    lanes_static = extract_waypoints(lane_wps)
    scene_rels_static = build_lane_topology(lane_wps)
    print(f"[*] Static lanes={len(lanes_static)}, static lane topology edges={len(scene_rels_static)}")

    for i in range(args.frames):
        world.tick()
        actors = world.get_actors()
        vehicles = actors.filter("vehicle.*")
        walkers = actors.filter("walker.*")
        tls = actors.filter("traffic.traffic_light*")
        weather = world.get_weather()

        actors_list = []
        for v in vehicles:
            t = v.get_transform(); vel = v.get_velocity()
            # 找出当前车辆的 (road_id, lane_id)
            lane_info = None
            try:
                wp = map_.get_waypoint(t.location, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
                if wp is not None:
                    lane_info = (int(wp.road_id), int(wp.lane_id))
            except Exception:
                pass
            actor_dict = {
                "type": "vehicle", "id": str(v.id), "type_id": v.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            }
            if lane_info is not None:
                actor_dict["road_id"] = lane_info[0]
                actor_dict["lane_id"] = lane_info[1]
            actors_list.append(actor_dict)
        for wk in walkers:
            t = wk.get_transform(); vel = wk.get_velocity()
            actors_list.append({
                "type": "walker", "id": str(wk.id), "type_id": wk.type_id,
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
            })
        tl_list = []
        for tl in tls:
            t = tl.get_transform()
            tl_list.append({
                "id": str(tl.id), "state": str(tl.get_state()),
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
            })
        weather_dict = {
            "cloudiness": weather.cloudiness, "precipitation": weather.precipitation,
            "precipitation_deposits": weather.precipitation_deposits,
            "wind_intensity": weather.wind_intensity,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density, "wetness": weather.wetness,
        }
        phase1_frames.append({
            "frame_id": i, "elapsed_seconds": i * tick_s,
            "actors": actors_list, "traffic_lights": tl_list, "weather": weather_dict,
            "waypoints": lane_wps,  # 复用同一份静态车道
        })
        if (i+1) % 20 == 0 or i == 0 or i == args.frames - 1:
            print(f"  frame {i+1:>3}/{args.frames}: v={len(vehicles)} w={len(walkers)} tl={len(tls)}")

    with open(out_dir / "phase1_extraction.json", "w", encoding="utf-8") as f:
        json.dump(phase1_frames, f, ensure_ascii=False, indent=2)
    timings["phase1_extraction"] = time.time() - t0
    print(f"[OK] Phase 1 done ({timings['phase1_extraction']:.1f}s)")

    # ===== Phase 2: Scene layer =====
    banner("Phase 2: Scene layer")
    t0 = time.time()

    from stk.scenario.snapshot_builder import FrameData, build_snapshot
    from stk.scenario.spatial import (
    compute_in_lane, compute_ahead_of,
    compute_beside, compute_nearby_pedestrian,
)

    phase2_frames = []
    for raw in phase1_frames:
        actors = extract_all_actors(raw)
        # 在每个 actor 上加 lane_id (前面已经从 waypoint 拿到了)
        for idx, av in enumerate(actors.get("vehicles", [])):
            src = next((a for a in raw["actors"] if str(a["id"]) == str(av.get("entity_id"))), None)
            if src and "lane_id" in src:
                av["lane_id"] = src["lane_id"]
                av["road_id"] = src.get("road_id", -1)
                av["current_lane_id"] = f"road_{src.get('road_id',0)}_lane_{src['lane_id']}"

        tl = extract_all_traffic_lights(raw.get("traffic_lights", []))
        weather = build_environment_snapshot(raw.get("weather", {}), raw["frame_id"])
        lanes = extract_waypoints(raw.get("waypoints", []))
        topo = build_lane_topology(raw.get("waypoints", []))

        # 用 spatial.py 计算 4 类场景关系 (dict -> SimpleNamespace)

        # 空间关系: 用 types.SimpleNamespace + dict 适配 spatial.py
        spatial_rels = []
        try:
            from types import SimpleNamespace
            vehs_raw = actors.get("vehicles", [])
            peds_raw = actors.get("pedestrians", [])
            # 把每条 dict 转为 SimpleNamespace,同时保留 attrs 作为 dict
            vehs_adapted = []
            for v in vehs_raw:
                sn = SimpleNamespace(**v)
                sn.attrs = dict(v)
                if not hasattr(sn, "entity_id") or sn.entity_id is None:
                    sn.entity_id = str(v.get("entity_id", v.get("id", "")))
                vehs_adapted.append(sn)
            peds_adapted = []
            for p in peds_raw:
                sn = SimpleNamespace(**p)
                sn.attrs = dict(p)
                if not hasattr(sn, "entity_id") or sn.entity_id is None:
                    sn.entity_id = str(p.get("entity_id", p.get("id", "")))
                peds_adapted.append(sn)

            spatial_rels.extend(compute_in_lane(vehs_adapted, lanes, raw["frame_id"]))
            spatial_rels.extend(compute_ahead_of(vehs_adapted, raw["frame_id"]))
            spatial_rels.extend(compute_beside(vehs_adapted, raw["frame_id"]))
            spatial_rels.extend(compute_nearby_pedestrian(vehs_adapted, peds_adapted, raw["frame_id"]))
        except Exception as e:
            import traceback
            fid = raw.get("frame_id", -1)
            print(f"[!] spatial err frame {fid}: {e}")
            import traceback; traceback.print_exc()
            traceback.print_exc()

        # 把 BaseRelation 转成 dict, 供 serialize_graph 使用
        # BaseRelation 是 pydantic BaseModel, 字段直接访问
        spatial_rels_dicts = []
        for r in spatial_rels:
            spatial_rels_dicts.append({
                "src_id": str(getattr(r, "src_id", "")),
                "dst_id": str(getattr(r, "dst_id", "")),
                "relation_type": getattr(r, "relation_type", ""),
                "frame_id": getattr(r, "frame_id", raw["frame_id"]),
            })
        all_scene_rels = topo + spatial_rels_dicts

        fd = FrameData(
            frame_id=raw["frame_id"],
            elapsed_seconds=raw["elapsed_seconds"],
            delta_seconds=tick_s,
            vehicles=actors.get("vehicles", []),
            pedestrians=actors.get("pedestrians", []),
            traffic_lights=tl,
            lanes=lanes,
            weather=raw.get("weather", {}),
            map_name=map_.name,
        )
        try:
            scen_snap, env_snap = build_snapshot(fd)
            scen_dict = scen_snap.__dict__ if hasattr(scen_snap, "__dict__") else str(scen_snap)
            env_dict = env_snap.__dict__ if hasattr(env_snap, "__dict__") else str(env_snap)
        except Exception as e:
            scen_dict = str(e); env_dict = str(e)

        snap_dict = {
            "frame_id": raw["frame_id"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "delta_seconds": tick_s,
            "vehicles": actors.get("vehicles", []),
            "pedestrians": actors.get("pedestrians", []),
            "traffic_lights": tl,
            "lanes": lanes,
            "scene_rels": all_scene_rels,
            "weather": weather,
            "scenario_snapshot": scen_dict,
            "environment_snapshot": env_dict,
        }
        phase2_frames.append(snap_dict)

    with open(out_dir / "phase2_scenario.json", "w", encoding="utf-8") as f:
        json.dump(phase2_frames, f, ensure_ascii=False, indent=2, default=str)

    n_veh = sum(len(s["vehicles"]) for s in phase2_frames)
    n_ped = sum(len(s["pedestrians"]) for s in phase2_frames)
    n_lane = len(phase2_frames[0]["lanes"]) if phase2_frames else 0
    n_srel = sum(len(s["scene_rels"]) for s in phase2_frames)
    timings["phase2_scene"] = time.time() - t0
    print(f"[OK] Phase 2 done ({timings['phase2_scene']:.2f}s) "
          f"veh={n_veh} ped={n_ped} lanes/frame={n_lane} scene_rels total={n_srel}")

    # ===== Phase 3: Behavior + Rules =====
    banner("Phase 3: Behavior + Rules")
    t0 = time.time()
    from stk.behavior.generator import BehaviorRelationGenerator
    from stk.rules.generator import RuleEnforcer

    beh_gen = BehaviorRelationGenerator()
    rule_enf = RuleEnforcer()
    phase3_beh = []
    phase3_rul = []
    for snap in phase2_frames:
        veh_str = [{**v, "entity_id": str(v.get("entity_id", v.get("id", "")))} for v in snap["vehicles"]]
        ped_str = [{**p, "entity_id": str(p.get("entity_id", p.get("id", "")))} for p in snap.get("pedestrians", [])]
        tl_str  = [{**t, "entity_id": str(t.get("entity_id", t.get("id", "")))} for t in snap.get("traffic_lights", [])]

        beh_out = beh_gen.generate(
            frame_id=snap["frame_id"],
            vehicles=veh_str,
            pedestrians=ped_str,
            traffic_lights=tl_str,
            scene_relations=snap.get("scene_rels", []),
        )
        phase3_beh.append({
            "frame_id": snap["frame_id"],
            "n_maneuvers": len(beh_out.get("maneuvers", [])),
            "n_interactions": len(beh_out.get("interactions", [])),
            "n_behavior_rels": len(beh_out.get("behavior_rels", [])),
            "n_cross_layer_rels": len(beh_out.get("cross_layer_rels", [])),
        })

        rule_out = rule_enf.enforce(
            frame_id=snap["frame_id"],
            vehicles=veh_str,
            pedestrians=ped_str,
            traffic_lights=tl_str,
            scene_rels=snap.get("scene_rels", []),
        )
        phase3_rul.append({
            "frame_id": snap["frame_id"],
            "n_violations": len(rule_out["violations"]),
            "n_responsibilities": len(rule_out.get("responsibilities", [])),
        })

    with open(out_dir / "phase3_behavior.json", "w", encoding="utf-8") as f:
        json.dump(phase3_beh, f, ensure_ascii=False, indent=2, default=str)
    with open(out_dir / "phase3_rules.json", "w", encoding="utf-8") as f:
        json.dump(phase3_rul, f, ensure_ascii=False, indent=2, default=str)

    total_man = sum(p["n_maneuvers"] for p in phase3_beh)
    total_int = sum(p["n_interactions"] for p in phase3_beh)
    total_vio = sum(p["n_violations"] for p in phase3_rul)
    timings["phase3_behavior_rules"] = time.time() - t0
    print(f"[OK] Phase 3 done ({timings['phase3_behavior_rules']:.2f}s) "
          f"maneuvers={total_man} interactions={total_int} violations={total_vio}")

    # ===== Phase 4: Incremental update =====
    banner("Phase 4: Incremental update")
    t0 = time.time()
    from stk.dynamic.incremental_updater import IncrementalEngine
    engine = IncrementalEngine()
    phase4_deltas = []
    for snap in phase2_frames:
        delta = engine.process_frame(snap)
        de = getattr(delta, "delta_entities", None)
        dr = getattr(delta, "delta_relations", None)
        da = getattr(delta, "delta_attrs", None)
        re_ = getattr(delta, "rule_events", None)
        phase4_deltas.append({
            "frame_id": snap["frame_id"],
            "n_entities_added": len(de.added) if de else 0,
            "n_entities_removed": len(de.removed) if de else 0,
            "n_entities_unchanged": len(de.unchanged) if de else 0,
            "n_relations_added": len(dr.added) if dr else 0,
            "n_relations_removed": len(dr.removed) if dr else 0,
            "n_attr_changes": len(da) if da else 0,
            "n_rule_events": len(re_) if re_ else 0,
        })
    with open(out_dir / "phase4_deltas.json", "w", encoding="utf-8") as f:
        json.dump(phase4_deltas, f, ensure_ascii=False, indent=2, default=str)
    timings["phase4_delta"] = time.time() - t0
    print(f"[OK] Phase 4 done ({timings['phase4_delta']:.2f}s) - {engine.n_deltas} delta graphs")

    # ===== Phase 5: Storage & graph output =====
    banner("Phase 5: Storage & graph summary")
    t0 = time.time()
    from stk.storage.serializer import serialize_graph

    graph_obj = serialize_graph(phase2_frames, with_relations=True)
    with open(out_dir / "phase5_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_obj, f, ensure_ascii=False, indent=2, default=str)
    g_nodes = len(graph_obj.get("nodes", []))
    g_edges = len(graph_obj.get("edges", []))
    type_counts = Counter(n["type"] for n in graph_obj.get("nodes", []))
    edge_type_counts = Counter(e["type"] for e in graph_obj.get("edges", []))
    print(f"[+] serialize_graph: nodes={g_nodes} edges={g_edges}")
    print(f"[+] node types: {dict(type_counts)}")
    print(f"[+] edge types: {dict(edge_type_counts)}")

    all_ids = set()
    for s in phase2_frames:
        for v in s["vehicles"]:
            all_ids.add(str(v.get("entity_id", v.get("id", ""))))
        for p in s["pedestrians"]:
            all_ids.add(str(p.get("entity_id", p.get("id", ""))))
        for tl in s["traffic_lights"]:
            all_ids.add(str(tl.get("entity_id", tl.get("id", ""))))
        for ln in s["lanes"]:
            all_ids.add(str(ln.get("entity_id", "")))

    summary = {
        "total_unique_entities": len(all_ids),
        "total_frames": args.frames,
        "town": world.get_map().name,
        "lanes_per_frame": n_lane,
        "scene_rels_total": n_srel,
        "phase3_maneuvers": total_man,
        "phase3_interactions": total_int,
        "phase3_violations": total_vio,
        "phase4_deltas": engine.n_deltas,
        "phase5_graph_nodes": g_nodes,
        "phase5_graph_edges": g_edges,
        "phase5_node_types": dict(type_counts),
        "phase5_edge_types": dict(edge_type_counts),
    }
    with open(out_dir / "phase5_kg_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[+] KG summary: {summary}")
    timings["phase5_storage"] = time.time() - t0

    # ===== Cleanup =====
    print("\n[*] Cleaning up spawned actors ...")
    for v in spawned_vehicles:
        try: v.destroy()
        except Exception: pass
    actor_map = {a.id: a for a in world.get_actors()}
    for wid, cid in spawned_walkers:
        if cid in actor_map:
            try: actor_map[cid].destroy()
            except Exception: pass
        if wid in actor_map:
            try: actor_map[wid].destroy()
            except Exception: pass
    print("[+] cleanup done")

    meta = {
        "host": args.host, "port": args.port,
        "frames": args.frames,
        "vehicles_spawned": len(spawned_vehicles),
        "walkers_spawned": len(spawned_walkers),
        "town": world.get_map().name, "tick_s": tick_s,
        "seed": args.seed, "cuda_visible_devices": gpu_env,
        "lane_waypoints_collected": len(lane_wps),
        "timings": timings, "total_time_s": sum(timings.values()),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    banner("Summary")
    print(f"Output: {out_dir}")
    print(f"Total:  {sum(timings.values()):.2f}s")
    for k, v in timings.items():
        print(f"  {k}: {v:.2f}s")
    print()
    print("[OK] All 5 phases passed!")
    print(f"     KG: {g_nodes} nodes, {g_edges} edges")
    print(f"     Lanes: {n_lane}/frame, scene_rels total: {n_srel}")


if __name__ == "__main__":
    main()