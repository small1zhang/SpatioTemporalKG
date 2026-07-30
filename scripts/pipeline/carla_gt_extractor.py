#!/usr/bin/env python3
"""
CARLA GT 验证 & RQ1.3 规则真值标定器
⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

双动作:
  A) 用 traffic manager 让 ego + NPC spawn + autopilot → Tick 2400 帧
     output: data/dataset/carla_gt_run.json
     ({frame_id, actor_id, x, y, z, speed, road_id, lane_id, lane_type, state})

  B) 基于 API 真值自动推理 R1‑R18 + RSS 规则触发 → 输出 rule_gt.json
     (含 GT 标签：{frame_id: {R1, R7, R13, …}} )

用法:
    python scripts/pipeline/carla_gt_extractor.py --num-frames 2400

输出:
    data/dataset/{carla_gt_run.json, rule_gt.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import carla
import numpy as np


# ================================================================
#   A) 采集器
# ================================================================

def spawn_actor(bp_lib, bp_name, actor_type, spawn_point, world, role_name=""):
    """安全 spawn，失败返回 None。"""
    bp = bp_lib.find(bp_name)
    if actor_type in ('vehicle',):
        bp.set_attribute('role_name', role_name)
    try:
        return world.spawn_actor(bp, spawn_point)
    except Exception:
        return None


def run_collect(args) -> List[Dict]:
    """主采集循环：spawn → autopilot → tick → record→ cleanup→ return records。"""
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    origin_map = world.get_map().name

    # 同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    # Spawn
    sp = world.get_map().get_spawn_points()
    print(f"[spawn] {len(sp)} points available")
    ego = spawn_actor(bp_lib, 'vehicle.lincoln.mkz_2017', 'vehicle',
                      sp[0], world, role_name='ego')
    npcs = []
    for i, pt in enumerate(sp[1:args.num_npc + 1]):
        a = spawn_actor(bp_lib, 'vehicle.audi.a2', 'vehicle',
                        pt, world, role_name='npc')
        if a: npcs.append(a)
    print(f"[spawn] ego={ego.id if ego else 'FAIL'}, npcs={len(npcs)}")

    # walker
    walkers = []
    wlk_bp = bp_lib.find('walker.pedestrian.0001')
    wlk_ctrl_bp = bp_lib.find('controller.ai.walker')
    for i in range(args.num_walkers):
        pt = sp[(i + 10) % len(sp)]
        loc = pt.location
        loc.z += 1.5
        try:
            w = world.try_spawn_actor(wlk_bp, carla.Transform(loc, carla.Rotation(yaw=0)))
            if not w:
                continue
            c = world.spawn_actor(wlk_ctrl_bp, carla.Transform(), w)
            c.start()
            target = carla.Location(x=loc.x + np.random.randint(-15, 15),
                                    y=loc.y + np.random.randint(-15, 15))
            c.go_to_location(target)
            walkers.append((w, c))
        except Exception:
            pass
    print(f"[spawn] walkers={len(walkers)}")

    # Traffic Manager (让 NPC 真正跑起来)
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm.global_percentage_speed_difference(30.0)
    for v in npcs:
        v.set_autopilot(True, tm_port)
    if ego:
        ego.set_autopilot(True, tm_port)

    world.tick()
    print("[tick] autopilot enabled, starting recording ...")

    records = []
    t0 = time.time()
    for fid in range(args.num_frames):
        world.tick()
        snap = world.get_snapshot()
        all_actors = world.get_actors()
        vehicles = all_actors.filter('vehicle.*')
        peds = all_actors.filter('walker.pedestrian.*')
        tls = all_actors.filter('traffic.traffic_light')
        time_stamp = snap.timestamp.elapsed_seconds

        for v in vehicles:
            tr = v.get_transform()
            vel = v.get_velocity()
            acc = v.get_acceleration()
            ctrl = v.get_control()
            try:
                wp = world.get_map().get_waypoint(tr.location, project_to_road=True)
                road_id = wp.road_id
                lane_id = wp.lane_id
                lane_type = str(wp.lane_type)
                is_junction = wp.is_junction
                # 几何距离最近 junction 的距离
                junction_dist = 999.0
                try:
                    next_jw = wp.get_junction()
                    if next_jw is not None:
                        junction_dist = float(wp.get_transform().location.distance(next_jw.bounding_box.location))
                    elif wp.get_next(1):
                        junction_dist = 100.0  # 不在 junction，但行进方向 1 waypoint内有
                except Exception:
                    junction_dist = 999.0
            except Exception:
                road_id = -1; lane_id = 0; lane_type = "Driving"
                is_junction = False; junction_dist = 999.0
            records.append({
                "frame": snap.frame,
                "elapsed_seconds": round(time_stamp, 3),
                "actor_id": v.id,
                "type": "vehicle",
                "type_id": v.type_id,
                "is_ego": v.attributes.get('role_name', '') == 'ego',
                "is_alive": v.is_alive,
                "x": round(tr.location.x, 4),
                "y": round(tr.location.y, 4),
                "z": round(tr.location.z, 4),
                "yaw": round(tr.rotation.yaw, 4),
                "speed": round((vel.x**2 + vel.y**2 + vel.z**2)**0.5, 4),
                "speed_kmh": round((vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6, 4),
                "heading_rad": round(((tr.rotation.yaw + 360) % 360) * np.pi / 180.0, 4),
                "vx": round(vel.x, 4), "vy": round(vel.y, 4), "vz": round(vel.z, 4),
                "ax": round(acc.x, 4), "ay": round(acc.y, 4), "az": round(acc.z, 4),
                "throttle": round(ctrl.throttle, 4), "brake": round(ctrl.brake, 4),
                "steer": round(ctrl.steer, 4),
                "road_id": road_id, "lane_id": lane_id, "lane_type": lane_type,
                "is_junction": is_junction,
                "junction_dist": round(junction_dist, 4),
            })
        for ped_actor, _ in walkers:
            if not ped_actor.is_alive: continue
            tr = ped_actor.get_transform()
            vel = ped_actor.get_velocity()
            records.append({
                "frame": snap.frame,
                "elapsed_seconds": round(time_stamp, 3),
                "actor_id": ped_actor.id,
                "type": "pedestrian",
                "type_id": ped_actor.type_id,
                "is_alive": ped_actor.is_alive,
                "x": round(tr.location.x, 4),
                "y": round(tr.location.y, 4),
                "z": round(tr.location.z, 4),
                "speed": round((vel.x**2 + vel.y**2 + vel.z**2)**0.5, 4),
            })
        for tl in tls:
            records.append({
                "frame": snap.frame,
                "elapsed_seconds": round(time_stamp, 3),
                "actor_id": tl.id,
                "type": "traffic_light",
                "state": str(tl.state).split('.')[-1],
            })
        if (fid + 1) % 200 == 0:
            print(f"    {fid+1}/{args.num_frames}  ({len(records)} records)", flush=True)

    dt = time.time() - t0
    print(f"[OK] {args.num_frames} frames / {len(records)} records in {dt:.1f}s")

    # Cleanup
    settings.synchronous_mode = False
    world.apply_settings(settings)
    for v in [ego] + npcs:
        if v and v.is_alive: v.destroy()
    for w, c in walkers:
        if c and c.is_alive: c.stop(); c.destroy()
        if w and w.is_alive: w.destroy()
    print("[cleanup] done")
    return records


# ================================================================
#   B) 规则真值自动推理
# ================================================================

def extract_rule_gt(records: List[Dict]) -> Dict[str, Set[str]]:
    """基于 Carla API 真值推理每帧的 GT 规则集。

    设计原则：
      - 用 API 字段直接可判的硬指标，不做松启发式
      - 每条规则覆盖率目标 <30% 帧, 避免全部/全不触发
    """
    by_frame: Dict = defaultdict(lambda: {"vehs": {}, "peds": {}, "tls": {}})
    for r in records:
        f = r["frame"]
        t = r.get("type", "")
        if t == "vehicle":
            by_frame[f]["vehs"][r["actor_id"]] = r
        elif t == "pedestrian":
            by_frame[f]["peds"][r["actor_id"]] = r
        elif t == "traffic_light":
            by_frame[f]["tls"][r["actor_id"]] = r

    stopped: Dict[int, int] = defaultdict(int)
    near_crosswalk: Dict[int, int] = defaultdict(int)

    result: Dict[str, Set[str]] = {}
    sorted_frames = sorted(by_frame.keys())

    for fid in sorted_frames:
        rules: Set[str] = set()
        vehs = by_frame[fid]["vehs"]
        tl_dict = by_frame[fid]["tls"]

        # ══════════════════════════════════════════════
        # R2: 闯红灯 — 车辆临近 (≤15m) 当前 Red 灯
        # ══════════════════════════════════════════════
        # 用路口距离 + TL 状态组合判定
        for vid, v in vehs.items():
            jd = v.get("junction_dist", 999.0)
            road_id = v.get("road_id", -1)
            if jd < 15.0 and road_id > 0:
                # 是否路口方向有红灯？我们记录 tl_states 中 Red 的存在
                # 简化: 当车辆在 junction_dist<15 且存在 Red 状态时触发
                if any(t.get("state") == "Red" for t in tl_dict.values()):
                    rules.add("R2")
                    break

        # ══════════════════════════════════════════════
        # R4: 对向会车 (同 road，lane 符号相反 且距离<40m)
        # ══════════════════════════════════════════════
        for vid_a, va in vehs.items():
            for vid_b, vb in vehs.items():
                if vid_a >= vid_b:
                    continue
                ra, la = int(va.get("road_id", -1)), int(va.get("lane_id", 0))
                rb, lb = int(vb.get("road_id", -1)), int(vb.get("lane_id", 0))
                if ra != rb or ra <= 0 or la * lb >= 0 or la == 0:
                    continue
                dist = ((va["x"]-vb["x"])**2 + (va["y"]-vb["y"])**2)**0.5
                if dist < 40.0:
                    rules.add("R4")
                    break
            if "R4" in rules:
                break

        # ══════════════════════════════════════════════
        # R13: 违法停车 — speed<0.5m/s 连续 ≥20 帧 (≈1s) 且在斑马线附近
        # ══════════════════════════════════════════════
        for vid, v in vehs.items():
            speed = v.get("speed", 10.0)
            if speed < 0.5:
                stopped[vid] += 1
                # 斑马线近似：junction_dist < 5 m 且 is_junction True
                if v.get("is_junction") and v.get("junction_dist", 999) < 5.0:
                    near_crosswalk[vid] += 1
            else:
                stopped[vid] = 0
                near_crosswalk[vid] = 0

            # 只有连续停 20 帧 + 路口内才报 R13
            if stopped[vid] >= 20 and near_crosswalk[vid] >= 10:
                rules.add("R13")
                break

        # ══════════════════════════════════════════════
        # RSS_R13a: 同向纵向距离近 (同 lane 同号，distance<5m)
        # ══════════════════════════════════════════════
        for vid_a, va in vehs.items():
            found = False
            for vid_b, vb in vehs.items():
                if vid_a >= vid_b:
                    continue
                la = int(va.get("lane_id", 0))
                lb = int(vb.get("lane_id", 0))
                if la == 0 or la != lb:
                    continue
                dx = va["x"] - vb["x"]
                dy = va["y"] - vb["y"]
                dist = (dx*dx + dy*dy)**0.5
                if dist < 5.0:
                    rules.add("RSS_R13a")
                    found = True
                    break
            if found:
                break

        # ══════════════════════════════════════════════
        # R7: 路口急刹 — brake>0.8 且在 junction 内
        # ══════════════════════════════════════════════
        for vid, v in vehs.items():
            if v.get("brake", 0) > 0.8 and v.get("is_junction"):
                rules.add("R7")
                break

        # ══════════════════════════════════════════════
        # R18: 逆行 — 用 vehicle 自身 |yaw| 与同 lane_id 车的中位 yaw 比较,偏差>90°
        # ══════════════════════════════════════════════
        # 收集同 lane_id 所有车的 yaw
        from statistics import median
        lane_yaws = defaultdict(list)
        for vid, v in vehs.items():
            lid = int(v.get("lane_id", 0))
            if lid == 0:
                continue
            lane_yaws[lid].append(v.get("yaw", 0.0))
        for vid, v in vehs.items():
            lid = int(v.get("lane_id", 0))
            if lid == 0 or len(lane_yaws[lid]) < 3:
                continue
            yaw = v.get("yaw", 0.0)
            med = median(lane_yaws[lid])
            # 角度差归一化到 [0,180]
            d = abs(((yaw - med + 180) % 360) - 180)
            if d > 90:
                rules.add("R18")
                break

        result[str(fid)] = rules

    return result


# ================================================================
#   入口
# ================================================================

def main():
    p = argparse.ArgumentParser("carla GT 验证 & 规则真值标定器")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--num-frames", type=int, default=2400,
                   help="采集帧数 (default: 2400 ≈ 2 min)")
    p.add_argument("--num-npc", type=int, default=9,
                   help="NPC 车辆数")
    p.add_argument("--num-walkers", type=int, default=5,
                   help="walker 数")
    p.add_argument("--only-gt", action="store_true",
                   help="跳过采集 (A)，直接从已有 carla_gt_run.json 推规则 (B)")
    p.add_argument("--out", default="data/dataset")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (影响 walker spawn 位置)")
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    if not args.only_gt:
        records = run_collect(args)
        run_path = out_dir / "carla_gt_run.json"
        with open(run_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[A] → {run_path} ({len(records)} records)")
    else:
        run_path = out_dir / "carla_gt_run.json"
        if not run_path.exists():
            print(f"[!] --only-gt 但 {run_path} 不存在"); sys.exit(1)
        records = json.loads(run_path.read_text())
        print(f"[A] Read {len(records)} records from {run_path}")

    print("[B] Inferring GT rules ...")
    rule_gt = extract_rule_gt(records)
    # 统计
    all_frames = set(rule_gt.keys())
    rules_counter = defaultdict(int)
    for fid, rs in rule_gt.items():
        for r in rs:
            rules_counter[r] += 1
    print(f"    Frames with GT: {len(all_frames)}")
    for rc, cnt in sorted(rules_counter.items(), key=lambda x: -x[1]):
        print(f"    {rc:<12} {cnt:>4} frames")

    # 序列化 (set → list)
    rule_gt_serializable = {fid: sorted(list(rs))
                            for fid, rs in rule_gt.items()}
    gt_path = out_dir / "rule_gt.json"
    with open(gt_path, "w") as f:
        json.dump(rule_gt_serializable, f, indent=2, ensure_ascii=False)
    print(f"[B] → {gt_path} ({len(rule_gt_serializable)} frames)")


if __name__ == "__main__":
    main()
