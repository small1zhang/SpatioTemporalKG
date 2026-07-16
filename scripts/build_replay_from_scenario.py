"""
scripts/build_replay_from_scenario.py  -  将 scenario_library 14 场景转成
scene_graph_*.json 供 dashboard 离线回放
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 确保项目根在 sys.path 中
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from stk.scenario.scenario_library import all_scenarios, get_scenario
from stk.rules.generator import RuleEnforcer


def build_replay_json(scenario_id: str) -> List[Dict[str, Any]]:
    """将 scenario_library 的一个场景转成 scene_graph JSON 序列."""
    frames = get_scenario(scenario_id)
    if not frames:
        return []
    n_frames = len(frames)
    enforcer = RuleEnforcer()

    outputs = []
    for i, fd in enumerate(frames):
        progress = i / max(1, n_frames - 1)

        # ── 计算 Risk 等级 ──────────────────────────────────────
        # 从场景数据提取距离/TTC
        risk_level = "LOW"
        reasons = []
        min_dist = float("inf")
        min_ttc = float("inf")

        for v in fd.vehicles:
            for p in fd.pedestrians:
                dx = v["location_x"] - p["location_x"]
                dy = v["location_y"] - p["location_y"]
                dist = (dx*dx + dy*dy)**0.5
                if dist < min_dist:
                    min_dist = dist
                # 假设行人速度 1.4 m/s
                rel_speed = v.get("speed", 8.0) + p.get("speed", 1.4)
                ttc = dist / max(rel_speed, 0.1)
                if ttc < min_ttc:
                    min_ttc = ttc
            for v2 in fd.vehicles:
                if v["entity_id"] == v2["entity_id"]:
                    continue
                dx = v["location_x"] - v2["location_x"]
                dy = v["location_y"] - v2["location_y"]
                dist = (dx*dx + dy*dy)**0.5
                if dist < min_dist and dist > 0.5:
                    min_dist = dist
                rel_speed = abs(v.get("speed", 8.0) - v2.get("speed", 8.0))
                ttc = dist / max(rel_speed, 0.1)
                if ttc < min_ttc and ttc > 0:
                    min_ttc = ttc

        # 天气因素
        weather = fd.weather or {}
        fog = weather.get("fog_density", 0)
        rain = weather.get("precipitation", 0)
        night = weather.get("sun_altitude_angle", 60) < 0

        rss_profile = "Urban-Dry"
        response_time = 0.5
        accel_max = 3.5
        brake_min = 4.0
        semantic_margin = 0.0
        explore_speed = 5.0

        if fog > 10 or rain > 20:
            rss_profile = "Urban-Wet"
            response_time = 0.8
            accel_max = 2.0
            brake_min = 2.5
            semantic_margin = 1.0
            explore_speed = 4.0
            if risk_level == "LOW":
                risk_level = "MEDIUM"
                reasons.append("Wet road/rain: lower accel and braking assumptions")

        if night:
            rss_profile = "Night-Reduced"
            response_time = 1.0
            semantic_margin = 2.0

        if min_dist < 18:
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
            reasons.append(f"Nearby vulnerable road user: distance={min_dist:.1f}m")

        if min_ttc < 3.0:
            risk_level = "HIGH"
            response_time = 1.2
            brake_min = max(brake_min, 4.5)
            reasons.append(f"Low TTC conflict: ttc={min_ttc:.2f}s")
            if "Nearby" not in str(reasons):
                reasons.append(f"Close proximity: distance={min_dist:.1f}m")

        # 遮挡模拟 (基于场景特征)
        occlusion_active = False
        hidden_active = False
        static_actor = None
        for lane in fd.lanes:
            for v in fd.vehicles:
                if v.get("heading_rad", 0) != 0:
                    continue

        # 有 static 实体或场景名含遮挡含义
        occluder_id = ""
        hidden_type = ""

        # ── 构建 node 列表 ──────────────────────────────────────
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Ego 车辆（取第一辆为 ego）
        ego_v = fd.vehicles[0] if fd.vehicles else {"entity_id":"ego","location_x":0,"location_y":0,"speed":0,"heading_rad":0.0}
        nodes.append({
            "id": "Ego",
            "type": "ego_vehicle",
            "speed_mps": ego_v.get("speed", 0),
            "speed": ego_v.get("speed", 0),
            "location_x": ego_v.get("location_x", 0),
            "location_y": ego_v.get("location_y", 0),
            "is_junction": bool(fd.traffic_lights),
        })

        # Environment
        nodes.append({
            "id": "Environment",
            "type": "environment",
            "precipitation": rain,
            "wetness": weather.get("wetness", 0),
            "fog_density": fog,
            "sun_altitude_angle": weather.get("sun_altitude_angle", 60),
            "phase": f"Frame {i}",
        })

        # Lanes
        for lane in fd.lanes:
            lid = lane.get("entity_id", f"lane_{lane.get('road_id','?')}_{lane.get('lane_id','?')}")
            nodes.append({
                "id": lid,
                "type": "lane",
                "road_id": lane.get("road_id", 0),
                "lane_id": lane.get("lane_id", 0),
                "center_x": lane.get("center_x", 0),
                "center_y": lane.get("center_y", 0),
                "length": lane.get("length", 100),
                "speed_limit": lane.get("speed_limit", 15),
            })
            edges.append({
                "source": "Ego", "target": lid,
                "relation": "ON_LANE",
            })

        # Vehicles (非 ego)
        for v in fd.vehicles:
            if v["entity_id"] == ego_v["entity_id"]:
                continue
            nodes.append({
                "id": v["entity_id"],
                "type": "vehicle",
                "speed_mps": v.get("speed", 0),
                "speed": v.get("speed", 0),
                "location_x": v.get("location_x", 0),
                "location_y": v.get("location_y", 0),
                "heading_rad": v.get("heading_rad", 0),
                "vehicle_type": v.get("vehicle_type", "vehicle"),
            })
            dx = ego_v.get("location_x",0) - v.get("location_x",0)
            dy = ego_v.get("location_y",0) - v.get("location_y",0)
            dist = (dx*dx + dy*dy)**0.5
            edges.append({
                "source": "Ego", "target": v["entity_id"],
                "relation": "NEAR_BY", "distance": dist,
            })
            # SAME_LANE 如果 heading_rad 接近
            h_diff = abs(ego_v.get("heading_rad",0) - v.get("heading_rad",0))
            if h_diff < 0.5 or abs(h_diff - math.pi) < 0.5:
                edges.append({
                    "source": "Ego", "target": v["entity_id"],
                    "relation": "SAME_LANE",
                })

        # Pedestrians
        for p in fd.pedestrians:
            nodes.append({
                "id": p["entity_id"],
                "type": "pedestrian",
                "speed_mps": p.get("speed", 0),
                "speed": p.get("speed", 0),
                "location_x": p.get("location_x", 0),
                "location_y": p.get("location_y", 0),
                "action": p.get("action", ""),
                "ttc_s": min_ttc if min_ttc != float("inf") else 10.0,
            })
            dx = ego_v.get("location_x",0) - p.get("location_x",0)
            dy = ego_v.get("location_y",0) - p.get("location_y",0)
            dist = (dx*dx + dy*dy)**0.5
            edges.append({
                "source": "Ego", "target": p["entity_id"],
                "relation": "NEAR_BY", "distance": dist,
            })
            edges.append({
                "source": "Ego", "target": p["entity_id"],
                "relation": "APPROACHING", "ttc_s": min_ttc if min_ttc != float("inf") else 10.0,
            })

        # Traffic lights
        for tl in fd.traffic_lights:
            nodes.append({
                "id": tl["entity_id"],
                "type": "traffic_light",
                "state": tl.get("state", "Green"),
                "position_x": tl.get("position_x", 0),
                "position_y": tl.get("position_y", 0),
            })
            edges.append({
                "source": "Ego", "target": tl["entity_id"],
                "relation": "CONTROLLED_BY",
                "state": tl.get("state", "Green"),
            })

        # ── 语义层逻辑 (仿照参考代码) ────────────────────────────
        road_wet = weather.get("wetness", 0) > 20
        if road_wet:
            nodes.append({
                "id": "RoadSurface:wetness",
                "type": "road_surface_state",
                "wetness": weather.get("wetness", 0),
                "friction_level": "medium_wet" if weather.get("wetness", 0) >= 60 else "high",
            })
            edges.append({
                "source": "Environment", "target": "RoadSurface:wetness",
                "relation": "HAS_ROAD_SURFACE_STATE",
                "wetness": weather.get("wetness", 0),
            })

        if rss_profile != "Urban-Dry":
            nodes.append({
                "id": "MicroODD",
                "type": "micro_odd",
                "profile_name": rss_profile,
                "confidence": 0.92,
                "phase": f"Frame {i}",
            })
            if road_wet:
                edges.append({
                    "source": "RoadSurface:wetness", "target": "MicroODD",
                    "relation": "ACTIVE_MICRO_ODD",
                    "profile": rss_profile,
                })
            else:
                edges.append({
                    "source": "Environment", "target": "MicroODD",
                    "relation": "ACTIVE_MICRO_ODD",
                    "profile": rss_profile,
                })

            nodes.append({
                "id": f"RSSProfile:{rss_profile}",
                "type": "rss_profile",
                "profile_name": rss_profile,
            })
            edges.append({
                "source": "MicroODD", "target": f"RSSProfile:{rss_profile}",
                "relation": "ACTIVATES_PROFILE",
                "profile": rss_profile,
            })
            edges.append({
                "source": f"RSSProfile:{rss_profile}", "target": "Ego",
                "relation": "CONSTRAINS",
                "response_time_s": response_time,
            })

        # SafetyEvent / Fallback (HIGH risk)
        if risk_level == "HIGH":
            nodes.append({
                "id": f"SafetyEvent:rss_margin_{i}",
                "type": "safety_event",
                "event_type": "RSS_MARGIN_VIOLATION",
                "tick": i,
            })
            nodes.append({
                "id": f"FallbackAction:low_speed_{i}",
                "type": "fallback_action",
                "target_speed_mps": explore_speed,
            })
            ped_id = fd.pedestrians[0]["entity_id"] if fd.pedestrians else "ego"
            edges.append({
                "source": f"SafetyEvent:rss_margin_{i}", "target": "Ego",
                "relation": "EXPLAINS_RISK",
                "level": risk_level,
            })
            edges.append({
                "source": f"SafetyEvent:rss_margin_{i}", "target": f"FallbackAction:low_speed_{i}",
                "relation": "EXPLAINS_ACTION",
            })
            edges.append({
                "source": f"FallbackAction:low_speed_{i}", "target": "Ego",
                "relation": "CONSTRAINS",
            })

        # ── 裁剪 risk 结构 ──────────────────────────────────────
        risk = {
            "level": risk_level,
            "reasons": reasons or ["Nominal driving"],
            "response_time": response_time,
            "accel_max": accel_max,
            "brake_min": brake_min,
            "brake_front_max": 8.0,
            "active_profile": rss_profile,
            "semantic_margin_m": semantic_margin,
            "explore_speed_mps": explore_speed,
            "evidence": [
                {"fact": "scenario_distance", "min_dist_m": round(min_dist, 1) if min_dist != float("inf") else 100.0},
                {"fact": "weather", "rain": rain, "fog": fog, "night": night},
            ],
        }

        outputs.append({
            "tick": i,
            "phase": f"Frame {i}  {scenario_id}",
            "scenario_id": scenario_id,
            "risk": risk,
            "nodes": nodes,
            "edges": edges,
        })

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=None,
                        help="场景 ID 列表 (默认全部 14 个)")
    parser.add_argument("--out", default="data/replay_json",
                        help="输出目录 (相对于仓库根)")
    args = parser.parse_args()

    scenes = all_scenarios()
    target_ids = args.ids if args.ids else sorted(scenes.keys())
    out_dir = Path(_REPO) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in target_ids:
        if sid not in scenes:
            print(f"[skip] unknown id: {sid}")
            continue
        seq = build_replay_json(sid)
        if not seq:
            print(f"[skip] {sid}: no frames")
            continue
        # 每个场景存为一个 JSON (含所有帧)
        for i, payload in enumerate(seq):
            fname = out_dir / f"scene_graph_{i:06d}_{payload['risk']['level']}.json"
            # 同时把 scenario_id 编码进文件名
        # 实际按 tick 分开存 (参考代码读取方式)
        for i, payload in enumerate(seq):
            level = payload["risk"]["level"]
            fname = out_dir / sid / f"scene_graph_{i:06d}_{level}.json"
            fname.parent.mkdir(parents=True, exist_ok=True)
            fname.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {sid}: {len(seq)} frames -> {out_dir / sid}")

    print(f"\n[done] 共 {len(target_ids)} 个场景")


if __name__ == "__main__":
    main()
