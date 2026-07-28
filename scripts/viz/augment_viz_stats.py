#!/usr/bin/env python3
"""augment_viz_stats.py -- 为 viz_stats.json 补齐新增字段 (node_type_dist / edge_type_dist / severity_hist / anomaly_event_log / shard_summary / ego_tail / pair_interact)"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter

_REPO = Path(__file__).resolve().parent.parent.parent
_TOWNS = {
    "Town01_20min": _REPO / "data" / "long_run" / "Town01_20min" / "run_20260724_104140_24000f",
    "Town02_20min": _REPO / "data" / "long_run" / "Town02_20min" / "run_20260724_152253_24000f",
    "Town04_20min": _REPO / "data" / "long_run" / "Town04_20min" / "run_20260724_154452_24000f",
    "Town05_20min": _REPO / "data" / "long_run" / "Town05_20min" / "run_20260724_104822_24000f",
    "Town10HD_20min": _REPO / "data" / "long_run" / "Town10HD_20min" / "run_20260724_105524_24000f",
}
_VIZ = _REPO / "viz_output"


def load_phase5_graph(viz_dir: Path):
    ph = viz_dir / "phase5_graph.json"
    if ph.exists():
        with open(ph) as f:
            return json.load(f)
    # merge shards
    nodes, edges = [], []
    for sh in sorted(viz_dir.glob("graph_*_*.json")):
        with open(sh) as f:
            d = json.load(f)
        nodes.extend(d.get("nodes", []))
        edges.extend(d.get("edges", []))
    if not nodes:
        return None
    # dedupe
    seen_n, seen_e = set(), set()
    uniq_n, uniq_e = [], []
    for n in nodes:
        if n["id"] in seen_n:
            continue
        seen_n.add(n["id"])
        uniq_n.append(n)
    for e in edges:
        key = (e.get("src_id"), e.get("dst_id"), e.get("type"), e.get("first_frame"))
        if key in seen_e:
            continue
        seen_e.add(key)
        uniq_e.append(e)
    return {"nodes": uniq_n, "edges": uniq_e}


def build_new_stats(viz_dir: Path, town_name: str = "") -> dict:
    g = load_phase5_graph(viz_dir)
    if not g:
        return {}
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    node_type_dist = Counter(n.get("type", "?") for n in nodes)
    edge_type_dist = Counter(e.get("type", "?") for e in edges)

    # severity_hist
    severity_hist = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for n in nodes:
        sm = (n.get("attrs") or {}).get("severity_max") or 0
        try:
            bucket = int(float(sm) // 0.1)
        except (ValueError, TypeError):
            bucket = 0
        bucket = max(0, min(bucket, 9))
        severity_hist[f"{bucket/10:.1f}-{(bucket+1)/10:.1f}"] += 1

    # anomaly_event_log from anomaly_log.json (raw, accurate)
    anom_log_path = viz_dir / "anomaly_log.json"
    anomaly_event_log = []
    if anom_log_path.exists():
        with open(anom_log_path) as f:
            raw_log = json.load(f)
        seen = set()
        for entry in raw_log:
            key = (entry.get("frame_id"), entry.get("event_id"))
            if key not in seen:
                seen.add(key)
                anomaly_event_log.append({
                    "frame": entry.get("frame_id"),
                    "event_id": entry.get("event_id"),
                    "type": entry.get("anomaly_type"),
                    "target_actor_id": entry.get("target_actor_id"),
                })

    # shard_summary
    shard_summary = {"total_shards": 0, "avg_nodes_per_shard": 0, "avg_edges_per_shard": 0, "total_frames": 0, "shards": []}
    summary_path = viz_dir / "phase5_kg_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            sm = json.load(f)
        shard_summary = {
            "total_shards": sm.get("n_shards", 0),
            "avg_nodes_per_shard": round(
                sm.get("total_graph_nodes", 0) / max(sm.get("n_shards", 1), 1)
            ),
            "avg_edges_per_shard": round(
                sm.get("total_graph_edges", 0) / max(sm.get("n_shards", 1), 1)
            ),
            "total_frames": sm.get("total_frames", 0),
            "shards": [
                {
                    "idx": s["shard_idx"],
                    "frames": s["frame_end"] - s["frame_start"] + 1,
                    "nodes": s["graph_nodes"],
                    "edges": s["graph_edges"],
                }
                for s in (sm.get("shards") or [])
            ],
        }

    # ego_tail: collect from graph nodes (single snapshot)
    ego_nodes = [n for n in nodes if (n.get("attrs") or {}).get("is_ego")]
    ego_tail = []
    for n in ego_nodes:
        a = n.get("attrs") or {}
        if "location_x" in a and "location_y" in a:
            ego_tail.append({
                "frame": n.get("first_frame"),
                "x": a["location_x"],
                "y": a["location_y"],
                "heading_rad": a.get("heading_rad", 0),
                "speed_ms": a.get("speed", 0) or a.get("speed_ms", 0) or 0,
            })
    ego_tail.sort(key=lambda p: p["frame"] or 0)
    # Dedupe same frame
    seen_f = set()
    deduped = []
    for pt in ego_tail:
        fr = pt["frame"] or 0
        if fr not in seen_f:
            seen_f.add(fr)
            deduped.append(pt)
    ego_tail = deduped
    # If only 1 position from graph, replace with chunk data (sampled every 200 frames)
    if len(ego_tail) <= 1:
        run_dir = _find_run_dir_for_town(_REPO, town_name)
        if run_dir:
            chunk_positions = _extract_ego_from_chunks(run_dir, sample_every=200)
            if len(chunk_positions) > len(ego_tail):
                ego_tail = chunk_positions

    # pair_interact: InteractionEvent type pairs
    ie_types = {"opposite_direction", "following", "overtaking", "changing_lane",
                "blocked_view", "approaching_pedestrian", "yielding_to", "ahead_of",
                "approaching", "standing_still", "beside", "nearby_pedestrian"}
    ie_edges = [e for e in edges if e.get("type") in ie_types]
    pair_counter = Counter()
    for e in ie_edges:
        pair_key = f"{e.get('src_id','?')}→{e.get('dst_id','?')}"
        pair_counter[pair_key] += 1
    pair_interactions = dict(sorted(pair_counter.items(), key=lambda x: -x[1])[:30])

    return {
        "node_type_dist": dict(node_type_dist),
        "edge_type_dist": dict(edge_type_dist),
        "severity_hist": severity_hist,
        "anomaly_event_log": anomaly_event_log,
        "shard_summary": shard_summary,
        "ego_tail": ego_tail,
        "pair_interact": pair_interactions,
    }


def _find_run_dir_for_town(repo: Path, town_name: str):
    """Find the data/long_run/<town>/run_* directory."""
    lr = repo / "data" / "long_run" / town_name
    if not lr.exists():
        return None
    run_dirs = sorted(lr.glob("run_*"))
    return run_dirs[0] if run_dirs else None


def _extract_ego_from_chunks(run_dir: Path, sample_every: int = 200) -> list:
    """Extract ego positions from all chunk files, sampling every N frames."""
    chunks = sorted(run_dir.glob("chunk_*.json"))
    if not chunks:
        return []
    ego_positions = []
    frame_step = 0
    for chunk_path in chunks:
        try:
            with open(chunk_path) as f:
                frames = json.load(f)
        except Exception:
            continue
        for frame_data in frames:
            frame_step += 1
            if frame_step % max(sample_every, 1) != 0:
                continue
            fid = frame_data.get("frame_id", 0)
            actors = frame_data.get("actors", [])
            for a in actors:
                if a.get("is_ego"):
                    loc = a.get("location", {})
                    rot = a.get("rotation", {})
                    vel = a.get("velocity", {})
                    ego_positions.append({
                        "frame": fid,
                        "x": loc.get("x", 0),
                        "y": loc.get("y", 0),
                        "heading_rad": rot.get("yaw", 0),
                        "speed_ms": (vel.get("x", 0) ** 2 + vel.get("y", 0) ** 2) ** 0.5,
                    })
                    break
    ego_positions.sort(key=lambda p: p["frame"])
    return ego_positions


def merge(viz_dir: Path, extra: dict) -> None:
    stats_path = viz_dir / "viz_stats.json"
    data = {}
    if stats_path.exists():
        with open(stats_path) as f:
            data = json.load(f)
    data.update(extra)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    for town_name, run_dir in _TOWNS.items():
        viz_dir = _VIZ / town_name
        if not viz_dir.exists():
            print(f"[skip] {town_name}: viz dir not found")
            continue
        print(f"[*] {town_name} ...")
        extra = build_new_stats(viz_dir, town_name=town_name)
        if not extra:
            print(f"  no data")
            continue
        merge(viz_dir, extra)
        print(f"  node_type_dist: {extra['node_type_dist']}")
        print(f"  severity_hist: {extra['severity_hist']}")
        print(f"  shard_summary: shards={extra['shard_summary']['total_shards']}")
        print(f"  ego_tail: {len(extra['ego_tail'])} positions")
        print(f"  pair_interact: top3={list(extra['pair_interact'].items())[:3]}")


if __name__ == "__main__":
    main()
