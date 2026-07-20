#!/usr/bin/env python3
"""export_viz_data.py -- 导出批量采集数据为可视化 JSON"""
import json
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent.parent
OUT = _REPO / "viz_output"
OUT.mkdir(exist_ok=True)
BATCH = _REPO / "data" / "runs" / "batch"

def main():
    print("[*] Loading results...")
    results = []
    for f in sorted(BATCH.glob("*/S*/result.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    print(f"  Found {len(results)} results")

    print("[*] Loading graph and trajectory data...")
    graph_data = {}
    for result_dir in sorted(BATCH.glob("*/S*")):
        map_name = result_dir.parent.name
        scenario = result_dir.name
        phases_dirs = sorted(result_dir.glob("phases_*"))
        if not phases_dirs:
            continue
        pd = phases_dirs[-1]
        entry = {"map": map_name, "scenario": scenario}

        kg_path = pd / "phase5_kg_summary.json"
        if kg_path.exists():
            with open(kg_path) as f:
                entry["kg_summary"] = json.load(f)

        graph_path = pd / "phase5_graph.json"
        if graph_path.exists():
            with open(graph_path) as f:
                g = json.load(f)
            entry["graph"] = {"nodes": g["nodes"][:500], "edges": g["edges"][:2000]}

        frames_path = pd / "phase2_scenario.json"
        if frames_path.exists():
            with open(frames_path) as f:
                frames = json.load(f)
            trajectories = defaultdict(list)
            for frame in frames[:100]:
                fid = frame.get("frame_id", 0)
                t = frame.get("elapsed_seconds", fid * 0.05)
                for v in frame.get("vehicles", []):
                    trajectories[str(v.get("entity_id", ""))].append({
                        "f": fid, "t": round(t, 3),
                        "x": round(v.get("location_x", 0), 2),
                        "y": round(v.get("location_y", 0), 2),
                        "s": round(v.get("speed", 0), 2),
                    })
                for p in frame.get("pedestrians", []):
                    trajectories[str(p.get("entity_id", ""))].append({
                        "f": fid, "t": round(t, 3),
                        "x": round(p.get("location_x", 0), 2),
                        "y": round(p.get("location_y", 0), 2),
                        "s": round(p.get("speed", 0), 2),
                    })
            entry["trajectories"] = dict(trajectories)
            entry["weather_tl"] = [
                {"f": f.get("frame_id"), "w": f.get("weather", {})}
                for f in frames[:100]
            ]
        graph_data[f"{map_name}/{scenario}"] = entry
    print(f"  Loaded {len(graph_data)} scenario datasets")

    # Summary
    by_map = defaultdict(list)
    by_scenario = defaultdict(list)
    for r in results:
        by_map[r["map"]].append(r)
        by_scenario[r["scenario"]].append(r)

    tier_map = {"S00":"A","S01":"A","S02":"A","S10":"B","S11":"B","S12":"B","S13":"B",
                "S20":"C","S21":"C","S22":"C","S30":"D","S31":"D","S32":"D","S33":"D"}

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.get("status") == "PASS"),
        "maps": sorted(by_map.keys()),
        "scenarios": sorted(by_scenario.keys()),
        "by_map": {m: {
            "n": len(ts),
            "nodes": round(sum(r.get("graph_nodes",0) for r in ts)/len(ts)),
            "edges": round(sum(r.get("graph_edges",0) for r in ts)/len(ts)),
            "time": round(sum(r.get("elapsed_s",0) for r in ts)/len(ts), 1),
        } for m, ts in by_map.items()},
        "by_scenario": {s: {
            "n": len(ts), "tier": tier_map.get(s, "?"),
            "nodes": round(sum(r.get("graph_nodes",0) for r in ts)/len(ts)),
            "edges": round(sum(r.get("graph_edges",0) for r in ts)/len(ts)),
            "time": round(sum(r.get("elapsed_s",0) for r in ts)/len(ts), 1),
        } for s, ts in by_scenario.items()},
    }

    results_table = [{
        "map": r["map"], "scenario": r["scenario"], "status": r["status"],
        "nodes": r.get("graph_nodes", 0), "edges": r.get("graph_edges", 0),
        "time": r.get("elapsed_s", 0),
    } for r in results]

    output = {"summary": summary, "results": results_table, "scenarios": graph_data}
    out_path = OUT / "viz_data.json"
    print(f"[*] Writing {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, default=str)
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"[OK] Exported {sz:.1f} MB to {out_path}")

if __name__ == "__main__":
    main()
