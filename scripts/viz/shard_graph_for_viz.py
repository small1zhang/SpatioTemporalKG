#!/usr/bin/env python3
"""shard_graph_for_viz.py -- 将 phase5_graph.json 切为 frame_snapshot_kg.html 所需的 sharded 格式

用法:
    python3 scripts/viz/shard_graph_for_viz.py \\
        --graph data/runs/test_3min_observe_v1/phases_20260723_183240_3600f/phase5_graph.json \\
        --out data/runs/test_3min_observe_v1/phases_20260723_183240_3600f/ \\
        --shard-size 2000

输出:
    out/phase5_kg_summary.json   (含 shards[] 列表)
    out/graph_0001_0_1999.json
    out/graph_0002_2000_3599.json
    ...
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def shard_graph(graph_path: Path, out_dir: Path, shard_size: int = 2000):
    with open(graph_path) as f:
        g = json.load(f)

    nodes = g["nodes"]
    edges = g["edges"]

    # 确定帧范围
    first_frames = [n.get("first_frame", 0) for n in nodes]
    last_frames = [n.get("last_frame", 0) for n in nodes]
    min_frame = min(first_frames)
    max_frame = max(last_frames)
    print(f"[*] Graph: {len(nodes)} nodes, {len(edges)} edges, frames {min_frame}~{max_frame}")

    all_shard_infos = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx, start in enumerate(range(min_frame, max_frame + 1, shard_size), start=1):
        end = min(start + shard_size - 1, max_frame)
        f_start = start
        f_end = end

        # 帧级过滤：节点 / 边在 [start, end] 范围内
        shard_nodes = [
            n for n in nodes
            if n.get("first_frame", 0) <= end and n.get("last_frame", 0) >= start
        ]
        shard_edges = [
            e for e in edges
            if not (e.get("first_frame", 0) > end or e.get("last_frame", 0) < start)
        ]

        shard_id_set = {n["id"] for n in shard_nodes}
        shard_edges = [e for e in shard_edges if e["src_id"] in shard_id_set and e["dst_id"] in shard_id_set]

        shard_data = {"nodes": shard_nodes, "edges": shard_edges}
        fname = f"graph_{shard_idx:04d}_{f_start}_{f_end}.json"
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False)
        sz_mb = path.stat().st_size / 1024 / 1024
        print(f"  [shard {shard_idx}] frames {f_start}-{f_end}: "
              f"{len(shard_nodes)} nodes, {len(shard_edges)} edges → {fname} ({sz_mb:.1f} MB)")

        all_shard_infos.append({
            "shard_idx": shard_idx,
            "frame_start": f_start,
            "frame_end": f_end,
            "frame_count": f_end - f_start + 1,
            "graph_nodes": len(shard_nodes),
            "graph_edges": len(shard_edges),
        })

    # 写 summary
    summary = {
        "output_mode": "sharded",
        "shard_frames": shard_size,
        "total_frames": max_frame - min_frame + 1,
        "n_shards": len(all_shard_infos),
        "total_graph_nodes": len(nodes),
        "total_graph_edges": len(edges),
        "shards": all_shard_infos,
    }
    summary_path = out_dir / "phase5_kg_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[+] summary saved: {summary_path} (n_shards={len(all_shard_infos)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, help="phase5_graph.json 路径")
    p.add_argument("--out", required=True, help="输出目录 (与 phase5_graph.json 同级)")
    p.add_argument("--shard-size", type=int, default=2000, help="每分片帧数")
    args = p.parse_args()

    graph_path = Path(args.graph)
    out_dir = Path(args.out)
    if not graph_path.exists():
        print(f"[FATAL] {graph_path} not found")
        sys.exit(1)

    shard_graph(graph_path, out_dir, shard_size=args.shard_size)
    print("[OK] 分片完成，可用 frame_snapshot_kg.html 查看")


if __name__ == "__main__":
    main()
