#!/usr/bin/env python3
"""独立 Phase5 序列化脚本: 从 pipeline 产出的临时文件流式构造 KG.

读:
  - data/long_run/<run_dir>/phase5/_phase2_tmp.jsonl  (per-frame snapshots)
  - data/long_run/<run_dir>/phase5/_ruleouts_tmp.jsonl  (per-frame rule_out)
  - data/long_run/<run_dir>/phase5/pipeline_checkpoint.json  (cross-chunk 累积容器)

写:
  - data/long_run/<run_dir>/phase5/phase5_graph.json
  - data/long_run/<run_dir>/phase5/phase5_kg_summary.json
"""
import json
import sys
from pathlib import Path
from collections import Counter

# 一定要在 import serializer 之前设置 PYTHONPATH
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from stk.storage.serializer import serialize_graph


def main(run_dir: str):
    run = Path(run_dir)
    phase5 = run / "phase5"

    phase2_path = phase5 / "_phase2_tmp.jsonl"
    ruleouts_path = phase5 / "_ruleouts_tmp.jsonl"
    ckpt_path = phase5 / "pipeline_checkpoint.json"

    if not phase2_path.exists():
        raise SystemExit(f"missing {phase2_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"missing {ckpt_path}")

    print(f"[*] reading checkpoint: {ckpt_path}")
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    all_maneuvers_raw = ckpt.get("all_maneuvers_raw", [])
    all_interactions_raw = ckpt.get("all_interactions_raw", [])
    all_behavior_rels_raw = ckpt.get("all_behavior_rels_raw", [])
    all_cross_layer_rels_raw = ckpt.get("all_cross_layer_rels_raw", [])
    print(f"    maneuvers={len(all_maneuvers_raw)}, interactions={len(all_interactions_raw)}, "
          f"beh_rels={len(all_behavior_rels_raw)}, cross_rels={len(all_cross_layer_rels_raw)}")

    print(f"[*] streaming rule_out from disk")
    rule_out_list = []
    with open(ruleouts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rule_out_list.append(json.loads(line))
    print(f"    rule_out records: {len(rule_out_list)}")

    print(f"[*] streaming phase2 frames into serializer (generator, 不一次 materialize)")
    # 关键: 传 generator, 但 serializer 内部仍会 list() — 这次我们试单一 materialize
    # 同时监控内存
    import resource

    def _phase2_stream(p):
        with open(p) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    yield json.loads(line)

    print(f"    calling serialize_graph ...")
    graph_obj = serialize_graph(
        _phase2_stream(phase2_path), with_relations=True,
        maneuvers=all_maneuvers_raw,
        interactions=all_interactions_raw,
        behavior_rels=all_behavior_rels_raw,
        cross_layer_rels=all_cross_layer_rels_raw,
        rule_out=rule_out_list,
    )

    g_nodes = len(graph_obj.get("nodes", []))
    g_edges = len(graph_obj.get("edges", []))
    type_counts = Counter(n["type"] for n in graph_obj.get("nodes", []))
    edge_type_counts = Counter(e["type"] for e in graph_obj.get("edges", []))

    print(f"[+] serialize_graph: nodes={g_nodes} edges={g_edges}")
    print(f"[+] node types: {dict(type_counts)}")
    print(f"[+] edge types: {dict(edge_type_counts)}")

    out_path = phase5 / "phase5_graph.json"
    print(f"[*] writing {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_obj, f, ensure_ascii=False, indent=2, default=str)
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"[+] phase5_graph.json saved ({sz:.1f} MB)")

    # 释放 graph_obj 内存
    del graph_obj
    del rule_out_list

    summary = {
        "total_frames": ckpt.get("total_frames", 0),
        "chunks_processed": ckpt.get("chunk_idx", 0) + 1,
        "graph_nodes": g_nodes,
        "graph_edges": g_edges,
        "node_types": dict(type_counts),
        "edge_types": dict(edge_type_counts),
        "beh_gen_stats": ckpt.get("beh_gen_stats", {}),
    }
    summary_path = phase5 / "phase5_kg_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[+] {summary_path} saved")

    # 清理临时文件
    print(f"[*] cleaning temp files")
    try:
        phase2_path.unlink()
        ruleouts_path.unlink()
        print(f"[+] temp removed")
    except Exception as e:
        print(f"[!] clean fail: {e}")

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[OK] Phase5 done. peak RSS: {peak_kb/1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
