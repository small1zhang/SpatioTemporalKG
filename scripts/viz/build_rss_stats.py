#!/usr/bin/env python3
"""build_rss_stats.py -- 为可视化 HTML 增加 RSS 板块所需的聚合数据

读取 phase5_graph.json + 所有 shard 文件 (graph_XXXX_*.json)，
扫描所有 SafetyViolation / ResponsibilityAssignment 节点 + violates/responsibleFor 边，
生成/合并到 viz_stats.json 中新增以下字段:

    rss_dist: {
        "by_code":  { "<rule_code>": {"count": N, "sev_max": f, "sev_avg": f, "fired_total": N, "layer": "..."} },
        "by_layer": { "<rule_layer>": N },
        "resp_top": [ {"actor_id": "...", "count": N, "reasons": {...} }, ... ],
        "fired_time": { "<rule_code>": [ (frame_start, frame_end, sev_max) ... ] },
        "pair_count": { "<src>_<dst>": N },
        "total_sv":  N,
        "total_resp": N,
        "total_violates_edge": N,
        "total_responsibleFor_edge": N,
    }

用法:
    python3 scripts/viz/build_rss_stats.py --viz-dir viz_output/Town01_20min
    python3 scripts/viz/build_rss_stats.py --all
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict, Counter


# RSS / TrafficLaw 规则码中文名映射 (尽量覆盖已观测到的码)
RULE_CODE_DESC = {
    # TrafficLaw 层
    "R4":  ("TrafficLaw", "对向会车违规 (opposite meeting)"),
    "R5":  ("TrafficLaw", "红灯闯行"),
    "R6":  ("TrafficLaw", "路口未让行"),
    "R7":  ("TrafficLaw", "变道未让行"),
    "R8":  ("TrafficLaw", "禁行方向行驶"),
    # RSS 层
    "R13a": ("RSS", "RSS 纵向安全距离不足"),
    "R13b": ("RSS", "RSS 纵向危险制动"),
    "R14a": ("RSS", "RSS 侧向危险距离"),
    "R14b": ("RSS", "RSS 侧向危险并道"),
    "R15":  ("RSS", "RSS 路权优先级错误"),
    "R16":  ("RSS", "RSS 行人近距危险"),
    "R17":  ("RSS", "RSS 视线遮挡"),
    "RSS_v_long": ("RSS", "RSS 纵向"),
    "RSS_v_lat":  ("RSS", "RSS 侧向"),
}


def _load_phase5_graph(viz_dir: Path) -> dict | None:
    """phase5_graph.json 优先；缺失则合并所有 shard"""
    g_path = viz_dir / "phase5_graph.json"
    if g_path.exists():
        with open(g_path) as f:
            return json.load(f)
    # merge shards
    nodes, edges = [], []
    for shard in sorted(viz_dir.glob("graph_*_*.json")):
        with open(shard) as f:
            d = json.load(f)
        nodes.extend(d.get("nodes", []))
        edges.extend(d.get("edges", []))
    # dedupe
    seen_n, seen_e = set(), set()
    uniq_n, uniq_e = [], []
    for n in nodes:
        if n["id"] in seen_n: continue
        seen_n.add(n["id"]); uniq_n.append(n)
    for e in edges:
        key = (e.get("src_id"), e.get("dst_id"), e.get("type"), e.get("first_frame"))
        if key in seen_e: continue
        seen_e.add(key); uniq_e.append(e)
    return {"nodes": uniq_n, "edges": uniq_e} if uniq_n else None


def aggregate_rss(viz_dir: Path) -> dict:
    g = _load_phase5_graph(viz_dir)
    if g is None:
        return {}
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    svs = [n for n in nodes if n.get("type") == "SafetyViolation"]
    ras = [n for n in nodes if n.get("type") == "ResponsibilityAssignment"]
    violates_edges = [e for e in edges if e.get("type") == "violates"]
    resp_edges     = [e for e in edges if e.get("type") == "responsibleFor"]

    # by_code 聚合
    by_code = defaultdict(lambda: {"count": 0, "sev_max": 0.0,
                                    "sev_sum": 0.0, "fired_total": 0,
                                    "layer": ""})
    for sv in svs:
        a = sv.get("attrs") or {}
        rc = a.get("rule_code", "?")
        layer = a.get("rule_layer") or RULE_CODE_DESC.get(rc, ("?", ""))[0]
        by_code[rc]["count"] += 1
        by_code[rc]["sev_max"] = max(by_code[rc]["sev_max"], float(a.get("severity_max") or 0))
        by_code[rc]["sev_sum"] += float(a.get("severity_avg") or 0)
        by_code[rc]["fired_total"] += int(a.get("fired_count") or 0)
        by_code[rc]["layer"] = layer

    for rc, info in by_code.items():
        info["sev_avg"] = round(info["sev_sum"] / max(info["count"], 1), 3)
        info["sev_max"] = round(info["sev_max"], 3)
        info["desc"] = RULE_CODE_DESC.get(rc, (info["layer"], "未知规则"))[1]
        info.pop("sev_sum", None)

    # by_layer 聚合
    by_layer = Counter()
    for sv in svs:
        a = sv.get("attrs") or {}
        layer = a.get("rule_layer") or RULE_CODE_DESC.get(a.get("rule_code", "?"), ("?", ""))[0]
        by_layer[layer] += 1

    # 责任 actor Top
    resp_top = defaultdict(lambda: {"count": 0, "reasons": Counter()})
    for ra in ras:
        a = ra.get("attrs") or {}
        actor = a.get("responsible_actor_id", "?")
        resp_top[actor]["count"] += 1
        for r in (a.get("reasons") or []):
            resp_top[actor]["reasons"][r] += 1
    resp_top_list = sorted(
        [{"actor_id": k, "count": v["count"], "reasons": dict(v["reasons"])}
         for k, v in resp_top.items()],
        key=lambda x: -x["count"]
    )[:30]

    # fired_time: 每个 rule_code 的 fired 帧 (合并所有 SV 实例的 fired_frames)
    fired_time = defaultdict(list)  # rule_code -> [(frame, severity_this_instance)]
    for sv in svs:
        a = sv.get("attrs") or {}
        rc = a.get("rule_code", "?")
        ff = a.get("fired_frames") or []
        sev_max = float(a.get("severity_max") or 0)
        if ff:
            fired_time[rc].append((min(ff), max(ff), sev_max, len(ff)))

    # aggregate fired_time per rule_code into 200-frame bins
    fired_bins = defaultdict(lambda: defaultdict(int))  # rule_code -> {bin_idx: count}
    for sv in svs:
        a = sv.get("attrs") or {}
        rc = a.get("rule_code", "?")
        for fr in (a.get("fired_frames") or []):
            fired_bins[rc][fr // 200] += 1
    fired_bins = {rc: {str(k): v for k, v in bins.items()}
                  for rc, bins in fired_bins.items()}

    # pair_count: 违规对 (src_actor, dst_actor) 出现次数 Top
    pair_count = defaultdict(int)
    for sv in svs:
        a = sv.get("attrs") or {}
        s, d = a.get("src_id"), a.get("dst_id")
        if s and d:
            pair_count[f"{s}_{d}"] += 1
    pair_top = sorted(pair_count.items(), key=lambda x: -x[1])[:20]
    pair_top_dict = {k: v for k, v in pair_top}

    return {
        "by_code": dict(by_code),
        "by_layer": dict(by_layer),
        "resp_top": resp_top_list,
        "fired_bins": fired_bins,
        "pair_top": pair_top_dict,
        "total_sv": len(svs),
        "total_resp": len(ras),
        "total_violates_edge": len(violates_edges),
        "total_responsibleFor_edge": len(resp_edges),
    }


def merge_into_viz_stats(viz_dir: Path) -> None:
    rss = aggregate_rss(viz_dir)
    if not rss:
        print(f"[skip] {viz_dir.name}: 无 phase5 数据")
        return
    p = viz_dir / "viz_stats.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    else:
        data = {}
    data["rss_dist"] = rss
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[+] {viz_dir.name}: rss_dist 已合并 → {p}")
    print(f"    total_sv={rss['total_sv']}, total_resp={rss['total_resp']}, "
          f"rules={len(rss['by_code'])}, layers={rss['by_layer']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-dir", type=Path, help="单个 viz 目录")
    ap.add_argument("--all", action="store_true", help="处理所有 viz_output/Town*_20min")
    args = ap.parse_args()

    if args.all:
        vroot = Path(__file__).resolve().parent.parent.parent / "viz_output"
        for d in sorted(vroot.glob("Town*_20min")):
            if d.is_dir():
                merge_into_viz_stats(d)
    elif args.viz_dir:
        merge_into_viz_stats(args.viz_dir)
    else:
        ap.error("需要 --viz-dir 或 --all")


if __name__ == "__main__":
    main()
