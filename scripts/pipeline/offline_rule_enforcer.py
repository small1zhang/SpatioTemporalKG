#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_rule_enforcer.py -- 离线跑 RuleEnforcer × frame_actors.csv，输出真实 DR/FAR

数据流:
  1. data/dataset/frame_actors.csv (1.34M 行 / 24k 帧 actor 属性) → 按 frame_id 分组
  2. data/dataset/frame_labels.csv (41,150 行) → 拿到每帧 GT rule_codes
  3. 对每帧重建 vehicles/pedestrians → 喂给 RuleEnforcer.enforce()
  4. 收集每帧触发规则码集合 (去重 + 归一化)
  5. 与 GT 比对 → 按规则码统计 TP/FP/FN/TN → DR/FAR
  6. 写 data/dataset/rule_detection_stats.json
  7. 直接 patch docs/thesis/chapter6_01.md 表 6-6

性能: 约 0.01 s/frame (ROI 模式 + skip_first=100 帧 + stride=1)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from stk.rules.generator import RuleEnforcer

# ─── 路径 ───────────────────────────────────────────────────
FRAME_LABELS = _REPO / "data" / "dataset" / "frame_labels.csv"
FRAME_ACTORS = _REPO / "data" / "dataset" / "frame_actors.csv"
STATS_OUT = _REPO / "data" / "dataset" / "rule_detection_stats.json"
CHAPTER6 = _REPO / "docs" / "thesis" / "chapter6_01.md"

# ─── frame_actors.csv 仅覆盖此 run ────────────────────
ACTORS_RUN_ID = "run_20260721_150239_24000f"

# ─── walker → pedestrian 映射 ──────────────────────────
WALKER_TO_PEDESTRIANS = {"walker", "pedestrian", "people"}

# ─── 表 6-6 规则码列表 ─────────────────────────────────
TABLE_6_6_RULES = [
    ("R1", "行人优先", "交规"),
    ("R2", "闯红灯", "交规"),
    ("R3", "实线变道", "交规"),
    ("R4", "对向会车违规", "交规"),
    ("R7", "路口未让行", "交规"),
    ("R8", "弱势参与者保护", "交规"),
    ("R11", "恶劣天气限速", "交规"),
    ("R13", "违法停车", "交规"),
    ("R14", "违反交通标志", "交规"),
    ("R17", "不按规定车道", "交规"),
    ("R18", "逆行车道", "交规"),
    ("RSS_R13a", "纵向安全距离", "RSS"),
    ("RSS_R14a", "横向安全距离", "RSS"),
    ("RSS_R15a", "横向危险状态", "RSS"),
]

# ─── enforcer 输出 → GT 规则码归一化 ──────────────────
CODE_NORMALIZE = {
    "R13a": "RSS_R13a",
    "R14a": "RSS_R14a",
    "R15a": "RSS_R15a",
}


def normalize_codes(raw_codes: set[str]) -> set[str]:
    return {CODE_NORMALIZE.get(c, c) for c in raw_codes if c}


def load_gt() -> dict[str, set[str]]:
    """读 frame_labels.csv，返回 {frame_id: set(rule_code)}"""
    print("[1/5] Loading GT ...", flush=True)
    gt: dict[str, set[str]] = {}
    with open(FRAME_LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            codes = r.get("rule_codes", "").strip()
            if codes:
                gt[r["frame_id"]] = {c.strip() for c in codes.split(",") if c.strip()}
    print(f"    {len(gt)} frames with non-empty GT (of all GT rows)", flush=True)
    return gt


def actor_to_entity(r: dict) -> dict:
    """frame_actors.csv 一行 → enforce() entity dict。"""
    f = lambda v, t=float, d=0.0: (lambda x: (t(x) if x not in ("", None) else d))(v)
    x = f(r["x"]); y = f(r["y"]); z = f(r["z"])
    return {
        "entity_id": str(r.get("actor_id", "")),
        "type": r.get("type", ""),
        "is_ego": r.get("is_ego", "0") == "1",
        "is_alive": r.get("is_alive", "1") == "1",
        "location_x": x, "location_y": y, "location_z": z,
        "speed": f(r.get("speed")),
        "speed_kmh": f(r.get("speed_kmph", r.get("speed_kmh"))),
        "heading_rad": f(r.get("heading_rad")),
        "road_id": r.get("road_id", ""),
        "lane_id": r.get("lane_id", ""),
        "is_emergency": r.get("is_emergency", "0") == "1",
        "is_on_crosswalk": r.get("is_on_crosswalk", "0") == "1",
        "is_on_sidewalk": r.get("is_on_sidewalk", "0") == "1",
        "bbox_x": f(r.get("bbox_x")), "bbox_y": f(r.get("bbox_y")),
        "bbox_z": f(r.get("bbox_z")),
        "throttle": f(r.get("throttle")),
        "brake": f(r.get("brake")),
        "steer": f(r.get("steer")),
        "type_id": r.get("type_id", ""),
    }


def stream_frames(skip_first: int = 100, stride: int = 1) -> dict[str, list[dict]]:
    """流式读 frame_actors.csv，按 frame_id 分组。"""
    print(f"[2/5] Streaming actors (skip_first={skip_first}, stride={stride}) ...", flush=True)
    frames: dict[str, list[dict]] = defaultdict(list)
    n = 0
    t0 = time.time()
    with open(FRAME_ACTORS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            if fid < skip_first:
                n += 1
                continue
            if (fid - skip_first) % stride != 0:
                n += 1
                continue
            # walker → pedestrian
            t = r.get("type", "")
            if t in WALKER_TO_PEDESTRIANS:
                r = dict(r)
                r["type"] = "pedestrian"
            frames[r["frame_id"]].append(r)
            n += 1
            if n % 200_000 == 0:
                print(f"    {n:,} actors, {len(frames):,} frames  [{time.time()-t0:.0f}s]", flush=True)
    print(f"    [OK] {n:,} actors → {len(frames):,} frames  ({time.time()-t0:.1f}s)", flush=True)
    return frames


def run_enforcer(frames: dict[str, list[dict]], verbose: bool = True) -> dict[str, set[str]]:
    """对每帧跑 enforce()，返回 {frame_id: 触发 rule_code 集合}。"""
    print(f"[3/5] Running RuleEnforcer on {len(frames)} frames ...", flush=True)
    enforcer = RuleEnforcer()
    result: dict[str, set[str]] = {}
    t0 = time.time()
    total_vio = 0

    for idx, (fid, actors) in enumerate(sorted(frames.items(), key=lambda x: int(x[0]))):
        veh_str = [actor_to_entity(a) for a in actors if a.get("type") == "vehicle"]
        ped_str = [actor_to_entity(a) for a in actors if a.get("type") == "pedestrian"]

        try:
            out = enforcer.enforce(
                frame_id=int(fid),
                vehicles=veh_str,
                pedestrians=ped_str,
                traffic_lights=[],
                scene_rels=[],
                behavior_rels=[],
            )
        except Exception as e:
            print(f"    [warn] frame {fid}: {e}", flush=True)
            result[fid] = set()
            continue

        raw = {v.rule_code for v in out.get("violations", [])}
        codes = normalize_codes(raw)
        result[fid] = codes
        total_vio += len(codes)

        if verbose and (idx + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"    {idx+1:,} frames, {total_vio:,} violations, "
                  f"{dt:.1f}s ({500/dt:.0f} fps)", flush=True)

    dt = time.time() - t0
    print(f"    [OK] {len(result):,} frames, {total_vio:,} violations in {dt:.1f}s", flush=True)
    return result


def compute_dr_far(preds: dict[str, set[str]], gt: dict[str, set[str]]) -> dict:
    """按规则码计算 TP/FP/FN/TN → DR/FAR。"""
    intersection = sorted(set(preds.keys()) & set(gt.keys()), key=lambda x: int(x))
    all_rules = set()
    for s in preds.values(): all_rules |= s
    for s in gt.values(): all_rules |= s
    # 优先级: 表 6-6 的 14+3=17 条 + 其他实际出现的
    table_codes = {r[0] for r in TABLE_6_6_RULES}
    eval_rules = sorted(table_codes | all_rules)

    per_rule = {}
    for rc in eval_rules:
        tp = fp = fn = tn = 0
        for fid in intersection:
            ip = rc in preds.get(fid, set())
            ig = rc in gt.get(fid, set())
            if ip and ig: tp += 1
            elif ip and not ig: fp += 1
            elif not ip and ig: fn += 1
            else: tn += 1
        dr = tp / (tp + fn) if (tp + fn) > 0 else None
        far = fp / (fp + tn) if (fp + tn) > 0 else None
        per_rule[rc] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "dr": dr, "far": far}

    total_tp = sum(p["tp"] for p in per_rule.values())
    total_fp = sum(p["fp"] for p in per_rule.values())
    total_fn = sum(p["fn"] for p in per_rule.values())
    total_tn = sum(p["tn"] for p in per_rule.values())
    overall_dr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    overall_far = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else None
    return {
        "n_frames_gt": len(gt),
        "n_frames_evaluated": len(intersection),
        "actors_run_id": ACTORS_RUN_ID,
        "per_rule": per_rule,
        "overall_dr": overall_dr,
        "overall_far": overall_far,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }


def _pct(x: float | None) -> str:
    if x is None: return "—"
    return f"{100*x:.1f}"


def _pct2(x: float | None) -> str:
    if x is None: return "—"
    return f"{100*x:.2f}"


def build_table_6_6(stats: dict) -> str:
    """生成表 6-6 markdown。"""
    rows_md = []
    traffic_drs, traffic_fars = [], []
    rss_drs, rss_fars = [], []

    for rc, name, layer in TABLE_6_6_RULES:
        p = stats["per_rule"].get(rc, {"dr": None, "far": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        # 显示 DR/FAR (None = 该规则无 GT 数据)
        dr_disp = _pct(p["dr"])
        far_disp = _pct2(p["far"])
        # 若 TP=FP=FN=TN=0 则说明该帧上无 GT 也无预测
        if all(v == 0 for v in (p["tp"], p["fp"], p["fn"], p["tn"])):
            dr_disp = "—"
            far_disp = "—"
        rows_md.append(f"| {rc} | {name} | {layer} | {dr_disp} | {far_disp} |")
        if p["dr"] is not None:
            if layer == "交规": traffic_drs.append(p["dr"]); traffic_fars.append(p["far"])
            else: rss_drs.append(p["dr"]); rss_fars.append(p["far"])

    def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

    traffic_avg_dr = _avg(traffic_drs); traffic_avg_far = _avg(traffic_fars)
    rss_avg_dr = _avg(rss_drs); rss_avg_far = _avg(rss_fars)

    rows_md.append(f"| **平均（交规）** | — | — | "
                   f"**{_pct(traffic_avg_dr)}** | **{_pct2(traffic_avg_far)}** |")
    rows_md.append(f"| **平均（RSS）** | — | — | "
                   f"**{_pct(rss_avg_dr)}** | **{_pct2(rss_avg_far)}** |")
    rows_md.append(f"| **总平均** | — | — | "
                   f"**{_pct(stats['overall_dr'])}** | **{_pct2(stats['overall_far'])}** |")

    return (
        "| 规则码 | 中文名称 | 子层 | DR (%) | FAR (%) |\n"
        "|--------|---------|------|:------:|:-------:|\n"
        + "\n".join(rows_md)
    )


def patch_chapter6(stats: dict, dry_run: bool = False) -> None:
    """把表 6-6 写入 chapter6_01.md，使用 REAL_FILL 标记区域。"""
    new_table = build_table_6_6(stats)
    text = CHAPTER6.read_text(encoding="utf-8")

    start_marker = "<!-- REAL_FILL:rule_detection_dr_far -->"
    end_marker = "<!-- /REAL_FILL:rule_detection_dr_far -->"
    new_block = f"{start_marker}\n\n{new_table}\n\n{end_marker}"

    if start_marker in text and end_marker in text:
        s = text.index(start_marker)
        e = text.index(end_marker) + len(end_marker)
        new_text = text[:s] + new_block + text[e:]
    else:
        anchor = "**表 6-6** 规则检测能力（14+3=17 条规则 × DR/FAR）"
        if anchor not in text:
            print("[!] 找不到表 6-6 标题，跳过 patch", flush=True)
            return
        anchor_end = text.index(anchor) + len(anchor)
        # 找下一个 "**分析**：STKG 规则层" 作为替换终点
        end_idx = text.find("**分析**：STKG 规则层", anchor_end)
        if end_idx < 0:
            end_idx = len(text)
        new_text = (
            text[:anchor_end]
            + "\n\n[三线表]\n\n" + new_table + "\n\n"
            + text[end_idx:]
        )

    if dry_run:
        print("\n[DRY-RUN] 表 6-6 预览:", flush=True)
        print(new_table, flush=True)
        return

    import shutil
    bak = CHAPTER6.with_suffix(".md.bak.rule6")
    shutil.copy2(CHAPTER6, bak)
    CHAPTER6.write_text(new_text, encoding="utf-8")
    print(f"[+] Patched → {CHAPTER6}", flush=True)


def main():
    p = argparse.ArgumentParser(description="离线跑 RuleEnforcer → 表 6-6 DR/FAR")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1, help="采样步长")
    p.add_argument("--skip-first", type=int, default=100,
                   help="跳过前 N 帧 (早期同质 + O(N²) 瓶颈帧)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-patch", action="store_true")
    args = p.parse_args()

    print(f"# offline_rule_enforcer.py — {ACTORS_RUN_ID}", flush=True)
    print(f"# skip_first={args.skip_first}, stride={args.stride}, max_frames={args.max_frames}", flush=True)

    gt = load_gt()
    frames = stream_frames(skip_first=args.skip_first, stride=args.stride)
    if args.max_frames:
        fids = sorted(frames.keys(), key=lambda x: int(x))[:args.max_frames]
        frames = {fid: frames[fid] for fid in fids}

    preds = run_enforcer(frames)
    stats = compute_dr_far(preds, gt)

    print("\n=== DR/FAR 汇总 ===", flush=True)
    print(f"{'rule':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7}  {'DR':>7} {'FAR':>7}", flush=True)
    for rc in [r[0] for r in TABLE_6_6_RULES]:
        p_ = stats["per_rule"].get(rc, {"tp":0,"fp":0,"fn":0,"tn":0,"dr":None,"far":None})
        if all(v==0 for v in (p_["tp"], p_["fp"], p_["fn"], p_["tn"])):
            print(f"  {rc:<12} {'-':>6} {'-':>6} {'-':>6} {'-':>7}  {'无GT':>7} {'无GT':>7}", flush=True)
        else:
            print(f"  {rc:<12} {p_['tp']:>6} {p_['fp']:>6} {p_['fn']:>6} {p_['tn']:>7}  "
                  f"{_pct(p_['dr']):>7} {_pct2(p_['far']):>7}", flush=True)
    print(f"\n  Overall DR = {_pct(stats['overall_dr'])}   "
          f"Overall FAR = {_pct2(stats['overall_far'])}", flush=True)
    print(f"  Evaluated {stats['n_frames_evaluated']} frames  "
          f"(GT non-empty {stats['n_frames_gt']})", flush=True)

    if not args.dry_run:
        STATS_OUT.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[+] Stats → {STATS_OUT}", flush=True)

    if not args.no_patch:
        patch_chapter6(stats, dry_run=args.dry_run)


if __name__ == "__main__":
    main()