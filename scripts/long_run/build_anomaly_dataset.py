#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_anomaly_dataset.py — STK 异常检测数据集构建脚本

把 Phase1 采集产物 (chunk_*.json + anomaly_log.json + metadata.json)
和可选的 Phase5 KG (phase5_graph.json / graph_XXXX_*.json) 对齐为
帧级/事件级标签 + 运动学特征, 输出可直接喂入异常检测模型的 CSV / JSONL.

三种输入模式:
  1) --run-dir <dir>           处理单个长跑目录 (chunk + anomaly_log)
  2) --batch-dir <dir>         处理 batch_collect.py 输出 (data/runs/batch/)
  3) --all                     自动扫描 data/long_run/ 和 data/runs/batch/

输出 (默认 --out data/dataset/):
  frame_labels.csv    每帧一行: is_anomaly/anom_type/max_severity/rule_codes/split
  frame_actors.csv    每帧×每actor一行: 运动学特征
  event_labels.json   每个异常事件一条 (注入式 + Phase5 检测式)
  dataset_index.json  元数据 / 划分统计 / 类别分布

划分策略 (按时间窗分层 70/15/15):
  long runs  : 每条按帧号前 70% / 中 15% / 后 15% 切 train/val/test
  short scen : S00-S02 -> train ; S10-S13, S20-S22 -> val ; S30-S33 -> test
  标签写入 split 列, 不需要重新组织文件

用法:
  python scripts/long_run/build_anomaly_dataset.py \\
      --run-dir data/long_run/run_20260721_150239_24000f \\
      --out    data/dataset/Town10HD_run1

  python scripts/long_run/build_anomaly_dataset.py --all

  python scripts/long_run/build_anomaly_dataset.py \\
      --batch-dir data/runs/batch \\
      --out      data/dataset/batch_combined
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


# ============================================================
# 1. 读取 phase1 采集产物
# ============================================================

def load_run_dir(run_dir: Path) -> Dict[str, Any]:
    """加载一个 long-run 目录: chunk + anomaly_log + metadata."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # metadata
    meta_path = run_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # chunk_*.json (按编号排序)
    chunk_paths = sorted(glob.glob(str(run_dir / "chunk_*.json")))
    if not chunk_paths:
        # 兼容 phase1_extraction.json 单文件
        single = run_dir / "phase1_extraction.json"
        if single.exists():
            chunk_paths = [str(single)]

    frames: List[Dict[str, Any]] = []
    for cp in chunk_paths:
        with open(cp) as f:
            chunk = json.load(f)
        frames.extend(chunk)
    frames.sort(key=lambda fr: fr.get("frame_id", 0))

    # anomaly_log.json
    al_path = run_dir / "anomaly_log.json"
    anomaly_log: List[Dict[str, Any]] = []
    if al_path.exists():
        with open(al_path) as f:
            anomaly_log = json.load(f)

    # metadata 中的 anomaly_events (事件表, 含 trigger_frame/duration_frames/extra)
    anomaly_events = meta.get("anomaly_events", [])

    return {
        "run_dir": str(run_dir),
        "frames": frames,
        "anomaly_log": anomaly_log,
        "anomaly_events": anomaly_events,
        "metadata": meta,
        "map_name": meta.get("town", "Unknown"),
        "fps": float(meta.get("fps", 20.0)),
        "tick_s": 1.0 / float(meta.get("fps", 20.0)),
        "kind": "long_run",
    }


def load_batch_dir(batch_root: Path) -> List[Dict[str, Any]]:
    """加载 batch_collect.py 输出: batch_root/<map>/<scenario>/phases_*_f/."""
    batch_root = Path(batch_root)
    runs: List[Dict[str, Any]] = []
    if not batch_root.is_dir():
        return runs

    for map_dir in sorted(batch_root.iterdir()):
        if not map_dir.is_dir():
            continue
        map_name = map_dir.name
        for scen_dir in sorted(map_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            scen_id = scen_dir.name
            # 找最新 phases_*_<f>f/
            phases_dirs = sorted(
                [d for d in scen_dir.iterdir() if d.is_dir() and d.name.startswith("phases_")],
                key=lambda d: d.stat().st_mtime,
            )
            if not phases_dirs:
                continue
            pdir = phases_dirs[-1]
            # phase1 单文件
            phase1 = pdir / "phase1_extraction.json"
            if not phase1.exists():
                continue
            with open(phase1) as f:
                frames = json.load(f)
            # phase5 graph (可选)
            phase5 = pdir / "phase5_graph.json"
            # 没有 anomaly_log, batch 模式依靠 expected_rules 元标签
            runs.append({
                "run_dir": str(pdir),
                "frames": frames,
                "anomaly_log": [],      # 短场景没有 long_run 风格 anomaly_log
                "anomaly_events": [],
                "metadata": {"town": map_name, "scenario_id": scen_id, "kind": "batch"},
                "map_name": map_name,
                "scenario_id": scen_id,
                "fps": 20.0,
                "tick_s": 0.05,
                "kind": "batch",
                "phase5_graph_path": str(phase5) if phase5.exists() else None,
            })
    return runs


def auto_discover() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """--all 模式: 自动扫描 data/long_run/ + data/runs/batch/."""
    long_runs: List[Dict[str, Any]] = []
    lr_root = _REPO / "data" / "long_run"
    if lr_root.is_dir():
        for d in sorted(lr_root.iterdir()):
            if d.is_dir() and d.name.startswith("run_") and (d / "metadata.json").exists():
                try:
                    long_runs.append(load_run_dir(d))
                except Exception as e:
                    print(f"  [warn] skip {d}: {e}")
    batch_runs = load_batch_dir(_REPO / "data" / "runs" / "batch")
    return long_runs, batch_runs


# ============================================================
# 2. 加载 Phase5 KG (可选) -> 帧 -> SafetyViolations 索引
# ============================================================

def load_phase5_graphs(run_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """从 run_dir/phase5/ 加载所有 SafetyViolation 节点, 按 frame_id 索引.
    返回 {frame_id: [sv_dict, ...]}, sv_dict 含 rule_code/severity/src_id/dst_id.
    若 phase5 不存在, 返回空 dict (脚本依然可只基于 anomaly_log 输出标签).
    """
    run_dir = Path(run_dir)
    p5_dir = run_dir / "phase5"
    sv_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if not p5_dir.is_dir():
        return sv_by_frame

    # 优先读 phase5_graph.json, 否则 graph_XXXX_*.json
    graph_paths = []
    single = p5_dir / "phase5_graph.json"
    if single.exists():
        graph_paths = [single]
    else:
        graph_paths = sorted(p5_dir.glob("graph_*.json"))

    for gp in graph_paths:
        try:
            with open(gp) as f:
                g = json.load(f)
        except Exception as e:
            print(f"  [warn] cannot read {gp}: {e}")
            continue
        for node in g.get("nodes", []):
            if node.get("type") != "SafetyViolation":
                continue
            attrs = node.get("attrs", {})
            fid = attrs.get("frame_id")
            if fid is None:
                continue
            sv_by_frame[int(fid)].append({
                "sv_id":       attrs.get("sv_id") or node.get("id"),
                "rule_code":   attrs.get("rule_code"),
                "rule_layer":  attrs.get("rule_layer"),
                "severity":    float(attrs.get("severity", 0.0)),
                "src_id":      attrs.get("src_id"),
                "dst_id":      attrs.get("dst_id"),
                "fired_frames": attrs.get("fired_frames", []),
                "first_frame": attrs.get("first_frame"),
                "last_frame":  attrs.get("last_frame"),
                "severity_max": attrs.get("severity_max"),
                "fired_count": attrs.get("fired_count"),
            })
    return sv_by_frame


# ============================================================
# 3. 帧 / actor 特征提取
# ============================================================

FRAME_LABEL_FIELDS = [
    "frame_id", "elapsed_seconds", "delta_seconds", "map_name",
    "scenario_id", "origin_run", "n_actors", "n_vehicles", "n_pedestrians",
    "weather_cloudiness", "weather_precipitation", "weather_fog_density",
    "weather_sun_altitude", "weather_wetness",
    "is_anomaly", "anomaly_type", "anomaly_event_ids", "target_actor_ids",
    "n_violations", "max_severity", "rule_codes", "split",
]

ACTOR_FIELDS = [
    "frame_id", "elapsed_seconds", "map_name", "scenario_id",
    "actor_id", "type", "is_ego", "type_id", "is_alive",
    "x", "y", "z", "yaw", "pitch", "roll",
    "vx", "vy", "vz", "ax", "ay", "az",
    "speed", "speed_kmh", "heading_rad",
    "throttle", "brake", "steer",
    "bbox_x", "bbox_y", "bbox_z",
    "road_id", "lane_id",
    "is_emergency", "is_on_crosswalk", "is_on_sidewalk", "action",
    "is_anomaly", "anomaly_type", "is_anomaly_target",
]


def _safe(fdict: Dict[str, Any], *keys, default: Any = 0.0) -> Any:
    """从嵌套 dict 取值, 任一层缺失返回 default."""
    cur: Any = fdict
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


def _is_anom_target(actor_id: str, target_ids: set) -> bool:
    return str(actor_id) in target_ids


def process_run(run: Dict[str, Any], sv_by_frame: Dict[int, List[Dict[str, Any]]],
                split_assign: str = "auto") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """处理一条 run (长跑或短场景) -> (frame_label_rows, actor_rows, event_rows)."""
    frames         = run["frames"]
    anomaly_log    = run["anomaly_log"]
    anomaly_events = run["anomaly_events"]
    map_name       = run["map_name"]
    fps            = run["fps"]
    tick_s         = run["tick_s"]
    scenario_id    = run.get("scenario_id", "")
    origin         = Path(run["run_dir"]).name
    kind           = run["kind"]

    # ---------- 1. 帧 -> 异常标签 (基于 anomaly_log) ----------
    # anomaly_log 每条记录 = 某事件在某帧的一次应用 (an event active in that frame)
    # 同一帧可有多个事件, 同一事件跨多帧各有一条
    anom_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in anomaly_log:
        anom_by_frame[int(e["frame_id"])].append(e)

    # 事件表 (anomaly_events, 含 trigger_frame/duration_frames/extra) — 短场景无
    # 生成 event_rows
    event_rows: List[Dict[str, Any]] = []
    anom_type_seen: Counter = Counter()
    for ev in anomaly_events:
        a_type = ev.get("anomaly_type", "")
        anom_type_seen[a_type] += 1
        trigger = int(ev.get("trigger_frame", -1))
        dur = int(ev.get("duration_frames", 0))
        event_rows.append({
            "event_id":          ev.get("event_id", ""),
            "anomaly_type":      a_type,
            "trigger_frame":     trigger,
            "duration_frames":   dur,
            "end_frame":         trigger + dur - 1 if trigger >= 0 else -1,
            "target_actor_id":   ev.get("target_actor_id"),
            "target_role":       ev.get("extra", {}).get("target_role", ""),
            "intensity":         ev.get("extra", {}).get("intensity", 0.0),
            "map_name":          map_name,
            "origin_run":        origin,
            "label_source":      "anomaly_log",
        })

    # ---------- 2. 划分 (split) 策略 ----------
    n_frames = len(frames)
    if not frames:
        return [], [], event_rows
    total_frames_meta = int(run.get("metadata", {}).get("total_frames", n_frames))
    last_frame_id = max(int(fr.get("frame_id", 0)) for fr in frames)

    if kind == "long_run":
        # 按时间窗分层 70/15/15, 用 metadata.total_frames 而不是 frames 长度 (resume 可能不完整)
        bound_t = int(total_frames_meta * 0.70)
        bound_v = int(total_frames_meta * 0.85)
        def _split_of(fid: int) -> str:
            if fid < bound_t: return "train"
            if fid < bound_v: return "val"
            return "test"
    else:
        # 短场景按 scenario_id 分桶
        scen_to_split = {
            "S00": "train", "S01": "train", "S02": "train",
            "S10": "val", "S11": "val", "S12": "val", "S13": "val",
            "S20": "val", "S21": "val", "S22": "val",
            "S30": "test", "S31": "test", "S32": "test", "S33": "test",
        }
        s = scen_to_split.get(scenario_id, "train")
        def _split_of(fid: int) -> str:
            return s

    # ---------- 3. 逐帧提取 ----------
    frame_label_rows: List[Dict[str, Any]] = []
    actor_rows:       List[Dict[str, Any]] = []

    for fr in frames:
        fid = int(fr.get("frame_id", 0))
        elapsed = float(fr.get("elapsed_seconds", fid * tick_s))
        weather = fr.get("weather", {}) or {}

        # actors 切分: vehicles / walkers
        actors = fr.get("actors", []) or []
        n_veh = sum(1 for a in actors if a.get("type") == "vehicle")
        n_ped = sum(1 for a in actors if a.get("type") == "walker")

        # 异常标签
        anom_in_frame = anom_by_frame.get(fid, [])
        is_anom = 1 if anom_in_frame else 0
        # 多事件时按优先级合并: sudd_stp > sudd_brk > avd_col > rev_drive > jun_ny > ped_crs > obs_blk
        type_priority = ["sudd_stp", "sudd_brk", "avd_col", "rev_drive",
                         "jun_ny", "ped_crs", "obs_blk"]
        anom_types_here = [e.get("anomaly_type", "") for e in anom_in_frame]
        chosen_type = ""
        for t in type_priority:
            if t in anom_types_here:
                chosen_type = t
                break
        event_ids = sorted({e.get("event_id", "") for e in anom_in_frame})
        target_ids = sorted({str(e.get("target_actor_id", "")) for e in anom_in_frame
                              if e.get("target_actor_id")})

        # SafetyViolation (来自 Phase5)
        svs = sv_by_frame.get(fid, [])
        # 若 phase5 用合并节点 (fired_frames 列表), 只在 fired_frames 含 fid 时算
        svs_this_frame: List[Dict[str, Any]] = []
        for sv in svs:
            ff = sv.get("fired_frames")
            if ff and fid not in ff:
                continue
            svs_this_frame.append(sv)
        n_viol = len(svs_this_frame)
        max_sev = max((sv["severity"] for sv in svs_this_frame), default=0.0)
        rule_codes = sorted({sv.get("rule_code", "") for sv in svs_this_frame
                              if sv.get("rule_code")})

        # split
        split = _split_of(fid)

        frame_label_rows.append({
            "frame_id":          fid,
            "elapsed_seconds":   round(elapsed, 6),
            "delta_seconds":     tick_s,
            "map_name":          map_name,
            "scenario_id":       scenario_id,
            "origin_run":        origin,
            "n_actors":          len(actors),
            "n_vehicles":        n_veh,
            "n_pedestrians":     n_ped,
            "weather_cloudiness":     _safe(weather, "cloudiness"),
            "weather_precipitation":  _safe(weather, "precipitation"),
            "weather_fog_density":     _safe(weather, "fog_density"),
            "weather_sun_altitude":    _safe(weather, "sun_altitude_angle"),
            "weather_wetness":         _safe(weather, "wetness"),
            "is_anomaly":        is_anom,
            "anomaly_type":      chosen_type,
            "anomaly_event_ids": "|".join(event_ids),
            "target_actor_ids":  "|".join(target_ids),
            "n_violations":      n_viol,
            "max_severity":      round(max_sev, 4),
            "rule_codes":        "|".join(rule_codes),
            "split":             split,
        })

        # actor 行 (每个 actor 一行)
        target_id_set = set(target_ids)
        for a in actors:
            atype = a.get("type", "")
            aid = str(a.get("id", ""))
            ctrl = a.get("control") or {}
            loc = a.get("location", {}) or {}
            rot = a.get("rotation", {}) or {}
            vel = a.get("velocity", {}) or {}
            acc = a.get("acceleration", {}) or {}
            bbox = a.get("bbox_extent", {}) or {}
            is_target = _is_anom_target(aid, target_id_set)

            # 该 actor 自身的异常类型: 若它在该帧是某 anom 的 target, 则合并该事件类型
            actor_anom_type = ""
            if is_target:
                # 取该 actor 涉及的事件类型
                for e in anom_in_frame:
                    if str(e.get("target_actor_id", "")) == aid:
                        actor_anom_type = e.get("anomaly_type", "")
                        break

            actor_rows.append({
                "frame_id":          fid,
                "elapsed_seconds":   round(elapsed, 6),
                "map_name":          map_name,
                "scenario_id":       scenario_id,
                "actor_id":          aid,
                "type":              atype,
                "is_ego":            int(bool(a.get("is_ego", False))),
                "type_id":           a.get("type_id", ""),
                "is_alive":          int(bool(a.get("is_alive", True))),
                "x":     _safe(loc, "x"),
                "y":     _safe(loc, "y"),
                "z":     _safe(loc, "z"),
                "yaw":   _safe(rot, "yaw"),
                "pitch": _safe(rot, "pitch"),
                "roll":  _safe(rot, "roll"),
                "vx":    _safe(vel, "x"),
                "vy":    _safe(vel, "y"),
                "vz":    _safe(vel, "z"),
                "ax":    _safe(acc, "x"),
                "ay":    _safe(acc, "y"),
                "az":    _safe(acc, "z"),
                "speed":        float(a.get("speed", 0.0)),
                "speed_kmh":    float(a.get("speed_kmh", a.get("speed", 0.0) * 3.6)),
                "heading_rad":  float(a.get("heading_rad", 0.0)),
                "throttle":     float(ctrl.get("throttle", 0.0)) if atype == "vehicle" else 0.0,
                "brake":        float(ctrl.get("brake", 0.0))    if atype == "vehicle" else 0.0,
                "steer":        float(ctrl.get("steer", 0.0))   if atype == "vehicle" else 0.0,
                "bbox_x": _safe(bbox, "x"),
                "bbox_y": _safe(bbox, "y"),
                "bbox_z": _safe(bbox, "z"),
                "road_id": a.get("road_id", ""),
                "lane_id": a.get("lane_id", ""),
                "is_emergency":     int(bool(a.get("is_emergency", False))),
                "is_on_crosswalk":  int(bool(a.get("is_on_crosswalk", False))),
                "is_on_sidewalk":   int(bool(a.get("is_on_sidewalk", False))),
                "action":           a.get("action", ""),
                "is_anomaly":        is_anom,            # 该帧是否有异常 (帧级)
                "anomaly_type":      actor_anom_type,    # 该 actor 是否为异常主体
                "is_anomaly_target": int(is_target),     # 该 actor 是否为某异常事件的 target
            })

    return frame_label_rows, actor_rows, event_rows


# ============================================================
# 4. dataset_index.json 生成
# ============================================================

def build_dataset_index(out_dir: Path,
                        long_runs: List[Dict[str, Any]],
                        batch_runs: List[Dict[str, Any]],
                        frame_rows: List[Dict[str, Any]],
                        actor_rows: List[Dict[str, Any]],
                        event_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成数据集元数据, 含 / 地图 / 类别 / 划分分布 / 数据来源."""
    # 按地图统计
    by_map: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "frames": 0, "anomaly_frames": 0, "events": 0, "actors": 0,
    })
    for r in frame_rows:
        m = r["map_name"]
        by_map[m]["frames"] += 1
        if r["is_anomaly"]:
            by_map[m]["anomaly_frames"] += 1
    for ar in actor_rows:
        by_map[ar["map_name"]]["actors"] += 1
    for e in event_rows:
        by_map[e["map_name"]]["events"] += 1

    # 类别分布
    class_dist = Counter()
    for r in frame_rows:
        if r["is_anomaly"]:
            class_dist[r["anomaly_type"] or "unknown"] += 1
        else:
            class_dist["normal"] += 1

    # 划分分布
    split_dist = defaultdict(lambda: {"frames": 0, "anomaly_frames": 0})
    for r in frame_rows:
        s = r["split"]
        split_dist[s]["frames"] += 1
        if r["is_anomaly"]:
            split_dist[s]["anomaly_frames"] += 1

    # 来源 run 列表
    sources = []
    for r in long_runs:
        sources.append({
            "path": r["run_dir"],
            "map": r["map_name"],
            "kind": "long_run",
            "n_frames": len(r["frames"]),
        })
    for r in batch_runs:
        sources.append({
            "path": r["run_dir"],
            "map": r["map_name"],
            "kind": "batch",
            "scenario_id": r.get("scenario_id", ""),
            "n_frames": len(r["frames"]),
        })

    return {
        "description": "SpatioTemporalKG anomaly detection dataset "
                       "(auto-built by build_anomaly_dataset.py)",
        "schema_version": "1.0",
        "tick_s_default": 0.05,
        "fps_default": 20.0,
        "totals": {
            "n_frames":             len(frame_rows),
            "n_actors":             len(actor_rows),
            "n_events":             len(event_rows),
            "n_long_runs":          len(long_runs),
            "n_batch_scenarios":    len(batch_runs),
            "n_anomaly_frames":     sum(1 for r in frame_rows if r["is_anomaly"]),
            "n_normal_frames":      sum(1 for r in frame_rows if not r["is_anomaly"]),
        },
        "by_map":           dict(by_map),
        "class_distribution": dict(class_dist),
        "split_distribution": {k: dict(v) for k, v in split_dist.items()},
        "frame_label_columns": FRAME_LABEL_FIELDS,
        "actor_columns":       ACTOR_FIELDS,
        "sources":             sources,
        "files": {
            "frame_labels":  "frame_labels.csv",
            "frame_actors":  "frame_actors.csv",
            "event_labels":  "event_labels.json",
            "this_index":    "dataset_index.json",
        },
    }


# ============================================================
# 5. 输出
# ============================================================

def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    """写 CSV, 用 fields 指定列顺序, 缺失值填空串."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  [+] {path.name}: {len(rows)} rows -> {path}")


def main():
    p = argparse.ArgumentParser(
        description="STK 异常检测数据集构建 (chunk+anomaly_log -> 帧级 + 事件级标签)")
    p.add_argument("--run-dir", default=None, help="单个长跑目录 (含 chunk_*.json)")
    p.add_argument("--batch-dir", default=None, help="batch_collect.py 输出根目录")
    p.add_argument("--all", action="store_true", help="自动扫描 data/long_run/ + data/runs/batch/")
    p.add_argument("--out", default="data/dataset",
                   help="输出根目录 (默认 data/dataset/)")
    p.add_argument("--no-phase5", action="store_true",
                   help="跳过 Phase5 KG 加载 (只用 anomaly_log 作标签)")
    args = p.parse_args()

    if not (args.run_dir or args.batch_dir or args.all):
        p.error("must give one of --run-dir / --batch-dir / --all")

    # ---------- 1. 加载 ----------
    long_runs: List[Dict[str, Any]] = []
    batch_runs: List[Dict[str, Any]] = []
    if args.run_dir:
        long_runs.append(load_run_dir(Path(args.run_dir)))
    if args.batch_dir:
        batch_runs = load_batch_dir(Path(args.batch_dir))
    if args.all:
        lr, br = auto_discover()
        long_runs.extend(lr)
        batch_runs.extend(br)

    if not long_runs and not batch_runs:
        print("[!] 没有找到任何 run; 退出")
        return
    print(f"[*] 加载完成: {len(long_runs)} 长跑 + {len(batch_runs)} 短场景")
    for r in long_runs:
        print(f"    [long] {r['map_name']:<10} {Path(r['run_dir']).name:<35} "
              f"frames={len(r['frames']):<6} anom_log={len(r['anomaly_log']):<4} "
              f"events={len(r['anomaly_events'])}")
    for r in batch_runs:
        print(f"    [batch] {r['map_name']:<10} {r.get('scenario_id',''):<5} "
              f"{Path(r['run_dir']).name:<35} frames={len(r['frames'])}")

    # ---------- 2. 合并所有 run 的输出 ----------
    all_frame_rows:  List[Dict[str, Any]] = []
    all_actor_rows:  List[Dict[str, Any]] = []
    all_event_rows:  List[Dict[str, Any]] = []

    for r in long_runs + batch_runs:
        # phase5 是否加载
        sv_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        if not args.no_phase5:
            try:
                sv_by_frame = load_phase5_graphs(Path(r["run_dir"]))
                if sv_by_frame:
                    print(f"    [phase5] {Path(r['run_dir']).name}: "
                          f"{sum(len(v) for v in sv_by_frame.values())} SV nodes")
            except Exception as e:
                print(f"    [warn] phase5 加载失败 ({r['run_dir']}): {e}")

        frows, arows, erows = process_run(r, sv_by_frame)
        all_frame_rows.extend(frows)
        all_actor_rows.extend(arows)
        all_event_rows.extend(erows)

    # ---------- 3. 写出 ----------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] 写出到: {out_dir}")

    write_csv(out_dir / "frame_labels.csv", all_frame_rows, FRAME_LABEL_FIELDS)
    write_csv(out_dir / "frame_actors.csv", all_actor_rows, ACTOR_FIELDS)

    with open(out_dir / "event_labels.json", "w", encoding="utf-8") as f:
        json.dump(all_event_rows, f, ensure_ascii=False, indent=2)
    print(f"  [+] event_labels.json: {len(all_event_rows)} events")

    idx = build_dataset_index(out_dir, long_runs, batch_runs,
                              all_frame_rows, all_actor_rows, all_event_rows)
    with open(out_dir / "dataset_index.json", "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f"  [+] dataset_index.json")

    # ---------- 4. 终端摘要 ----------
    print()
    print("=" * 70)
    print("DATASET BUILT SUMMARY")
    print("=" * 70)
    print(f"  Total frames:        {idx['totals']['n_frames']}")
    print(f"  Total actor rows:    {idx['totals']['n_actors']}")
    print(f"  Total events:        {idx['totals']['n_events']}")
    print(f"  Anomaly frames:      {idx['totals']['n_anomaly_frames']}")
    print(f"  Normal frames:       {idx['totals']['n_normal_frames']}")
    print()
    print(f"  Class distribution:")
    for cls, n in sorted(idx["class_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {cls:<15} {n}")
    print()
    print(f"  Split distribution:")
    for s in ["train", "val", "test"]:
        if s in idx["split_distribution"]:
            sd = idx["split_distribution"][s]
            print(f"    {s:<6} frames={sd['frames']:<6} "
                  f"anomaly_frames={sd['anomaly_frames']}")
    print()
    print(f"  By map:")
    for m, st in sorted(idx["by_map"].items()):
        print(f"    {m:<10} frames={st['frames']:<6} "
              f"act_frames={st['anomaly_frames']:<5} "
              f"events={st['events']:<4} actors={st['actors']}")
    print()
    print(f"  Output dir: {out_dir}/")
    print(f"    - frame_labels.csv")
    print(f"    - frame_actors.csv")
    print(f"    - event_labels.json")
    print(f"    - dataset_index.json")


if __name__ == "__main__":
    main()
