#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — 跨分块编排 Phase2→3→4→5 (长期连续采集方案 §五)

用法:
    python scripts/long_run/pipeline.py --run-dir data/long_run/Town10HD_xxx/run_20260720_120000_24000f

设计要点:
  • 复用现有 stk/scenario, stk/behavior, stk/rules, stk/dynamic, stk/storage 模块,
    不修改这些模块内部代码 (仅循环调用). 唯一需要改造的是实例化对象跨分块复用,
    确保状态不因分块边界被重置.
  • IncrementalEngine 和 BehaviorRelationGenerator 实例在分块循环外部创建,
    每个 chunk 继续推进, 不清空内部状态.
  • Phase2 的 waypoint 采集 (static lanes / lane topology) 只需做一次, 跨分块共用.
  • 最终统一写入同一个 Neo4j 图 (同一批 actor.id 对应同一批节点, 持续更新属性).
  • 若无法连接 Neo4j, 输出 phase5_graph.json (KG 全图 JSON), 与现有异常单场景兼容.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


def banner(label):
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)


def load_chunks(run_dir: Path) -> List[Path]:
    """加载 run_dir 下的所有 chunk_*.json, 按帧号排序返回.

    同时支持新格式(light): {'vehicles':[...], 'pedestrians':[...], ...} 的 "已提取" 数据,
    和 run_phases_1_5.py 输出的 phase1_extraction.json 原始格式.

    这里假定 collect.py 产出的 chunk 文件是原始格式:
      [{"frame_id": i, "elapsed_seconds": i*tick_s,
        "actors": [...], "traffic_lights": [...], "weather": {...},
        "waypoints": [...], "events": [...]}, ...]
    """
    chunks = sorted(run_dir.glob("chunk_*.json"))
    if not chunks:
        # 也支持单文件
        fallback = run_dir / "phase1_extraction.json"
        if fallback.exists():
            chunks = [fallback]
    return chunks


def extract_static_lanes(chunks: List[Path]) -> dict:
    """从所有 chunk 的 waypoints 汇总静态车道 + 拓扑, 跨分块共用."""
    from stk.extraction.waypoint_extractor import extract_waypoints, build_lane_topology

    all_wps = []
    for cf in chunks[:3]:  # 前 3 个 chunk 够收集路网
        with open(cf) as f:
            data = json.load(f)
        for frame in data[:5]:
            for wp in frame.get("waypoints", []):
                all_wps.append(wp)
    # dedup by (road_id, lane_id)
    seen = set()
    unique_wps = []
    for wp in all_wps:
        key = (wp.get("road_id"), wp.get("lane_id"))
        if key not in seen:
            seen.add(key)
            unique_wps.append(wp)

    lanes = extract_waypoints(unique_wps)
    topo = build_lane_topology(unique_wps)
    return {"lanes": lanes, "topo": topo}


def process_chunks(
    chunks: List[Path],
    static: dict,
    tick_s: float = 0.05,
    map_name: str = "Town10HD",
    seed: int = 42,
    neo4j_config: Optional[Dict[str, Any]] = None,
    out_dir: Path = None,
    checkpoint_path: Optional[Path] = None,
    shard_frames: Optional[int] = None,
    coalesce_containment: bool = True,
    ego_id: Optional[str] = None,
    importance_threshold: float = -1.0,
    exclude_lanes: bool = False,
    prune_edges: bool = False,
):
    """跨分块执行 Phase2→3→4→5.

    Args:
        chunks: 按帧顺序排列的 chunk 文件路径列表
        static: extract_static_lanes() 的结果 (lanes + topo)
        tick_s: 帧间隔 (秒)
        map_name: 地图名
        seed: 随机种子 (用于 traffic_density 等)
        neo4j_config: {host, port, user, password}, None=只输出 JSON
        out_dir: phase5 输出目录
        checkpoint_path: pipeline checkpoint 路径 (若有则从中恢复, 也定期写入)
        shard_frames: Phase5 分片帧数 (None=单文件模式, 向后兼容)
        coalesce_containment: 是否启用边压缩 (默认 True, --no-coalesce 时禁用)

    核心设计:
      → Phase1 已由 collect.py 完成.
      → Phase2-5 在 chunk 循环外部建立状态, 跨 chunk 不停.
        - BehaviorRelationGenerator (含防抖) 持续
        - IncrementalEngine (delta) 持续
      → 输出: phase5_graph.json (单文件, 默认) 或 graph_XXXX_<start>_<end>.json 分片 (--shard-frames)
      → checkpoint 支持: 每个 chunk 处理完写一次 pipeline_checkpoint.json,
        下次重启时跨 chunk 累积容器 + 引擎状态全部恢复.
    """
    from stk.extraction.actor_extractor import extract_all_actors
    from stk.extraction.trafficlight_extractor import extract_all_traffic_lights
    from stk.extraction.weather_extractor import build_environment_snapshot
    from stk.scenario.snapshot_builder import FrameData, build_snapshot
    from stk.scenario.spatial import (
        compute_in_lane, compute_ahead_of, compute_beside,
        compute_nearby_pedestrian, compute_in_junction,
    )
    from stk.behavior.generator import BehaviorRelationGenerator
    from stk.rules.generator import RuleEnforcer
    from stk.dynamic.incremental_updater import IncrementalEngine
    from stk.storage.serializer import serialize_graph
    from stk.config import EgoCentricConfig
    from stk.filter.importance import ImportanceScorer
    from stk.filter.edge_pruner import EdgePruner
    from stk.filter.background_filter import BackgroundFilter

    # 阶段 3 可选裁剪配置 (CLI 参数驱动, 默认全 None = 不过滤)
    _importance_cfg = None
    _background_cfg = None
    _edge_pruner_cfg = None
    _filter_ego_id = ego_id
    if importance_threshold >= 0.0:
        _ego_cfg = EgoCentricConfig.default()
        _ego_cfg.importance_threshold = importance_threshold
        _importance_cfg = ImportanceScorer(_ego_cfg)
    if exclude_lanes:
        _background_cfg = BackgroundFilter(EgoCentricConfig.default())
    if prune_edges:
        _ego_cfg = EgoCentricConfig.default()
        if _importance_cfg is None:
            _ego_cfg.importance_threshold = 0.30
        _edge_pruner_cfg = EdgePruner(_ego_cfg)

    # ---- Phase 3-5 跨 chunk 状态 ----
    beh_gen = BehaviorRelationGenerator()
    rule_enf = RuleEnforcer()
    engine = IncrementalEngine()

    lanes = static["lanes"]
    topo = static["topo"]

    # 累积容器:
    # - rule_out / maneuvers / interactions / behrels / crossrels: 保内存 (~30MB 内)
    # - all_phase2_frames: 巨大 (24000帧×100KB+), 流式写 jsonl 临时文件
    tmp_phase2     = Path(out_dir / "_phase2_tmp.jsonl") if out_dir else None

    all_maneuvers_raw = []
    all_interactions_raw = []
    all_behavior_rels_raw = []
    all_cross_layer_rels_raw = []
    all_ruleouts_raw = []

    total_frames = 0

    # ---- RQ2 性能计数器 (§10.6 任务#3, 表 6-8/9/10/11 真实化) ----
    # 跨 chunk 累积各 phase 总耗时 (秒) 与帧数, 运行结束输出 perf_metrics.json
    perf_t = {
        "phase2_extract": 0.0,    # Phase 2: actor/TL/weather 提取
        "phase2_spatial": 0.0,    # Phase 2: 空间关系计算 + FrameData
        "phase3_behavior": 0.0,   # Phase 3: BehaviorRelationGenerator
        "phase3_rules": 0.0,      # Phase 3: RuleEnforcer
        "phase4_engine": 0.0,      # Phase 4: IncrementalEngine.process_frame
        "phase5_serialize": 0.0,   # Phase 5: serialize_graph (单次)
    }
    perf_n = {
        "phase2_extract": 0, "phase2_spatial": 0,
        "phase3_behavior": 0, "phase3_rules": 0,
        "phase4_engine": 0,
    }
    # 输出路径: out_dir/perf_metrics.json
    perf_out_path = Path(out_dir) / "perf_metrics.json" if out_dir else None

    # 清空临时文件 (首次写)
    if tmp_phase2 and tmp_phase2.exists():
        tmp_phase2.unlink()

    # ---- Checkpoint 恢复 ----
    start_chunk_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            # 恢复引擎状态
            if "beh_gen" in ckpt:
                beh_gen.load_dict(ckpt["beh_gen"])
            if "engine" in ckpt:
                engine.load_dict(ckpt["engine"])
            # 注: maneuvers/interactions/behrels/crossrels 是内存累积 (pydantic 对象需保留),
            # phase2/ruleouts 走 jsonl 临时文件.
            # resume 场景下 all_*_raw 是空 list (跨进程不可恢复), 因此 Phase5 会被跳过,
            # 用户需从头跑 (--no-resume) 才能输出完整 KG.
            total_frames = ckpt.get("total_frames", 0)
            start_chunk_idx = ckpt.get("chunk_idx", 0) + 1  # 跳过已完成的
            print(f"[+] RESUME from checkpoint: {checkpoint_path.name}")
            print(f"    last chunk processed: {ckpt.get('chunk_idx')} -> start at chunk {start_chunk_idx+1}")
            print(f"    total_frames so far: {total_frames}")
            print(f"    beh_gen stats: {ckpt.get('beh_gen_stats', {})}")
        except Exception as e:
            print(f"[!] checkpoint load failed ({e}), starting from scratch")
            start_chunk_idx = 0
    elif checkpoint_path:
        print(f"[*] no checkpoint at {checkpoint_path}, starting fresh")

    # skip 已处理 chunk
    chunks_to_process = chunks[start_chunk_idx:]
    if start_chunk_idx > 0:
        print(f"[+] skipping {start_chunk_idx} already-processed chunks; "
              f"remaining: {len(chunks_to_process)}")

    for chunk_offset, cf in enumerate(chunks_to_process):
        chunk_idx = start_chunk_idx + chunk_offset  # 真实 chunk 编号
        banner(f"Phase2-5: Processing chunk {chunk_idx+1}/{len(chunks)} ({cf.name})")
        t0 = time.time()

        with open(cf) as f:
            chunk_data = json.load(f)

        # 兼容两种输入格式: 原始 (run_phases_1_5.py / collect.py) 和 已提取字典
        # 自动检测: 如果首 frame 有 "actors" 键, 即为原始; 否则假设为 {vehicles,pedestrians,...}
        first_frame = chunk_data[0] if chunk_data else {}
        needs_extraction = "actors" in first_frame or "waypoints" in first_frame

        chunk_phase2 = []
        chunk_maneuvers_raw = []
        chunk_interactions_raw = []
        chunk_behavior_rels_raw = []
        chunk_cross_layer_rels_raw = []
        chunk_ruleouts_raw = []

        # 临时文件 handle (仅 phase2 帧流式, 其余保内存)
        f_phase2 = open(tmp_phase2, "a", encoding="utf-8") if tmp_phase2 else None

        for raw in chunk_data:
            fid = raw["frame_id"]

            # Phase 2a: actor/TL/weather 提取 (RQ2 perf)
            _t0 = time.perf_counter()
            # Phase 2 转换 (若需要提取)
            if needs_extraction:
                actors = extract_all_actors(raw)
                for av in actors.get("vehicles", []):
                    src = next((a for a in raw["actors"]
                                if str(a["id"]) == str(av.get("entity_id"))), None)
                    if src and "lane_id" in src:
                        av["lane_id"] = src["lane_id"]
                        av["road_id"] = src.get("road_id", -1)
                        av["current_lane_id"] = f"road_{src.get('road_id',0)}_lane_{src['lane_id']}"
                tl = extract_all_traffic_lights(raw.get("traffic_lights", []))
                weather = build_environment_snapshot(
                    raw.get("weather", {}), fid,
                    elapsed_seconds=raw.get("elapsed_seconds", 0.0),
                    delta_seconds=tick_s, map_name=map_name,
                    traffic_density=len(actors.get("vehicles", [])),
                    random_seed=seed,
                )
                # 注: waypoints/lanes 重用 static 而非重新提取
            else:
                # 已是 phase2 格式 (兼容 batch 脚本)
                actors = {"vehicles": raw.get("vehicles", []),
                          "pedestrians": raw.get("pedestrians", [])}
                tl = raw.get("traffic_lights", [])
                weather = raw.get("weather", {})
            perf_t["phase2_extract"] += time.perf_counter() - _t0
            perf_n["phase2_extract"] += 1

            # Phase 2b: 空间关系计算 (RQ2 perf)
            _t0 = time.perf_counter()
            # 空间关系 (每帧使用 static lanes)
            from types import SimpleNamespace
            vehs_raw = actors.get("vehicles", [])
            peds_raw = actors.get("pedestrians", [])
            vehs_adapted = []
            for v in vehs_raw:
                sn = SimpleNamespace(**v); sn.attrs = dict(v)
                if not hasattr(sn, "entity_id") or sn.entity_id is None:
                    sn.entity_id = str(v.get("entity_id", v.get("id", "")))
                vehs_adapted.append(sn)
            peds_adapted = []
            for p in peds_raw:
                sn = SimpleNamespace(**p); sn.attrs = dict(p)
                if not hasattr(sn, "entity_id") or sn.entity_id is None:
                    sn.entity_id = str(p.get("entity_id", p.get("id", "")))
                peds_adapted.append(sn)

            space_rels = []
            try:
                space_rels.extend(compute_in_lane(vehs_adapted, lanes, fid))
                space_rels.extend(compute_ahead_of(vehs_adapted, fid))
                space_rels.extend(compute_beside(vehs_adapted, fid))
                space_rels.extend(compute_nearby_pedestrian(vehs_adapted, peds_adapted, fid))
                space_rels.extend(compute_in_junction(vehs_adapted, fid, lanes))
            except Exception as e:
                print(f"  [!] spatial err f{fid}: {e}")

            spatial_rels_dicts = []
            for r in space_rels:
                spatial_rels_dicts.append({
                    "src_id": str(getattr(r, "src_id", "")),
                    "dst_id": str(getattr(r, "dst_id", "")),
                    "relation_type": getattr(r, "relation_type", ""),
                    "frame_id": getattr(r, "frame_id", fid),
                })
            # T2.3 聚合关系
            scene_id = f"scenario_frame_{fid}"
            env_id = f"env_frame_{fid}"
            tl_ids_seen = set()
            for tl_dict in tl:
                tl_eid = str(tl_dict.get("entity_id", ""))
                if not tl_eid or tl_eid in tl_ids_seen:
                    continue
                tl_ids_seen.add(tl_eid)
                spatial_rels_dicts.append({
                    "src_id": scene_id, "dst_id": tl_eid,
                    "relation_type": "containsTrafficLight", "frame_id": fid,
                })
            road_lane_seen = set()
            for ln in lanes:
                ln_id = str(ln.get("entity_id", ""))
                if not ln_id or ln_id in road_lane_seen:
                    continue
                road_lane_seen.add(ln_id)
                spatial_rels_dicts.append({
                    "src_id": scene_id, "dst_id": ln_id,
                    "relation_type": "containsRoad", "frame_id": fid,
                })
            spatial_rels_dicts.append({
                "src_id": scene_id, "dst_id": env_id,
                "relation_type": "hasEnvironment", "frame_id": fid,
            })
            spatial_rels_dicts.append({
                "src_id": env_id, "dst_id": scene_id,
                "relation_type": "weather_context", "frame_id": fid,
            })
            all_scene_rels = topo + spatial_rels_dicts

            snap_dict = {
                "frame_id": fid,
                "elapsed_seconds": raw.get("elapsed_seconds", fid * tick_s),
                "delta_seconds": tick_s,
                "vehicles": actors.get("vehicles", []),
                "pedestrians": actors.get("pedestrians", []),
                "traffic_lights": tl,
                "lanes": lanes,
                "scene_rels": all_scene_rels,
                "weather": weather,
            }
            chunk_phase2.append(snap_dict)
            perf_t["phase2_spatial"] += time.perf_counter() - _t0
            perf_n["phase2_spatial"] += 1

            # Phase 3a: 行为检测 (RQ2 perf)
            _t0 = time.perf_counter()
            # Phase 3: 行为 + 规则
            veh_str = [{**v, "entity_id": str(v.get("entity_id", v.get("id", "")))}
                       for v in snap_dict["vehicles"]]
            ped_str = [{**p, "entity_id": str(p.get("entity_id", p.get("id", "")))}
                       for p in snap_dict.get("pedestrians", [])]
            tl_str = [{**t, "entity_id": str(t.get("entity_id", t.get("id", "")))}
                      for t in snap_dict.get("traffic_lights", [])]

            beh_out = beh_gen.generate(
                frame_id=fid, vehicles=veh_str, pedestrians=ped_str,
                traffic_lights=tl_str, scene_relations=snap_dict.get("scene_rels", []),
            )
            chunk_maneuvers_raw.extend(beh_out.get("maneuvers", []))
            chunk_interactions_raw.extend(beh_out.get("interactions", []))
            chunk_behavior_rels_raw.extend(beh_out.get("behavior_rels", []))
            chunk_cross_layer_rels_raw.extend(beh_out.get("cross_layer_rels", []))
            perf_t["phase3_behavior"] += time.perf_counter() - _t0
            perf_n["phase3_behavior"] += 1

            # Phase 3b: 规则检测 (RQ2 perf)
            _t0 = time.perf_counter()
            rule_out = rule_enf.enforce(
                frame_id=fid, vehicles=veh_str, pedestrians=ped_str,
                traffic_lights=tl_str, scene_rels=snap_dict.get("scene_rels", []),
            )
            chunk_ruleouts_raw.append({
                "frame_id": fid,
                "violations": rule_out.get("violations", []),
                "responsibilities": rule_out.get("responsibilities", []),
            })
            perf_t["phase3_rules"] += time.perf_counter() - _t0
            perf_n["phase3_rules"] += 1

            # Phase 4: 增量 (跨 chunk 复用 engine) (RQ2 perf)
            _t0 = time.perf_counter()
            engine.process_frame(snap_dict)
            perf_t["phase4_engine"] += time.perf_counter() - _t0
            perf_n["phase4_engine"] += 1

            total_frames += 1

	# 累积 chunk 输出: maneuvers/interactions/behrels/crossrels 保内存 (对象小),
        # phase2 帧 + ruleouts 流式写盘 (避免 OOM)
        all_maneuvers_raw.extend(chunk_maneuvers_raw)
        all_interactions_raw.extend(chunk_interactions_raw)
        all_behavior_rels_raw.extend(chunk_behavior_rels_raw)
        all_cross_layer_rels_raw.extend(chunk_cross_layer_rels_raw)
        # phase2 帧 + ruleouts 流式写盘
        if f_phase2 is not None:
            for snap in chunk_phase2:
                f_phase2.write(json.dumps(snap, ensure_ascii=False, default=str) + "\n")
            f_phase2.close()
	        # 累积本 chunk 的 rule_out
        all_ruleouts_raw.extend(chunk_ruleouts_raw)
        # 记下 beh_rels 数 (在清空前)
        n_scene_rels_this_chunk = sum(len(s['scene_rels']) for s in chunk_phase2)
        chunk_phase2.clear()  # release memory, but keep as list (NOT set to None)
        elapsed = time.time() - t0
        print(f"  [chunk {chunk_idx+1}] {len(chunk_data)} frames, "
              f"beh_rels={n_scene_rels_this_chunk}, "
              f"time={elapsed:.1f}s")

        # ---- 写 checkpoint (每 chunk 一次) ----
        if checkpoint_path is not None:
            try:
                _write_pipeline_checkpoint(
                    checkpoint_path, chunk_idx,
                    beh_gen, engine,
                    total_frames,
                )
                print(f"    [ckpt] saved -> {checkpoint_path.name}")
            except Exception as e:
                print(f"    [!] checkpoint write failed: {e}")

    # ---- Phase 5: 序列化 ----
    # 注: 如果本次是 resume (start_chunk_idx > 0), 则 phase2 帧只覆盖了部分 chunk;
    # 序列化出来的图不完整. 因此 resume 完成后只更新 checkpoint, 等用户跑完最后一轮才输出最终图.
    is_full_run = (start_chunk_idx == 0)
    if not is_full_run:
        print("\n[*] Resume run completed. Phase5 SKIPPED (partial frames).")
        print(f"    To produce final graph, run again with --no-resume to process all chunks from scratch.")
        return {
            "is_full_run": False,
            "total_frames_in_run": total_frames,
            "chunks_processed": len(chunks_to_process),
        }

    banner("Phase 5: Storage & graph output")
    t0 = time.time()

    # 把 jsonl 临时文件读回内存 (frame dict 本身不大，大的是边)
    all_frames: list = []
    if tmp_phase2 and tmp_phase2.exists():
        with open(tmp_phase2, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    all_frames.append(json.loads(line))
        print(f"[+] phase2 frames loaded: {len(all_frames)}")

    print(f"[+] in-memory: maneu={len(all_maneuvers_raw)} inter={len(all_interactions_raw)} "
          f"beh={len(all_behavior_rels_raw)} cross={len(all_cross_layer_rels_raw)} rule_out={len(all_ruleouts_raw)}")
    print(f"[+] output mode: shard_frames={shard_frames}, coalesce={coalesce_containment}")

    shard_use = shard_frames is not None and shard_frames > 0
    all_shard_infos: list = []
    total_nodes = 0
    total_edges_g = 0

    # ---- 分片序列化 (shard mode) ----
    if shard_use:
        for start_idx in range(0, len(all_frames), shard_frames):
            shard_data = all_frames[start_idx:start_idx + shard_frames]
            if not shard_data:
                continue
            f_start = shard_data[0]["frame_id"]
            f_end = shard_data[-1]["frame_id"]
            print(f"\n[shard] frames {f_start}..{f_end} ({len(shard_data)} frames)")

            graph_obj = serialize_graph(
                shard_data, with_relations=True,
                maneuvers=all_maneuvers_raw,
                interactions=all_interactions_raw,
                behavior_rels=all_behavior_rels_raw,
                cross_layer_rels=all_cross_layer_rels_raw,
                rule_out=all_ruleouts_raw,
                coalesce_containment=coalesce_containment,
                importance_cfg=_importance_cfg,
                background_cfg=_background_cfg,
                edge_pruner_cfg=_edge_pruner_cfg,
                ego_id=_filter_ego_id,
            )
            n_nodes = len(graph_obj.get("nodes", []))
            n_edges = len(graph_obj.get("edges", []))
            total_nodes += n_nodes
            total_edges_g += n_edges
            shard_info = {
                "shard_idx": len(all_shard_infos) + 1,
                "frame_start": f_start, "frame_end": f_end,
                "frame_count": len(shard_data),
                "graph_nodes": n_nodes, "graph_edges": n_edges,
            }
            all_shard_infos.append(shard_info)
            print(f"  nodes={n_nodes} edges={n_edges}")

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                graph_path = out_dir / f"graph_{shard_info['shard_idx']:04d}_{f_start}_{f_end}.json"
                with open(graph_path, "w", encoding="utf-8") as f:
                    json.dump(graph_obj, f, ensure_ascii=False, indent=2, default=str)
                sz = graph_path.stat().st_size / (1024 * 1024)
                print(f"  saved: {graph_path.name} ({sz:.1f} MB)")

        summary = {
            "output_mode": "sharded",
            "coalesce_containment": coalesce_containment,
            "shard_frames": shard_frames,
            "total_frames": total_frames,
            "chunks_processed": len(chunks),
            "n_shards": len(all_shard_infos),
            "total_graph_nodes": total_nodes,
            "total_graph_edges": total_edges_g,
            "shards": all_shard_infos,
            "engine_n_deltas": engine.n_deltas,
        }
        if out_dir:
            summary_path = out_dir / "phase5_kg_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[+] summary saved: {summary_path} (n_shards={len(all_shard_infos)})")

    # ---- 原单文件模式 (向后兼容) ----
    else:
        graph_obj = serialize_graph(
            all_frames, with_relations=True,
            maneuvers=all_maneuvers_raw,
            interactions=all_interactions_raw,
            behavior_rels=all_behavior_rels_raw,
            cross_layer_rels=all_cross_layer_rels_raw,
            rule_out=all_ruleouts_raw,
            coalesce_containment=coalesce_containment,
            importance_cfg=_importance_cfg,
            background_cfg=_background_cfg,
            edge_pruner_cfg=_edge_pruner_cfg,
            ego_id=_filter_ego_id,
        )
        g_nodes = len(graph_obj.get("nodes", []))
        g_edges = len(graph_obj.get("edges", []))
        total_nodes = g_nodes
        total_edges_g = g_edges
        type_counts = Counter(n["type"] for n in graph_obj.get("nodes", []))
        edge_type_counts = Counter(e["type"] for e in graph_obj.get("edges", []))

        print(f"[+] serialize_graph: nodes={g_nodes} edges={g_edges}")
        print(f"[+] node types: {dict(type_counts)}")
        print(f"[+] edge types: {dict(edge_type_counts)}")

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            graph_path = out_dir / "phase5_graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(graph_obj, f, ensure_ascii=False, indent=2, default=str)
            sz = graph_path.stat().st_size / (1024 * 1024)
            print(f"[+] graph saved: {graph_path} ({sz:.1f} MB)")

            summary = {
                "output_mode": "single",
                "coalesce_containment": coalesce_containment,
                "total_frames": total_frames, "chunks_processed": len(chunks),
                "graph_nodes": g_nodes, "graph_edges": g_edges,
                "node_types": dict(type_counts), "edge_types": dict(edge_type_counts),
                "engine_n_deltas": engine.n_deltas,
            }
            summary_path = out_dir / "phase5_kg_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[+] summary saved: {summary_path}")

        # 若配置了 Neo4j, 写入 (仅在单文件模式写一次)
        if neo4j_config:
            try:
                _write_to_neo4j(graph_obj, neo4j_config, out_dir)
            except Exception as e:
                print(f"[!] Neo4j write failed: {e}")
        else:
            print("[*] Neo4j not configured. SKIP.")

    # 清理临时文件
    if out_dir:
        try:
            for p in (tmp_phase2,):
                if p and p.exists():
                    p.unlink()
            print(f"[+] cleaned tmp files")
        except Exception:
            pass

    tim = time.time() - t0
    perf_t["phase5_serialize"] = tim
    print(f"[OK] Phase5 done ({tim:.1f}s) total_nodes={total_nodes} total_edges={total_edges_g}")

    # ---- RQ2 性能指标输出 (§10.6 任务#3, 表 6-8/9/10/11) ----
    # 计算每帧均耗 (ms/frame) + 总吞吐 (FPS), 写 perf_metrics.json
    import statistics as _stat
    total_per_run = sum(perf_t.values())
    total_frames_per_run = total_frames if total_frames > 0 else 1
    perf_metrics = {
        "total_frames": total_frames,
        "total_time_s": round(total_per_run, 3),
        "throughput_fps": round(total_frames_per_run / total_per_run, 2) if total_per_run > 0 else 0.0,
        "phase_breakdown_sec": {k: round(v, 3) for k, v in perf_t.items()},
        "phase_per_frame_ms": {
            k: round(perf_t[k] / perf_n[k] * 1000.0, 3) if perf_n[k] > 0 else 0.0
            for k in perf_n
        },
        "phase_call_counts": dict(perf_n),
    }
    # 各帧均耗 sum (≈单帧总耗时, ms)
    perf_metrics["total_per_frame_ms"] = round(
        sum(perf_metrics["phase_per_frame_ms"].values()), 3
    )
    # 表 6-9/10/11 等 RQ2 列直接对应"单帧均耗 + 总吞吐"
    if perf_out_path is not None:
        try:
            perf_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(perf_out_path, "w", encoding="utf-8") as f:
                json.dump(perf_metrics, f, ensure_ascii=False, indent=2)
            print(f"[+] perf_metrics saved: {perf_out_path}")
            print(f"    total_frames={total_frames}, total_time={total_per_run:.1f}s, "
                  f"throughput={perf_metrics['throughput_fps']} fps")
            print(f"    per-frame (ms): "
                  f"P2ext={perf_metrics['phase_per_frame_ms']['phase2_extract']:.2f}, "
                  f"P2spat={perf_metrics['phase_per_frame_ms']['phase2_spatial']:.2f}, "
                  f"P3beh={perf_metrics['phase_per_frame_ms']['phase3_behavior']:.2f}, "
                  f"P3rule={perf_metrics['phase_per_frame_ms']['phase3_rules']:.2f}, "
                  f"P4eng={perf_metrics['phase_per_frame_ms']['phase4_engine']:.2f}, "
                  f"P5ser={perf_t['phase5_serialize']:.1f}s (one-time)")
        except Exception as e:
            print(f"[!] perf_metrics write failed: {e}")


# ------------- Pipeline Checkpoint -------------

def _write_pipeline_checkpoint(
    ckpt_path: Path, chunk_idx: int,
    beh_gen, engine,
    total_frames: int,
) -> None:
    """序列化当前 pipeline 状态到 checkpoint 文件 (每 chunk 写入一次)."""
    ckpt = {
        "chunk_idx": chunk_idx,
        "total_frames": total_frames,
        "beh_gen": beh_gen.to_dict(),
        "engine": engine.to_dict(),
        "beh_gen_stats": beh_gen.stats(),
        "timestamp": __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(ckpt_path)


def _write_to_neo4j(graph_obj, config, out_dir):
    """通过 stk/storage/writer.py 写入 Neo4j.

    如果不想引入 Neo4j 依赖, 保持 JSON 输出兼容.
    """
    from stk.storage.connector import Neo4jConnectionPool
    from stk.storage.writer import GraphWriter

    pool = Neo4jConnectionPool(
        host=config["host"], port=config["port"],
        user=config["user"], password=config["password"],
    )
    writer = GraphWriter(pool)
    writer.write_graph(graph_obj)
    writer.close()
    pool.close()


def main():
    p = argparse.ArgumentParser(description="Cross-chunk Phase2→5 orchestrator")
    p.add_argument("--run-dir", required=True, help="collect.py 输出目录 (含 chunks)")
    p.add_argument("--tick-s", type=float, default=0.05, help="帧间隔 (秒)")
    p.add_argument("--map-name", default="Town10HD", help="地图名")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neo4j-host", default=None, help="Neo4j 地址 (不指定则仅输出 JSON)")
    p.add_argument("--neo4j-port", type=int, default=7687)
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default="password")
    p.add_argument("--out", default=None, help="输出目录 (默认 = run_dir/phase5)")
    p.add_argument("--no-resume", action="store_true",
                   help="忽略已有 pipeline_checkpoint.json, 从头处理所有 chunk")
    # Phase5 输出优化 (long-run ≥20min 数据用)
    p.add_argument("--shard-frames", type=int, default=None,
                   help="Phase5 按时间窗口分片输出 (每 N 帧一个 graph_XXXX_<start>_<end>.json). "
                        "不指定则保持原单文件 phase5_graph.json 行为 (向后兼容)")
    p.add_argument("--no-coalesce", action="store_true",
                   help="禁用边合并 (默认开启 coalesce_containment 以压缩 containsXXX 等冗余边). "
                        "禁用后退回逐帧 scenario_frame_F + 逐帧包含边的老行为")
    # 阶段 3: 节点/边裁剪 (默认关闭, 显式 opt-in)
    p.add_argument("--importance-threshold", type=float, default=-1.0,
                   help="重要性打分阈值 (<0 表示关闭). 启用后低分实体在序列化时被剔除.")
    p.add_argument("--ego-id", default=None,
                   help="显式自车 entity_id, 用于重要性打分与 ROI (留空自动识别).")
    p.add_argument("--exclude-lanes", action="store_true",
                   help="启用静态背景外移: lane 节点/边不进 KG.")
    p.add_argument("--prune-edges", action="store_true",
                   help="启用边稀疏化: ROI 外 spatia/behavior 边被剔除.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[FATAL] run-dir not found: {run_dir}")
        sys.exit(1)

    chunks = load_chunks(run_dir)
    if not chunks:
        print(f"[FATAL] no chunk_*.json found in {run_dir}")
        sys.exit(1)
    print(f"[+] found {len(chunks)} chunk files")

    out_dir = Path(args.out) if args.out else (run_dir / "phase5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline checkpoint path
    checkpoint_path = None if args.no_resume else (out_dir / "pipeline_checkpoint.json")

    # Static lanes
    banner("Static lane extraction (跨分块共用)")
    static = extract_static_lanes(chunks)
    print(f"[+] static: lanes={len(static['lanes'])}, topo_edges={len(static['topo'])}")

    neo4j_config = None
    if args.neo4j_host:
        neo4j_config = {
            "host": args.neo4j_host, "port": args.neo4j_port,
            "user": args.neo4j_user, "password": args.neo4j_password,
        }

    process_chunks(
        chunks=chunks, static=static,
        tick_s=args.tick_s, map_name=args.map_name,
        seed=args.seed, neo4j_config=neo4j_config,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        shard_frames=args.shard_frames,
        coalesce_containment=not args.no_coalesce,
        ego_id=args.ego_id,
        importance_threshold=args.importance_threshold,
        exclude_lanes=args.exclude_lanes,
        prune_edges=args.prune_edges,
    )

    print("\n[OK] Pipeline done.")
    print(f"     chunks processed: {len(chunks)}")
    print(f"     output: {out_dir}")


if __name__ == "__main__":
    main()
