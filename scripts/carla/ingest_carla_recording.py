#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_carla_recording.py — 从 CARLA chunk JSON / 模拟数据 / SinD2.0 加载帧并实时检测

用法:
    # 1. 从 CARLA chunk JSON（long_run 数据）回放检测
    python scripts/carla/ingest_carla_recording.py \
        --source chunk --path data/long_run/run_20260721_144806_2400f/chunk_0001.json \
        --frames 10

    # 2. 从 SinD2.0 模拟数据回放检测
    python scripts/carla/ingest_carla_recording.py \
        --source sind2 --path data/sind2_dataset/frames_chunk_001.json \
        --frames 10

    # 3. 从模拟 CARLA 数据集（含 actors）回放
    python scripts/carla/ingest_carla_recording.py \
        --source carla-sim --path data/carla_simulated/chunk_001.json \
        --frames 10

    # 4. 对比运行所有 5 个模型
    python scripts/carla/ingest_carla_recording.py \
        --source chunk --path data/long_run/run_20260721_144806_2400f/chunk_0001.json \
        --models all --frames 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn.functional as F

from model.data_adapter import (
    chunk_frame_to_snapshot,
    sind2_frame_to_snapshot,
    load_simulated_frame,
)
from stk.gnn.exporter import extract_stkg_tensors
from stk.gnn.k_hstgan import K_HSTGAN
from model.ks_nbcf.model import KS_NBCF_Fuser, KS_NBCF_LoopFeedback, KS_NBCF_Arbiter
from model.re_gcn.re_gcn import RE_GCN
from model.gdn.gdn import GDN
from model.general_dyg.general_dyg import GeneralDyG


# ============================================================
# 推理辅助
# ============================================================

def _build_snapshot_from_chunk(raw: dict, source: str) -> dict:
    """根据 source 类型，调用对应的 adapter 转换帧。"""
    if source == "sind2":
        return sind2_frame_to_snapshot(raw)
    elif source == "carla-sim":
        return _load_carla_sim_snapshot(raw)
    else:
        return chunk_frame_to_snapshot(raw)


def _load_carla_sim_snapshot(raw: dict) -> dict:
    """carla_simulated 格式帧转 snapshot。"""
    weather = raw.get("weather", {})
    extracted = {
        "frame_id": raw.get("frame_id", 0),
        "elapsed_seconds": raw.get("elapsed_seconds", 0.0),
        "vehicles": raw.get("vehicles", []),
        "pedestrians": raw.get("pedestrians", []),
        "traffic_lights": raw.get("traffic_lights", []),
        "weather": {
            "fog_density": float(weather.get("fog_density", 0.0)),
            "cloudiness": float(weather.get("cloudiness", 50.0)),
            "precipitation": float(weather.get("precipitation", 0.0)),
            "wetness": float(weather.get("wetness", 0.0)),
            "sun_altitude_angle": float(weather.get("sun_altitude_angle", 45.0)),
            "wind_intensity": float(weather.get("wind_intensity", 0.0)),
        },
    }
    return {"extracted": extracted, "delta": None, "rule_out": {"violations": []}}


def _run_model_inference(
    model,
    data,
    device: torch.device,
    model_name: str,
) -> dict:
    """单模型推理，返回指标摘要。"""
    model.eval()
    with torch.no_grad():
        out = model(data)
    # out: (y_a, y_s, y_b, y_r) or (y_a, y_s, y_b, y_r, extras)
    y_a = out[0]
    n_anom = int((y_a.squeeze(-1) > 0.5).sum().item())
    y_mean = float(y_a.mean().item())
    y_max = float(y_a.max().item())
    return {
        "model": model_name,
        "n_anomaly_nodes": n_anom,
        "y_anomaly_mean": round(y_mean, 4),
        "y_anomaly_max": round(y_max, 4),
    }


# ============================================================
# 主流程
# ============================================================

def run_detection(
    chunk_path: str,
    source: str = "chunk",
    n_frames: int = 5,
    device_name: str = "cpu",
    models_filter: str = "k_hstgan",
) -> list:
    """
    逐帧回放，返回每帧的检测结果列表。
    """
    device = torch.device(device_name)
    print(f"[ingest] source={source} path={chunk_path} n_frames={n_frames} device={device}")

    # 加载 chunk
    with open(chunk_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    if isinstance(frames, dict) and "frames" in frames:
        frames = frames["frames"]

    # 初始化模型
    models = {}
    if models_filter in ("k_hstgan", "all"):
        m = K_HSTGAN(hidden_dim=64).to(device).eval()
        models["K-HSTGAN"] = m
    if models_filter in ("ks_nbcf", "all"):
        m = K_HSTGAN(hidden_dim=64).to(device).eval()
        models["KS-NBCF"] = m
    if models_filter in ("re_gcn", "all"):
        m = RE_GCN(input_dim=18, hidden_dim=64).to(device).eval()
        models["RE-GCN"] = m
    if models_filter in ("gdn", "all"):
        m = GDN(input_dim=18, hidden_dim=64).to(device).eval()
        models["GDN"] = m
    if models_filter in ("general_dyg", "all"):
        m = GeneralDyG(input_dim=18, hidden_dim=64).to(device).eval()
        models["GeneralDyG"] = m

    if not models:
        print("[ingest] ERROR: no models selected")
        return []

    # KS-NBCF 融合模块
    loop = KS_NBCF_LoopFeedback(num_rules=14, device=device)
    fuser = KS_NBCF_Fuser(tau_K=0.3).to(device)
    arbiter = KS_NBCF_Arbiter().to(device)

    all_results = []
    t_total = 0.0

    for i in range(min(n_frames, len(frames))):
        raw = frames[i]
        snapshot = _build_snapshot_from_chunk(raw, source)

        # 构建 STKG
        try:
            data = extract_stkg_tensors(snapshot).to(device)
        except Exception as e:
            print(f"  [frame {i}] STKG build failed: {e}")
            all_results.append({"frame_id": i, "status": "stkg_error", "error": str(e)})
            continue

        n_nodes = data.x.shape[0]
        if n_nodes == 0:
            all_results.append({"frame_id": i, "status": "skip", "reason": "0 nodes"})
            continue

        frame_result = {"frame_id": i, "status": "ok", "n_nodes": n_nodes, "models": {}}
        t0 = time.time()

        # 逐模型推理
        for model_name, model in models.items():
            try:
                mr = _run_model_inference(model, data, device, model_name)
                frame_result["models"][model_name] = mr
            except Exception as e:
                frame_result["models"][model_name] = {"error": str(e)}

        # KS-NBCF D-S 融合（只在使用 k_hstgan 时运行）
        if "K-HSTGAN" in models or "KS-NBCF" in models:
            khstgan = models.get("K-HSTGAN") or models.get("KS-NBCF")
            with torch.no_grad():
                y_a, y_s, y_b, y_r, extras = khstgan(data, return_extras=True)

            # loop feedback
            node_ids = [v.get("entity_id", "") for v in snapshot["extracted"].get("vehicles", [])]
            node_ids += [p.get("entity_id", "") for p in snapshot["extracted"].get("pedestrians", [])]
            if not node_ids:
                node_ids = [f"n{j}" for j in range(n_nodes)]
            rule_out = snapshot.get("rule_out", {"violations": []})

            y_weak = loop.compute_weak_labels(rule_out, node_ids, num_rules=14)
            epsilon = extras["per_head_anomaly"].var(dim=-1, unbiased=False)
            kappa_rule = getattr(data, "kappa_rule", torch.zeros(n_nodes, 14)).to(device)
            s_rule = kappa_rule.max(dim=-1).values.clamp(0.0, 1.0)
            rule_fires = (kappa_rule.sum(dim=-1) > 0).float()

            p_anomaly = y_a.to(device)
            fusion_result = fuser(p_anomaly, epsilon, s_rule, rule_fires)

            # arbiter
            edge_index = extras.get("edge_index")
            edge_type = extras.get("edge_type")
            if edge_index is not None and edge_type is not None:
                edge_index = edge_index.to(device)
                edge_type = edge_type.to(device)
                try:
                    arb_result = arbiter(
                        fusion_result, rule_out,
                        extras.get("rgat_attention", {}) or {},
                        edge_index, edge_type, node_ids,
                        p_anomaly=p_anomaly,
                    )
                except Exception:
                    arb_result = None
            else:
                arb_result = None

            frame_result["ks_nbcf_fusion"] = {
                "decision": fusion_result["decision"],
                "K": round(fusion_result["K"], 4),
                "m_fused_anomaly": round(fusion_result["m_fused_anomaly"], 4),
                "is_consistent": fusion_result["is_consistent"],
            }
            if arb_result:
                frame_result["ks_nbcf_arbiter"] = {
                    "resolve_type": arb_result["resolve_type"],
                    "y_fused": round(arb_result["y_fused"], 4),
                    "overlap": round(arb_result["overlap"], 4),
                    "explanation": arb_result["explanation"][:120],
                }

        elapsed = time.time() - t0
        t_total += elapsed
        frame_result["processing_time_ms"] = round(elapsed * 1000, 1)

        # 打印摘要
        anom_str = " | ".join(
            f"{name}:{mr.get('n_anomaly_nodes', mr.get('error','?'))}"
            for name, mr in frame_result.get("models", {}).items()
        )
        fusion_str = ""
        if "ks_nbcf_fusion" in frame_result:
            f_ = frame_result["ks_nbcf_fusion"]
            fusion_str = f" | D-S:{f_['decision']} K={f_['K']:.3f}"
        print(f"  frame {i}: nodes={n_nodes} {anom_str}{fusion_str}  "
              f"({frame_result['processing_time_ms']:.0f}ms)")

        all_results.append(frame_result)

    avg_ms = (t_total / len(all_results) * 1000) if all_results else 0
    print(f"\n[ingest] done: {len(all_results)} frames  avg={avg_ms:.0f}ms/frame")
    return all_results


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="CARLA/SinD2.0 实时检测回放")
    p.add_argument("--source", choices=["chunk", "sind2", "carla-sim"],
                    default="chunk", help="数据源类型")
    p.add_argument("--path", required=True, help="chunk JSON 文件路径")
    p.add_argument("--frames", type=int, default=5, help="处理帧数")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="推理设备")
    p.add_argument("--models", default="k_hstgan",
                    choices=["k_hstgan", "ks_nbcf", "re_gcn", "gdn", "general_dyg", "all"],
                    help="使用的模型")
    p.add_argument("--output", default=None, help="结果输出 JSON 路径")
    args = p.parse_args()

    results = run_detection(
        chunk_path=args.path,
        source=args.source,
        n_frames=args.frames,
        device_name=args.device,
        models_filter=args.models,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[ingest] saved results to {out_path}")


if __name__ == "__main__":
    main()