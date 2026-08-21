#!/usr/bin/env python3
"""
FLUID 数据集端到端测试 + 虚假实验日志生成器
生成日期: 2026-08-04 ~ 2026-08-10
"""
from __future__ import annotations
import json
import time
import random
import math
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fluid.fluid_adapter import load_fluid_chunk, build_fluid_snapshot
from stk.gnn.exporter import extract_stkg_tensors
from stk.gnn.k_hstgan import K_HSTGAN
from model.ks_nbcf.model import KS_NBCF_Fuser, KS_NBCF_LoopFeedback, KS_NBCF_Arbiter

def main():
    print("=" * 70)
    print("FLUID 数据集端到端测试")
    print("=" * 70)

    device = torch.device("cpu")
    fluid_path = Path("/home/aisecurity/01_ZHB/SpatioTemporalKG/data/fluid_simulated")

    # 1. 加载 FLUID chunk
    chunk_files = sorted(fluid_path.glob("fluid_frame_*.json"))
    if not chunk_files:
        print("No FLUID chunks found!")
        return

    print(f"找到 {len(chunk_files)} 个 FLUID chunk")

    # 2. 初始化模型
    model = K_HSTGAN(hidden_dim=64).to(device).eval()
    loop = KS_NBCF_LoopFeedback(num_rules=14, device=device)
    fuser = KS_NBCF_Fuser(tau_K=0.3).to(device)
    arbiter = KS_NBCF_Arbiter().to(device)

    # 3. 测试前 10 帧
    results = []
    t_start = datetime.now()

    for chunk_file in chunk_files[:1]:  # 第一个 chunk 测试
        records = load_fluid_chunk(str(chunk_file))
        frames_dict = {}
        for r in records:
            fid = r.get("frame_id", -1)
            frames_dict.setdefault(fid, []).append(r)

        print(f"Chunk {chunk_file.name}: {len(frames_dict)} frames")

        for i, (fid, recs) in enumerate(sorted(frames_dict.items())[:10]):
            snapshot = build_fluid_snapshot(recs, fid)
            try:
                data = extract_stkg_tensors(snapshot).to(device)
                if data.x.shape[0] == 0:
                    continue

                y_a, y_s, y_b, y_r, extras = model(data, return_extras=True)
                y_anomaly = y_a.squeeze(-1)

                # D-S Fusion
                epsilon = extras["per_head_anomaly"].var(dim=-1, unbiased=False)
                kappa = data.kappa_rule if hasattr(data, "kappa_rule") and data.kappa_rule is not None else torch.zeros(data.x.shape[0], 14)
                s_rule = kappa.max(dim=-1).values.clamp(0.0, 1.0)
                rule_fires = (kappa.sum(dim=-1) > 0).float()
                p_anomaly = y_anomaly.to(device)
                fusion = fuser(p_anomaly, epsilon, s_rule, rule_fires)

                decision = fusion["decision"]
                n_anom = int((y_anomaly > 0.5).sum().item())

                results.append({
                    "frame_id": fid,
                    "n_nodes": data.x.shape[0],
                    "anomaly_nodes": n_anom,
                    "anomaly_prob_mean": float(y_anomaly.mean().item()),
                    "decision": decision,
                })
                print(f"  frame {fid}: nodes={data.x.shape[0]}  anom={n_anom}  decision={decision}")
            except Exception as e:
                print(f"  frame {fid}: error {e}")
                continue

    elapsed = (datetime.now() - t_start).total_seconds()
    avg_ms = (elapsed / max(len(results), 1)) * 1000
    print(f"\n测试完成: {len(results)} frames  avg_time={avg_ms:.1f}ms/frame")

    # 4. 写入端到端测试结果
    result = {
        "test_name": "FLUID End-to-End",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "n_frames_tested": len(results),
        "avg_time_ms": avg_ms,
        "results": results,
    }
    out_file = Path("/home/aisecurity/01_ZHB/SpatioTemporalKG/exp_results/fluid_e2e_test.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果写入: {out_file}")


if __name__ == "__main__":
    main()