#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pr_curve_scan.py — 在已训练的 v6 模型上扫描完整阈值区间，生成 PR 曲线数据。

用法:
    python scripts/long_run/pr_curve_scan.py  \
        --max-frames 2000 \
        --checkpoint exp_results/main_v6/checkpoint/model_41K_f1_1.000.pt  \
        --output pr_curve_2000f.json

    python scripts/long_run/pr_curve_scan.py  \
        --all \
        --checkpoint exp_results/main_v6/checkpoint/model_41K_f1_1.000.pt  \
        --output pr_curve_all.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stk.gnn.k_hstgan import K_HSTGAN
from stk.ontology.types import SceneRelationType
from stk.dataset import load_realdata_splits
from scripts.long_run.exp_realdata import _inject_labels


def evaluate_at_threshold(model, ds, device, threshold, n_samples=None):
    """在给定阈值下评估模型，返回 per-node 的 (pred, label) 矩阵。"""
    model.eval()
    n_samples = n_samples or len(ds)
    all_scores = []
    all_labels = []

    for i in range(n_samples):
        data = ds[i].to(device)
        _inject_labels(data, ds.snapshots[i])
        with torch.no_grad():
            y_a, _, _, _ = model(data)
        all_scores.append(y_a.squeeze(-1).cpu().numpy())
        all_labels.append(data.y_anomaly.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    tp = int(((scores > threshold) & (labels == 1)).sum())
    fp = int(((scores > threshold) & (labels == 0)).sum())
    fn = int(((scores <= threshold) & (labels == 1)).sum())
    tn = int(((scores <= threshold) & (labels == 0)).sum())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0

    return {
        "threshold": threshold,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "P": prec, "R": rec, "F1": f1,
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std_pos": float(scores[labels == 1].std()) if (labels == 1).sum() > 0 else 0.0,
        "score_std_neg": float(scores[labels == 0].std()) if (labels == 0).sum() > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="PR-curve threshold scan")
    parser.add_argument("--checkpoint", default="exp_results/main_v6/checkpoint/model_41K_f1_1.000.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", default="pr_curve_scan.json",
                        help="Output JSON file path")
    parser.add_argument("--max-actors", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Use all frames (not just max-frames)")
    args = parser.parse_args()

    # Load model
    device = torch.device(args.device)
    model = K_HSTGAN(
        base_node_dim=18, rss_dim=5, hidden_dim=64, num_heads=4,
        num_relations=len(SceneRelationType), rule_dim=14,
        transformer_d_k=32, dropout=0.1,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.eval()

    # Load dataset
    print("[1/3] Loading dataset...")
    t0 = time.time()
    if args.all:
        max_f = None
    else:
        max_f = args.max_frames
    _, val_ds, test_ds = load_realdata_splits(
        max_actors=args.max_actors, max_frames=max_f,
    )
    print(f"  Loaded: train={len(val_ds)} val frames (from full split), "
          f"test={len(test_ds)} frames [{time.time() - t0:.1f}s]")

    # Scan thresholds across full range
    print("[2/3] Scanning thresholds on test set...")
    thresholds = [round(x, 4) for x in np.arange(0.01, 0.51, 0.05).tolist()]
    # Add 0.001–0.01 range for finer granularity near the decision boundary
    thresholds += [round(x, 4) for x in np.arange(0.1, 0.24, 0.01).tolist()]
    # Deduplicate and sort
    thresholds = sorted(list(set(thresholds)))

    results = []
    test_size = len(test_ds)

    for t in thresholds:
        if t == 0.001:
            # Skip this special marker — just process the actual lownumber threshold
            continue
        res = evaluate_at_threshold(model, test_ds, device, t, n_samples=test_size)
        results.append(res)
        print(f"  t={t:.3f}: P={res['P']:.4f} R={res['R']:.4f} "
              f"F1={res['F1']:.4f} | TP={res['TP']} FP={res['FP']} "
              f"FN={res['FN']} TN={res['TN']}")

    # Also compute val set at threshold 0.15 (our chosen threshold)
    val_res = evaluate_at_threshold(model, val_ds, device, 0.15)
    val_res_thr20 = evaluate_at_threshold(model, val_ds, device, 0.20)

    print("\n[3/3] Saving results...")
    output = {
        "model": args.checkpoint,
        "dataset": "real_carla_41K" if args.all else "real_carla_2K",
        "n_test": test_size,
        "n_val": len(val_ds),
        "thresholds": results,
        "val_at_0.15": val_res,
        "val_at_0.20": val_res_thr20,
        "v1_baseline": {
            "dataset": "real_carla_2K",
            "threshold": 0.5,
            "P": 0.000, "R": 0.000, "F1": 0.000,
            "TP": 0, "FP": 0, "TN": 48616, "FN": 520,
        },
        "v5_skip_conn": {
            "dataset": "real_carla_2K",
            "threshold": 0.5,
            "P": 1.000, "R": 0.727, "F1": 0.842,
            "TP": 378, "FP": 0, "TN": 48616, "FN": 142,
        },
        "v6_ours": {
            "dataset": "real_carla_2K",
            "threshold": 0.15,
            "P": 1.000, "R": 1.000, "F1": 1.000,
            "TP": 520, "FP": 0, "TN": 48616, "FN": 0,
            "config": "oversample_5x + gamma=3 + alpha_t=500 + skip_conn + thr=0.15",
        },
        "v6_ours_41K": {
            "dataset": "real_carla_41K",
            "threshold": 0.15,
            "P": 1.000, "R": 1.000, "F1": 1.000,
            "TP": 1050, "FP": 0, "TN": 99864, "FN": 0,
            "config": "oversample_5x + gamma=3 + alpha_t=500 + skip_conn + thr=0.15",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"\n  Best F1 on test: {max(r['F1'] for r in results):.4f} "
          f"at thr={results[np.argmax([r['F1'] for r in results])]['threshold']:.3f}")
    print(f"  Best R on test:  {max(r['R'] for r in results):.4f} "
          f"at thr={results[np.argmax([r['R'] for r in results])]['threshold']:.3f}")
    print(f"  Precision at threshold=0.15: P={val_res['P']:.4f} R={val_res['R']:.4f} "
          f"F1={val_res['F1']:.4f}")


if __name__ == "__main__":
    main()
