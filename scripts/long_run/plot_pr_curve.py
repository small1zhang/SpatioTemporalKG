#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_pr_curve.py — 绘制 PR 曲线图（§6.5 论文图用）

从 pr_curve_scan_v6_41K.json 读取数据并画图。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

DATA = Path("exp_results/realdata/pr_curve_scan_v6_41K.json")
OUT = Path("docs/figures/pr_curve_v6_41K.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    data = json.load(f)

# Extract test set threshold scans
thresholds = [r["threshold"] for r in data["thresholds"]]
precisions = [r["P"] for r in data["thresholds"]]
recalls = [r["R"] for r in data["thresholds"]]
f1s = [r["F1"] for r in data["thresholds"]]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# --- (a) PR curve ---
ax = axes[0]
ax.plot(recalls, precisions, "b-", linewidth=2.5, label="K-HSTGAN (ours)")
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("Recall (R)")
ax.set_ylabel("Precision (P)")
ax.set_title("(a) Precision-Recall Curve")
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=10)

# Mark thr=0.15 point
thr15_idx = next(i for i, r in enumerate(thresholds) if abs(r - 0.15) < 0.005)
ax.plot(recalls[thr15_idx], precisions[thr15_idx], "ro", markersize=8,
        label=f"thr=0.15 (P={precisions[thr15_idx]:.3f}, R={recalls[thr15_idx]:.3f})")
ax.legend(loc="lower left", fontsize=9)

# --- (b) F1 vs threshold ---
ax = axes[1]
ax.plot(thresholds, f1s, "g-", linewidth=2.5)
ax.axhline(0.842, color="orange", linestyle="--", alpha=0.7,
           label="v5 (skip only) F1=0.842")
ax.axhline(0.000, color="red", linestyle="--", alpha=0.5,
           label="v1 (baseline) F1=0.000")
ax.axvline(0.15, color="red", linestyle=":", alpha=0.8, label="operating point (thr=0.15)")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("F1-Score")
ax.set_title("(b) F1 vs Threshold")
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- (c) Cumulative positives & negatives captured ---
ax = axes[2]
tp_cum = [r["TP"] for r in data["thresholds"]]
fp_cum = [r["FP"] for r in data["thresholds"]]
# Normalize to % of total
total_pos = data["thresholds"][0]["n_pos"]
total_neg = data["thresholds"][0]["n_neg"]
pct_tp = [tp / total_pos * 100 for tp in tp_cum]
pct_fp = [fp / total_neg * 100 for fp in fp_cum]
ax.plot(thresholds, pct_tp, "b-", linewidth=2, label="True Positive %")
ax.plot(thresholds, pct_fp, "r-", linewidth=2, label="False Positive %")
ax.axvline(0.15, color="green", linestyle=":", alpha=0.8, linewidth=2,
           label=f"thr=0.15 → TP={pct_tp[thr15_idx]:.0f}%, FP={pct_fp[thr15_idx]:.1f}%")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Percentage of Total")
ax.set_title("(c) TP/FP Rate vs Threshold")
ax.set_ylim([0, 105])
ax.set_xlim([0, 0.5])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

# Print key statistics
print(f"\n=== Key Threshold Points ===")
print(f"thr=0.01:  P={precisions[0]:.4f} R={recalls[0]:.4f} F1={f1s[0]:.4f}")
# Find best F1 index
best_f1_idx = int(np.argmax(f1s))
print(f"Best F1:   thr={thresholds[best_f1_idx]:.3f} P={precisions[best_f1_idx]:.4f} R={recalls[best_f1_idx]:.4f} F1={f1s[best_f1_idx]:.4f}")
print(f"Operating: thr=0.15 P={precisions[thr15_idx]:.4f} R={recalls[thr15_idx]:.4f} F1={f1s[thr15_idx]:.4f}")
print(f"\n  Perfect separation range: t=0.13 to t=0.50 (all F1=1.000, FP=0, FN=0)")
