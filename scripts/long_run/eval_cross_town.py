#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cross_town.py — K-HSTGAN 跨 Town 泛化评估 (v6 checkpoint)

背景:
  frame_labels.csv 含 5 个 Town: Town10HD (33350 帧) + Town01/02/04/05 (各 1950 帧)。
  所有 anomaly 帧全部来自 Town10HD（其余 4 Town 为正常采集）。
  Town01/02/04/05 的 frame_id 范围都是 [0, 149]（与 Town10HD 部分区间重叠），
  因此默认 load_realdata_splits() 仅保留 Town10HD 以避免 frame_id 串扰，
  其余 4 Town 未参与训练或评估。

评估策略:
  1. OOD 集: Town01/02/04/05 (val+test 合并 ~3,900 帧, 全部正常)
     指标: FPR (FP / (FP + TN)) — 越低越好，0 表示对 OOD 零误报
  2. In-distribution: Town10HD test (4,110 帧, 正常 + 1050 anomaly 节点)
     对照 F1=1.000 baseline
  3. 域内正常但非 anomaly 的子集: Town10HD test 中 is_anomaly=0 帧的 FPR

实现:
  绕过 load_realdata_splits，对每个 Town 单独构造 RealDataDataset 子集
  (按 map_name 过滤 labels_df + actors_df)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stk.gnn.k_hstgan import K_HSTGAN
from stk.ontology.types import SceneRelationType
from stk.dataset.real_data_loader import RealDataDataset
from scripts.long_run.exp_realdata import _inject_labels


def build_town_subset(
    town: str,
    split: str,
    actors_path: str = "data/dataset/frame_actors.csv",
    labels_path: str = "data/dataset/frame_labels.csv",
    events_path: str = "data/dataset/event_labels.json",
    max_actors: int = 30,
    max_frames: int | None = None,
) -> RealDataDataset:
    """为指定 (town, split) 构造独立 RealDataDataset，仅含该 Town 的 actor/label 行。"""
    actors_df = pd.read_csv(actors_path, dtype={"actor_id": str, "scenario_id": str}, low_memory=False)
    actors_df = actors_df[actors_df["map_name"] == town].reset_index(drop=True)

    labels_df = pd.read_csv(labels_path, dtype={"scenario_id": str, "split": str}, low_memory=False)
    labels_df = labels_df[(labels_df["map_name"] == town) & (labels_df["split"] == split)].reset_index(drop=True)

    with open(events_path, "r") as f:
        events = json.load(f)

    if max_frames is not None and max_frames < len(labels_df):
        labels_df = labels_df.sample(n=max_frames, random_state=42).sort_index().reset_index(drop=True)

    return RealDataDataset(
        actors_df=actors_df,
        labels_df=labels_df,
        events=events,
        split=split,
        max_actors=max_actors,
    )


def eval_dataset(model, ds, device, threshold=0.15, town_tag=""):
    """对 dataset 做完整推理，返回聚合 P/R/F1 + per-frame 分布。"""
    model.eval()
    all_scores, all_labels = [], []
    per_frame = []

    for idx in range(len(ds)):
        data = ds[idx].to(device)
        _inject_labels(data, ds.snapshots[idx])
        with torch.no_grad():
            y_a, _, _, _ = model(data)
        scores = y_a.squeeze(-1).cpu().numpy()
        labels = data.y_anomaly.cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

        is_anom_frame = int(labels.sum() > 0)
        per_frame.append({
            "idx": idx,
            "frame_id": ds.snapshots[idx]['extracted']['frame_id'],
            "map_name": ds.snapshots[idx]['extracted']['map_name'],
            "n_nodes": len(scores),
            "n_anom_nodes": int(labels.sum()),
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "pred_pos_count": int((scores > threshold).sum()),
            "is_frame_anomaly_gt": is_anom_frame,
        })

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    tp = int(((all_scores > threshold) & (all_labels == 1)).sum())
    fp = int(((all_scores > threshold) & (all_labels == 0)).sum())
    fn = int(((all_scores <= threshold) & (all_labels == 1)).sum())
    tn = int(((all_scores <= threshold) & (all_labels == 0)).sum())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn + 1e-8)

    # 不要在循环中存全部 per_frame (dataframe 写出转 CSV)
    return {
        "threshold": threshold,
        "P": prec, "R": rec, "F1": f1, "FPR": fpr,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_pos": int((all_labels == 1).sum()),
        "n_neg": int((all_labels == 0).sum()),
        "n_total": len(all_labels),
        "n_frames": len(per_frame),
        "tag": town_tag,
    }, per_frame


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-town OOD evaluation")
    parser.add_argument("--checkpoint", default="exp_results/main_v6/checkpoint/model_41K_f1_1.000.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", default="exp_results/cross_town/eval_full.json")
    parser.add_argument("--max-frames-per-town", type=int, default=None,
        help="limit per-town frames for quick eval (default full)")
    parser.add_argument("--max-actors", type=int, default=30, help="default 30 to match training")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---------- 1. Load v6 model ----------
    print("[1/4] Loading v6 model...")
    model = K_HSTGAN(
        base_node_dim=18, rss_dim=5, hidden_dim=64, num_heads=4,
        num_relations=len(SceneRelationType), rule_dim=14,
        transformer_d_k=32, dropout=0.1,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd)
    model.eval()
    print(f"  Loaded: {args.checkpoint}")

    # ---------- 2. Evaluate each Town ----------
    print("\n[2/4] Evaluating cross-town on val+test splits OOD...")
    towns_to_eval = ["Town01", "Town02", "Town04", "Town05", "Town10HD"]
    results = {"threshold": args.threshold, "towns": {}, "summary": {}}
    by_split_fpr = {"in_distribution": {"Town10HD_test": None}, "ood": {}}

    for town in towns_to_eval:
        for split in ["val", "test"]:
            t0 = time.time()
            print(f"\n  >>> {town} / {split} ...", end=" ", flush=True)
            try:
                ds = build_town_subset(town, split, max_actors=args.max_actors,
                                        max_frames=args.max_frames_per_town)
                print(f"n_frames={len(ds)} ({time.time()-t0:.1f}s load)", end=" ", flush=True)
                if len(ds) == 0:
                    print("-> empty, skip")
                    results["towns"].setdefault(town, {})[split] = {"n_frames": 0, "skipped": True}
                    continue
                t1 = time.time()
                agg, _ = eval_dataset(model, ds, device, args.threshold, town_tag=f"{town}/{split}")
                print(f"eval [{time.time()-t1:.1f}s] P={agg['P']:.4f} R={agg['R']:.4f} "
                      f"F1={agg['F1']:.4f} FPR={agg['FPR']:.4f} "
                      f"TP={agg['TP']} FP={agg['FP']} TN={agg['TN']} FN={agg['FN']}")
                results["towns"].setdefault(town, {})[split] = agg
                # Track FPR for OOD (non-Town10HD)
                if town != "Town10HD":
                    by_split_fpr["ood"][f"{town}_{split}"] = agg["FPR"]
                else:
                    # Town10HD val/test used as in-distribution
                    by_split_fpr.setdefault("in_distribution", {})[f"{town}_{split}"] = {
                        "FPR": agg["FPR"], "R": agg["R"], "F1": agg["F1"]
                    }
            except Exception as e:
                print(f"-> error: {type(e).__name__}: {e}")
                results["towns"].setdefault(town, {})[split] = {"error": str(e)}

    # ---------- 3. Aggregate summaries ----------
    print("\n[3/4] Aggregating summaries...")
    # OOD = town01/02/04/05 val+test frames all should be normal (anomaly=0)
    # Aggregate all 4 OOD towns' val+test into single bag
    ood_total = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "n_frames": 0, "n_pos": 0, "n_neg": 0}
    for town in ["Town01", "Town02", "Town04", "Town05"]:
        for split in ["val", "test"]:
            agg = results["towns"].get(town, {}).get(split, {})
            if "TP" in agg:
                ood_total["TP"] += agg["TP"]
                ood_total["FP"] += agg["FP"]
                ood_total["TN"] += agg["TN"]
                ood_total["FN"] += agg["FN"]
                ood_total["n_frames"] += agg["n_frames"]
                ood_total["n_pos"] += agg["n_pos"]
                ood_total["n_neg"] += agg["n_neg"]

    tp, fp, tn, fn = ood_total["TP"], ood_total["FP"], ood_total["TN"], ood_total["FN"]
    ood_total["P"] = tp / (tp + fp + 1e-8)
    ood_total["R"] = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    ood_total["F1"] = 2 * ood_total["P"] * ood_total["R"] / (ood_total["P"] + ood_total["R"] + 1e-8) if (ood_total["P"] + ood_total["R"]) > 0 else 0.0
    ood_total["FPR"] = fp / (fp + tn + 1e-8)
    results["summary"]["OOD_aggregate"] = ood_total
    print(f"  OOD (Town01/02/04/05 val+test): n_frames={ood_total['n_frames']} "
          f"n_pos={ood_total['n_pos']} n_neg={ood_total['n_neg']}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    P={ood_total['P']:.4f} R={ood_total['R']:.4f} "
          f"F1={ood_total['F1']:.4f} FPR={ood_total['FPR']:.6f}")

    # ID (Town10HD test only)
    agg_test = results["towns"].get("Town10HD", {}).get("test", {})
    if "F1" in agg_test:
        results["summary"]["ID_test"] = agg_test
        print(f"  ID Town10HD test: P={agg_test['P']:.4f} R={agg_test['R']:.4f} "
              f"F1={agg_test['F1']:.4f} FPR={agg_test['FPR']:.6f}")
    agg_val = results["towns"].get("Town10HD", {}).get("val", {})
    if "F1" in agg_val:
        results["summary"]["ID_val"] = agg_val
        print(f"  ID Town10HD val : P={agg_val['P']:.4f} R={agg_val['R']:.4f} "
              f"F1={agg_val['F1']:.4f} FPR={agg_val['FPR']:.6f}")

    # ---------- 4. Save ----------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Saved: {out_path}")


if __name__ == "__main__":
    main()
