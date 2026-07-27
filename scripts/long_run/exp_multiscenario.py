#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_multiscenario.py — K-HSTGAN + KS-NBCF 多场景训练 + 评估

数据来源：stk.scenario.scenario_library 的 14 个预置场景（S00–S33）
每场景 6 帧，共 84 帧 → 按帧切分 train/val/test（8:1:1 近似）

实验输出：
  exp_results/
    rq1/
      training_curve.json    训练损失 + 各阶段 metrics
      confusion_matrix.json  {TP, FP, TN, FN}
      model.pt               最佳模型权重
    rq2/
      ablation.json          消融实验结果（逐模块移除）
    summary.json             全局实验摘要

Run:
    python scripts/long_run/exp_multiscenario.py
    python scripts/long_run/exp_multiscenario.py --epochs 20 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stk.pipeline.orchestrator import PipelineOrchestrator
from stk.gnn.exporter import STKGGraphDataset, extract_stkg_tensors
from stk.gnn.k_hstgan import K_HSTGAN
from stk.gnn.trainer import K_HSTGANTrainer
from stk.fusion import DempsterShaferFuser, EvidenceChainArbiter

# ============================================================
# 1. 数据收集
# ============================================================
ALL_SCENARIOS = [
    "S00", "S01", "S02",
    "S10", "S11", "S12", "S13",
    "S20", "S21", "S22",
    "S30", "S31", "S32", "S33",
]

SCENARIO_TIER = {
    "S00": "A", "S01": "A", "S02": "A",
    "S10": "B", "S11": "B", "S12": "B", "S13": "B",
    "S20": "C", "S21": "C", "S22": "C",
    "S30": "D", "S31": "D", "S32": "D", "S33": "D",
}


def collect_all_snapshots(scenarios: list[str] = ALL_SCENARIOS,
                          max_frames: int = 6) -> list[tuple[str, dict]]:
    """
    运行所有场景的 pipeline，返回 [(scenario_id, snapshot), ...] 列表。
    """
    all_snaps: list[tuple[str, dict]] = []
    print(f"[collect] running {len(scenarios)} scenarios, max_frames={max_frames} ...")
    for sid in scenarios:
        orch = PipelineOrchestrator()
        orch.run_scenario(sid, max_frames=max_frames)
        for fid in sorted(orch.snapshot_store.list_frame_ids()):
            snap = orch.snapshot_store.get(fid)
            all_snaps.append((sid, snap))
    print(f"[collect] {len(all_snaps)} frames collected")
    return all_snaps


def build_dataset(all_snaps: list[tuple[str, dict]]) -> STKGGraphDataset:
    """将原始 snapshot 列表转为 STKGGraphDataset（含标签）。"""
    snapshots = [snap for _, snap in all_snaps]
    ds = STKGGraphDataset(snapshots)
    print(f"[dataset] {len(ds)} frames, "
          f"anomaly positives: {sum(d.y_anomaly.sum().item() for d in ds):.0f}")
    return ds


def split_dataset(ds: STKGGraphDataset,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  seed: int = 42):
    """按帧随机划分 train/val/test。"""
    rng = np.random.RandomState(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # 统计各集合的 anomaly 比例
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        n_pos = sum(1 for i in idx if ds[i].y_anomaly.sum().item() > 0)
        print(f"  {name}: {len(idx)} frames, {n_pos} anomaly-positive ({n_pos/len(idx)*100:.1f}%)")
    return train_idx, val_idx, test_idx


# ============================================================
# 2. 训练
# ============================================================
def train_model(ds: STKGGraphDataset,
                train_idx: list[int],
                val_idx: list[int],
                epochs: int = 20,
                device: torch.device = torch.device("cpu"),
                lr: float = 1e-3,
                patience: int = 5) -> tuple[K_HSTGAN, dict]:
    """
    训练 K-HSTGAN 模型，返回 (best_model, training_history)。
    """
    def collate(batch):
        return batch[0] if isinstance(batch, list) else batch

    train_sub = torch.utils.data.Subset(ds, train_idx)
    val_sub = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_sub, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, collate_fn=collate)

    model = K_HSTGAN(
        base_node_dim=18, rss_dim=5, hidden_dim=64,
        num_heads=4, num_relations=15, rule_dim=14,
        transformer_d_k=32, dropout=0.1,
    ).to(device)

    trainer = K_HSTGANTrainer(
        model=model, lr=lr, max_epochs=epochs, patience=patience,
        grad_clip=5.0, lambda_reg=1e-4,
    )

    history: dict = {"train": [], "val": [], "stages": []}
    best_val_f1 = 0.0
    best_model_state = None

    print(f"\n[train] epochs={epochs}, lr={lr}, patience={patience}, device={device}")
    print(f"{'ep':>4} {'stage':>6} {'loss':>8} {'L0':>7} {'L1':>7} {'g_norm':>8} | "
          f"{'val_P':>6} {'val_R':>6} {'val_F1':>7} {'val_acc':>8} {'val_loss':>9}")
    print("-" * 95)

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader)
        dt = time.time() - t0

        history["train"].append({
            "epoch": epoch,
            "stage": trainer.stage_scheduler.get_stage(epoch),
            **train_metrics,
        })
        history["val"].append({
            "epoch": epoch,
            **val_metrics,
        })
        history["stages"].append({
            "epoch": epoch,
            "stage": trainer.stage_scheduler.get_stage(epoch),
            "lr": train_metrics.get("lr", 0),
            "gamma_3": trainer.ws_scheduler.get_gamma(epoch),
            "w_main": trainer.stage_scheduler.get_w_main(epoch),
        })

        stage = trainer.stage_scheduler.get_stage(epoch)
        print(f"{epoch:>4} {stage:>6} {train_metrics['L_total']:>8.4f} "
              f"{train_metrics['L0']:>7.4f} {train_metrics['L1']:>7.4f} "
              f"{train_metrics['grad_norm']:>8.4f} | "
              f"{val_metrics['P']:>6.3f} {val_metrics['R']:>6.3f} {val_metrics['F1']:>7.3f} "
              f"{val_metrics['accuracy']:>8.3f} {val_metrics['val_loss']:>9.4f} "
              f"[{dt:.2f}s]")

        # 保存最佳模型
        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if trainer.should_stop():
            print(f"  → early stopping at epoch {epoch} (best F1={best_val_f1:.4f})")
            break

    # 恢复最佳权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"[train] best_val_F1={best_val_f1:.4f}")

    return model, history


# ============================================================
# 3. 评估
# ============================================================
@torch.no_grad()
def evaluate_model(model: K_HSTGAN,
                   ds: STKGGraphDataset,
                   indices: list[int],
                   device: torch.device = torch.device("cpu")) -> dict:
    """在指定索引集上评估，返回完整指标。"""
    model.eval()
    all_preds = []
    all_targets = []

    for i in indices:
        data = ds[i].to(device)
        y_a, _, _, _ = model(data)
        preds = (y_a.squeeze(-1) > 0.5).long()
        all_preds.append(preds.cpu())
        all_targets.append(data.y_anomaly.cpu().long())

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)

    tp = int(((preds_t == 1) & (targets_t == 1)).sum())
    fp = int(((preds_t == 1) & (targets_t == 0)).sum())
    fn = int(((preds_t == 0) & (targets_t == 1)).sum())
    tn = int(((preds_t == 0) & (targets_t == 0)).sum())
    n_total = tp + fp + fn + tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (n_total + 1e-8)
    auc_approx = (recall + precision) / 2  # 粗略近似

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "P": round(precision, 4), "R": round(recall, 4),
        "F1": round(f1, 4), "accuracy": round(accuracy, 4),
        "AUC_approx": round(auc_approx, 4),
        "n_total": n_total,
    }


def evaluate_fusion(model: K_HSTGAN,
                    ds: STKGGraphDataset,
                    indices: list[int],
                    device: torch.device = torch.device("cpu")) -> dict:
    """运行 KS-NBCF 完整融合链路，统计 resolve_type 分布。"""
    fuser = DempsterShaferFuser(tau_K=0.3)
    arbiter = EvidenceChainArbiter()
    resolve_counts = {"consistent": 0, "trust_GNN": 0, "trust_rule": 0, "needs_review": 0}
    k_values = []

    for i in indices:
        data = ds[i].to(device)
        y_a, _, _, _, extras = model(data, return_extras=True)
        # 规则触发
        kappa_rule = data.kappa_rule
        s_rule = kappa_rule.max(dim=-1).values.clamp(0, 1)
        rule_fires = (kappa_rule.sum(dim=-1) > 0).float()
        # 补充 violations
        # (简化：不再逐帧构造 node_ids，直接用 kappa_rule 判定)
        epsilon = extras["per_head_anomaly"].var(dim=-1, unbiased=False)
        fr = fuser(y_a, epsilon, s_rule, rule_fires)
        resolve_counts[fr.decision] = resolve_counts.get(fr.decision, 0) + 1
        k_values.append(fr.K)

    return {
        "resolve_distribution": resolve_counts,
        "K_mean": round(float(np.mean(k_values)), 4) if k_values else 0,
        "K_std": round(float(np.std(k_values)), 4) if k_values else 0,
    }


# ============================================================
# 4. 消融实验
# ============================================================
def run_ablation(ds: STKGGraphDataset,
                 train_idx: list[int],
                 val_idx: list[int],
                 test_idx: list[int],
                 epochs: int = 10,
                 device: torch.device = torch.device("cpu")) -> dict:
    """
    RQ2 消融实验：逐个移除关键模块，观察 F1 下降幅度。
    消融方案：
      - full:       完整模型
      - no_rule_inject: 移除规则强度残差注入（kappa_rule 全 0）
      - no_rss:          移除 RSS 残差注入（kappa_rss 全 0）
      - no_delta_gate:   移除差分门控（delta_feat 全 0）
    """
    results = {}

    configs = {
        "full": {},
        "no_rule_inject": {"zero_kappa_rule": True},
        "no_rss": {"zero_kappa_rss": True},
        "no_delta_gate": {"zero_delta_feat": True},
    }

    for name, flags in configs.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}  flags={flags}")
        print(f"{'='*60}")

        # 创建数据集副本（如有 zero 标志，修改 kappa）
        ds_copy = STKGGraphDataset([ds.snapshots[i] for i in range(len(ds))])

        # 应用消融标志
        if flags.get("zero_kappa_rule"):
            for j in range(len(ds_copy)):
                d = ds_copy[j]
                d.kappa_rule.zero_()
        if flags.get("zero_kappa_rss"):
            for j in range(len(ds_copy)):
                d = ds_copy[j]
                d.kappa_rss.zero_()
        if flags.get("zero_delta_feat"):
            for j in range(len(ds_copy)):
                d = ds_copy[j]
                d.delta_feat.zero_()

        # 训练
        model, hist = train_model(
            ds_copy, train_idx, val_idx,
            epochs=epochs, device=device, lr=1e-3, patience=3,
        )

        # 评估
        test_metrics = evaluate_model(model, ds_copy, test_idx, device=device)
        results[name] = test_metrics
        print(f"  → {name}: P={test_metrics['P']:.3f} R={test_metrics['R']:.3f} "
              f"F1={test_metrics['F1']:.3f} acc={test_metrics['accuracy']:.3f}")

    # 计算相对 F1 下降
    if "full" in results:
        base_f1 = results["full"]["F1"]
        for name in results:
            if name != "full":
                delta = results[name]["F1"] - base_f1
                results[name]["F1_delta_vs_full"] = round(delta, 4)
                results[name]["F1_pct_drop"] = round(abs(delta) / max(base_f1, 1e-8) * 100, 1)

    return results


# ============================================================
# 5. 主入口
# ============================================================
def main(args):
    device = torch.device(args.device)
    print("=" * 70)
    print("K-HSTGAN Multi-Scenario Experiment (RQ1 + RQ2)")
    print(f"  epochs={args.epochs}, lr={args.lr}, device={args.device}")
    print(f"  scenarios={len(ALL_SCENARIOS)}, frames_per_scenario={args.max_frames}")
    print("=" * 70)

    # 确保输出目录存在
    out_dir = Path(PROJECT_ROOT) / "exp_results"
    rq1_dir = out_dir / "rq1"
    rq2_dir = out_dir / "rq2"
    rq1_dir.mkdir(parents=True, exist_ok=True)
    rq2_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据收集
    all_snaps = collect_all_snapshots(ALL_SCENARIOS, max_frames=args.max_frames)
    ds = build_dataset(all_snaps)

    # 2. 划分
    train_idx, val_idx, test_idx = split_dataset(ds, seed=args.seed)

    # 3. 训练
    t0 = time.time()
    model, history = train_model(
        ds, train_idx, val_idx,
        epochs=args.epochs, device=device, lr=args.lr,
        patience=args.patience,
    )
    train_time = time.time() - t0

    # 4. 保存训练曲线
    with open(rq1_dir / "training_curve.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"[saved] {rq1_dir / 'training_curve.json'}")

    # 5. 评估（test set）
    test_metrics = evaluate_model(model, ds, test_idx, device=device)
    with open(rq1_dir / "confusion_matrix.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[saved] {rq1_dir / 'confusion_matrix.json'}")

    # 6. KS-NBCF 融合评估
    fusion_metrics = evaluate_fusion(model, ds, test_idx, device=device)
    with open(rq1_dir / "fusion_metrics.json", "w") as f:
        json.dump(fusion_metrics, f, indent=2)
    print(f"[saved] {rq1_dir / 'fusion_metrics.json'}")

    # 7. 保存模型权重
    torch.save(model.state_dict(), rq1_dir / "model.pt")
    print(f"[saved] {rq1_dir / 'model.pt'}")

    # 8. 消融实验（RQ2）
    print("\n" + "=" * 70)
    print("  Ablation Experiments (RQ2)")
    print("=" * 70)
    ablation = run_ablation(
        ds, train_idx, val_idx, test_idx,
        epochs=args.ablation_epochs, device=device,
    )
    with open(rq2_dir / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"[saved] {rq2_dir / 'ablation.json'}")

    # 9. 汇总
    summary = {
        "dataset": {
            "n_total": len(ds),
            "n_scenarios": len(ALL_SCENARIOS),
            "scenarios": ALL_SCENARIOS,
        },
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "training": {
            "epochs_ran": len(history["train"]),
            "best_val_f1": max(h["F1"] for h in history["val"]) if history["val"] else 0,
            "total_time_sec": round(train_time, 2),
        },
        "rq1_test": test_metrics,
        "rq1_fusion": fusion_metrics,
        "rq2_ablation": ablation,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_dir / 'summary.json'}")

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Test P={test_metrics['P']:.3f}  R={test_metrics['R']:.3f}  F1={test_metrics['F1']:.3f}  acc={test_metrics['accuracy']:.3f}")
    print(f"  TP={test_metrics['TP']} FP={test_metrics['FP']} TN={test_metrics['TN']} FN={test_metrics['FN']}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Fusion K: mean={fusion_metrics['K_mean']:.4f} std={fusion_metrics['K_std']:.4f}")
    print()
    for name, m in ablation.items():
        drop = m.get("F1_pct_drop", "-")
        print(f"  {name:<20} F1={m['F1']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}  (drop={drop}%)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-HSTGAN Multi-Scenario Experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ablation-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(main(args))
