#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_realdata.py — 真实 CARLA 数据 K-HSTGAN 训练实验（§6.3）

基于 `stk.dataset.RealDataDataset` 加载真实数据（41K 帧），
训练 K-HSTGAN + KS-NBCF 融合框架，收集论文第 6 章实验数据。

数据来源：data/dataset/frame_actors.csv + frame_labels.csv + event_labels.json
预切分：train=23,540 / val=9,930 / test=7,680（keep-out temporal split）

Usage:
    # 最小验证（50 帧，3 epochs）
    python scripts/long_run/exp_realdata.py --max-frames 50 --epochs 3 --device cpu

    # 小规模训练
    python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 10 --device cpu

    # 全量训练
    python scripts/long_run/exp_realdata.py --all --epochs 30 --device cpu
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stk.dataset import RealDataDataset, load_realdata_splits
from stk.gnn.k_hstgan import K_HSTGAN
from stk.gnn.trainer import K_HSTGANTrainer
from stk.fusion import DempsterShaferFuser, EvidenceChainArbiter

# ============================================================
# 配置
# ============================================================
DEFAULT_OUT_DIR = PROJECT_ROOT / "exp_results" / "realdata"

# 六个异常类型的情境编码（用于 y_scene / y_behavior 伪标签）
ANOMALY_TYPE_TO_SCENE = {
    "sudd_brk":  0,   # 高速公路急刹 → Highway
    "avd_col":   0,   # 避撞 → Highway
    "rev_drive": 1,   # 逆向 → Urban
    "obs_blk":   1,   # 障碍 → Urban
    "jun_ny":    2,   # 路口不避让 → Intersection
    "sudd_stp":  0,   # 突然停车 → Highway
}
ANOMALY_TYPE_TO_BEHAVIOR = {
    "sudd_brk":  1,   # SuddenBrake
    "avd_col":   2,   # CollisionAvoidance
    "rev_drive": 3,   # Reversing
    "obs_blk":   4,   # Blocking
    "jan_ny":    5,   # JunctionNonYield
    "sudd_stp":  1,   # SuddenStop (same as SuddenBrake)
}

# ============================================================
# 辅助函数：为 Data 对象补充伪标签
# ============================================================
def _inject_labels(data: torch.Tensor, snapshot: dict) -> None:
    """
    从 snapshot 的 extracted 字段补充 y_scene / y_behavior / y_rule 伪标签。
    基于 violation 中的 anomaly_type 与 rule_code 映射。
    """
    n_nodes = data.x.size(0)
    n_rules = 14

    # 默认全零
    y_scene = torch.zeros(n_nodes, dtype=torch.long)
    y_behavior = torch.zeros(n_nodes, dtype=torch.long)
    y_rule = torch.zeros(n_nodes, n_rules, dtype=torch.float)

    violations = snapshot.get("rule_out", {}).get("violations", [])
    node_ids = snapshot["extracted"].get("_node_ids", [])
    id2row = {nid: i for i, nid in enumerate(node_ids)}

    for vi in violations:
        # 从 attrs 字典取属性（兼容 pydantic SafetyViolation 和 CSVViolation）
        attrs = getattr(vi, "attrs", {}) or {}
        rule_code = attrs.get("rule_code", "") or getattr(vi, "rule_code", "")
        src = attrs.get("src_id") or getattr(vi, "src_id", "")
        raw_type = attrs.get("anomaly_type", "") or getattr(vi, "anomaly_type", "")
        sev = float(attrs.get("severity", 0.5) or getattr(vi, "severity", 0.5))

        if src not in id2row:
            continue

        row = id2row[src]

        # y_scene（简化）
        if rule_code in ("R1", "R4", "R7", "R8"):
            y_scene[row] = 1  # Intersection
        elif rule_code in ("R2",):
            y_scene[row] = 0  # Highway
        else:
            y_scene[row] = 2  # Residential

        # y_behavior
        if raw_type in ANOMALY_TYPE_TO_BEHAVIOR:
            bp = ANOMALY_TYPE_TO_BEHAVIOR[raw_type]
            if bp < 7:
                y_behavior[row] = bp
        elif rule_code == "R13":
            y_behavior[row] = 1  # SuddenBrake

        # y_rule（14 维）
        rule_name_map = {
            "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R7": 6,
            "R8": 7, "R9": 8, "R10": 9, "R11": 10, "R13": 11,
            "R16": 12, "R17": 13, "R18": 14,
        }
        if rule_code in rule_name_map:
            ridx = rule_name_map[rule_code]
            if ridx < n_rules:
                y_rule[row, ridx] = sev

    data.y_scene = y_scene
    data.y_behavior = y_behavior
    data.y_rule = y_rule


# ============================================================
# 主实验流程
# ============================================================
def main(args):
    device = torch.device(args.device)
    print("=" * 72)
    print("K-HSTGAN Real CARLA Data Experiment")
    print(f"  device={args.device}, max_actors={args.max_actors}, epochs={args.epochs}")
    print("=" * 72)

    # ---- 1. 加载数据 ----
    print("\n[1/5] Loading real CARLA data...")
    t0 = time.time()

    if args.all:
        print("  Mode: ALL frames (full train/val/test)")
        train_ds, val_ds, test_ds = load_realdata_splits(
            actors_path=args.actors_path,
            labels_path=args.labels_path,
            events_path=args.events_path,
            max_actors=args.max_actors,
        )
    else:
        max_f = args.max_frames
        print(f"  Mode: max_frames={max_f}")
        train_ds, val_ds, test_ds = load_realdata_splits(
            actors_path=args.actors_path,
            labels_path=args.labels_path,
            events_path=args.events_path,
            max_actors=args.max_actors,
            max_frames=max_f,
        )

    # 注入标签
    for ds in (train_ds, val_ds, test_ds):
        for i in range(len(ds)):
            snap = ds.snapshots[i]
            _inject_labels(ds[i], snap)

    print(f"  train: {len(train_ds)} frames, val: {len(val_ds)}, test: {len(test_ds)}")
    print(f"  Data loading: {time.time() - t0:.1f}s")

    # ---- 2. 数据集统计 ----
    print("\n[2/5] Dataset stats (sampled first 50 frames):")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        n_anom = 0
        n_nodes = 0
        n_edges = 0
        n_samples = min(50, len(ds))
        for i in range(n_samples):
            d = ds[i]
            n_nodes += d.x.size(0)
            n_edges += d.edge_index.size(1)
            n_anom += int(d.y_anomaly.sum().item())
        print(f"  {name}: {len(ds)} frames, "
              f"avg_nodes={n_nodes / n_samples:.1f}, "
              f"avg_edges={n_edges / n_samples:.1f}, "
              f"anom_nodes(sampled)={n_anom}")

    # ---- 3. 初始化模型 ----
    print("\n[3/5] Initializing K-HSTGAN model...")
    from stk.ontology.types import SceneRelationType
    model = K_HSTGAN(
        base_node_dim=18,
        rss_dim=5,
        hidden_dim=64,
        num_heads=4,
        num_relations=len(SceneRelationType),
        rule_dim=14,
        transformer_d_k=32,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # ---- 4. 训练 ----
    print("\n[4/5] Training...")
    t1 = time.time()

    trainer = K_HSTGANTrainer(
        model=model,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        grad_clip=5.0,
        lambda_reg=1e-4,
    )

    history = {"train": [], "val": [], "stages": []}
    best_val_f1 = 0.0
    best_state = None

    sep = "=" * 95
    print(sep)
    print(f"{'ep':>4} {'stage':>6} {'loss':>8} {'L0':>7} {'L1':>7} {'g_norm':>8} | "
          f"{'val_P':>6} {'val_R':>6} {'val_F1':>7} {'val_acc':>8} {'val_loss':>9}")
    print("-" * 95)

    for epoch in range(args.epochs):
        t_ep = time.time()

        # Train (on train_ds, sampled)
        n_train_use = min(len(train_ds), args.train_batch)
        train_metrics = _train_epoch_real(
            model, trainer, train_ds, n_train_use, device, epoch,
        )

        # Validate
        val_metrics = _eval_epoch_real(
            model, val_ds, min(len(val_ds), args.val_batch), device,
        )

        dt = time.time() - t_ep

        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})
        history["stages"].append({
            "epoch": epoch,
            "stage": trainer.stage_scheduler.get_stage(epoch),
            "lr": train_metrics.get("lr", 0),
        })

        print(f"{epoch:>4} {trainer.stage_scheduler.get_stage(epoch):>6} "
              f"{train_metrics['L_total']:>8.4f} {train_metrics['L0']:>7.4f} "
              f"{train_metrics['L1']:>7.4f} {train_metrics['grad_norm']:>8.4f} | "
              f"{val_metrics['P']:>6.3f} {val_metrics['R']:>6.3f} {val_metrics['F1']:>7.3f} "
              f"{val_metrics['accuracy']:>8.3f} {val_metrics['val_loss']:>9.4f} "
              f"[{dt:.1f}s]")

        # Save best
        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if trainer.should_stop():
            print(f"  -> early stopping at epoch {epoch} (best F1={best_val_f1:.4f})")
            break

    train_time = time.time() - t1

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 5. 评估 ----
    print("\n[5/5] Evaluation on test set...")
    test_metrics = _eval_epoch_real(
        model, test_ds, len(test_ds), device,
    )

    result = {
        "test": test_metrics,
        "best_val_f1": best_val_f1,
        "epochs_ran": len(history["train"]),
        "train_time_sec": train_time,
        "config": {
            "max_actors": args.max_actors,
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "test_frames": len(test_ds),
        },
    }

    # 保存
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Test:  P={test_metrics['P']:.3f}  R={test_metrics['R']:.3f}  "
          f"F1={test_metrics['F1']:.3f}  acc={test_metrics['accuracy']:.3f}")
    print(f"  TP={test_metrics['TP']} FP={test_metrics['FP']} "
          f"TN={test_metrics['TN']} FN={test_metrics['FN']}")
    print(f"  Best val F1: {best_val_f1:.3f}")
    print(f"  Train time:  {train_time:.1f}s")
    print(f"  Saved: {out_dir / 'results.json'}")
    print("=" * 72)

    return result


def _train_epoch_real(
    model: K_HSTGAN,
    trainer: K_HSTGANTrainer,
    ds: RealDataDataset,
    n_samples: int,
    device: torch.device,
    epoch: int,
) -> dict:
    """在真实数据的 train_ds 上训练一个 epoch（采样 n_samples 帧）"""
    model.train()
    trainer.optimizer.zero_grad()

    total_loss = 0.0
    total_steps = 0
    total_grad_norm = 0.0
    total_L0 = 0.0
    total_L1 = 0.0

    # 随机采样
    indices = torch.randperm(min(n_samples, len(ds))).tolist()

    for idx in indices:
        data = ds[idx].to(device)
        _inject_labels(data, ds.snapshots[idx])

        try:
            y_a, y_s, y_b, y_r = model(data)
            loss, metrics = trainer._compute_loss(
                y_a, y_s, y_b, y_r,
                data.y_anomaly, data.y_scene, data.y_behavior, data.y_rule,
                epoch=epoch,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.ema.update()

            total_loss += loss.item()
            total_L0 += metrics["L0"]
            total_L1 += metrics.get("L1", 0.0)
            total_grad_norm += metrics.get("grad_norm", 0.0)
            total_steps += 1
        except Exception as e:
            print(f"  [WARN] Skip frame idx={idx}: {e}")
            continue

    if total_steps == 0:
        return {"L_total": 0.0, "L0": 0.0, "L1": 0.0, "grad_norm": 0.0, "lr": 0.0}

    return {
        "L_total": total_loss / total_steps,
        "L0": total_L0 / total_steps,
        "L1": total_L1 / total_steps,
        "grad_norm": total_grad_norm / total_steps,
        "lr": trainer.optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def _eval_epoch_real(
    model: K_HSTGAN,
    ds: RealDataDataset,
    n_samples: int,
    device: torch.device,
) -> dict:
    """在真实数据上评估。"""
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_steps = 0

    indices = list(range(min(n_samples, len(ds))))

    for idx in indices:
        data = ds[idx].to(device)
        _inject_labels(data, ds.snapshots[idx])

        try:
            y_a, y_s, y_b, y_r = model(data)
            preds = (y_a.squeeze(-1) > 0.5).long()
            all_preds.append(preds.cpu())
            all_targets.append(data.y_anomaly.long().cpu())

            loss = nn.functional.binary_cross_entropy(
                y_a.squeeze(-1), data.y_anomaly.float()
            )
            total_loss += loss.item()
            total_steps += 1
        except Exception as e:
            print(f"  [WARN] Skip eval idx={idx}: {e}")
            continue

    if not all_preds:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "accuracy": 0.0,
                "val_loss": 0.0, "TP": 0, "FP": 0, "TN": 0, "FN": 0}

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

    return {
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "val_loss": total_loss / max(total_steps, 1),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n_total": n_total,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-HSTGAN Real CARLA Data Training Experiment"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actors", type=int, default=25)
    parser.add_argument("--train-batch", type=int, default=2000,
                        help="Max training frames per epoch")
    parser.add_argument("--val-batch", type=int, default=1000,
                        help="Max validation frames per epoch")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--actors-path", default="data/dataset/frame_actors.csv")
    parser.add_argument("--labels-path", default="data/dataset/frame_labels.csv")
    parser.add_argument("--events-path", default="data/dataset/event_labels.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max-frames", type=int, default=None,
                       help="Max frames total (debug mode)")
    group.add_argument("--all", action="store_true",
                       help="Use ALL frames (train/val/test full)")
    args = parser.parse_args()
    sys.exit(main(args))
