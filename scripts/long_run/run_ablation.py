#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py — 运行消融实验 A/B/C/D

每次迭代只改一个组件：
  基线 (v6):   skip_conn=True, gamma=3.0, alpha_cap=500, oversample=True
  A:           跳过 oversample (skip_conn=True, gamma=3.0, alpha_cap=500)
  B:           跳过 skip_conn (oversample=True, gamma=3.0, alpha_cap=500)
  C:           gamma=2.0 (oversample=True, skip_conn=True, alpha_cap=500)
  D:           alpha_cap=100 (oversample=True, skip_conn=True, gamma=3.0)

用法:
    python scripts/long_run/run_ablation.py [--max-frames 2000 | --all]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.long_run.exp_realdata import main as exp_main
from stk.gnn.trainer import FocalLoss

# ============================================================
# 消融配置表
# ============================================================
ABLATIONS = {
    "v6_full": {
        "label": "Full (v6)",
        "oversample": True,
        "gamma": 3.0,
        "alpha_cap": 500.0,
        "skip_conn": True,
        "train_batch": 2000,
        "epochs": 20,
    },
    "A_no_oversample": {
        "label": "w/o Oversample",
        "oversample": False,
        "gamma": 3.0,
        "alpha_cap": 500.0,
        "skip_conn": True,
        "train_batch": 2000,
        "epochs": 20,
    },
    "B_no_skipconn": {
        "label": "w/o Skip Connection",
        "oversample": True,
        "gamma": 3.0,
        "alpha_cap": 500.0,
        "skip_conn": False,
        "train_batch": 2000,
        "epochs": 20,
    },
    "C_gamma2": {
        "label": "gamma=2.0",
        "oversample": True,
        "gamma": 2.0,
        "alpha_cap": 500.0,
        "skip_conn": True,
        "train_batch": 2000,
        "epochs": 20,
    },
    "D_alpha100": {
        "label": "alpha_cap=100",
        "oversample": True,
        "gamma": 3.0,
        "alpha_cap": 100.0,
        "skip_conn": True,
        "train_batch": 2000,
        "epochs": 20,
    },
}


def run_single_ablation(name: str, cfg: dict, device: str, max_frames: int) -> dict:
    """
    在隔离环境中运行单个消融实验。

    通过直接修改 global state 的前哨来注入配置（FocalLoss gamma、alpha_cap）。
    由于 exp_realdata.py 内部通过 /'__main__'/ 的函数导入调用，通过
    修改 trainer 的 FocalLoss 和 compute_loss 实现。
    """
    import argparse

    # Build args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=device)
    parser.add_argument("--max-actors", type=int, default=25)
    parser.add_argument("--train-batch", type=int, default=cfg["train_batch"])
    parser.add_argument("--val-batch", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "exp_results" / "ablations" / name
    ))
    parser.add_argument("--actors-path", default="data/dataset/frame_actors.csv")
    parser.add_argument("--labels-path", default="data/dataset/frame_labels.csv")
    parser.add_argument("--events-path", default="data/dataset/event_labels.json")

    if max_frames is not None:
        parser.add_argument("--max-frames", type=int, default=max_frames)
        parser.add_argument("--all", action="store_true")
    else:
        parser.add_argument("--max-frames", type=int, default=None)
        parser.add_argument("--all", action="store_true", default=True)

    if cfg["oversample"]:
        parser.add_argument("--oversample-pos", action="store_true", default=True)
    else:
        parser.add_argument("--oversample-pos", action="store_true", default=False)

    args = parser.parse_args([])

    print(f"\n{'='*60}")
    print(f"  Ablation: {name} ({cfg['label']})")
    print(f"  gamma={cfg['gamma']}, alpha_cap={cfg['alpha_cap']}, "
          f"skip_conn={cfg['skip_conn']}, oversample={cfg['oversample']}")
    print(f"{'='*60}")

    # Apply skip connection change to K_HSTGAN
    if not cfg["skip_conn"]:
        from stk.gnn import k_hstgan
        orig_forward = k_hstgan.K_HSTGAN.forward

        def patched_forward(self, data, return_extras=False):
            """Override: remove h_temporal += h_spatial skip connection."""
            from stk.gnn.knowledge_injector import RSSResidualInjector, RuleStrengthEncoder
            x = data.x
            edge_index = data.edge_index
            edge_type = data.edge_type
            kappa_rss = data.kappa_rss
            kappa_rule = data.kappa_rule

            x_aug = self.rss_injector(x, kappa_rss)

            if return_extras:
                h_spatial, rgat_attn, per_head_h = self.rgat(
                    x_aug, edge_index, edge_type, return_attention=True)
            else:
                h_spatial = self.rgat(x_aug, edge_index, edge_type)

            h_spatial = self.rule_encoder(h_spatial, kappa_rule)

            N = x.size(0)
            h_seq = h_spatial.unsqueeze(1)
            delta_feat = data.delta_feat
            if delta_feat.dim() == 1:
                d_t = delta_feat.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)
            h_lstm, _ = self.delta_lstm(h_seq, d_t)
            h_lstm = h_lstm.squeeze(1)

            h_behavior = self.behavior_attn(h_lstm.unsqueeze(1))
            h_temporal = self.scene_transformer(h_lstm.unsqueeze(1))
            # --- DELIBERATELY REMOVED skip connection: h_temporal += h_spatial ---

            y_scene = self.scene_head(h_temporal)
            y_behavior = self.behavior_head(h_temporal)
            y_anomaly = self.anomaly_head(h_temporal)
            y_rule = self.rule_head(h_temporal)

            if return_extras:
                extras = {
                    "per_head_anomaly": torch.sigmoid(
                        self.anomaly_head(per_head_h)).squeeze(-1).detach(),
                    "rgat_attention": rgat_attn or {},
                    "h_spatial": h_spatial.detach(),
                    "h_temporal": h_temporal.detach(),
                    "edge_index": edge_index.detach(),
                    "edge_type": edge_type.detach(),
                    "delta_feat": delta_feat.detach(),
                }
                return y_anomaly, y_scene, y_behavior, y_rule, extras
            return y_anomaly, y_scene, y_behavior, y_rule

        k_hstgan.K_HSTGAN.forward = patched_forward

    # Apply gamma and alpha_cap via mock on trainer
    from stk.gnn import trainer as trainer_mod

    # Monkey-patch FocalLoss gamma only for this config
    original_focal = copy.deepcopy(trainer_mod.FocalLoss)

    class PatchedFocalLoss(trainer_mod.FocalLoss):
        def __init__(self):
            super().__init__(gamma=cfg["gamma"])

    trainer_mod.FocalLoss = PatchedFocalLoss

    # Monkey-patch alpha_t cap in _compute_loss
    original_compute_loss = trainer_mod.K_HSTGANTrainer._compute_loss

    def patched_compute_loss(self, y_anomaly, y_scene, y_behavior, y_rule,
                              target_anomaly, target_scene, target_behavior,
                              target_rule, epoch, attn_weights=None):
        """Use the configured alpha_cap."""
        # Copy original logic but with our alpha_cap
        w_main = self.stage_scheduler.get_w_main(epoch)
        stage = self.stage_scheduler.get_stage(epoch)
        gamma_3 = self.ws_scheduler.get_gamma(epoch)

        n_normal = (target_anomaly == 0).sum().float()
        n_anomaly = (target_anomaly == 1).sum().float()
        alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=cfg["alpha_cap"])
        alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())
        L0 = self.focal_loss(y_anomaly, target_anomaly, alpha=alpha_per_sample)

        L1 = torch.tensor(0.0, device=y_anomaly.device)
        L2 = torch.tensor(0.0, device=y_anomaly.device)
        L3 = torch.tensor(0.0, device=y_anomaly.device)

        L_reg = torch.tensor(0.0, device=y_anomaly.device)
        if attn_weights is not None:
            L_reg = L_reg + self.beta_attn_sparse * (attn_weights ** 2).mean()
        for p in self.model.parameters():
            L_reg = L_reg + self.lambda_reg * (p ** 2).sum()

        L_total = w_main * L0 + L_reg

        return L_total, {"L0": L0.item(), "L1": 0.0, "L2": 0.0,
                         "L3": 0.0, "L_reg": L_reg.item(),
                         "L_total": L_total.item()}

    trainer_mod.K_HSTGANTrainer._compute_loss = patched_compute_loss

    # Run experiment via main from exp_realdata
    try:
        result = exp_main(args)
        print(f"\n  ✓ {name} completed")
        return result
    except Exception as e:
        print(f"\n  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--max-frames", type=int, default=2000,
                        help="Use max-frames for faster ablation")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for training (default: cuda:0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  K-HSTGAN Ablation Experiments")
    print(f"  Device: {args.device}, max_frames={args.max_frames}")
    print("=" * 70)

    results = {}
    for name, cfg in ABLATIONS.items():
        t0 = time.time()
        res = run_single_ablation(name, cfg, args.device, args.max_frames)
        dt = time.time() - t0
        results[name] = {
            "config": cfg,
            "result": res,
            "time_sec": round(dt, 1),
        }

    # Save ablation results
    out_dir = PROJECT_ROOT / "exp_results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved all ablation results to {out_path}")

    # Print final summary table
    print("\n" + "=" * 70)
    print("  Ablation Results Summary")
    print("=" * 70)
    print(f"{'Name':20} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} "
          f"{'FN':>4} {'TN':>6}")
    print("-" * 70)
    for name, data in results.items():
        r = data.get("result", {})
        if isinstance(r, dict):
            tr = r.get("test", r)
            p = tr.get("P", 0)
            rec = tr.get("R", 0)
            f1 = tr.get("F1", 0)
            tp = tr.get("TP", 0)
            fp = tr.get("FP", 0)
            fn = tr.get("FN", 0)
            tn = tr.get("TN", 0)
        else:
            p = rec = f1 = tp = fp = fn = tn = 0
        print(f"{data['config']['label']:20} {p:>6.3f} {rec:>6.3f} "
              f"{f1:>6.3f} {tp:>4} {fp:>4} {fn:>4} {tn:>6}")
    print("=" * 70)


if __name__ == "__main__":
    main()
