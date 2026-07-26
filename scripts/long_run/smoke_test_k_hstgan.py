#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K-HSTGAN Smoke Test —— 端到端流程验证（HANDOVER §6.2 红灯缓解）

目标：
  1. 跑通 PipelineOrchestrator → snapshot_store → STKGGraphDataset 的数据管线
  2. 实例化 K_HSTGAN 模型，前向 + 反向各 1 轮
  3. 验证输出维度与损失有限（非 NaN/Inf）
  4. 验证梯度范数 > 0
  5. （可选）跑 1 个 epoch 的训练验证 K_HSTGANTrainer

验收标准：forward 无报错 + loss finite + grad_norm > 0

Run:
    python scripts/long_run/smoke_test_k_hstgan.py
    python scripts/long_run/smoke_test_k_hstgan.py --scenario S00 --backend cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from stk.pipeline.orchestrator import PipelineOrchestrator
from stk.gnn.exporter import STKGGraphDataset, extract_stkg_tensors
from stk.gnn.k_hstgan import K_HSTGAN
from stk.gnn.trainer import K_HSTGANTrainer


def main(scenario_id: str = "S00", max_frames: int = 6,
         device_name: str = "cpu", train_one_epoch: bool = True) -> int:
    print("=" * 72)
    print(f"K-HSTGAN Smoke Test")
    print(f"  scenario: {scenario_id}, max_frames: {max_frames}, device: {device_name}")
    print("=" * 72)

    device = torch.device(device_name)
    t0 = time.time()

    # ========== Step 1. 运行 pipeline 获取 snapshots ==========
    print("\n[1/5] Running pipeline orchestrator...")
    orchestrator = PipelineOrchestrator()
    summary = orchestrator.run_scenario(scenario_id, max_frames=max_frames)
    print(f"  OK: scenario={summary['scenario']}, frames={summary['frames']}")
    for fr in summary["results"][:3]:
        print(f"    frame {fr['frame_id']}: n_violations={fr['n_violations']}, n_deltas={fr['n_deltas']}")
    if summary["frames"] == 0:
        print("[ERROR] No frames generated — pipeline failed")
        return 1

    # ========== Step 2. 构造数据集 ==========
    print("\n[2/5] Building STKGGraphDataset...")
    frame_ids = sorted(orchestrator.snapshot_store.list_frame_ids())[:max_frames]
    snapshots = [orchestrator.snapshot_store.get(fid) for fid in frame_ids]
    dataset = STKGGraphDataset(snapshots)
    print(f"  dataset length: {len(dataset)}")

    # 检查第一帧
    data0 = dataset[0]
    print(f"  first frame: x.shape={tuple(data0.x.shape)}, "
          f"edge_index={tuple(data0.edge_index.shape)}, "
          f"edge_type={tuple(data0.edge_type.shape)}")
    print(f"  kappa_rss: {tuple(data0.kappa_rss.shape)}, "
          f"kappa_rule: {tuple(data0.kappa_rule.shape)}, "
          f"env_feat: {tuple(data0.env_feat.shape)}, "
          f"delta_feat: {tuple(data0.delta_feat.shape)}")
    print(f"  labels: y_anomaly={tuple(data0.y_anomaly.shape)}, "
          f"y_scene={tuple(data0.y_scene.shape)}, "
          f"y_behavior={tuple(data0.y_behavior.shape)}, "
          f"y_rule={tuple(data0.y_rule.shape)}")
    print(f"  anomaly positives: {int(data0.y_anomaly.sum().item())}")

    # ========== Step 3. 实例化模型 + 前向传播 ==========
    print("\n[3/5] Instantiating K_HSTGAN model...")
    model = K_HSTGAN(
        base_node_dim=18, rss_dim=5, hidden_dim=64,
        num_heads=4, num_relations=15, rule_dim=14,
        transformer_d_k=32, dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    print("  forward pass...")
    data0 = data0.to(device)
    y_a, y_s, y_b, y_r = model(data0)
    print(f"  outputs: y_anomaly={tuple(y_a.shape)}, y_scene={tuple(y_s.shape)}, "
          f"y_behavior={tuple(y_b.shape)}, y_rule={tuple(y_r.shape)}")
    assert torch.isfinite(y_a).all(), "y_anomaly contains NaN/Inf"
    assert torch.isfinite(y_s).all(), "y_scene contains NaN/Inf"
    assert torch.isfinite(y_b).all(), "y_behavior contains NaN/Inf"
    assert torch.isfinite(y_r).all(), "y_rule contains NaN/Inf"
    print(f"  y_anomaly stats: min={y_a.min().item():.4f}, max={y_a.max().item():.4f}, "
          f"mean={y_a.mean().item():.4f}")
    print(f"  y_rule nonzero cells: {int((y_r > 0.5).sum().item())}")

    fused = model.fused_score(y_a, y_s, y_b, y_r)
    print(f"  fused score: {fused.item():.4f}")

    # ========== Step 4. 反向传播 + 梯度 ==========
    print("\n[4/5] Backward pass...")
    manual_loss = (
        torch.nn.functional.binary_cross_entropy(y_a.squeeze(-1), data0.y_anomaly.float())
        + torch.nn.functional.cross_entropy(y_s, data0.y_scene.long())
        + torch.nn.functional.cross_entropy(y_b, data0.y_behavior.long())
        + torch.nn.functional.binary_cross_entropy(y_r, data0.y_rule.float())
    )
    manual_loss.backward()
    grad_norm = 0.0
    n_grad_params = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item()) ** 2
            n_grad_params += 1
    grad_norm = grad_norm ** 0.5
    print(f"  backward loss: {manual_loss.item():.4f}")
    print(f"  gradient norm: {grad_norm:.4f} (over {n_grad_params} params)")
    assert grad_norm > 0, "All gradients are zero"
    assert torch.isfinite(manual_loss).all(), "Loss is NaN/Inf"

    # ========== Step 5. Trainer 1 个 epoch ==========
    if train_one_epoch:
        print("\n[5/5] Running 1 epoch via K_HSTGANTrainer...")
        trainer = K_HSTGANTrainer(model, lr=1e-3, max_epochs=1)

        # 简易 DataLoader：把 dataset 中每个 Data 当作一个独立 batch
        def collate(batch):
            return batch[0] if isinstance(batch, list) else batch
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
        metrics = trainer.train_epoch(loader, epoch=0)
        print(f"  train metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        eval_metrics = trainer.evaluate(loader)
        print(f"  eval metrics:")
        for k, v in eval_metrics.items():
            print(f"    {k}: {v:.4f}")
    else:
        print("\n[5/5] Skipping trainer test (pass --train to enable)")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"  ✅ SMOKE TEST PASSED ({elapsed:.2f}s)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-HSTGAN Smoke Test")
    parser.add_argument("--scenario", default="S00", help="Scenario ID (default: S00)")
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no-train", action="store_true", help="Skip trainer test")
    args = parser.parse_args()

    train_one_epoch = not args.no_train
    exit_code = main(scenario_id=args.scenario, max_frames=args.max_frames,
                     device_name=args.device, train_one_epoch=train_one_epoch)
    sys.exit(exit_code)
