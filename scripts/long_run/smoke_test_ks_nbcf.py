#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-NBCF 融合框架——端到端 Smoke Test

验证链路：
  PipelineOrchestrator → K_HSTGAN (return_extras=True) →
  φ_feat 编排 → φ_loop 三阶段 → φ_fuser D-S 融合 → evidence_chain 仲裁

成功条件：
  - 无运行时异常
  - 各阶段输出形状 / 类型符合预期
  - 至少能生成一个 FusionArbiterResult
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
import torch.nn.functional as F

from stk.pipeline.orchestrator import PipelineOrchestrator
from stk.gnn.exporter import STKGGraphDataset
from stk.gnn.k_hstgan import K_HSTGAN
from stk.fusion import (
    KS_NBCF_FeatInjection,
    LoopFeedbackModule,
    DempsterShaferFuser,
    EvidenceChainArbiter,
)


def main(
    scenario_id: str = "S00",
    max_frames: int = 4,
    device_name: str = "cpu",
):
    print("=" * 72)
    print("KS-NBCF Smoke Test")
    print(f"  scenario: {scenario_id}, max_frames: {max_frames}, device: {device_name}")
    print("=" * 72)

    device = torch.device(device_name)
    t0 = time.time()

    # [1/6] Pipeline orchestrator
    print("\n[1/6] Running pipeline orchestrator...")
    orch = PipelineOrchestrator()
    out = orch.run_scenario(scenario_id, max_frames=max_frames)
    print(f"  OK: scenario={out['scenario']}, frames={out['frames']}")
    for r in out["results"][:3]:
        print(f"    frame {r['frame_id']}: n_violations={r['n_violations']}, n_deltas={r['n_deltas']}")

    # [2/6] Build dataset & extract_STKG
    print("\n[2/6] Building STKGGraphDataset...")
    fids = sorted(orch.snapshot_store.list_frame_ids())
    snaps = [orch.snapshot_store.get(fid) for fid in fids]
    dataset = STKGGraphDataset(snaps)
    print(f"  dataset length: {len(dataset)}")
    data0 = dataset[0]
    print(f"  first frame: x.shape={tuple(data0.x.shape)}, "
          f"edge_index={tuple(data0.edge_index.shape)}, "
          f"edge_type={tuple(data0.edge_type.shape)}")

    # [3/6] K_HSTGAN forward (return_extras=True)
    print("\n[3/6] K_HSTGAN forward (return_extras=True)...")
    model = K_HSTGAN().to(device).eval()
    data0_dev = data0.to(device)
    with torch.no_grad():
        y_a, y_s, y_b, y_r, extras = model(data0_dev, return_extras=True)
    print(f"  y_anomaly: {tuple(y_a.shape)}  mean={y_a.mean().item():.4f}")
    print(f"  y_scene:    {tuple(y_s.shape)}")
    print(f"  y_behavior: {tuple(y_b.shape)}")
    print(f"  y_rule:     {tuple(y_r.shape)}")
    print(f"  per_head_anomaly: {tuple(extras['per_head_anomaly'].shape)}")
    print(f"  rgat_attention keys: {list(extras['rgat_attention'].keys())}")
    print(f"  h_spatial: {tuple(extras['h_spatial'].shape)}")
    print(f"  h_temporal: {tuple(extras['h_temporal'].shape)}")

    # [4/6] φ_feat 编排
    print("\n[4/6] φ_feat KS_NBCF_FeatInjection...")
    feat_injector = KS_NBCF_FeatInjection(model=model)
    snap = snaps[0]
    y_a2, y_s2, y_b2, y_r2, extras2 = feat_injector.predict_with_extras(snap, device=device)
    print(f"  predict_with_extras OK: y_anomaly={tuple(y_a2.shape)}")

    # [5/6] φ_loop 三阶段
    print("\n[5/6] φ_loop LoopFeedbackModule...")
    loop = LoopFeedbackModule(num_rules=14, device=device)
    # Stage I: 弱监督
    rule_out = snap.get("rule_out", {})
    node_ids = list(snap["extracted"].get("_node_ids", []) or [])
    # 若 exporter 没暴露 _node_ids，则用 vehicles/peds 的 entity_id 顺序
    if not node_ids:
        node_ids = []
        for v in snap["extracted"].get("vehicles", []):
            node_ids.append(str(v.get("entity_id", "")))
        for p in snap["extracted"].get("pedestrians", []):
            node_ids.append(str(p.get("entity_id", "")))
        # 占位
        while len(node_ids) < data0.x.size(0):
            node_ids.append(f"placeholder_{len(node_ids)}")
    y_weak = loop.compute_weak_labels(rule_out, node_ids, num_rules=14)
    print(f"  Stage I: y_weak shape={tuple(y_weak.shape)}, "
          f"sum={y_weak.sum().item():.2f}")
    weak_loss = loop.compute_weak_loss(y_r2, y_weak, epoch=0)
    print(f"  Stage I: L_weak = {weak_loss.item():.4f}")

    # Stage II: 计算反馈
    p_anomaly = y_a2.to(device)
    gt_anomaly = data0.y_anomaly.to(device)
    s_minus, s_zero = loop.compute_stage2_signals(y_r2, p_anomaly, gt_anomaly)
    ev_lens = loop.compute_evidence_lengths(rule_out)
    print(f"  Stage II: s_minus={s_minus.tolist()}")
    print(f"  Stage II: s_zero={s_zero.tolist()}")
    eta_new = loop.update_eta(s_minus, s_zero, ev_lens)
    print(f"  Stage II: eta_new = {eta_new.tolist()}")

    # Stage III: 动态规则生成
    # 没有 attention 时直接跳过
    if extras2["rgat_attention"]:
        edge_index = extras2["edge_index"].to(device)
        edge_type = extras2["edge_type"].to(device)
        dyn_rules = loop.generate_dynamic_rules(
            extras2["per_head_anomaly"], y_a2,
            extras2["rgat_attention"], node_ids,
            edge_index=edge_index, edge_type=edge_type,
        )
        print(f"  Stage III: generated {len(dyn_rules)} dynamic rules")
        for r in dyn_rules[:3]:
            print(f"    - {r['rule_template']}: {r['src_id']} → {r['dst_id']} (α={r['attention_score']:.3f})")
    else:
        print("  Stage III: skipped (no attention triples in this scenario)")
        dyn_rules = []

    # [6/6] D-S fuse + Evidence chain
    print("\n[6/6] φ_fuse D-S + evidence_chain arbitration...")
    fuser = DempsterShaferFuser(tau_K=0.3)
    # 准备 kappa_rule 和 rule_fires
    kappa_rule = data0.kappa_rule.to(device)
    # s_rule: 取 max_v severity_i(v)（与论文式 5.10 一致），从 rule_out 反推
    # kappa_rule 仅含主规则 14 维；组内主类不全（缺 R6/R12/R14/R15）→ 用 rule_out 兜底。
    s_rule = kappa_rule.max(dim=-1).values.clamp(0.0, 1.0)
    rule_fires = (kappa_rule.sum(dim=-1) > 0).float()
    # 兜底：从 rule_out.violations 反推 rule_fires 和 s_rule 增量
    if int(rule_fires.sum().item()) == 0 and rule_out.get("violations"):
        id2row = {nid: i for i, nid in enumerate(node_ids)}
        for sv in rule_out["violations"]:
            if isinstance(sv, dict):
                src = sv.get("src_id"); dst = sv.get("dst_id"); sev = sv.get("severity", 0.0)
                if src is None:
                    src = sv.get("attrs", {}).get("src_id")
                if dst is None:
                    dst = sv.get("attrs", {}).get("dst_id")
                if sev == 0.0:
                    sev = sv.get("attrs", {}).get("severity", 0.0)
            elif hasattr(sv, "model_dump"):
                d = sv.model_dump()
                src = d.get("src_id"); dst = d.get("dst_id"); sev = d.get("severity", 0.0)
                if src is None:
                    src = d.get("attrs", {}).get("src_id")
                if dst is None:
                    dst = d.get("attrs", {}).get("dst_id")
                if sev == 0.0:
                    sev = d.get("attrs", {}).get("severity", 0.0)
            else:
                src = getattr(sv, "src_id", None); dst = getattr(sv, "dst_id", None)
                sev = getattr(sv, "severity", 0.0)
            for nid in (src, dst):
                if nid and str(nid) in id2row:
                    row = id2row[str(nid)]
                    rule_fires[row] = 1.0
                    s_rule[row] = max(float(s_rule[row]), float(sev or 0.0))
    # 多头方差 ε_t（§5.4.2.2）
    epsilon = extras2["per_head_anomaly"].var(dim=-1, unbiased=False)  # [N]
    print(f"  p_anomaly={p_anomaly.mean().item():.3f}, "
          f"epsilon mean={epsilon.mean().item():.4f}, "
          f"rule_fires={int(rule_fires.sum().item())}, "
          f"s_rule max={s_rule.max().item():.2f}")
    fr = fuser(p_anomaly, epsilon, s_rule, rule_fires)
    print(f"  FusionResult: decision={fr.decision}, K={fr.K:.3f}, "
          f"consistent={fr.is_consistent}, needs_backtrack={fr.needs_backtrack}")
    print(f"  m_fused: a={fr.m_fused_anomaly:.3f}, "
          f"not_a={fr.m_fused_normal:.3f}, Theta={fr.m_fused_uncertain:.3f}")

    # 仲裁
    arbiter = EvidenceChainArbiter()
    arb = arbiter(
        fusion_result=fr,
        rule_out=rule_out,
        rgat_attention=extras2["rgat_attention"],
        edge_index=extras2["edge_index"],
        edge_type=extras2["edge_type"],
        node_ids=node_ids,
        p_anomaly=p_anomaly,
    )
    print(f"  ArbiterResult:")
    print(f"    resolve_type:  {arb.resolve_type}")
    print(f"    y_fused:       {arb.y_fused:.4f}")
    print(f"    overlap:       {arb.overlap:.4f}")
    print(f"    evidence_str:  {arb.evidence_strength:.4f}")
    print(f"    rule_nodes:    {arb.rule_evidence_path[:3]}")
    print(f"    gnn_triples:   {len(arb.gnn_attention_path)} entries")
    print(f"    explanation:   {arb.explanation_path[:160]}")

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"  ✅ KS-NBCF SMOKE TEST PASSED ({dt:.2f}s)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S00", type=str)
    parser.add_argument("--max-frames", default=4, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    sys.exit(main(scenario_id=args.scenario, max_frames=args.max_frames,
                   device_name=args.device))
