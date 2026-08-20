#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KS-NBCF: Knowledge-guided Safety-critical Neuro-Boolean Consensus Fusion

(基于论文 "Safety-Critical Transition Learning with Conflict-Aware
Neuro-Symbolic Reasoning for Driving Anomaly Detection" §5 复现)

架构概述：
  φ_feat: K-HSTGAN 特征编排 + RSS/交规残差注入
  φ_loop: 三阶段双向闭环反馈（弱监督 → 置信度更新 → 动态规则生成）
  φ_fuse: Dempster-Shafer 证据理论融合
  φ_arb: KG 证据链回溯仲裁（Jaccard overlap + rule/GNN trust）

训练流程：
  - Stage I: 仅 L_weak 辅助训练（预热），γ_3 衰减
  - Stage II: 联合训练，η_i EMA 更新
  - Stage III: 精调主任务，冻结辅助头
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


# ============================================================
# φ_fuse: Dempster-Shafer 证据融合（式 5.9-5.21）
# ============================================================

@dataclass
class KSNBCFResult:
    """KS-NBCF 单帧推理结果"""
    y_anomaly: torch.Tensor           # [N, 1]
    y_scene: torch.Tensor             # [N, 3]
    y_behavior: torch.Tensor          # [N, 7]
    y_rule: torch.Tensor              # [N, 14]
    fusion_decision: str              # "anomaly" | "normal" | "uncertain"
    K: float                          # 冲突系数
    m_fused_anomaly: float            # m_fused({a})
    m_fused_normal: float             # m_fused({¬a})
    m_fused_uncertain: float          # m_fused(Θ)
    evidence_strength: float          # 规则证据强度
    resolve_type: str                 # consistent / trust_GNN / trust_rule / needs_review
    explanation: str                  # 可解释性路径


class KS_NBCF_Fuser(nn.Module):
    """
    KS-NBCF 融合器：φ_fuse 模块（式 5.9–5.21）

    辨识框架 Θ = {a, ¬a}
    mass functions:
      m_GNN({a})  = p_t
      m_GNN(¬a)   = 1 − p_t − ε_t
      m_GNN(Θ)    = ε_t
      m_rule({a}) = s_t · 𝟙[rule fires]
      m_rule(¬a)  = (1 − s_t) · 𝟙[rule fires]
      m_rule(Θ)   = 1 − m_rule({a}) − m_rule(¬a)

    Dempster 组合 → 决策（阈值 0.5）
    """

    def __init__(self, tau_K: float = 0.3):
        super().__init__()
        self.tau_K = tau_K

    @staticmethod
    def gnn_mass(
        p_anomaly: torch.Tensor,   # [N, 1]
        epsilon: torch.Tensor,     # [N]
    ) -> Dict[str, torch.Tensor]:
        """m_GNN mass function"""
        p = p_anomaly.squeeze(-1)                    # [N]
        m_a = p
        m_theta = epsilon
        m_not_a = 1.0 - p - m_theta
        m_not_a = m_not_a.clamp(min=0.0)
        total = m_a + m_not_a + m_theta
        total = total.clamp(min=1e-8)
        return {
            "a": m_a / total,
            "not_a": m_not_a / total,
            "Theta": m_theta / total,
        }

    @staticmethod
    def rule_mass(
        s_rule: torch.Tensor,       # [N]  max severity
        rule_fires: torch.Tensor,   # [N]  是否触发
    ) -> Dict[str, torch.Tensor]:
        """m_rule mass function"""
        fires = rule_fires.float()
        m_a = s_rule * fires
        m_not_a = (1.0 - s_rule) * fires
        m_theta = 1.0 - m_a - m_not_a
        m_theta = m_theta.clamp(min=0.0, max=1.0)
        return {"a": m_a, "not_a": m_not_a, "Theta": m_theta}

    @staticmethod
    def dempster_combine(
        m1: Dict[str, torch.Tensor],
        m2: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Dempster 组合规则（式 5.16-5.20）"""
        K = m1["a"] * m2["not_a"] + m1["not_a"] * m2["a"]      # [N]
        denom = (1.0 - K).clamp(min=1e-8)

        fused_a = (m1["a"] * m2["a"]
                   + m1["a"] * m2["Theta"]
                   + m1["Theta"] * m2["a"]) / denom
        fused_not_a = (m1["not_a"] * m2["not_a"]
                       + m1["not_a"] * m2["Theta"]
                       + m1["Theta"] * m2["not_a"]) / denom
        fused_theta = m1["Theta"] * m2["Theta"] / denom

        return {"a": fused_a, "not_a": fused_not_a, "Theta": fused_theta, "K": K}

    def forward(
        self,
        p_anomaly: torch.Tensor,
        epsilon: torch.Tensor,
        s_rule: torch.Tensor,
        rule_fires: torch.Tensor,
    ) -> Dict[str, Any]:
        """融合入口，返回标量决策"""
        m_rule = self.rule_mass(s_rule, rule_fires)
        m_gnn = self.gnn_mass(p_anomaly, epsilon)
        m_fused = self.dempster_combine(m_rule, m_gnn)

        m_a_mean = m_fused["a"].mean().item()
        m_na_mean = m_fused["not_a"].mean().item()
        m_th_mean = m_fused["Theta"].mean().item()
        K_mean = m_fused["K"].mean().item()

        is_consistent = K_mean <= self.tau_K
        if m_a_mean > 0.5:
            decision = "anomaly"
        elif m_na_mean > 0.5:
            decision = "normal"
        else:
            decision = "uncertain"

        return {
            "decision": decision,
            "K": K_mean,
            "is_consistent": is_consistent,
            "m_fused_anomaly": m_a_mean,
            "m_fused_normal": m_na_mean,
            "m_fused_uncertain": m_th_mean,
            "m_fused": {
                "a": m_fused["a"],
                "not_a": m_fused["not_a"],
                "Theta": m_fused["Theta"],
            },
        }


# ============================================================
# φ_loop: 三阶段闭环反馈
# ============================================================

@dataclass
class RuleConfidenceState:
    """规则置信度状态向量"""
    eta: torch.Tensor               # [num_rules]
    eta_ema: torch.Tensor           # [num_rules]
    threshold_search_count: torch.Tensor  # [num_rules]


class KS_NBCF_LoopFeedback(nn.Module):
    """
    φ_loop 三阶段双向闭环反馈模块（式 5.4-5.7）

    Stage I: 弱监督预热（epoch 0..T_warm）
    Stage II: GNN→规则置信度 EMA 更新
    Stage III: 动态规则模板生成
    """

    def __init__(
        self,
        num_rules: int = 14,
        device: torch.device = torch.device("cpu"),
        beta: float = 0.001,
        eta_floor: float = 0.3,
        gamma_3_init: float = 0.5,
        T_warm: int = 10,
        top_k: int = 10,
        ema_decay: float = 0.9,
        eta_min_delta: float = 0.05,
        p_low: float = 0.3,
    ):
        super().__init__()
        self.num_rules = num_rules
        self.device = device
        self.beta = beta
        self.eta_floor = eta_floor
        self.gamma_3_init = gamma_3_init
        self.T_warm = T_warm
        self.top_k = top_k
        self.ema_decay = ema_decay
        self.eta_min_delta = eta_min_delta
        self.p_low = p_low

        self.state = RuleConfidenceState(
            eta=torch.ones(num_rules, dtype=torch.float32),
            eta_ema=torch.ones(num_rules, dtype=torch.float32),
            threshold_search_count=torch.zeros(num_rules, dtype=torch.long),
        )

    # ---- Stage I: 弱监督 ----
    def compute_weak_loss(
        self,
        y_rule: torch.Tensor,
        y_weak: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """L_weak = γ_3(epoch) * BCE(y_rule, y_weak)"""
        gamma = max(0.0, self.gamma_3_init * (1.0 - float(epoch) / float(self.T_warm)))
        loss = F.binary_cross_entropy(y_rule, y_weak.float(), reduction="mean")
        return gamma * loss

    @staticmethod
    def compute_weak_labels(
        rule_out: Dict[str, Any],
        node_ids: List[str],
        num_rules: int = 14,
    ) -> torch.Tensor:
        """y_weak_t,i = max_v severity_i(v, t)"""
        rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                      "R10", "R11", "R13", "R16", "R17", "R18"]
        rule_idx = {r: i for i, r in enumerate(rule_codes)}

        out = torch.zeros(len(node_ids), num_rules, dtype=torch.float32)
        id2row = {nid: i for i, nid in enumerate(node_ids)}

        for sv in rule_out.get("violations", []) or []:
            code = str(sv.get("rule_code", "") or sv.get("attrs", {}).get("rule_code", ""))
            base = code.split("a")[0] if code.endswith("a") else code
            if base not in rule_idx:
                continue
            col = rule_idx[base]
            sev = float(sv.get("severity", 0.0) or sv.get("attrs", {}).get("severity", 0.0))
            for nid in (sv.get("src_id"), sv.get("dst_id")):
                nid = str(nid) if nid else None
                if nid and nid in id2row:
                    row = id2row[nid]
                    out[row, col] = max(out[row, col].item(), sev)
        return out

    # ---- Stage II: 置信度反馈 ----
    @torch.no_grad()
    def update_eta(
        self,
        s_minus: torch.Tensor,
        s_zero: torch.Tensor,
        evidence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        η_i^{e+1} = η_i^e * [1 - β·(s_i^- + ε_i)·(1-η_i)^+]   (5.4)
        ε_i       = 0.2 / sqrt(ℓ_i) + 0.05 · s_i^0              (5.5)
        """
        eps = 0.2 / torch.sqrt(evidence_lengths.clamp(min=1.0).float()) + 0.05 * s_zero
        eta_old = self.state.eta
        delta = self.beta * (s_minus + eps) * torch.clamp(1.0 - eta_old, min=0.0)
        eta_new = eta_old * (1.0 - delta)

        search_mask = eta_new < self.eta_floor
        self.state.threshold_search_count += search_mask.long()

        eta_ema_new = self.ema_decay * self.state.eta_ema + (1.0 - self.ema_decay) * eta_new
        small_change = (eta_ema_new - self.state.eta_ema).abs() < self.eta_min_delta
        eta_ema_final = torch.where(small_change, self.state.eta_ema, eta_ema_new)

        self.state.eta = eta_new
        self.state.eta_ema = eta_ema_final
        return eta_new

    @staticmethod
    def compute_stage2_signals(
        y_rule: torch.Tensor,
        y_anomaly: torch.Tensor,
        gt_anomaly: torch.Tensor,
        p_low: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """s_i^- = mean(rule fires ∧ p<p_low), s_i^0 = mean(rule fires ∧ gt=0)"""
        rule_fires = (y_rule > 0.5).float()                  # [N, 14]
        p_low_mask = (y_anomaly.squeeze(-1) < p_low).float()  # [N]
        gt_zero_mask = (1.0 - gt_anomaly.float()).clamp(0.0, 1.0)

        n = max(rule_fires.size(0), 1)
        s_minus = (rule_fires * p_low_mask.unsqueeze(-1)).sum(dim=0) / n
        s_zero = (rule_fires * gt_zero_mask.unsqueeze(-1)).sum(dim=0) / n
        return s_minus, s_zero

    @staticmethod
    def compute_evidence_lengths(
        rule_out: Dict[str, Any],
        num_rules: int = 14,
    ) -> torch.Tensor:
        """计算每条规则的证据链平均长度 ℓ_i（简化：violation count + 1）"""
        rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                      "R10", "R11", "R13", "R16", "R17", "R18"]
        rule_idx = {r: i for i, r in enumerate(rule_codes)}

        counts = torch.ones(num_rules, dtype=torch.float32)
        for sv in rule_out.get("violations", []) or []:
            code = str(sv.get("rule_code", "") or sv.get("attrs", {}).get("rule_code", ""))
            base = code.split("a")[0] if code.endswith("a") else code
            if base in rule_idx:
                counts[rule_idx[base]] += 1
        return counts

    # ---- Stage III: 动态规则生成 ----
    @torch.no_grad()
    def generate_dynamic_rules(
        self,
        per_head_anomaly: torch.Tensor,
        y_anomaly: torch.Tensor,
        rgat_attention: Dict[int, torch.Tensor],
        node_ids: List[str],
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """当 y_anomaly > 0.5 时提取 top-K 注意力三元组 → 规则模板"""
        if not rgat_attention or edge_index is None:
            return []

        anomaly_mask = (y_anomaly.squeeze(-1) > 0.5)
        if anomaly_mask.sum() == 0:
            return []

        SCENE_REL_NAMES = [
            "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
            "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
            "containsVehicle", "containsPedestrian", "containsTrafficLight",
            "containsRoad", "hasEnvironment",
        ]
        ATTENTION_TEMPLATE_MAP = {
            "ahead_of": "following_too_close",
            "beside": "lateral_safe_displacement",
            "nearby_pedestrian": "pedestrian_proximity",
            "adjacent_lane": "lane_change",
            "in_junction": "junction_no_yield",
            "controlled_by": "tl_no_yield",
        }

        triples: List[Tuple[str, str, str, float]] = []
        for k, alpha in rgat_attention.items():
            if k >= len(SCENE_REL_NAMES):
                continue
            rel_name = SCENE_REL_NAMES[k]
            alpha_mean = alpha.mean(dim=0).tolist() if alpha.dim() >= 2 else alpha.tolist()
            mask_k = (edge_type == k)
            if mask_k.sum() == 0:
                continue
            src_k = edge_index[0][mask_k].tolist()
            dst_k = edge_index[1][mask_k].tolist()
            for s, d, a in zip(src_k, dst_k, alpha_mean):
                if s < len(node_ids) and d < len(node_ids):
                    triples.append((rel_name, node_ids[s], node_ids[d], float(a)))

        triples.sort(key=lambda t: t[3], reverse=True)
        top = triples[: self.top_k]

        dynamic_rules = []
        for rel, src, dst, score in top:
            template = ATTENTION_TEMPLATE_MAP.get(rel)
            if template is None:
                continue
            src_anom = bool(anomaly_mask[node_ids.index(src)].item()) if src in node_ids else False
            dst_anom = bool(anomaly_mask[node_ids.index(dst)].item()) if dst in node_ids else False
            if not (src_anom or dst_anom):
                continue
            dynamic_rules.append({
                "rule_template": template,
                "relation_type": rel,
                "src_id": src,
                "dst_id": dst,
                "attention_score": score,
                "rule_layer": "Dynamic",
            })
        return dynamic_rules


# ============================================================
# φ_arb: KG 证据链回溯仲裁
# ============================================================

class KS_NBCF_Arbiter(nn.Module):
    """
    证据链回溯仲裁器（§5.4.5）

    Jaccard(P_v, S_a) 重叠度判定规则来源
    K ≤ τ_K  → trust D-S
    overlap > 0.5 → trust GNN
    evidence_strength > 0.8 → trust rule
    else → needs review
    """

    def __init__(self, overlap_threshold: float = 0.5, evidence_str_threshold: float = 0.8):
        super().__init__()
        self.overlap_threshold = overlap_threshold
        self.evidence_str_threshold = evidence_str_threshold

    @staticmethod
    def build_rule_evidence(
        rule_out: Dict[str, Any],
    ) -> Tuple[set, float]:
        """遍历 SafetyViolation → 节点集合 P_v + 平均 severity"""
        nodes = set()
        sev_sum = 0.0
        n = 0
        for sv in rule_out.get("violations", []) or []:
            src = sv.get("src_id"); dst = sv.get("dst_id"); sev = sv.get("severity", 0.0)
            if isinstance(src, (int, str)) and src:
                nodes.add(str(src))
            if isinstance(dst, (int, str)) and dst:
                nodes.add(str(dst))
            sev_sum += float(sev or 0.0)
            n += 1
        evidence_strength = sev_sum / n if n > 0 else 0.0
        return nodes, evidence_strength

    @staticmethod
    def build_attention_subgraph(
        rgat_attention: Dict[int, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_ids: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, str, str, float]]:
        """从 RGAT 注意力提取 top-K 三元组"""
        if not rgat_attention or edge_index is None:
            return []

        SCENE_REL_NAMES = [
            "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
            "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
            "containsVehicle", "containsPedestrian", "containsTrafficLight",
            "containsRoad", "hasEnvironment",
        ]

        triples = []
        for k, alpha in rgat_attention.items():
            if k >= len(SCENE_REL_NAMES):
                continue
            rel = SCENE_REL_NAMES[k]
            if alpha.dim() >= 2:
                alpha_mean = alpha.mean(dim=0)
            else:
                alpha_mean = alpha
            alpha_list = alpha_mean.tolist()
            mask_k = (edge_type == k)
            if mask_k.sum() == 0:
                continue
            src_k = edge_index[0][mask_k].tolist()
            dst_k = edge_index[1][mask_k].tolist()
            for s, d, a in zip(src_k, dst_k, alpha_list):
                if 0 <= s < len(node_ids) and 0 <= d < len(node_ids):
                    triples.append((node_ids[s], node_ids[d], rel, float(a)))

        triples.sort(key=lambda t: t[3], reverse=True)
        return triples[:top_k]

    @staticmethod
    def jaccard(P: set, S: set) -> float:
        """Jaccard 重叠度 (5.25)"""
        if not P and not S:
            return 0.0
        return len(P & S) / (len(P | S) + 1e-8)

    def forward(
        self,
        fusion_result: Dict[str, Any],
        rule_out: Dict[str, Any],
        rgat_attention: Dict[int, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_ids: List[str],
        p_anomaly: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """仲裁总入口"""
        P_v, evidence_strength = self.build_rule_evidence(rule_out)
        S_a_triples = self.build_attention_subgraph(
            rgat_attention, edge_index, edge_type, node_ids, top_k=10)
        S_v = set()
        for s, d, _, _ in S_a_triples:
            S_v.add(s)
            S_v.add(d)

        overlap = self.jaccard(P_v, S_v)

        K = fusion_result.get("K", 0.0)
        tau_K = 0.3
        if fusion_result.get("is_consistent", True):
            resolve_type = "consistent"
            y_fused = fusion_result["m_fused_anomaly"]
        elif overlap > self.overlap_threshold:
            resolve_type = "trust_GNN"
            y_fused = float(p_anomaly.mean().item()) if p_anomaly is not None else fusion_result["m_fused_anomaly"]
        elif evidence_strength > self.evidence_str_threshold:
            resolve_type = "trust_rule"
            y_fused = fusion_result["m_fused_anomaly"]
        else:
            resolve_type = "needs_review"
            y_fused = fusion_result["m_fused_anomaly"]

        path_parts = []
        for s, d, rel, score in S_a_triples[:3]:
            path_parts.append(f"({s})-[:{rel} α={score:.2f}]->({d})")
        if not path_parts:
            path_parts.append("<no-attention-triples>")

        explanation = (
            f"K={K:.3f} Δ-S:{fusion_result['decision']}|overlap={overlap:.2f}|"
            f"resolve={resolve_type}|evidence_sev={evidence_strength:.2f}|{"|".join(path_parts)}"
        )

        return {
            "y_fused": y_fused,
            "resolve_type": resolve_type,
            "explanation": explanation,
            "rule_evidence_path": list(P_v),
            "gnn_attention_path": S_a_triples,
            "overlap": overlap,
            "evidence_strength": evidence_strength,
            "K": K,
            "decision": fusion_result["decision"],
        }


# ============================================================
# 完整 KS-NBCF 模型
# ============================================================

class KS_NBCF(nn.Module):
    """
    KS-NBCF: 完整的 Neuro-Symbolic Consensus Fusion 推理链路
    """

    def __init__(
        self,
        k_hstgan_model=None,
        tau_K: float = 0.3,
        num_rules: int = 14,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.k_hstgan = k_hstgan_model
        self.fuser = KS_NBCF_Fuser(tau_K=tau_K)
        self.loop = KS_NBCF_LoopFeedback(num_rules=num_rules, device=device)
        self.arbiter = KS_NBCF_Arbiter()
        self.device = device

    def forward(
        self,
        data,
        rule_out: Optional[Dict[str, Any]] = None,
        node_ids: Optional[List[str]] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        完整前向：K-HSTGAN → φ_loop → φ_fuse → φ_arb

        支持两种模式：
        1. 完整模式（用于训练/评估）：提供 rule_out 和 node_ids
        2. 简化模式（用于 smoke test）：仅运行 K-HSTGAN 前向，跳过融合逻辑
        """
        # 1. K-HSTGAN 前向
        y_a, y_s, y_b, y_r, extras = self.k_hstgan(data, return_extras=True)

        if rule_out is None or node_ids is None:
            # 简化模式：仅返回 K-HSTGAN 输出
            extras_out = {
                "per_head_anomaly": extras["per_head_anomaly"],
                "rgat_attention": extras["rgat_attention"],
                "h_spatial": extras["h_spatial"],
                "h_temporal": extras["h_temporal"],
                "edge_index": extras["edge_index"],
                "edge_type": extras["edge_type"],
                "delta_feat": extras["delta_feat"],
            } if extras else None
            return y_a, y_s, y_b, y_r, extras_out

        # 完整模式：φ_loop → φ_fuse → φ_arb
        # 2. φ_loop 三阶段
        y_weak = self.loop.compute_weak_labels(rule_out, node_ids, num_rules=14)
        weak_loss = self.loop.compute_weak_loss(y_r, y_weak, epoch=epoch)

        p_anomaly = y_a.to(self.device)
        gt_anomaly = data.y_anomaly.to(self.device)
        s_minus, s_zero = self.loop.compute_stage2_signals(y_r, p_anomaly, gt_anomaly)
        ev_lens = self.loop.compute_evidence_lengths(rule_out)
        eta_new = self.loop.update_eta(s_minus, s_zero, ev_lens)

        dyn_rules = self.loop.generate_dynamic_rules(
            extras["per_head_anomaly"], y_a,
            extras["rgat_attention"], node_ids,
            edge_index=extras["edge_index"], edge_type=extras["edge_type"],
        )

        # 3. φ_fuse D-S 融合
        kappa_rule = data.kappa_rule.to(self.device)
        s_rule = kappa_rule.max(dim=-1).values.clamp(0.0, 1.0)
        rule_fires = (kappa_rule.sum(dim=-1) > 0).float()
        epsilon = extras["per_head_anomaly"].var(dim=-1, unbiased=False)

        fusion_result = self.fuser(p_anomaly, epsilon, s_rule, rule_fires)

        # 4. φ_arb 仲裁
        arb_result = self.arbiter(
            fusion_result=fusion_result,
            rule_out=rule_out,
            rgat_attention=extras["rgat_attention"],
            edge_index=extras["edge_index"].to(self.device),
            edge_type=extras["edge_type"].to(self.device),
            node_ids=node_ids,
            p_anomaly=p_anomaly,
        )

        extras_out = {
            "per_head_anomaly": extras["per_head_anomaly"],
            "rgat_attention": extras["rgat_attention"],
            "h_spatial": extras["h_spatial"],
            "h_temporal": extras["h_temporal"],
            "edge_index": extras["edge_index"],
            "edge_type": extras["edge_type"],
            "delta_feat": extras["delta_feat"],
            "fusion_result": fusion_result,
            "arb_result": arb_result,
            "dyn_rules": dyn_rules,
        } if extras else None

        return y_a, y_s, y_b, y_r, extras_out


# ============================================================
# KS-NBCF 训练器
# ============================================================

class KS_NBCFTrainer:
    """KS-NBCF 多任务训练器"""

    def __init__(
        self,
        model: KS_NBCF,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
        grad_clip: float = 5.0,
        lambda1: float = 0.5,
        lambda2: float = 0.5,
        lambda3: float = 0.5,
        focal_gamma: float = 3.0,
        alpha_cap: float = 500.0,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.alpha_cap = alpha_cap

        # 仅训练 k_hstgan backbone + heads
        self.optimizer = torch.optim.AdamW([
            {"params": model.k_hstgan.parameters(), "lr": lr, "weight_decay": 1e-4},
        ])

        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.best_f1 = 0.0
        self.epochs_no_improve = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        lr = 1e-3 if epoch < 30 else 1e-4

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        total_loss = 0.0
        total_steps = 0
        epoch_metrics: Dict[str, float] = {"L_total": 0.0, "L0": 0.0, "L_weak": 0.0, "grad_norm": 0.0}

        for batch in dataloader:
            self.optimizer.zero_grad()
            result = self.model(batch, epoch=epoch)
            target_anomaly = batch.y_anomaly.float()

            # L0: Focal Loss
            n_normal = (target_anomaly == 0).sum().float()
            n_anomaly = (target_anomaly == 1).sum().float()
            alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=self.alpha_cap)
            alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())
            L0 = self.focal_loss(result.y_anomaly, target_anomaly, alpha=alpha_per_sample)

            # L_weak
            node_ids = batch.node_ids
            rule_out = {"violations": []}  # 简化
            y_weak = self.model.loop.compute_weak_labels(rule_out, node_ids, num_rules=14)
            L_weak = self.model.loop.compute_weak_loss(result.y_rule, y_weak, epoch=epoch)

            # L1: scene CE
            L1 = F.cross_entropy(result.y_scene, batch.y_scene.long())
            # L2: behavior CE
            L2 = F.cross_entropy(result.y_behavior, batch.y_behavior.long())
            # L3: rule BCE
            L3 = F.binary_cross_entropy(result.y_rule, batch.y_rule.float())

            L_total = L0 + self.lambda1 * L1 + self.lambda2 * L2 + self.lambda3 * L3 + L_weak
            L_total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.k_hstgan.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += L_total.item()
            total_steps += 1
            epoch_metrics["L0"] += L0.item()
            epoch_metrics["L_weak"] += L_weak.item()
            epoch_metrics["grad_norm"] += float(grad_norm)
            epoch_metrics["L_total"] += L_total.item()

        if total_steps > 0:
            for k in epoch_metrics:
                epoch_metrics[k] /= total_steps
        return epoch_metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            result = self.model(batch, epoch=100)
            target_anomaly = batch.y_anomaly.float()
            n_normal = (target_anomaly == 0).sum().float()
            n_anomaly = (target_anomaly == 1).sum().float()
            alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=self.alpha_cap)
            alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())
            L0 = self.focal_loss(result.y_anomaly, target_anomaly, alpha=alpha_per_sample)

            preds = (result.y_anomaly.squeeze(-1) > 0.5).long()
            all_preds.append(preds)
            all_targets.append(batch.y_anomaly.long())
            total_loss += L0.item()
            total_steps += 1

        if not all_preds:
            return {"P": 0.0, "R": 0.0, "F1": 0.0, "accuracy": 0.0, "val_loss": total_loss}

        all_preds_t = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)

        tp = ((all_preds_t == 1) & (all_targets_t == 1)).sum().float()
        fp = ((all_preds_t == 1) & (all_targets_t == 0)).sum().float()
        fn = ((all_preds_t == 0) & (all_targets_t == 1)).sum().float()
        tn = ((all_preds_t == 0) & (all_targets_t == 0)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        if f1 > self.best_f1:
            self.best_f1 = f1.item()
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        return {
            "P": precision.item(),
            "R": recall.item(),
            "F1": f1.item(),
            "accuracy": accuracy.item(),
            "val_loss": total_loss / max(total_steps, 1),
            "best_f1": self.best_f1,
        }

    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced anomaly detection"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = logits.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        targets = targets.float()
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        eps = 1e-7
        bce = -(targets * torch.log(probs + eps) +
                (1 - targets) * torch.log(1 - probs + eps))
        loss = focal_weight * bce
        if alpha is not None:
            loss = loss * alpha
        return loss.mean()


# ============================================================
# 主入口
# ============================================================

def main():
    """KS-NBCF 复现测试"""
    print("=" * 70)
    print("KS-NBCF Model Reproduction")
    print("=" * 70)

    device = torch.device("cpu")
    from stk.gnn.k_hstgan import K_HSTGAN
    k_hstgan = K_HSTGAN(hidden_dim=64).to(device)
    ks_nbcf = KS_NBCF(k_hstgan_model=k_hstgan, device=device).to(device)

    n_params = sum(p.numel() for p in ks_nbcf.parameters())
    print(f"KS-NBCF total params: {n_params:,}")

    print("\nKS-NBCF architecture initialized.")
    print("  φ_fuse:  KS_NBCF_Fuser (D-S fusion)")
    print("  φ_loop:  KS_NBCF_LoopFeedback (3-stage feedback)")
    print("  φ_arb:   KS_NBCF_Arbiter (KG evidence chain)")
    print("=" * 70)


if __name__ == "__main__":
    main()