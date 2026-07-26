# -*- coding: utf-8 -*-
"""
φ_fuse D-S 证据理论融合核心（§5.4，算法 5.4 入口，式 5.9–5.21）

辨识框架 Θ = {a, ¬a}（异常/正常）；焦元：{a}, {¬a}, Θ

m_rule({a}) = s_t,                m_rule(¬a) = (1−s_t)·𝟙[rule fires],     m_rule(Θ) = 1−m_rule({a})−m_rule(¬a)   (5.9–5.11)
m_GNN({a})  = p_t,                m_GNN(¬a)  = 1−p_t−ε_t,                  m_GNN(Θ)  = ε_t                        (5.12–5.14)
K           = m_rule({a})·m_GNN(¬a) + m_rule(¬a)·m_GNN({a})                                                     (5.18)

Dempster 组合 → m_fused                                                                                          (5.16–5.20)
决策：
  m_fused({a})  > 0.5 → anomaly
  m_fused(¬a)  > 0.5 → normal
  否则            → uncertain → 触发 evidence_chain 回溯
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class FusionResult:
    """D-S 融合结果（§5.4.3 输出 schema）。"""
    m_fused_anomaly: float       # m_fused({a})
    m_fused_normal: float        # m_fused(¬a)
    m_fused_uncertain: float     # m_fused(Θ)
    K: float                     # 冲突系数 (5.18)
    is_consistent: bool          # K ≤ τ_K
    needs_backtrack: bool        # 不一致，需 evidence_chain
    decision: str                # "anomaly" | "normal" | "uncertain"
    # 规范化 mass 向量（方便下游）
    m_fused: Dict[str, float]    # {"a": ..., "not_a": ..., "Theta": ...}


class DempsterShaferFuser(nn.Module):
    """
    D-S 证据理论融合核心。
    """
    # === 超参数（§5.4.4 表 5-3）===
    tau_K: float = 0.3       # 冲突阈值（论文正文 + 表 5-3）

    def __init__(self, tau_K: float = 0.3):
        super().__init__()
        self.tau_K = tau_K

    # ============================================================
    # Mass 分配函数
    # ============================================================
    @staticmethod
    def rule_mass(
        s_rule: torch.Tensor,       # [N] max_i severity (不为零的规则触发)
        rule_fires: torch.Tensor,   # [N] bool/int: 是否存在规则触发
    ) -> Dict[str, torch.Tensor]:
        """
        m_rule({a}) = s_t·𝟙[rule fires]
        m_rule(¬a)  = (1−s_t)·𝟙[rule fires]
        m_rule(Θ)   = 1 − m_rule({a}) − m_rule(¬a)

        Returns dict {"a": [N], "not_a": [N], "Theta": [N]}
        """
        fires = rule_fires.float()
        m_a = s_rule * fires
        m_not_a = (1.0 - s_rule) * fires
        m_theta = 1.0 - m_a - m_not_a
        # 数值钳位
        m_theta = m_theta.clamp(min=0.0, max=1.0)
        return {"a": m_a, "not_a": m_not_a, "Theta": m_theta}

    @staticmethod
    def gnn_mass(
        p_anomaly: torch.Tensor,   # [N, 1] → [N]
        epsilon: torch.Tensor,     # [N] 多头方差（§5.4.2.2）
    ) -> Dict[str, torch.Tensor]:
        """
        m_GNN({a})  = p_t
        m_GNN(¬a)   = 1 − p_t − ε_t
        m_GNN(Θ)    = ε_t

        Returns dict {"a": [N], "not_a": [N], "Theta": [N]}
        """
        p = p_anomaly.squeeze(-1)
        m_a = p
        m_theta = epsilon
        m_not_a = 1.0 - p - m_theta
        # 钳位
        m_not_a = m_not_a.clamp(min=0.0)
        # 归一化（允许浮点微调）
        total = m_a + m_not_a + m_theta
        total = total.clamp(min=1e-8)
        return {"a": m_a / total, "not_a": m_not_a / total, "Theta": m_theta / total}

    # ============================================================
    # Dempster 组合（§5.4.2.3，式 5.16–5.20）
    # ============================================================
    @staticmethod
    def dempster_combine(
        m1: Dict[str, torch.Tensor],
        m2: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Dempster 组合规则（两源）：
          K         = m1({a})·m2(¬a) + m1(¬a)·m2({a})               (5.18)
          m_fused({a})  = [m1({a})·m2({a}) + m1({a})·m2(Θ) + m1(Θ)·m2({a})] / (1−K)   (5.16)
          m_fused(¬a)   = [m1(¬a)·m2(¬a) + m1(¬a)·m2(Θ) + m1(Θ)·m2(¬a)] / (1−K)       (5.17)
          m_fused(Θ)    = m1(Θ)·m2(Θ) / (1−K)                       (5.19)

        Returns dict {"a": [N], "not_a": [N], "Theta": [N], "K": [N]}
        """
        K = m1["a"] * m2["not_a"] + m1["not_a"] * m2["a"]  # [N]
        denom = (1.0 - K).clamp(min=1e-8)

        fused_a = (m1["a"] * m2["a"]
                   + m1["a"] * m2["Theta"]
                   + m1["Theta"] * m2["a"]) / denom
        fused_not_a = (m1["not_a"] * m2["not_a"]
                       + m1["not_a"] * m2["Theta"]
                       + m1["Theta"] * m2["not_a"]) / denom
        fused_theta = m1["Theta"] * m2["Theta"] / denom

        return {"a": fused_a, "not_a": fused_not_a, "Theta": fused_theta, "K": K}

    # ============================================================
    # 决策
    # ============================================================
    def decide(self, m_fused: Dict[str, torch.Tensor]) -> FusionResult:
        """
        决策规则（5.21）：
          m_fused({a})  > 0.5 → anomaly
          m_fused(¬a)  > 0.5 → normal
          否则            → uncertain（触发 evidence_chain 回溯）

        输入为标量（单样本）
        """
        m_a = float(m_fused["a"].item()) if torch.is_tensor(m_fused["a"]) else float(m_fused["a"])
        m_na = float(m_fused["not_a"].item()) if torch.is_tensor(m_fused["not_a"]) else float(m_fused["not_a"])
        m_th = float(m_fused["Theta"].item()) if torch.is_tensor(m_fused["Theta"]) else float(m_fused["Theta"])
        K_val = float(m_fused["K"].item()) if torch.is_tensor(m_fused["K"]) else float(m_fused["K"])

        is_consistent = K_val <= self.tau_K
        if m_a > 0.5:
            decision = "anomaly"
        elif m_na > 0.5:
            decision = "normal"
        else:
            decision = "uncertain"

        # 触发 evidence_chain 回溯的必要条件：
        #   (1) D-S 决策不确定 OR (2) 冲突系数超过 τ_K（信号不一致）
        #   两者任一成立即回溯，避免漏检冲突情形下的风险决策。
        needs_backtrack = (decision == "uncertain") or (not is_consistent)

        return FusionResult(
            m_fused_anomaly=m_a,
            m_fused_normal=m_na,
            m_fused_uncertain=m_th,
            K=K_val,
            is_consistent=is_consistent,
            needs_backtrack=needs_backtrack,
            decision=decision,
            m_fused={"a": m_a, "not_a": m_na, "Theta": m_th},
        )

    # ============================================================
    # 融合总入口
    # ============================================================
    def forward(
        self,
        p_anomaly: torch.Tensor,       # [N, 1] GNN 异常概率
        epsilon: torch.Tensor,         # [N]    多头方差
        s_rule: torch.Tensor,          # [N]    max severity (规则触发)
        rule_fires: torch.Tensor,      # [N]    0/1 是否规则触发
    ) -> FusionResult:
        """
        融合 D-S 入口（对单样本推理；batch 时取 mean）。

        Returns:
            FusionResult（标量级别）
        """
        # 转为 [N] 向量
        m_rule = self.rule_mass(s_rule, rule_fires)
        m_gnn = self.gnn_mass(p_anomaly, epsilon)
        m_fused = self.dempster_combine(m_rule, m_gnn)
        # 取均值 → 决策
        m_avg = {
            "a": m_fused["a"].mean(),
            "not_a": m_fused["not_a"].mean(),
            "Theta": m_fused["Theta"].mean(),
            "K": m_fused["K"].mean(),
        }
        return self.decide(m_avg)
