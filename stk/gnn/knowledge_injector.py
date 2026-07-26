# -*- coding: utf-8 -*-
"""
知识注入模块（§4.4, 式 4.24–4.28）

两种注入策略：
  1. RSSResidualInjector — RSS 残差向量拼接（Feature-Level，改变输入维度 18→23）
  2. RuleStrengthEncoder — 交规触发强度残差加（Feature-Level，不改变维度）

弱监督（Weak Supervision）在 trainer.py 中实现，此处仅提供编码器。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RSSResidualInjector(nn.Module):
    """
    RSS 残差向量拼接（4.24–4.25）。

    kappa_rss(v) = [
        d_min_long - d_long,   # 纵向安全距离余量
        d_min_lat  - d_lat,    # 横向安全距离余量
        TTC        - tau_safe, # 碰撞时间余量
        v          - v_limit,  # 超速余量
        brake      - brake_min # 制动余量
    ]

    拼接后输入维度：F_aug = F_base + 5（默认 18 + 5 = 23）

    Args:
        base_dim:  基础节点特征维度（默认 18）
        rss_dim:   RSS 残差维度（固定 5）
        normalize: 是否对拼接后的向量做 LayerNorm
    """

    def __init__(self, base_dim: int = 18, rss_dim: int = 5,
                 normalize: bool = True):
        super().__init__()
        self.base_dim = base_dim
        self.rss_dim = rss_dim
        self.aug_dim = base_dim + rss_dim
        self.norm = nn.LayerNorm(self.aug_dim) if normalize else nn.Identity()

    def forward(self, x: torch.Tensor, kappa_rss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         [N, base_dim]  基础节点特征（Vehicle/Ped/TL 均为 base_dim）
            kappa_rss: [N, rss_dim]   RSS 残差（非 vehicle 节点为全 0）

        Returns:
            x_aug:     [N, base_dim + rss_dim]  拼接后特征
        """
        if kappa_rss.dim() == 1:
            kappa_rss = kappa_rss.unsqueeze(0)
        x_aug = torch.cat([x, kappa_rss], dim=-1)  # [N, F+5]
        x_aug = self.norm(x_aug)
        return x_aug


class RuleStrengthEncoder(nn.Module):
    """
    交规触发强度编码 + 残差注入（4.26–4.28）。

    kappa_rule(v) ∈ R^{14}    (每条规则的 severity ∈ [0, 1])
    z_v^rule = MLP_rule(kappa_rule(v)) ∈ R^{F'}     (14 → 32 → 64, ReLU)
    h_v^spatial' = h_v^spatial + z_v^rule             (残差连接)

    参数量：14 × 32 + 32 × 64 = 2496

    Args:
        hidden_dim: F'（默认 64）
        rule_dim:   交规触发维度（固定 14）
    """

    def __init__(self, hidden_dim: int = 64, rule_dim: int = 14):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rule_dim = rule_dim
        self.mlp = nn.Sequential(
            nn.Linear(rule_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

    def forward(self, h_spatial: torch.Tensor, kappa_rule: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_spatial: [N, hidden_dim]  空间编码输出
            kappa_rule: [N, rule_dim]    交规触发强度

        Returns:
            h_out:     [N, hidden_dim]  注入后输出（残差）
        """
        z_rule = self.mlp(kappa_rule)  # [N, hidden_dim]
        h_out = h_spatial + z_rule
        return h_out
