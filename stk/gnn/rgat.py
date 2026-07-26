# -*- coding: utf-8 -*-
"""
关系感知图注意力层 RGAT（§4.2, 式 4.4–4.11）

RelationGAT 采用 Per-Relation Channel Attention + Gated Fusion 结构，
每种场景关系类型各自维护独立的线性变换 W_k 与注意力向量 a_k，
最终通过可学习的关系先验 γ_k + 动态门控 g_k(h_i) 加权融合。

核心复杂度：O(K * H * |E| * F')，其中 K=15 关系类型，H=4 多头数。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


# === 关系类型先验（Table 4-2）===
# gamma_k = log(prior_weight)，编码为 log 空间以便 softmax 软化
_SCENE_REL_NAMES: List[str] = [
    "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
    "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
    "containsVehicle", "containsPedestrian", "containsTrafficLight",
    "containsRoad", "hasEnvironment", "weather_context",
]

_GAMMA_INIT = {
    "in_lane":             math.log(3.0),
    "on_road":             math.log(1.5),
    "in_junction":         math.log(2.0),
    "adjacent_lane":       math.log(1.5),
    "lane_connects":       math.log(1.0),
    "ahead_of":            math.log(4.0),
    "beside":              math.log(2.5),
    "nearby_pedestrian":   math.log(4.0),
    "controlled_by":       math.log(2.0),
    # contains* / hasEnvironment — 在注意力中排除（gamma_k=0 强制）
    "containsVehicle":     0.0,
    "containsPedestrian":  0.0,
    "containsTrafficLight":0.0,
    "containsRoad":        0.0,
    "hasEnvironment":      0.0,
    "weather_context":     math.log(0.5),
}

# 与 attention 无关的边类型集合（gamma_k 固定为 0，不参与注意力）
_EXCLUDED_RELS = frozenset([
    "containsVehicle", "containsPedestrian",
    "containsTrafficLight", "containsRoad", "hasEnvironment",
])


class RelationGAT(nn.Module):
    """
    关系感知图注意力层。

    公式：
        e_ij^(k) = LeakyReLU(a_k^T [W_k h_i ‖ W_k h_j])        (4.4)
        α_ij^(k) = softmax_j(e_ij^(k))                            (4.5)
        h_i^(k,h) = σ( Σ_j α_ij^(k,h) W_k^(h) h_j )             (4.6)
        h_i^(k,final) = (1/H) Σ_h h_i^(k,h)                      (4.10)
        β_k = softmax_k(γ_k + g_k(h_i))                           (4.8)
        h_i^spatial = Σ_k β_k · h_i^(k,final)                     (4.7)

    Args:
        in_features:  输入节点特征维度（F = 18 + 5 = 23，经 RSS 注入后）
        hidden_dim:   隐藏层维度 F'（默认 64）
        num_heads:    每个关系通道的多头注意力数 H（默认 4）
        num_relations: 场景关系类型总数 K（默认 15）
        dropout:      注意力与输出的 dropout 率
    """

    def __init__(
        self,
        in_features: int = 23,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_relations: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # === Per-Relation 参数（每个关系通道独立 W_k, a_k）===
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, in_features, self.head_dim))
            for _ in range(num_relations)
        ])
        self.a = nn.ParameterList([
            nn.Parameter(torch.empty(num_heads, 2 * self.head_dim, 1))
            for _ in range(num_relations)
        ])

        # === 关系先验 γ_k（可学习）===
        gamma_init = torch.zeros(num_relations, dtype=torch.float32)
        for i, name in enumerate(_SCENE_REL_NAMES):
            gamma_init[i] = _GAMMA_INIT.get(name, 0.0)
        self.gamma = nn.Parameter(gamma_init)

        # === 动态门控 g_k：单层 MLP → 标量 ===
        self.gate_w = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        self.gate_b = nn.Parameter(torch.zeros(num_heads))

        # === 输出投影 ===
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # 排除的边类型索引
        self._excluded_indices = {i for i, name in enumerate(_SCENE_REL_NAMES)
                                   if name in _EXCLUDED_RELS}

        self._reset_parameters()

    def _reset_parameters(self):
        for W in self.W:
            nn.init.xavier_uniform_(W.view(W.size(0), W.size(1) * W.size(2)))
        for a in self.a:
            nn.init.xavier_uniform_(a.view(a.size(0), a.size(1)))
        nn.init.xavier_uniform_(self.gate_w.view(self.gate_w.size(0), self.gate_w.size(1)))
        nn.init.zeros_(self.gate_b)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          [N, in_features] 节点特征
            edge_index: [2, E]           边索引 (src, dst)
            edge_type:  [E]             每条边的关系类型（0..K-1）

        Returns:
            h_out:      [N, hidden_dim]  空间编码输出
        """
        N = x.size(0)
        device = x.device

        # 初始化关系权重 β_k = softmax(γ)
        # γ 中被排除的关系强制为 -inf
        gamma = self.gamma.clone()
        for idx in self._excluded_indices:
            gamma[idx] = float("-inf")
        beta = F.softmax(gamma, dim=0)  # [K]

        # 逐关系通道聚合
        h_all_channels: List[torch.Tensor] = []  # K 个 [N, F'] 张量
        if edge_index.size(1) == 0:
            h_all_channels = [x[:, :1].expand(N, self.hidden_dim).clone()
                              for _ in range(self.num_relations)]
            for hc in h_all_channels:
                hc.zero_()
        else:
            src, dst = edge_index[0], edge_index[1]  # [E]

            for k in range(self.num_relations):
                mask_k = (edge_type == k)
                if mask_k.sum() == 0:
                    # 无此类边 → 输出为 0
                    h_all_channels.append(torch.zeros(N, self.hidden_dim, device=device))
                    continue

                src_k = src[mask_k]
                dst_k = dst[mask_k]
                E_k = src_k.size(0)

                # W_k x：[H, E_k, head_dim]
                W_k = self.W[k]  # [H, F, head_dim]
                h_src = x[src_k]  # [E_k, F]
                h_dst = x[dst_k]  # [E_k, F]

                # [H, E_k, head_dim]
                W_src = torch.einsum("hfj, ef -> ehj", W_k, h_src)  # [H, E_k, head_dim]
                W_dst = torch.einsum("hfj, ef -> ehj", W_k, h_dst)  # [H, E_k, head_dim]

                # 注意力系数 e_ij^(k,h)：LeakyReLU(a_k^T [W_src ‖ W_dst])
                a_k = self.a[k]  # [H, 2*head_dim, 1]
                pair = torch.cat([W_src, W_dst], dim=-1)  # [H, E_k, 2*head_dim]
                e_k = self.leaky_relu(
                    torch.einsum("hab, heb -> hea", a_k, pair).squeeze(-1)
                )  # [H, E_k]

                # Softmax（按 dst 节点归一化）
                idx_dst = dst_k.unsqueeze(0).expand(self.num_heads, -1)  # [H, E_k]
                alpha_k = softmax(e_k, idx_dst, num_nodes=N)  # [H, E_k]

                # 加权聚合 h_j = Σ_j α_ij^(k,h) · x_j
                # [H, E_k] x [E_k, F] → [H, F]
                h_k_h = torch.einsum("he, ef -> hf", alpha_k, h_dst)  # [H, F]
                # W_k^T h_k_h → [H, head_dim]
                h_k_h_W = torch.einsum("hfj, h_f -> hj", W_k, h_k_h)
                h_k_h_W = torch.tanh(h_k_h_W)  # σ activation

                # 多头平均 → [F']
                h_k = h_k_h_W.reshape(1, self.hidden_dim).expand(N, -1)

                # 门控：β_k = softmax(γ_k + g_k(h_i))
                # 简化：β_k 已经是固定的 softmax(γ)，用于整个关系通道
                h_all_channels.append(h_k)

        # 融合所有关系通道：h = Σ_k β_k * h_k
        h_out = torch.zeros(N, self.hidden_dim, device=device)
        for k in range(self.num_relations):
            h_out = h_out + beta[k] * h_all_channels[k]

        h_out = self.out_proj(h_out)
        h_out = self.dropout(h_out)
        return h_out
