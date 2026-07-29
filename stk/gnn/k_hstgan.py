# -*- coding: utf-8 -*-
"""
K-HSTGAN 完整模型（§4.1–4.5, 式 4.1–4.39）

四层结构：
  Input Layer → Spatial Encoding (RGAT) → Temporal Encoding (DHLSTM-Attn)
  → Knowledge Injection → Multi-Task Fusion Heads

输入：STKGGraphDataset 输出的 torch_geometric.Data 对象
输出：(y_anomaly, y_scene, y_behavior, y_rule) 四路多任务预测

关键超参数：
  F_in = 23 (18 基础 + 5 RSS)
  F_hidden = 64
  K = 15 (scene relations)
  H = 4 (attention heads)
  T = 30 (temporal window, max)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgat import RelationGAT
from .dhlstm_attn import DeltaGatedLSTM, BehaviorAttention, SceneTransformer
from .knowledge_injector import RSSResidualInjector, RuleStrengthEncoder


class MLP(nn.Module):
    """简单 MLP 分类头：in → hidden → out_dim。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class K_HSTGAN(nn.Module):
    """
    Knowledge-guided Hierarchical Spatio-Temporal Graph Attention Network.

    Architecture:
      1. Input Layer:     x_aug = [x ‖ LayerNorm(kappa_rss)]     (4.24–4.25)
      2. Spatial Layer:   h_spatial = RGAT(x_aug, edge_index, edge_type)  (4.4–4.11)
      3. Knowledge Inject:h_spatial' = h_spatial + RuleStrengthEncoder(kappa_rule)  (4.28)
      4. Temporal Layer:  ΔGatedLSTM → BehaviorAttention → SceneTransformer  (4.13–4.23)
      5. Fusion Heads:    Anomaly/Scene/Behavior/Rule classification  (4.35–4.38)
      6. Final Fusion:    y_fused = w0*y_a + w1*max(y_s) + w2*max(y_b) + w3*max(y_r)  (4.39)

    Args:
        base_node_dim:   基础节点特征维度（默认 18）
        rss_dim:         RSS 残差维度（默认 5）
        hidden_dim:      隐藏层维度 F'（默认 64）
        num_heads:       RGAT 多头注意力头数 H（默认 4）
        num_relations:   场景关系类型数 K（默认 15）
        rule_dim:        交规触发维度（默认 14）
        transformer_d_k: Transformer key 维度（默认 32）
        dropout:         全局 dropout 率
    """

    def __init__(
        self,
        base_node_dim: int = 18,
        rss_dim: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_relations: int = 15,
        rule_dim: int = 14,
        transformer_d_k: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rule_dim = rule_dim
        F_in = base_node_dim + rss_dim  # 23

        # === 1. RSS 残差注入（4.24–4.25）===
        self.rss_injector = RSSResidualInjector(
            base_dim=base_node_dim, rss_dim=rss_dim, normalize=True)

        # === 2. RGAT 空间编码（4.4–4.11）===
        self.rgat = RelationGAT(
            in_features=F_in, hidden_dim=hidden_dim,
            num_heads=num_heads, num_relations=num_relations, dropout=dropout)

        # === 3. 规则强度编码 + 残差注入（4.26–4.28）===
        self.rule_encoder = RuleStrengthEncoder(
            hidden_dim=hidden_dim, rule_dim=rule_dim)

        # === 4. 时序编码（4.13–4.23）===
        self.delta_lstm = DeltaGatedLSTM(
            hidden_dim=hidden_dim, input_dim=hidden_dim,
            delta_input_dim=4, dropout=dropout)
        self.behavior_attn = BehaviorAttention(hidden_dim=hidden_dim)
        self.scene_transformer = SceneTransformer(
            hidden_dim=hidden_dim, d_k=transformer_d_k, dropout=dropout)

        # === 5. 多任务融合头（4.32–4.38）===
        # Scene: MeanPool over Vehicle+Pedestrian → MLP → 3 classes
        self.scene_head = MLP(hidden_dim, hidden_dim, 3, dropout=dropout)
        # Behavior: MeanPool per event → MLP → 7 classes
        self.behavior_head = MLP(hidden_dim, hidden_dim, 7, dropout=dropout)
        # Anomaly (primary): MLP → scalar
        self.anomaly_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout)
        # Rule: Multi-label → 14-dim sigmoid
        self.rule_head = MLP(hidden_dim, hidden_dim, rule_dim, dropout=dropout)

        # === 6. 最终融合权重（4.39）===
        # y_fused = w0*y_a + w1*max(y_s) + w2*max(y_b) + w3*max(y_r)
        self.fusion_weights = nn.Parameter(
            torch.tensor([1.0, 0.1, 0.2, 0.3], dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, data, return_extras: bool = False):
        """
        单帧前向传播。

        Args:
            data: torch_geometric.Data，含以下字段：
                x:                  [N, F]    节点特征（F = base_dim = 18）
                edge_index:         [2, E]    场景边索引
                edge_type:          [E]       场景边类型
                kappa_rss:          [N, 5]    RSS 残差
                kappa_rule:         [N, 14]   交规触发强度
                delta_feat:         [4]       Δg_t 四元组
            return_extras: 是否返回多头方差/注意力等扩展字段（KS-NBCF 融合模块用）。

        Returns:
            y_anomaly:  [N, 1]   节点级异常概率（sigmoid 后）
            y_scene:    [N, 3]   场景类分布（softmax 后）
            y_behavior: [N, 7]   行为类分布（softmax 后）
            y_rule:     [N, 14]  规则触发概率（sigmoid 后）
            extras (optional): dict，含
                per_head_anomaly:  [N, H]  各注意力头异常概率（用于 §5.4.2.2 ε_t）
                rgat_attention:    Dict[int, Tensor]  关系类型 k → [H, E_k] 注意力
                h_spatial:         [N, F']           空间编码输出
                h_temporal:        [N, F']           时序编码输出
                edge_index:        [2, E]
                edge_type:         [E]
                delta_feat:        [4]
        """
        x = data.x              # [N, F]
        edge_index = data.edge_index
        edge_type = data.edge_type
        kappa_rss = data.kappa_rss    # [N, 5]
        kappa_rule = data.kappa_rule  # [N, 14]

        # === 1. RSS 残差拼接 → [N, 23] ===
        x_aug = self.rss_injector(x, kappa_rss)

        # === 2. RGAT 空间编码 → [N, F'] ===
        if return_extras:
            h_spatial, rgat_attention, per_head_h = self.rgat(
                x_aug, edge_index, edge_type, return_attention=True)
        else:
            h_spatial = self.rgat(x_aug, edge_index, edge_type)

        # === 3. 规则残差注入 → [N, F'] ===
        h_spatial = self.rule_encoder(h_spatial, kappa_rule)

        # === 4. 时序编码 ===
        # DeltaGatedLSTM 输入 [B, T, F']，B = N（节点），T = 1（单帧模式）。
        # 单帧模式下每个节点独立编码，T 维后续多帧窗口用 K-HSTGAN.forward_sequence。
        N = x.size(0)
        h_seq = h_spatial.unsqueeze(1)  # [N, 1, F']
        # delta_feat: [4] → 扩展到 [N, 1, 4]
        delta_feat = data.delta_feat
        if delta_feat.dim() == 1:
            d_t = delta_feat.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)  # [N, 1, 4]
        elif delta_feat.dim() == 2:
            d_t = delta_feat.unsqueeze(1)
        else:
            d_t = delta_feat
        h_lstm, _ = self.delta_lstm(h_seq, d_t)  # [N, 1, F']
        h_lstm = h_lstm.squeeze(1)  # [N, F']

        # 行为注意力（单帧：B=0，退化为节点内均匀权重）— 输入 [B, T, F']
        h_behavior = self.behavior_attn(
            h_lstm.unsqueeze(1))  # [N, F']

        # Scene Transformer：LSTM 序列 + 行为序列
        h_temporal = self.scene_transformer(
            h_lstm.unsqueeze(1))  # [N, F']
        # Add skip connection from spatial encoding to preserve node-level identity
        h_temporal = h_temporal + h_spatial

        # === 5. 多任务融合头（节点级预测：对每个节点输出独立分布）===
        y_scene = F.softmax(self.scene_head(h_temporal), dim=-1)        # [N, 3]
        y_behavior = F.softmax(self.behavior_head(h_temporal), dim=-1)  # [N, 7]
        y_anomaly = torch.sigmoid(self.anomaly_head(h_temporal))        # [N, 1]
        y_rule = torch.sigmoid(self.rule_head(h_temporal))              # [N, 14]

        if return_extras:
            # per-head anomaly 概率：用 RGAT 逐头 h（[N, H, F'] 副本）走 anomaly_head。
            # per_head_h: [N, H, F'] 经过同一 anomaly_head → [N, H, 1] → [N, H]
            per_head_logits = self.anomaly_head(per_head_h)        # [N, H, 1]
            per_head_anomaly = torch.sigmoid(per_head_logits).squeeze(-1)  # [N, H]
            extras = {
                "per_head_anomaly": per_head_anomaly.detach(),    # [N, H]
                "rgat_attention": rgat_attention or {},            # Dict[int, [H, E_k]]
                "h_spatial": h_spatial.detach(),                    # [N, F']
                "h_temporal": h_temporal.detach(),                  # [N, F']
                "edge_index": edge_index.detach(),
                "edge_type": edge_type.detach(),
                "delta_feat": delta_feat.detach(),
            }
            return y_anomaly, y_scene, y_behavior, y_rule, extras

        return y_anomaly, y_scene, y_behavior, y_rule

    def fused_score(self, y_anomaly, y_scene, y_behavior, y_rule) -> torch.Tensor:
        """
        最终融合分数（4.39）。
        y_fused = w0*y_a + w1*max(y_s) + w2*max(y_b) + w3*max(y_r)
        """
        w = self.fusion_weights
        s_a = y_anomaly.mean()
        s_s = y_scene.max(dim=-1).values.mean()
        s_b = y_behavior.max(dim=-1).values.mean()
        s_r = y_rule.max(dim=-1).values.mean()
        return w[0] * s_a + w[1] * s_s + w[2] * s_b + w[3] * s_r
