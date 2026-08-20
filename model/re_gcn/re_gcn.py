#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RE-GCN: Relation Enhancement Graph Convolutional Network

基线模型 1：RE-GCN (IEEE TITS 2021)

核心思路：
  - 在 GCN 基础上引入关系增强机制
  - 对边特征进行编码，结合节点特征共同学习
  - 使用多关系图卷积进行异常检测

参考论文：
  "RE-GCN: Relational Event-based Graph Convolutional Network
   for Human Activity Recognition" (相关方法迁移至交通场景)
  或：
  "Relation Enhancement for Graph Neural Networks on
   Spatio-Temporal Anomaly Detection"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from typing import List, Optional, Tuple


class RE_GCN_NodeEncoder(nn.Module):
    """节点特征编码器"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RelationEdgeEncoder(nn.Module):
    """
    关系边编码器：对边类型进行嵌入
    论文中边类型数 K=15，嵌入维度 32
    """

    def __init__(self, num_relations: int = 15, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_relations, embed_dim)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.embed(edge_type)


class RelationEnhancedGCNLayer(MessagePassing):
    """
    关系增强图卷积层（式 3.2 - 3.4）

    输入：
      x: [N, F_in] 节点特征
      edge_index: [2, E] 边索引
      edge_type: [E] 边类型
      edge_attr: [E, F_edge] 边特征（可选）

    聚合方式：
      h_i' = σ( Σ_{j∈N(i)} α_ij ⊙ e_{ij} ⊙ x_j )
    其中 α_ij 为注意力权重，e_{ij} 为边类型嵌入
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int = 15,
        embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 节点变换矩阵（应用于聚合后的消息，维度为 out_channels）
        self.linear = nn.Linear(out_channels, out_channels)

        # 关系嵌入（用于边特征）
        self.relation_embed = nn.Embedding(num_relations, embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 注意力权重（使用边特征维度）
        self.attn_weight = nn.Parameter(torch.Tensor(out_channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight)
        nn.init.normal_(self.attn_weight, std=0.02)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        # 边特征编码
        edge_attr = self.edge_mlp(self.relation_embed(edge_type))  # [E, out]

        # 注意力权重（基于节点特征）
        src_nodes = x[edge_index[0]]  # [E, out]
        # attn_weight: [out, 1] -> unsqueeze 后使用 dot product
        # 避免 broadcast 错误
        attn_logits = (src_nodes * self.attn_weight.t()).sum(dim=-1, keepdim=True)  # [E, 1]
        alpha = torch.sigmoid(attn_logits)  # [E, 1]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, alpha=alpha, size=(x.size(0), x.size(0)))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        # 节点特征 + 边特征融合
        return (x_j * edge_attr) * alpha

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.linear(aggr_out)


class RE_GCN(nn.Module):
    """
    RE-GCN 完整模型

    架构：
      Input Layer → RelationEnhancedGCN × 2 layers → Multi-Task Heads

    输入：torch_geometric.Data
    输出：(y_anomaly, y_scene, y_behavior, y_rule)
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_relations: int = 15,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入编码
        self.node_encoder = RE_GCN_NodeEncoder(input_dim, hidden_dim)

        # 两层关系增强 GCN
        self.gc1 = RelationEnhancedGCNLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            dropout=dropout,
        )
        self.gc2 = RelationEnhancedGCNLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            dropout=dropout,
        )

        # 多任务融合头
        self.scene_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.behavior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 14),
        )

    def forward(self, data, return_extras: bool = False):
        """
        前向传播

        Args:
            data: torch_geometric.Data
                - x: [N, 18]
                - edge_index: [2, E]
                - edge_type: [E]
            return_extras: 是否返回中间特征

        Returns:
            y_anomaly: [N, 1]
            y_scene: [N, 3]
            y_behavior: [N, 7]
            y_rule: [N, 14]
            extras: 可选，含 h_spatial 等
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # 节点编码
        h = self.node_encoder(x)  # [N, hidden]

        # 第一层 GCN
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gc1(h, edge_index, edge_type)
        h = h.relu_()

        # 第二层 GCN
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gc2(h, edge_index, edge_type)
        h = h.relu_()

        # 多任务输出
        y_scene = F.softmax(self.scene_head(h), dim=-1)         # [N, 3]
        y_behavior = F.softmax(self.behavior_head(h), dim=-1)   # [N, 7]
        y_anomaly = torch.sigmoid(self.anomaly_head(h))          # [N, 1]
        y_rule = torch.sigmoid(self.rule_head(h))                # [N, 14]

        extras = {"h_spatial": h.detach()} if return_extras else None

        return y_anomaly, y_scene, y_behavior, y_rule, extras

    def fused_score(self, y_anomaly, y_scene, y_behavior, y_rule) -> torch.Tensor:
        """最终融合分数（固定权重）"""
        s_a = y_anomaly.mean()
        s_s = y_scene.max(dim=-1).values.mean()
        s_b = y_behavior.max(dim=-1).values.mean()
        s_r = y_rule.max(dim=-1).values.mean()
        w = torch.tensor([1.0, 0.1, 0.2, 0.3])
        return w[0] * s_a + w[1] * s_s + w[2] * s_b + w[3] * s_r


class RE_GCNTrainer:
    """RE-GCN 训练器"""

    def __init__(self, model: RE_GCN, lr: float = 1e-3, max_epochs: int = 50):
        self.model = model
        self.lr = lr
        self.max_epochs = max_epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.focal_loss = FocalLoss(gamma=3.0)

    def train_epoch(self, dataloader, epoch: int = 0) -> dict:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            y_a, y_s, y_b, y_r, _ = self.model(batch)

            target_anomaly = batch.y_anomaly.float()
            n_normal = (target_anomaly == 0).sum().float()
            n_anomaly = (target_anomaly == 1).sum().float()
            alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=500.0)
            alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())

            L0 = self.focal_loss(y_a, target_anomaly, alpha=alpha_per_sample)
            L1 = F.cross_entropy(y_s, batch.y_scene.long())
            L2 = F.cross_entropy(y_b, batch.y_behavior.long())
            L3 = F.binary_cross_entropy(y_r, batch.y_rule.float())
            L_total = L0 + 0.5 * L1 + 0.5 * L2 + 0.5 * L3

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += L_total.item()
            total_steps += 1

        return {"L_total": total_loss / max(total_steps, 1)}

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in dataloader:
            y_a, _, _, _, _ = self.model(batch)
            preds = (y_a.squeeze(-1) > 0.5).long()
            all_preds.append(preds)
            all_targets.append(batch.y_anomaly.long())

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

        return {"P": precision.item(), "R": recall.item(), "F1": f1.item(), "accuracy": accuracy.item()}


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, alpha=None):
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


def main():
    """RE-GCN 复现测试"""
    print("=" * 70)
    print("RE-GCN Model Reproduction")
    print("=" * 70)

    device = torch.device("cpu")
    model = RE_GCN(input_dim=18, hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RE-GCN total params: {n_params:,}")

    # 简单前向测试
    from torch_geometric.data import Data
    x = torch.randn(10, 18)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_type = torch.randint(0, 15, (20,))
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    y_a, y_s, y_b, y_r, extras = model(data)
    print(f"\nForward pass OK:")
    print(f"  y_anomaly: {tuple(y_a.shape)}")
    print(f"  y_scene: {tuple(y_s.shape)}")
    print(f"  y_behavior: {tuple(y_b.shape)}")
    print(f"  y_rule: {tuple(y_r.shape)}")
    print("=" * 70)


if __name__ == "__main__":
    main()