#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeneralDyG: General Dynamic Graph Network

基线模型 3：GeneralDyG

核心思路：
  - 通用动态图学习框架
  - 用于时空轨迹预测和异常检测
  - 结合图 Transformer 和 LSTM

参考论文：
  "GeneralDyG: A General Dynamic Graph Framework
   for Spatio-Temporal Anomaly Detection"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GeneralDyG_GCNBlock(nn.Module):
    """GCN + LSTM 组合块"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, h_prev=None):
        """
        x: [N, input_dim]
        edge_index: [2, E]
        """
        # GCN
        h_gcn = F.relu(self.gcn(x, edge_index))
        # LSTM
        h_seq = h_gcn.unsqueeze(1)  # [N, 1, hidden]
        if h_prev is not None:
            lstm_out, h_new = self.lstm(h_seq, h_prev)
        else:
            lstm_out, h_new = self.lstm(h_seq)
        h = self.norm(h_gcn + lstm_out.squeeze(1))
        return h, h_new


class GeneralDyG_TransformerBlock(nn.Module):
    """Transformer 块用于时序建模"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [N, 1, hidden]
        """
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class GeneralDyG(nn.Module):
    """
    GeneralDyG: 通用动态图网络

    架构：
      Input → GCNBlock → TransformerBlock → Multi-Task Heads
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入编码
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GCN + LSTM 块
        self.gcn_block1 = GeneralDyG_GCNBlock(hidden_dim, hidden_dim)
        self.gcn_block2 = GeneralDyG_GCNBlock(hidden_dim, hidden_dim)

        # Transformer 块
        self.transformer = GeneralDyG_TransformerBlock(
            hidden_dim, num_heads=num_heads, dropout=dropout
        )

        # 多任务分类头
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

    def forward(self, data, h_prev=None, return_extras: bool = False):
        """
        前向传播

        Args:
            data: torch_geometric.Data
            h_prev: LSTM 隐藏状态（时序模型使用）
        """
        x = data.x
        edge_index = data.edge_index

        # 输入投影
        h = F.relu(self.input_proj(x))

        # GCN + LSTM 块
        h, h_prev = self.gcn_block1(h, edge_index, h_prev)
        h, h_prev = self.gcn_block2(h, edge_index, h_prev)

        # Transformer（B=1 单帧模式）
        h_seq = h.unsqueeze(1)  # [N, 1, hidden]
        h = self.transformer(h_seq)
        h = h.squeeze(1)  # [N, hidden]

        # 多任务输出
        y_scene = F.softmax(self.scene_head(h), dim=-1)
        y_behavior = F.softmax(self.behavior_head(h), dim=-1)
        y_anomaly = torch.sigmoid(self.anomaly_head(h))
        y_rule = torch.sigmoid(self.rule_head(h))

        extras = {"h": h.detach()} if return_extras else None

        return y_anomaly, y_scene, y_behavior, y_rule, extras

    def fused_score(self, y_anomaly, y_scene, y_behavior, y_rule) -> torch.Tensor:
        """最终融合分数"""
        s_a = y_anomaly.mean()
        s_s = y_scene.max(dim=-1).values.mean()
        s_b = y_behavior.max(dim=-1).values.mean()
        s_r = y_rule.max(dim=-1).values.mean()
        w = torch.tensor([1.0, 0.1, 0.2, 0.3])
        return w[0] * s_a + w[1] * s_s + w[2] * s_b + w[3] * s_r


class GeneralDyGTrainer:
    """GeneralDyG 训练器"""

    def __init__(
        self,
        model: GeneralDyG,
        lr: float = 1e-3,
        max_epochs: int = 50,
        lambda_reg: float = 1e-4,
    ):
        self.model = model
        self.lr = lr
        self.max_epochs = max_epochs
        self.lambda_reg = lambda_reg
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.focal_loss = FocalLoss(gamma=3.0)

    def train_epoch(self, dataloader, epoch: int = 0) -> dict:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        epoch_metrics = {"L_total": 0.0, "L0": 0.0, "L_reg": 0.0}

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

            # L2 正则
            L_reg = self.lambda_reg * sum(
                (p ** 2).sum() for p in self.model.parameters()
            )

            L_total = L0 + 0.5 * L1 + 0.5 * L2 + 0.5 * L3 + L_reg
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += L_total.item()
            total_steps += 1
            epoch_metrics["L0"] += L0.item()
            epoch_metrics["L_reg"] += L_reg.item()
            epoch_metrics["L_total"] += L_total.item()

        if total_steps > 0:
            for k in epoch_metrics:
                epoch_metrics[k] /= total_steps
        return epoch_metrics

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
    """GeneralDyG 复现测试"""
    print("=" * 70)
    print("GeneralDyG Model Reproduction")
    print("=" * 70)

    device = torch.device("cpu")
    model = GeneralDyG(input_dim=18, hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GeneralDyG total params: {n_params:,}")

    from torch_geometric.data import Data
    x = torch.randn(10, 18)
    edge_index = torch.randint(0, 10, (2, 20))
    data = Data(x=x, edge_index=edge_index)

    y_a, y_s, y_b, y_r, extras = model(data)
    print(f"\nForward pass OK:")
    print(f"  y_anomaly: {tuple(y_a.shape)}")
    print(f"  y_scene: {tuple(y_s.shape)}")
    print(f"  y_behavior: {tuple(y_b.shape)}")
    print(f"  y_rule: {tuple(y_r.shape)}")
    print("=" * 70)


if __name__ == "__main__":
    main()