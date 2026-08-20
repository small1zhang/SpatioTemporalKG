#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDN: Graph Domain Network

基线模型 2：GDN

核心思路：
  - 结合图神经网络与领域自适应
  - 用于跨域交通异常检测
  - 使用对比学习进行领域对齐

参考论文：
  "GDN: Graph Domain Network for Transferable Anomaly Detection
   in Autonomous Driving"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GDN_ContrastiveLoss(nn.Module):
    """对比损失：拉近正例，推远负例"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        SimCLR-style contrastive loss
        z_i, z_j: [N, D] 两个增强视图的表示
        """
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)   # [2N, D]
        z = F.normalize(z, dim=1)

        # 相似度矩阵
        sim = torch.matmul(z, z.t()) / self.temperature  # [2N, 2N]
        # 标签：同一节点的两个视图为正例
        label = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)], dim=0)
        loss = self.ce(sim, label)
        return loss


class GDN_Encoder(nn.Module):
    """GDN 编码器：GCN + 对比学习"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        """前向传播，返回投影后的表示"""
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        z = self.projection(h)
        return F.normalize(z, dim=1)


class GDN_DomainClassifier(nn.Module):
    """领域分类器：区分源域/目标域"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # 源域/目标域
        )

    def forward(self, h):
        return self.net(h)


class GDN(nn.Module):
    """
    GDN: Graph Domain Network
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 编码器
        self.encoder = GDN_Encoder(input_dim, hidden_dim, output_dim)

        # 多任务分类头
        self.scene_head = nn.Linear(output_dim, 3)
        self.behavior_head = nn.Linear(output_dim, 7)
        self.anomaly_head = nn.Linear(output_dim, 1)
        self.rule_head = nn.Linear(output_dim, 14)

        # 领域分类器
        self.domain_classifier = GDN_DomainClassifier(hidden_dim)

        # 对比损失
        self.contrastive_loss = GDN_ContrastiveLoss()

    def forward(self, data, domain_label: Optional[torch.Tensor] = None,
                return_extras: bool = False):
        """
        前向传播

        Args:
            data: torch_geometric.Data
            domain_label: [N] 0=源域, 1=目标域（用于领域适应）
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # 编码器
        z = self.encoder(x, edge_index)  # [N, output_dim]

        # 分类头
        y_scene = F.softmax(self.scene_head(z), dim=-1)
        y_behavior = F.softmax(self.behavior_head(z), dim=-1)
        y_anomaly = torch.sigmoid(self.anomaly_head(z))
        y_rule = torch.sigmoid(self.rule_head(z))

        extras = {"z": z.detach(), "domain_logits": None}

        # 领域分类损失（如果提供了 domain_label）
        if domain_label is not None:
            domain_logits = self.domain_classifier(z)
            extras["domain_logits"] = domain_logits

        return y_anomaly, y_scene, y_behavior, y_rule, extras

    def contrastive_forward(self, data1, data2) -> torch.Tensor:
        """
        对比学习前向：对两个视图的编码结果计算对比损失

        Args:
            data1, data2: 两个增强视图的 Data 对象
        """
        z1 = self.encoder(data1.x, data1.edge_index)
        z2 = self.encoder(data2.x, data2.edge_index)
        return self.contrastive_loss(z1, z2)


class GDNTrainer:
    """GDN 训练器"""

    def __init__(
        self,
        model: GDN,
        lr: float = 1e-3,
        max_epochs: int = 50,
        lambda_contrastive: float = 0.5,
        lambda_domain: float = 0.1,
    ):
        self.model = model
        self.lr = lr
        self.max_epochs = max_epochs
        self.lambda_contrastive = lambda_contrastive
        self.lambda_domain = lambda_domain
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.focal_loss = FocalLoss(gamma=3.0)

    def train_epoch(self, dataloader, epoch: int = 0) -> dict:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        epoch_metrics = {"L_total": 0.0, "L0": 0.0, "L_contrastive": 0.0, "L_domain": 0.0}

        for batch in dataloader:
            self.optimizer.zero_grad()

            y_a, y_s, y_b, y_r, extras = self.model(batch)
            target_anomaly = batch.y_anomaly.float()

            # L0: Focal Loss
            n_normal = (target_anomaly == 0).sum().float()
            n_anomaly = (target_anomaly == 1).sum().float()
            alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=500.0)
            alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())
            L0 = self.focal_loss(y_a, target_anomaly, alpha=alpha_per_sample)

            # L1/L2/L3
            L1 = F.cross_entropy(y_s, batch.y_scene.long())
            L2 = F.cross_entropy(y_b, batch.y_behavior.long())
            L3 = F.binary_cross_entropy(y_r, batch.y_rule.float())

            # 对比损失
            L_contrastive = torch.tensor(0.0, device=batch.x.device)
            if hasattr(batch, 'data_aug') and batch.data_aug is not None:
                L_contrastive = self.model.contrastive_forward(batch, batch.data_aug)

            # 领域分类损失
            L_domain = torch.tensor(0.0, device=batch.x.device)
            if extras.get("domain_logits") is not None and batch.domain_label is not None:
                L_domain = F.cross_entropy(
                    extras["domain_logits"],
                    batch.domain_label.long()
                )

            L_total = L0 + 0.5 * L1 + 0.5 * L2 + 0.5 * L3 \
                      + self.lambda_contrastive * L_contrastive \
                      + self.lambda_domain * L_domain

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += L_total.item()
            total_steps += 1
            epoch_metrics["L0"] += L0.item()
            epoch_metrics["L_contrastive"] += L_contrastive.item()
            epoch_metrics["L_domain"] += L_domain.item()
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
    """GDN 复现测试"""
    print("=" * 70)
    print("GDN (Graph Domain Network) Model Reproduction")
    print("=" * 70)

    device = torch.device("cpu")
    model = GDN(input_dim=18, hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GDN total params: {n_params:,}")

    from torch_geometric.data import Data
    x = torch.randn(10, 18)
    edge_index = torch.randint(0, 10, (2, 20))
    data = Data(x=x, edge_index=edge_index)

    y_a, y_s, y_b, y_r, extras = model(data)
    print(f"\nForward pass OK:")
    print(f"  y_anomaly: {tuple(y_a.shape)}")
    print(f"  z shape: {tuple(extras['z'].shape)}")
    print("=" * 70)


if __name__ == "__main__":
    main()