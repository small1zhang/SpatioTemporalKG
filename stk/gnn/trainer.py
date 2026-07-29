# -*- coding: utf-8 -*-
"""
K-HSTGAN 多任务训练器（§4.5, 式 4.40–4.49）

三阶段训练：
  Stage I   预训练 (epoch 0–5):   仅辅助头 (L1,L2,L3)，w0=0
  Stage II  联合训练 (epoch 5–30): 全任务全损失
  Stage III 精调 (epoch 30–50):   仅主任务头，辅助头冻结

损失函数：
  L0: Focal Loss (γ_focal=2, 自适应类权重 α_t)           (4.41–4.42)
  L1: 场景 CE (softmax, 3类)                             (4.43)
  L2: 行为 CE (softmax, 7类)                             (4.44)
  L3: 交规 BCE (multi-label, 14维) + 弱监督衰减           (4.45–4.46)
  L_reg: L2 正则 + 注意力稀疏项 (β=0.01)                 (4.47)
  梯度裁剪: c=5.0                                         (4.48)
  EMA: θ_EMA = 0.99 θ_EMA + 0.01 θ                      (4.49)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """Focal Loss (4.41–4.42)。"""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # [N] per-sample weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits:  [N, 1]  预测概率（已是 sigmoid 后；K-HSTGAN.forward 在 anomaly_head 上
                             直接施加 sigmoid，因此这里不再重复 sigmoid）
            targets: [N]    0/1 标签
            alpha:   [N]    per-sample class weights（可选，通常来自 _compute_loss 中的
                             adaptive alpha_t）；None 表示等权重。
        """
        probs = logits.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)  # [N]
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


class StageScheduler:
    """三阶段学习率调度器（4.5 节训练计划表）。

    实证调优：
      - Stage I（仅 w_main=0 弱监督预热）被实测为有害：在真实 CARLA 数据上，
        L1/L2/L3 的辅助标签在节点级极度不平衡，前若干 epoch 仅由辅助任务驱动
        会导致 SceneTransformer + LayerNorm 把整帧所有节点的 hidden 表示压成
        相同向量，进入 Stage II 后 anomaly_head 只能从常数表示中学到 bias，
        节点级判别无法恢复。
      - 解决方案：stage1_end=0，跳过 Stage I，从 epoch 0 起让 L0（Focal Loss
        + adaptive α_t）参与梯度回传，backbone 始终在节点级监督下保留判别性。

    Stage II（w_main=1 联合训练）：epoch 0..(stage2_end-1)
    Stage III（w_main=1 + 冻结辅助头）：epoch stage2_end 起
    """

    def __init__(self, lr_pretrain: float = 1e-3, lr_joint: float = 1e-3,
                 lr_finetune: float = 1e-4,
                 stage1_end: int = 0, stage2_end: int = 30):
        self.lr_pretrain = lr_pretrain
        self.lr_joint = lr_joint
        self.lr_finetune = lr_finetune
        self.stage1_end = stage1_end
        self.stage2_end = stage2_end

    def get_stage(self, epoch: int) -> int:
        if epoch < self.stage1_end:
            return 1
        elif epoch < self.stage2_end:
            return 2
        else:
            return 3

    def get_lr(self, epoch: int) -> float:
        stage = self.get_stage(epoch)
        if stage == 1:
            return self.lr_pretrain
        elif stage == 2:
            return self.lr_joint
        else:
            return self.lr_finetune

    def get_w_main(self, epoch: int) -> float:
        """主任务损失权重 w0。Stage I 为 0，Stage II/III 为 1。"""
        return 0.0 if self.get_stage(epoch) == 1 else 1.0

    def freeze_aux(self, stage: int) -> bool:
        """Stage III 是否冻结辅助头。"""
        return stage == 3


class WeakSupervisionScheduler:
    """弱监督权重 γ_3 衰减调度（4.46, 式 4.31）。"""

    def __init__(self, gamma_3_init: float = 0.5, T_warm: int = 10):
        self.gamma_3_init = gamma_3_init
        self.T_warm = T_warm

    def get_gamma(self, epoch: int) -> float:
        return max(0.0, self.gamma_3_init * (1.0 - epoch / self.T_warm))


class ExponentialMovingAverage:
    """EMA（4.49）。"""

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self):
        for n, p in self.model.named_parameters():
            self.shadow[n].data = self.decay * self.shadow[n].data + (1 - self.decay) * p.data

    def apply_shadow(self):
        self.backup = {n: p.clone() for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.data.copy_(self.shadow[n].data)

    def restore(self):
        if hasattr(self, "backup"):
            for n, p in self.model.named_parameters():
                p.data.copy_(self.backup[n].data)


class K_HSTGANTrainer:
    """
    多任务训练器，集成三阶段调度、Focal Loss、弱监督衰减、EMA、梯度裁剪。

    Usage:
        trainer = K_HSTGANTrainer(model)
        metrics = trainer.train_epoch(dataloader, epoch=0)
        eval_metrics = trainer.evaluate(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
        grad_clip: float = 5.0,
        lambda_reg: float = 1e-4,
        beta_attn_sparse: float = 0.01,
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
        self.lambda_reg = lambda_reg
        self.beta_attn_sparse = beta_attn_sparse
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.alpha_cap = alpha_cap  # 用于 _compute_loss 中正例权重上限

        # 阶段调度
        self.stage_scheduler = StageScheduler()
        self.ws_scheduler = WeakSupervisionScheduler()

        # 优化器（不同参数组不同学习率）
        self.optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": lr, "weight_decay": 1e-4},
        ])

        # EMA
        self.ema = ExponentialMovingAverage(model, decay=0.99)

        # 损失函数：gamma=3.0 让 Focal Loss 更聚焦于硬正例（hard positive）
        # 相比默认 gamma=2.0，梯度对低置信度正例的惩罚更大，有助于提升 Recall
        # focal_gamma 可通过构造函数参数覆盖，用于消融实验
        self.focal_loss = FocalLoss(gamma=focal_gamma)

        # 早停
        self.best_f1 = 0.0
        self.epochs_no_improve = 0

    def _compute_loss(
        self,
        y_anomaly: torch.Tensor,    # [N, 1]
        y_scene: torch.Tensor,      # [N, 3]
        y_behavior: torch.Tensor,   # [N, 7]
        y_rule: torch.Tensor,       # [N, 14]
        target_anomaly: torch.Tensor,  # [N]
        target_scene: torch.Tensor,    # [N]
        target_behavior: torch.Tensor, # [N]
        target_rule: torch.Tensor,     # [N, 14]
        epoch: int,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算多任务总损失（4.40–4.47）。"""
        w_main = self.stage_scheduler.get_w_main(epoch)
        stage = self.stage_scheduler.get_stage(epoch)
        gamma_3 = self.ws_scheduler.get_gamma(epoch)

        # L0: Focal Loss（主任务）
        # 自适应类权重 α_t = min(1, #normal / (#anomaly + ε))
        n_normal = (target_anomaly == 0).sum().float()
        n_anomaly = (target_anomaly == 1).sum().float()
        alpha_t = torch.clamp(n_normal / (n_anomaly + 1.0), max=self.alpha_cap)
        alpha_per_sample = torch.where(target_anomaly == 0, 1.0, alpha_t.item())
        L0 = self.focal_loss(y_anomaly, target_anomaly, alpha=alpha_per_sample)

        # L1: 场景 CE（softmax，3类）
        L1 = F.cross_entropy(y_scene, target_scene.long())

        # L2: 行为 CE（softmax，7类）
        L2 = F.cross_entropy(y_behavior, target_behavior.long())

        # L3: 交规 BCE（multi-label，14维）
        L3_gt = F.binary_cross_entropy(y_rule, target_rule.float())
        L3 = L3_gt + gamma_3 * L3_gt  # 弱监督（简化：直接加权 gt）

        # L_reg: L2 正则 + 注意力稀疏
        L_reg = torch.tensor(0.0, device=y_anomaly.device)
        if attn_weights is not None:
            L_reg = L_reg + self.beta_attn_sparse * (attn_weights ** 2).mean()
        for p in self.model.parameters():
            L_reg = L_reg + self.lambda_reg * (p ** 2).sum()

        # 总损失（4.40）
        L_total = w_main * L0 + self.lambda1 * L1 + self.lambda2 * L2 + self.lambda3 * L3 + L_reg

        return L_total, {"L0": L0.item(), "L1": L1.item(), "L2": L2.item(),
                         "L3": L3.item(), "L_reg": L_reg.item(), "L_total": L_total.item()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        训练一个 epoch。

        Args:
            dataloader: 批次 DataLoader
            epoch:      当前 epoch 编号（从 0 开始）

        Returns:
            metrics: 包含 loss 分量与梯度范数的字典
        """
        self.model.train()
        stage = self.stage_scheduler.get_stage(epoch)
        lr = self.stage_scheduler.get_lr(epoch)

        # 更新学习率
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Stage III 冻结辅助头
        if self.stage_scheduler.freeze_aux(stage):
            for name, p in self.model.named_parameters():
                if "scene_head" in name or "behavior_head" in name or "rule_head" in name:
                    p.requires_grad = False

        total_loss = 0.0
        total_steps = 0
        epoch_metrics: Dict[str, float] = {
            "L0": 0.0, "L1": 0.0, "L2": 0.0, "L3": 0.0, "L_reg": 0.0,
            "L_total": 0.0, "grad_norm": 0.0, "lr": lr, "stage": float(stage),
        }

        for batch in dataloader:
            self.optimizer.zero_grad()
            # 前向
            y_a, y_s, y_b, y_r = self.model(batch)
            loss, metrics = self._compute_loss(
                y_a, y_s, y_b, y_r,
                batch.y_anomaly, batch.y_scene, batch.y_behavior, batch.y_rule,
                epoch=epoch,
            )
            # 反向
            loss.backward()
            # 梯度裁剪（4.48）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            # EMA 更新（4.49）
            self.ema.update()

            for k in ("L0", "L1", "L2", "L3", "L_reg", "L_total"):
                epoch_metrics[k] += metrics[k]
            epoch_metrics["grad_norm"] += float(grad_norm)
            total_steps += 1

        if total_steps > 0:
            for k in epoch_metrics:
                if k not in ("lr", "stage"):
                    epoch_metrics[k] /= total_steps
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估，计算 P / R / F1 / accuracy。

        Returns:
            metrics: 含 P, R, F1, accuracy, val_loss 的字典
        """
        self.model.eval()
        self.ema.apply_shadow()

        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        total_loss = 0.0
        total_steps = 0

        for batch in dataloader:
            y_a, y_s, y_b, y_r = self.model(batch)
            loss, _ = self._compute_loss(
                y_a, y_s, y_b, y_r,
                batch.y_anomaly, batch.y_scene, batch.y_behavior, batch.y_rule,
                epoch=100,
            )
            total_loss += loss.item()
            total_steps += 1

            preds = (y_a.squeeze(-1) > 0.5).long()  # [N]
            all_preds.append(preds)
            all_targets.append(batch.y_anomaly.long())

        self.ema.restore()

        if not all_preds:
            return {"P": 0.0, "R": 0.0, "F1": 0.0, "accuracy": 0.0,
                    "val_loss": total_loss, "best_f1": self.best_f1}

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
        val_loss = total_loss / max(total_steps, 1)

        # 早停
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
            "val_loss": val_loss,
            "best_f1": self.best_f1,
            "epochs_no_improve": float(self.epochs_no_improve),
        }

    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience
