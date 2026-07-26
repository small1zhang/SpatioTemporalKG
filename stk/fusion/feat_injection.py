# -*- coding: utf-8 -*-
"""
φ_feat 特征注入编排层（§5.2，算法 5.1）

策略 I（RSS 残差拼接）：x_aug = [x ‖ LayerNorm(kappa_rss)]  → dim 23
策略 II（规则强度残差注入）：h_spatial += MLP(kappa_rule)  → 不改 dim

注意：策略 I 和策略 II 的实际计算已由 K-HSTGAN 内部的
RSSResidualInjector 和 RuleStrengthEncoder 完成（见 stk/gnn/knowledge_injector.py）。
本模块作为 KS-NBCF 的顶层编排入口，负责：
  1. 从 snapshot dict 预处理 kappa_rss / kappa_rule 张量
  2. 调用 K_HSTGAN(data, return_extras=True)
  3. 返回统一的 (y_*, extras) 三元组，供下游 φ_loop / φ_fuse 消费
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data

from stk.gnn.k_hstgan import K_HSTGAN
from stk.gnn.exporter import extract_stkg_tensors


class KS_NBCF_FeatInjection:
    """
    KS-NBCF 特征注入编排层（§5.2）。

    本层不包含可学习参数，仅作为数据转换 + K-HSTGAN 推理的胶水。
    """

    def __init__(self, model: K_HSTGAN):
        """
        Args:
            model: 已加载权重的 K_HSTGAN 模型实例
        """
        self.model = model

    @torch.no_grad()
    def predict_with_extras(
        self,
        snapshot: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        ego_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        对单帧 snapshot 运行特征注入 + K-HSTGAN，返回扩展输出。

        Args:
            snapshot: PipelineOrchestrator.snapshot_store.get(frame_id) 返回的字典
            device:   推理设备
            ego_id:   可选 ego vehicle ID

        Returns:
            y_anomaly:  [N, 1]
            y_scene:    [N, 3]
            y_behavior: [N, 7]
            y_rule:     [N, 14]
            extras:     dict（per_head_anomaly, rgat_attention, h_spatial, h_temporal, ...）
        """
        data = extract_stkg_tensors(snapshot, ego_id=ego_id)
        data = data.to(device)
        y_a, y_s, y_b, y_r, extras = self.model(data, return_extras=True)
        return y_a, y_s, y_b, y_r, extras

    def predict_with_grad(
        self,
        data: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        对已有 Data 对象前向传播（训练用，保留梯度）。

        Args:
            data: torch_geometric.Data（由 STKGGraphDataset 或 extract_stkg_tensors 生成）

        Returns:
            (y_anomaly, y_scene, y_behavior, y_rule, extras)
        """
        y_a, y_s, y_b, y_r, extras = self.model(data, return_extras=True)
        return y_a, y_s, y_b, y_r, extras
