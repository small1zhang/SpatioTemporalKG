# -*- coding: utf-8 -*-
"""
K-HSTGAN：Knowledge-guided Hierarchical Spatio-Temporal Graph Attention Network
(论文第 4 章实现)

模块组成:
  - exporter: STKG 数据导出（snapshot_store → torch_geometric.Data）
  - rgat: 关系感知图注意力层（§4.2, 式 4.4–4.11）
  - dhlstm_attn: 差分驱动层次化 LSTM-Attention（§4.3, 式 4.13–4.23）
  - knowledge_injector: RSS 残差 + 交规强度注入（§4.4, 式 4.24–4.28）
  - k_hstgan: 完整模型（§4.1–4.5）
  - trainer: 多任务训练器 + 三阶段调度（§4.5, 式 4.40–4.49）
"""

from .exporter import STKGGraphDataset, extract_stkg_tensors
from .rgat import RelationGAT
from .dhlstm_attn import DeltaGatedLSTM, BehaviorAttention, SceneTransformer
from .knowledge_injector import RSSResidualInjector, RuleStrengthEncoder
from .k_hstgan import K_HSTGAN
from .trainer import K_HSTGANTrainer

__all__ = [
    "STKGGraphDataset",
    "extract_stkg_tensors",
    "RelationGAT",
    "DeltaGatedLSTM",
    "BehaviorAttention",
    "SceneTransformer",
    "RSSResidualInjector",
    "RuleStrengthEncoder",
    "K_HSTGAN",
    "K_HSTGANTrainer",
]
