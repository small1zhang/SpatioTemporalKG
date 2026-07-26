# -*- coding: utf-8 -*-
"""
KS-NBCF 融合框架（论文第 5 章）

模块组成:
  - feat_injection.py   φ_feat 编排层（§5.2，算法 5.1）
  - loop_feedback.py    φ_loop 三阶段闭环（§5.3，算法 5.2 + 5.3）
  - ds_fuser.py         φ_fuse D-S 证据理论融合核心（§5.4，算法 5.4）
  - evidence_chain.py   KG 证据链回溯仲裁（§5.4.5）
"""
from .feat_injection import KS_NBCF_FeatInjection
from .loop_feedback import LoopFeedbackModule
from .ds_fuser import DempsterShaferFuser, FusionResult
from .evidence_chain import EvidenceChainArbiter, FusionArbiterResult

__all__ = [
    "KS_NBCF_FeatInjection",
    "LoopFeedbackModule",
    "DempsterShaferFuser",
    "FusionResult",
    "EvidenceChainArbiter",
    "FusionArbiterResult",
]
