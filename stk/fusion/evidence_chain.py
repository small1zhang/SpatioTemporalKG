# -*- coding: utf-8 -*-
"""
KG 证据链回溯仲裁（§5.4.5，算法 5.4 后半）

当 D-S 融合结果中 K > τ_K 或 decision=="uncertain" 时触发回溯：

  Step 1（规则证据链）：
    遍历 SafetyViolation.supportedByEvidence 边 → 收集 P_v 节点集合
    evidence_strength = mean(sv.severity)

  Step 2（GNN 注意力子图）：
    从 RGAT 注意力权重提取 top-K (src, dst, rel_type, α)
    节点集 S_a.nodes = ∪{src, dst}

  Step 3（Jaccard 重叠度）：
    overlap = |P_v ∩ S_a| / (|P_v ∪ S_a| + ε)   (5.25)

  仲裁（算法 5.4）：
    K ≤ τ_K                  → trust D-S,   resolve_type="consistent"
    overlap > 0.5             → trust GNN,   resolve_type="trust_GNN"
    evidence_strength > 0.8  → trust rule,  resolve_type="trust_rule"
    else                     → needs review,resolve_type="needs_review"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .ds_fuser import FusionResult


# 与 stk/gnn/exporter.py:SCENE_REL_NAMES 一致
_SCENE_REL_NAMES: List[str] = [
    "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
    "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
    "containsVehicle", "containsPedestrian", "containsTrafficLight",
    "containsRoad", "hasEnvironment",
]


@dataclass
class FusionArbiterResult:
    """融合仲裁最终输出（§5.5 算法 5.5 输出）。"""
    y_fused: float                                  # 融合异常概率
    resolve_type: str                               # consistent / trust_GNN / trust_rule / needs_review
    explanation_path: str                           # 人类可读解释（Cypher-like 路径）
    rule_evidence_path: List[str]                   # KG 证据链节点 ID 列表
    gnn_attention_path: List[Tuple[str, str, str, float]]  # top-K 注意力三元组
    overlap: float                                  # Jaccard 重叠度
    evidence_strength: float                        # 规则证据强度
    K: float                                        # D-S 冲突系数
    decision: str                                   # D-S 原决策


class EvidenceChainArbiter(nn.Module):
    """
    KG 证据链回溯仲裁器（§5.4.5）。
    """
    # === 超参数（§5.4.5 + §5.4.4 表 5-3）===
    overlap_threshold: float = 0.5
    evidence_str_threshold: float = 0.8
    top_k: int = 10

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        evidence_str_threshold: float = 0.8,
        top_k: int = 10,
    ):
        super().__init__()
        self.overlap_threshold = overlap_threshold
        self.evidence_str_threshold = evidence_str_threshold
        self.top_k = top_k

    # ============================================================
    # Step 1：规则证据链
    # ============================================================
    @staticmethod
    def build_rule_evidence(
        rule_out: Dict[str, Any],
    ) -> Tuple[Set[str], float]:
        """
        遍历 SafetyViolation 列表，收集节点集合 P_v 和平均 severity。

        Args:
            rule_out: RuleEnforcer.enforce 返回的 dict
                      （含 violations 字段，每条 SafetyViolation 含 src_id/dst_id/severity）

        Returns:
            P_v: 节点 ID 集合
            evidence_strength: 平均 severity
        """
        nodes: Set[str] = set()
        sev_sum = 0.0
        n = 0
        for sv in rule_out.get("violations", []) or []:
            # 兼容 SafetyViolation pydantic 模型：字段在 attrs 子字典中
            if isinstance(sv, dict):
                src = sv.get("src_id"); dst = sv.get("dst_id"); sev = sv.get("severity", 0.0)
                if src is None:
                    src = sv.get("attrs", {}).get("src_id")
                if dst is None:
                    dst = sv.get("attrs", {}).get("dst_id")
                if sev == 0.0:
                    sev = sv.get("attrs", {}).get("severity", 0.0)
            elif hasattr(sv, "model_dump"):
                d = sv.model_dump()
                src = d.get("src_id"); dst = d.get("dst_id"); sev = d.get("severity", 0.0)
                if src is None:
                    src = d.get("attrs", {}).get("src_id")
                if dst is None:
                    dst = d.get("attrs", {}).get("dst_id")
                if sev == 0.0:
                    sev = d.get("attrs", {}).get("severity", 0.0)
            elif hasattr(sv, "__dict__"):
                d = sv.__dict__
                src = d.get("src_id"); dst = d.get("dst_id"); sev = d.get("severity", 0.0)
                if src is None:
                    src = d.get("attrs", {}).get("src_id")
                if dst is None:
                    dst = d.get("attrs", {}).get("dst_id")
                if sev == 0.0:
                    sev = d.get("attrs", {}).get("severity", 0.0)
            else:
                src = getattr(sv, "src_id", None); dst = getattr(sv, "dst_id", None)
                sev = getattr(sv, "severity", 0.0)
            if src:
                nodes.add(str(src))
            if dst:
                nodes.add(str(dst))
            sev_sum += float(sev or 0.0)
            n += 1
        evidence_strength = sev_sum / n if n > 0 else 0.0
        return nodes, evidence_strength

    # ============================================================
    # Step 2：GNN 注意力子图
    # ============================================================
    @torch.no_grad()
    def build_attention_subgraph(
        self,
        rgat_attention: Dict[int, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_ids: List[str],
    ) -> List[Tuple[str, str, str, float]]:
        """
        从 RGAT 注意力权重提取 top-K 三元组 S_a(t)。

        Returns:
            List of (src_id, dst_id, rel_name, alpha)
        """
        if not rgat_attention or edge_index is None:
            return []

        # 收集所有 (rel_name, src, dst, alpha)
        triples: List[Tuple[str, str, str, float]] = []
        for k, alpha in rgat_attention.items():
            if k >= len(_SCENE_REL_NAMES):
                continue
            rel = _SCENE_REL_NAMES[k]
            # alpha: [H, E_k]
            if alpha.dim() >= 2:
                alpha_mean = alpha.mean(dim=0)  # [E_k]
            else:
                alpha_mean = alpha
            alpha_list = alpha_mean.tolist()
            mask_k = (edge_type == k)
            if mask_k.sum() == 0:
                continue
            src_k = edge_index[0][mask_k].tolist() if edge_index.dim() == 2 else []
            dst_k = edge_index[1][mask_k].tolist() if edge_index.dim() == 2 else []
            for s, d, a in zip(src_k, dst_k, alpha_list):
                if 0 <= s < len(node_ids) and 0 <= d < len(node_ids):
                    triples.append((node_ids[s], node_ids[d], rel, float(a)))

        triples.sort(key=lambda t: t[3], reverse=True)
        return triples[: self.top_k]

    @staticmethod
    def jaccard(P: Set[str], S: Set[str]) -> float:
        """Jaccard 重叠度（5.25）"""
        if not P and not S:
            return 0.0
        union = P | S
        inter = P & S
        return len(inter) / (len(union) + 1e-8)

    # ============================================================
    # 主入口
    # ============================================================
    def forward(
        self,
        fusion_result: FusionResult,
        rule_out: Dict[str, Any],
        rgat_attention: Dict[int, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_ids: List[str],
        p_anomaly: Optional[torch.Tensor] = None,
    ) -> FusionArbiterResult:
        """
        仲裁总入口（§5.4.5 算法 5.4）。

        Args:
            fusion_result: D-S 融合输出（DempsterShaferFuser.forward 的返回）
            rule_out:     RuleEnforcer.enforce 输出
            rgat_attention: K_HSTGAN extras["rgat_attention"]
            edge_index:   K_HSTGAN extras["edge_index"]
            edge_type:    K_HSTGAN extras["edge_type"]
            node_ids:     节点 ID 列表（与 exporter 一致）
            p_anomaly:    [N, 1] 可选：用于 trust_GNN 时取 y_fused

        Returns:
            FusionArbiterResult：含 y_fused / resolve_type / explanation_path
        """
        # Step 1: 规则证据链
        P_v, evidence_strength = self.build_rule_evidence(rule_out)

        # Step 2: GNN 注意力子图
        S_a_triples = self.build_attention_subgraph(
            rgat_attention, edge_index, edge_type, node_ids)
        S_v: Set[str] = set()
        for s, d, _, _ in S_a_triples:
            S_v.add(s); S_v.add(d)

        # Step 3: 重叠度
        overlap = self.jaccard(P_v, S_v)

        # 仲裁逻辑
        if fusion_result.is_consistent:
            resolve_type = "consistent"
            y_fused = fusion_result.m_fused_anomaly
        elif overlap > self.overlap_threshold:
            resolve_type = "trust_GNN"
            # 取 GNN 输出 p_anomaly 的均值
            if p_anomaly is not None and p_anomaly.numel() > 0:
                y_fused = float(p_anomaly.mean().item())
            else:
                y_fused = fusion_result.m_fused_anomaly
        elif evidence_strength > self.evidence_str_threshold:
            resolve_type = "trust_rule"
            y_fused = fusion_result.m_fused_anomaly
        else:
            resolve_type = "needs_review"
            y_fused = fusion_result.m_fused_anomaly

        # 构造人类可读解释（Cypher-flavored 字符串）
        explanation = self._build_explanation(
            P_v, S_a_triples, evidence_strength, overlap, fusion_result, resolve_type)

        return FusionArbiterResult(
            y_fused=y_fused,
            resolve_type=resolve_type,
            explanation_path=explanation,
            rule_evidence_path=list(P_v),
            gnn_attention_path=S_a_triples,
            overlap=overlap,
            evidence_strength=evidence_strength,
            K=fusion_result.K,
            decision=fusion_result.decision,
        )

    def _build_explanation(
        self,
        P_v: Set[str],
        S_a: List[Tuple[str, str, str, float]],
        evidence_strength: float,
        overlap: float,
        fusion_result: FusionResult,
        resolve_type: str,
    ) -> str:
        """生成 Cypher-flavored 的人类可读解释路径。"""
        path = []
        for s, d, rel, score in S_a[:3]:
            path.append(f"({s})-[:{rel} α={score:.2f}]->({d})")
        if not path:
            path.append("<no-attention-triples>")
        evidence_summary = (
            f"RuleEvidence(nodes={list(P_v)[:3]}, severity_mean="
            f"{evidence_strength:.2f})")
        explanation = (
            f"K={fusion_result.K:.3f} Δ-S:{fusion_result.decision}"
            f"|overlap={overlap:.2f}|resolve={resolve_type}|{evidence_summary}|"
            "|".join(path)
        )
        return explanation
