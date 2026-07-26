# -*- coding: utf-8 -*-
"""
φ_loop 三阶段双向闭环反馈模块（§5.3，算法 5.2 + 5.3）

三阶段：
  Stage I（预训练弱监督，epoch 0..T_warm）：
      y_weak_t,i = max_v severity_i(v, t) →
      L_weak = γ_3(epoch) * BCE(y_rule, y_weak)              (5.2, 5.3)

  Stage II（训练中 GNN→规则置信度反馈）：
      s_i^- = mean(rule fires ∧ GNN p<0.3)                  (弱信号)
      s_i^0 = mean(rule fires ∧ gt_anomaly=0)                (误报)
      ℓ_i   = avg evidence chain length
      ε_i   = 0.2 / sqrt(ℓ_i) + 0.05 * s_i^0          (5.5)
      η_i^{e+1} = η_i^e * [1 - β·(s_i^- + ε_i)·(1-η_i)^+]   (5.4)

      Stabilization：η = 0.9·η_old + 0.1·η_new，|Δη|<5% → noop
      若 η_i < 0.3：阈值搜索 θ_i ∈ {θ_i-Δ, θ_i, θ_i+Δ} → argmax F1   (5.6)

  Stage III（推理时 GNN→规则模板）：
      当 y_anomaly > α_p 时，从 RGAT 注意力取 top-K 三元组 S_a(t)  (5.7)
      映射关系到 6 种规则模板，生成动态 SafetyViolation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Stage III 注意力→规则模板映射表（§5.3.4，附录 5-B）
ATTENTION_TEMPLATE_MAP: Dict[str, str] = {
    "ahead_of":           "following_too_close",
    "beside":             "lateral_safe_displacement",
    "nearby_pedestrian":  "pedestrian_proximity",
    "adjacent_lane":      "lane_change",
    "in_junction":        "junction_no_yield",
    "controlled_by":      "tl_no_yield",
}

# Scene 关系名映射到 K-HSTGAN edge_type 索引（与 stk/gnn/exporter.py:SCENE_REL_NAMES 一致）
_SCENE_REL_NAMES: List[str] = [
    "in_lane", "on_road", "in_junction", "adjacent_lane", "lane_connects",
    "ahead_of", "beside", "nearby_pedestrian", "controlled_by",
    "containsVehicle", "containsPedestrian", "containsTrafficLight",
    "containsRoad", "hasEnvironment",
]
_REL_NAME_TO_IDX = {n: i for i, n in enumerate(_SCENE_REL_NAMES)}


@dataclass
class RuleConfidenceState:
    """规则置信度状态向量（每条规则）。"""
    eta: torch.Tensor               # [14] 各规则置信度
    eta_ema: torch.Tensor           # [14] EMA 平滑后的 η
    threshold_search_count: torch.Tensor  # [14] 阈值搜索次数


class LoopFeedbackModule(nn.Module):
    """
    三阶段双向闭环反馈模块。
    """
    # === 超参数（§5.3.5 表 5-2）===
    beta: float = 0.001           # 置信度衰减率（论文 §5.3.3.2）
    eta_floor: float = 0.3        # η 下限，触发阈值搜索
    gamma_3_init: float = 0.5     # 弱监督初始权重
    T_warm: int = 10              # 弱监督预热周期
    top_k: int = 10               # 注意力 top-K
    alpha_anomaly: float = 0.5    # Stage III 异常判定阈值
    ema_decay: float = 0.9        # η EMA 平滑系数
    eta_min_delta: float = 0.05   # η 更新最小变化阈值
    p_low: float = 0.3            # GNN 弱信号阈值

    def __init__(self, num_rules: int = 14, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.num_rules = num_rules
        self.device = device
        # 初始化 η = 1.0（每条规则初始 100% 置信）
        self.state = RuleConfidenceState(
            eta=torch.ones(num_rules, dtype=torch.float32),
            eta_ema=torch.ones(num_rules, dtype=torch.float32),
            threshold_search_count=torch.zeros(num_rules, dtype=torch.long),
        )

    # ============================================================
    # Stage I：弱监督损失
    # ============================================================
    def compute_weak_loss(
        self,
        y_rule: torch.Tensor,        # [N, 14]
        y_weak: torch.Tensor,         # [N, 14]  预先按 max_v severity 算好
        epoch: int,
    ) -> torch.Tensor:
        """
        L_weak = γ_3(epoch) * BCE(y_rule, y_weak)              (5.2, 5.3)
        γ_3(epoch) = max(0, γ_3_init * (1 - epoch/T_warm))
        """
        gamma = max(0.0, self.gamma_3_init * (1.0 - float(epoch) / float(self.T_warm)))
        loss = F.binary_cross_entropy(y_rule, y_weak.float(), reduction="mean")
        return gamma * loss

    @staticmethod
    def compute_weak_labels(
        rule_out: Dict[str, Any],
        node_ids: List[str],
        num_rules: int = 14,
        rule_codes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        生成弱标签：y_weak_t,i = max_v severity_i(v, t)

        Args:
            rule_out: RuleEnforcer.enforce 返回 dict（violations 列表）
            node_ids: 节点 ID 列表（与 exporter 顺序一致）
            num_rules: 规则维度
            rule_codes: 14 维规则编码顺序（与 compute_kappa_rule 一致）

        Returns:
            y_weak: [N, 14] 取 max_v severity_i(v) ∈ [0, 1]
        """
        if rule_codes is None:
            rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                          "R10", "R11", "R13", "R16", "R17", "R18"]
        rule_idx = {r: i for i, r in enumerate(rule_codes)}
        out = torch.zeros(len(node_ids), num_rules, dtype=torch.float32)
        id2row = {nid: i for i, nid in enumerate(node_ids)}

        # 通配属性访问（兼容 SafetyViolation pydantic 模型将字段存于 attrs 子字典）
        def _attr(o, k, d=None):
            if o is None:
                return d
            if isinstance(o, dict):
                if o.get(k) is not None:
                    return o.get(k)
                attrs = o.get("attrs", {})
                if isinstance(attrs, dict):
                    return attrs.get(k, d)
                return d
            if hasattr(o, "model_dump"):
                dump = o.model_dump()
                if dump.get(k) is not None:
                    return dump.get(k)
                attrs = dump.get("attrs", {})
                if isinstance(attrs, dict):
                    return attrs.get(k, d)
                return d
            if hasattr(o, "__dict__"):
                if o.__dict__.get(k) is not None:
                    return o.__dict__.get(k)
                attrs = o.__dict__.get("attrs", {})
                if isinstance(attrs, dict):
                    return attrs.get(k, d)
                return d
            return getattr(o, k, d)

        for sv in rule_out.get("violations", []) or []:
            code = str(_attr(sv, "rule_code", "") or "")
            base = code.split("a")[0] if code.endswith("a") else code
            if base not in rule_idx:
                continue
            col = rule_idx[base]
            sev = float(_attr(sv, "severity", 0.0) or 0.0)
            for nid in (_attr(sv, "src_id"), _attr(sv, "dst_id")):
                nid = str(nid) if nid is not None else None
                if nid and nid in id2row:
                    row = id2row[nid]
                    out[row, col] = max(out[row, col].item(), sev)
        return out

    # ============================================================
    # Stage II：GNN→规则置信度反馈
    # ============================================================
    @torch.no_grad()
    def update_eta(
        self,
        s_minus: torch.Tensor,        # [14] rule fires ∧ GNN p<p_low 比例
        s_zero: torch.Tensor,         # [14] rule fires ∧ gt_anomaly=0 比例
        evidence_lengths: torch.Tensor,  # [14] 平均证据链长度（≥1）
    ) -> torch.Tensor:
        """
        η_i^{e+1} = η_i^e * [1 - β·(s_i^- + ε_i)·(1-η_i)^+]   (5.4, 5.5)
        ε_i       = 0.2 / sqrt(ℓ_i) + 0.05 · s_i^0              (5.5)
        Stabilization：η_new EMA = 0.9·η_old + 0.1·η_new
        |Δη| < 5% → noop

        Returns:
            eta_new: [14] 更新后 η
        """
        eps = 0.2 / torch.sqrt(evidence_lengths.clamp(min=1.0).float()) + 0.05 * s_zero
        eta_old = self.state.eta
        # (1 - η)^+ = max(0, 1-η)
        delta = self.beta * (s_minus + eps) * torch.clamp(1.0 - eta_old, min=0.0)
        eta_new = eta_old * (1.0 - delta)

        # 阈值搜索：η < floor 时计数++
        search_mask = eta_new < self.eta_floor
        self.state.threshold_search_count += search_mask.long()

        # EMA 平滑：η_ema = 0.9·η_ema + 0.1·η_new
        eta_ema_new = self.ema_decay * self.state.eta_ema + (1.0 - self.ema_decay) * eta_new

        # |Δη| < 5% → noop（保留 η_old）
        small_change = (eta_ema_new - self.state.eta_ema).abs() < self.eta_min_delta
        eta_ema_final = torch.where(small_change, self.state.eta_ema, eta_ema_new)

        self.state.eta = eta_new
        self.state.eta_ema = eta_ema_final
        return eta_new

    def get_eta(self) -> torch.Tensor:
        """返回当前 η（[14]）"""
        return self.state.ema_ema.detach() if False else self.state.eta_ema.detach()

    # ============================================================
    # Stage III：GNN→规则模板（推理时）
    # ============================================================
    @torch.no_grad()
    def generate_dynamic_rules(
        self,
        per_head_anomaly: torch.Tensor,   # [N, H]
        y_anomaly: torch.Tensor,           # [N, 1]
        rgat_attention: Dict[int, torch.Tensor],  # k → [H, E_k]
        node_ids: List[str],
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        当 y_anomaly > α_p 时，提取 top-K 注意力三元组 S_a(t) → 映射到规则模板。

        Args:
            per_head_anomaly: [N, H] 各头异常概率
            y_anomaly:        [N, 1] 融合后异常概率
            rgat_attention:   RGAT 注意力权重（关系类型 k → [H, E_k]）
            node_ids:         节点 ID 列表
            edge_index:       [2, E] 边索引（仅用于重排 attention 顺序）
            edge_type:        [E] 边类型

        Returns:
            List[Dict[rule_template, src_id, dst_id, attention_score]]
        """
        # 没有 attention 数据 → 直接返回
        if not rgat_attention or edge_index is None or edge_type is None:
            return []

        # 节点级异常 flag
        anomaly_mask = (y_anomaly.squeeze(-1) > self.alpha_anomaly)
        if anomaly_mask.sum() == 0:
            return []

        # 收集所有 (rel_name, src, dst, alpha)
        triples: List[Tuple[str, str, str, float]] = []
        for k, alpha in rgat_attention.items():
            if k >= len(_SCENE_REL_NAMES):
                continue
            rel_name = _SCENE_REL_NAMES[k]
            # alpha: [H, E_k] → 取多头均值 [E_k]
            alpha_mean = alpha.mean(dim=0).tolist()
            # 反查 src/dst：从原始 edge_index 中按 edge_type==k 选出
            mask_k = (edge_type == k)
            if mask_k.sum() == 0:
                continue
            src_k = edge_index[0][mask_k].tolist() if edge_index.dim() == 2 else []
            dst_k = edge_index[1][mask_k].tolist() if edge_index.dim() == 2 else []
            for s, d, a in zip(src_k, dst_k, alpha_mean):
                if s < len(node_ids) and d < len(node_ids):
                    triples.append((rel_name, node_ids[s], node_ids[d], float(a)))

        # top-K
        triples.sort(key=lambda t: t[3], reverse=True)
        top = triples[: self.top_k]

        # 映射到模板
        dynamic_rules: List[Dict[str, Any]] = []
        for rel, src, dst, score in top:
            template = ATTENTION_TEMPLATE_MAP.get(rel)
            if template is None:
                continue
            # 至少一端为异常节点
            src_anom = bool(anomaly_mask[node_ids.index(src)].item()) if src in node_ids else False
            dst_anom = bool(anomaly_mask[node_ids.index(dst)].item()) if dst in node_ids else False
            if not (src_anom or dst_anom):
                continue
            dynamic_rules.append({
                "rule_template": template,
                "relation_type": rel,
                "src_id": src,
                "dst_id": dst,
                "attention_score": score,
                "rule_layer": "Dynamic",
                "evidence": "GNN-attention top-K",
            })
        return dynamic_rules

    # ============================================================
    # 工具：统计 Stage II 输入
    # ============================================================
    @staticmethod
    @torch.no_grad()
    def compute_stage2_signals(
        y_rule: torch.Tensor,        # [N, 14] sigmoid
        y_anomaly: torch.Tensor,      # [N, 1] sigmoid
        gt_anomaly: torch.Tensor,      # [N] 0/1
        p_low: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 Stage II 反馈所需的两个比例：

          s_i^- = mean(rule fires ∧ GNN p < p_low)   [14]
          s_i^0 = mean(rule fires ∧ gt_anomaly = 0)   [14]

        Args:
            y_rule:    [N, 14]
            y_anomaly: [N, 1]
            gt_anomaly:[N] 0/1 真值
            p_low:     弱信号阈值

        Returns:
            s_minus: [14]
            s_zero:  [14]
        """
        rule_fires = (y_rule > 0.5).float()                  # [N, 14]
        p_low_mask = (y_anomaly.squeeze(-1) < p_low).float()  # [N]
        gt_zero_mask = (1.0 - gt_anomaly.float()).clamp(0.0, 1.0)  # [N]

        n = max(rule_fires.size(0), 1)
        s_minus = (rule_fires * p_low_mask.unsqueeze(-1)).sum(dim=0) / n
        s_zero = (rule_fires * gt_zero_mask.unsqueeze(-1)).sum(dim=0) / n
        return s_minus, s_zero

    @staticmethod
    @torch.no_grad()
    def compute_evidence_lengths(
        rule_out: Dict[str, Any],
        num_rules: int = 14,
        rule_codes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        计算每条规则的证据链平均长度 ℓ_i。简化实现：用每个规则的 violation 计数。
        """
        if rule_codes is None:
            rule_codes = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9",
                          "R10", "R11", "R13", "R16", "R17", "R18"]
        rule_idx = {r: i for i, r in enumerate(rule_codes)}

        counts = torch.ones(num_rules, dtype=torch.float32)  # 防 0
        for sv in rule_out.get("violations", []) or []:
            code = ""
            if isinstance(sv, dict):
                code = str(sv.get("rule_code", "") or sv.get("attrs", {}).get("rule_code", "") or "")
            elif hasattr(sv, "model_dump"):
                dump = sv.model_dump()
                code = str(dump.get("rule_code", "") or dump.get("attrs", {}).get("rule_code", "") or "")
            elif hasattr(sv, "__dict__"):
                code = str(sv.__dict__.get("rule_code", "") or sv.__dict__.get("attrs", {}).get("rule_code", "") or "")
            else:
                code = str(getattr(sv, "rule_code", "") or "")
            base = code.split("a")[0] if code.endswith("a") else code
            if base in rule_idx:
                counts[rule_idx[base]] += 1
        return counts
