# -*- coding: utf-8 -*-
"""
差分驱动层次化 LSTM-Attention（§4.3, 式 4.13–4.23）

三级层次结构：
  1. DeltaGatedLSTM   — 帧级差分门控 LSTM（4.13–4.16）
  2. BehaviorAttention — 行为级注意力聚合（4.17–4.19）
  3. SceneTransformer — 场景级自注意力 Transformer（4.20–4.23）

时序窗口长度 T = 30 帧（默认），每帧对应一个 h_i^spatial → 经 LSTM 编码后，
行为窗口做 attention 聚合，最终经 Transformer 得到 h^temporal。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaGatedLSTM(nn.Module):
    """
    帧级差分门控 LSTM（4.13–4.16）。

    delta_t = MLP_δ([|ΔE.added|, |ΔE.removed|, ||ΔA||_F, |ΔR.added|])   (4.14)
    g_t^in  = σ(W_in[h_t^spatial, delta_t]) · sigmoid(W_gate · delta_t)  (4.15)
    c_t, h_t = LSTM(g_t^in, c_{t-1}, h_{t-1})                           (4.16)

    当 delta_t 接近零向量时，sigmoid(W_gate · delta_t) → 0，LSTM 输入被抑制，
    实现"跳帧"效果，跳过变化极小的冗余帧。

    Args:
        hidden_dim:   LSTM 隐藏维度 F'（默认 64）
        input_dim:    输入特征维度（= in_features，如 23）
        delta_input_dim: delta_t 的输入维度（固定 4）
        num_layers:   LSTM 层数
        dropout:      LSTM 层间 dropout
    """

    def __init__(self, hidden_dim: int = 64, input_dim: int = 23,
                 delta_input_dim: int = 4, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Δt MLP：4 → F'（输出作为 LSTM 输入的 gate）
        self.delta_mlp = nn.Sequential(
            nn.Linear(delta_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 输入门 W_in：[F', F' + F']
        self.W_in = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Gate：W_gate：[F' → 1] → sigmoid
        self.W_gate = nn.Linear(hidden_dim, 1)
        # 标准 LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        h_spatial: torch.Tensor,
        delta_feat: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_spatial:  [B, T, F'] 或 [T, F']  空间编码序列
            delta_feat: [B, T, 4]  或 [T, 4]   Δg_t 特征
            h_prev:     [1, B, F'] 或 [B, F']   初始 LSTM 隐状态
            c_prev:     [1, B, F'] 或 [B, F']   初始 LSTM cell 状态

        Returns:
            h_out:  [B, T, F'] 或 [T, F']  LSTM 输出序列
            c_out:  最终 cell 状态
        """
        # 归一化到 3D
        if h_spatial.dim() == 2:
            h_spatial = h_spatial.unsqueeze(0)  # [1, T, F']
            delta_feat = delta_feat.unsqueeze(0)
        B, T, _ = h_spatial.shape

        # Δt 编码：[B, T, F']
        delta_t = self.delta_mlp(delta_feat)

        # LSTM 逐步计算（含门控）
        h_out_list: list[torch.Tensor] = []
        if h_prev is not None and h_prev.dim() == 2:
            h_prev = h_prev.unsqueeze(0)  # [1, B, F']
            c_prev = c_prev.unsqueeze(0)
        h_cur = h_prev if h_prev is not None else torch.zeros(1, B, self.hidden_dim, device=h_spatial.device)
        c_cur = c_prev if c_prev is not None else torch.zeros(1, B, self.hidden_dim, device=h_spatial.device)

        for t in range(T):
            h_t = h_spatial[:, t, :]       # [B, F']
            d_t = delta_t[:, t, :]         # [B, F']
            # W_in 输入：拼接 h_t^spatial 与 delta_t → gate
            gate_in = torch.sigmoid(self.W_gate(d_t))  # [B, 1]
            g_in = torch.sigmoid(self.W_in(
                torch.cat([h_t, d_t], dim=-1)
            ))  # [B, F']
            g_in = g_in * gate_in  # 差分门控：变化小时抑制

            # LSTM step
            out, (h_cur, c_cur) = self.lstm(g_in.unsqueeze(1), (h_cur, c_cur))
            h_out_list.append(out.squeeze(1))  # [B, F']

        h_out = torch.stack(h_out_list, dim=1)  # [B, T, F']
        return h_out, c_cur


class BehaviorAttention(nn.Module):
    """
    行为级注意力聚合（4.17–4.19）。

    对于每个行为窗口 W_b，计算注意力权重并聚合窗口内 LSTM 帧：
        α_b^beh = softmax_{t∈W_b}(a^T tanh(W h_t^LSTM))    (4.18)
        h_b^behavior = Σ_t α_b,t^beh · h_t^LSTM             (4.19)

    当行为窗口 B=0 时（无行为事件），退化为对所有 T 帧的均匀注意力。

    Args:
        hidden_dim: F'（LSTM 隐藏维度）
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        h_lstm: torch.Tensor,
        behavior_windows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_lstm:          [B, T, F']  LSTM 输出序列
            behavior_windows: [B, T] bool/tensor — 哪些帧属于某个行为窗口
                             若为 None，则使用均匀注意力。

        Returns:
            h_behavior: [B, F']  行为级聚合向量
        """
        B, T, Fd = h_lstm.shape
        device = h_lstm.device

        # 计算注意力得分：a^T tanh(W h_t)
        score = self.attn_a(torch.tanh(self.attn_w(h_lstm)))  # [B, T, 1]
        score = score.squeeze(-1)  # [B, T]

        if behavior_windows is not None:
            # 用 mask 将非行为窗口帧设为 -inf
            score = score.masked_fill(~behavior_windows.bool(), float("-inf"))

        alpha = F.softmax(score, dim=-1)  # [B, T]

        # 逐 B 聚合
        h_behavior = torch.einsum("bt, btf -> bf", alpha, h_lstm)  # [B, Fd]
        return h_behavior


class SceneTransformer(nn.Module):
    """
    场景级自注意力 Transformer（4.20–4.23）。

    H_seq = Concat(h_1^LSTM..h_T^LSTM, h_1^behavior..h_B^behavior)   (4.20)
    h^temporal = LayerNorm(ScaledDotProduct(H_seq))[T-1, :]           (4.23)

    Args:
        hidden_dim: F'（默认 64）
        d_k:        Transformer key 维度（默认 32）
        num_heads:  Transformer 多头数（默认 4）
        dropout:    dropout 率
    """

    def __init__(self, hidden_dim: int = 64, d_k: int = 32,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.num_heads = num_heads

        # Q, K, V 投影
        self.W_Q = nn.Linear(hidden_dim, num_heads * d_k, bias=False)
        self.W_K = nn.Linear(hidden_dim, num_heads * d_k, bias=False)
        self.W_V = nn.Linear(hidden_dim, num_heads * d_k, bias=False)
        self.W_out = nn.Linear(num_heads * d_k, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = d_k ** 0.5

    def forward(
        self,
        h_lstm: torch.Tensor,
        h_behavior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_lstm:      [B, T, F']  LSTM 输出序列
            h_behavior:  [B, B_seq, F']  可选的行为聚合序列（通常 B_seq ≤ 5）

        Returns:
            h_temporal:  [B, F']  最后一帧的 Transformer 输出（取 T-1 位置）
        """
        B, T, Fd = h_lstm.shape
        device = h_lstm.device

        # 拼接 LSTM 序列与行为序列 → [B, T + B_seq, Fd]
        if h_behavior is not None and h_behavior.dim() == 3:
            # h_behavior: [B, B_seq, Fd]
            H_seq = torch.cat([h_lstm, h_behavior], dim=1)
        elif h_behavior is not None and h_behavior.dim() == 2:
            # [B, Fd] → [B, 1, Fd]
            H_seq = torch.cat([h_lstm, h_behavior.unsqueeze(1)], dim=1)
        else:
            H_seq = h_lstm
        L = H_seq.size(1)  # 序列总长度

        # Multi-head self-attention
        Q = self.W_Q(H_seq).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.W_K(H_seq).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(H_seq).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B, H, L, d_k]
        context = context.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_k)
        out = self.W_out(context)  # [B, L, Fd]

        # 残差连接 + LayerNorm
        out = self.layer_norm(out + H_seq)

        # 提取最后一帧（T-1 位置）
        h_temporal = out[:, -1, :]  # [B, Fd]
        return h_temporal
