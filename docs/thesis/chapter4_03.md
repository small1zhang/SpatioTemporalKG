# 4.3 时序编码层：差分驱动的层次化 LSTM-Attention

## 4.3.1 设计动机

4.2 节 RGAT 仅编码单帧的瞬时空间关系。自动驾驶安全异常往往跨多帧演化：跟车从安全到危险是一个长度约 1-2 秒（20-40 帧）的过程，超车涉及速度差累积和横向位移达 3-5 秒（60-100 帧）。单帧图注意力无法捕获此类"时间轴上的减速积累/横向漂移"，时序建模不可或缺。

第 3.6 节的动态更新机制为时序建模带来一个独特资源：**差分图 $\Delta g_t$**。$\Delta g_t$ 天然标注了帧间变化量——当 $\Delta g_t$ 为空时，意味着帧间无变化，此时时序模型可"跳帧"而不损失信息；当 $\Delta g_t$ 携带属性变化或实体进出时，时序模型应充分响应该帧变化。现有时序模型（LSTM、GRU、Transformer）均匀消费每一帧，不利用这一稀疏先验。

基于此，本节提出**差分驱动的层次化 LSTM-Attention**（Delta-gated Hierarchical LSTM-Attention，简称 DHLSTM-Attn）。核心设计为三层结构：帧级 LSTM（处理单帧节点特征与 $\Delta g_t$ 门控）→ 行为级注意力（基于对应行为窗口做跨帧聚焦）→ 场景级自注意力（跨越全部时间步做相似性建模），层次递进、分辨率由细到粗。

## 4.3.2 帧级差分门控 LSTM

### 4.3.2.1 基础 LSTM 单元

设 $[ \mathbf{h}_t^{\text{spatial}} \in \mathbb{R}^{N \times F'}]_{t=1}^T$ 为 RGAT 在 $T$ 帧窗口内输出的空间嵌入序列。标准 LSTM 逐帧处理序列：

$$
\mathbf{c}_t,\ \mathbf{h}_t^{\text{LSTM}}\ =\ \mathrm{LSTM}_{\theta}\!\left(\ \mathbf{h}_t^{\text{spatial}},\ \mathbf{c}_{t-1},\ \mathbf{h}_{t-1}^{\text{LSTM}}\ \right)
\tag{4.13}
$$

其中 $\mathbf{c}_t$ 与 $\mathbf{h}_t^{\text{LSTM}}$ 分别为 LSTM 单元的状态向量与隐藏向量，$\theta$ 为 LSTM 参数。

### 4.3.2.2 差分门控机制

差分门控的核心思想：$\Delta g_t$ 承载的帧间变化量越大，LSTM 应该对此帧投入更多注意力容量；$\Delta g_t$ 为空或极小时，LSTM 应维持上一帧状态，避免冗余计算。

将 $\Delta g_t$ 编码为门控向量 $\mathbf{\delta}_t$：

$$
\mathbf{\delta}_t\ =\ \mathrm{MLP}_{\delta}\!\left(\ \big[\ \text{sum}(\Delta_{\mathcal{E}}.added),\ \text{sum}(\Delta_{\mathcal{E}}.removed),\ \|\Delta_{\mathcal{A}}\|_F,\ \text{sum}(\Delta_{\mathcal{R}}.added)\ \big]\ \right)
\tag{4.14}
$$

其中 $\|\Delta_{\mathcal{A}}\|_F$ 是属性差分的 Frobenius 范数，四维输入拼接后由单层 MLP 映射至 $F'$ 维。

门控与 LSTM 状态更新整合为：

$$
\mathbf{g}_t^{\text{in}}\ =\ \sigma\!\left(\ \mathbf{W}_{\text{in}}\,[\,\mathbf{h}_t^{\text{spatial}},\ \mathbf{\delta}_t\,]\ +\ \mathbf{b}_{\text{in}}\ \right) \odot \text{sigmoid}(\mathbf{W}_{\text{gate}}\,\mathbf{\delta}_t)
\tag{4.15}
$$

$$
\mathbf{c}_t,\ \mathbf{h}_t^{\text{LSTM}}\ =\ \mathrm{LSTM}_{\theta}\!\left(\ \mathbf{g}_t^{\text{in}},\ \mathbf{c}_{t-1},\ \mathbf{h}_{t-1}^{\text{LSTM}}\ \right)
\tag{4.16}
$$

关键在公式 (4.15) 中 $\text{sigmoid}(\mathbf{W}_{\text{gate}}\,\mathbf{\delta}_t)$ 项：当 $\mathbf{\delta}_t \approx \mathbf{0}$（帧间无变化），该 sigmoid 输出接近 $\mathbf{0}$，门控输入趋近 $\mathbf{0}$，LSTM 输入权重被压制，状态几乎不更新 → 等价于"跳过该帧"。当 $\mathbf{\delta}_t$ 非零（有实体/属性/关系变化），sigmoid 输出接近 $\mathbf{1}$，LSTM 正常处理该帧。

该门控相比标准 LSTM 避免约 30%–50% 的冗余帧前向计算（来自 $\Delta g_t$ 的经验稀疏率）。实验定量分析见第 6 章 RQ2.4 消融实验。

## 4.3.3 行为级注意力

LSTM 输出 $\{\mathbf{h}_1^{\text{LSTM}}, \dots, \mathbf{h}_T^{\text{LSTM}}\}$ 已编码了每帧的时序上下文。但 STKG 行为层中 InteractionEvent 往往跨多帧持续，有必要在行为持续窗口内做**聚焦**——例如一次超车行为持续 20 帧，其关键帧是"超车开始"（src 变道切入）和"超车结束"（dst 变道切出），而非其中间匀速推进的 18 帧。

行为级注意力利用 InteractionEvent 节点的 `frame_start` 和 `frame_end` 自动构建行为窗口：

$$
\mathcal{W}_b = [t_{\text{start}}^{(b)},\ t_{\text{end}}^{(b)}],\quad b = 1, \dots, B
\tag{4.17}
$$

对每个行为窗口 $\mathcal{W}_b$：

$$
\mathbf{\alpha}_b^{\text{beh}}\ =\ \text{softmax}_{t \in \mathcal{W}_b}\!\left(\ \mathbf{a}_{\text{beh}}^{\top}\,\tanh\!\big(\ \mathbf{W}_{\text{beh}}\,\mathbf{h}_t^{\text{LSTM}} + \mathbf{b}_{\text{beh}}\ \big)\ \right)
\tag{4.18}
$$

$$
\mathbf{h}_b^{\text{behavior}}\ =\ \sum_{t \in \mathcal{W}_b} \alpha_{b,t}^{\text{beh}}\ \mathbf{h}_t^{\text{LSTM}}
\tag{4.19}
$$

$B$ 为当前窗口内全部行为事件数。当当前窗口无行为事件（$B = 0$）时，行为级注意力退化为对全部 $T$ 帧的均匀注意力。

行为级注意力的另一设计细节：若行为窗口 $\mathcal{W}_b$ 跨出当前 $T$ 帧滑动窗口边界，则仅取落在窗口内的部分参与计算，边界外的帧被丢弃——这一截断对模型推理延迟无影响，因为 $T$ 帧窗口以步长 $T$ 滑动覆盖整个时间轴。

## 4.3.4 场景级自注意力

帧级 LSTM 的输出 $\mathbf{h}_t^{\text{LSTM}}$ 与行为级注意力输出 $\mathbf{h}_b^{\text{behavior}}$ 在上述两层处理后，段落向量加总后输入场景级 Transformer 自注意力层：

$$
\mathbf{H}_{\text{seq}}\ =\ \text{Concat}\!\big(\ \mathbf{h}_1^{\text{LSTM}},\ \dots,\ \mathbf{h}_T^{\text{LSTM}},\ \mathbf{h}_1^{\text{behavior}},\ \dots,\ \mathbf{h}_B^{\text{behavior}}\ \big)
\tag{4.20}
$$

$$
\mathbf{Q},\ \mathbf{K},\ \mathbf{V}\ =\ \mathbf{H}_{\text{seq}}\mathbf{W}_Q,\ \mathbf{H}_{\text{seq}}\mathbf{W}_K,\ \mathbf{H}_{\text{seq}}\mathbf{W}_V
\tag{4.21}
$$

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\ =\ \text{softmax}\!\left(\ \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\ \right) \mathbf{V}
\tag{4.22}
$$

场景级注意力的参数规模并未显著增加（$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 维度为 $F' \times d_k$，默认 $d_k = 32$），但由于涉及 $\mathbf{H}_{\text{seq}}$ 的全体自注意力，其复杂度为 $\mathcal{O}((T+B)^2 d_k)$。典型情况下 $T=30$，$B \leq 5$，即约 $35 \times 35 = 1225$ 个注意力位置，单帧开销完全可接受。

场景级自注意力的输出 $\mathbf{H}_{\text{transformer}} \in \mathbb{R}^{(T+B) \times d_k}$ 中仅取其"最后一帧"行（即第 $T$ 帧的 $\mathbf{h}_T^{\text{LSTM}}$ 对应的输出）作为该窗口的时序特征摘要 $\mathbf{h}^{\text{temporal}}$：

$$
\mathbf{h}^{\text{temporal}}\ =\ \mathbf{H}_{\text{transformer}}[T-1,:]
\tag{4.23}
$$

这是 4.5 节融合层的直接输入之一。

## 4.3.5 层次结构总览

图 4-2 展示了层次化时序编码的三个层次与数据流向。

```
帧序列 t=1,...,T      行为窗口 W_1...W_B     帧+行为序列
     │                      │                     │
     ▼                      │                     │
 ┌─────────┐                │                     │
 │ 帧级 LSTM│ ── h_t^LSTM ──┤                     │
 │ (差分门控) │               │                     │
 └─────────┘                │                     │
     │                      ▼                     │
     │               ┌──────────────┐              │
     │               │ 行为级注意力  │              │
     │               │ h_b^behavior  │              │
     │               └──────────────┘              │
     │                      │                     │
     └──────────────────────┼─────────────────────┘
                            ▼
                    ┌────────────────────────┐
                    │ 场景级自注意力          │
                    │ (Transformer)          │
                    │ h^temporal              │
                    └────────────────────────┘
                            │
                            ▼
                     → 4.5 融合层
```

**图 4-2** 层次化时序编码三层结构：帧级 LSTM（含差分门控）→ 行为级注意力 → 场景级 Transformer

## 4.3.6 复杂度分析

设窗口长度 $T$，行为事件数 $B$，帧特征维 $F'$，注意力头 $H$。

| 组件 | 单帧/窗口复杂度 | 典型值 |
|------|--------------|--------|
| 帧级 LSTM | $\mathcal{O}(F'^2)$ | $64^2 = 4096$ |
| 差分门控 | $\mathcal{O}(F')$ | 64 |
| 行为级注意力 | $\mathcal{O}(B \cdot T \cdot F')$ | $5 \times 30 \times 64 = 9600$ |
| 场景级 Transformer | $\mathcal{O}((T+B)^2 \cdot d_k)$ | $35^2 \times 32 = 39200$ |
| **总计** | **约 ~52000 参数/帧** | — |

总复杂度低于典型单层 Transformer 的 $\mathcal{O}(N^2 d)$（$N \gg 35$ 的情况），属于轻量级。且差分门控可跳过约 30-50% 的冗余帧，实际计算量可能等效为 $\mathcal{O}(0.6 \times 52000) \approx 31200$ 参数/帧。

## 4.3.7 与标准 LSTM 的对比

表 4-2 总结了 DHLSTM-Attn 与标准 LSTM 在时序编码能力上的差异。

**表 4-2** DHLSTM-Attn 与标准 LSTM 对比

| 维度 | 标准 LSTM | DHLSTM-Attn |
|------|---------|-------------|
| 帧间变化利用 | 无 | 差分门控 $\delta_t \to$ 跳过无变化帧 |
| 行为窗口感知 | 无 | 行为级注意力聚焦 InteractionEvent 窗口 |
| 帧间相似性建模 | 单方向链式传播 | Transformer 双向自注意力 |
| 帧信息浪费比 | 0%（均匀处理每帧）| 跳过 30-50% 无用帧 |
| 行为时间感知 | 全部帧同等对待 | 行为窗口内做跨帧聚焦 |

## 4.3.8 小结

本节设计了 K-HSTGAN 的时序编码层 DHLSTM-Attn：三层层次化结构，差分门控提升了计算效率，行为级注意力注入 STKG InteractionEvent 窗口先验，场景级 Transformer 增强了长距离帧间交互。输出 $\mathbf{h}^{\text{temporal}}$ 传递到 4.5 节融合层与 4.4 节知识注入层的输出合并，完成异常预测。

下一节将介绍模型如何将 RSS 公式与交规规则编码为向量知识，注入到时序与空间特征的决策路径中。