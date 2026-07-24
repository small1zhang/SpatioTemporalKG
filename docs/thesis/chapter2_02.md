# 2.2 图神经网络、循环网络与自注意力机制

## 2.2.1 图神经网络的基本形式

图神经网络（Graph Neural Network, GNN）是一类直接在图结构数据上运行的神经网络模型，其核心思想是通过邻里聚合（neighbor aggregation）在图上传播节点表示。给定图 $G = (\mathcal{V}, \mathcal{E})$ 与节点初始特征矩阵 $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times F}$，GNN 在每层 $l$ 上将每个节点 $v \in \mathcal{V}$ 的隐表示 $\mathbf{h}_v^{(l)}$ 按下式更新：

\$$
\mathbf{h}_v^{(l+1)} = \sigma\left( \mathbf{W}^{(l)} \mathbf{h}_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathbf{W}_{\text{agg}}^{(l)} \mathbf{h}_u^{(l)} \right)
\tag{2.1}
\$$

其中 $\mathcal{N}(v)$ 为 $v$ 的邻居集合，$\mathbf{W}^{(l)}, \mathbf{W}_{\text{agg}}^{(l)}$ 为可训练参数，$\sigma$ 为非线性激活函数。文献 [Bruce, AAAI 2017] 给出了 GNN 在图分类与节点分类任务上的最早形式化定义。GNN 的核心共性在于：（i）输入为图，输出仍是图（每层之后图结构不变，仅节点表示更新）；（ii）参数在所有节点之间共享，与图大小无关。

## 2.2.2 图卷积网络

Kipf 与 Welling 在 ICLR 2017 上提出图卷积网络（Graph Convolutional Network, GCN），将图卷积定义为归一化邻接矩阵上的线性变换：

\$$
\mathbf{H} = \sigma\left( \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{X} \mathbf{W} \right)
\tag{2.2}
\$$

其中 $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ 为加自环的邻接矩阵，$\tilde{\mathbf{D}}$ 为对应的度矩阵。该公式是谱图卷积的一阶近似，将每个节点的更新表达为"自身与邻居的归一化加权和"的线性变换。

GCN 的局限在于：对所有邻居节点采用相同的归一化权重，无法体现邻居之间的相对重要性差异。这一局限催生了图注意力网络。

## 2.2.3 图注意力网络

Veličković 等人在 ICLR 2018 上提出图注意力网络（Graph Attention Network, GAT），通过注意力机制让每个节点自适应地分配邻居权重：

\$$
\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_i \| \mathbf{W} \mathbf{h}_j] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_i \| \mathbf{W} \mathbf{h}_k] \right) \right)}
\tag{2.3}
\$$

\$$
\mathbf{h}_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right)
\tag{2.4}
\$$

其中 $\mathbf{W}$ 为共享权重矩阵，$\mathbf{a}$ 为注意力向量，$\sigma$ 为激活函数。GAT 的关键性质是：邻居权重 $\alpha_{ij}$ 由节点对 $(i, j)$ 的特征计算得到，对每个目标节点 $i$ 跨邻居做 softmax 归一化。这与 GCN 中固定归一化权重不同，使得 GAT 可以表达"邻居之间相对重要性不同"的语义。

GAT 的扩展包括多头注意力 GAT（Multi-Head GAT）：使用 $K$ 组独立注意力头 $\{(\mathbf{W}_k, \mathbf{a}_k)\}_{k=1}^K$，将 $K$ 头输出按拼接或平均合并。本论文第 4 章 §4.2 RGAT 在 GAT 的基础上引入"关系感知"机制，为 STKG 的 15 种场景关系各自分配独立注意力通道，并引入可学习关系先验 $\gamma_k$ 与门控 $g_k(\cdot)$，详见第 4.2 节。

## 2.2.4 循环神经网络与长短期记忆

循环神经网络（Recurrent Neural Network, RNN）用于处理序列数据。给定输入序列 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$，RNN 在每个时间步 $t$ 维护一个隐状态 $\mathbf{h}_t$，按下式更新：

\$$
\mathbf{h}_t = \tanh\left( \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b} \right)
\tag{2.5}
\$$

标准 RNN 在长序列上存在梯度消失/爆炸问题。Hochreiter 与 Schmidhuber 在 1997 年提出长短期记忆网络（Long Short-Term Memory, LSTM），通过门控机制（输入门、遗忘门、输出门）控制信息流：

\$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
\tag{2.6}
\$$

\$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
\tag{2.7}
\$$

\$$
\mathbf{g}_t = \tanh(\mathbf{W}_g [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_g)
\tag{2.8}
\$$

\$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t
\tag{2.9}
\$$

\$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
\tag{2.10}
\$$

\$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\tag{2.11}
\$$

其中 $\sigma$ 为 sigmoid，$\odot$ 为元素积，$\mathbf{c}_t$ 为细胞状态。LSTM 通过遗忘门 $\mathbf{f}_t$ 与输入门 $\mathbf{i}_t$ 的乘性控释，使得梯度可在时间轴上长期传递而不衰减。

第 4 章 §4.3 DHLSTM-Attn 的帧级 LSTM 层以 LSTM 为基本单元，并在其输入路径上引入差分门控 $\mathbf{W}_{\text{gate}} \boldsymbol{\delta}_t$，根据 STKG 的差分图 $\Delta g_t$ 自适应决定是否更新细胞状态——具体设计见第 4.3.2 节公式 (4.15)。

## 2.2.5 自注意力与 Transformer

Vaswani 等人在 NeurIPS 2017 上提出 Transformer 架构，核心是自注意力机制（Self-Attention）：

\$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
\tag{2.12}
\$$

其中 $Q, K, V$ 分别为查询（Query）、键（Key）、值（Value）矩阵，$d_k$ 为键维度。$\sqrt{d_k}$ 用于缩放点积避免梯度消失。多头自注意力（Multi-Head Self-Attention, MHSA）将 $Q, K, V$ 投影至 $h$ 个子空间，每个头独立做注意力，最后拼接：

\$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O
\tag{2.13}
\$$

\$$
\text{head}_i = \text{Attention}(Q \mathbf{W}_i^Q, K \mathbf{W}_i^K, V \mathbf{W}_i^V)
\tag{2.14}
\$$

Transformer 的核心优势是：所有位置之间的依赖关系都通过点积注意力直接建立，避免了 RNN 中沿时间步逐步传递的链式依赖。这使得长距离依赖建模能力显著优于 RNN，且训练更易并行化。

## 2.2.6 三类机制在本文中的协同

本节介绍的 GNN、LSTM、Transformer 三类机制分别对应第 4 章 K-HSTGAN 的三个核心子模块：

![三线表]
**表 2-3** 神经网络机制与第 4 章设计的对应
[三线表]

| 本节理论工具 | 第 4 章对应设计 | 节号 | 关键改造 |
|------------|---------------|------|---------|
| GAT 单通道注意力 | 关系感知 GAT（RGAT）：15 种关系独立通道 | §4.2 | 引入可学习先验 $\gamma_k$ + 门控 $g_k(\cdot)$ |
| LSTM 标准 cell | 帧级 DHLSTM：差分 $\Delta g_t$ 作为门控信号 | §4.3.2 | 增加 $\text{sigmoid}(\mathbf{W}_{\text{gate}} \boldsymbol{\delta}_t)$ 项 |
| Transformer MHSA | 场景级 Transformer 自注意力 | §4.3.4 | 跨全部时间步的隐表示做自注意力 |
| LSTM 序列建模 + Transformer 长距依赖 | 三层层次化时序编码 | §4.3 | 帧级 LSTM → 行为级注意力 → 场景级 Transformer |

三类机制在本文中并非独立运行，而是按"局部—中观—全局"的层次结构协同：帧级 LSTM 处理相邻帧的演化，行为级注意力对 InteractionEvent 窗口做跨帧聚焦，场景级 Transformer 做全时间步自注意力。这一层次结构是第 4.3 节设计的核心。