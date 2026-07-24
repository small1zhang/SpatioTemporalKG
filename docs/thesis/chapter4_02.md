# 4.2 空间编码层：关系感知图注意力 GAT

## 4.2.1 设计动机

标准 GAT（Veličković et al. [2018]）对节点 $i$ 的邻居 $j$ 计算注意力权重：

$$
\alpha_{ij}\ =\ \sigma\big(\ \mathbf{a}^{\top}\big[\ \mathbf{W}\mathbf{h}_i\ \|\ \mathbf{W}\mathbf{h}_j\ \big]\ \big)
\tag{4.3}
$$

其中 $\mathbf{W}$ 为共享权重矩阵，$\mathbf{a}$ 为注意力向量，$\sigma$ 为 LeakyReLU 激活。该公式的核心假设是"所有邻居通过同一注意力通道被关注"。但 STKG 场景层的 15 种关系对异常检测的判别价值差异显著：`ahead_of` 关系对跟车异常判别至关重要，`adjacent_lane` 仅用作拓扑参考，`weather_context` 与异常无直接物理联系。把所有关系塞进同一个注意力通道将损失大量先验信息。

为此，本节提出**关系感知 GAT**（Relation-aware GAT，简称 RGAT），核心思想是：在每帧场景图上，为 STKG 的 15 种场景关系各自分配独立的注意力通道，并在通道之间做加权融合——融合权重由关系先验向量 $\boldsymbol{\beta} \in \mathbb{R}^{15}$ 与可学习的注意力门控共同决定。

## 4.2.2 RGAT 形式化

### 4.2.2.1 邻接张量与关系通道

设第 $t$ 帧场景图共 $N$ 个节点，节点特征矩阵 $\mathbf{X} \in \mathbb{R}^{N \times F}$。邻接张量 $\mathbf{A} \in \mathbb{R}^{N \times N \times K}$，其中 $K = 15$ 为场景关系类型数，$\mathbf{A}_{ij}^{(k)} = 1$ 表示节点 $i$ 与 $j$ 之间存在第 $k$ 类关系。本节使用 $k \in \{1, 2, \dots, K\}$ 索引关系通道，按 `SceneRelationType` 的枚举顺序对应。

### 4.2.2.2 单通道 GAT

对每个关系通道 $k$，独立计算注意力权重：

$$
\mathbf{e}_{ij}^{(k)}\ =\ \mathrm{LeakyReLU}\Big(\ \mathbf{a}_k^{\top}\big[\ \mathbf{W}_k \mathbf{h}_i\ \|\ \mathbf{W}_k \mathbf{h}_j\ \big]\ \Big)
\tag{4.4}
$$

其中 $\mathbf{W}_k \in \mathbb{R}^{F' \times F}$ 为该通道的线性变换矩阵，$\mathbf{a}_k$ 为该通道的注意力向量。通道间权重不共享。该机制使 `ahead_of` 关系拥有独立的注意力参数空间，与 `beside`、`nearby_pedestrian` 等关系互不干扰。

经 softmax 归一化（沿 $j$ 维）得到注意力系数：

$$
\alpha_{ij}^{(k)}\ =\ \frac{\exp\big(\mathbf{e}_{ij}^{(k)}\big)}{\sum_{j' \in \mathcal{N}_k(i)} \exp\big(\mathbf{e}_{ij'}^{(k)}\big)}
\tag{4.5}
$$

其中 $\mathcal{N}_k(i)$ 是节点 $i$ 在第 $k$ 类关系下的邻居集合。每个通道的节点更新：

$$
\mathbf{h}_i^{(k)}\ =\ \sigma\!\left(\ \sum_{j \in \mathcal{N}_k(i)} \alpha_{ij}^{(k)}\, \mathbf{W}_k \mathbf{h}_j\ \right)
\tag{4.6}
$$

### 4.2.2.3 关系先验加权融合

15 个通道输出 $\{\mathbf{h}_i^{(1)}, \dots, \mathbf{h}_i^{(K)}\}$ 经加权融合得到该层最终节点嵌入：

$$
\mathbf{h}_i^{\text{spatial}}\ =\ \sum_{k=1}^{K} \beta_k\, \mathbf{h}_i^{(k)}
\tag{4.7}
$$

其中 $\beta_k$ 是关系 $k$ 的融合权重，由先验向量与门控函数联合决定：

$$
\beta_k\ =\ \frac{\exp\!\big(\ \gamma_k + g_k(\mathbf{h}_i)\ \big)}{\sum_{k'=1}^{K} \exp\!\big(\ \gamma_{k'} + g_{k'}(\mathbf{h}_i)\ \big)}
\tag{4.8}
$$

其中：

- $\gamma_k \in \mathbb{R}$：可学习的关系先验对数权重，初始化为表 4-1 的先验值；
- $g_k(\cdot)$：单层 MLP，输出标量，作为对关系 $k$ 在当前节点上下文下的"动态调节"项。

### 4.2.2.4 关系先验初始化

关系先验的初始值由 STKG 设计经验给出，体现"哪些关系对异常检测更重要"：

**表 4-1** 场景关系先验权重（初始 $\gamma_k$）

| 类别 | 关系 | 初始 $\gamma_k$ | 理由 |
|------|------|----------------|------|
| 高先验 | `ahead_of` | $\log 4$ | 跟车与追尾判定核心 |
| 高先验 | `nearby_pedestrian` | $\log 4$ | 行人安全核心 |
| 高先验 | `in_lane` | $\log 3$ | 同车道判定基础 |
| 高先验 | `beside` | $\log 3$ | 超车与变道核心 |
| 中先验 | `controlled_by` | $\log 2$ | 闯红灯判定 |
| 中先验 | `in_junction` | $\log 2$ | 路口场景识别 |
| 中先验 | `adjacent_lane` | $\log 1.5$ | 变道可能性来源 |
| 中先验 | `on_road` | $\log 1.5$ | 行驶路段识别 |
| 中先验 | `lane_connects` | $\log 1$ | 路段拓扑连通 |
| 低先验 | `weather_context` | $\log 0.5$ | 间接影响 |
| 低先验 | `containsVehicle` | 0（基础项） | 帧聚合关系，不参与注意力 |
| 低先验 | `containsPedestrian` | 0 | 同上 |
| 低先验 | `containsTrafficLight` | 0 | 同上 |
| 低先验 | `containsRoad` | 0 | 同上 |
| 低先验 | `hasEnvironment` | 0 | 同上 |

注意：`contains*` 与 `hasEnvironment` 这五类帧聚合关系不参与邻居注意力计算（它们的"邻居"是帧根节点，物理意义不显著），仅作为 4.1.3 节输入层中环境向量的桥接通道使用。因此有效注意力通道数为 $K' = 10$。

### 4.2.2.5 多头机制

借鉴 GAT 的多头机制，本节在每个关系通道内独立使用 $H$ 个注意力头：

$$
\mathbf{h}_i^{(k)}\ =\ \big\|\_{h=1}^{H}\ \sigma\!\left(\ \sum_{j \in \mathcal{N}_k(i)} \alpha_{ij}^{(k,h)}\, \mathbf{W}_k^{(h)} \mathbf{h}_j\ \right)
\tag{4.9}
$$

实验中默认 $H = 4$，每个头输出维度 $F' / H = 16$，总输出维度 $F' = 64$。多头输出在最后一层做平均而非拼接：

$$
\mathbf{h}_i^{(k),\text{final}}\ =\ \frac{1}{H} \sum_{h=1}^{H} \mathbf{h}_i^{(k,h)}
\tag{4.10}
$$

## 4.2.3 复杂度分析

RGAT 单层前向计算复杂度：

$$
\mathcal{O}\!\left(\ K \cdot H \cdot |\mathcal{E}|\cdot F'\ \right)
\tag{4.11}
$$

其中 $|\mathcal{E}|$ 为图的总边数。与传统 GAT 的复杂度 $\mathcal{O}(H \cdot |\mathcal{E}| \cdot F')$ 相比，RGAT 引入了 $K = 10$ 倍的常数因子。但得益于 STKG 关系的稀疏性（每帧平均 $|\mathcal{E}| \approx 4.2 \times 10^3$），实际单帧前向延迟与传统 GAT 在同规模无关系区分的图上相当。第 6 章 RQ2.1 实测数据将验证这一论断。

## 4.2.4 与 RE-GCN 的对比

RE-GCN [Li et al., SIGIR 2021] 同样在关系种类丰富的情况下做图卷积，但其对所有关系做"统一变换 + 拼接"，缺乏先验加权机制：

$$
\mathbf{h}_i^{\text{RE-GCN}}\ =\ \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_k(i)} \frac{1}{|\mathcal{N}_k(i)|}\ \mathbf{W}_k \mathbf{h}_j
\tag{4.12}
$$

对比 RGAT 公式 (4.4)-(4.8)，可看出 RGAT 在三方面优于 RE-GCN：

1. **可学习注意力 vs 均匀采样**：RE-GCN 邻居贡献度相同，RGAT 通过注意力学习不同邻居的相对重要性；
2. **关系先验可学习 vs 固定同等权重**：RE-GCN 对 15 种关系融合时采用同等权重，RGAT 引入可学习先验 $\gamma_k$ 表达先验差异；
3. **门控动态调节**：RGAT 的 $g_k(\cdot)$ 允许同一关系在不同节点上下文下贡献不同，是 RE-GCN 完全缺乏的机制。

## 4.2.5 实现伪代码

```
算法 4.2: RGATLayer.forward(X, A)
输入: X ∈ R^{N×F} (节点特征), A ∈ R^{N×N×K} (邻接张量)
输出: H_spatial ∈ R^{N×F'} (空间编码后节点嵌入)

1. N, F ← shape(X)
2. H_channel ← []          // 收集 K 个通道输出
3. for k in 1..K:
4.    if γ_k == 0:  // contains* 类不参与注意力
5.        continue
6.    end if
7.    // === 单通道 GAT 多头 ===
8.    H_k ← []
9.    for h in 1..H:
10.       W_k_h ← weight_k[h]  // F' x F
11.       a_k_h ← attention_k[h]  // 2F'
12.       X' ← X @ W_k_h.T  // N x F'
13.       e_ij ← LeakyReLU(a_k_h.T @ concat(X'_i, X'_j))  // 仅 A[:,:,k]=1 处
14.       alpha_ij ← softmax_j(e_ij)
15.       h_i_h ← sum_j alpha_ij * X'_j  // N x F'
16.       H_k.append(h_i_h)
17.   end for
18.   H_k ← concat(H_k, dim=-1)  // N x H*F'
19.   H_channel.append((k, H_k))
20. end for
21. // === 关系先验加权融合 ===
22. beta ← []
23. for (k, H_k) in H_channel:
24.   g_k ← MLP_g_k(H_k).squeeze(-1)  // N (标量)
25.   beta_k ← exp(γ_k + g_k)  // N
26.   beta.append(beta_k)
27. end for
28. beta ← stack(beta, dim=-1)  // N x K'
29. beta ← beta / beta.sum(dim=-1, keepdim=True)
30. H_spatial ← sum_k beta[:,:,k] * H_channel[k]  // N x F'
31. return H_spatial
```

`stk/gnn/rgat.py`（待实现）将上述算法封装为 PyTorch Geometric `MessagePassing` 类，使用稀疏矩阵操作以利用 STKG 的稀疏邻接结构。

## 4.2.6 行为邻接的使用

除场景层 15 种关系外，STKG 还提供 13 种行为关系（`following`、`approaching` 等）。行为关系本身是 4.3 节时序编码层的关键输入，但其注意力直接通过 4.2 节 RGAT 处理也存在两种选择：

- **方案 A**：行为关系作为 RGAT 的额外通道，与场景关系并列。总共 $K = 15 + 13 = 28$ 通道；
- **方案 B**：行为关系仅在时序编码层使用，空间编码层只消费场景关系。

本文采用**方案 B**，理由如下：

1. 行为关系在 STKG 中已带有时态信息（`valid_from`，`frame_start`/`frame_end`），其语义本质属于"跨帧事件"，与 RGAT 的"单帧瞬时"语境不平衡；
2. 行为关系相比场景关系稀疏得多（每帧行为关系 $\approx$ 场景关系的 1/10），独立通道可能被场景关系淹没；
3. 时序编码层可专门针对行为关系做窗口注意力，发挥其时间维度的信息。

因此 4.2 节 RGAT 仅消费场景关系，行为关系将在 4.3 节时序编码层独立处理。

## 4.2.7 小结

本节设计 K-HSTGAN 的空间编码层——关系感知 GAT（RGAT）。RGAT 为 STKG 的 15 种场景关系各自分配独立注意力通道，并通过可学习先验 $\gamma_k$ 与门控 $g_k(\cdot)$ 做通道间加权融合。与标准 GAT 和 RE-GCN 相比，RGAT 显式利用 STKG 关系类型的语义差异，使注意力机制与场景本体融合。RGAT 单帧输出节点空间嵌入 $\mathbf{h}_i^{\text{spatial}} \in \mathbb{R}^{F'}$，将作为 4.3 节时序编码层的输入。
