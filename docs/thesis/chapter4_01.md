# 第 4 章 知识引导的层次化时空图注意力网络 K-HSTGAN

# 4.1 问题映射与 K-HSTGAN 总体架构

自动驾驶仿真安全验证的核心任务之一是异常检测——即在长时仿真过程中，自动识别"违反物理安全约束、违反交通法规或偏离预期行为模式"的帧片段。第 3 章构建的 STKG 为该任务提供了结构化、时态可追溯、规则可推理的输入。然而，将 STKG 直接用于异常检测仍面临三方面挑战：

1. **场景层 15 种关系重要性差异显著**：同车道跟车的 `in_lane`、`ahead_of` 与车辆并排的 `beside` 对异常判别的贡献不同，传统 GAT 对所有邻居统一注意力无法表达这一先验；
2. **节点时序演化具有显著差分结构**：STKG 的增量机制天然提供了 $\Delta g_t$，但传统时序模型（LSTM、Transformer）按帧均匀消费，未利用差分稀疏性，导致大量冗余计算；
3. **规则知识是异常检测的强先验但缺乏注入路径**：RSS 公式与交规在符号引擎中固定触发，深度模型若不能"消化"这些知识，将无法在小样本异常上实现高准确率。

针对上述挑战，本章提出**知识引导的层次化时空图注意力网络**（Knowledge-guided Hierarchical Spatio-Temporal Graph Attention Network，K-HSTGAN）。该网络以 STKG 为输入，输出多任务异常预测，并支持注意力权重的可解释可视化。

## 4.1.1 任务定义与符号约定

### 4.1.1.1 异常检测任务形式化

给定 $\mathcal{STKG}$ 演进序列 $\{G_1, G_2, \dots, G_T\}$ 与对应的差分序列 $\{\Delta g_1, \Delta g_2, \dots, \Delta g_T\}$、规则知识 $\mathcal{K} = \{\text{RSS params}, \text{rule defs}\}$，定义异常检测任务为：

$$
\hat{y}_t\ =\ f_{\theta}\Big(\ G_1, \dots, G_t,\ \Delta g_1, \dots, \Delta g_t,\ \mathcal{K}\ \Big),\quad t \in [1, T]
\tag{4.1}
$$

其中 $\hat{y}_t \in [0,1]$ 为帧 $t$ 的异常预测概率。实际应用中 $T$ 通常取 30（即 1.5 秒滚动窗口 @ 20 fps），覆盖典型的跟车超车行为时间尺度。

多任务输出扩展为：

$$
\hat{y}_t\ =\ \big(\ \hat{y}_t^{\text{anomaly}},\ \hat{\mathbf{y}}_t^{\text{scene}},\ \hat{\mathbf{y}}_t^{\text{behavior}},\ \hat{\mathbf{y}}_t^{\text{rule}}\ \big)
\tag{4.2}
$$

分别对应主异常二分类、场景层异常 3 类、行为层异常 7 类、规则层触发 24 类（含 14 条交规 + 3 项 RSS + 7 类常识违规）。

### 4.1.1.2 输入张量约定

设第 $t$ 帧场景图中存在 $N_t$ 个车辆实体。各模型输入张量定义为：

| 输入 | 形状 | 含义 | STKG 来源 |
|------|------|------|----------|
| $\mathbf{X}_t$ | $N_t \times F$（$F = 18$） | 车辆物理特征 | `VehicleEntity.attrs` |
| $\mathbf{A}_t$ | $N_t \times N_t \times 16$ | 邻接张量（15 种场景关系 + 1 自环） | `SceneRelationType` 的 15 种 |
| $\mathbf{B}_t$ | $N_t \times N_t \times 14$ | 行为邻接张量（13 种行为关系 + 1 自环） | `BehaviorRelationType` 的 13 种 |
| $\Delta \mathbf{X}_t, \Delta \mathbf{A}_t, \Delta \mathbf{B}_t$ | 同上 | 差分张量 | `DeltaGraph` |
| $\boldsymbol{\kappa}_{\text{rss}}$ | $N_t \times 5$ | RSS 残差向量 | `stk/rules/rss/model.py` |
| $\boldsymbol{\kappa}_{\text{rule}}$ | $N_t \times 14$ | 交规触发强度 | `stk/rules/traffic/rules.py` |
| $\mathbf{e}_t$ | $12$ | 环境特征 | `EnvironmentSnapshot` |

为简化符号，后文在无歧义时省略下标 $t$。

## 4.1.2 K-HSTGAN 总体架构

K-HSTGAN 采用**四层结构**：输入层、空间编码层、时序编码层、知识注入与融合层。各层之间形成清晰的数据流：输入层将 STKG 转为张量；空间编码层在每帧做关系感知图注意力，输出节点帧内嵌入；时序编码层对长度 $T$ 的帧序列做层次化 LSTM-Attention，输出节点时序嵌入；知识注入层将 RSS 与交规知识编码为向量特征并融合到主回路；最终融合层做特征聚合与多任务输出。

```
┌────────────────────────────────────────────────────────────────┐
│                     K-HSTGAN 总体架构                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  STKG 输入流         Δg_t 差分流           规则知识 K            │
│  {G_1..T}           {Δg_1..T}              {RSS params,         │
│         │              │                    Rule definitions}   │
│         ▼              ▼                          │             │
│  ┌──────────────────────────┐                     │             │
│  │ 输入层 (§4.1.3)          │                     │             │
│  │ STKG → 张量导出          │                     │             │
│  └────────────┬─────────────┘                     │             │
│               │ X_t, A_t, B_t, Δ_t                │             │
│               ▼                                   │             │
│  ┌──────────────────────────┐    ┌────────────────┴─────┐       │
│  │ 空间编码层 §4.2          │    │ 知识注入层 §4.4       │       │
│  │ 关系感知 GAT             │◀───│ κ_rss + κ_rule +     │       │
│  │ (15 类关系先验)          │    │ Rule Embedding        │       │
│  └────────────┬─────────────┘    └────────────────┬─────┘       │
│               │ 帧内 h_t^(spatial)                  │             │
│               ▼                                     ▼             │
│  ┌──────────────────────────────────────────────────┐         │
│  │ 时序编码层 §4.3                                   │         │
│  │ 帧级 LSTM → 行为级 Attention → 场景级 Transformer  │         │
│  │ Δg_t 作为帧级 LSTM 门控信号                       │         │
│  └────────────┬─────────────────────────────────────┘         │
│               │ h_t^(temporal)                                │
│               ▼                                                │
│  ┌──────────────────────────────────────────────────┐         │
│  │ 融合层 §4.5                                      │         │
│  │ 多任务头：p_anomaly, p_scene, p_behavior, p_rule  │         │
│  │ 加权融合最终异常分数                              │         │
│  └──────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────┘
```

**图 4-1** K-HSTGAN 总体架构。空间编码层消费 STKG 单帧图；时序编码层消费跨帧序列；知识注入层将规则知识注入到上述两层的下游特征中；融合层输出多任务预测。

## 4.1.3 输入层设计

输入层是 K-HSTGAN 与 STKG 的接口层，负责将第 3 章构建的图结构转化为 PyTorch Geometric 可消费的张量。其核心组件是 `STKGToPyGExporter`（待实现于 `stk/gnn/exporter.py`），主要流程如下：

```
算法 4.1: STKGToPyGExporter.export(frame_sequence)
输入: STKG 帧序列 [G_1, ..., G_T] 与差分流 [Δg_1, ..., Δg_T]
输出: PyG Data 序列 [Data_1, ..., Data_T]

1. for each (G_t, Δg_t) in zip(frame_sequence, delta_stream):
2.    // 1. 节点特征矩阵
3.    node_features ← extract_node_features(G_t.vehicles)
4.                  ++ extract_env_features(G_t.env_snapshot)
5.    // 2. 邻接张量：按 16 个关系通道分别构造
6.    A_spatial ← stack([adjacency_matrix(G_t.scene_rels, type=rt) for rt in SceneRelationType])
7.    A_behavior ← stack([adjacency_matrix(G_t.behavior_rels, type=rt) for rt in BehaviorRelationType])
8.    // 3. 差分张量：复用 DeltaGraph
9.    ΔX_t ← compute_feature_diff(G_t.vehicles, G_{t-1}.vehicles)
10.   ΔA_t ← compute_adj_diff(G_t.scene_rels, G_{t-1}.scene_rels)
11.   // 4. 知识注入
12.   κ_rss ← compute_rss_residuals(G_t.vehicles, params)
13.   κ_rule ← compute_rule_strengths(G_t.vehicles, G_t.vehicles)
14.   // 5. 标签
15.   y_anomaly, y_scene, y_behavior, y_rule ← load_labels(G_t)
16.   // 6. 封装为 PyG Data
17.   Data_t ← make_pyg_data(x=node_features, edge_index=A_spatial, ...)
18. end for
19. return [Data_1, ..., Data_T]
```

输入层支持**普通模式**与**差分稀疏模式**两种加载策略。前者加载全部 $T$ 帧的完整张量；后者利用 $\Delta g_t$ 的稀疏性：当 $\Delta g_t.\Delta_{\mathcal{E}}^{\text{unchanged}}$ 占比超过 90% 时，仅复制上一帧的节点特征，仅更新发生属性变化的行的特征。这一策略在第 6 章实验中将体现每帧前向传播时间 30%~50% 的节省。

## 4.1.4 与 RE-GCN、GDN 的差异

K-HSTGAN 与目前时序知识图谱推理和异常检测的代表模型的差异主要体现在三方面。第一，与 RE-GCN [Li et al., SIGIR 2021] 相比，RE-GCN 在文本三元组上做链路预测，输入是离散的实体-关系-实体三元组；K-HSTGAN 输入是带属性版本化和差分结构的 STKG，可承载物理连续值并利用稀疏差分降低计算量。第二，与 GDN [Deng & Hooi, AAAI 2021] 相比，GDN 通过结构学习自动构造传感器图，K-HSTGAN 复用 STKG 的 15 种场景关系作为强先验，省去结构学习且符合驾驶语义。第三，上述二者均无规则知识注入机制，K-HSTGAN 通过 4.4 节的 RSS 编码、交规 Embedding 与弱监督三策略实现先验注入。

## 4.1.5 小结

本节给出 K-HSTGAN 的总体设计：以 STKG 为输入，定义异常检测任务形式化 $\hat{y}_t = f_\theta(G_{1..t}, \Delta g_{1..t}, \mathcal{K})$；模型采用四层结构——输入层、空间编码层、时序编码层、知识注入与融合层。后续 4.2、4.3、4.4、4.5 节将分别展开四个层的设计细节。
