# 5.2 子模块①：特征层规则先验注入 $\phi_{\text{feat}}$

## 5.2.1 设计定位

$\phi_{\text{feat}}$ 的目标是在 K-HSTGAN 的输入层与中间层将符号规则的连续值引入网络。它不作决策、不改变网络结构，仅以**特征分量**的形式存在于 GNN 的整个前向路径中。这种设计的核心优势是：规则以连续值进入网络后，梯度可经规则分量反向传播到 GNN 参数，使网络**自动学习每条规则分量对当前训练场景的贡献权重**——若某条规则对特定场景意义不大（如 RSS 在无跟车场景），网络会收敛让该特征进入注意力低值区域。

$\phi_{\text{feat}}$ 完全复用第 4.4 节两种特征注入策略：策略 I（RSS 残差向量拼接入初始特征）与策略 II（交规触发强度 Embedding 以残差路径作用）。本节在 4.4 节的基础上进一步分析这两类特征注入在"符号—神经融合"语境下的语义内涵。

## 5.2.2 策略 I 形式化回顾与耦合分析

回顾 4.4.2.1 节：对车辆 $v$，RSS 残差向量 $\boldsymbol{\kappa}_{\text{rss}}(v) \in \mathbb{R}^5$ 由五维残差拼接并 $\mathrm{LayerNorm}$ 后与原始特征 $\mathbf{x}_v \in \mathbb{R}^{18}$ 合并为 $F^{\text{aug}} = 23$ 维初始特征：

$$
\mathbf{x}_v^{\text{aug}} = [\mathbf{x}_v \| \mathrm{LayerNorm}(\boldsymbol{\kappa}_{\text{rss}}(v))]
$$

在 KS-NBCF 语境下，$\boldsymbol{\kappa}_{\text{rss}}(v)$ 的维度含义可改写为更抽象的"物理安全力矩"表达：

| 残差分量 $j$ | 对应物理约束 | 零残差含义 | 正残差含义 | 负残差含义 |
|-------------|-----------|----------|----------|----------|
| 1 | $\Delta d_{\text{long}}$ | $d = d_{\min}$（边界）| 安全 | 危险 |
| 2 | $\Delta d_{\text{lat}}$ | $d = d_{\min}^{\text{lat}}$ | 安全 | 危险 |
| 3 | TTC - 2.5s | TTC = 2.5s | 安全 | 危险 |
| 4 | $v - v_{\text{limit}}$ | 等于限速 | 未超速 | 超速 |
| 5 | brake - 0.3 | 制动 0.3（NoProperResponse 阈值）| 充分制动 | 未充分制动 |

$\boldsymbol{\kappa}_{\text{rss}}(v)$ 在 RGAT 中的前向传播不依赖 GNN 的预测结果，因此具有"规则先验的正交性"——GNN 自身参数 $\theta$ 的更新不影响 $\boldsymbol{\kappa}_{\text{rss}}$ 的计算，极大降低了"神经网络吸收规则信号但遗忘规则"的风险。

## 5.2.3 策略 II 形式化回顾与耦合分析

回顾 4.4.2.2 节：对每个节点 $v$：

$$
\mathbf{z}_v^{\text{rule}} = \mathrm{MLP}_{\text{rule}}(\boldsymbol{\kappa}_{\text{rule}}(v)) \in \mathbb{R}^{F'}
$$
$$
\mathbf{h}_v^{\text{spatial'}} = \mathbf{h}_v^{\text{spatial}} + \mathbf{z}_v^{\text{rule}}
$$

在 KS-NBCF 语境下，$\boldsymbol{\kappa}_{\text{rule}}(v)$ 的极端稀疏性需要特别处理。当 $\boldsymbol{\kappa}_{\text{rule}}(v) = \mathbf{0}^{(14)}$ 时：

- $\boldsymbol{\kappa}_{\text{rule}}(v)$ → MLP ReLU → $\mathbf{z}_v^{\text{rule}}$ 为近似零向量；
- $\mathbf{h}_v^{\text{spatial'}} \approx \mathbf{h}_v^{\text{spatial}}$，即规则残差路径退化为恒等映射；
- GNN 行为不受规则影响，在众多"正常"帧上保持纯洁的几何特征编码。

只有当规则触发（severity > 0）时，$\mathbf{z}_v^{\text{rule}}$ 产生非零偏置，"规则信号"才参与决策。这种"异常才介入"的设计符合安全验证的效用原则——正常场景中只要 GNN 表现好，不要干扰；异常场景中规则信号必须强制参与。

## 5.2.4 实现伪代码

```
算法 5.1: FeatureInjector.apply(X, scenario_entities, vehicles)
输入: 原始特征矩阵 X ∈ ℝ^{N×F}, 场景实体集, 车辆子集
输出: 注入后特征矩阵 X_aug ∈ ℝ^{N×F_aug}

1. κ_rss_list ← []
2. for each vehicle v in vehicles:
3.    front_v ← get_front_vehicle(v, scene_relations)
4.    d_long ← longitudinal_distance(v, front_v)
5.    d_min ← compute_dmin_long(v.speed, front_v.speed, RSS_PARAMS)
6.    ttc ← compute_ttc(d_long, v.speed - front_v.speed)
        7.    κ ← [
        8.        d_min - d_long, compute_dmin_lat(...) - compute_dlat(v, front_v),
        9.        ttc - 2.5, v.speed - v.speed_limit,
        10.       v.brake - 0.3
        11.   ]
12.   κ_rss_list.append(LayerNorm(κ))
13. end for
14. κ_rss ← stack(κ_rss_list)  # N×5
15. X_aug ← concat([X, κ_rss], dim=-1)  # N×F_aug
16.
17. // 策略 II 在 GAT 输出后以残差调用
18. // (在 4.2 节算法 4.2 的 line 29 之后被调用)
19. return X_aug
```

15 行为直接从 D-S 证据理论 1 级 2 个子任务（RSS 残差、规则触发强度 Embedding、弱监督训练）中继承自第 4 章。

## 5.2.5 与 4.4 节的关系

4.4 节知识注入层与 5.2 节 $\phi_{\text{feat}}$ 在实现上完全复用（strategies I & II），但本节将相同的机制放置在"符号-神经闭环融合"的大框架下进行了语义定位与耦合分析：

- 4.4 节回答"知识注入怎么注入"（实现细节）；
- 5.2 节回答"知识注入在融合框架中扮演什么角色"（定位协同）。前者是后者在 KS-NBCF 框架下的具体实现。

$\phi_{\text{feat}}$ 为本框架提供三方面的融合支持：
1. **规则先验的特征正交性**，确保正常帧 GNN 不受规则信号干扰；
2. **规则信号的紧急介入能力**，异常帧规则强制偏置特征空间；
3. **对下游 $\phi_{\text{loop}}$ 的输入保证**：$\phi_{\text{feat}}$ 注入后的损失函数变化可作为 5.3.2 节 GNN 反馈调整规则置信度的输入信号之一。