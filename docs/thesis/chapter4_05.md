# 4.5 多模态融合头与多任务训练策略

## 4.5.1 设计动机

K-HSTGAN 的最终输出是异常二分类预测 $\hat{y}^{\text{anomaly}}_t \in [0,1]$，但若仅以单一二分类头做端到端训练，存在两个问题：第一，自动驾驶异常类别多样（行为异常、规则异常、物理危险等），同一帧可能同时具备多种异常属性，单二分类信号无法表达这一多标签语义；第二，单任务训练易陷入局部最优，对小样本异常（如夜间+行人鬼探头）的检测率受限。

本节在 4.2 节空间编码层、4.3 节时序编码层与 4.4 节知识注入层之后，设计**多模态融合头**与**多任务训练策略**：以 STKG 的三层结构（场景、行为、规则）作为辅助任务锚点，引导主任务异常二分类的全局损失曲面更加平坦，从而提升对小样本异常的鲁棒性。

## 4.5.2 多模态融合头

### 4.5.2.1 节点时序嵌入与 STKG 三层对齐

设 4.3 节输出 $\mathbf{h}^{\text{temporal}} \in \mathbb{R}^{N \times F'}$ 是单帧窗口下所有 $N$ 个节点（车辆）的时序摘要。但 K-HSTGAN 最终需要对**帧**给出异常预测，因此需要做节点→帧的聚合。本节采用三个不同视角的聚合头，分别对齐 STKG 的三层结构。

**① 场景层头（聚合到节点集合）**：

\$$
\mathbf{h}^{\text{scene}}_t\ =\ \mathrm{MeanPool}\!\left(\ \mathbf{h}^{\text{temporal}}_v\ \middle|\ v \in \text{Vehicle} \cup \text{Pedestrian}\ \right)\ \in\ \mathbb{R}^{F'}
\tag{4.32}
\$$

将所有车辆与行人的时序特征做平均池化，得到一个"场景级摘要向量"。

**② 行为层头（聚合到 InteractionEvent）**：

设当前窗口内 $B$ 个 InteractionEvent，每个事件 $b$ 由其 actor 集合 $\mathcal{A}_b$ 与 frame 窗口 $\mathcal{W}_b$ 限定。对每个 $b$：

\$$
\mathbf{h}^{\text{behavior}}_b\ =\ \mathrm{MeanPool}\!\left(\ \mathbf{h}^{\text{temporal}}_v\ \middle|\ v \in \mathcal{A}_b\ \right)\ \in\ \mathbb{R}^{F'}
\tag{4.33}
\$$

得到 $B$ 个行为层摘要向量。这些向量构成行为层任务的 $B$ 个独立预测单元。当 $B = 0$（当前窗口无行为事件）时，行为层任务让位于默认 dummy 标签。

**③ 规则层头（聚合到 SafetyViolation）**：

设当前窗口内 $K_t$ 个 SafetyViolation 节点（来自 3.5 节规则层）。每个 $sv_k$ 关联一对违规主体 $(s_k, d_k)$，则：

\$$
\mathbf{h}^{\text{rule}}_k\ =\ \big[\ \mathbf{h}^{\text{temporal}}_{s_k}\ \|\ \mathbf{h}^{\text{temporal}}_{d_k}\ \big]\ \in\ \mathbb{R}^{2F'}
\tag{4.34}
\$$

将违规的源与目标节点的时序特征拼接，得到 $K_t$ 个规则层预测单元。

### 4.5.2.2 多头分类器

三个聚合头各自经过一个 2 层 MLP，得到对应任务的预测：

\$$
\hat{y}^{\text{anomaly}}_t\ =\ \sigma\!\left(\ \mathrm{MLP}_{\text{anomaly}}\!\left(\ \mathbf{h}^{\text{scene}}_t\ \right)\ \right)\ \in\ [0,1]
\tag{4.35}
\$$

\$$
\hat{\mathbf{y}}^{\text{scene}}_t\ =\ \mathrm{softmax}\!\left(\ \mathrm{MLP}_{\text{scene}}\!\left(\ \mathbf{h}^{\text{scene}}_t\ \right)\ \right)\ \in\ \mathbb{R}^3
\tag{4.36}
\$$

\$$
\hat{\mathbf{y}}^{\text{behavior}}_t\ =\ \mathrm{softmax}\!\left(\ \mathrm{MLP}_{\text{behavior}}\!\left(\ \mathbf{h}^{\text{behavior}}_b\ \right)\ \right)\ \in\ \mathbb{R}^7
\tag{4.37}
\$$

\$$
\hat{\mathbf{y}}^{\text{rule}}_t\ =\ \mathrm{sigmoid}\!\left(\ \mathrm{MLP}_{\text{rule}}\!\left(\ \mathbf{h}^{\text{rule}}_k\ \right)\ \right)\ \in\ \mathbb{R}^{14}
\tag{4.38}
\$$

每个 MLP 默认结构为 $F' \to 64 \to \text{out\_dim}$，中间 ReLU 激活，末端未激活（softmax 或 sigmoid 在外层应用）。

注意 $\hat{\mathbf{y}}^{\text{rule}}_t$ 是多标签输出（同一帧可能触发多个交规），用 sigmoid；$\hat{\mathbf{y}}^{\text{scene}}$ 与 $\hat{\mathbf{y}}^{\text{behavior}}$ 是单标签多分类，用 softmax。

### 4.5.2.3 主任务最终融合

主任务 $\hat{y}^{\text{anomaly}}_t$ 与三个辅助任务的输出做加权融合：

\$$
\hat{y}^{\text{fused}}_t\ =\ w_0 \cdot \hat{y}^{\text{anomaly}}_t + w_1 \cdot \max_j \hat{y}^{\text{scene}}_{t,j} + w_2 \cdot \max_b \max_j \hat{y}^{\text{behavior}}_{b,j} + w_3 \cdot \max_k \hat{y}^{\text{rule}}_{k,\cdot}
\tag{4.39}
\$$

权重 $w_0$–$w_3$ 默认值为 $1.0, 0.1, 0.2, 0.3$，可在 `config/training.yaml` 中调整。融合最终输出 $\hat{y}^{\text{fused}}_t$ 取代单任务 $\hat{y}^{\text{anomaly}}_t$ 作为推理阶段最终异常分数。

## 4.5.3 多任务损失

### 4.5.3.1 总损失函数

设 $\mathcal{L}_0$ 为主损失（异常二分类 BCE），$\mathcal{L}_1, \mathcal{L}_2, \mathcal{L}_3$ 为三个辅助任务的损失（场景层、行为层、规则层）。总损失：

\$$
\mathcal{L}_{\text{total}}\ =\ \mathcal{L}_0\ +\ \lambda_1\, \mathcal{L}_1\ +\ \lambda_2\, \mathcal{L}_2\ +\ \lambda_3\, \mathcal{L}_3\ +\ \lambda_{\text{reg}}\, \mathcal{L}_{\text{reg}}
\tag{4.40}
\$$

权重 $\lambda_1, \lambda_2, \lambda_3$ 默认为 $0.5, 0.5, 0.5$，$\lambda_{\text{reg}}$ 默认 $10^{-4}$。

### 4.5.3.2 主任务损失与类别平衡

异常/正常帧比例极度不均衡（100:1），主损失采用 Focal Loss [Lin et al., 2017]：

\$$
\mathcal{L}_0\ =\ -\ \frac{1}{T \cdot N} \sum_{t, v} \alpha_t\,(1 - \hat{p}_{t,v})^{\gamma_{\text{focal}}} \cdot \big[\ y_{t,v} \log \hat{p}_{t,v} + (1 - y_{t,v}) \log (1 - \hat{p}_{t,v})\ \big]
\tag{4.41}
\$$

其中 $\gamma_{\text{focal}} = 2$ 是 Focal Loss 调节参数，$\alpha_t$ 为类别权重，由窗口内异常帧比例自适应计算：

\$$
\alpha_t\ =\ \min\!\left(\ 1,\ \frac{\#\text{normal}}{\#\text{anomaly} + \epsilon}\ \right)
\tag{4.42}
\$$

默认 $\epsilon = 1$ 防止除零。$\alpha_t$ 上限设为 100 避免极端值导致训练不稳定。

### 4.5.3.3 辅助任务损失

辅助任务采用标准交叉熵：

\$$
\mathcal{L}_1\ =\ -\ \frac{1}{T} \sum_t \sum_{j=1}^{3} y^{\text{scene}}_{t,j} \log \hat{y}^{\text{scene}}_{t,j}
\tag{4.43}
\$$

\$$
\mathcal{L}_2\ =\ -\ \frac{1}{B} \sum_b \sum_{j=1}^{7} y^{\text{behavior}}_{b,j} \log \hat{y}^{\text{behavior}}_{b,j}
\tag{4.44}
\$$

\$$
\mathcal{L}_3\ =\ -\ \frac{1}{K_t} \sum_k \sum_{j=1}^{14} \big[\ y^{\text{rule}}_{k,j} \log \hat{y}^{\text{rule}}_{k,j} + (1 - y^{\text{rule}}_{k,j}) \log (1 - \hat{y}^{\text{rule}}_{k,j})\ \big]
\tag{4.45}
\$$

注意 $\mathcal{L}_3$ 是 BCE（多标签），而非交叉熵。

### 4.5.3.4 规则弱监督温控

策略 III（4.4 节）的弱监督标签 $y^{\text{weak}}$ 通过 $\mathcal{L}_3^{\text{weak}}$ 注入到 $\mathcal{L}_3$：

\$$
\mathcal{L}_3\ =\ \mathcal{L}_3^{\text{gt}}\ +\ \gamma_3(\text{epoch})\, \mathcal{L}_3^{\text{weak}}
\tag{4.46}
\$$

其中 $\gamma_3(\text{epoch})$ 按 (4.31) 线性递减。当 epoch $\geq T_{\text{warm}} = 10$ 时，$\gamma_3 = 0$，弱监督完全淡出，最终训练仅依赖真实标签。

### 4.5.3.5 正则化项

正则化项包含 L2 正则与图注意力稀疏正则：

\$$
\mathcal{L}_{\text{reg}}\ =\ \sum_{\theta \in \Theta} \|\theta\|_2^2\ +\ \beta\, \sum_{k} \sum_{i,j} |\alpha_{ij}^{(k)}|^2
\tag{4.47}
\$$

第二项鼓励注意力分布稀疏化，避免所有节点获得相近的注意力权重，影响可解释性。$\beta$ 默认 $0.01$。

## 4.5.4 训练策略

### 4.5.4.1 多阶段训练流程

K-HSTGAN 训练分三阶段进行，以平衡主任务的细调与辅助任务的先验注入：

| 阶段 | Epoch 范围 | 描述 | 学习率 | 弱监督权重 |
|------|----------|------|--------|----------|
| 阶段 I：预训练 | 0-5 | 仅训练辅助头 $\mathcal{L}_1, \mathcal{L}_2, \mathcal{L}_3$，主头 $w_0 = 0$ | $10^{-3}$ | $\gamma_3 = 0.5$ |
| 阶段 II：联合微调 | 5-30 | 全任务联合训练，所有损失权重均衡 | $10^{-4}$ | $\gamma_3$ 从 0.4 线性减至 0 |
| 阶段 III：主任务微调 | 30-50 | 冻结辅助头，仅训练主头 | $10^{-5}$ | $\gamma_3 = 0$ |

三阶段对应"先学结构→联合微调→聚焦主任务"的训练哲学。阶段 I 利用规则弱监督快速给模型一个合理参数空间方向；阶段 II 在所有任务监督下达到全局最优；阶段 III 仅做主任务微调，避免辅助任务过拟合。

### 4.5.4.2 优化器与学习率

| 优化对象 | 优化器 | 学习率 | 调度 |
|---------|--------|--------|------|
| RGAT 参数 $\mathbf{W}_k, \mathbf{a}_k$ | AdamW | $10^{-4}$ | Cosine annealing |
| LSTM 参数 $\theta$ | AdamW | $10^{-3}$ | Step decay |
| Transformer 自注意 | AdamW | $10^{-4}$ | Linear warmup (5k steps) → Cosine |
| MLP 头 | AdamW | $10^{-3}$ | Step decay |
| 关系先验 $\gamma_k$ | Adam | $10^{-2}$ | 固定 |

不同组件使用不同学习率，反映出对参数空间的不同先验——关系先验 $\gamma_k$ 学习率高，使先验能快速适应数据；RGAT、Transformer 学习率低，避免深度结构破坏稳定训练流。

### 4.5.4.3 梯度裁剪与稳定性

训练过程中 LSTM 易出现梯度爆炸，对所有参数施加梯度裁剪：

\$$
\mathbf{g}\ \leftarrow\ \min\!\left(\ 1,\ \frac{\|\mathbf{g}\|_2}{c}\ \right)\ \cdot\ \mathbf{g},\quad c = 5.0
\tag{4.48}
\$$

同时使用 EMA（Exponential Moving Average）做参数平均，缓解训练后期梯度震荡：

\$$
\theta_{\text{EMA}}^{(t)}\ =\ 0.99\, \theta_{\text{EMA}}^{(t-1)} + 0.01\, \theta^{(t)}
\tag{4.49}
\$$

测试期使用 $\theta_{\text{EMA}}$ 替代 $\theta$ 做推理。

### 4.5.4.4 早停策略

主任务 F1 在验证集连续 5 个 epoch 无提升时触发早停。同时维护 Best F1 与 Best Recall 两个指标，分别保存对应权重。最终选择 F1 最高的 checkpoint 作为产出版本。

## 4.5.5 端到端伪代码

```
算法 4.3: K-HSTGAN.forward(G_1..T, Δg_1..T, K, env_1..T)
输入: STKG 帧序列、差分序列、规则知识、环境序列
输出: 多任务预测 (y_anomaly, y_scene, y_behavior, y_rule)

# 步骤 1: 输入层将 STKG 转为张量
1. (X, A, B, Δ, κ_rss, κ_rule, env) ← STKGExport(G_1..T, Δg_1..T, K, env_1..T)

# 步骤 2: 知识注入策略 I 拼接初始特征
2. X_in ← concat([X, normalize(κ_rss)], dim=-1)  # N x (F+5)

# 步骤 3: 4.2 RGAT 空间编码（每帧独立）
3. H_spatial ← RGAT(X_in, A)  # T x N x F'

# 步骤 4: 知识注入策略 II 残差路径注入规则 Embedding
4. Z_rule ← MLP_rule(κ_rule)  # T x N x F'
5. H_spatial' ← H_spatial + Z_rule

# 步骤 5: 4.3 DHLSTM-Attn 时序编码
6. δ_t ← construct_delta(Δ_1..T)  # 差分门控输入
7. H_lstm ← DeltaGatedLSTM(H_spatial', δ_t)  # T x N x F'
8. H_behavior ← BehaviorAttention(H_lstm, behavior_events)  # B x F'
9. H_temporal ← TransformerSelfAttn(concat(H_lstm, H_behavior))  # (T+B) x d_k
10. h_temporal ← H_temporal[T-1, :]  # F'

# 步骤 6: 4.5 多模态融合头
11. h_scene ← MeanPool(H_temporal[0:T, :])  # F'
12. h_behavior_agg ← MeanPool(H_temporal[T:, :])  # F'
13. h_rule ← RuleAggregator(H_temporal, sv_relations)  # K_t x 2F'
14. y_anomaly ← sigmoid(MLP_anomaly(h_scene))
15. y_scene ← softmax(MLP_scene(h_scene))
16. y_behavior ← softmax(MLP_behavior(h_behavior_agg))
17. y_rule ← sigmoid(MLP_rule(h_rule))
18. y_fused ← w0 * y_anomaly + w1 * max(y_scene) + w2 * max(y_behavior) + w3 * max(y_rule)

return (y_anomaly, y_scene, y_behavior, y_rule, y_fused)
```

## 4.5.6 小结

本节描述了 K-HSTGAN 的融合头与训练策略。融合头从时序编码层的输出 $\mathbf{h}^{\text{temporal}}$ 出发，对 STKG 三层结构分别做聚合（场景层 MeanPool、行为层 InteractionEvent 聚合、规则层 SafetyViolation 聚合），输出四任务预测。多任务损失含主任务 Focal Loss 与三个辅助任务交叉熵，规则弱监督按 (4.31) 递减。三阶段训练（先验预训练 → 联合微调 → 主任务微调）配合梯度裁剪、EMA 与早停保证训练稳定性。下一节将以与现有方法的对比收束本章。