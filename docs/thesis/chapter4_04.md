# 4.4 知识注入层：RSS 公式与交规规则的嵌入

## 4.4.1 设计动机

深度异常检测模型在标注样本充分时表现优秀，但自动驾驶安全场景中异常样本天然稀疏（正常帧与异常帧的比例通常超过 100:1）。规则引擎不依赖样本，但覆盖范围有限。知识注入层的设计目标正是**将符号规则的强先验注入到深度模型中**，使 K-HSTGAN 在小样本异常场景下仍能保持高检测率，同时不损失对未知异常的泛化能力。

知识注入的设计遵循三个原则：**可微分性**——注入过程必须保持端到端可训练；**不破坏泛化**——注入不应造成模型对已知规则的过拟合；**可解释性**——注入途径应留有关键节点特征，便于后续注意力回溯分析。

## 4.4.2 三种知识注入策略

本节设计三种互补的知识注入策略，分别作用于 K-HSTGAN 的三种特征层次。

### 4.4.2.1 策略 I：安全度量残差向量（特征层注入）

**策略**：将 RSS 公式与工程安全度量的混合计算结果以连续值向量的形式拼接至车辆节点的初始特征 $\mathbf{X}_t$。该向量含 5 个分量，其中 2 个来自 RSS 公式（`Δd_long`、`Δd_lat`），3 个来自工程安全度量（`TTC_res`、`v_res`、`brake_res`），统称安全度量残差向量 $\boldsymbol{\kappa}_{\text{safety}}(v)$——尽管为行文方便仍沿用 $\boldsymbol{\kappa}_{\text{rss}}$ 记号，但其含义已不限于纯 RSS 公式。

对每个车辆节点 $v$，计算 5 维安全度量残差向量 $\boldsymbol{\kappa}_{\text{rss}}(v)$：

\$$
\boldsymbol{\kappa}_{\text{rss}}(v) = \Big[
d_{\min}^{\text{long}} - d_{\text{long}},\
d_{\min}^{\text{lat}} - d_{\text{lat}},\
\text{TTC} - \tau_{\text{safe}},\
v - v_{\text{limit}},\
\text{brake} - \text{brake}_{\min}
\Big]
\tag{4.24}
\$$

各分量的含义与计算方式：

| 分量 | 来源 | 含义 | 公式来源 | 正常范围 | 异常方向 |
|------|------|------|---------|---------|---------|
| `Δd_long` | RSS 纵向公式 | 纵向安全距离余量 | Eq.(3.12) | < 0（安全）| > 0 |
| `Δd_lat` | RSS 横向公式 | 横向安全距离余量 | Eq.(3.14) | < 0 | > 0 |
| `TTC_res` | 工程安全度量 | 碰撞时间余量 | TTC = d / (v_A - v_B)，$\tau$=2.5s | > 0 | < 0 |
| `v_res` | 工程安全度量 | 速度限速余量 | current_speed - v_limit | < 0 | > 0 |
| `brake_res` | 工程安全度量 | 制动余量 | brake - brake_min(0.3) | < 0（正常巡航） | > 0（持续制动 → 异常）|

注：`Δd_long` 与 `Δd_lat` 采用"危险程度正向"约定（即 $d_{\min} - d > 0$ 表示实际距离小于安全阈值、处于危险区），与 §2.4.3 公式 (2.28) 残差方向一致，也与源码 `stk/gnn/exporter.py:compute_kappa_rss` 中的工程实现（`d_min_long - d_long`）严格对应。`TTC_res` / `v_res` / `brake_res` 虽在公式中看起来方向不一，但均经 `LayerNorm` 归一化后输入网络，符号方向不影响模型容量。

**关于 5 维分量的归属说明**：5 维残差向量并非严格 5 维纯 RSS 公式残差，而是 2 维 RSS 公式残差 + 3 维工程安全度量的混合向量。`Δd_long` 与 `Δd_lat` 来自 §3.3.3.1 的 RSS 公式 (3.11) 与 (3.13)；`TTC_res` / `v_res` / `brake_res` 三个分量则不属于 RSS 原论文定义，而是本文根据工程实践筛选出与连续物理安全高度相关的辅助指标，用于在 RSS 公式之外为网络补充速度-制动维度的连续证据（因 RSS 公式仅覆盖距离维度，速度与制动维度的连续证据由这三项工程指标提供）。三 维工程指标均与 §3.3.3.2 的部分交规存在弱映射关系（如 `v_res` 与 R10/R11 限速规则、`brake_res` 与 §3.3.3.1 公式 (3.16) NoProperResponse 阈值），但其在本注入策略中以连续值形式进入网络，与 §4.4.2.2 的离散 0/1 触发强度向量形成互补。此外，§3.3.3.1a 表 3-17a 列出的 4 项 RSS 扩充规则（`RSS_CUTIN` / `RSS_CUTOUT` / `RSS_NPR_ENH` / `RSS_CZ_ADAPT`）为框架性描述，未参与本注入策略的 5 维残差向量构成，待后续代码扩展时可独立加入以扩展残差维度。

残差向量 $\boldsymbol{\kappa}_{\text{rss}}(v)$ 经 $\text{LayerNorm}$ 归一化后与原始特征拼接：

\$$
\mathbf{x}_v^{\text{aug}}\ =\ \big[\ \mathbf{x}_v\ \|\ \text{LayerNorm}(\boldsymbol{\kappa}_{\text{rss}}(v))\ \big]
\tag{4.25}
\$$

拼接后的特征维数 $F^{\text{aug}} = F + 5 = 18 + 5 = 23$（原 18 维车辆特征 + 5 维残差）。RGAT 的输入层接受 $F^{\text{aug}}$ 维输入，使 GAT 在注意力计算时天然获得 RSS 先验信息。

**优势**：RSS 残差以连续值形式进入网络，梯度可经残差分量反向传播到网络参数，使 GAT **自动学习每个 RSS 分量的相对重要性**——例如若 RSS 纵向残差对某场景的异常检测贡献极低（可能因为该场景 RSS 参数设置不合理），网络会通过注意力将纵向残差特征的权重压低。

### 4.4.2.2 策略 II：规则触发强度向量的 Embedding（特征层注入）

与 RSS 残差不同，14 条交规的触发是离散事件（触发 / 不触发）。直接使用 0/1 向量会引入稀疏性，不利于训练。因此采用 Embedding 方式。

对每个车辆节点 $v$，交规触发强度向量：

\$$
\boldsymbol{\kappa}_{\text{rule}}(v)\ =\ \big[\ \text{severity}_{R1}(v),\ \text{severity}_{R2}(v),\ \dots,\ \text{severity}_{R18}(v)\ \big]\ \in\ \mathbb{R}^{14}
\tag{4.26}
\$$

其中 $\text{severity}_i(v) \in [0,1]$，0 表示规则 $i$ 未触发，>0 表示触发且严重度。该向量的特点是稀疏——正常帧中最多 1-2 项非零。

将此向量通过一个小型两层 MLP 映射至 $F' = 64$ 维：

\$$
\mathbf{z}_v^{\text{rule}}\ =\ \mathrm{MLP}_{\text{rule}}\big(\ \boldsymbol{\kappa}_{\text{rule}}(v)\ \big)\ \in\ \mathbb{R}^{F'}
\tag{4.27}
\$$

MLP 结构为 $14 \to 32 \to 64$，ReLU 激活。该向量不参与初始特征拼接，而是作为额外信号接入 RGAT 层输出后的残差路径：

\$$
\mathbf{h}_v^{\text{spatial'}}\ =\ \mathbf{h}_v^{\text{spatial}} + \mathbf{z}_v^{\text{rule}}
\tag{4.28}
\$$

残差路径的好处：当交规触发强度极低（$\boldsymbol{\kappa}_{\text{rule}}(v) \approx \mathbf{0}$），MLP 输出约为 $0$，不干扰正常的空间编码；当交规触发显著（如闯红灯有高 severity），$\mathbf{z}_v^{\text{rule}}$ 非零，给空间嵌入加一个"规则信号偏置"。

### 4.4.2.3 策略 III：规则弱监督训练（标签层注入）

前两策略在特征层面注入规则知识。策略 III 在**训练标签**层面利用规则引擎的 `SafetyViolation` 输出。

定义弱监督标签：

\$$
y_t^{\text{weak}}\ =\ \begin{cases}
1 & \text{if } \max_{v} \text{severity}(v, t) > \tau_{\text{weak}}\ (\text{default }0.3) \\
0 & \text{otherwise}
\end{cases}
\tag{4.29}
\$$

该标签仅用于多任务训练的**规则层辅助头**的损失计算：

\$$
\mathcal{L}_{\text{rule}}^{\text{weak}}\ =\ \mathrm{BCE}\big(\ \hat{\mathbf{y}}_t^{\text{rule}},\ y_t^{\text{weak}} \big)
\tag{4.30}
\$$

弱监督标签并非最终 GT 替代——它仅指导规则层辅助头（对应 4.5 节的 $p_{\text{rule}}$ 输出）的初始训练方向。训练后期（约 epoch 10 后），弱监督损失的权重 $\gamma_3$ 线性递减至 0，让模型自主发现规则引擎可能遗漏的异常模式。

弱监督训练的"温度控制"通过动态权重实现：

\$$
\gamma_3(\text{epoch})\ =\ \max\!\big(0,\ \gamma_3^{\text{init}} \cdot (1 - \text{epoch} / T_{\text{warm}})\big)
\tag{4.31}
\$$

默认 $\gamma_3^{\text{init}} = 0.5$，$T_{\text{warm}} = 10$ epochs。这一递减策略保证了模型不会永久性依赖规则信号，符合"规则是脚手架，训练完成后可拆除"的设计哲学。

## 4.4.3 知识注入的完整性验证

表 4-3 对比三种策略在五个维度的表现。

**表 4-3** 三种知识注入策略对比
[三线表]

| 维度 | 策略 I（RSS 残差） | 策略 II（交规 Embedding） | 策略 III（弱监督） |
|------|-------------------|------------------------|------------------|
| 作用层次 | 初始特征 | 空间编码后残差 | 训练损失 |
| 是否可微分 | 是 | 是 | 不需要梯度（标签层）|
| 作用时机 | 训练+推理 | 训练+推理 | 仅训练 |
| 对未知异常泛化影响 | 无（残差正交于分类器）| 极低（残差路径）| 正（引导但逐步退化）|
| 参数引入量 | 0（规则公式固定）| 14×32+32×64=2496 | 0（loss 函数）|
| 对推理速度影响 | 无 | 可忽略 | 无（仅训练阶段）|

## 4.4.4 代码对应

三策略的实现分别分布在 `stk/gnn/` 的以下模块：

- 策略 I：`knowledge_injector.py` → `RSSResidualInjector` 类，调用 `stk/rules/rss/model.py` 的 `run_rss_check`
- 策略 II：`knowledge_injector.py` → `RuleStrengthEncoder` 类，使用 `stk/rules/traffic/rules.py` 的 `check_*` 函数
- 策略 III：`trainer.py` → `WeakSupervisionScheduler` 类，管理 $\gamma_3$ 与 epoch 的递减调度

## 4.4.5 小结

本节描述了 K-HSTGAN 的知识注入层，包含三种互补策略：RSS 残差向量以连续值形式注入初始特征，交规 Embedding 以残差路径作用于空间编码后特征，规则弱监督训练以标签形式引导规则层辅助头。三者共同使符号规则知识以"脚手架"方式影响深度模型的训练与推理，但不破坏模型对未知异常的泛化能力。