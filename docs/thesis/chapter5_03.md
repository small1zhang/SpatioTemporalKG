# 5.3 子模块②：训练-推理三阶段双向闭环 $\phi_{\text{loop}}$

## 5.3.1 设计动机

5.2 节 $\phi_{\text{feat}}$ 实现了规则 → GNN 的单向流。但仅有单向流不足以构成闭环——规则对 GNN 的影响只是"注入式"，GNN 不能反作用于规则。本节设计 $\phi_{\text{loop}}$ 实现**双向**信息流：训练前用规则做弱监督标签，训练中用 GNN 输出调整规则置信度，推理时用 GNN 注意力激活并扩展规则模板。

三阶段对应训练流程的三个关键时点：

| 阶段 | 时点 | 信息流向 | 主要作用 |
|------|------|---------|---------|
| 阶段 I | 训练前 | 规则 → GNN | 弱监督标签（数据扩充）|
| 阶段 II | 训练中 | GNN → 规则 | 反馈调整规则置信度 |
| 阶段 III | 推理时 | GNN → 规则 | GNN 注意力激活规则模板 |

三阶段不仅时序上前后衔接，且各自形成独立完整的"规则与 GNN 双向互动"。本节将分别展开三阶段的设计。

## 5.3.2 阶段 I：训练前——规则弱监督标签生成

### 5.3.2.1 问题背景

K-HSTGAN 的主任务（异常二分类）需要充足的正例样本驱动训练，但人工/仿真标注的异常帧占比仅 1%–5%。在没有外部规则的辅助下，GNN 主任务训练将陷入严重的正负样本不均衡问题，Focal Loss 也难以彻底解决。

阶段 I 利用规则引擎的 `SafetyViolation` 输出生成弱监督标签 $y^{\text{weak}}_t$，作为多任务训练中**规则辅助头**的标签来源——而非替代真实标签。

### 5.3.2.2 弱监督标签定义

弱监督标签按帧汇总规则触发情况：

$$
y^{\text{weak}}_t \in \mathbb{R}^{14},\quad y^{\text{weak}}_{t, i} = \max_v \text{severity}_i(v, t)
\tag{5.1}
$$

其中 $\text{severity}_i(v, t)$ 是帧 $t$ 中规则 $i$ 对车辆 $v$ 的触发严重度（来自 `stk/rules/traffic/rules.py`）。当窗口内有多个车辆触发同一条规则，取严重度最大值作为该帧的弱标签。

注意：弱监督标签通过 4.5 节规则层辅助头 $\mathbf{y}^{\text{rule}}_t$ 与 (4.45) 定义的 BCE 损失 $\mathcal{L}_3^{\text{gt}}$ 进入模型训练：

$$
\mathcal{L}_3^{\text{weak}} = -\frac{1}{T} \sum_t \sum_{i=1}^{14} \big[\ y^{\text{weak}}_{t,i} \log \hat{y}^{\text{rule}}_{t,i} + (1 - y^{\text{weak}}_{t,i}) \log (1 - \hat{y}^{\text{rule}}_{t,i})\ \big]
\tag{5.2}
$$

### 5.3.2.3 温度递减控制

弱监督仅在训练前 10 个 epoch 起作用，权重 $\gamma_3(\text{epoch})$ 线性递减：

$$
\gamma_3(\text{epoch}) = \max\big(0,\ 0.5 \cdot (1 - \text{epoch}/10)\big)
\tag{5.3}
$$

epoch 10 后 $\gamma_3 = 0$，弱监督完全淡出。这是为了保证 GNN 不**永久依赖规则信号**——训练完成后规则可"拆除"，GNN 应能独立检测该 14 类规则覆盖的异常。

### 5.3.2.4 弱监督与真实标签的关系

| 信号类型 | 来源 | 作用 |
|---------|------|------|
| 真实标签 $y^{\text{gt}}$ | 仿真注入日志（anomaly_log.json）+ 人工标注 | 主任务 $\mathcal{L}_0$ 监督 |
| 弱监督 $y^{\text{weak}}$ | 规则引擎 SafetyViolation 输出 | 辅助 $\mathcal{L}_3$ 初始训练信号 |

二者独立——真实标签来自第一手仿真日志，弱监督来自规则引擎的间接推理。两者交叉的部分（如 R10 高速限速同时被规则与人工标注）通过加权融合到 $\mathcal{L}_3$；不交叉的部分（规则触发但真实标签为 0）则是规则误报，温控机制让 GNN 逐渐远离规则信号。

## 5.3.3 阶段 II：训练中——GNN 反馈调整规则置信度

### 5.3.3.1 问题背景

阶段 I 时规则对 GNN 起到"教师"作用。但规则本身**每条触发阈值固定**（如 R10 限速 120 km/h 的阈值在 `config/traffic_rules.yaml` 中固定）。在不同场景下规则的"合理阈值"可能略有差异——例如在恶劣天气，限速应严格到 90 km/h；在晴朗高速可放宽到 130 km/h。固定阈值使规则在某些场景下误报率过高或漏报率过高。

阶段 II 利用 GNN 在训练中观察到的预测不一致性自动**反馈调整规则置信度**，激发规则对场景的适应性。

### 5.3.3.2 规则置信度调整模型

设训练时第 $i$ 条规则的置信度为 $\eta_i \in [0, 1]$，初始 $\eta_i = 1$（强信）。对当前 epoch，统计三条流：

| 流 | 信号 | 计算方式 |
|---|------|---------|
| GNN 流 | 规则触发但 GNN 不视异常的帧比例 | $s_{i}^- = \frac{1}{|T_i|} \sum_{t \in T_i} [\hat{y}^{\text{anomaly}}_t < 0.3]$ |
| 真值流 | 规则触发但真值正常的帧比例 | $s_{i}^0 = \frac{1}{|T_i|} \sum_{t \in T_i} [\text{gt\_anomaly}_t = 0]$ |
| 支撑流 | 同规则触发的 STKG 证据链长度 | $\ell_i = \text{avg\_evidence\_chain\_len}(T_i)$ |

则规则 $i$ 的置信度更新：

$$
\eta_i^{(\text{epoch}+1)} \leftarrow \eta_i^{(\text{epoch})} \cdot \big[\ 1 - \beta\, (s_i^- + \varepsilon_i) \cdot (1 - \eta_i)^{+}\ \big]
\tag{5.4}
$$

其中 $\beta$ 是学习步长（默认 0.001），$\varepsilon_i$ 是规则的"可信度惩罚"项：

$$
\varepsilon_i = \frac{0.2}{\sqrt{\ell_i}} + 0.05\, s_i^0
\tag{5.5}
$$

当 $\eta_i$ 低于阈值（默认 0.3）时触发**规则参数调整**：

$$
\theta_i \leftarrow \theta_i - \nabla_{\theta_i} \mathcal{L}\!\left(\ \text{rule}_i(\theta_i) \,\text{vs.}\, \text{gt\_anomaly}\ \right)
\tag{5.6}
$$

$\theta_i$ 是规则 $i$ 的阈值参数（如 R10 的限速 120 km/h）。梯度下降需要规则可微，对一些阈值固定的规则（R1 行人优先距离 15 m）采用区间搜索：尝试 $\theta_i \in \{\theta_i - \Delta, \theta_i, \theta_i + \Delta\}$ 三种参数值，取该规则触发后与真值交集最大的阈值。

### 5.3.3.3 阶段 II 伪代码

```
算法 5.2: FeedbackAdjuster.update(epoch, rule_pool, gnn_outputs, gt_labels)
输入: epoch, 规则池, GNN 输出列表, 真值标签列表
输出: 更新后的规则置信度 {η_i} 和参数 {θ_i}

1. for each rule i in rule_pool:
2.    triggered_frames ← get_triggered_frames(rule_i)
3.    s_i_minus ← mean([1 if gnn(t) < 0.3 else 0 for t in triggered_frames])
4.    s_i_zero ← mean([1 if not gt_anomaly(t) else 0 for t in triggered_frames])
5.    ell_i ← avg_evidence_chain_len(triggered_frames, rule_i)
6.    epsilon_i ← 0.2 / sqrt(ell_i) + 0.05 * s_i_zero
7.    eta_i ← eta_i * (1 - beta * (s_i_minus + epsilon_i) * max(0, 1 - eta_i))
8.    if eta_i < 0.3:
9.         # 阈值触发调整
10.        candidate_theta ← {theta_i - delta, theta_i, theta_i + delta}
11.        theta_i ← argmax_theta(F1_score(rule_i(theta), gt_anomaly))
12.    end if
13. end for
14. save(rule_pool.eta, rule_pool.theta)
```

### 5.3.3.4 反馈稳定保证

为避免规则置信度因噪声振荡，引入两个稳定机制：

- **EMA 平滑**：$\eta_i^{(\text{new})} \leftarrow 0.9\, \eta_i^{(\text{old})} + 0.1\, \eta_i^{(\text{new})}$
- **触发边界**：每次更新若 $\eta_i$ 变化幅度 $< 5\%$，则不更新（避免无意义微小调整）

## 5.3.4 阶段 III：推理时——GNN 注意力激活规则模板

### 5.3.4.1 问题背景

阶段 III 是推理时的双向闭环——GNN 检测到异常时，将注意力权重对应的子图反向提交给规则引擎，激活已有规则的新模板变体。这是 KS-NBCF 中最创新的环节，使规则能根据 GNN 输入**动态扩展**。

### 5.3.4.2 GNN 注意力子图提取

对推理时 K-HSTGAN 给出 $y^{\text{anomaly}}_t > 0.5$ 的帧（GNN 触发异常），抽取 RGAT 的注意力权重 $\alpha_{ij}^{(k)}$ 排序前 $K_{\text{top}} = 10$ 个 $(i, j, k)$ 三元组，构成"异常注意力子图 $\mathcal{S}_a$":

$$
\mathcal{S}_a(t) = \big\{\ (i, j, k)\ \big|\ \text{top-}K_{\text{top}} \alpha_{ij}^{(k)}(t)\ \big\}
\tag{5.7}
$$

### 5.3.4.3 规则模板匹配

对每个三元组 $(i, j, k)$，根据关系类型 $k$ 查找规则模板库：

| 关系类型 $k$ | 关联规则模板 |
|------------|------------|
| `ahead_of` | `following_too_close` |
| `beside` | `lateral_safe_displacement` |
| `nearby_pedestrian` | `pedestrian_proximity` |
| `adjacent_lane` | `lane_change` |
| `in_junction` | `junction_no_yield` |

每个模板是一个**带有可填充参数槽**的规则原型，例如 `following_too_close` 模板：

```python
Template("following_too_close",
         front_v = SLOT,
         behind_v = SLOT,
         d_threshold = 8.0,
         severity = SLOT)
```

### 5.3.4.4 模板实例化

抽取 $\mathcal{S}_a$ 中 $(i, j, k)$ 配合 STKG 当帧的 `ahead_of(v_i, v_j)` 关系，填充模板得：

```python
instance = Template("following_too_close",
                   front_v = v_j, behind_v = v_i,
                   d_threshold = 8.0,
                   severity = gnn_attention_score)
```

规则引擎接收实例化模板后做一次轻量级规则评估：若该当帧 $d_{\text{long}}(v_i, v_j) < 8.0$，则触发该动态规则，并入 `SafetyViolation` 列表（规则层标签 $\text{rule\_layer} = \text{"Dynamic"}$）。

### 5.3.4.5 阶段 III 伪代码

```
算法 5.3: TemplateActivator.activate(gnn_output, attention_weights, frame_data)
输入: GNN 输出, 注意力权重, 当前帧数据
输出: 新增动态 SafetyViolation 列表 (可能为空)

1. if gnn_output.y_anomaly < 0.5:  # GNN 未触发异常
2.    return []
3. end if
4. S_a ← top_K_attentions(attention_weights, K=10)
5. new_violations ← []
6. for each (i, j, k) in S_a:
7.    template ← match_template(k, frame_data)
8.    if template is None:
9.        continue
10.   end if
11.   instance ← template.fill(front_v=v_j, behind_v=v_i,
12.                            severity=attention_score)
13.   sv ← instance.evaluate(frame_data)
14.   if sv.is_violation:
15.       sv.rule_layer = "Dynamic"
16.       new_violations.append(sv)
17.   end if
18. end for
19. return new_violations
```

阶段 III 的输出 `new_violations` 会按下文 5.4 节 D-S 融合的"规则证据质量函数"重新参与决策融合，最终接受或拒绝该 GNN 异常判断。

## 5.3.5 三阶段协同

三阶段在训练与推理全周期内协同工作：

```
训练前 ─ 阶段I（弱监督标签生成）
       │       ↓
训练过程 阶段II（GNN反馈调整规则置信度）  每个epoch末
       │   └─→ 规则参数变化反馈到 5.4 节质量函数构造
       ▼
推理时 阶段III（注意力激活规则模板）       └─→ 5.4 节融合做最终判断
       │
       ▼
   输出 y_fused + explanation_path
```

**图 5-2** 三阶段闭环数据流动

阶段 I 与阶段 III 同时引入"规则 → GNN" 与 "GNN → 规则" 两个方向的信息，再通过阶段 II 的反馈调整机制将 GNN 的训练表现注入到规则的参数中。由此闭环构成。阶段 III 完成后输出（新触发的动态 SafetyViolation）进入 5.4 节的 D-S 融合作为"规则证据源"。

## 5.3.6 小结

本节设计 KS-NBCF 的核心机制 $\phi_{\text{loop}}$：将双向闭环按训练-推理时序分为三阶段——训练前的弱监督标签生成、训练中的规则置信度反馈调整、推理时的 GNN 注意力激活规则模板。三阶段互为支撑、前后衔接，构成了区别于单向规则注入的关键所在，是 KS-NBCF 双向闭环特性的根本来源。