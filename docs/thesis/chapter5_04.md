# 5.4 子模块③：D-S 证据理论融合与冲突消解 $\phi_{\text{fuse}}$

## 5.4.1 设计动机

$\phi_{\text{feat}}$ 将规则知识注入 GNN 特征空间，$\phi_{\text{loop}}$ 在 GNN 与规则间建立双向信息流。但到最终决策 phase，规则与 GNN 依然可能给出矛盾的异常判定。此时需要一个形式化的融合框架**显式建模双方的不确定性**，并把调解过程本身作为可解释输出的一部分。

本节采用 **Dempster-Shafer 证据理论**（简称 D-S 理论）作为融合框架的理论基础。选择 D-S 而非贝叶斯融合或简单加权的原因有三：

1. **显式建模不确定性**：D-S 用 $\Theta$（全集）上的质量函数 $m(A)$ 表达对假设 $A$ 的确信度，$m(\Theta)$（分配给人集）表达"不确定"——这在规则与 GNN 输出不一致时极其可贵；
2. **冲突系数 $K$ 作为决策信号**：D-S 组合规则产出的 $K$ 介于 0 到 1，0 表示完全无冲突，1 表示完全冲突。将 $K > \tau_K$ 作为触发冲突消解的阈值，可自动区分场景；
3. **不需要先验概率**：与贝叶斯需要 $P(\text{anomaly})$ 不同，D-S 直接从证据出发，更适合异常检测中异常先验不可靠的问题。

## 5.4.2 焦元定义与质量函数构造

D-S 框架的识别框架（Frame of Discernment）定义为：

$$
\Theta = \{\, \text{anomaly},\ \neg\text{anomaly} \,\}
\tag{5.8}
$$

焦元（Focal Element）定义为 $\Theta$ 的幂集 $\{ \{\text{anomaly}\},\ \{\neg\text{anomaly}\},\ \Theta \}$。

### 5.4.2.1 规则引擎质量函数 $m_{\text{rule}}$

设当前帧 $t$ 规则引擎输出的最大严重度为 $s_t = \max_i \text{severity}_i(v_i)$，定义：

$$
m_{\text{rule}}\big(\{\text{anomaly}\}\big) = s_t \cdot \mathbb{1}[\,\exists_i \text{rule}_i \text{ triggers}\,]
\tag{5.9}
$$

$$
m_{\text{rule}}\big(\{\neg\text{anomaly}\}\big) = (1 - s_t) \cdot \mathbb{1}[\,\exists_i \text{rule}_i \text{ triggers}\,]
\tag{5.10}
$$

$$
m_{\text{rule}}(\Theta) = 1 - m_{\text{rule}}(\{\text{anomaly}\}) - m_{\text{rule}}(\{\neg\text{anomaly}\})
\tag{5.11}
$$

当无规则触发时，$m_{\text{rule}}(\{\text{anomaly}\}) = 0$，$m_{\text{rule}}(\Theta) = 1$——所有质量分配给"不确定性"，表示规则无法判断。

### 5.4.2.2 GNN 质量函数 $m_{\text{GNN}}$

设 K-HSTGAN 对帧 $t$ 的异常预测概率为 $p_t$，方差为 $\epsilon_t$（来自 4.2 节多头注意力的多头输出方差）。定义：

$$
m_{\text{GNN}}\big(\{\text{anomaly}\}\big) = p_t
\tag{5.12}
$$

$$
m_{\text{GNN}}\big(\{\neg\text{anomaly}\}\big) = 1 - p_t - \epsilon_t
\tag{5.13}
$$

$$
m_{\text{GNN}}(\Theta) = \epsilon_t
\tag{5.14}
$$

$\epsilon_t$ 的计算：对 K-HSTGAN 的 $H = 4$ 个注意力头在帧 $t$ 的预测取方差：

$$
\epsilon_t = \frac{1}{H} \sum_{h=1}^{H} (p_t^{(h)} - \bar{p}_t)^2
\tag{5.15}
$$

$\epsilon_t$ 越大，$m_{\text{GNN}}(\Theta)$ 越大——即 GNN 对自己越不确定，分配给"未知"的质量越多。

### 5.4.2.3 两种质量函数的对比

**表 5-2** 规则与 GNN 质量函数的差异

| 属性 | $m_{\text{rule}}$ | $m_{\text{GNN}}$ |
|------|-----------------|-----------------|
| $\Theta$ 的赋值条件 | 规则未触发 → 全不确定 | 多头预测方差大 → 不确定高 |
| $\{\text{anomaly}\}$ 的赋值基础 | 规则触发与否 + severity | 网络输出概率 |
| $\{\neg\text{anomaly}\}$ 的赋值基础 | 规则未触发(但规则在活动) | 1 - p - ε |
| 对零输入的反应 | 全不确定性($m(\Theta)=1$) | $p \approx 0$，$\epsilon \approx 0$ |
| 适用场景 | 规则可覆盖的已知异常 | 全部异常未知/已知 |

## 5.4.3 Dempster 组合规则

对于两质量函数 $m_{\text{rule}}$ 与 $m_{\text{GNN}}$，Dempster 组合规则：

$$
m_{\text{fused}}(A) = \frac{1}{1-K} \sum_{B \cap C = A} m_{\text{rule}}(B) \cdot m_{\text{GNN}}(C)
\tag{5.16}
$$

其中冲突系数：

$$
K = \sum_{B \cap C = \emptyset} m_{\text{rule}}(B) \cdot m_{\text{GNN}}(C)
\tag{5.17}
$$

对于本框架的二分类简化情形，$K$ 的计算可展开为：

$$
K = m_{\text{rule}}(\{a\}) \cdot m_{\text{GNN}}(\{\neg a\}) + m_{\text{rule}}(\{\neg a\}) \cdot m_{\text{GNN}}(\{a\})
\tag{5.18}
$$

融合质量函数的展开式：

$$
m_{\text{fused}}(\{a\}) = \frac{m_{\text{rule}}(\{a\}) \cdot m_{\text{GNN}}(\{a\}) + m_{\text{rule}}(\{a\}) \cdot m_{\text{GNN}}(\Theta) + m_{\text{GNN}}(\{a\}) \cdot m_{\text{rule}}(\Theta)}{1 - K}
\tag{5.19}
$$

$$
m_{\text{fused}}(\{\neg a\}) = \frac{m_{\text{rule}}(\{\neg a\}) \cdot m_{\text{GNN}}(\{\neg a\}) + m_{\text{rule}}(\{\neg a\}) \cdot m_{\text{GNN}}(\Theta) + m_{\text{GNN}}(\{\neg a\}) \cdot m_{\text{rule}}(\Theta)}{1 - K}
\tag{5.20}
$$

**最终判别**：

$$
\hat{y}^{\text{fused}} = \begin{cases}
1\ (\text{anomaly}) & \text{if } m_{\text{fused}}(\{a\}) > 0.5 \\
0\ (\text{normal})  & \text{if } m_{\text{fused}}(\{\neg a\}) > 0.5 \\
\text{uncertain}     & \text{otherwise (触发冲突消解)}
\end{cases}
\tag{5.21}
$$

## 5.4.4 冲突系数 $K$ 作为决策触发条件

KS-NBCF 的策略是：当 $K \leq \tau_K$（默认 0.3），两证据基本一致，直接使用 Dempster 组合的 $m_{\text{fused}}$ 做判断；当 $K > \tau_K$，冲突程度高，触发 5.4.5 节的 KG 证据链回溯仲裁。

此处的阈值 $\tau_K$ 可通过一个独立验证集搜索确定（在第 6 章实验中 $\tau_K$ 的敏感性分析中量化）：

$$
\tau_K^* = \arg\max_{\tau \in [0,1]} \text{F1}(y_{\text{fused}}(\tau), y_{\text{truth}})
\tag{5.22}
$$

## 5.4.5 冲突消解——KG 证据链路径回溯仲裁

当 $K > \tau_K$ 且 $y^{\text{fused}} = \text{"uncertain"}$ 时，触发器 KS-NBCF 的抗误判机制：回溯 STKG 证据链，利用覆盖度统计与证据质量综合裁决。

### 5.4.5.1 规则证据链获取

对每帧冲突帧 $t$ 中所有 `SafetyViolation` 节点 $V = \{sv_1, \dots, sv_k\}$，执行 `MATCH (sv)-[:supportedByEvidence]->(e) RETURN e` 获取证据链（复用 `stk/storage/queries.py` 的 `anomaly_trace_query`）：

$$
\mathcal{P}_v = \{\, e_1, e_2, \dots, e_{L_i} \mid e_j \text{ is evidence node for } sv_i \,\}
\tag{5.23}
$$

### 5.4.5.2 GNN 注意力子图提取

对当前帧 $t$ 的 RGAT 层注意力权重 $\alpha_{ij}^{(k)}$，抽取 top-10 注意力边构成 GNN 关键子图 $\mathcal{S}_a$（同 5.3.4.2 节）：

$$
\mathcal{S}_a(t) = \{\, (i,j,k) \mid \text{top-10 } \alpha_{ij}^{(k)}(t) \,\}
\tag{5.24}
$$

### 5.4.5.3 覆盖度计算

证据链节点集合与 GNN 注意力子图节点的交集率：

$$
\text{overlap} = \frac{|\mathcal{P}_v.\text{nodes} \cap \mathcal{S}_a.\text{nodes}|}{|\mathcal{P}_v.\text{nodes} \cup \mathcal{S}_a.\text{nodes}| + \epsilon}
\tag{5.25}
$$

式中 $\epsilon$ 防止除零。

### 5.4.5.4 仲裁规则

```
算法 5.4: ConflictResolver.resolve(m_rule, m_gnn, K, graph_t)
输入: 规则/gnn 质量函数, 冲突系数 K, 当前帧图
输出: 最终判定 (y_final, explanation)

1. if K <= τ_K:                           # 融合可行
2.    y_final ← argmax(m_fused)
3.    return (y_final, "D-S consistent")
4. end if
5.
6. # K > τ_K — 冲突消解开始
7. P_v ← anomaly_trace_query(graph_t)     # 规则证据链
8. S_a ← extract_attention_subgraph(graph_t.alpha)  # GNN 子图
9. overlap ← compute_overlap(P_v, S_a)
10.
11. if overlap > 0.5:                     # GNN 关注区域与证据链高度重叠
12.    # GNN 的解释子图与规则的证据链匹配 → 信任双方但是 GNN 压倒规则
13.    y_final ← argmax(m_gnn)
14.    return (y_final, f"overlap={overlap:.2f}, trust GNN")
15. end if
16.
17. evidence_strength ← mean([sv.severity for sv in P_v])
18. if evidence_strength > 0.8:
19.    # 规则证据链强度极高 → 信任规则
20.    y_final ← argmax(m_rule)
21.    return (y_final, f"strength={evidence_strength:.2f}, trust rule")
22. end if
23.
24. # 两者都不确定：标记为人工评审
25. return ("needs_review",
26.         f"K={K:.2f}, overlap={overlap:.2f}, strength={evidence_strength:.2f}")
```

### 5.4.5.5 仲裁可解释性输出

算法 5.4 的返回值 `explanation` 直接构成可读的解释短语。对于"信任 GNN"的情形，该短语可以扩展为一条完整解释链：

> 规则引擎触发 `sv_R13a_2048_veh123`（d_actual=4.3m < d_min=7.5m，severity=0.78）
> GNN 预测异常概率仅 0.12（置信中）
> 冲突系数 K = 0.67 > τ_K（0.3）
> 回溯证据链：`supportedByEvidence` 边包含 `in_lane`、`ahead_of`、`violates` 总计 3 条边
> 注意力子图重叠率 overlap = 0.33（低于 0.5 但证据强度 0.78 < 0.8）
> **裁决结果**：`needs_review`——规则与 GNN 的双重不一致，建议人工复核。

这种级别的解释性在现有的自动驾驶安全验证系统中不存在。

## 5.4.6 D-S 融合前后效果定性分析

考虑三类冲突帧在 D-S 融合前后的质量函数变化：

**类型 A（规则高异常 + GNN 低异常）**：

| 质量函数 | $\{a\}$ | $\{\neg a\}$ | $\Theta$ |
|----------|---------|-------------|---------|
| $m_{\text{rule}}$ | 0.78 | 0.15 | 0.07 |
| $m_{\text{GNN}}$ | 0.12 | 0.83 | 0.05 |
| $m_{\text{fused}}$（$K$=0.67,冲突） → 消解 | 0.10 | 0.82 | 0.08 |

D-S 直接融合 (K=0.67) 会给出"不确定" → 冲突消解。若 overlap < 0.5 且 evidence_strength < 0.8，则"needs_review"。

**类型 B（规则低异常 + GNN 高异常）**：

| 质量函数 | $\{a\}$ | $\{\neg a\}$ | $\Theta$ |
|----------|---------|-------------|---------|
| $m_{\text{rule}}$ | 0.10 | 0.00（无触发→全Θ）| 0.90 |
| $m_{\text{GNN}}$ | 0.80 | 0.15 | 0.05 |
| $m_{\text{fused}}$ | 0.82 | 0.13 | 0.05 |

$K=0.01$，无冲突——GNN 被融合结果完全接受，规则因为无触发（全 $\Theta$）不抵制 GNN。

**类型 C（两者都高，GNN 不确定）**：

| 质量函数 | $\{a\}$ | $\{\neg a\}$ | $\Theta$ |
|----------|---------|-------------|---------|
| $m_{\text{rule}}$ | 0.78 | 0.15 | 0.07 |
| $m_{\text{GNN}}$ | 0.65 | 0.10 | 0.25 |
| $m_{\text{fused}}$ | 0.88 | 0.05 | 0.07 |

$K=0.02$（规则与 GNN 都倾向 $\{a\}$），无冲突 → 融合后 $m_{\text{fused}}(\{a\}) = 0.88 > 0.5$ → 异常。GNN 的 $\epsilon = 0.25$（多头不一致）被 D-S 转化为 $\Theta$ 质量，但是规则的高 $m(\{a\})$ 将融合偏量进一步推向异常。

三类演示了 D-S 对规则与 GNN 输出的自适应权重分配——不需要人工设定。

## 5.4.7 小结

本节设计 KS-NBCF 的决策层融合核心 $\phi_{\text{fuse}}$：通过 D-S 证据理论采纳规则与 GNN 的输出构建质量函数，采用 Dempster 组合规则计算融合质量函数，冲突系数 $K$ 超过阈值 $\tau_K$ 时触发 STKG 证据链回溯仲裁。三类冲突融合前后的质量函数变化演示了 D-S 融合理赔在自适应权重分配上的优势。