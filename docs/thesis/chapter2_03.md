# 2.3 D-S 证据理论

Dempster-Shafer 证据理论（Dempster-Shafer Theory, D-S evidence theory）是一种用于不确定性建模与证据融合的数学框架，由 Dempster 在 1968 年提出 [35]、Shafer 在 1976 年出版专题著作 [36] 完善。D-S 理论的核心特征是：（i）能表达"无知"（ignorance）状态；（ii）能显式建模冲突；（iii）能融合多个独立证据源而不退化为简单加权。这三点恰对应第 1.2.4 节指出的符号-神经融合方法"无冲突建模、不可解释"两项核心缺失。

## 2.3.1 识别框架

D-S 理论从一个识别框架（Frame of Discernment, FoD）出发，定义为互斥且完备的命题集合：

\$$
\Theta = \{\theta_1, \theta_2, \ldots, \theta_N\}, \quad \theta_i \cap \theta_j = \emptyset \ (i \neq j), \quad \bigcup_{i=1}^{N} \theta_i = \Omega
$$

其中 $\Omega$ 为论域。识别框架的子集称为"焦元"（focal element），记为 $A \subseteq \Theta$。在本文第 5 章 §5.4 自动驾驶异常检测的应用中，识别框架固定为二分类：

$$
\Theta = \{\text{anomaly}, \text{normal}\}
\tag{2.15}
\$$

对应的焦元有四个：$\{a\}$（异常）、$\{n\}$（正常）、$\Theta$（不确定）、$\emptyset$（不可能）。

## 2.3.2 基本概率指派函数

D-S 理论用基本概率指派函数（Basic Probability Assignment, BPA，又称质量函数 mass function）表达证据：

\$$
m: 2^{\Theta} \to [0, 1]
\tag{2.16}
\$$

满足：（i）$m(\emptyset) = 0$；（ii）$\sum_{A \subseteq \Theta} m(A) = 1$。

$m(A)$ 表示证据**精确**支持命题 $A$ 的程度（而非支持 $A$ 的子命题）。$m(\Theta)$ 是关键量，表示"证据对任意子命题均无足够支持"，即无知（ignorance）。在 $\Theta = \{a, n\}$ 二分类下，质量函数共三个非零量 $m(\{a\}), m(\{n\}), m(\Theta)$，其和为 1。

质量函数优于概率分布之处在于：概率分布要求 $P(\{a\}) + P(\{n\}) = 1$，无法表达"两个命题都不知道"的状态；质量函数则允许 $m(\Theta) > 0$ 表达无知。

## 2.3.3 信任函数与似然函数

由质量函数可诱导两类对偶测度：

**信任函数（Belief Function）**：证据**支持**命题 $A$ 的总置信度：

\$$
\text{Bel}(A) = \sum_{B \subseteq A} m(B)
\tag{2.17}
\$$

**似然函数（Plausibility Function）**：证据**不反对**命题 $A$ 的总置信度：

\$$
\text{Pl}(A) = \sum_{B \subseteq \Theta, B \cap A \neq \emptyset} m(B) = 1 - \text{Bel}(\bar{A})
\tag{2.18}
\$$

信任与似然构成对 $A$ 真值的区间估计 $[\text{Bel}(A), \text{Pl}(A)]$。区间宽度 $\text{Pl}(A) - \text{Bel}(A) = m(\Theta)$，恰好等于无知度。当两个证据源对命题 $A$ 都无知（$m_1(\Theta) = m_2(\Theta) = 1$），融合后仍无知——这是 D-S 理论合理性的体现。

## 2.3.4 Dempster 组合规则

Dempster 组合规则（Dempster's Rule of Combination）将两个独立证据的质量函数 $m_1, m_2$ 融合为 $m_{12}$：

\$$
m_{12}(A) = \frac{1}{1 - K} \sum_{B \cap C = A} m_1(B) \cdot m_2(C), \quad A \neq \emptyset
\tag{2.19}
\$$

\$$
m_{12}(\emptyset) = 0
\tag{2.20}
\$$

其中冲突系数 $K$ 表示两证据的矛盾程度：

\$$
K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)
\tag{2.21}
\$$

冲突系数 $K \in [0, 1]$：
- $K = 0$：两证据无矛盾；
- $K \to 1$：两证据高度冲突，1/(1-K) 趋于 $\infty$，归一化严重放大矛盾；
- $K = 1$：两证据完全矛盾，组合规则失效。

冲突系数 $K$ 是第 5 章 §5.4 KS-NBCF φ_fuse 子模块的核心决策量：当 $K$ 超过阈值 $\tau_K$ 时（默认 $\tau_K = 0.5$），融合结果进入"高冲突情形"，触发 STKG 证据链回溯仲裁机制，详见 §5.4.5。

## 2.3.5 Dempster 组合规则的两点特性

**结合律**：Dempster 组合满足结合律 $(m_1 \oplus m_2) \oplus m_3 = m_1 \oplus (m_2 \oplus m_3)$，但**不满足交换律相关的归一化稳定性**——多证据融合时，$K$ 系数会因融合顺序不同而累积。本文采用"规则→GNN"两源融合，且为单步融合，无需考虑多源归一化顺序问题。

**归一化的非对称影响**：归一化因子 $1/(1-K)$ 在 $K$ 接近 1 时会显著放大单源置信度。这一性质是 Yager 等学者对 D-S 理论的核心批评——它会"掩盖"两源矛盾对最终决策的负面影响。本文通过引入冲突阈值 $\tau_K$ 与回溯仲裁路径（§5.4.5）规避了这一陷阱。

## 2.3.6 二分类情形的解析式

对于第 5 章实际使用的二分类 $\Theta = \{a, n\}$，每个证据源可写为三元组 $(m(\{a\}), m(\{n\}), m(\Theta))$。给两个源 $(m_1^a, m_1^n, m_1^\Theta)$ 与 $(m_2^a, m_2^n, m_2^\Theta)$，Dempster 组合的解析形式如下：

\$$
m_{12}(\{a\}) = \frac{m_1^a \cdot m_2^a + m_1^a \cdot m_2^\Theta + m_1^\Theta \cdot m_2^a}{1 - K}
\tag{2.22}
\$$

\$$
m_{12}(\{n\}) = \frac{m_1^n \cdot m_2^n + m_1^n \cdot m_2^\Theta + m_1^\Theta \cdot m_2^n}{1 - K}
\tag{2.23}
\$$

\$$
m_{12}(\Theta) = \frac{m_1^\Theta \cdot m_2^\Theta}{1 - K}
\tag{2.24}
\$$

其中冲突系数：

\$$
K = m_1^a \cdot m_2^n + m_1^n \cdot m_2^a
\tag{2.25}
\$$

公式 (2.22)–(2.25) 是第 5 章 §5.4 算法 5.4（`ConflictResolver.resolve`）的核心计算式。

## 2.3.7 与第 5 章设计的接续关系

![三线表]
**表 2-4** D-S 证据理论工具与第 5 章设计对应
[三线表]

| 本节理论工具 | 第 5 章对应设计 | 节号 |
|------------|---------------|------|
| 识别框架 $\Theta$ | 二分类 $\{\text{anomaly}, \text{normal}\}$ | §5.4.2 |
| 质量函数 $m(\cdot)$ | 规则引擎与 GNN 各自构造的质量函数 $m_{\text{rule}}, m_{\text{gnn}}$ | §5.4.3 |
| Dempster 组合规则 | $m_{\text{fuse}} = m_{\text{rule}} \oplus m_{\text{gnn}}$ | §5.4.4 |
| 冲突系数 $K$ | 触发阈值 $\tau_K = 0.5$，超阈值进入回溯仲裁 | §5.4.5 |
| 区间 $[\text{Bel}, \text{Pl}]$ | 决策可信度评估，输出"信任规则/信任 GNN/需人工复核" | §5.4.5 |

本节内容仅给出 D-S 理论的标准化形式。第 5 章将基于这些形式化定义构造规则与 GNN 的质量函数，并设计高冲突情形下的仲裁路径——这是本文创新点三的具体技术载体。