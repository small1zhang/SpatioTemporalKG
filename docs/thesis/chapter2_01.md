# 2.1 知识图谱及时空扩展

## 2.1.1 知识图谱的基本定义

知识图谱（Knowledge Graph, KG）是一种以图结构表示实体与实体间语义关系的知识表示形式。其形式化定义为：知识图谱 $G = (\mathcal{E}, \mathcal{R}, \mathcal{F})$，其中 $\mathcal{E}$ 为实体集合，$\mathcal{R}$ 为关系类型集合，$\mathcal{F} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$ 为事实三元组集合，每个事实 $f = (h, r, t) \in \mathcal{F}$ 表示头实体 $h$ 与尾实体 $t$ 之间存在关系 $r$ [11]。

知识图谱的本体（ontology）层为其上层 schema，定义实体类型（class）与关系类型（property）的层次结构与约束。W3C 推荐的资源描述框架（RDF）与 Web 本体语言（OWL）是知识图谱的两个核心数据模型标准。RDF 以"主-谓-宾"三元组表示事实，OWL 在 RDF 之上增加描述逻辑（Description Logic, DL）以支持类层次、属性约束、基数限制等结构化推理。

知识图谱的推理任务主要包括三类：
- **链接预测（Link Prediction）**：给定部分三元组，预测缺失的 $h/r/t$。
- **实体对齐（Entity Alignment）**：在不同 KG 之间识别指代同一真实世界的实体。
- **知识问答（Question Answering）**：将自然语言问题转化为结构化查询。

本节中知识图谱的基本定义将作为第 3 章 STKG 本体设计的形式化基础。

## 2.1.2 时序知识图谱

时序知识图谱（Temporal Knowledge Graph, TKG）在标准 KG 上引入时间维度，每个事实可携带有效时间区间。其形式化定义为 $G_T = (\mathcal{E}, \mathcal{R}, \mathcal{T}, \mathcal{F}_T)$，其中 $\mathcal{T}$ 为时间戳或时间区间集合，$\mathcal{F}_T \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E} \times \mathcal{T}$ 为带时间的事实四元组集合 [12]。

时序事实可分为以下四类时态语义 [14]：

![三线表]
**表 2-1** 时序事实的四类时态语义

| 类别 | 形式 | 含义 | 例子 |
|------|------|------|------|
| 瞬时事实 | $(h, r, t, \tau)$ | 在时刻 $\tau$ 发生的事件 | (车辆A, 变道进入, 车道B, $t_3$) |
| 区间事实 | $(h, r, t, [\tau_s, \tau_e])$ | 在时间区间内持续成立 | (车辆A, 跟车, 车辆B, $[t_1, t_5]$) |
| 演化事实 | $(h, r, t, \tau) \to (h, r', t', \tau')$ | 时间推移下关系变化 | (A, ahead_of, B) $\to$ (A, beside, B) |
| 周期事实 | $(h, r, t, \text{every } \Delta)$ | 周期性发生 | (信号灯, 切换, 红, every 30s) |

TKG 上的核心推理任务为时序链接预测：给定查询 $(h, r, ?, t)$，预测在时间 $t$ 满足该关系的尾实体。ICEWS14/18、GDELT 等公开数据集是该任务的标准评测基准 [15]。但 ICEWS18 时间粒度为天，GDELT 粒度为 15 分钟——这远粗于本文所需 50 ms 帧级粒度。

## 2.1.3 时空知识图谱

时空知识图谱（Spatio-Temporal Knowledge Graph, STKG）在 TKG 基础上进一步引入空间关系作为一类显式关系类型 [13]。其形式化定义为 $G_{ST} = (\mathcal{E}, \mathcal{R}_s \cup \mathcal{R}_t, \mathcal{T}, \mathcal{F}_{ST})$，其中 $\mathcal{R}_s$ 为空间关系集合（如 `located_in`、`near`、`ahead_of`），$\mathcal{R}_t$ 为非空间语义关系集合（如 `interacts_with`）。

STKG 与传统 TKG 的关键差异在于：空间关系附带几何约束（如距离、方向），这种约束在更新过程中具有"几何确定性"——只要实体的位置、朝向已知，空间关系是否成立可以通过几何计算精确判定（无需经验学习）。这一性质是第 3 章场景层 15 种空间关系（`in_lane`、`ahead_of`、`beside` 等）的设计基础。

STKG 的另一个特征是其图谱结构的稀疏性变化。在自动驾驶仿真帧序列下，相邻帧间的实体演化具有稀疏性——大部分车辆的属性在大部分帧中保持不变。这一稀疏性已在第 1 章相关工作中提及，是本文 STKG 设计差异图 $\Delta g_t$ 增量更新机制的物理基础，也是第 4 章 DHLSTM-Attn 差分门控机制的输入信号来源。

## 2.1.4 STKG 与本文工作的接续关系

本节内容对应第 3 章 §3.1 STKG 形式化定义与 §3.2 四层本体架构：第 3 章将基于本节定义具体设计 14 类实体、42 种关系、节点生命周期与属性版本化机制。表 2-2 汇总本节理论工具与第 3 章设计的对应关系。

![三线表]
**表 2-2** 知识图谱理论工具与第 3 章设计对应

| 本节理论工具 | 第 3 章对应设计 | 节号 |
|------------|---------------|------|
| KG 形式化定义 $G = (\mathcal{E}, \mathcal{R}, \mathcal{F})$ | STKG 元层形式化 $G_t = G_{t-1} \oplus \Delta g_t$ | §3.1 |
| RDF/OWL 数据模型 | 14 类实体类型与 4 大类 42 种关系 | §3.2 |
| TKG 时态语义四分类 | 节点生命周期四状态机（candidate→active→inactive→deleted）| §3.6 |
| STKG 几何确定性 | 场景层 15 种空间关系的纯函数计算 | §3.3 |