# 3.1 问题定义与形式化

自动驾驶仿真平台（如 CARLA、LGSVL、Apollo Simulation）持续输出包含车辆、行人、信号灯、道路、环境等多模态要素的实时观测数据。这些原始数据具有以下特征：**多源异构**（同一时刻存在物理、几何、控制、规则等多类信息）、**时空同步**（所有要素在统一帧时钟下采样）、**演化交织**（场景结构、行为状态、违规事件在时间轴上层层递进）。仅依赖原始数据难以为下游安全验证任务提供结构化、可推理、可解释的支撑。本章在仿真数据与下游应用之间引入时空动态知识图谱（Spatio-Temporal Dynamic Knowledge Graph, STKG）作为中间表示层，旨在将一帧或多帧仿真观测转为语义明确、时态可追溯、规则可推理的图结构。

本节首先给出 STKG 的形式化定义，随后阐明时态三元组的语义、节点生命周期模型与命名空间机制的统一约定，为后续 3.2 节本体设计、3.3--3.5 节三层构建、3.6 节动态更新奠定符号基础。

## 3.1.1 知识图谱与时空知识图谱

经典知识图谱 $\mathcal{KG} = (\mathcal{E}, \mathcal{R}, \mathcal{F})$ 由实体集 $\mathcal{E}$、关系集 $\mathcal{R}$ 与事实集 $\mathcal{F} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$ 构成，每条事实 $f = (s, p, o)$ 表示主体 $s$ 经谓词 $p$ 指向客体 $o$。该定义关注"什么是什么"的静态认知，缺乏对"何时为真"的时间维度建模。

为引入时态，本文采用 **时态三元组**（Temporal Triple） 的扩展定义：

$$
\tau := (s,\ p,\ o,\ t,\ [t_s, t_e],\ \mathbf{a},\ c)
\tag{3.1}
$$

其中 $s \in \mathcal{E}$ 为.subject、$o \in \mathcal{E}$ 为 object、$p \in \mathcal{R}$ 为谓词关系；$t \in \mathbb{N}^+$ 为采样帧号（与仿真 tick 对齐）；$[t_s, t_e]$ 为该三元组的有效时间区间（valid time），$t_s$ 为生效帧、$t_e$ 为失效帧（可空）；$\mathbf{a} \in \mathbb{R}^k$ 为附加属性向量（如距离、速度差等数值化证据）；$c \in [0,1]$ 为置信度。

**时空动态知识图谱** 在此基础上引入演进机制：

$$
\mathcal{STKG} := \big\langle\ \mathcal{E},\ \mathcal{R},\ \mathcal{A},\ \mathcal{T},\ \mathcal{P},\ \{G_t\}_{t=1}^{T}\ \big\rangle
\tag{3.2}
$$

其中 $\mathcal{E}$ 为实体类型集合、$\mathcal{R}$ 为关系类型集合、$\mathcal{A}$ 为属性集合、$\mathcal{T}$ 为时间索引集合、$\mathcal{P}$ 为公理约束集合（详见 3.2.2 节）；$\{G_t\}_{t=1}^{T}$ 为按帧组织的图序列，$G_t$ 表示第 $t$ 帧的瞬时认知状态，由该帧全部节点、边与时态信息构成。

设 $\Delta g_t := G_t \ominus G_{t-1}$ 表示相邻两帧的差分图（详见 3.6.1 节形式化定义），则 STKG 的演进过程可写为：

$$
G_t = G_{t-1} \oplus \Delta g_t,\quad t = 2, \dots, T
\tag{3.3}
$$

其中 $\oplus$ 为增量融合算子，满足公理 $A_7$（详见 3.2.2 节）。该递推形式是后续 3.6 节"动态更新机制"的理论基础。

## 3.1.2 实体、关系与时态三元组的类化表达

为便于代码实现与图谱存储，本文将时态三元组 $\tau$ 拆解为三个独立类：

- **实体节点** $e$（`BaseEntity`）：承载 subject/object 角色，由全局唯一 `entity_id`、`entity_type`、`valid_from`、`valid_to`、属性字典 `attrs` 与 `confidence` 构成；
- **关系边** $r$（`BaseRelation`）：承载 predicate 角色，由 `src_id`、`dst_id`、`relation_type`、`frame_id`、`valid_from`、`valid_to`、`attrs` 与 `confidence` 构成；
- **时态三元组** $\tau$（`TemporalTriple`）：逻辑上等价于 $(s, p, o, t, [t_s, t_e], \mathbf{a}, c)$，作为"事实链"的最终表达形式，可与上述节点-边模型一一互转。

拆分的好处有三：

1. 节点（`BaseEntity`）与边（`BaseRelation`）可独立索引与查询，便于图数据库（Neo4j）原生存储；
2. 关系边携带 `frame_id`，天然支持时态重放与帧切片查询；
3. 同一实体或关系可在不同帧复用，避免重复存档，符合本体公理 $A_2$（实体类型固定）与 $A_3$（属性版本化）。

## 3.1.3 节点生命周期模型

自动驾驶仿真中实体进出场景频繁，车辆的"消失""再现""离开路口"等都需要在图上清晰区分。本文为每个实体引入 **节点生命周期** 模型，定义四状态有限状态机：

$$
\text{NodeLifecycleStatus} := \{\textsf{CREATED},\ \textsf{ACTIVE},\ \textsf{STALE},\ \textsf{INACTIVE}\}
\tag{3.4}
$$

状态转移关系为：

$$
\textsf{CREATED} \xrightarrow{\text{appear}} \textsf{ACTIVE} \xrightarrow{\text{timeout}} \textsf{STALE} \xrightarrow{\text{forget}} \textsf{INACTIVE}
\tag{3.5}
$$

具体语义：

- **CREATED**：实体首次在场景出现（生命周期管理器在帧 $t$ 标记其诞生）；
- **ACTIVE**：实体在最近若干帧持续可见，可参与增量更新与下游推理；
- **STALE**：实体连续 $N_{\text{stale}}$ 帧未被观测到，进入"暂忘"状态，仍保留在图中以备恢复；
- **INACTIVE**：实体连续 $N_{\text{forget}}$ 帧（$N_{\text{forget}} > N_{\text{stale}}$）仍未见，关闭其所有属性版本。

该模型对 3.6 节"动态更新机制"至关重要：增量更新引擎 `IncrementalEngine` 在每帧根据当前帧 ID 集合与上一帧 ID 集合之差，自动驱动状态机的转移。生命周期还跟踪每个属性的版本链 $\{(\text{value}_k,\ \text{valid\_from}_k,\ \text{valid\_to}_k)\}_{k=1}^{K}$，满足公理 $A_3$ 的"属性版本化"。

## 3.1.4 命名空间与全局唯一标识

为保证图谱在多场景、多地图、多次运行间不发生 ID 冲突，本文设计了**分层命名空间机制**。所有实体 ID 由前缀 + 业务主键合成：

| 前缀 | 实体类型 | ID 模板 | 示例 |
|------|---------|---------|------|
| `veh_` | 车辆 | `veh_<actor_id>` | `veh_123` |
| `ped_` | 行人 | `ped_<actor_id>` | `ped_42` |
| `tl_` | 信号灯 | `tl_<actor_id>` | `tl_5` |
| `road_` | 道路元素 | `road_<road_id>_lane_<lane_id>` | `road_3_lane_2` |
| `man_` | 行为节点 | `man_<veh_id>_<frame_start>` | `man_veh_123_2048` |
| `int_` | 交互节点 | `int_<src>_<dst>_<type>_<frame>` | `int_veh_123_veh_456_following_2048` |
| `sv_` | 违规节点 | `sv_<rule_code>_<frame>` | `sv_R13a_2052` |
| `resp_` | 责任节点 | `resp_<sv_id>_<actor_id>` | `resp_sv_R13a_2052_veh_123` |

`IDGenerator` 单例（`GLOBAL_ID_GENERATOR`）在全局维护已分配 ID 集合，每次创建实体时校验唯一性（满足公理 $A_1$）。该命名空间为 3.7 节流式长时采集中的多 chunk 拼接、多地图交叉验证、Neo4j 图谱合并提供了一致性基础。

## 3.1.5 小结

本节给出了 STKG 的形式化定义 $\mathcal{STKG} = \langle \mathcal{E}, \mathcal{R}, \mathcal{A}, \mathcal{T}, \mathcal{P}, \{G_t\}\rangle$、时态三元组 $\tau$ 的扩展七元组形式、节点-边-三元组三类表达、节点生命周期四状态机与命名空间 ID 规则。这些定义构成后续各节技术表达的"语法层"，确保 3.2 节本体设计、3.3--3.5 节三层构建、3.6 节动态更新具有统一的符号基础。

---

# 3.2 四层本体总体设计

## 3.2.1 设计动机

自动驾驶仿真数据存在三层递进的语义抽象：

| 语义层级 | 内容 | 时间尺度 | 典型问题 |
|---------|------|---------|---------|
| 场景层 | "在哪里" | 单帧（$\sim$50ms） | "A 车在几号车道？""B 行人在车辆前方几米？" |
| 行为层 | "在做什么" | 多帧（$\sim$秒级） | "A 在跟车还是变道？""B 在横穿吗？" |
| 规则层 | "是否合规" | 跨帧（$\sim$事件级） | "A 是否违反安全距离？""谁负责？" |

仅靠单帧空间关系无法回答"行为"问题；仅靠行为无法判断"是否违反交通法规"。同时，仿真数据本质上是一个**时间序列**，每帧之间都存在增量演化、属性版本更新、规则事件反向插入。本文由此提出 **四层本体设计**：场景、行为、规则三层纵向递进，外加横向的**动态更新机制**统一管理时态演化。

```
                  ┌──────────────────────────────────────┐
                  │        横向机制：动态更新              │
横向机制           │   Δg_t = (Δentities, Δattrs,          │
(Dynamic)         │            Δrelations, rule_events)   │
                  └──────────────────────────────────────┘
                                   ▲
                                   │ 时间轴增量更新
                                   │
   ┌───────────────────────────────┼───────────────────────────┐
   │                               │                           │
   ▼                               ▼                           ▼
┌──────────┐                ┌──────────────┐              ┌──────────────┐
│ 场景层    │ ── manifestsAs→│  行为层       │ ── violates→ │  规则层       │
│ Scene    │                │  Behavior    │              │  Rule        │
│ 6 类节点 │                │  2 类节点     │              │  4 类节点     │
│15 种关系 │                │ 13 种关系     │              │ 7 种关系      │
└──────────┘                └──────────────┘              └──────────────┘
   ▲                                                          │
   │                                                          │ supportedByEvidence
   └──────────────────────────────────────────────────────────┘
              证据链反向注入：违规 → 场景/行为证据节点
```

**图 3-1** 四层本体总体架构：纵向"场景-行为-规则"递进 + 横向"动态更新"机制

## 3.2.2 实体与关系类型总览

依据 `stk/ontology/types.py` 的形式化定义，本文设计 14 种实体类型与 4 大类共 42 种关系类型。**表 3-1** 给出实体类型清单：

**表 3-1** 实体类型与所属层级

| 实体类型 | 标签 | 所属层级 | 角色 |
|---------|------|---------|------|
| `VEHICLE` | `Vehicle` | 场景层 | 动态主体 |
| `PEDESTRIAN` | `Pedestrian` | 场景层 | 弱势主体 |
| `TRAFFIC_LIGHT` | `TrafficLight` | 场景层 | 控制设备 |
| `LANE` | `Lane` | 场景层 | 道路结构单元 |
| `ROAD` | `Road` | 场景层 | 道路结构单元 |
| `JUNCTION` | `Junction` | 场景层 | 道路结构单元 |
| `ENV_SNAPSHOT` | `EnvSnapshot` | 场景层 | 帧环境状态 |
| `SCENE_SNAPSHOT` | `SceneSnapshot` | 场景层 | 帧聚合根 |
| `MANEUVER` | `Maneuver` | 行为层 | 单实体持续行为 |
| `INTERACTION_EVENT` | `Interaction` | 行为层 | 多实体交互 |
| `RULE_DEFINITION` | `Rule` | 规则层 | 规则定义 |
| `RULE_PARAMETER` | `Param` | 规则层 | 规则参数 |
| `SAFETY_VIOLATION` | `SafetyViolation` | 规则层 | 违规事件 |
| `RESPONSIBILITY_ASSIGNMENT` | `Responsibility` | 规则层 | 责任归因 |

**表 3-2** 关系类型清单（按四大类分组）

| 类别 | 关系数 | 典型关系 |
|------|------|---------|
| 场景层关系 `SceneRelationType` | 15 | `in_lane`, `on_road`, `in_junction`, `adjacent_lane`, `lane_connects`, `ahead_of`, `beside`, `nearby_pedestrian`, `controlled_by`, `containsVehicle`, `containsPedestrian`, `containsTrafficLight`, `containsRoad`, `hasEnvironment`, `weather_context` |
| 行为层关系 `BehaviorRelationType` | 13 | `standing_still`, `changing_lane`, `following`, `approaching`, `yielding_to`, `overtaking`, `wrong_side_meeting`, `opposite_direction`, `same_direction`, `blocked_view`, `approaching_pedestrian`, `approaching_intersection`, `crossing` |
| 规则层关系 `RuleRelationType` | 7 | `definedBy`, `usesParam`, `supportedByEvidence`, `violates`, `triggers`, `responsibleFor`, `causedBy` |
| 跨层桥接关系 `CrossLayerRelationType` | 7 | `manifestsAs`, `actor`, `src`, `dst`, `hasVersion`, `has_maneuver`, `has_interaction` |
| **合计** | **42** | — |

跨层桥接关系是连接相邻纵向层的"针灸线"。其中 `manifestsAs` 把行为节点（Maneuver/Interaction）与对应的行为关系边绑定，实现"节点+边双轨表达"——同一行为可同时通过节点参与图遍历、通过边参与拓扑查询；`actor/src/dst` 则将行为节点与其主体实体、相互作用的客体实体连接起来。

## 3.2.3 节点+边双轨表达原则

本节阐述贯穿三层本体的核心设计原则——**节点+边双轨表达**（dual-track representation）。其本质是：同一语义对象在图中既以节点形式存在，又以关系形式存在，二者并存且语义一致。

以行为层为例：检测器 `detect_following(v_A, v_B)` 输出一条关系 `following(v_A, v_B, t)`，但同时 `BehaviorRelationGenerator` 还会创建一个 `InteractionEvent` 节点 `int_<A>_<B>_following_<t>`。该节点既包含行为属性（duration、severity、related_rule 等），又通过 `manifestsAs` 边与 following 关系绑定，再通过 `actor/src/dst` 边分别与 $v_A$ 和 $v_B$ 连接。如此设计带来三重收益：

1. **图遍历灵活性**：图查询可以从 $v_A$ 出发沿 `following` 边直达 $v_B$（短路径），也可经 `actor` 边到 `InteractionEvent` 节点再经 `dst` 边到 $v_B$（含行为属性的长路径）；
2. **属性承载能力**：关系的 `attrs` 可承载"边属性"，节点的 `attrs` 可承载"对象属性"——例如 `following` 关系边可携带 `distance`、`relative_speed`、`ttc`，对应 `InteractionEvent` 节点可额外携带 `state`、`severity`、`related_rule`、`source_relations` 等，双方分工明确；
3. **跨层桥接友好**：规则层 `SafetyViolation` 节点通过 `supportedByEvidence` 边反向指回行为节点 / 场景节点，证据链天然指向"某个具体 InteractionEvent"或"某条场景关系"，与单一边表达相比可解释性更强。

该原则在 3.3 节场景层、3.4 节行为层、3.5 节规则层中均有体现。

## 3.2.4 公理体系

为统一约束三层本体的合法状态，本文设计 **七条核心公理** $A_1$--$A_7$。它们既是设计意图的明确化，也是后续单元测试与运行时一致性校验的依据。

**公理 $A_1$（实体 ID 唯一性）**  
每个实体 ID 在全图空间内全局唯一，非空且不含空格类字符。`IDGenerator` 单例确保跨场景、跨地图时不重复。

**公理 $A_2$（实体类型固定）**  
每个实体的 `entity_type` 字段一经 instantiation 即固定，且必须属于 `EntityType` 枚举。这保证类型系统在后续查询和规则推理中无歧义。

**公理 $A_3$（属性版本化）**  
任意实体的任意属性 $a$ 在时间轴上的取值序列 $\{(\text{val}_k,\ t_{s,k},\ t_{e,k})\}$ 必须可查询，且版本区间两两不相交。该公理由 `NodeLifecycle.add_version()` 与 `VersionManager` 协同保障（详见 3.6.3 节）。

**公理 $A_4$（关系必有时态）**  
每条关系边必须携带非空的 `valid_from` 字段（缺省取 `frame_id`）；周期性关系还需带 `valid_to`。不接受"无时间戳的边"。

**公理 $A_5$（三层证据约束）**  
规则层任何节点（如 `SafetyViolation`）必须通过 `supportedByEvidence` 边连接到至少一个场景层或行为层证据节点。该公理保证"违规"一定能追溯到原始观测，满足可解释性要求。

**公理 $A_6$（事件可追溯性）**  
对任意 `SafetyViolation` 节点 $v$，存在至少一条由 $k \geq 1$ 条 `supportedByEvidence` 边组成的路径，终点为场景层实体节点或行为层 Maneuver/Interaction 节点。$A_5$ 是 $A_6$ 的弱形式，$A_6$ 是对完整证据链长度的要求。

**公理 $A_7$（增量一致性）**  
相邻两帧的图状态满足 $G_t = G_{t-1} \oplus \Delta g_t$，且 $\Delta g_t$ 不在已被 $A_3$ 版本化的属性上推出"撤销版本"。换言之，增量只录入"新版本"，不删除旧版本——实体在消失时进入 `INACTIVE` 状态而非从图上抹除。

七条公理在 `stk/ontology/axioms.py` 中以独立函数实现，并在 `tests/test_ontology.py` 中通过 31 个 case 全数覆盖。运行时若任一公理被违反，将触发异常并阻止该帧写入 Neo4j。

## 3.2.5 四层结构的语义分工

四层本体并非简单堆叠，而是有明确的语义分工，下面以纵向"递进抽象"+ 横向"动态支撑"两条线总结：

**纵向递进抽象**（场景 → 行为 → 规则）：

| 转换 | 数据流 | 信息增益 |
|------|------|---------|
| 场景层 → 行为层 | 计算 spatial relation、防抖、Maneuver/Interaction 节点生成 | 引入"做什么"的语义 |
| 行为层 → 规则层 | 规则匹配、RSS 安全距离计算、责任归因 | 引入"是否合规"的判断与责任归属 |

**横向动态支撑**（动态更新机制）：

| 支撑点 | 机制 | 详见 |
|--------|------|------|
| 实体级 | DiffSet 增删保持 + 生命周期驱动 | 3.6.1 节 |
| 属性级 | AttrVersion 版本化 + 阈值过滤 | 3.6.3 节 |
| 关系级 | 增删保持 + 防抖状态机接管 | 3.6.1 节 + 3.4.4 节 |
| 规则事件 | 反向插入 + 证据链生成 | 3.6.5 节 |

如此设计的核心好处是：**每一层的输出都既是下一层的输入，也是规则层证据链的源头**。3.2.6 节给出完整的数据流示意。

## 3.2.6 数据流总览

```
[CARLA Tick t]
       │
       ▼
[extraction/ 6 类提取器]
       │        vehicles[], pedestrians[], traffic_lights[],
       │        lanes[], weather{}, traffic_density
       ▼
[scenario/snapshot_builder.build_snapshot]
       │        ScenarioSnapshot(t) + EnvSnapshot(t) + 6 类实体
       │        + 4 类 containsX 关系 + hasEnvironment
       ▼
[scenario/spatial.compute_*]
       │        in_lane × N, ahead_of × M, beside × K,
       │        nearby_pedestrian × L, controlled_by × P, ...
       ▼
[behavior/BehaviorRelationGenerator.generate]
       │        Maneuver[] + Interaction[] + 13 种行为关系
       │        + 跨层 manifestsAs / actor / src / dst
       ▼
[rules/RuleEnforcer.enforce]
       │        RSS 违规、R1-R18 交规违规、SafetyViolation[]、
       │        ResponsibilityAssignment[]、违反规则边、证据链
       ▼
[storage/serializer.serialize_graph]
       │        4 层节点 + 4 类边 → Neo4j MERGE 批量语句
       │        或 JSON 序列化导出
       ▼
[Neo4j / JSON 文件]
       │
       ▼
[downstream]   ──  3D 可视化 Dashboard (viz/)
              ──  GNN 异常检测导出 (storage/queries.export_for_gnn_cypher)
              ──  规则验证与回放 (storage/replay.replay_violation)
```

**图 3-2** STKG 单帧数据流：6 类提取器 → 4 层本体构建 → 存储 → 下游应用

横向的动态更新机制则贯穿以上每个阶段，分别对实体、属性、关系、规则事件四个层面进行增量管理。3.6 节将详细展开。

## 3.2.7 小结

本节给出 STKG 的本体总体设计：以 14 种实体类型和 4 大类 42 种关系类型为骨架，遵循"节点+边双轨表达"原则与七条核心公理 $A_1$--$A_7$ 的约束，构成纵向递进（场景-行为-规则）+ 横向支撑（动态更新）的四层结构。该本体为后续 3.3 节场景层提取、3.4 节行为层防抖检测、3.5 节规则层 RSS 与交规推理提供了清晰、可形式化、可测试的设计基础。
