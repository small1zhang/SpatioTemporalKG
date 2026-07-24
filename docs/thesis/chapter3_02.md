# 3.2 四层本体总体设计

在 3.1 节形式化定义的基础上，本节给出 STKG 的本体总体架构。该架构以"纵向三层递进 + 横向动态支撑"为核心组织原则：纵向由场景层、行为层、规则层构成语义抽象递进链路，横向由动态更新机制贯穿时间维度。本节聚焦于本体设计与公理约束，详细的三层构建方法在 3.3 节展开。

## 3.2.1 设计动机与总体架构

自动驾驶仿真数据存在三层递进的语义抽象：场景层回答"在哪里"，时间尺度约 50 ms；行为层回答"在做什么"，时间尺度约为秒级；规则层回答"是否合规"，时间尺度跨越整个事件期。仅靠单帧空间关系无法回答行为问题；仅靠行为无法判断是否违反交通法规。三层之间存在明确的依赖链：行为层依赖场景层的空间关系作为输入（如检测 `following` 需要 `in_lane` 与 `ahead_of`），规则层依赖行为层与场景层的双重输入（如 R1 行人优先既需要场景层的行人在横道线、车辆距离，也需要行为层的车辆减速行为）。这种"上层依赖下层、下层不依赖上层"的递进结构是分层本体得以成立的逻辑基础。

若将三层分别独立建模为三个图谱，则会引入跨图谱引用难题：行为层的 `InteractionEvent` 通过何种机制定位到场景层的具体车辆 $v_A$？规则层的 `SafetyViolation` 又如何追溯到触发该违规的具体空间关系？若仅以"实体 ID"做软引用而不在图上显式表达，跨层路径需经多次外部查找拼接而成，可解释性大打折扣。与之相比，本文采用**单图谱三层共存**的方案：通过跨层桥接关系（`manifestsAs`、`actor`、`src`、`dst`、`supportedByEvidence`）在图上显式建立跨层路径，使得任意违规节点可经图遍历直接到达原始场景事实，无须外部查找。

同时，仿真数据本质上是一个**时间序列**，每帧之间存在增量演化、属性版本更新、规则事件反向插入。本文由此提出 **四层本体设计**：场景、行为、规则三层纵向递进，外加横向的**动态更新机制**统一管理时态演化。整体架构如图 3-1 所示。

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

纵向箭头体现了知识的三阶段升华：场景层实体经 `manifestsAs` 桥接边生成行为节点，行为关系经 `violates` 桥接边判定为规则违规，规则层的 `SafetyViolation` 节点再经 `supportedByEvidence` 边反向指回场景/行为证据，形成完整的语义闭环。横向的动态更新机制则同时管理实体、属性、关系、规则事件四个层面的时态演化。

需要指出的是，"四层本体"在本体论意义上并非四类并列的实体集合，而是"三类实体集合（场景、行为、规则）+ 一类横向机制（动态更新）"的复合结构。横向机制不引入新的实体类型，而是为前述三类实体提供时态管理能力——所有实体均通过生命周期状态机与属性版本链与时间轴关联。这种"实体三类 + 横向机制一类"的设计避免了实体类型的过度膨胀，同时保证了时态管理的统一性。

## 3.2.2 实体与关系类型总览

依据 `stk/ontology/types.py` 的形式化定义，本文设计 **14 种实体类型**与 **4 大类共 42 种关系类型**。表 3-5 给出实体类型清单，表 3-6 给出关系类型清单（含源→目标节点约束）。

**表 3-5** 实体类型与所属层级

[三线表]

| 实体类型 | Neo4j 标签 | 所属层级 | 角色 |
|---------|-----------|---------|------|
| `VEHICLE` | `Vehicle` | 场景层 | 动态主体 |
| `PEDESTRIAN` | `Pedestrian` | 场景层 | 弱势主体 |
| `TRAFFIC_LIGHT` | `TrafficLight` | 场景层 | 控制设备 |
| `LANE`/`ROAD`/`JUNCTION` | `Lane`/`Road`/`Junction` | 场景层 | 道路结构单元 |
| `ENV_SNAPSHOT` | `EnvSnapshot` | 场景层 | 帧环境状态 |
| `SCENE_SNAPSHOT` | `SceneSnapshot` | 场景层 | 帧聚合根 |
| `MANEUVER` | `Maneuver` | 行为层 | 单实体持续行为 |
| `INTERACTION_EVENT` | `Interaction` | 行为层 | 多实体交互 |
| `RULE_DEFINITION` | `Rule` | 规则层 | 规则定义 |
| `RULE_PARAMETER` | `Param` | 规则层 | 规则参数 |
| `SAFETY_VIOLATION` | `SafetyViolation` | 规则层 | 违规事件 |
| `RESPONSIBILITY_ASSIGNMENT` | `Responsibility` | 规则层 | 责任归因 |

**表 3-6** 关系类型清单（按四大类分组，含源→目标节点约束）

[三线表]

| 类别 | 关系名 | 源节点 → 目标节点 | 用途 |
|------|--------|------------------|------|
| 场景层 `SceneRelationType`（15 种） | `in_lane` | `Vehicle` → `Lane` | 车辆所在车道 |
| | `on_road` | `Vehicle` → `Road` | 车辆所在路段 |
| | `in_junction` | `Vehicle` → `Junction` | 车辆所在路口 |
| | `adjacent_lane` | `Lane` → `Lane` | 车道邻接 |
| | `lane_connects` | `Lane` → `Lane` | 车道拓扑连接 |
| | `ahead_of` | `Vehicle` → `Vehicle` | 同车道纵向前方 |
| | `beside` | `Vehicle` → `Vehicle` | 横向并排 |
| | `nearby_pedestrian` | `Vehicle` → `Pedestrian` | 行人邻近 |
| | `controlled_by` | `Lane` → `TrafficLight` | 灯控车道 |
| | `containsVehicle`/`containsPedestrian`/`containsTrafficLight`/`containsRoad` | `SceneSnapshot` → 实体 | 帧聚合包含 |
| | `hasEnvironment` | `SceneSnapshot` → `EnvSnapshot` | 帧环境上下文 |
| | `weather_context` | `EnvSnapshot` → `SceneSnapshot` | 全局语境 |
| 行为层 `BehaviorRelationType`（13 种） | `standing_still`/`changing_lane` | `Vehicle` → `Vehicle` (self) | 单实体持续状态 |
| | `following`/`approaching`/`overtaking`/`yielding_to`/`wrong_side_meeting`/`opposite_direction`/`same_direction`/`blocked_view` | `Vehicle` → `Vehicle` | 双车交互 |
| | `approaching_pedestrian` | `Vehicle` → `Pedestrian` | 车-行人交互 |
| | `approaching_intersection` | `Vehicle` → `Junction` | 车-路口交互 |
| | `crossing` | `Pedestrian` → `Vehicle` (路径冲突) | 行人横穿 |
| 规则层 `RuleRelationType`（7 种） | `defined_by` | `SafetyViolation` → `Rule` | 违规归属规则 |
| | `uses_param` | `Rule` → `Param` | 规则使用参数 |
| | `supportedByEvidence` | `SafetyViolation` → 场景/行为证据 | 证据链 |
| | `violates` | 违规源实体 → 违规目标实体 | 违规边 |
| | `triggers`/`causedBy` | `SafetyViolation` ↔ `SafetyViolation` | 因果链 |
| | `responsibleFor` | `Responsibility` → `SafetyViolation` | 责任归属 |
| 跨层桥接 `CrossLayerRelationType`（7 种） | `manifestsAs` | `Maneuver`/`Interaction` ↔ 行为边 | 节点+边双轨 |
| | `actor` | `Maneuver` → `Vehicle` | 主体引用 |
| | `src`/`dst` | `Interaction` → 实体 | 交互主客体 |
| | `hasVersion` | 实体 → `AttrVersion` | 属性版本指针 |
| | `has_maneuver`/`has_interaction` | SceneSnapshot → 行为节点 | 帧级聚合 |

跨层桥接关系是连接相邻纵向层的"针灸线"。`manifestsAs` 把行为节点（Maneuver/Interaction）与对应的行为关系边绑定，实现"节点+边双轨表达"——同一行为可同时通过节点参与图遍历、通过边参与拓扑查询；`actor/src/dst` 则将行为节点与其主体实体、相互作用的客体实体连接起来。

## 3.2.3 节点+边双轨表达原则

本节阐述贯穿三层本体的核心设计原则——**节点+边双轨表达**（dual-track representation）。其本质是：同一语义对象在图中既以节点形式存在，又以关系形式存在，二者并存且语义一致。表 3-7 对比了"仅边""仅节点""节点+边双轨"三种方案的差异。

**表 3-7** 行为表达方案对比

[三线表]

| 方案 | 灵活性 | 属性承载 | 跨层桥接 | 子图检索 | 本文取舍 |
|------|-------|---------|---------|---------|---------|
| 仅边表达 | 弱（边属性单一） | 边 attrs | 弱（边无法被反向引用） | 短路径快 | 否 |
| 仅节点表达 | 弱（需经节点跳转） | 节点 attrs | 强 | 长路径慢 | 否 |
| **双轨表达（选）** | 强 | 节点+边分工 | 强 | 长短路径均可 | **是** |

以行为层为例：检测器 `detect_following(v_A, v_B)` 输出一条关系 `following(v_A, v_B, t)`，但同时 `BehaviorRelationGenerator` 还会创建一个 `InteractionEvent` 节点 `int_<A>_<B>_following_<t>`。该节点既包含行为属性（duration、severity、related_rule 等），又通过 `manifestsAs` 边与 following 关系绑定，再通过 `actor/src/dst` 边分别与 $v_A$ 和 $v_B$ 连接。如此设计带来三重收益：

1. **图遍历灵活性**：图查询可从 $v_A$ 出发沿 `following` 边直达 $v_B$（短路径），也可经 `actor` 边到 `InteractionEvent` 节点再经 `dst` 边到 $v_B$（含行为属性的长路径）；
2. **属性承载能力**：关系的 `attrs` 可承载"边属性"，节点的 `attrs` 可承载"对象属性"——例如 `following` 关系边可携带 `distance`、`relative_speed`、`ttc`，对应 `InteractionEvent` 节点可额外携带 `state`、`severity`、`related_rule`、`source_relations` 等，双方分工明确；
3. **跨层桥接友好**：规则层 `SafetyViolation` 节点通过 `supportedByEvidence` 边反向指回行为节点 / 场景节点，证据链天然指向"某个具体 InteractionEvent"或"某条场景关系"，可解释性更强。

双轨表达的底层逻辑在于：图谱中的"边"与"节点"在图论模型中本就是相互对偶的两种表达单元——边擅长表达二元关系，节点擅长承载属性与多跳关联。当语义对象本身既是"两实体间的关系"、又是"具备自身属性的实体"时（如 InteractionEvent 既是"v_A 与 v_B 的跟驰关系"、又是"一个具有持续帧数与严重度的行为事件"），单一表达就显得不足。在工程上，仅以边表达时边属性表容易膨胀且难以作为"反查起点"——典型查询"找出过去 100 帧所有 severity > 0.7 的 InteractionEvent"在纯边模型下需扫描边表全量才能筛选，而双轨表达则可经节点索引直接命中。

双轨表达带来的存储代价是每条行为平均多 1 个节点 + 4 条桥接边（`manifestsAs`、`actor`、`src`、`dst`）。考虑到 Neo4j 在节点与边存储上的均衡代价（同等属性字数时节点存储略优于边），此代价相对收益可接受。后续 3.5 节通过三道正交裁剪（重要性打分、边稀疏化、静态外移）进一步控制图谱规模，使双轨表达在长时运行时仍保持存储友好。

该原则在 3.3 节场景层、行为层、规则层中均有体现，是三层本体的统一语义桥梁。

## 3.2.4 公理体系

为统一约束三层本体的合法状态，本文设计 **七条核心公理** $A_1$–$A_7$。它们既是设计意图的明确化，也是后续单元测试与运行时一致性校验的依据。每条公理均有一个具体的设计动机：它约束的是哪种在实践中已经出现（或可能出现）的"图状态异常"。以下逐条说明其动机与约束。

**公理 $A_1$（实体 ID 唯一性）**
每个实体 ID 在全图空间内全局唯一，非空且不含空格类字符。`IDGenerator` 单例确保跨场景、跨地图时不重复。
> **动机**：在长时仿真（如 24000 帧、20 分钟）中，同一车辆在跨 chunk 恢复、地图切换或 Simulation Reset 后可能被再次创建。若不强制 ID 唯一性，可能出现"同一物理车辆在图谱中存在两个 ID"的歧义，导致规则层证据链断裂——前一个 chunk 中有违规记录的车辆，后一个 chunk 中以新 ID 出现时历史违规信息无法关联。IDGenerator 单例的全局 ID 校验正是为此设计。

**公理 $A_2$（实体类型固定）**
每个实体的 `entity_type` 字段一经 instantiation 即固定，且必须属于 `EntityType` 枚举。这保证类型系统在后续查询和规则推理中无歧义。
> **动机**：若某实体在生命周期内类型可变（如从 Vehicle 变为 Pedestrian），则针对"车辆"的查询与推理将产生语义歧义。类型固定是 RDF 三元组与属性图模型的共同约定，此处显式声明为公理以加强其在代码层面的强制执行。

**公理 $A_3$（属性版本化与区间不相交）**
任意实体的任意属性 $a$ 在时间轴上的取值序列 $\{(\text{val}_k,\ t_{s,k},\ t_{e,k})\}$ 必须可查询，且版本区间两两不相交：
$$
\forall a,\ \forall i \neq j,\ [t_{s,i},\ t_{e,i}] \cap [t_{s,j},\ t_{e,j}] = \emptyset
\tag{3.7}
$$
该公理由 `NodeLifecycle.add_version()` 与 `VersionManager` 协同保障（详见 3.4.3 节）。
> **动机**：区间不相交约束是为了防止因并发写入或重采导致同一属性的两个版本在时间上重叠——例如因 checkpoint 恢复时的帧错位，同一帧被处理两次导致 speed 属性的两个版本都具有 valid_from=1024。这种重叠在时间旅行查询中会导致歧义（`get_version_at(attr, t)` 应返回哪个版本？）。设置区间不相交后，查询可通过单个比较操作确定当前帧的属性版本，而不需要处理版本优先级规则。

**公理 $A_4$（关系必有时态）**
每条关系边必须携带非空的 `valid_from` 字段（缺省取 `frame_id`）；周期性关系还需带 `valid_to`。不接受"无时间戳的边"。
> **动机**：在仿真场景中，所有事实本质上是时态的——"车辆 A 在车道 L 上"仅在特定帧成立，离开该帧后关系可能不再保持。如果允许"无时间戳边"（如将 `in_lane` 关系中 `valid_from` 设为 NULL），则跨帧查询时无法判断该关系在目标帧是否有效。此公理保证了所有场景/行为/规则关系的时态可查询性。

**公理 $A_5$（证据链至少一项）**
规则层任何节点（如 `SafetyViolation`）必须通过 `supportedByEvidence` 边连接到至少一个场景层或行为层证据节点。即：
$$
\forall v \in \text{SafetyViolation},\ \exists e \in \text{Scene} \cup \text{Behavior},\ \text{supportedByEvidence}(v, e)
\tag{3.8}
$$
该公理保证"违规"一定能追溯到原始观测，是最小可解释性要求。
> **动机**：在实践中曾遇到因规则引擎内部逻辑错误导致 `SafetyViolation` 节点被创建后无任何证据链——该违规成为"孤儿节点"。下游解释模块在尝试追溯原因时发现 `MATCH (sv)-[:supportedByEvidence]->(e)` 返回空，无法提供任何解释。A5 将此情形定义为违反公理，从工程上强制规则引擎必须为每个 `SafetyViolation` 填充 evidence_path。

**公理 $A_6$（证据链完整路径可达）**
对任意 `SafetyViolation` 节点 $v$，存在至少一条由 $k \geq 1$ 条 `supportedByEvidence` 边组成的路径，终点为场景层实体节点或行为层 Maneuver/Interaction 节点：
$$
\forall v,\ \exists \pi = (v, e_1, e_2, \dots, e_k),\ |\pi| \geq 1,\ e_k \in \text{SceneEntity} \cup \text{BehaviorNode}
\tag{3.9}
$$
$A_5$ 与 $A_6$ 的区别：$A_5$ 仅要求"至少连接一个证据"，$A_6$ 要求"证据路径可经图遍历完整到达原始观测"。$A_6$ 是 $A_5$ 的强化形式，二者均由 `tests/test_ontology.py` 中的 `axiom_A5` 与 `axiom_A6` 独立验证。
> **动机**：在实践中存在这样一种情况——`SafetyViolation` 的 evidence_path 包含了证据 ID，但这些 ID 对应的节点已被垃圾回收或未创建，导致虽然形式上满足 A5（边存在），但路径的终点是一个空节点，无法追溯有效信息。A6 要求路径的终点必须是一个图遍历可到达的"活"实体节点，从根源上堵住了"断头路"的异常。

**公理 $A_7$（增量一致性）**
相邻两帧的图状态满足
$$
G_t = G_{t-1} \oplus \Delta g_t,\qquad \Delta g_t.\text{cancel}(\cdot) = \emptyset
\tag{3.10}
$$
即 $\Delta g_t$ 不在已被 $A_3$ 版本化的属性上推出"撤销版本"。换言之，增量只录入"新版本"，不删除旧版本——实体在消失时进入 `INACTIVE` 状态而非从图上抹除。
> **动机**：如果在增量更新中删除了旧版本的属性记录，则时间旅行查询在访问删除帧时将返回 0 条记录，A3 的公理无法再得到满足。此公理将 A3 的"版本可查询"要求延伸到了增量更新过程——任何增量更新都不应破坏历史版本的可追溯性。

七条公理在 `stk/ontology/axioms.py` 中以独立函数实现，并在 `tests/test_ontology.py` 中通过 31 个 case 全数覆盖。运行时若任一公理被违反，将触发异常并阻止该帧写入 Neo4j，保证图谱的语义一致性。

## 3.2.5 与现有本体的借鉴关系

STKG 的本体设计借鉴了多种现有知识图谱的合理成分。表 3-8 列出主要借鉴对象与本文改进方向。

**表 3-8** STKG 与现有本体借鉴对比

[三线表]

| 现有工作 | 借鉴点 | 本文改进 |
|---------|-------|---------|
| nuScenes KG [Cesari, 2023] | 实体类型分层、属性字段定义 | 引入时态三元组与属性版本化，从离线转为增量 |
| roadscene2vec [Venkatraman, 2021] | 单帧场景图抽取流程 | 引入跨帧行为层与防抖状态机 |
| CoSI 本体 [Bagschik, 2018] | 驾驶场景概念分类 | 引入 RSS 物理安全子层，与场景知识形成互补 |
| Dynamic KGs 综述 [Cao, 2023] | 流式摄取、时间戳事件、快照批量更新策略对比 | 采用事件驱动 + 增量合并模式，避免完整图重建 |
| 电力通信领域 KG [Wang, 2022] | 基于时序感知的滑动窗口增量更新算法 | 滑动窗口用于行为层模式检测与高频事实管理 |
| KG-as-World-Models [Huang, 2025] | 知识图谱作为语义世界模型的可能性 | 引入规则层事件作为图谱自身动态更新的一部分 |

在以上借鉴基础上，本文的核心创新点有三：① 以"节点+边双轨表达"统一三层本体，比现有"仅节点"或"仅边"方案具备更强的图遍历灵活性；② 以"四状态生命周期 + 属性版本化"解决仿真中实体频繁进出场景与属性演进的双重时态管理需求；③ 以"RSS 强先验 + GNN 统计建模"实现规则驱动与数据驱动的融合（详见 3.3.3 节末尾前瞻讨论）。

需要进一步指出的是：表 3-8 的借鉴关系并不暗示本文是对现有工作简单的"集成式合并"。每一项借鉴在 STKG 中均经历了从"原始方法"到"STKG 场景"的适配改造。例如，nuScenes KG 的实体分层被保留但引入了时态三元组与属性版本化，从离线静态结构改造为在线增量结构；roadscene2vec 的逐帧抽取流程被保留但摒弃了帧间重建策略，将防抖状态机与周期关系更新嵌入行为生成器中。在借鉴对象的所有结合点上，增量更新、时态管理与防抖机制是最主要的适配性贡献——这三个设计不是为了"有特色"而加入，而是为了应对"20 Hz 仿真帧率 × 24000 帧长时运行"这一特定工程约束而不得不解决的问题。

## 3.2.6 数据流总览

四层本体设计在一帧仿真 tick 内的数据流如图 3-2 所示。

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

横向的动态更新机制贯穿以上每个阶段，分别对实体、属性、关系、规则事件四个层面进行增量管理。3.4 节将详细展开。

## 3.2.7 小结

本节给出 STKG 的本体总体设计：以 14 种实体类型和 4 大类 42 种关系类型为骨架，遵循"节点+边双轨表达"原则与七条核心公理 $A_1$–$A_7$ 的约束，构成纵向递进（场景-行为-规则）+ 横向支撑（动态更新）的四层结构。该本体为后续 3.3 节三层构建、3.4 节动态更新、3.5 节流式采集提供了清晰、可形式化、可测试的设计基础。
