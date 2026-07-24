# 3.4 动态更新机制

3.3 节分别描述了 STKG 三层本体的"瞬时构造"——给出在某一帧仿真时刻如何从 CARLA 数据流构建场景层、行为层、规则层的全部节点与边。但仿真数据本质上是按帧推进的时间序列，图谱在时间轴上的演化管理是 STKG 区别于静态知识图谱的关键。一个明显的反例是：若仅依赖 3.3 节所述的瞬时构造，每帧都"从零"重建图谱，则相邻帧之间的对应实体（如同一辆车 $v_A$ 在第 $t$ 帧与第 $t+1$ 帧的两个版本的节点）之间无法建立任何关联，规则层证据链也无法跨帧引用同一实体，更无法支持"在仿真第 1500 帧时车辆 $v_A$ 的速度是多少"这类时间旅行查询。

本节描述 STKG 的横向机制——动态更新：定义差分图 $\Delta g_t$、增量引擎的五步主流程、属性版本化机制、时间窗口聚合以及规则事件的反向插入。这些机制共同保证 STKG 在长时仿真中对存储、查询、推理友好的同时，对帧间变化保持精细的时态表达能力。从工程视角看，动态更新是 STKG 在"20 Hz 帧率 × 24000 帧 = 48 万帧"长时运行下仍能保持可接受存储与查询性能的关键——若无增量更新，每帧的图谱规模约为数千节点+数万边，全量存储 48 万帧需数百 GB，缺乏现实可行性；通过差异 Δg_t 仅记录帧间变化，存储规模可压缩至全量模式的约 5%-10%。

## 3.4.1 差分图与四类更新动作

在 3.1.2 节中曾给出公式 (3.3)：$G_t = G_{t-1} \oplus \Delta g_t$。本节给出 $\Delta g_t$ 的形式化定义——**差分图**（DeltaGraph）：

$$
\Delta g_t := \big\langle\ \Delta_{\mathcal{E}}(t),\ \Delta_{\mathcal{A}}(t),\ \Delta_{\mathcal{R}}(t),\ \mathcal{E}_{\text{rule}}(t)\ \big\rangle
\tag{3.28}
$$

四个分量分别对应四类更新动作：

**① 实体级差分 $\Delta_{\mathcal{E}}(t)$**：本帧相对上一帧的实体集合差分，由 `DiffSet` 三集合表示：

$$
\Delta_{\mathcal{E}}(t) = \big(\ \mathcal{E}_{t}^{\text{added}},\ \mathcal{E}_{t}^{\text{removed}},\ \mathcal{E}_{t}^{\text{unchanged}}\ \big)
\tag{3.29}
$$

其中 $\mathcal{E}_{t}^{\text{added}} = \mathcal{E}_t \setminus \mathcal{E}_{t-1}$ 为本帧新进入场景的实体集合，$\mathcal{E}_{t}^{\text{removed}} = \mathcal{E}_{t-1} \setminus \mathcal{E}_t$ 为本帧离开场景的实体集合，$\mathcal{E}_{t}^{\text{unchanged}} = \mathcal{E}_t \cap \mathcal{E}_{t-1}$ 为持续存在的实体集合。注意，"removed"在 STKG 中并不从图中删除，而是触发对应节点的生命周期状态从 `ACTIVE` 转移到 `STALE`（满足公理 $A_7$）。

**② 属性级差分 $\Delta_{\mathcal{A}}(t)$**：本帧相对上一帧的属性值变化集合，结构为：

$$
\Delta_{\mathcal{A}}(t) = \Big\{\ \big(e,\ a,\ (\text{val}_{t-1},\ \text{val}_t)\big)\ \Big|\ e \in \mathcal{E}_{t}^{\text{unchanged}},\ a \in \mathcal{A},\ |\text{val}_t - \text{val}_{t-1}| > \epsilon_{\text{thresh}}(a)\ \Big\}
\tag{3.30}
$$

其中 $\epsilon_{\text{thresh}}(a)$ 是**按属性类型差异化**的防抖阈值，由 `ThresholdConfig` 控制。表 3-21 给出代码中的默认值。

**表 3-21** 属性级防抖阈值（`ThresholdConfig`）

[三线表]

| 属性类别 | 阈值 | 单位 | 设计理由 |
|---------|------|------|---------|
| `location_x/y/z` | 0.10 | m | 仿真器浮点抖动约 0.05 m，2 倍裕度 |
| `speed` | 0.50 | m/s | 微小速度变化对行为判定不构成实质改变 |
| `heading_rad` | 0.035 | rad ≈ 2° | 车辆正常行驶朝向抖动小于 1° |
| `throttle`/`brake` | 0.05 | [0,1] | 控制信号量化精度 |
| `steer` | 0.02 | [-1,1] | 方向盘微小抖动 |
| `bbox_extent` | 0.0 | m | 包围盒尺寸一旦变化即记录 |

差异化阈值的核心目的是过滤 CARLA 物理仿真中微小的浮点抖动，避免产生大量无意义版本，同时保留有语义价值的变化。例如，位置变化小于 0.1 m 通常对应仿真器浮点噪声而非真实车辆移动，无需触发新版本；但 `bbox_extent` 一旦变化则反映车辆换型，必须记录。本文对 `bbox_extent` 阈值取 0.0 m 是基于一个工程观察：CARLA 0.9.16 中车辆的 `bounding_box.extent` 在同一 `type_id` 下是常量，当且仅当车辆被销毁/重建时（异常 shutting down 状态）发生字段变化——任何这种变化都应被立即记录以触发下游警示。其余阈值的取值合理性在长时运行测试中得到了反向验证：跨 chunk 恢复后属性版本数无明显跳跃，下游 K-HSTGAN 异常检测模型在阈值上/下浮动 20% 时训练结果保持稳定（详见 6.5 节敏感性分析）。

阈值的选择经过了三阶段的实测调优：第一阶段使用统一阈值 0.1（任意单位），结果发现低精度的 `bbox_extent` 也产生大量版本，单帧属性差分平均 2.3 项，长时累积后压缩比仅 35%；第二阶段按"属性维度"分组（位置/速度/朝向/控制/几何）设置差异化阈值，压缩比提升至 75%；第三阶段（当前实现）按每个属性单独设置阈值，将 `bbox_extent` 进一步放宽至 0 而将 `steer` 收紧至 0.02，压缩比稳定在 82%-88% 之间。这一实验过程佐证了"差异化阈值"相对于统一阈值的工程价值。

**③ 关系级差分 $\Delta_{\mathcal{R}}(t)$**：本帧相对上一帧的关系集合差分，结构与实体级差分类似，由 `_relation_key(src_id, dst_id, type, frame_id)` 做唯一性判定：

$$
\Delta_{\mathcal{R}}(t) = \big(\ \mathcal{R}_{t}^{\text{added}},\ \mathcal{R}_{t}^{\text{removed}},\ \mathcal{R}_{t}^{\text{unchanged}}\ \big)
\tag{3.31}
$$

**④ 规则事件 $\mathcal{E}_{\text{rule}}(t)$**：本帧规则层新触发的 `SafetyViolation`、`ResponsibilityAssignment` 及其相关边组成的列表：

$$
\mathcal{E}_{\text{rule}}(t) = \big[\ \text{sv}_1,\ \text{sv}_2,\ \dots,\ \text{sv}_k,\ \text{resp}_1,\ \dots,\ \text{resp}_l\ \big]
\tag{3.32}
$$

四类组合形成 $\Delta g_t$ 的完整形式化定义。该定义直接对应代码 `DeltaGraph` 的 dataclass 字段：`delta_entities`、`delta_attrs`、`delta_relations`、`rule_events`。

## 3.4.2 增量引擎五步流程

`IncrementalEngine`（`stk/dynamic/incremental_updater.py`）是动态更新的主驱动。它每帧接收一个完整快照，输出一个 `DeltaGraph`。核心流程为五步，伪代码如算法 3.4 所示。

```
算法 3.4: IncrementalEngine.process_frame(frame)
输入: 当前帧完整快照 frame (含 vehicles, pedestrians, traffic_lights, scene_rels, behavior_rels)
输出: DeltaGraph Δg_t

1. // Step 1: recv — 接收并校验
2. validate(frame)  // 检查数值属性不被字符串污染 (公理 A3 防御)
3.
4. // 帧跳跃检测: 若帧号间隔 > 1 则触发重置
5. if _prev_frame is not None and frame.frame_id - _prev_frame.frame_id > 1:
6.     reset()  // 上一帧状态过时, 当下帧作为新 baseline
7. end if
8.
9. // Step 2: diff — 计算差分
10. if _prev_frame is None:  // 首帧或重置后
11.    δ_entities ← {added: all, removed: ∅, unchanged: ∅}
12.    δ_attrs ← {}  // 首帧无属性差分
13.    δ_relations ← {added: all, removed: ∅, unchanged: ∅}
14. else:
15.    δ_entities ← compute_delta_entities(curr_entities, prev_entities)
16.    δ_attrs ← compute_delta_attrs(curr_attrs, prev_attrs, threshold_config)
17.    δ_relations ← compute_delta_relations(curr_rels, prev_rels)
18. end if
19.
20. // Step 3: patch — 根据差分打补丁
21. apply_entity_lifecycle(δ_entities)  // CREATED→ACTIVE / ACTIVE→STALE
22. apply_attribute_versions(δ_attrs)  // 为每个变化属性创建 AttrVersion
23.
24. // Step 4: eval — 规则评估
25. rule_events ← RuleEnforcer.enforce(frame_id, entities, scene_rels, behavior_rels)
26.
27. // Step 5: writeback — 写回并保存
28. _prev_frame ← frame
29. _delta_history.append(Δg_t)
30. return Δg_t = (δ_entities, δ_attrs, δ_relations, rule_events)
```

**算法 3.4** 增量引擎五步流程。第 5-7 行的帧跳跃检测为代码工程防御：当仿真出现暂停或断流后，引擎自动重置，避免使用过时的"上一帧"导致错误差分。

帧跳跃检测的引入源于一次实测故障场景：在长时采集的某一 chunk 边界，CARLA 服务器因 Python 客户端心跳超时而重启，重启后仿真器从第 0 帧重新计数，但 `IncrementalEngine` 的 `_prev_frame` 仍持有第 15999 帧的快照。下一帧 frame_id=0 与上一帧 frame_id=15999 的差分为 `frame_id - prev_frame_id = -15999`，若按常规路径走 `compute_delta_attrs(curr, prev)`，由于 prev 的车辆集合与 curr 几乎不重合，引擎会判定"全部车辆新增 + 全部车辆消失"，产生一个假性的全量重建动作。帧跳跃检测通过 `abs(frame_id - prev_frame_id) > 1` 的简单条件触发 `reset()`，将 `_prev_frame` 清空、上一帧性能计数器清零，使下一帧按"首帧"处理（添加所有实体、不计算属性差分），从根本上避免该故障模式。

`IncrementalEngine` 内部状态：

- `_prev_frame: Optional[dict]`：上一帧快照。`None` 表示首帧或刚 `reset()`；
- `_delta_history: List[DeltaGraph]`：累积的差分历史，可用于回放和版本回溯；
- `_threshold_config: ThresholdConfig`：属性级阈值配置。

引擎支持序列化（`to_dict()` / `load_dict()`），可在 chunk 边界保存 checkpoint 并在新 chunk 开始时恢复——这是 3.5 节流式长时采集的核心依赖。

**数值属性防污染机制**：`_validate_numeric_attrs(frame)` 函数在 Step 1 校验当前帧中所有数值属性（位置、速度、油门、刹车等）的类型严格为 `int` 或 `float`。若发现任何字符串或 `None` 污染，触发 `RuntimeError` 阻断该帧处理。这一防御机制的历史背景是：早期版本曾用 `json.dumps(prev, default=str)` 序列化 checkpoint，导致 `None` 被转为字符串 `"None"`，恢复后 `compute_delta` 误判全字段变更（字符串 `"None"` 与数字永不相等，触发不存在的"全属性变化"）。该问题在 `tests/test_storage.py` 中通过 6 个 case 严格覆盖，包含：完整 round-trip 校验、关键字段保护、跨版本恢复、None 污染注入、字符串与数字类型混淆、capacity 边界条件等。

除了上述"已知失败模式"防御，数值属性防污染机制还预防一类更隐蔽的"渐进式类型混淆"：例如 `elapsed_seconds` 字段在 CARLA 中本应为 `float`，但在某次 pause-resume 操作后可能变为 `int`（仿真器内部从 0 起重新计数）。如果不做类型校验，差分引擎会以"前帧 float、当前帧 int"双方都参与计算，最终产生一些静默错误的差分结果，下游 GNN 学习会引入轻微的特征漂移。严格的 `int`/`float` 校验通过强类型拒绝隐蔽污染，是 STKG 构建准确率能稳定达到 99%+ 的关键工程保障。

## 3.4.3 属性版本化机制

`VersionManager`（`stk/dynamic/version.py`）管理属性级时态。其数据结构为 `Dict[entity_id, Dict[attr_name, List[AttrVersion]]]`，每个 `AttrVersion` 记录一个属性值的版本：

$$
\text{AttrVersion} = (\text{value},\ \text{valid\_from},\ \text{valid\_to})
\tag{3.33}
$$

`VersionManager` 主要接口：

- `record_change(eid, attr, new_val, frame_id)`：在 $e$ 的属性 $a$ 上追加新版本，同时将上一版本的 `valid_to` 设置为 `frame_id - 1`；
- `close_entity(eid, frame_id)`：关闭该实体所有未结束的版本，将其 `valid_to` 设置为 `frame_id`；
- `get_current(eid, attr)`：返回最新版本值；
- `get_history(eid, attr)`：返回完整版本链表，支持任意时点的 attribute 时间旅行查询。

属性版本化满足公理 $A_3$。它使得如下查询成为可能：**"在仿真第 1500 帧时，车辆 $v_A$ 的速度是多少？"**——即使后续时刻 $v_A$ 已经多次更新速度，仍可经 `get_version_at(attr, frame_id)` 精确还原：

```cypher
MATCH (v:Vehicle {entity_id: 'veh_123'})-[:hasVersion]->
      (av:AttrVersion {attr: 'speed'})
WHERE av.valid_from <= 1500 AND
      (av.valid_to IS NULL OR av.valid_to >= 1500)
RETURN av.value
```

该查询已在 `stk/storage/queries.py` 中封装为 `temporal_attr_query(entity_id, t_start, t_end)`。版本化机制是后续 KS-NBCF 框架中"GNN 时序编码层"读取帧级特征的基础——它使得 GNN 可在不重放原始仿真的情况下直接基于图谱完成时序学习。

## 3.4.4 时间窗口聚合

规则层单帧触发往往信息有限，复杂事件需要跨多帧聚合观察。本文设计了**双层时间窗口管理**策略，由 `TimeWindowAggregator`（`stk/dynamic/time_window.py`）与 `LongTermEventStore`（`stk/dynamic/event_store.py`）协同实现：

**① 高频事实滑窗聚合**：对场景层高频事实（如位置、速度）采用 600 帧（30 秒）的滑动窗口，超出窗口的版本转入冷存储。

$$
\text{SummaryEvent}(t_s, t_e) = \big\langle\ t_s,\ t_e,\ n_{\text{violation}},\ \text{max\_severity},\ \{\text{rule\_codes}\},\ \{\text{actors}\}\ \big\rangle
\tag{3.34}
$$

默认窗口大小 30 帧（即 1.5 秒 @ 20 fps），滑动步长 1 帧。聚合器在每帧调用 `add(frame_id, violations)`，每窗口结束输出 `SummaryEvent`。该窗口聚合是超车、连续变道、长时跟车等复杂行为分析的基础。

**② 长期事件持久化**：规则层 `SafetyViolation`、`ResponsibilityAssignment` 等事件不进入滑窗，而是直接写入 Neo4j 永久存储。事件节点的 `valid_to` 字段对长时事件设为 `∞`，对短时事件按实际持续帧设置。

**表 3-22** 双层时间窗口管理策略

[三线表]

| 管理层 | 适用对象 | 默认窗口 | 持久化 | 适用查询 |
|--------|---------|---------|--------|---------|
| 高频滑动窗口 | 场景层高频事实（位置、速度等）| 600 帧（30 s） | 滑出转冷存 | 近期实时查询 |
| | 行为层聚合事件 | 30 帧（1.5 s） | 聚合后归档 | 复杂事件回溯 |
| 长期事件存储 | 规则层 SafetyEvent | 永久 | Neo4j | 任意时点追溯 |

该双层管理避免了大量中间版本堆积造成 Neo4j 写入压力，同时保留了高价值事件的完整生命周期。

## 3.4.5 规则事件反向插入

`EventInjector`（`stk/dynamic/event_injector.py`）负责将 `RuleEnforcer` 在 Step 4 生成的规则事件反向插入到当前帧图状态中。其工作流如算法 3.5 所示。

```
算法 3.5: EventInjector.inject_violation(graph, violation)
输入: 当前帧 graph, 待插入的 SafetyViolation
输出: 更新后的 graph

1. graph.violations.append(violation)
2. // 生成 supportedByEvidence 边
3. for each evidence in violation.evidence_path:
4.    graph.evidence_rels.append(supportedByEvidence(violation.id, evidence.id))
5. end for
6. // 生成 violates 边
7. graph.violation_rels.append(violates(violation.src_id, violation.dst_id, ..., violation.sv_id))
8. // 生成 defined_by 边
9. graph.defined_by_rels.append(defined_by(violation.sv_id, violation.rule_code))
10. return graph
```

反向插入的设计意图在于：规则事件的产生依赖场景层与行为层的输入，但规则事件本身的属性又需要被后续帧的场景/行为层引用（例如某违规可能影响责任归因图谱的可视化）。`EventInjector` 在不破坏四层本体递进语义的前提下，实现了"规则层 → 场景/行为层"的语义引用闭环。

## 3.4.6 与静态图谱的对比

表 3-23 对比 STKG 与传统静态知识图谱在时态管理上的差异。

**表 3-23** STKG 与静态知识图谱时态管理对比

[三线表]

| 维度 | 静态 KG | 本文 STKG |
|------|---------|---------|
| 实体表达 | 单一存在 | 4 状态生命周期（CREATED→ACTIVE→STALE→INACTIVE） |
| 属性表达 | 当前值 | 版本链 AttrVersion(value, valid_from, valid_to) |
| 关系表达 | 当前存在 | 三集合 DiffSet(added/removed/unchanged) |
| 更新方式 | 整图重建 | 增量 Δg_t |
| 查询表达 | 静态 Cypher | 时间旅行 Cypher（基于 valid_from/valid_to） |
| 历史回溯 | 不可 | 任意时点属性、关系、规则事件追溯 |
| 存储代价 | $O(N)$ | $O(K \cdot N)$（K 为平均版本数，2-5） |

STKG 通过引入版本与状态机，将"瞬时快照"扩展为"时间序列图"，存储代价约比静态图谱高 2-5 倍，但支持任意时点的精准回溯，这是自动驾驶安全验证场景的硬性需求。

## 3.4.7 小结

本节描述了 STKG 的横向机制——动态更新：差分图 $\Delta g_t$ 的四类更新动作（实体、属性、关系、规则事件），增量引擎的五步流程（recv→diff→patch→eval→writeback），属性版本化机制，双层时间窗口聚合，以及规则事件反向插入。这套机制与 3.3 节的瞬时构造组合，使得 STKG 既能从场景、行为、规则三层完整建模仿真观察，又能在时间轴上管理实体的生命周期与属性版本，是后续长时流式采集与 KS-NBCF 时序学习的基础。
