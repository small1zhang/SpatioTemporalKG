# 3.6 动态更新机制

前述 3.3–3.5 节分别描述了 STKG 三层本体的"瞬时构造"，但仿真数据本质上是按帧推进的时间序列，图谱在时间轴上的演化管理是 STKG 区别于静态知识图谱的关键。本节描述 STKG 的横向机制——动态更新：定义差分图 $\Delta g_t$、增量引擎的"五步"主流程、属性版本化机制、时间窗口聚合以及规则事件的反向插入。这些机制共同保证 STKG 在长时仿真中对存储、查询、推理友好的同时，对帧间变化保持精细的时态表达能力。

## 3.6.1 差分图与四类更新动作

在第 3.1.1 节中我们曾给出公式 (3.3)：$G_t = G_{t-1} \oplus \Delta g_t$。本节给出 $\Delta g_t$ 的形式化定义——**差分图**（DeltaGraph）：

\$$
\Delta g_t := \big\langle\ \Delta_{\mathcal{E}}(t),\ \Delta_{\mathcal{A}}(t),\ \Delta_{\mathcal{R}}(t),\ \mathcal{E}_{\text{rule}}(t)\ \big\rangle
\tag{3.13}
\$$

四个分量分别对应四类更新动作：

**① 实体级差分 $\Delta_{\mathcal{E}}(t)$**：本帧相对上一帧的实体集合差分，由 `DiffSet` 三集合表示：

\$$
\Delta_{\mathcal{E}}(t) = \big(\ \mathcal{E}_{t}^{\text{added}},\ \mathcal{E}_{t}^{\text{removed}},\ \mathcal{E}_{t}^{\text{unchanged}}\ \big)
\tag{3.14}
\$$

其中 $\mathcal{E}_{t}^{\text{added}} = \mathcal{E}_t \setminus \mathcal{E}_{t-1}$ 为本帧新进入场景的实体集合，$\mathcal{E}_{t}^{\text{removed}} = \mathcal{E}_{t-1} \setminus \mathcal{E}_t$ 为本帧离开场景的实体集合，$\mathcal{E}_{t}^{\text{unchanged}} = \mathcal{E}_t \cap \mathcal{E}_{t-1}$ 为持续存在的实体集合。注意，"removed"在 STKG 中并不从图中删除，而是触发对应节点的生命周期状态从 `ACTIVE` 转移到 `STALE`（满足公理 $A_7$）。

**② 属性级差分 $\Delta_{\mathcal{A}}(t)$**：本帧相对上一帧的属性值变化集合，结构为：

\$$
\Delta_{\mathcal{A}}(t) = \Big\{\ \big(e,\ a,\ (\text{val}_{t-1},\ \text{val}_t)\big)\ \Big|\ e \in \mathcal{E}_{t}^{\text{unchanged}},\ a \in \mathcal{A},\ |\text{val}_t - \text{val}_{t-1}| > \epsilon_{\text{thresh}}\ \Big\}
\tag{3.15}
\$$

其中 $\epsilon_{\text{thresh}}$ 是属性防抖阈值（默认 0.01），用以过滤 CARLA 物理仿真中微小的浮点抖动，避免产生大量无意义版本。对涉及位置、速度等连续值的属性，阈值由 `ThresholdConfig` 控制。

**③ 关系级差分 $\Delta_{\mathcal{R}}(t)$**：本帧相对上一帧的关系集合差分，结构与实体级差分类似，由 `_relation_key(src_id, dst_id, type, frame_id)` 做唯一性判定：

\$$
\Delta_{\mathcal{R}}(t) = \big(\ \mathcal{R}_{t}^{\text{added}},\ \mathcal{R}_{t}^{\text{removed}},\ \mathcal{R}_{t}^{\text{unchanged}}\ \big)
\tag{3.16}
\$$

**④ 规则事件 $\mathcal{E}_{\text{rule}}(t)$**：本帧规则层新触发的 `SafetyViolation`、`ResponsibilityAssignment` 及其相关边组成的列表：

\$$
\mathcal{E}_{\text{rule}}(t) = \big[\ \text{sv}_1,\ \text{sv}_2,\ \dots,\ \text{sv}_k,\ \text{resp}_1,\ \dots,\ \text{resp}_l\ \big]
\tag{3.17}
\$$

四类组合形成 $\Delta g_t$ 的完整形式化定义。该定义直接对应代码 `DeltaGraph` 的 dataclass 字段：`delta_entities`、`delta_attrs`、`delta_relations`、`rule_events`。

## 3.6.2 增量引擎五步流程

`IncrementalEngine`（`stk/dynamic/incremental_updater.py`）是动态更新的主驱动。它每帧接收一个完整快照，输出一个 `DeltaGraph`。核心流程为五步：

```
算法 3.4: IncrementalEngine.process_frame(frame)
输入: 当前帧完整快照 frame (含 vehicles, pedestrians, traffic_lights, scene_rels, behavior_rels)
输出: DeltaGraph Δg_t

1. // Step 1: recv — 接收并校验
2. validate(frame)  // 检查数值属性不被字符串污染 (公理 A3 防御)
3.
4. // Step 2: diff — 计算差分
5. if _prev_frame is None:  // 首帧或重置后
6.    δ_entities ← {added: all, removed: ∅, unchanged: ∅}
7.    δ_attrs ← {}  // 首帧无属性差分
8.    δ_relations ← {added: all, removed: ∅, unchanged: ∅}
9. else:
10.   δ_entities ← compute_delta_entities(curr_entities, prev_entities)
11.   δ_attrs ← compute_delta_attrs(curr_attrs, prev_attrs, threshold)
12.   δ_relations ← compute_delta_relations(curr_rels, prev_rels)
13. end if
14.
15. // Step 3: patch — 根据差分打补丁
16. apply_entity_lifecycle(δ_entities)  // CREATED→ACTIVE / ACTIVE→STALE
17. apply_attribute_versions(δ_attrs)  // 为每个变化属性创建 AttrVersion
18.
19. // Step 4: eval — 规则评估
20. rule_events ← RuleEnforcer.enforce(frame_id, entities, scene_rels, behavior_rels)
21.
22. // Step 5: writeback — 写回并保存
23. _prev_frame ← frame
24. _delta_history.append(Δg_t)
25. return Δg_t = (δ_entities, δ_attrs, δ_relations, rule_events)
```

`IncrementalEngine` 内部状态：

- `_prev_frame: Optional[dict]`：上一帧快照。`None` 表示首帧或刚 reset。
- `_delta_history: List[DeltaGraph]`：累积的差分历史，可用于回放和版本回溯。

引擎支持序列化（`to_dict()` / `load_dict()`），可在 chunk 边界保存 checkpoint 并在新 chunk 开始时恢复——这是 3.7 节流式长时采集的核心依赖。

**帧跳检测**：若 `_prev_frame` 的 `frame_id` 与当前帧间隔 $> 1$（即仿真出现暂停或断流），引擎自动触发 `reset()`，当下帧作为新的 baseline 处理。这避免了使用过时的"上一帧"导致错误差分的问题。

**数值属性防污染机制**：`_validate_numeric_attrs(frame)` 函数在 Step 1 校验当前帧中所有数值属性（位置、速度、油门、刹车等）的类型严格为 `int` 或 `float`。若发现任何字符串或 `None` 污染，触发 `RuntimeError` 阻断该帧处理。这一防御机制的历史背景是：早期版本曾用 `json.dumps(prev, default=str)` 序列化 checkpoint，导致 `None` 被转为字符串 `"None"`，恢复后 `compute_delta` 误判全字段变更（字符串 `"None"` 与数字永不相等，触发不存在的"全属性变化"）。该问题在 `tests/test_storage.py` 中通过 6 个 case 严格覆盖。

## 3.6.3 属性版本化机制

`VersionManager`（`stk/dynamic/version.py`）管理属性级时态。其数据结构为 `Dict[entity_id, Dict[attr_name, List[AttrVersion]]]`，每个 `AttrVersion` 记录一个属性值的版本：

\$$
\text{AttrVersion} = (\text{value},\ \text{valid\_from},\ \text{valid\_to})
\tag{3.18}
\$$

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

该查询已在 `stk/storage/queries.py` 中封装为 `temporal_attr_query(entity_id, t_start, t_end)`。版本化机制是下游 KS-NBCF 框架中"GNN 时序编码层"读取帧级特征的基础——它使得 GNN 可在不重放原始仿真的情况下直接基于图谱完成时序学习。

## 3.6.4 时间窗口聚合

规则层单帧触发往往信息有限，复杂事件需要跨多帧聚合观察。`TimeWindowAggregator`（`stk/dynamic/time_window.py`）实现了一个简单的滑动窗口聚合器：

\$$
\text{SummaryEvent}(t_s, t_e) = \big\langle\ t_s,\ t_e,\ n_{\text{violation}},\ \text{max\_severity},\ \{\text{rule\_codes}\},\ \{\text{actors}\}\ \big\rangle
\tag{3.19}
\$$

默认窗口大小 30 帧（即 1.5 秒 @ 20 fps），滑动步长 1 帧。聚合器在每帧调用 `add(frame_id, violations)`，每窗口结束输出 `SummaryEvent`。该窗口聚合是超车、连续变道、长时跟车等复杂行为分析的基础。

## 3.6.5 规则事件反向插入

`EventInjector`（`stk/dynamic/event_injector.py`）负责将 `RuleEnforcer` 在 Step 4 生成的规则事件反向插入到当前帧图状态中。其工作流：

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
8. // 生成 definedBy 边
9. graph.defined_by_rels.append(defined_by(violation.sv_id, violation.rule_code))
10. return graph
```

反向插入的设计意图在于：规则事件的产生依赖场景层与行为层的输入，但规则事件本身的属性又需要被后续帧的场景/行为层引用（例如某违规可能影响责任归因图谱的可视化）。`EventInjector` 在不破坏四层本体递进语义的前提下，实现了"规则层 → 场景/行为层"的语义引用闭环。

## 3.6.6 与静态图谱的对比

表 3-12 对比 STKG 与传统静态知识图谱在时态管理上的差异。

**表 3-12** STKG 与静态知识图谱时态管理对比
[三线表]

| 维度 | 静态 KG | STKG |
|------|---------|------|
| 实体表达 | 单一存在 | 4 状态生命周期（CREATED→ACTIVE→STALE→INACTIVE） |
| 属性表达 | 当前值 | 版本链 AttrVersion(value, valid_from, valid_to) |
| 关系表达 | 当前存在 | 三集合 DiffSet(added/removed/unchanged) |
| 更新方式 | 整图重建 | 增量 Δg_t |
| 查询表达 | 静态 Cypher | 时间旅行 Cypher（基于 valid_from/valid_to） |
| 历史回溯 | 不可 | 任意时点属性、关系、规则事件追溯 |
| 存储代价 | O(1) × N | O(K) × N（K 为平均版本数） |

STKG 通过引入版本与状态机，将"瞬时快照"扩展为"时间序列图"，存储代价约比静态 KG 高 2-5 倍（K 平均 2-5），但支持任意时点的精准回溯，这是自动驾驶安全验证场景的硬性需求。

## 3.6.7 小结

本节描述了 STKG 的横向机制——动态更新：差分图 $\Delta g_t$ 的四类更新动作（实体、属性、关系、规则事件），增量引擎的五步流程（recv→diff→patch→eval→writeback），属性版本化机制，时间窗口聚合，以及规则事件反向插入。这套机制与 3.3–3.5 节的瞬时构造组合，使得 STKG 既能从场景、行为、规则三层完整建模仿真观察，又能在时间轴上管理实体的生命周期与属性版本，是后续长时流式采集与 KS-NBCF 时序学习的基础。
