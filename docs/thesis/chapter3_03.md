# 3.4 行为层：行为检测与防抖

如果说场景层回答了"此刻实体在哪里"，那么**行为层**回答的是"此刻实体在做什么"。行为是一个跨多帧的连续概念，单帧静态关系无法构成行为。本节描述 STKG 行为层的设计：定义两类行为节点、十一种行为检测器、十三种行为关系类型，并引入防抖状态机解决单帧抖动引入的关系闪烁问题。

## 3.4.1 设计动机与节点分类

自动驾驶中"行为"一词的语义可分两层：

- **单实体持续行为**（maneuver）：一辆车的纵向行为状态，如巡航、加速、减速、停车、变道、静止；
- **多实体交互行为**（interaction）：两个或多个实体间的相对行为，如跟车、让行、超车、对向行驶、横穿。

二者在表观速度、相对距离、相对朝向上均呈现不同的时间持续性。为统一表达，本文将行为层节点设计为两类，对齐 `stk/behavior/nodes.py`：

- `ManeuverNode`（Neo4j Label: `Maneuver`）：单实体持续行为的**事件节点**，承载 actor、行为类型、起止帧、严重度等属性；
- `InteractionEvent`（Neo4j Label: `Interaction`）：多实体交互行为的**事件节点**，承载 src、dst、交互类型、起止帧、源关系集合等属性。

二者均继承 `BaseEntity`，并通过 3.2.3 节所述的"节点+边双轨表达"原则与 13 种 `BehaviorRelationType` 关联。Maneuver 共有 6 种类型：`standing_still`、`changing_lane`、`accelerating`、`decelerating`、`cruising`、`stopping`。Interaction 共有 13 种类型：`standing_still`、`changing_lane`、`following`、`approaching`、`yielding_to`、`overtaking`、`wrong_side_meeting`、`opposite_direction`、`same_direction`、`blocked_view`、`approaching_pedestrian`、`approaching_intersection`、`crossing`。

表 3-5 列出全部行为节点的关键属性。

**表 3-5** 行为层节点属性

| 节点 | 属性 | 描述 |
|------|------|------|
| ManeuverNode | `maneuver_type` | 6 种之一 |
| | `actor_id` | 行为主体 ID |
| | `frame_start`, `frame_end` | 起止帧 |
| | `duration_frames` | 持续帧数 |
| | `state` | `"active"` 或 `"ended"` |
| | `severity` | 严重度 [0,1] |
| | `derived_attrs` | 派生属性字典（如平均速度） |
| | `related_rule` | 相关规则 ID（可空） |
| InteractionEvent | `interaction_type` | 13 种之一 |
| | `src_id`, `dst_id` | 交互主客体 ID |
| | `frame_start`, `frame_end` | 起止帧 |
| | `duration_frames`, `state`, `severity` | 同上 |
| | `source_relations` | 触发该交互的场景关系集合 |
| | `related_rule` | 关联的规则 ID |

`related_rule` 字段的存在，使得行为节点在生成时即可"声明"其与未来规则层异常之间可能的对应关系，是后续 KS-NBCF 框架 5.5 节中"决策层融合"的语义桥接点。

## 3.4.2 行为检测器

`stk/behavior/detectors.py` 实现 11 个 `detect_*` 函数，每个函数均返回 `Optional[BaseRelation]` 或其列表。表 3-6 给出全部检测器的判定条件与它们的输出行为类型。

**表 3-6** 行为检测器与判定条件

| 检测器 | 输出行为类型 | 判定条件 | 关键阈值 |
|--------|-------------|---------|---------|
| `detect_standing_still` | `standing_still` | $\|v\| < 0.1$ m/s 持续 $N$ 帧 | 速度阈值 0.1 m/s |
| `detect_changing_lane` | `changing_lane` | 横向速度 $\|v_y\| > 0.5$ m/s 且 `in_lane` 关系发生 lane_id 变化 | 横向速度 0.5 m/s |
| `detect_following` | `following` | 同车道 + 距离 $<12$ m + 前车速度 $>$ 后车速度 - 1 m/s | 距离 12 m |
| `detect_approaching` | `approaching` | 同车道 + 距离 $<20$ m + 相对速度 $>1$ m/s（后车接近） | 相对速度 1 m/s |
| `detect_yielding_to` | `yielding_to` | 车辆与行人 $<8$ m + 车辆减速至 $\|v\|<1$ m/s + 行人在横道线 | 行人距离 8 m |
| `detect_overtaking` | `overtaking` | `beside` + 后车速度 $>$ 前车速度 +2 m/s + 持续 3 帧 | 速度差 2 m/s |
| `detect_opposite_direction` | `opposite_direction` | 两车朝向差 $>143°$ + 距离 $<30$ m | 朝向差 143° |
| `detect_blocked_view` | `blocked_view` | 视线方向三车共线 + 中车侵入 $>30\%$ 屏蔽面积 | 屏蔽比 30% |
| `detect_approaching_pedestrian` | `approaching_pedestrian` | `nearby_pedestrian` + 车速 $>5$ m/s + 距离缩短趋势 | 车速 5 m/s |
| `detect_approaching_intersection` | `approaching_intersection` | `in_junction` + 车辆距离路口边界 $<15$ m | 距离 15 m |
| `detect_crossing` | `crossing` | 行人在 `crosswalk` 区域内 + 行人位移方向与车速方向夹角 $>60°$ | 夹角 60° |

11 个检测器全部为纯函数，输入是场景层输出（`vehicles`、`pedestrians`、`traffic_lights`、`junctions`、`crosswalks`、`scene_relations`），输出是行为关系候选列表。检测器的设计素材参照 NHTSA 100-Car 自然驾驶研究及 v3 设计文档 §3.4。

`run_all_detectors(...)` 是检测器批量执行入口，按类型聚合并返回 `{"following": [...], "approaching": [...], ...}`，作为 `BehaviorRelationGenerator.generate(...)` 的核心驱动。

## 3.4.3 行为关系生成器

`BehaviorRelationGenerator` 是行为层主流程的入口。其核心算法如下：

```
算法 3.1: BehaviorRelationGenerator.generate(frame_id, ...)
输入: frame_id, 当前帧场景层输出
输出: maneuvers[], interactions[], behavior_rels[], cross_layer_rels[]

1. candidates ← run_all_detectors(scene_layer_data)
2. // 可选：Ego×ROI 对子过滤
3. if ego_filter_config.enabled:
4.     candidates ← ego_filter.apply(candidates, ego_id)
5. end if
6. for each candidate in candidates:
7.     action ← debouncer.update(candidate.id, candidate.type, frame_id)
8.     if action == "create":
9.         create ManeuverNode/InteractionEvent node
10.        create behavior_rel, manifestsAs, actor/src/dst edges
11.    elif action == "delete":
12.        close existing behavior node (state ← "ended")
13.        close corresponding behavior_rel (valid_to ← frame_id - 1)
14.    elif action == "keep":
15.        update existing node frame_end ← frame_id
16.    end if
17. end for
18. return maneuvers, interactions, behavior_rels, cross_layer_rels
```

算法 3.1 的关键在于 `debouncer.update(...)` 的防抖逻辑。每次调用，防抖器根据当前帧关系是否被检测到，与历史连续出现帧数比较，决定下一步动作是 `create`、`keep`、还是 `delete`。

## 3.4.4 防抖状态机

行为关系在时间上具有"持续时间"约束，单帧的瞬时闪烁不应引发 Maneuver/Interaction 节点的反复创建与删除。例如，`approaching` 关系需要在两车接近时持续 3 帧以上才算"接近"行为；`following` 关系需要持续 5 帧以上才算"跟车"。为此，本文为每种行为关系定义独立的防抖阈值。

**表 3-7** 行为关系防抖阈值（DEFAULT_DEBOUNCE_THRESHOLDS）

| 关系类型 | 进入阈值（持续帧数） | 消失阈值（消失帧数） | 模式 |
|---------|------------------|-----------------|------|
| `following` | 3 | 3 | 双向防抖 |
| `approaching` | 3 | 3 | 双向防抖 |
| `yielding_to` | 3 | 3 | 双向防抖 |
| `overtaking` | 5 | 3 | 进入更稳健 |
| `changing_lane` | 2 | 2 | 双向防抖 |
| `blocked_view` | 3 | 3 | 双向防抖 |
| `approaching_pedestrian` | 3 | 3 | 双向防抖 |
| `approaching_intersection` | 2 | 2 | 双向防抖 |
| `crossing` | 3 | 3 | 双向防抖 |
| `standing_still` | 2 | 2 | 双向防抖 |
| `wrong_side_meeting` | 1 | 1 | 瞬时反应 |
| `opposite_direction` | 1 | 1 | 瞬时反应 |
| `same_direction` | 1 | 1 | 瞬时反应 |

`RelationDebouncer` 为每一对 `(src_id, dst_id, type)` 维护一个内部状态：`active`（关系当前是否存在）、`on_counter`（连续被检测到帧数）、`off_counter`（连续消失帧数）。状态机算法如下：

```
算法 3.2: RelationDebouncer.update(cand_id, type, frame_id)
输入: 当前候选边 id, 类型, 帧号
输出: "create" | "keep" | "delete" | "none"

1. state ← _table.get((cand_id, type))
2. if state is None:  // 首次出现
3.    state ← {active: false, on_counter: 1, off_counter: 0}
4.    _table[(cand_id, type)] ← state
5. else:
6.    state.on_counter += 1; state.off_counter ← 0
7. end if
8.
9. enter_thresh, exit_thresh ← THRESHOLDS[type]
10. if not state.active and state.on_counter >= enter_thresh:
11.   state.active ← true
12.   return "create"
13. end if
14. if state.active:
15.    return "keep"
16. end if
17. return "none"

// 当某帧未检测到该关系时，调用 decay():
18. state.off_counter += 1; state.on_counter ← 0
19. if state.active and state.off_counter >= exit_thresh:
20.    state.active ← false
21.    return "delete"
22. end if
23. return "none"
```

防抖机制的另一关键功能是处理真实消失：当关系连续消失帧数达到 `exit_thresh` 时，触发 `delete` 动作，关闭对应 `ManeuverNode`/`InteractionEvent` 的 `frame_end` 字段并将 `state` 设置为 `"ended"`，同时将行为关系边的 `valid_to` 字段设置为 `frame_id - 1`。这样既保留了行为节点的完整时长信息，又保证了后续帧若该关系复现，将被识别为新的行为节点，而非沿用旧节点。

防抖状态机的实现位于 `stk/behavior/debouncer.py`，并在 `tests/test_debouncer.py` 中通过 18 个 case 覆盖：包括重置、激活、关闭、瞬时关系的快速切换、与持续关系的稳健切换等场景。

## 3.4.5 跨层桥接机制

行为层与场景层之间通过 `stk/behavior/manifest.py` 中的四类跨层边连接：

- `manifestsAs(behavior_node_id, src, dst, type, frame_id)`：将行为节点与对应的行为关系边绑定，实现"节点+边双轨表达"——同一行为既可经节点参与图遍历，也可经关系边参与拓扑查询；
- `actor_edge(maneuver_id, vehicle_id, frame_id)`：连接 Maneuver 与其行为主体车辆；
- `src_edge(interaction_id, src_entity_id, frame_id)`、`dst_edge(interaction_id, dst_entity_id, frame_id)`：连接 Interaction 与其交互主客体。

`link_maneuver_to_scene(...)` 与 `link_interaction_to_scene(...)` 函数封装了一次性生成行为节点 + 跨层桥接边的逻辑，由 `BehaviorRelationGenerator` 在创建新行为时统一调用。

这一桥接设计的精妙之处在于：当 3.5 节规则层的 `RuleEnforcer` 检测到违规时，可通过 `supportedByEvidence` 边反向指回行为节点，再经由 `src`/`dst` 边定位到具体的车辆实体——形成"违规 → 行为 → 实体"的完整证据链，满足 3.2.4 节公理 $A_6$ 的"事件可追溯性"要求。

## 3.4.6 小结

本节详述了行为层的设计：通过 2 类节点、13 种关系、11 个检测器与防抖状态机的有机协作，将场景层瞬时空间关系聚合成跨帧的"行为语义"。防抖机制是行为层的核心算法贡献，使图谱能够过滤单帧抖动，同时保持对真实消失的快速响应。跨层桥接机制则为下一节的规则层提供了清晰的"证据入口"。
