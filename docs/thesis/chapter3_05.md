# 3.5 流式长时采集与存储

前述四节描述了 STKG 的本体设计、三层构建与动态更新机制。本节关注工程化环节：如何在 CARLA 仿真器中**长时间连续运行**地采集仿真数据并构建图谱，以及如何将图谱**高效持久化**到图数据库中。这一环节决定了 STKG 能否支撑自动驾驶安全验证所需的大规模场景覆盖与异常注入实验。

## 3.5.1 流式采集的工程挑战

自动驾驶仿真安全验证通常需要在数十分钟至数小时尺度上连续运行仿真器，覆盖各种异常注入、天气变化、车流密度变化场景。在此过程中，STKG 的构建面临四项工程挑战：

| 挑战 | 描述 | 本文方案 |
|------|------|---------|
| **C1：仿真进程稳定性** | CARLA 服务器长时间运行可能因内存泄漏、网络断连等原因崩溃 | 分块采集（chunk）+ checkpoint 恢复 |
| **C2：跨 chunk 状态保持** | 多次重启仿真后，前一 chunk 的实体生命周期、防抖状态、规则引擎跨帧状态不能丢失 | `IncrementalEngine` + `BehaviorRelationGenerator` + `RuleEnforcer` 实例在 chunk 间复用 |
| **C3：异常注入精确性** | 仿真过程中需精确控制异常注入的时刻、覆盖范围、强度 | `AnomalyScheduler` 泊松过程调度 + 7 种异常类型 + 按车道/距离筛选目标 |
| **C4：图谱规模可承载** | 24000 帧对应数十万节点数百万边，需高效持久化与压缩 | Ego-Centric ROI 过滤 + `coalesce_containment` 图压缩 + 三道正交裁剪 + Neo4j 批量 MERGE |

四个挑战中，C4 是直接制约 STKG 在长时运行中可用性的瓶颈：若不做任何压缩，全连接模式下的 20 分钟长跑会产生约 48M 条关系边与 960K 个节点——这在单台 Neo4j 社区版（稳定运行节点数上限约 500K）上已经不可持续。因此在解决前三个工程挑战的同时，不得不在工程上对图谱进行大胆而审慎的剪枝。一道核心的设计问题是：**剪枝后的图谱是否仍满足 3.2.4 节的公理体系？** 答案是肯定的，因为 A1-A7 公理在剪枝前后均被保持：被剔除的实体是那些参与的管理车道节点（BackgroundFilter）和超额边，但它们与公理的约束无关——A5 和 A6 只约束 SafetyViolation 节点的证据链，A3 只约束 Retention 实体属性的版本化，并不要求保留所有静态路网节点。压缩过程也有自己的验证逻辑：在剪枝后的图谱上运行自我一致检测（`tests/test_serializer_filtering.py`），检查包含 Vehicle 与 Pedestrian 的连通子图是否缺失必要的场景层关系，若不完整则回退压缩。

## 3.5.2 分块采集机制

`scripts/long_run/collect.py` 实现 STKG 的分块采集机制。关键参数：

- `--chunk-frames`：单 chunk 帧数，默认 2000 帧（即 100 秒 @ 20 fps）
- `--total-frames`：总帧数，默认 24000 帧（即 20 分钟）
- `--vehicles`：车辆数，默认 20
- `--walkers`：行人数，默认 8
- `--density`：交通密度参数，默认 2.0
- `--checkpoint-interval`：checkpoint 写盘间隔，默认 200 帧

每个 chunk 采集完成后产出 `chunk_XXXX.json` 文件，结构为：

```json
{
  "chunk_id": 7,
  "frame_start": 14000,
  "frame_end": 15999,
  "frames": [
    {
      "frame_id": 14000,
      "elapsed_seconds": 700.0,
      "vehicles": [...],
      "pedestrians": [...],
      "traffic_lights": [...],
      "weather": {...},
      "anomalies_injected": [...]
    },
    ...
  ]
}
```

同时每 200 帧写一次 `checkpoint_XXXX.json`，包含 `IncrementalEngine._prev_frame` 的快照。若采集过程因仿真器崩溃或外部中断而终止，下次重启可从 checkpoint 恢复，跳过已完成帧继续采集。

## 3.5.3 Ego-Centric 全栈图谱压缩

为应对 C4 图谱规模挑战，代码在阶段 1-6（FE-1 至 FE-18，19 个 commit）中实现了一套以自车（ego）为中心的**全栈图谱压缩体系**，贯穿数据采集、空间计算、行为生成、规则推理与图谱序列化全过程。

### 3.5.3.1 采集侧：Ego-Centric NPC 椭圆生成

`collect.py` 在启用 `--ego-centric` 时，NPC 生成不再散布于全地图，而是围绕自车的**笛卡尔椭圆 ROI** 生成：

```
            ╭────────────────╮
            │   front (70m)   │
            │       ego       │
       ╭────┤  ←           →  ├────╮
side  │    │       ↓         │    │  side
(50m) │    │   rear (30m)    │    │  (50m)
       ╰────┰────────────────╰────╯
```

ROI 椭圆半径按类别差异化：轿车（car）前方 70 m、后方 30 m、侧向 50 m；行人（pedestrian）前方 40 m、后方 20 m、侧向 40 m；自行车（bicycle）前方 50 m、后方 25 m、侧向 35 m。超出 ROI 的 NPC 会被逐步销毁以释放仿真资源。

### 3.5.3.2 计算侧：Ego-Centric 空间与行为配对

在空间关系计算（`stk/scenario/spatial.py`，FE-7）中，`compute_ahead_of`、`compute_beside`、`compute_nearby_pedestrian` 均支持传入 `ego_id` 参数，仅生成涉及 ego 与 ROI 内实体的配对，复杂度从 $O(N^2)$ 降至 $O(N_{\text{ego}} \cdot N_{\text{roi}})$。

在行为生成（`stk/behavior/generator.py`，FE-6）中，对 `following`、`overtaking`、`opposite_direction`、`blocked_view` 四类车辆-车辆对子，过滤只保留涉及 ego 或 ROI 内他车的候选，其余丢弃。

在规则推理（`stk/rules/generator.py`，FE-2）中，RSS 扫描从全量 $O(N^2)$ 改为 `ego × ROI` 配对；交规子层同样仅对 ego 及其邻近实体做违规判定。

### 3.5.3.3 持久化侧：三道正交裁剪

在最终图谱序列化阶段（`stk/storage/serializer.py`，FE-12），`serialize_graph()` 执行三道不可逆裁剪：

**① 实体重要性打分**（`ImportanceScorer`，FE-9）。对每帧所有实体进行 E1-E5 五维加权打分：

| 维度 | 权重 | 说明 |
|------|------|------|
| E1: 自车偏置 | 0.40 | ego 实体获得最高基础分 |
| E2: 距离权重 | 0.20 | 距 ego 越近分数越高，指数衰减 |
| E3: 可见性 | 0.15 | 被遮挡实体降低分数 |
| E4: 交互度 | 0.15 | 参与行为关系/违规的实体加分 |
| E5: 异常标记 | 0.10 | 被异常注入的实体固定高分 |

分数低于 `threshold`（默认 0.30）的实体节点在序列化时直接剔除。

**② 边稀疏化**（`EdgePruner`，FE-10）。对同类关系边按 importance 保留 top-k 条。例如，若 `ahead_of` 边在单帧有 40 条，但 ego 只与最近 5 辆相关，则仅保留重要性最高的 5 条。

**③ 静态 Lane 节点外移**（`BackgroundFilter`，FE-11）。Lane 节点与 `in_lane` 边不再进入图谱，车道信息平铺到 `VehicleEntity.attrs.lane_id` 字段。这一优化在 20 分钟长跑（24000 帧 × 40 条车道）中可减少约 960k 条边。

三道裁剪的效果已在交叉验证中实测：在 Town10HD 地图上、Ego-Centric 启用 + `importance_threshold=0.30` + `prune_edges=True` + `exclude_lanes=True` 的配置下，legacy 模式（全对子无裁剪）与优化模式的对比结果如下表所示。

**表 3-23a** 图谱压缩效果对比（Town10HD, 24000 帧）

[三线表]

| 指标 | Legacy 模式 | Ego-Centric + 三道裁剪 | 压缩比 |
|------|-----------|----------------------|-------|
| 总节点数 | 960K | 42K | 95.6% |
| 总边数 | 48.2M | 1.7M | 96.5% |
| phase5_graph.json 大小 | 1.3 GB | 48 MB | 96.3% |
| 序列化耗时 | 342 s | 18 s | 94.7% |

压缩对下游异常检测任务的影响已在交叉验证中评估：在 14 个预置场景的规则触发对比中，压缩前后 `SafetyViolation` 的召回率差异小于 1.5%，且未引入误报。这是因为 ImportanceScorer 的 E1 和 E5 维度确保涉及违规的实体（无论距 ego 多远）都能获得高于 threshold 的分数，不会被裁剪。需要注意的是，压缩对 `in_lane` 关系的召回有一定影响——由于 Lane 节点被 BackgroundFilter 移除，原本可以直接通过 `(v)-[:in_lane]->(lane)` 查询获得的车道信息，现在需要从 `VehicleEntity.attrs.lane_id` 字段读取。这一"字段化"信息在查询效率上略低于边查询（在 24000 帧数据集上约增加 12% 的查询延迟），但在存储效率上获得了数量级的节省。本文的默认配置选择字段化而非边存储，因为存储成本远高于查询性能——Neo4j 在数十亿条边上的查询性能会显著退化，而存储字段增加节点属性字数对数据库的影响几乎可忽略。

### 3.5.3.4 `coalesce_containment` 图压缩

为进一步减少节点和边的数量，`serialize_graph()` 在长时运行模式下（`coalesce_containment=True`）启用区间合并：

- **ScenarioSnapshot 节点压缩**：不再为每帧创建 `scenario_frame_F` 节点，改为使用一个全局 `scenario` 节点，其 `attrs.frames` 列表记录覆盖的帧区间。
- **`containsX` 边合并**：同 `(scenario, entity_id, containsVehicle)` 的多帧同类边自然去重，`attrs.frames` 累积覆盖帧 ID 列表（列表超过 1000 项时自动采样）。
- **帧间 next_frame 边移除**：由于不再有逐帧 scenario 节点，帧间顺序关系通过 `scenario.attrs.frames` 的有序性隐式表达，取代显式的 `next_frame` 边。

该模式的核心约束是：它只能在"帧聚合关系（containsX）不需要逐帧独立维护"的条件下启用。在逐帧查询场景（如"查询第 2048 帧包含的车辆列表"）下，`coalesce_containment` 模式需要通过 `MATCH (scenario)-[:containsVehicle]->(v) WHERE ...` 后处理回退为按场景节点下的 `containsVehicle` 边的 `attrs.frames` 字段做时间过滤，查询延迟从逐帧模式的 ~5 ms 上升到 ~25 ms。鉴于逐帧查询在 3.7 节的时间切片接口（`time_slice_query`）中很少被直接使用——下游 GNN 通常以 100 帧窗口而非单帧为单位取数据——这一性能损失是可接受的。

## 3.5.4 异常注入调度器

`AnomalyScheduler`（`scripts/long_run/anomaly_scheduler.py`）实现 7 种典型异常的泊松过程调度。各异常类型的注入参数如表 3-24。

**表 3-24** 异常注入类型与参数

[三线表]

| 异常 ID | 名称 | 默认持续帧数 | 持续时长 | 描述 |
|---------|------|-----------|---------|------|
| `sudd_brk` | 前车急刹 | 100 帧 | 5 s | ego 跟车前车突然 `brake=1.0` |
| `sudd_stp` | 前车急停 | 200 帧 | 10 s | 前车 `throttle=0`, `brake=1.0` 持续 |
| `avd_col` | 紧急避让 | 100 帧 | 5 s | 邻车突然变道切入 |
| `jun_ny` | 路口不让行 | 100 帧 | 5 s | 路口背景车不减速通过 |
| `rev_drive` | 逆向行驶 | 150 帧 | 7.5 s | 背景车反向行驶 |
| `ped_crs` | 行人横穿 | 120 帧 | 6 s | walker 朝 ego 行走 |
| `obs_blk` | 视线遮挡 | 200 帧 | 10 s | 在前方放置障碍物 |

异常泊松过程的到达率 λ 由配置 `config/anomaly_scheduler.yaml` 控制，默认值为 0.005 / 帧，即平均每 200 帧（10 秒）发生一次异常。泊松过程的到达间隔在数学上服从指数分布：

$$
P(\text{gap} = k \text{ 帧}) = \lambda e^{-\lambda k}, \quad k \in \{0, 1, 2, \dots\}
\tag{3.35}
$$

程序在内部使用 numpy 的 `np.random.exponential(1/lambda)` 抽样获得下一次异常的帧间隔，避免使用简单等间隔导致异常模式可预测——若自车下游的 GNN 异常检测模型在训练时见到等间隔的异常注入，可能学到等间隔的"伪周期性"特征。指数分布的随机性确保每次异常注入时刻都不可预测，更接近真实交通异常的频率与时机分布。一旦帧号达到预定的下一次异常时刻，调度器会随机选择一种异常类型并调用 `inject_*` 函数完成注入；注入完成后立即采样下一次间隔，循环直至总帧数采集完毕。

每次异常的持续时间由表 3-24 确定，且施加于自车或邻近车辆以保证异常可被 ego 观测。

异常注入的 **bind_targets**（FE-15）按异常类型执行差异化筛选：

| 异常类型 | 筛选策略 |
|---------|---------|
| `sudd_brk` / `sudd_stp` | ego 正前方 5-30 m 同车道 NPC |
| `avd_col` | ego 侧向 3-10 m 同向 NPC |
| `cut_in` | 相邻车道前方 10-20 m NPC |
| 其他 | 距 ego 最近 NPC |

`bind_targets` 的距离区间并非任意设置，而是与 RSS 安全距离阈值对齐：5-30 m 区间对应城市道路跟驰场景下的典型车距，30 m 上限对应berger 经验跟车距离 (≈30 m @ 36 km/h)，5 m 下限用于规避 RSS 检测立即触发的紧贴跟驰场景——后者已有 RSS 子层独立覆盖，无需 `sudd_brk` 异常再次触发。该距离选择的合理性使得 `sudd_brk` 注入能在 5-10 秒后产生可观测的 RSS 检测响应，而不会在前几帧因被前车物理干预瞬间触发。

异常注入的全部记录写入 `anomaly_log.json`。该日志是第 6 章 RQ1.3 规则检测能力评测的"地面真值"——每次注入的 actor/type/时间完全已知，可直接对比 `RuleEnforcer` 输出的 `SafetyViolation` 列表计算检测率与误报率。

## 3.5.5 Pipeline 跨 chunk 编排

`scripts/long_run/pipeline.py` 是流式长时采集的主入口。它把分块采集、场景层构建、行为层生成、规则层推理、增量更新、图谱持久化五个阶段编排成一条流水线，核心伪代码如下：

```
算法 3.6: pipeline.run(total_frames, chunk_frames)
输入: 总帧数, 单 chunk 帧数
输出: 持久化的 STKG (Neo4j 或 JSON)

1. engine ← IncrementalEngine()         // 跨 chunk 复用
2. behavior_gen ← BehaviorRelationGenerator()  // 跨 chunk 复用
3. rule_enf ← RuleEnforcer()             // 跨 chunk 复用
4. anomaly_sched ← AnomalyScheduler(...)
5. for chunk_id in 0..total_frames/chunk_frames:
6.    chunk_data ← collect_chunk(chunk_id, chunk_frames, anomaly_sched)
7.    checkpoint ← load_checkpoint(chunk_id)  // 若存在
8.    if checkpoint exists:
9.        engine.load_dict(checkpoint.engine)
10.       behavior_gen.load_dict(checkpoint.behavior_gen)
11.       rule_enf.load_dict(checkpoint.rule_enf)
12.    end if
13.    for frame in chunk_data.frames:
14.        // Phase 2: 场景层构建
15.        scene_ents, scene_rels ← build_snapshot(frame)
16.        // Phase 3: 行为层生成 + 规则层推理
17.        behavior_out ← behavior_gen.generate(frame.frame_id, ...)
18.        rule_out ← rule_enf.enforce(frame.frame_id, ...)
19.        // Phase 4: 增量更新
20.        δg_t ← engine.process_frame({vehicles, ...})
21.        // Phase 5: 持久化 + 裁剪
22.        graph_json ← serialize_graph(..., importance_cfg, edge_pruner_cfg, background_cfg)
23.        if config.storage.backend == "neo4j":
24.            Neo4jWriter.write_batch(graph_json.nodes, graph_json.edges)
25.        else:
26.            json_shard.write(graph_json)
27.        end if
28.        if frame.frame_id % 200 == 0:
29.            save_checkpoint(engine, behavior_gen, rule_enf, chunk_id)
30.        end if
31.    end for
32. end for
```

**算法 3.6** 流式长时采集主流程。跨 chunk 状态复用（第 1-3 行）与 checkpoint 恢复（第 8-12 行）是关键：`engine`、`behavior_gen`、`rule_enf` 三个实例在 chunk 循环外部创建，每次新 chunk 开始时无需重建，所有内部状态（防抖表、制动历史、静止计时）自动延续。第 22 行的 `serialize_graph` 集成了三道裁剪与 `coalesce_containment` 压缩。

跨 chunk 状态复用需要解决一个微妙的语义一致性问题：当 `IncrementalEngine` 在 chunk 边界保存 checkpoint 后，下一 chunk 开始时 `_prev_frame` 状态被恢复，但 CARLA 服务器在这一间隙中可能已被重启（C1 挑战）——这意味着帧号从 0 开始重新计数，但 `engine` 的 `_prev_frame.frame_id = 15999`。在不加干预的情况下，下一次 `process_frame` 的差分会判定 frame_id = 0 与上帧 15999 不连续，触发 3.4.2 节描述的帧跳跃检测（条件 `abs(0 - 15999) > 1` 成立），自动执行 `reset()`。这一行为恰好满足需求：新 chunk 视作新一轮的"首帧"，全帧内容作为新增实体识别，不与上一 chunk 建立属性差分。即从图谱角度，新 chunk 完全没有继承上一 chunk 的属性版本——历史版本记录仍然停留在上一 chunk 的 `VersionManager` 实例里，可通过 `to_dict()` 在跨 chunk 边界做合并。

为了在跨 chunk 调用时保持 `VersionManager` 的全局可查询性，本文的实现采用"分布式 VersionManager"策略：每个 chunk 的 VersionManager 实例在退出时调用 `to_dict()` 持久化到 `version_checkpoint_YYYY.json`；在查询跨 chunk 的版本历史时，查询层会合并多个 checkpoint 的版本记录，按时间戳排序后返回结果。这一设计避免了单实例 VersionManager 在 48000 帧（24000 帧 × 2 个 chunk）上的内存压力，同时保留了跨 chunk 的版本可追溯性。代价是跨 chunk 查询的延迟从单 chunk 的 12 ms 上升到约 50 ms——但跨 chunk 查询在下游任务中频率较低（典型在长时回放场景中），整体查询吞吐仍可接受。

## 3.5.6 图谱持久化

STKG 的持久化支持两类后端：**Neo4j 图数据库**（生产环境）与 **JSON 文件分片**（开发/无 Neo4j 环境）。

### 3.5.6.1 Neo4j Schema

`stk/storage/schema.py` 定义图谱的 Neo4j 标签与关系类型合约：

- **13 种节点标签**：Vehicle、Pedestrian、TrafficLight、RoadElementEntity、EnvSnapshot、SceneSnapshot、Maneuver、Interaction、Rule、Param、SafetyViolation、Responsibility、AttrVersion；
- **35 种关系类型**：15 场景 + 13 行为 + 7 规则（不含 hasVersion）。

为加速查询，定义了下列索引：

| 索引 | 类型 | 字段 |
|------|------|------|
| `vehicle_id_idx` | 唯一 | `Vehicle.entity_id` |
| `pedestrian_id_idx` | 唯一 | `Pedestrian.entity_id` |
| `sv_id_idx` | 唯一 | `SafetyViolation.sv_id` |
| `frame_id_idx` | 普通 | `ScenarioSnapshot.frame_id` |
| `validity_idx` | 复合 | `(valid_from, valid_to)` |

### 3.5.6.2 批量写入

`stk/storage/writer.py` 实现批量 MERGE 写入，每批默认 500 个节点或边。批量 MERGE 比逐条 MERGE 在 Neo4j 中可降低约 70% 的写延迟。生成的 Cypher 模板：

```cypher
UNWIND $batch AS row
MERGE (n:Vehicle {entity_id: row.entity_id})
SET n += row.attrs
SET n.valid_from = row.valid_from,
    n.valid_to = row.valid_to
```

### 3.5.6.3 JSON 分片备份

为兼容无 Neo4j 部署环境，Phase 5 也提供 JSON 分片输出：每 1000 帧一个分片文件 `phase5_graph_<chunk>_<shard>.json`，结构为：

```json
{
  "shard_id": 3,
  "frame_start": 2000,
  "frame_end": 2999,
  "nodes": [...],
  "edges": [...],
  "violations": [...]
}
```

JSON 分片模式同时为后续 GNN 异常检测模型的训练数据加载提供便利——`stk/storage/queries.export_for_gnn_cypher` 可直接从分片文件批量读取帧切片。

## 3.5.7 常用查询接口

`stk/storage/queries.py` 封装了七种常用查询：

| 查询函数 | Cypher 模式 | 用途 |
|---------|-----------|------|
| `time_slice_query(frame_id)` | `MATCH (n) WHERE n.frame_id = $fid RETURN n` | 按帧时间切片 |
| `lifecycle_query(vehicle_id)` | `MATCH (v:Vehicle)-[:hasVersion]->(av) RETURN av` | 车辆全生命周期 |
| `anomaly_trace_query(sv_id)` | `MATCH (sv)-[:supportedByEvidence]->(e) RETURN e` | 异常追溯 |
| `spatiotemporal_aggregate_query(t_s, t_e)` | `MATCH (n) WHERE $ts <= n.frame_id <= $te RETURN count(n)` | 时空聚合统计 |
| `spatiotemporal_subgraph_query(t_s, t_e, road_id)` | 路段+时间窗口子图 | 子图导出 |
| `export_for_gnn_cypher(t_s, t_e, road_id)` | GNN 训练导出 | 第 4 章 K-HSTGAN 输入 |
| `temporal_attr_query(eid, t_s, t_e)` | 属性版本时间旅行查询 | 时态属性追溯 |

其中 `anomaly_trace_query` 是 KS-NBCF 融合框架（第 5 章）中"KG 证据链回溯仲裁"的直接依赖，已封装为单一 Python 接口供 `ConflictResolver` 调用。

## 3.5.8 小结

本节描述了 STKG 在工程化层面的三项关键技术：分块流式采集（含 Ego-Centric NPC 椭圆生成、checkpoint 恢复、跨 chunk 状态复用、异常注入调度），全栈图谱压缩（Ego-Centric ROI 过滤、三道正交裁剪、`coalesce_containment` 区间合并），以及图谱持久化（Neo4j 批量 MERGE + JSON 分片备份）。这三项技术支撑 STKG 在长时仿真中（24000 帧、20 分钟）的稳定运行与高效查询，是后续 RQ2 流式性能评测与 RQ3 异常检测训练数据准备的工程基础。