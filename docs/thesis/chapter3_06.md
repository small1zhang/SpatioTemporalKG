# 3.7 流式长时采集与存储

前述六节描述了 STKG 的本体设计、三层构建与动态更新机制。本节关注工程化环节：如何在 CARLA 仿真器中**长时间连续运行**地采集仿真数据并构建图谱，以及如何将图谱**高效持久化**到图数据库中。这一环节决定了 STKG 能否支撑自动驾驶安全验证所需的大规模场景覆盖与异常注入实验。

## 3.7.1 流式采集的工程挑战

自动驾驶仿真安全验证通常需要在数十分钟至数小时尺度上连续运行仿真器，覆盖各种异常注入、天气变化、车流密度变化场景。在此过程中，STKG 的构建面临四项工程挑战：

| 挑战 | 描述 | 本文方案 |
|------|------|---------|
| **C1：仿真进程稳定性** | CARLA 服务器长时间运行可能因内存泄漏、网络断连等原因崩溃 | 分块采集（chunk）+ checkpoint 恢复 |
| **C2：跨 chunk 状态保持** | 多次重启仿真后，前一 chunk 的实体生命周期、防抖状态、规则引擎跨帧状态不能丢失 | `IncrementalEngine` + `BehaviorRelationGenerator` 实例在 chunk 间复用 |
| **C3：异常注入精确性** | 仿真过程中需精确控制异常注入的时刻、覆盖范围、强度 | `AnomalyScheduler` 泊松过程调度 + 7 种异常类型 |
| **C4：图谱规模可承载** | 24000 帧对应数十万节点数百万边，需高效持久化 | Neo4j 批量 MERGE + JSON 分片输出 |

## 3.7.2 分块采集机制

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

## 3.7.3 异常注入调度器

`AnomalyScheduler`（`scripts/long_run/anomaly_scheduler.py`）实现 7 种典型异常的泊松过程调度。各异常类型的注入参数如表 3-13。

**表 3-13** 异常注入类型与参数
[三线表]

| 异常 ID | 名称 | 默认持续帧数 | 持续时长 | 描述 |
|---------|------|-----------|---------|------|
| `sudd_brk` | 前车急刹 | 100 帧 | 5 s | ego 跟车前车突然 `brake=1.0` |
| `sudd_stp` | 前车急停 | 200 帧 | 10 s | 前车 `throttle=0`, `brake=1.0` 持续 |
| `avd_col` | 紧急避让 | 100 帧 | 5 s | ego 邻车突然变道切入 |
| `jun_ny` | 路口不让行 | 100 帧 | 5 s | 路口背景车不减速通过 |
| `rev_drive` | 逆向行驶 | 150 帧 | 7.5 s | 背景车反向行驶 |
| `ped_crs` | 行人横穿 | 120 帧 | 6 s | walker 朝 ego 行走 |
| `obs_blk` | 视线遮挡 | 200 帧 | 10 s | 在前方放置障碍物 |

异常泊松过程的到达率 `λ` 由配置 `config/anomaly_scheduler.yaml` 控制，默认值为 0.005 / 帧，即平均每 200 帧（10 秒）发生一次异常。每次异常的持续时间由上述表确定，且施加于 ego 车辆或 ego 邻近车辆以保证异常可被 ego 观测。

异常注入的全部记录写入 `anomaly_log.json`，结构为：

```json
[
  {
    "anomaly_id": 17,
    "type": "sudd_brk",
    "trigger_frame": 2048,
    "duration_frames": 100,
    "actor_id": "veh_42",
    "params": {"brake_force": 1.0}
  },
  ...
]
```

`anomaly_log.json` 是第 6 章 RQ1.3 规则检测能力评测的"地面真值"——每次注入的 actor/type/时间完全已知，可直接对比 `RuleEnforcer` 输出的 `SafetyViolation` 列表计算检测率与误报率。

## 3.7.4 Pipeline 跨 chunk 编排

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
8.    for frame in chunk_data.frames:
9.        // Phase 2: 场景层构建
10.       scene_ents, scene_rels ← build_snapshot(frame)
11.       // Phase 3: 行为层生成 + 规则层推理
12.       behavior_out ← behavior_gen.generate(frame.frame_id, ...)
13.       rule_out ← rule_enf.enforce(frame.frame_id, ...)
14.       // Phase 4: 增量更新
15.       δg_t ← engine.process_frame({
16.           vehicles: scene_ents.vehicles, ...,
17.           behavior_rels: behavior_out.behavior_rels
18.       })
19.       // Phase 5: 持久化
20.       if config.storage.backend == "neo4j":
21.           Neo4jWriter.write_entities(scene_ents + behavior_out.maneuvers + ...)
22.           Neo4jWriter.write_relations(scene_rels + behavior_out.behavior_rels + ...)
23.       else:
24.           json_shard.write(frame.frame_id, scene_ents, ...)
25.       end if
26.       if frame.frame_id % 200 == 0:
27.           save_checkpoint(engine, chunk_id, frame.frame_id)
28.       end if
29.   end for
30. end for
```

跨 chunk 状态复用是关键：`engine`、`behavior_gen`、`rule_enf` 三个实例在 chunk 循环外部创建，每次新 chunk 开始时无需重建，所有内部状态（防抖表、制动历史、静止计时）自动延续。这一设计保证长时仿真中行为的连贯性（如某超车行为横跨 chunk 边界时，防抖状态机会延续其持续帧计数）。

## 3.7.5 图谱持久化

STKG 的持久化支持两类后端：**Neo4j 图数据库**（生产环境）与 **JSON 文件分片**（开发/无 Neo4j 环境）。

### 3.7.5.1 Neo4j Schema

`stk/storage/schema.py` 定义图谱的 Neo4j 标签与关系类型合约：

- **13 种节点标签**：Vehicle、Pedestrian、TrafficLight、RoadElementEntity、EnvSnapshot、SceneSnapshot、Maneuver、Interaction、Rule、Param、SafetyViolation、Responsibility、AttrVersion；
- **35 种关系类型**：15 场景 + 13 行为 + 7 规则（不含 hasVersion）。

为加速查询，配置中定义了下列索引：

| 索引 | 类型 | 字段 |
|------|------|------|
| `vehicle_id_idx` | 唯一 | `Vehicle.entity_id` |
| `pedestrian_id_idx` | 唯一 | `Pedestrian.entity_id` |
| `sv_id_idx` | 唯一 | `SafetyViolation.sv_id` |
| `frame_id_idx` | 普通 | `ScenarioSnapshot.frame_id` |
| `validity_idx` | 复合 | `(valid_from, valid_to)` |

### 3.7.5.2 批量写入

`stk/storage/writer.py` 实现批量 MERGE 写入，每批默认 500 个节点或边。批量 MERGE 比逐条 MERGE 在 Neo4j 中可降低约 70% 的写延迟。`write_entity_batch(entities, batch_size)` 与 `write_relation_batch(relations, batch_size)` 是两个核心入口。生成的 Cypher 模板：

```cypher
UNWIND $batch AS row
MERGE (n:Vehicle {entity_id: row.entity_id})
SET n += row.attrs
SET n.valid_from = row.valid_from,
    n.valid_to = row.valid_to
```

### 3.7.5.3 JSON 分片备份

为兼容无 Neo4j 部署环境，`Phase 5` 也提供 JSON 分片输出：每 1000 帧一个分片文件 `phase5_graph_<chunk>_<shard>.json`，结构为：

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

JSON 分片模式同时为后续 GNN 异常检测模型的训练数据加载提供便利——`stk/gnn/exporter.py` 可直接从分片文件批量读取帧切片，无需连数据库。

## 3.7.6 常用查询接口

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

其中 `anomaly_trace_query` 是 KS-NBCF 融合框架（大论文第 5 章）中"KG 证据链回溯仲裁"的直接依赖，已封装为单一 Python 接口供 `ConflictResolver` 调用。

## 3.7.7 小结

本节描述了 STKG 在工程化层面的两项关键技术：分块流式采集（含 checkpoint 恢复、跨 chunk 状态复用、异常注入调度）和图谱持久化（Neo4j 批量 MERGE + JSON 分片备份）。这两项技术支撑 STKG 在长时仿真中（24000 帧、20 分钟）的稳定运行与高效查询，是后续 RQ2 流式性能评测与 RQ3 异常检测训练数据准备的工程基础。
