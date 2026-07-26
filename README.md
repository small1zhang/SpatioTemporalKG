# SpatioTemporalKG — 时空动态知识图谱

> 基于 **CARLA 0.9.16** 仿真器真值数据，面向自动驾驶安全验证的**时空动态知识图谱**（Spatio-Temporal Dynamic Knowledge Graph, STKG）构建框架。
>
> 将仿真原始观测（车辆/行人/信号灯/路网/天气）转化为**语义明确、时态可追溯、规则可推理**的四层图结构，支撑下游异常检测、规则验证与安全分析。

---

## 目录

- [项目概述](#项目概述)
- [核心数据统计](#核心数据统计)
- [系统架构](#系统架构)
- [模块详解](#模块详解)
  - [0️⃣ 本体层 (Ontology)](#0️⃣-本体层-ontology)
  - [1️⃣ 场景层 (Scenario)](#1️⃣-场景层-scenario)
  - [2️⃣ 行为层 (Behavior)](#2️⃣-行为层-behavior)
  - [3️⃣ 规则层 (Rules)](#3️⃣-规则层-rules)
  - [4️⃣ 动态更新 (Dynamic)](#4️⃣-动态更新-dynamic)
  - [5️⃣ 提取层 (Extraction)](#5️⃣-提取层-extraction)
  - [6️⃣ 存储层 (Storage)](#6️⃣-存储层-storage)
  - [7️⃣ 滤波层 (Filter)](#7️⃣-滤波层-filter)
  - [8️⃣ 流水线编排 (Pipeline)](#8️⃣-流水线编排-pipeline)
  - [9️⃣ 可视化 (Visualization)](#9️⃣-可视化-visualization)
- [数据流总览](#数据流总览)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [运行方式](#运行方式)
- [测试矩阵](#测试矩阵)
- [项目结构](#项目结构)
- [关键设计决策](#关键设计决策)
- [论文与引用](#论文与引用)

---

## 项目概述

### 解决的问题

自动驾驶仿真平台（CARLA、LGSVL 等）持续输出多源异构的实时观测数据：
- **多源异构**：同一时刻存在物理、几何、控制、规则等多类信息
- **时空同步**：所有要素在统一帧时钟下采样
- **演化交织**：场景结构、行为状态、违规事件在时间轴上层层递进

仅依赖原始数据难以支撑下游安全验证任务的结构化、可推理、可解释需求。STKG 在仿真数据与下游应用之间引入**时空动态知识图谱**作为中间表示层。

### 核心思想

**四层本体**（场景 → 行为 → 规则）纵向递进 + **动态更新**横向支撑：

```
                  ┌──────────────────────────────────────┐
                  │        横向机制：动态更新              │
                  │   Δg_t = (Δentities, Δattrs,          │
                  │            Δrelations, rule_events)   │
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

---

## 核心数据统计

| 维度 | 数值 | 代码位置 |
|------|------|---------|
| **实体类型** | 14 类 | `stk/ontology/types.py` |
| **场景层关系** | 15 种 | `SceneRelationType` |
| **行为层关系** | 13 种 | `BehaviorRelationType` |
| **规则层关系** | 7 种 | `RuleRelationType` |
| **跨层桥接关系** | 7 种 | `CrossLayerRelationType` |
| **关系总数** | **42 种** | 四大类之和 |
| **核心公理** | 7 条 (A1–A7) | `stk/ontology/axioms.py` |
| **场景节点属性** | 6 类 × 4~18 个字段 | `stk/scenario/nodes.py` |
| **行为检测器** | 11 个 | `stk/behavior/detectors.py` |
| **防抖关系** | 13 种，阈值 1~5 帧 | `stk/behavior/debouncer.py` |
| **RSS 参数** | 7 个 | `stk/rules/rss/model.py` |
| **交规规则** | 14 条（R1–R18 含跳号） | `stk/rules/traffic/rules.py` |
| **预置场景** | 14 个（A/B/C/D 四类） | `stk/scenario/scenario_library.py` |
| **异常注入类型** | 7 种 | `scripts/long_run/anomaly_scheduler.py` |
| **流式分块** | 2000 帧/块，24000 帧最大 | `scripts/long_run/collect.py` |
| **配置文件** | 6 个 | `config/*.yaml` |
| **可驾车地图** | 5 张 | `map_configs/*.yaml` |

---

## 系统架构

### 四层本体语义分工

| 层级 | 回答的问题 | 时间尺度 | 输入 | 输出 |
|------|-----------|---------|------|------|
| **场景层** (Scene) | "此刻世界长什么样" | 单帧 (~50ms) | CARLA 原始观测 | 实体属性 + 空间拓扑关系 |
| **行为层** (Behavior) | "实体在做什么" | 多帧 (~秒级) | 场景层输出 | 行为节点 + 交互关系 |
| **规则层** (Rule) | "是否合规" | 跨帧 (~事件级) | 场景+行为层 | SafetyViolation + 证据链 |
| **动态更新** (Dynamic) | "图谱如何随时间演化" | 所有帧 | 三层输出 | Δg_t 差分 + 版本管理 |

### 模块依赖关系

```
stk/
├── ontology/         ← 无依赖，被所有模块引用
├── extraction/       ← 依赖 ontology
├── scenario/         ← 依赖 ontology，被 behavior/rules 引用
├── behavior/         ← 依赖 ontology + scenario
├── rules/            ← 依赖 ontology + scenario + behavior
├── dynamic/          ← 依赖 ontology + rules
├── filter/           ← 依赖 ontology + config
├── pipeline/         ← 依赖所有上层模块
├── storage/          ← 依赖所有模块（序列化全图）
└── viz/              ← 依赖 storage
```

---

## 模块详解

### 0️⃣ 本体层 (Ontology)

**位置**：`stk/ontology/`

**核心类**：

| 模块 | 功能 |
|------|------|
| `entity.py` | `BaseEntity` 基类 — 所有实体节点的 pydantic BaseModel，含 `entity_id`, `entity_type`, `valid_from/to`, `attrs` |
| `relation.py` | `BaseRelation` 基类 — 所有关系边的 pydantic BaseModel，含 `src_id`, `dst_id`, `relation_type`, `frame_id`, `valid_from/to` |
| `types.py` | **14 种实体类型枚举** + **4 大类 42 种关系类型枚举**（SceneRelationType 15 种、BehaviorRelationType 13 种、RuleRelationType 7 种、CrossLayerRelationType 7 种）|
| `namespace.py` | 分层命名空间 & `IDGenerator` 单例，保证全局 ID 唯一性 |
| `axioms.py` | **七条核心公理 A1–A7** 的可调用验证函数 |
| `lifecycle.py` | `NodeLifecycle` — 四状态生命周期（CREATED → ACTIVE → STALE → INACTIVE）|
| `temporal_triple.py` | 时态三元组 τ 的七元组表达 |

**七条核心公理**：
- **A1**: 实体 ID 全局唯一
- **A2**: 实体类型一经创建即固定
- **A3**: 属性版本化（时间旅行可查）
- **A4**: 所有关系必须有 valid_from
- **A5**: 规则层节点必须连接证据
- **A6**: 违规节点可追溯到原始观测
- **A7**: 增量一致性（Δ 不删除实体）

**命名规则**：

| 前缀 | 实体类型 | 示例 |
|------|---------|------|
| `veh_` | 车辆 | `veh_123` |
| `ped_` | 行人 | `ped_42` |
| `tl_` | 信号灯 | `tl_5` |
| `road_` | 道路元素 | `road_3_lane_2` |
| `man_` | 行为节点 | `man_veh_123_2048` |
| `int_` | 交互节点 | `int_veh_123_veh_456_following_2048` |
| `sv_` | 违规节点 | `sv_R13a_2052` |
| `resp_` | 责任节点 | `resp_sv_R13a_2052_veh_123` |

---

### 1️⃣ 场景层 (Scenario)

**位置**：`stk/scenario/`

**节点类型（6 类）**：

| 节点 | Neo4j Label | 属性数 | 说明 |
|------|-------------|--------|------|
| VehicleEntity | `Vehicle` | 18 | 标识组(3) + 位置/运动组(9) + 派生计算组(4) + 物理状态组(4) |
| PedestrianEntity | `Pedestrian` | 13 | 含 is_on_crosswalk, is_on_sidewalk, action 特有字段 |
| TrafficLightEntity | `TrafficLight` | 7 | state(Red/Yellow/Green), elapsed_time, affected_lane_ids |
| RoadElementEntity | `Lane`/`Road`/`Junction` | 13 | 含 left/right_lane_id, junction_id, lane_type |
| EnvironmentSnapshot | `EnvSnapshot` | 12 | 天气六要素 + 帧信息 |
| ScenarioSnapshot | `SceneSnapshot` | 4 | 每帧的聚合根节点 |

**空间关系（15 种）**：

| 类别 | 关系 | 计算方式 |
|------|------|---------|
| **拓扑关系** | `in_lane` | 最近车道匹配（横向距离 < ε） |
| | `on_road` | 车道 → 路段包含关系派生 |
| | `in_junction` | 车道 → 路口包含关系（junction_id != -1）|
| | `adjacent_lane` | `left_lane_id` / `right_lane_id` 字段 |
| | `lane_connects` | 车道连接拓扑 |
| **空间关系** | `ahead_of` | 同车道纵向距离 > 0 |
| | `beside` | |横向| ≤ 3.0m，|纵向| < 5.0m |
| | `nearby_pedestrian` | 欧氏距离 < 20.0m |
| **控制关系** | `controlled_by` | 信号灯 → 车道映射表 |
| **帧聚合关系** | `containsVehicle/Pedestrian/TrafficLight/Road` | 帧根 → 实体 |
| | `hasEnvironment` | 帧根 → 环境节点 |
| | `weather_context` | 环境语境边 |

**主要模块**：

| 模块 | 功能 |
|------|------|
| `snapshot_builder.py` | `build_snapshot(FrameData)` 构建帧根 + 环境双根结构 |
| `spatial.py` | 全部空间关系纯函数计算（`compute_in_lane`, `compute_ahead_of`, `compute_beside` 等）|
| `lifecycle_manager.py` | 每帧差集计算生命周期状态变化 |
| `scenario_library.py` | 14 个预置测试场景工厂函数 |
| `nodes.py` | 6 类场景节点的 pydantic 定义 |
| `relations.py` | 场景层 15 种关系工厂函数 |

---

### 2️⃣ 行为层 (Behavior)

**位置**：`stk/behavior/`

**核心设计：节点+边双轨表达**

同一行为在图中既以**节点**形式存在（ManeuverNode / InteractionEvent），又以**关系**形式存在（behavior_rel），二者通过 `manifestsAs` 边绑定。

**节点类型（2 类）**：

| 节点 | 角色 | 类型数量 |
|------|------|---------|
| `ManeuverNode` | 单实体持续行为 | 6 种：standing_still, changing_lane, accelerating, decelerating, cruising, stopping |
| `InteractionEvent` | 多实体交互事件 | 13 种：following, approaching, yielding_to, overtaking, ... |

**行为检测器（11 个）**：

| 检测器 | 判定条件 | 关键阈值 |
|--------|---------|---------|
| `detect_standing_still` | \|v\| < 0.1 m/s 持续 N 帧 | 速度阈值 0.1 m/s |
| `detect_changing_lane` | 横向速度 > 0.5 m/s 且 lane_id 变化 | 横向速度 0.5 m/s |
| `detect_following` | 同车道 + 距离 < 12 m | 距离 12 m |
| `detect_approaching` | 同车道 + 距离 < 20 m + 相对速度 > 1 m/s | 相对速度 1 m/s |
| `detect_yielding_to` | 行人 < 8 m + 车辆减速 + 行人在横道线 | 行人距离 8 m |
| `detect_overtaking` | beside + 后车速度 > 前车 + 2 m/s + 持续 3 帧 | 速度差 2 m/s |
| `detect_opposite_direction` | 朝向差 > 143° + 距离 < 30 m | 朝向差 143° |
| `detect_blocked_view` | 三车共线 + 中车屏蔽 > 30% | 屏蔽比 30% |
| `detect_approaching_pedestrian` | nearby_pedestrian + 车速 > 5 m/s | 车速 5 m/s |
| `detect_approaching_intersection` | in_junction + 距离路口 < 15 m | 距离 15 m |
| `detect_crossing` | 行人在 crosswalk + 位移与车速夹角 > 60° | 夹角 60° |

**防抖状态机**（核心算法贡献）：

每种行为关系有独立的**进入阈值**和**消失阈值**（帧数）：

| 关系类型 | 进入阈值 | 消失阈值 | 模式 |
|---------|---------|---------|------|
| `following` | 3 | 3 | 双向防抖 |
| `approaching` | 3 | 3 | 双向防抖 |
| `overtaking` | 5 | 3 | 进入更稳健 |
| `changing_lane` | 2 | 2 | 双向防抖 |
| `wrong_side_meeting` | 1 | 1 | 瞬时反应 |
| `opposite_direction` | 1 | 1 | 瞬时反应 |

**跨层桥接**：`manifest.py` 通过 `manifestsAs` / `actor` / `src` / `dst` 四类边将行为节点与场景实体连接。

**主要模块**：

| 模块 | 功能 |
|------|------|
| `generator.py` | `BehaviorRelationGenerator` — 行为层主驱动 |
| `detectors.py` | 11 个 `detect_*` 纯函数 |
| `debouncer.py` | `RelationDebouncer` — 防抖状态机 |
| `manifest.py` | 跨层桥接边生成 |
| `nodes.py` | ManeuverNode + InteractionEvent 定义 |
| `relations.py` | 13 种行为关系工厂函数 |

---

### 3️⃣ 规则层 (Rules)

**位置**：`stk/rules/`

**双层架构**：

```
规则层 (Rule Layer)
├── RSS 子层 (物理安全校验)
│   ├── 纵向安全距离 d_min_long
│   ├── 横向安全距离 d_min_lat
│   ├── 反应不当 NoProperResponse
│   └── 责任归因 Responsible
│
├── 交通法规子层 (行为合规校验)
│   ├── R1 行人优先 · R2 闯红灯 · R3 实线变道
│   ├── R4 对向会车 · R5 逆行 · R7 路口让行
│   ├── R8 弱势保护 · R9 学区限速 · R10 高速限速
│   ├── R11 天气限速 · R13 禁停 · R16 黄灯抢行
│   └── R17 不按规定车道 · R18 导向车道
│
└── 输出
    ├── SafetyViolation 节点 + violates 边
    ├── supportedByEvidence 证据链
    └── ResponsibilityAssignment 责任归因
```

**RSS 纵向安全距离公式**：

$$d_{\min}^{\text{long}}(A, B, t) = \max\!\left(0,\ v_A \rho + \frac12 a_{\max,\text{accel}} \rho^2 + \frac{(v_A + a_{\max,\text{accel}}\rho)^2}{2\,a_{\min,\text{brake}}} - \frac{v_B^2}{2\,a_{\text{brake}}}\right)$$

**RSS 默认参数**：

| 参数 | 符号 | 值 | 描述 |
|------|------|-----|------|
| 反应时间 | ρ | 0.3 s | 感知到制动的时间 |
| 后车最大加速 | a_max_accel | 0.5 m/s² | 反应时间内合理加速上限 |
| 后车最小制动 | a_min_brake | 3.0 m/s² | 反应后能实现的制动 |
| 前车最大制动 | a_brake | 8.0 m/s² | 前车极端制动 |

**规则层节点（4 类）**：

| 节点 | 属性 |
|------|------|
| RuleDefinition | rule_id, rule_name, rule_layer, predicate_name |
| RuleParameter | param_id, name, value, unit |
| SafetyViolation | sv_id, rule_code, frame_id, severity, evidence_path |
| ResponsibilityAssignment | resp_id, sv_id, responsible_actor_id, reason |

**规则层关系（7 种）**：`definedBy`、`usesParam`、`violates`、`supportedByEvidence`、`triggers`、`responsibleFor`、`causedBy`

**主要模块**：

| 模块 | 功能 |
|------|------|
| `generator.py` | `RuleEnforcer` — 规则层主驱动，RSS 扫描 + 交规检查 |
| `nodes.py` | 4 类规则层节点定义 |
| `relations.py` | 7 种规则层关系工厂函数 |
| `rss/model.py` | RSS 三个核心算子 + 默认参数 |
| `traffic/rules.py` | 14 条 `check_Ri_*` 函数 |

---

### 4️⃣ 动态更新 (Dynamic)

**位置**：`stk/dynamic/`

**差分图 Δg_t 定义**：

$$\Delta g_t := \langle \Delta_{\mathcal{E}}(t),\ \Delta_{\mathcal{A}}(t),\ \Delta_{\mathcal{R}}(t),\ \mathcal{E}_{\text{rule}}(t) \rangle$$

| 分量 | 含义 | 结构 |
|------|------|------|
| Δ_ℰ(t) | 实体级差分 | DiffSet(added, removed, unchanged) |
| Δ_𝒜(t) | 属性级差分 | {(eid, attr): (old_val, new_val)} |
| Δ_ℛ(t) | 关系级差分 | DiffSet(added, removed, unchanged) |
| ℰ_rule(t) | 规则事件 | SafetyViolation 列表 |

**增量引擎五步流程**：

```
Algorithm: IncrementalEngine.process_frame(frame)
1. recv     — 接收并校验（数值属性防污染）
2. diff     — 计算三集合差分 + 属性变化
3. patch    — 应用生命周期转移 + 属性版本化
4. eval     — 规则引擎评估
5. writeback— 保存 prev_frame, 返回 Δg_t
```

**属性版本化**：

`VersionManager` 管理每个实体每个属性的版本链 `AttrVersion(value, valid_from, valid_to)`，支持任意时点的属性时间旅行查询：

```cypher
MATCH (v:Vehicle {entity_id: 'veh_123'})-[:hasVersion]->
      (av:AttrVersion {attr: 'speed'})
WHERE av.valid_from <= 1500 AND
      (av.valid_to IS NULL OR av.valid_to >= 1500)
RETURN av.value
```

**主要模块**：

| 模块 | 功能 |
|------|------|
| `incremental_updater.py` | `IncrementalEngine` — 增量主驱动 + checkpoint 序列化 |
| `diff.py` | `DeltaGraph`, `DiffSet`, `compute_delta` 差分计算 |
| `version.py` | `VersionManager`, `AttrVersion` 属性版本化 |
| `time_window.py` | `TimeWindowAggregator` 滑动窗口聚合器 |
| `event_injector.py` | `inject_violation` 规则事件反向插入 |
| `snapshot_store.py` | `SnapshotStore` 快照存储 |

---

### 5️⃣ 提取层 (Extraction)

**位置**：`stk/extraction/`

**6 类提取器**：

| 提取器 | CARLA API | 输出 |
|--------|-----------|------|
| `actor_extractor.py` | `world.get_actors().filter('vehicle.*'/'walker.*')` | Vehicle dict + Pedestrian dict |
| `waypoint_extractor.py` | `map.generate_waypoints(2.0)` | Lane/Road/Junction dict + 拓扑关系 |
| `trafficlight_extractor.py` | `world.get_actors().filter('traffic.*')` | TrafficLight dict 列表 |
| `weather_extractor.py` | `world.get_weather()` | 天气属性 dict |
| `sensor_extractor.py` | Collision/LaneInvasion 传感器 | 碰撞 + 车道入侵事件 |
| `pipeline.py` | 编排五类提取器 | 合并为单一 FrameData dict |

**`process_frame(frame_data)`** 串联所有提取器，输出兼容下游 scenario 模块的字典格式。

---

### 6️⃣ 存储层 (Storage)

**位置**：`stk/storage/`

**双后端持久化**：

| 后端 | 适用场景 | 核心模块 |
|------|---------|---------|
| **Neo4j** | 生产环境 | `writer.py` (批量 MERGE), `connector.py`, `schema.py` |
| **JSON 分片** | 开发/无 Neo4j | `serializer.py` (序列化全图) |

**Neo4j Schema**：

- **13 种节点标签**：Vehicle, Pedestrian, TrafficLight, RoadElementEntity, EnvSnapshot, SceneSnapshot, Maneuver, Interaction, Rule, Param, SafetyViolation, Responsibility, AttrVersion
- **35 种关系类型**：15 场景 + 13 行为 + 7 规则（不含 hasVersion）

**批量化写入**：`write_entity_batch` / `write_relation_batch` 使用 `UNWIND $batch AS row MERGE ...` 模板，比逐条 MERGE 降低 ~70% 写延迟。

**查询接口（7 种常用）**：

| 查询 | 函数 | 用途 |
|------|------|------|
| 按帧时间切片 | `time_slice_query(frame_id)` | 单帧快照 |
| 车辆全生命周期 | `lifecycle_query(vehicle_id)` | 属性版本追踪 |
| 异常追溯 | `anomaly_trace_query(sv_id)` | 证据链回溯 |
| 时空聚合 | `spatiotemporal_aggregate_query(t_s, t_e)` | 窗口统计 |
| 子图导出 | `spatiotemporal_subgraph_query(t_s, t_e, road_id)` | GNN 输入 |
| GNN 训练导出 | `export_for_gnn_cypher(t_s, t_e, road_id)` | K-HSTGAN 输入 |
| 时态属性查询 | `temporal_attr_query(eid, t_s, t_e)` | 属性时间旅行 |

---

### 7️⃣ 滤波层 (Filter)

**位置**：`stk/filter/`

**6 个正交滤波器**（阶段 1–3 增量引入）：

| 滤波器 | 模块 | 功能 |
|--------|------|------|
| **笛卡尔椭圆 ROI** | `roi.py` | 以 ego 为中心的椭圆区域过滤，差异化半径（轿车 70m/行人 40m 等） |
| **EgoCentricFilter** | `generator.py` | 整合 ROI + 类别半径 + 滞回逻辑 |
| **生命周期跟踪** | `lifecycle.py` | ENTER → UPDATE → EXIT → FORGET 状态机 |
| **重要性打分** | `importance.py` | E1–E5 五维加权（ego 权重 0.40, 距离 0.20, 可见性 0.15, 交互 0.15, 异常 0.10）|
| **边稀疏化** | `edge_pruner.py` | 同类边按 importance top-k 保留 |
| **静态背景外移** | `background_filter.py` | Lane 节点/边不进 KG，信息平铺到 Vehicle.attrs 中 |

**Ego-Centric ROI 配置**：

```yaml
config/ego_centric.yaml
enabled: true
radius_front: 70.0     # 正向 70m
radius_rear: 30.0      # 后方 30m
radius_side: 50.0      # 侧向 50m
radii_by_category:
  car:        {front: 70, rear: 30, side: 50}
  motorcycle: {front: 50, rear: 25, side: 35}
  bicycle:    {front: 50, rear: 25, side: 35}
pedestrian_radius_front: 40
```

---

### 8️⃣ 流水线编排 (Pipeline)

**位置**：`stk/pipeline/` + `scripts/long_run/`

**单场景流水线** (`PipelineOrchestrator`)：

1. 加载场景帧数据 → `extraction.pipeline.process_frame()` 提取
2. `IncrementalEngine.process_frame()` 增量更新
3. `RuleEnforcer.enforce()` 规则推理
4. 存入 `SnapshotStore`

**长时流式采集** (`scripts/long_run/`)：

| 脚本 | 功能 |
|------|------|
| `collect.py` | 长时间 Phase1 采集（默认 24000 帧/20 分钟），分块输出 chunk_XXXX.json |
| `pipeline.py` | 跨分块编排 Phase2→3→4→5，复用 IncrementalEngine / BehaviorRelationGenerator / RuleEnforcer 状态 |
| `anomaly_scheduler.py` | 7 种异常类型泊松过程调度 + `bind_targets` 按车道筛选 |
| `run_e2e_5min.sh` | 端到端一键运行脚本（5 分钟，ego-centric） |
| `import_neo4j.py` | 将 phase5 产出导入 Neo4j |

**异常注入类型**：

| 异常 ID | 名称 | 持续帧数 | 描述 |
|---------|------|---------|------|
| `sudd_brk` | 前车急刹 | 100 | ego 跟车前车突然刹车 |
| `sudd_stp` | 前车急停 | 200 | 前车完全停止 |
| `avd_col` | 紧急避让 | 100 | 邻车突然变道切入 |
| `jun_ny` | 路口不让行 | 100 | 背景车不减速通过路口 |
| `rev_drive` | 逆向行驶 | 150 | 背景车反向行驶 |
| `ped_crs` | 行人横穿 | 120 | walker 朝 ego 行走 |
| `obs_blk` | 视线遮挡 | 200 | 前方放置障碍物 |

---

### 9️⃣ 可视化 (Visualization)

**位置**：`stk/viz/`

| 模块 | 功能 |
|------|------|
| `anomaly_replay.py` | 异常事件回放 + Cypher 证据链输出 |
| `birds_eye.py` | 鸟瞰图渲染（CARLA 表面渲染 → PNG bytes）|
| `kg_dashboard.py` | KG 仪表盘（主版本）|
| `kg_dashboard_ref.py` | KG 仪表盘（参考实现）|

---

## 数据流总览

```
[CARLA Tick t]
       │
       ▼
[extraction/ 6 类提取器]  ← `process_frame()`
       │  vehicles[], pedestrians[], traffic_lights[], lanes[], weather{}
       ▼
[scenario/snapshot_builder.build_snapshot]
       │  ScenarioSnapshot(t) + EnvSnapshot(t) + 6 类实体 + containsX 关系
       ▼
[scenario/spatial.compute_*]
       │  in_lane × N, ahead_of × M, beside × K, ...
       ▼
[behavior/BehaviorRelationGenerator.generate]
       │  Maneuver[] + Interaction[] + 13 种行为关系 + 跨层桥接
       ▼
[rules/RuleEnforcer.enforce]
       │  RSS 违规 + R1-R18 交规 + SafetyViolation + 证据链 + 责任归因
       ▼
[storage/serializer.serialize_graph]
       │  4 层节点 + 边 → Neo4j MERGE / JSON 分片
       ▼
[Neo4j / JSON 文件]
       │
       ▼
[downstream]
   ├── 3D 可视化 Dashboard (viz/)
   ├── GNN 异常检测导出 (queries.export_for_gnn_cypher)
   └── 规则验证与回放 (replay.replay_violation)
```

---

## 快速开始

### 环境要求

- Python 3.10+
- CARLA 0.9.16 仿真器（可选，离线运行场景库无需 CARLA）
- Neo4j 5.x（可选，支持 JSON 分片模式）

### 安装

```bash
# 1. 创建 conda 环境
conda create -n stk python=3.10 -y
conda activate stk

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .

# 3. 验证安装
pytest tests/test_smoke.py -v
stk --help
```

### 离线运行基本测试

```bash
# 运行全部 14 个场景的端到端 pipeline
python -m stk.pipeline.runner --scenario all

# 指定单场景
python -m stk.pipeline.runner --scenario S10
```

### 启动长时采集（需 CARLA）

```bash
# 1. 启动 CARLA 服务器
# 2. 运行采集
python scripts/long_run/collect.py \
    --host localhost --port 2000 \
    --town Town10HD \
    --total-frames 24000 \
    --chunk-frames 2000 \
    --vehicles 20 --walkers 8 \
    --out data/long_run/Town10HD_$(date +%Y%m%d_%H%M%S)

# 3. 运行 pipeline（从 chunk 构建 KG）
python scripts/long_run/pipeline.py \
    --run-dir data/long_run/Town10HD_*

# 或一键端到端
bash scripts/long_run/run_e2e_5min.sh
```

---

## 配置说明

所有配置文件位于 `config/` 目录：

| 配置文件 | 控制内容 |
|---------|---------|
| `pipeline.yaml` | 流水线阶段开关、最大帧数、场景列表、输出目录 |
| `ego_centric.yaml` | Ego-Centric ROI 半径、类别差异化半径、重要性权重、过滤开关 |
| `rss_rules.yaml` | RSS 7 个参数（ρ, a_max_accel, a_min_brake, a_brake 等）|
| `traffic_rules.yaml` | 18 条交规触发阈值（速度、距离、持续帧数）|
| `ontology.yaml` | 本体配置 |
| `neo4j.yaml` | Neo4j 连接信息（URI、认证）|

阈值实现在 `stk/config.py` 的 `ThresholdConfig` dataclass，包含 lane/vehicle/pedestrian/junction/occlusion/traffic_rule/rss 七组阈值，均可在运行时动态调整。

---

## 运行方式

### CLI 接口

```bash
stk --help               # 查看所有命令
stk run scenario S10     # 运行单场景
stk run all              # 运行所有场景
stk run pipeline         # 运行流水线
```

### 关键脚本

| 脚本 | 用途 |
|------|------|
| `scripts/long_run/collect.py` | CARLA 长时间数据采集 |
| `scripts/long_run/pipeline.py` | 跨 chunk KG 构建 |
| `scripts/long_run/build_anomaly_dataset.py` | 构建异常检测数据集 |
| `scripts/long_run/import_neo4j.py` | 导入 Neo4j |
| `scripts/pipeline/run_phases_1_5.py` | 单次运行 Phase1→5 全流程 |
| `scripts/pipeline/cross_validate.py` | 交叉验证 |
| `scripts/pipeline/smoke_test.py` | 冒烟测试 |
| `scripts/replay/render_scenario_gif.py` | 场景回放 GIF 生成 |
| `scripts/replay/render_kg_dashboard_gif.py` | KG 仪表盘 GIF |
| `scripts/viz/shard_graph_for_viz.py` | 可视化图分片导出 |

---

## 测试矩阵

25 个测试文件，全部可在无 CARLA 环境下运行：

| 测试文件 | 覆盖模块 | Case 数 |
|---------|---------|---------|
| `test_ontology.py` | 7 条公理 + 类型系统 | 31 |
| `test_scenario.py` | 场景节点/关系/空间计算 | ✅ |
| `test_scenario_library.py` | 14 个场景工厂 | ✅ |
| `test_scenario_egocentric.py` | 场景层 ego×ROI | ✅ |
| `test_behavior.py` | 行为检测器/生成器 | ✅ |
| `test_behavior_egocentric.py` | 行为层 ego×ROI | ✅ |
| `test_debouncer.py` | 防抖状态机 | 18 |
| `test_rules.py` | 规则引擎 | ✅ |
| `test_rules_rss.py` | RSS 安全距离 | ✅ |
| `test_rules_traffic.py` | R1–R18 交规 | ✅ |
| `test_rules_regression.py` | 规则回归 | ✅ |
| `test_rules_integration.py` | 规则集成 | ✅ |
| `test_dynamic.py` | 增量引擎/版本管理 | ✅ |
| `test_incremental_resume.py` | checkpoint 恢复 | ✅ |
| `test_extraction.py` | 6 类提取器 | ✅ |
| `test_storage.py` | 序列化/写入/查询 | ✅ |
| `test_serializer_filtering.py` | 全链路滤波 | ✅ |
| `test_ego_centric_filter.py` | EgoCentricFilter | 26 |
| `test_importance_scorer.py` | E1–E5 打分 | ✅ |
| `test_edge_pruner.py` | 边稀疏化 | ✅ |
| `test_background_filter.py` | 背景外移 | ✅ |
| `test_lifecycle_tracker.py` | 生命周期跟踪 | ✅ |
| `test_pipeline.py` | 流水线编排 | ✅ |
| `test_smoke.py` | 烟囱测试 | ✅ |
| `test_longrun_shard_output.py` | 长时运行分片输出 | ✅ |

运行全部测试：

```bash
pytest tests/ -v --ignore=tests/test_pipeline.py  # 499 passed（无 Neo4j 时跳 6）
```

---

## 项目结构

```
SpatioTemporalKG/
├── stk/                           # 核心 Python 包
│   ├── __init__.py
│   ├── config.py                  # 配置加载 + ThresholdConfig + EgoCentricConfig
│   ├── cli.py                     # CLI 入口
│   ├── ontology/                  # 本体层
│   │   ├── entity.py              #   BaseEntity 基类
│   │   ├── relation.py            #   BaseRelation 基类
│   │   ├── types.py               #   14 实体类型 + 42 关系类型枚举
│   │   ├── namespace.py           #   命名空间 + ID 生成器
│   │   ├── axioms.py              #   7 条核心公理 A1-A7
│   │   ├── lifecycle.py           #   节点生命周期四状态机
│   │   ├── temporal_triple.py     #   时态三元组七元组
│   │   └── __init__.py
│   ├── scenario/                  # 场景层
│   │   ├── snapshot_builder.py    #   FrameData + build_snapshot
│   │   ├── spatial.py             #   空间关系纯函数计算
│   │   ├── lifecycle_manager.py   #   生命周期管理器
│   │   ├── scenario_library.py    #   14 个预置场景
│   │   ├── nodes.py               #   6 类场景节点
│   │   ├── relations.py           #   15 种场景关系
│   │   └── __init__.py
│   ├── behavior/                  # 行为层
│   │   ├── generator.py           #   BehaviorRelationGenerator
│   │   ├── detectors.py           #   11 个行为检测器
│   │   ├── debouncer.py           #   防抖状态机
│   │   ├── manifest.py            #   跨层桥接
│   │   ├── nodes.py               #   ManeuverNode + InteractionEvent
│   │   ├── relations.py           #   13 种行为关系
│   │   └── __init__.py
│   ├── rules/                     # 规则层
│   │   ├── generator.py           #   RuleEnforcer 主驱动
│   │   ├── nodes.py               #   4 类规则节点
│   │   ├── relations.py           #   7 种规则关系
│   │   ├── rss/                   #   RSS 子层
│   │   │   ├── model.py           #     RSS 三个核心算子
│   │   │   └── __init__.py
│   │   ├── traffic/               #   交规子层
│   │   │   ├── rules.py           #     14 条 check_Ri_* 函数
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── dynamic/                   # 动态更新
│   │   ├── incremental_updater.py #   IncrementalEngine 五步流程
│   │   ├── diff.py                #   DeltaGraph + DiffSet
│   │   ├── version.py             #   VersionManager + AttrVersion
│   │   ├── time_window.py         #   滑动窗口聚合器
│   │   ├── event_injector.py      #   规则事件反向插入
│   │   ├── snapshot_store.py      #   快照存储
│   │   └── __init__.py
│   ├── extraction/                # CARLA 数据提取
│   │   ├── pipeline.py            #   提取编排
│   │   ├── actor_extractor.py     #   车辆/行人提取
│   │   ├── waypoint_extractor.py  #   路网拓扑提取
│   │   ├── trafficlight_extractor.py # 信号灯提取
│   │   ├── sensor_extractor.py    #   传感器事件
│   │   ├── weather_extractor.py   #   天气提取
│   │   ├── api_mapping.py         #   API 映射表
│   │   └── __init__.py
│   ├── filter/                    # 滤波层（Ego-Centric）
│   │   ├── roi.py                 #   笛卡尔椭圆判定
│   │   ├── generator.py           #   EgoCentricFilter
│   │   ├── lifecycle.py           #   生命周期跟踪
│   │   ├── importance.py          #   重要性打分 E1-E5
│   │   ├── edge_pruner.py         #   边稀疏化
│   │   ├── background_filter.py   #   静态背景外移
│   │   └── __init__.py
│   ├── pipeline/                  # 流水线编排
│   │   ├── orchestrator.py        #   PipelineOrchestrator
│   │   ├── runner.py              #   pipeline runner
│   │   ├── checkpoint.py          #   checkpoint 管理器
│   │   └── __init__.py
│   ├── storage/                   # 存储层
│   │   ├── connector.py           #   Neo4j 连接
│   │   ├── schema.py              #   Neo4j Schema 定义
│   │   ├── serializer.py          #   全图序列化（核心）
│   │   ├── writer.py              #   批量写入
│   │   ├── importer.py            #   图谱导入
│   │   ├── queries.py             #   7 种查询接口
│   │   ├── replay.py              #   违规回放
│   │   └── __init__.py
│   ├── viz/                       # 可视化
│   │   ├── anomaly_replay.py      #   异常回放
│   │   ├── birds_eye.py           #   鸟瞰图渲染
│   │   ├── kg_dashboard.py        #   KG 仪表盘
│   │   ├── kg_dashboard_ref.py    #   KG 仪表盘（参考）
│   │   └── __init__.py
│   └── README.md                  # 包内说明
├── config/                        # YAML 配置文件
│   ├── pipeline.yaml
│   ├── ontology.yaml
│   ├── ego_centric.yaml
│   ├── rss_rules.yaml
│   ├── traffic_rules.yaml
│   └── neo4j.yaml
├── tests/                         # 25 个测试文件
│   ├── conftest.py
│   ├── test_smoke.py
│   ├── test_ontology.py           # 31 个公理验证 case
│   ├── test_debouncer.py          # 18 个防抖 case
│   ├── test_ego_centric_filter.py # 26 个 ROI case
│   ├── ... (共 25 个)
├── scripts/                       # 运维与实验脚本
│   ├── long_run/                  # 长时采集与 pipeline
│   │   ├── collect.py
│   │   ├── pipeline.py
│   │   ├── anomaly_scheduler.py
│   │   ├── build_anomaly_dataset.py
│   │   ├── import_neo4j.py
│   │   ├── run_e2e_5min.sh
│   │   └── run_phase5_shard.sh
│   ├── pipeline/                  # 流水线脚本
│   │   ├── smoke_test.py
│   │   ├── cross_validate.py
│   │   ├── run_phases_1_5.py
│   │   └── batch_collect.py
│   ├── carla/                     # CARLA 交互脚本
│   │   ├── run_extraction.py
│   │   ├── spawn_traffic.py
│   │   └── process_manager.py
│   ├── replay/                    # 回放脚本
│   │   ├── build_graph_from_frames.py
│   │   ├── build_replay_from_scenario.py
│   │   ├── render_scenario_gif.py
│   │   └── render_kg_dashboard_gif.py
│   ├── viz/                       # 可视化工具
│   │   └── export_viz_data.py
│   ├── query/                     # 查询脚本
│   │   └── query_anomaly.py
│   └── remote/                    # 远程服务器脚本
├── map_configs/                   # 地图配置
│   ├── Town01.yaml
│   ├── Town02.yaml
│   ├── Town04.yaml
│   ├── Town05.yaml
│   └── Town10HD.yaml
├── docs/                          # 文档
│   ├── thesis/                    # 毕业论文分章
│   │   ├── chapter1_*.md
│   │   ├── chapter2_*.md
│   │   ├── chapter3_*.md         # “时空动态知识图谱构建” (7 文件)
│   │   ├── chapter4_*.md
│   │   ├── chapter5_*.md
│   │   ├── PLAN_thesis_and_paper.md
│   │   └── HANDOVER.md
│   └── ego_centric_pipeline.md    # Ego-Centric 改动完整文档
├── docker/                        # Docker 部署
│   └── neo4j/
│       └── docker-compose.yml
├── notebooks/                     # Jupyter 探索笔记本
├── data/                          # 运行数据 (gitignored)
├── logs/                          # 运行日志 (gitignored)
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## 关键设计决策

### 1. 节点+边双轨表达

同一语义对象同时以**节点**和**关系边**形式存在。例如 `following(v_A, v_B)` 既是一条关系边，也对应一个 `InteractionEvent` 节点，二者通过 `manifestsAs` 边绑定。三重收益：图遍历灵活、属性承载分工清晰、跨层桥接可解释。

### 2. 防抖状态机

行为层最关键的设计。每种行为关系有独立进入/消失阈值（1–5 帧），`RelationDebouncer` 维护 `(src, dst, type)` 三元组的 `on_counter` / `off_counter`，在持续满足条件后才创建/删除行为节点，消除单帧浮点抖动。

### 3. 四层本体递进

场景层提供"几何确定性"的空间关系基座 → 行为层通过防抖聚合为行为语义 → 规则层进行物理安全与法规合规判断。每层的输出既是下一层的输入，也是违规证据链的源头。

### 4. 差分图驱动增量

用 `Δg_t` 的四类更新动作（实体/属性/关系/规则事件）替代全图重建。`IncrementalEngine` 的五步流程维护跨帧状态一致性，checkpoint 序列化支持长时分块采集中的断点恢复。

### 5. Ego-Centric 裁剪

围绕自车建立笛卡尔椭圆 ROI，通过**重要性打分 + 边稀疏化 + 静态背景外移**三个正交优化器，将长时运行的全图规模压缩 ≥95%（从 ~5.4M 边降至可管理规模），且所有滤波步骤为纯函数不可变转换。

### 6. 双后端持久化

Neo4j 批量 MERGE（生产环境）+ JSON 分片（开发环境/无需数据库），JSON 分片可直接为下游 GNN 模型提供批量帧切片。

### 7. 属性版本化

`VersionManager` 管理每个属性的版本链，支持任意时点的"时间旅行"查询——给定帧号即可还原该帧任意实体的属性值，无需重放原始仿真。

---

## 论文与引用

本项目的核心设计在以下论文中详细描述（见 `docs/thesis/`）：

- **第 3 章**：时空动态知识图谱构建（7 个文件覆盖形式化定义/本体/场景层/行为层/规则层/动态更新/流式采集）
- **第 4 章**：基于 K-HSTGAN 的时序异常检测
- **第 5 章**：KS-NBCF 融合框架

设计文档原始版本：`docs/v3_paragraphs.txt`（7 章完整设计）

---

## 开发阶段

| 阶段 | 名称 | 对应模块 |
|------|------|---------|
| 0 | 项目骨架与环境就绪 | — |
| 1 | 本体层 | `stk/ontology/` |
| 2 | 场景层 | `stk/scenario/` |
| 3 | 行为层 | `stk/behavior/` |
| 4 | 规则层（RSS + 交规 R1-R18） | `stk/rules/` |
| 5 | 动态更新 | `stk/dynamic/` |
| 6 | Neo4j 存储 | `stk/storage/` |
| 7 | CARLA 提取 | `stk/extraction/` |
| 8 | 集成流水线 | `stk/pipeline/` + `scripts/` |
| 9 | 滤波层 (Ego-Centric) | `stk/filter/` |
