# 3.3 场景层：实体与空间关系提取

场景层是 STKG 的最底层建模单元，回答"此刻这个世界长什么样"。它的输入是 CARLA 仿真器经由 `stk/extraction/` 六类提取器得到的原始观测数据，输出是本帧内所有物理实体的属性描述以及它们之间的空间拓扑关系。场景层设计遵循三个核心原则：（1）**可提取性**——每个字段必须在 CARLA 0.9.16 API 中有确定数据源；（2）**几何确定性**——每条空间关系可由仿真真值直接计算，无需依赖学习或推理；（3）**零防抖**——场景层为瞬时关系，不存在跨帧防抖需求（防抖只在行为层出现）。

## 3.3.1 节点类型与属性设计

场景层共定义 **8 类节点**（其中 6 类独立、2 类聚合辅助），覆盖自动驾驶场景中的全部物理要素。

### 3.3.1.1 动态实体：VehicleEntity

`VehicleEntity`（Neo4j Label: `Vehicle`）对应 CARLA 中 `carla.Vehicle`，是场景中最主要的行为主体。它携带 **18 个属性字段**，分为四组：

**① 标识组**：`vehicle_id`（`carla.Actor.id`）、`vehicle_type`（`carla.Vehicle.type_id`，如`"vehicle.nissan.patrol"`）、`is_ego`（是否自车）。

**② 位置/运动组**（直接来自 CARLA）：`location_x/y/z`（`get_location()`，单位 m）、`velocity_x/y/z`（`get_velocity()`，单位 m/s）、`acceleration_x/y/z`（`get_acceleration()`，单位 m/s²）。

**③ 派生计算组**（自原生属性计算）：`speed`（$\|\vec{v}\|$，m/s）、`speed_kmh`（$3.6 \times \text{speed}$，km/h）、`heading_rad`（由`get_transform().rotation.yaw`换算为弧度）、`pitch`、`roll`。

**④ 物理与状态组**：`bbox_extent_x/y/z`（包围盒半长，m）、`throttle`、`brake`、`steer`（控制信号，$[0,1]$）、`is_alive`（布尔）。

18 个字段中 14 个来自 CARLA 原生 API（无需计算），4 个为派生量。这种"天然—计算"的分治降低了提取模块与仿真器的耦合度——原生字段的提取器与派生装置可独立维护。

### 3.3.1.2 动态实体：PedestrianEntity

`PedestrianEntity`（Neo4j Label: `Pedestrian`）对应 CARLA `carla.Walker`，携带 **13 个属性**。除位置/速度/包围盒等与 VehicleEntity 结构相似的字段外，另有三个行人的特有字段：`action`（行为类别，如`"walking"`、`"running"`等来自 `carla.WalkerBoneControl`）、`is_on_crosswalk`（通过行人所在位置与斑马线多边形做空间包含判断，利用 `shapely.geometry.Polygon.contains()` 实现）、`is_on_sidewalk`（通过 `carla_map.get_waypoint(ped_location).lane_type` 是否为 `Sidewalk` 判定）。

### 3.3.1.3 控制设备：TrafficLightEntity

`TrafficLightEntity`（Neo4j Label: `TrafficLight`）对应 CARLA `carla.TrafficLight`，携带 **7 个属性**。其中关键字段 `state` 取值为 `{"Red", "Yellow", "Green"}`（CARLA 原生 `carla.TrafficLightState`）；`elapsed_time` 表示当前颜色已持续时间（s）；`affected_lane_ids` 是一个列表字段，记录该信号灯控制的所有车道 ID 集合。该集合并非 CARLA 直接提供，而是需要在地图加载时预处理：遍历 `carla_map.generate_waypoints(2.0)` 并为每个 waypoint 检查是否有 `waypoint.traffic_light` 关联，从而建立 $TL \to \{\text{lane\_id}\}$ 映射。

### 3.3.1.4 静态设施：RoadElementEntity

`RoadElementEntity`（Neo4j 多标签支持 `Lane`、`Road`、`Junction`）来自 CARLA `carla.Waypoint` 拓扑查询。每个 `Lane` 节点携带 **13 个属性**，涵盖唯一标识（`road_id`、`lane_id`、`junction_id`）、几何（`center_x/y/z`、`heading_rad`、`lane_width`、`speed_limit`）、拓扑关系（`left_lane_id`、`right_lane_id`、`has_traffic_light`）和类型信息（`lane_type`，对应 `carla.LaneType` 枚举，包括 `Driving`、`Sidewalk`、`Shoulder` 等）。

关于 `lane_id` 的符号约定：正数表示道路默认方向车道，负数表示对向车道——此约定对 3.5 节逆行检测（R5）至关重要。`junction_id` 等于 `-1` 表示不在路口内，正数表示所属路口 ID。

### 3.3.1.5 环境与聚合节点

**`EnvironmentSnapshot`**（Neo4j Label: `EnvSnapshot`）采集每帧的全局环境状态，携带 **12 个字段**：`frame_id`、`elapsed_seconds`、`delta_seconds`、`map_name`、`fog_density`、`cloudiness`、`precipitation`、`wetness`、`sun_altitude_angle`、`wind_intensity`、`random_seed`、`traffic_density`。其中 `wetness`、`fog_density`、`sun_altitude_angle` 等环境量与 3.5 节恶劣天气限速规则（R11）、弱势参与者保护规则（R8）直接相关。

**`ScenarioSnapshot`**（Neo4j Label: `SceneSnapshot`）是每帧唯一的聚合根节点，包含 `frame_id`、`elapsed_seconds`、`n_vehicles`、`n_pedestrians` 四个字段。该节点通过 `containsVehicle`、`containsPedestrian`、`containsTrafficLight`、`containsRoad`、`hasEnvironment` 五种帧聚合边与同帧所有场景层实体及环境节点一一连接，形成"以帧为根、以实体为叶"的树状结构。

表 3-3 汇总全部场景层节点的属性和字段数。

**表 3-3** 场景层节点属性统计
[三线表]

| 节点类 | Neo4j Label | 属性数 | 原生字段 | 派生字段 | 特有字段 |
|--------|-------------|--------|---------|---------|---------|
| VehicleEntity | `Vehicle` | 18 | 14 | 4 | is_ego, throttle, brake, steer |
| PedestrianEntity | `Pedestrian` | 13 | 10 | 3 | action, is_on_crosswalk, is_on_sidewalk |
| TrafficLightEntity | `TrafficLight` | 7 | 5 | 2 | state, elapsed_time, affected_lane_ids |
| RoadElementEntity | `Lane`/`Road`/`Junction` | 13 | 10 | 3 | lane_id, lane_type, left/right_lane_id |
| EnvironmentSnapshot | `EnvSnapshot` | 12 | 8 | 4 | weather 六要素 + time_of_day |
| ScenarioSnapshot | `SceneSnapshot` | 4 | 4 | 0 | 帧统计 |

## 3.3.2 空间关系计算

场景层关系描述"实体之间在同一帧内的空间、拓扑、包含关系"。所有关系均通过 `stk/scenario/spatial.py` 中的纯函数计算，满足前文所述的"几何确定性"原则。

### 3.3.2.1 拓扑关系

`in_lane`：给定车辆 $v$ 的位置坐标 $\mathbf{p}_v(t)$ 与车道集合 $\{L_1, \dots, L_k\}$，$v$ 所在车道由最近车道匹配确定。计算 $v$ 到每条车道中心线的横向距离 $d_{\perp}(v, L_i)$，取 $i^* = \arg\min_i d_{\perp}$。若 $d_{\perp}(v, L_{i^*}) < \epsilon_{\text{lane}}$（默认 $\epsilon_{\text{lane}}=2.0$ m，由配置 `config/rss_rules.yaml` 控制），则建立关系 `in_lane(v, L_{i^*}, t)`，边属性 `distance_to_lane_center = d_{\perp}`。

`adjacent_lane`：车道 $L_i$ 与 $L_j$ 相邻当且仅当 $L_i.\text{left\_lane\_id} = L_j.\text{lane\_id}$ 或 $L_i.\text{right\_lane\_id} = L_j.\text{lane\_id}$。该关系仅依靠 RoadElementEntity 的拓扑属性，不需几何计算。

`on_road` / `in_junction`：利用车道--路段--路口三级的包含关系派生。若 $v$ 所在车道的 `junction_id != -1`，则建立 `in_junction(v, J, t)`；否则建立 `on_road(v, R, t)$，其中 $R$ 为 $v$ 所在路段的聚合 Road 节点。

### 3.3.2.2 空间关系

`ahead_of`：两车 $v_A$、$v_B$ 在同一车道（`in_lane`）且 $v_A$ 在 $v_B$ 前方时建立 `ahead_of(v_A, v_B, t)`。判定条件是：两车纵向距离 $\Delta s = \text{longitudinal\_distance}(v_A, v_B) > 0$（正表示 $A$ 在 $B$ 前方）且横向偏移 $|d_{\perp}(v_A, L) - d_{\perp}(v_B, L)| < w_{\text{lane}}/2$。关系边携带属性 `longitudinal_distance`、`lateral_distance`。

`beside`：两车 $|d_{\text{lat}}(v_A, v_B)| < d_{\text{lat}}^{\text{max}}$（默认 3.0 m）且 $|\Delta s| < d_{\text{long}}^{\text{max}}$（默认 5.0 m）时建立 `beside(v_A, v_B, t)$。该关系是检测并行行驶、超车等行为的必要前置条件。

`nearby_pedestrian`：车辆 $v$ 与行人 $p$ 的欧氏距离 $\|\mathbf{p}_p - \mathbf{p}_v\| < d_{\text{ped}}^{\text{max}}$（默认 20.0 m）时建立。该关系是后续行为检测中 `detect_yielding_to` 和 `detect_approaching_pedestrian` 的主要输入来源。

### 3.3.2.3 控制关系

`controlled_by`：车道 $L$ 被信号灯 $T$ 控制当且仅当 $T.\text{affected\_lane\_ids}$ 包含 $L.\text{lane\_id}$。该关系在首次加载地图时通过预处理建立（`build_traffic_light_lane_map`），之后不再变动，但在全局生效期间属性 `valid_from_frame = 0` 标记为"始终有效"。

### 3.3.2.4 帧聚合关系

`containsVehicle` / `containsPedestrian` / `containsTrafficLight` / `containsRoad` / `hasEnvironment`：五类帧聚合关系统一把 `ScenarioSnapshot`（帧根节点）与同帧所有实例连通。每种 `containsX` 关系边携带 `frame_id` 属性。其中 `weather_context` 是 `EnvironmentSnapshot` 与 `ScenarioSnapshot` 之间的双向语境边，表达"该帧的环境上下文"。

### 3.3.2.5 对比与总结

**表 3-4** 场景层关系计算模式对比
[三线表]

| 关系类型 | 计算复杂度 | 准确率特征 | 特殊依赖 |
|---------|-----------|-----------|---------|
| `in_lane` | $O(V \cdot k)$ | 取决于 $\epsilon_{\text{lane}}$ 设定 | 预构建车道中心线数据集 |
| `ahead_of` | $O(N^2)$ 优化为 $O(N)$ | 依赖 `in_lane` 的正确性 | `in_lane` 先决 |
| `beside` | $O(N^2)$ | 阈值敏感，易受车道宽度影响 | — |
| `controlled_by` | $O(1)$ | 100%（预处理确定） | 地图加载时预处理 |
| `containsX` | $O(N)$ | 100% | 固定映射 |

场景层共计 15 种关系类型，每帧平均生成 $4.2 \times 10^3$ 条关系边（以 20 辆车、8 个行人、12 盏信号灯、40 条车道、5 个路口为典型规模）。这些关系边构成了行为层与规则层推理的几何基础。

## 3.3.3 快照构建与生命周期管理

`stk/scenario/snapshot_builder.py` 的 `build_snapshot(FrameData)` 函数接收一帧完整的原始数据（`FrameData`），将其转化为 `ScenarioSnapshot + EnvironmentSnapshot` 双根结构。关键步骤为：

1. 以 `ScenarioSnapshot(frame_id, ...)` 作为帧根节点；
2. 以 `EnvironmentSnapshot(frame_id, weather...)` 作为环境节点；
3. 将所有实体（车辆、行人、信号灯、道路元素逐个构建实例）通过五种 `containsX` 边接入帧根；
4. 通过 `hasEnvironment` 边接入环境节点；
5. 返回 `(ScenarioSnapshot, EnvironmentSnapshot)` 二元组。

与此同时，`LifecycleManager` 对每帧检测到的动态实体 ID 集合与上一帧 ID 集合做差集，输出每个实体的生命周期状态转换 `{"activated"|"deactivated"|"stable"|"created"}`。该状态是对应节点下一帧是否需要执行 CEATE 或 DEACTIVATE 操作的唯一依据。

## 3.3.4 提取器模块

`stk/extraction/` 包含六类提取器，各自从 CARLA 不同类型收据源提取并封装原始数据：

| 提取器 | 对应 CARLA API | 输出结构 |
|--------|---------------|---------|
| `ActorExtractor` | `world.get_actors().filter('vehicle.*')` / `'walker.*'` | VehicleEntity、PedestrianEntity 二元组 |
| `TrafficLightExtractor` | `world.get_actors().filter('traffic.*')` | TrafficLightEntity 列表 |
| `WaypointExtractor` | `carla_map.generate_waypoints(2.0)` | RoadElementEntity (Lane/Road/Junction) 列表 + 灯-车道映射表 |
| `WeatherExtractor` | `world.get_weather()` | WeatherSnapshot 属性字典 |
| `SensorExtractor` | Collision/LaneInvasion 传感器回调 | 碰撞与车道入侵事件 |
| `Pipeline`（编排器） | 将五者并行提取结果合并为 `FrameData` | 完整的 `FrameData` 结构 |

所有提取器由 `extraction/pipeline.py` 中的编排器统一调度，使用 `multiprocessing.pool.ThreadPool` 实现并行拉取，在 50 ms 帧循环窗口内完成全部数据提取。

## 3.3.5 小结

本节详述了 STKG 场景层的设计：6 类实体节点承载 68 个属性字段（原生 + 派生），15 种空间关系通过纯函数计算实现几何确定性。场景层的输出（实体属性 + 空间关系 + 帧聚合结构）是行为层检测与规则层推理的唯一下游数据源。其"可提取性"和"零防抖"的设计原则保证了约束链底部的高可靠性与低延迟。
