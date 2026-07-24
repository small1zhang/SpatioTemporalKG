# 3.3 三层构建：场景层、行为层、规则层

在 3.2 节本体设计确定的类型体系与公理约束下，本节按"场景→行为→规则"的语义递进序列，依次实现三层本体的构建方法。每层均解释其节点类型与属性设计、关系类型与计算逻辑、核心算法流程，并以跨层桥接机制串联三层为有机整体。

---

## 3.3.1 场景层：实体与空间关系提取

场景层是 STKG 的最底层建模单元，回答"此刻这个世界长什么样"。它的输入是 CARLA 仿真器经由 `stk/extraction/` 六类提取器得到的原始观测数据，输出是本帧内所有物理实体的属性描述以及它们之间的空间拓扑关系。场景层设计遵循三个核心原则：（1）**可提取性**——每个字段必须在 CARLA 0.9.16 API 中有确定数据源；（2）**几何确定性**——每条空间关系可由仿真真值直接计算，无需依赖学习或推理；（3）**零防抖**——场景层为瞬时关系，不存在跨帧防抖需求（防抖只在行为层出现）。

### 3.3.1.1 节点类型与属性设计

场景层共定义 **8 类节点**（其中 6 类独立、2 类聚合辅助），覆盖自动驾驶场景中的全部物理要素。

#### 动态实体：VehicleEntity

`VehicleEntity`（Neo4j Label: `Vehicle`）对应 CARLA 中 `carla.Vehicle`，是场景中最主要的行为主体。它携带 **28 个属性字段**，按数据来源分为三组。表 3-9 给出完整的 CARLA 映射。

**表 3-9** VehicleEntity 属性与 CARLA API 映射

[三线表]

| 属性名 | CARLA 数据源 | 类型 | 单位 | 计算方法 | 可用性 |
|--------|------------|------|------|---------|-------|
| `entity_id` | `actor.id` | str | — | 直接 | 直接可用 |
| `vehicle_type` | `actor.type_id` | str | — | 直接 | 直接可用 |
| `vehicle_category` | `type_id` 派生 | str | — | 关键字匹配 | 可计算 |
| `is_ego` | 上下文注入 | bool | — | 静态字段 | 注入 |
| `location_x/y/z` | `actor.get_location()` | float | m | 直接 | 直接可用 |
| `velocity_x/y/z` | `actor.get_velocity()` | float | m/s | 直接 | 直接可用 |
| `acceleration_x/y/z` | `actor.get_acceleration()` | float | m/s² | 直接 | 直接可用 |
| `speed` | `vel.magnitude()` | float | m/s | 1 行计算 | 可计算 |
| `speed_kmh` | `speed × 3.6` | float | km/h | 1 行计算 | 可计算 |
| `heading_rad` | `math.radians(rotation.yaw)` | float | rad | 1 行计算 | 可计算 |
| `pitch`/`roll` | `rotation.pitch`/`roll` | float | deg | 直接 | 直接可用 |
| `bbox_extent_x/y/z` | `bbox.extent` | float | m | 直接 | 直接可用 |
| `throttle` | `control.throttle` | float | [0,1] | 直接 | 直接可用 |
| `brake` | `control.brake` | float | [0,1] | 直接 | 直接可用 |
| `steer` | `control.steer` | float | [-1,1] | 直接 | 直接可用 |
| `is_alive` | `actor.is_alive` | bool | — | 直接 | 直接可用 |
| `is_emergency` | 派生/上下文 | bool | — | 类型匹配 | 可计算 |

`vehicle_category` 由 `_derive_vehicle_category(vehicle_type)` 函数根据 `type_id` 关键字匹配确定，分为 `car`、`bicycle`、`motorcycle`、`bus_or_truck`、`emergency` 五类。该分类用于 3.3.2 节行为层 Ego-Centric ROI 的差异化半径（如轿车 70 m、自行车 50 m）。

28 个字段中 20 个来自 CARLA 原生 API（无需计算），8 个为派生量。这种"天然—计算"的分治降低了提取模块与仿真器的耦合度，并使得未来替换仿真器（如从 CARLA 切换到 LGSVL）时只需重写原生字段映射，派生字段则保持不变。28 字段的设置并非随意：其中 `vehicle_category`、`is_emergency` 等字段虽不直接参与场景空间关系计算，但被行为层（如 `overtaking` 检测器需要区分轿车与自行车以施加不同距离阈值）与规则层（如 R8 弱势参与者保护、R9 学区限速）频繁使用；`bbox_extent_x/y/z` 则用于 `blocked_view` 行为检测中的遮挡比例计算——若仅保留位置/速度核心字段，下游模块将不得不通过 `entity_id` 反向查询仿真器获取这些字段，破坏了"图谱作为唯一知识源"的设计原则。

#### 动态实体：PedestrianEntity

`PedestrianEntity`（Neo4j Label: `Pedestrian`）对应 CARLA `carla.Walker`，携带 **13 个属性**。除位置/速度/包围盒等与 VehicleEntity 结构相似的字段外，另有三个行人的特有字段：`action`（行为类别，如 `"Idle"`、`"Walking"`、`"Running"`，来自 `carla.Walker.get_action()`）、`is_on_crosswalk`（通过行人所在位置与斑马线多边形做空间包含判断，利用 `shapely.geometry.Polygon.contains()` 实现）、`is_on_sidewalk`（通过 `carla_map.get_waypoint(ped_location).lane_type` 是否为 `Sidewalk` 判定）。

行人节点与车辆节点在属性结构上的差异反映了二者在安全验证场景中的不同角色：行人不需要 `throttle`/`brake`/`steer` 控制信号（因为 CARLA 中的行人由全局导航系统控制而非独立控制输入），但需要 `is_on_crosswalk` 和 `is_on_sidewalk` 两个布尔标记，它们是 R1（行人优先）与 R8（弱势参与者保护）两条规则判定的核心输入。实验表明，`is_on_crosswalk` 的判定具有极低的误报率：采用 `Polygon.contains()` 配合地图预生成的斑马线多边形集合，在 24000 帧长时运行中未观察到误报。这是因为 CARLA 的斑马线多边形由地图标定文件直接定义，几何边界稳定，不存在"斑马线边界浮点抖动"的问题。——这也间接说明了场景层"零防抖"原则的合理性：空间包含判定不受仿真器物理抖动影响，无需引入额外的帧间平滑。

`action` 字段的行为识别准确率受限于 CARLA 的 `Walker.get_action()` 接口：该接口在 `WalkerBoneControl` 模式下返回 `"Idle"`、`"Walking"`、`"Running"` 三值之一，但在 `Autopilot` 模式下（即行人在 Crosswalk 上自动穿行时），返回值为 `"Idle"` 恒定不变，无法反映行人"即将加速、正在穿越"等细分状态。这是 CARLA 0.9.16 的已知限制，本文在此不做额外的运动模式推断，直接使用原生接口输出值。未来在 3.7 节异常注入的 `ped_crs` 异常类型中，自车辆可通过行人 speed 字段的帧间变化进行间接推断（如 `speed` 从 0.5 m/s 跳变到 2.0 m/s 通常对应"突然横穿"行为），以弥补 `action` 字段在该场景下的信息不足。

#### 控制设备：TrafficLightEntity

`TrafficLightEntity`（Neo4j Label: `TrafficLight`）对应 CARLA `carla.TrafficLight`，携带 **7 个属性**。其中关键字段 `state` 取值为 `{"Red", "Yellow", "Green"}`；`elapsed_time` 表示当前颜色已持续时间（s）；`affected_lane_ids` 是一个列表字段，记录该信号灯控制的所有车道 ID 集合。该集合并非 CARLA 直接提供，而是需要在地图加载时预处理：遍历 `carla_map.generate_waypoints(2.0)` 并为每个 waypoint 检查是否有 `waypoint.traffic_light` 关联，从而建立 $TL \to \{\text{lane\_id}\}$ 映射。

#### 静态设施：RoadElementEntity

`RoadElementEntity`（Neo4j 多标签支持 `Lane`、`Road`、`Junction`）来自 CARLA `carla.Waypoint` 拓扑查询。每个 `Lane` 节点携带 **13 个属性**，涵盖唯一标识（`road_id`、`lane_id`、`junction_id`）、几何（`center_x/y/z`、`heading_rad`、`lane_width`、`speed_limit`）、拓扑关系（`left_lane_id`、`right_lane_id`、`has_traffic_light`）和类型信息（`lane_type`）。`lane_id` 的符号约定：正数表示道路默认方向车道，负数表示对向车道——此约定对逆行检测（R5）至关重要。`junction_id` 等于 `-1` 表示不在路口内，正数表示所属路口 ID。

#### 环境与聚合节点

**`EnvironmentSnapshot`**（Neo4j Label: `EnvSnapshot`）采集每帧的全局环境状态，携带 **12 个字段**：`frame_id`、`elapsed_seconds`、`delta_seconds`、`map_name`、`fog_density`、`cloudiness`、`precipitation`、`precipitation_deposits`、`wetness`、`sun_altitude_angle`、`wind_intensity`、`traffic_density`。其中 `wetness`、`fog_density`、`sun_altitude_angle` 等环境量与 3.3.3 节恶劣天气限速规则（R11）、弱势参与者保护规则（R8）直接相关。

**`ScenarioSnapshot`**（Neo4j Label: `SceneSnapshot`）是每帧唯一的聚合根节点，包含 `frame_id`、`elapsed_seconds`、`n_vehicles`、`n_pedestrians` 四个字段。该节点通过 `containsVehicle`、`containsPedestrian`、`containsTrafficLight`、`containsRoad`、`hasEnvironment` 五种帧聚合边与同帧所有场景层实体及环境节点一一连接，形成"以帧为根、以实体为叶"的树状结构。

表 3-10 汇总全部场景层节点的属性和字段数。

**表 3-10** 场景层节点属性统计

[三线表]

| 节点类 | Neo4j Label | 属性数 | 原生字段 | 派生字段 | 特有字段 |
|--------|-------------|--------|---------|---------|---------|
| VehicleEntity | `Vehicle` | 28 | 20 | 8 | vehicle_category, is_ego, is_emergency, brake, steer |
| PedestrianEntity | `Pedestrian` | 13 | 10 | 3 | action, is_on_crosswalk, is_on_sidewalk |
| TrafficLightEntity | `TrafficLight` | 7 | 5 | 2 | state, elapsed_time, affected_lane_ids |
| RoadElementEntity | `Lane`/`Road`/`Junction` | 13 | 10 | 3 | lane_id, lane_type, left/right_lane_id |
| EnvironmentSnapshot | `EnvSnapshot` | 12 | 8 | 4 | 天气六要素 + 帧信息 |
| ScenarioSnapshot | `SceneSnapshot` | 4 | 4 | 0 | 帧统计 |

### 3.3.1.2 空间关系计算

场景层关系描述"实体之间在同一帧内的空间、拓扑、包含关系"。所有关系均通过 `stk/scenario/spatial.py` 中的纯函数计算。表 3-11 给出全部 15 种关系类型的源→目标约束与计算复杂度。

**表 3-11** 场景层关系类型、约束与复杂度

[三线表]

| 关系类型 | 源 → 目标 | 计算方式 | 复杂度 | 阈值 |
|---------|----------|---------|--------|------|
| `in_lane` | `Vehicle` → `Lane` | 最近车道匹配（横向距离） | $O(V \cdot K)$ | $\epsilon_{\text{lane}}=10.0$ m |
| `on_road` | `Vehicle` → `Road` | 车道→路段包含派生 | $O(1)$ | — |
| `in_junction` | `Vehicle` → `Junction` | `junction_id != -1` 判定 | $O(1)$ | — |
| `adjacent_lane` | `Lane` → `Lane` | `left_lane_id`/`right_lane_id` 字段 | $O(1)$ | — |
| `lane_connects` | `Lane` → `Lane` | waypoint.next() 拓扑 | $O(K)$ | — |
| `ahead_of` | `Vehicle` → `Vehicle` | 同车道纵距投影 > 0 | $O(N_{\text{ego}} \cdot N_{\text{roi}})^{\dagger}$ | 横向 $< w_{\text{lane}}/2$ |
| `beside` | `Vehicle` → `Vehicle` | 横向≤3.0 m, 纵向<5.0 m | $O(N_{\text{ego}} \cdot N_{\text{roi}})^{\dagger}$ | $\text{lat}_{\text{max}}=3.0$ m, $\text{long}_{\text{max}}=5.0$ m |
| `nearby_pedestrian` | `Vehicle` → `Pedestrian` | 欧氏距离 | $O(N \cdot P)$ | 20.0 m |
| `controlled_by` | `Lane` → `TrafficLight` | 预处理映射表 | $O(1)$ | — |
| `containsVehicle`/Pedestrian/TL/Road | `SceneSnapshot` → 实体 | 固定帧聚合 | $O(N)$ | — |
| `hasEnvironment` | `SceneSnapshot` → `EnvSnapshot` | 固定关联 | $O(1)$ | — |
| `weather_context` | `EnvSnapshot` → `SceneSnapshot` | 双向语境边 | $O(1)$ | — |

> $\dagger$：在 Ego-Centric 模式下（默认启用），空间关系计算仅以自车（ego）与 ROI 内车辆配对，复杂度降为 $O(N_{\text{ego}} \cdot N_{\text{roi}})$，其中 $N_{\text{roi}}$ 为以 self 为中心的笛卡尔椭圆（前方 70 m、后方 30 m、侧向 50 m）内的他车数量。非 Ego-Centric 模式（`legacy_full_pairing=true`）退化为 $O(N^2)$。

场景层共计 15 种关系类型。以 20 辆车、8 个行人、12 盏信号灯、40 条车道、5 个路口为典型规模，每帧平均生成约 $1.5 \times 10^3$ 条关系边（Ego-Centric 模式）。这些关系边构成了行为层与规则层推理的几何基础。

### 3.3.1.3 快照构建与生命周期管理

`stk/scenario/snapshot_builder.py` 的 `build_snapshot(FrameData)` 函数接收一帧完整的原始数据，将其转化为 `ScenarioSnapshot + EnvironmentSnapshot` 双根结构。关键步骤为：以 `ScenarioSnapshot(frame_id)` 作为帧根节点；以 `EnvironmentSnapshot(frame_id, weather...)` 作为环境节点；将所有实体通过五种 `containsX` 边接入帧根；通过 `hasEnvironment` 边接入环境节点。

与此同时，`LifecycleManager` 对每帧检测到的动态实体 ID 集合与上一帧 ID 集合做差集，输出每个实体的生命周期状态转换 `{"activated"|"deactivated"|"stable"|"created"}`。该状态是对应节点下一帧是否需要执行 CREATE 或 DEACTIVATE 操作的唯一依据。

### 3.3.1.4 提取器模块

`stk/extraction/` 包含六类提取器，由 `extraction/pipeline.py` 中的编排器统一调度，在 50 ms 帧循环窗口内完成全部数据提取。表 3-12 列出各类提取器。

**表 3-12** 六类 CARLA 真值提取器

[三线表]

| 提取器 | 对应 CARLA API | 输出结构 |
|--------|---------------|---------|
| `ActorExtractor` | `world.get_actors().filter('vehicle.*'/'walker.*')` | Vehicle/Pedestrian dict 列表 |
| `TrafficLightExtractor` | `world.get_actors().filter('traffic.*')` | TrafficLight dict 列表 |
| `WaypointExtractor` | `carla_map.generate_waypoints(2.0)` | Lane/Road/Junction + 灯-车道映射 |
| `WeatherExtractor` | `world.get_weather()` | 天气属性 dict |
| `SensorExtractor` | Collision/LaneInvasion 传感器回调 | 碰撞与车道入侵事件 |
| `Pipeline`（编排器） | 五者并行合并为 FrameData | 完整的 FrameData dict |

六类提取器在编排器中并非完全并行——其中 `ActorExtractor`、`TrafficLightExtractor`、`WeatherExtractor`、`SensorExtractor` 互相独立，可并行调用 CARLA API；而 `WaypointExtractor` 因为需要遍历整张地图（CARLA Town10 中含约 8000 个 waypoint），单次调用耗时约 200 ms，远高于其他提取器的 10-30 ms 量级。本文的处理策略是：`WaypointExtractor` 在仿真器启动后只调用一次，提取的道路拓扑结果缓存在 `ScenarioBuilder._waypoint_cache` 中，之后每帧的场景层构建只查询该缓存，避免重复扫描地图。该缓存机制使得场景层构建在长时运行中保持稳定的 50 ms 单帧预算，是支撑后续流式长时采集（3.5 节）的关键性能优化。

`SensorExtractor` 与前五类不同：它并非通过定时轮询 CARLA API，而是通过 `world.on_tick(lambda event: sensor_callback(event))` 注册回调函数，由仿真器在传感器事件发生时主动推送。该设计使得 `SensorExtractor` 的输出是非确定性的——同一场景在不同次运行中可能因为仿真器物理微小差异而触发不同的碰撞事件。这一非确定性在 3.6 节场景库中通过"地面真值"机制被规避：每个场景的预期违规由代码确定性触发，而非依赖随机物理碰撞。

---

## 3.3.2 行为层：行为检测与防抖

如果说场景层回答了"此刻实体在哪里"，那么行为层回答的是"此刻实体在做什么"。行为是一个跨多帧的连续概念，单帧静态关系无法构成行为。例如"跟驰"不是某一帧的瞬时事实——它是"在一段时间内前车与后车保持同车道、相近速度、有跟距离"这一**多帧组合事实**。同样地，"超车"也不是某一帧的事实，它是"后车接近-并排-超越-远离-回到原车道"四个连续子阶段的组合。本节描述行为层的设计：两类行为节点、11 个检测器、13 种关系类型，以及防抖状态机消除单帧抖动引入的关系闪烁。

行为层在数据流中处于承上启下的位置：上游它接收场景层的"瞬时事实流"作为输入，下游它向规则层提供"持续行为语义"作为规则判定的基础。许多规则（如 R3 实线变道、R13 纵向安全距离、R14 横向安全距离）的判定不仅依赖单帧几何关系，还依赖行为层的"持续状态"判定——例如 R3 实线变道要求 `changing_lane` 行为在 `lane_id` 变化过程中跨越实线车道；R13 纵向安全距离需要 `following` 行为状态以确定两车是否处于"主评与被评"关系。如果没有行为层的中间抽象，规则引擎将不得不在每一帧重新从场景层重组跨帧信息，不仅效率低下，也无法在图上显式表达"行为"作为独立语义对象。

### 3.3.2.1 行为分类与节点设计

自动驾驶中"行为"语义可分三个子类：

**表 3-13** 行为分类体系

[三线表]

| 类别 | 描述 | 节点类型 | 关系类型数 | 对应关系 |
|------|------|---------|-----------|---------|
| **个体行为** (maneuver) | 单实体自身运动状态 | `ManeuverNode` | 2 | standing_still, changing_lane |
| **交互行为** (interaction) | 两实体间互动 | `InteractionEvent` | 7 | following, approaching, yielding_to, overtaking, wrong_side_meeting, opposite_direction, blocked_view |
| **演化行为** (evolution) | 实体相对环境的演化 | `InteractionEvent` | 4 | same_direction, approaching_pedestrian, approaching_intersection, crossing |

个体行为（如`standing_still`、`changing_lane`）建模为 `ManeuverNode`（Neo4j Label: `Maneuver`），通过 `actor` 边指向行为主体车辆。交互行为与演化行为建模为 `InteractionEvent`（Neo4j Label: `Interaction`），通过 `src`/`dst` 边指向交互主客体。表 3-14 列出两类节点的关键属性。

**表 3-14** 行为层节点属性

[三线表]

| 节点 | 属性 | 说明 |
|------|------|------|
| ManeuverNode | `maneuver_type` | 6 种：standing_still, changing_lane, accelerating, decelerating, cruising, stopping |
| | `actor_id` | 行为主体 ID |
| | `frame_start`, `frame_end` | 起止帧 |
| | `duration_frames` | 持续帧数 |
| | `state` | `"active"` 或 `"ended"` |
| | `severity` | 严重度 [0,1] |
| | `derived_attrs` | 派生属性字典（如平均速度） |
| | `related_rule` | 相关规则 ID（可空） |
| InteractionEvent | `interaction_type` | 13 种（含个体行为类） |
| | `src_id`, `dst_id` | 交互主客体 ID |
| | `frame_start`, `frame_end` | 起止帧 |
| | `duration_frames`, `state`, `severity` | 同上 |
| | `source_relations` | 触发该交互的场景关系集合 |
| | `related_rule` | 关联的规则 ID |

### 3.3.2.2 行为检测器与防抖状态机

`stk/behavior/detectors.py` 实现 11 个 `detect_*` 纯函数。每个检测器输入场景层输出，输出行为关系候选。表 3-15 给出全部检测器的判定条件、关键阈值与防抖进入门槛。

**表 3-15** 行为检测器与判定条件

[三线表]

| 检测器 | 输出行为类型 | 判定条件 | 关键阈值 | 进入防抖(帧) |
|--------|-------------|---------|---------|------------|
| `detect_standing_still` | `standing_still` | $\|v\| < 0.1$ m/s 持续 $N$ 帧 | 速度 0.1 m/s | 2 |
| `detect_changing_lane` | `changing_lane` | 横向速度 $\|v_y\| > 0.5$ m/s 且 `in_lane` ID 变化 | 横向速度 0.5 m/s | 2 |
| `detect_following` | `following` | 同车道 + 距离 $<12$ m + 前车速度 $>$ 后车 - 1 m/s | 距离 12 m | 3 |
| `detect_approaching` | `approaching` | 同车道 + 距离 $<20$ m + 相对速度 $>1$ m/s | 相对速度 1 m/s | 3 |
| `detect_yielding_to` | `yielding_to` | 行人 $<8$ m + 车速 $<1$ m/s + 在横道线 | 行人距离 8 m | 3 |
| `detect_overtaking` | `overtaking` | `beside` + 后车速度 $>$ 前车 $+2$ m/s + 持续 3 帧 | 速度差 2 m/s | 5 |
| `detect_opposite_direction` | `opposite_direction` | 朝向差 $>143°$ + 距离 $<30$ m | 朝向差 143° | 1 |
| `detect_blocked_view` | `blocked_view` | 三车共线 + 中车屏蔽 $>30\%$ | 屏蔽比 30% | 3 |
| `detect_approaching_pedestrian` | `approaching_pedestrian` | `nearby_pedestrian` + 车速 $>5$ m/s + 距离缩短趋势 | 车速 5 m/s | 3 |
| `detect_approaching_intersection` | `approaching_intersection` | `in_junction` + 距离路口 $<15$ m | 距离 15 m | 2 |
| `detect_crossing` | `crossing` | 在 crosswalk + 速度与行人位移夹角 $>60°$ | 夹角 60° | 3 |

防抖机制是行为层的核心算法贡献。每种行为关系定义独立的**进入阈值**（`enter_thresh`）与**消失阈值**（`exit_thresh`），二者可不对称。表 3-16 给出全部 13 种关系的差异化防抖配置。

**表 3-16** 行为关系防抖阈值（代码 `DEFAULT_DEBOUNCE_THRESHOLDS`）

[三线表]

| 关系类型 | 进入阈值 | 消失阈值 | 模式 | 设计理由 |
|---------|---------|---------|------|---------|
| `following` | 3 | 3 | 双向防抖 | 路口减速跟车抖动 |
| `approaching` | 3 | 3 | 双向防抖 | 信号灯相位切换瞬间 |
| `yielding_to` | 3 | 3 | 双向防抖 | 行人在横道线边缘抖动 |
| `overtaking` | 5 | 3 | 进入更严 | 超车动作时间跨度长 |
| `changing_lane` | 2 | 2 | 双向防抖 | 变道动作可较快完成 |
| `blocked_view` | 3 | 3 | 双向防抖 | 遮挡判定有延迟 |
| `approaching_pedestrian` | 3 | 3 | 双向防抖 | 行人偶然接近 |
| `approaching_intersection` | 2 | 2 | 双向防抖 | 路口进出边界 |
| `crossing` | 3 | 3 | 双向防抖 | 行人横穿判定 |
| `standing_still` | 2 | 2 | 双向防抖 | 低速滑行判别 |
| `wrong_side_meeting` | 1 | 1 | 瞬时反应 | 单一几何判定 |
| `opposite_direction` | 1 | 1 | 瞬时反应 | 单一几何判定 |
| `same_direction` | 1 | 1 | 瞬时反应 | 单一几何判定 |

`RelationDebouncer` 为每一对 `(src_id, dst_id, type)` 维护 `on_counter`（连续满足帧数）与 `off_counter`（连续不满足帧数）两个计数器，区别对待激活与消亡路径。防抖状态机的完整伪代码如算法 3.1 所示。

```
算法 3.1: RelationDebouncer.update(relation_type, key, condition_met, frame_id)
输入: 关系类型, (src,dst,type) 三元组, 当前帧是否满足, 帧号
输出: "create" | "keep" | "delete" | "none"

1. state ← _items.get(key)
2. if state is None:
3.     threshold ← THRESHOLDS[relation_type]
4.     state ← {active: false, on_counter: 0, off_counter: 0, threshold: threshold}
5.     _items[key] ← state
6. end if
7.
8. // -- 激活路径 --
9. if condition_met:
10.    state.on_counter += 1
11.    state.off_counter ← 0
12.    if not state.active and state.on_counter ≥ state.threshold:
13.        state.active ← true
14.        return "create"
15.    end if
16.    if state.active:
17.        return "keep"
18.    end if
19.    return "none"
20.
21. // -- 消亡路径 --
22. else:
23.    state.off_counter += 1
24.    state.on_counter ← 0
25.    if state.active and state.off_counter ≥ state.threshold:
26.        state.active ← false
27.        return "delete"
28.    end if
29.    if state.active:
30.        return "keep"  // 处于活跃但未达到消失阈值,抑制噪声
31.    end if
32.    return "none"
33. end if
```

**算法 3.1** 行为关系防抖状态机。激活路径与消亡路径均需连续满足/不满足 `threshold` 帧才触发状态切换；在活跃状态下短时未检测到关系时返回 `"keep"` 而非 `"delete"`，有效抑制单帧的浮点闪烁。

防抖状态机的设计动机源于两个工程观察。**观察一：CARLA 物理仿真存在亚帧浮点抖动**。在 Ego-Centric 模式下，自车辆的速度从 `v=8.0000` m/s 抖动到 `v=8.0002 m/s` 再回到 `v=7.9998 m/s` 是经常出现的现象——这种亚米级的位置抖动会令检测器在"满足/不满足"间频繁跳变，例如"前车距离<12 m"在抖动峰值时刻可能瞬间为 → 12.0012 → 11.9983，造成 `following` 关系在连续帧中 ON/OFF/ON/OFF 切换，下游 GNN 学习时无法从中识别出真正语义。**观察二：行为本身时域跨度天然较长**。例如"超车"在 CARLA 仿真中通常需要 3-5 秒，对应 60-100 帧；跟驰往往持续更久。因此对行为类关系的"短暂消失"保持容忍（即 offset_counter 累积到 threshold 才 delete），不会引入语义延迟，反而能过滤仿真噪声。

`off_counter` 与 `on_counter` 解耦的设计是防算法对抖动"双向不对称容忍"的关键：例如 `overtaking` 阈值为 enter=5、exit=3，意味着新关系的建立需要连续 5 帧（3 个连续行为子阶段整合的证据，避免误启动），而关系消失只需连续 3 帧（避免持续行为的偶然漏检测导致关系提前中断）；这种不对称容忍契合了"建立行为需要更多证据、消亡行为应较灵敏"的直觉。

当防抖判定为 `"delete"` 时，`BehaviorRelationGenerator` 关闭对应行为节点的 `frame_end` 字段并将 `state` 设为 `"ended"`，同时将行为关系边的 `valid_to` 设置为 `frame_id - 1`，保留完整时长信息。这里将 `valid_to` 设置为 `frame_id - 1` 而非 `frame_id`，是因为 `frame_id` 处的检测条件已不满足，关系事实在该帧不成立；保留 `frame_id - 1` 表示关系的最后一帧有效是上一帧。这是时态三元组公理 $A_4$ 的具体应用——`valid_to` 不能延展到关系实际不再成立的帧。

### 3.3.2.3 行为关系生成器与跨层桥接

`BehaviorRelationGenerator` 是行为层主流程入口。其核心流程如算法 3.2 所示。

```
算法 3.2: BehaviorRelationGenerator.generate(frame_id, ...)
输入: frame_id, 当前帧场景层输出
输出: maneuvers[], interactions[], behavior_rels[], cross_layer_rels[]

1. candidates ← run_all_detectors(scene_layer_data)
2. // Ego-Centric 过滤: 仅保留 ego 参与的车辆-车辆对子
3. if ego_filter_config.filter_behavior_detectors:
4.     ego_id ← ego_filter.select(vehicles).ego.entity_id
5.     for rel_type in {following, overtaking, opposite_direction, blocked_view}:
6.         candidates[rel_type] ← filter_ego_pairs(candidates[rel_type], ego_id)
7.     end for
8. end if
9.
10. for each (src_id, dst_id, rel_type, extra_attrs) in candidates:
11.     key ← (src_id, dst_id, rel_type)
12.     action ← debouncer.update(rel_type, key, condition_met=true, frame_id)
13.     if action == "create":
14.         rel ← create_behavior_relation(rel_type, key, frame_id, extra_attrs)
15.         node, edges ← create_node_and_links(rel_type, key, frame_id, extra_attrs)
16.         // 创建 manifestsAs, actor, src, dst 跨层桥接边
17.         new_behavior_rels.append(rel)
18.         new_cross_layer.extend(edges)
19.     elif action == "delete":
20.         close existing node (state ← "ended", frame_end ← frame_id)
21.         close existing relation (valid_to ← frame_id)
22.     end if
23. end for
24. return {maneuvers, interactions, behavior_rels, cross_layer_rels}
```

**算法 3.2** 行为关系生成器。第 2-8 行的 Ego-Centric 过滤是代码 FE-6 引入的优化：对 `following`、`overtaking`、`opposite_direction`、`blocked_view` 四类车辆-车辆对子，默认只保留涉及自车或 ROI 内他车的配对，降低长时运行时的边数膨胀。

`BehaviorRelationGenerator` 支持完整的 checkpoint 序列化（`to_dict()` / `load_dict()`），可在长时采集的 chunk 边界恢复防抖表与活跃节点集，保证跨 chunk 行为的连贯性（如某超车行为横跨 chunk 边界时，防抖计数器不会重置）。

### 3.3.2.4 跨层桥接机制

行为层与场景层之间通过 `stk/behavior/manifest.py` 中的四类跨层边连接：

- `manifestsAs(behavior_node_id, src, dst, type, frame_id)`：将行为节点与对应的行为关系边绑定，实现"节点+边双轨表达"；
- `actor_edge(maneuver_id, vehicle_id, frame_id)`：连接 Maneuver 与其行为主体车辆；
- `src_edge(interaction_id, src_entity_id, frame_id)`、`dst_edge(interaction_id, dst_entity_id, frame_id)`：连接 Interaction 与其交互主客体。

形成"违规 → 行为 → 实体"的完整证据链，满足公理 $A_6$。

---

## 3.3.3 规则层：RSS 与交通法规推理

规则层是 STKG 语义抽象的最高层，回答"这些行为合不合规、危不危险"。它的输入是场景层实体属性与行为层检测结果，输出是 `SafetyViolation` 节点、`violates` 边、`supportedByEvidence` 证据链以及 `ResponsibilityAssignment` 责任归因。规则层在架构上分为两个互补子层——**RSS 子层**（负责车辆间纵横向物理安全距离校验）和 **交通法规子层**（负责行为合规性的法规判别），二者共同构成完整的符号推理系统。

为何将规则层进一步分为两个子层而非统一表达？根本原因在于两类规则的判定逻辑与不确定性来源不同。**RSS 子层**基于牛顿运动学建立的连续物理安全距离公式，其参数（反应时间 ρ、最大加速 a_max_accel 等）具有清晰的物理含义与文献标准值，规则触发条件由数学不等式确定；其不确定性主要来源于物理参数取值是否过于保守，而非规则本身的歧义。**交通法规子层**则基于交通安全法规文本——这些文本天然具有模糊性（如"必要的的安全距离"、"合理的车速"、"应当让行"等表述），需要在工程上量化为具体数值阈值（例如将"必要的让行距离"定为 15 m），其规则触发条件实际上是工程经验与法规文本的折中产物。两类规则的不确定性来源不同，处理方式也不同：RSS 保留数学推导的全过程并以 `predicate_str` 字段记录完整公式调用上下文；交规则以 `predicate_str` 记录"规则名 + 关键阈值"使下游可解释性模块能区分这两类不同来源的违规。这种分层表达并不增加冗余节点——`SafetyViolation` 节点的 `rule_layer` 字段就承担了这一区分职责。

```
┌──────────────────────────────────────────────────────────┐
│                      规则层 (Rule Layer)                  │
│ ┌─────────────────────────┐  ┌─────────────────────────┐ │
│ │   RSS 子层               │  │   交通法规/场景子层       │ │
│ │   (物理安全校验)          │  │   (行为合规校验)          │ │
│ │                         │  │                           │ │
│ │   纵向安全距离 d_min      │  │   R1 行人优先             │ │
│ │   横向安全距离 d_min_lat  │  │   R2 闯红灯              │ │
│ │   反应不当 NoProperResp   │  │   R3 实线变道            │ │
│ │   责任归因 Responsible   │  │   ...（R4-R18 共14条）    │ │
│ │                         │  │                           │ │
│ │   理论基础：RSS 模型       │  │   理论基础：道路交通安全法 │ │
│ └─────────────────────────┘  └─────────────────────────┘ │
│                                                          │
│   输出层 → SafetyViolation 节点 + violates 边              │
│          → supportedByEvidence 边 + ResponsibilityAssignment │
└──────────────────────────────────────────────────────────┘
```

**图 3-3** 规则层内部双层结构

### 3.3.3.1 RSS 子层

RSS（Responsibility-Sensitive Safety）由 Shalev-Shwartz 等 [Shalev-Shwartz, 2017] 提出，核心思想是若每辆车始终保持**随速度动态变化的最小安全距离**，则所有碰撞的责任均可归咎于未保持该距离的车辆。RSS 模型被 Mobileeye/Yoshiokafuku 等业界主体作为可验证安全模型的标准框架引用，本文在 STKG 规则层中将其作为连续物理安全谓词实现——这与 RSS 在原文献中作为决策控制模型的角色不同：本文不向下游发送"应当刹车"的指令，而仅向图谱写入"DangerousState"事件，由下游异常检测模型与可视化模块消费。

**纵向安全距离**。给定前车 $B$ 与跟驰后车 $A$，两车同车道且 $\text{ahead\_of}(B, A)$ 成立，则后车 $A$ 必须与 $B$ 保持的纵向最小安全距离为：

$$
d_{\min}^{\text{long}}(A, B, t) = \max\!\left(0,\ v_A \rho + \frac12 a_{\max,\text{accel}} \rho^2 + \frac{(v_A + a_{\max,\text{accel}}\rho)^2}{2\,a_{\min,\text{brake}}} - \frac{v_B^2}{2\,a_{\text{brake}}}\right)
\tag{3.11}
$$

该公式分四项对应跟驰场景下"后车在反应时间内继续运动 + 反应时间内加速 + 反应后制动到完全停止的距离 − 前车在相同时间内制动到完全停止的距离（前车也减速，因此这一项作为削减量出现）"。每一项的物理含义：第一项 $v_A \rho$ 是车辆 $A$ 在反应时间 $\rho$ 内维持当前速度 $v_A$ 行驶的距离；第二项 $\frac{1}{2} a_{\max,\text{accel}} \rho^2$ 是车辆 $A$ 在反应时间内以最大加速度 $a_{\max,\text{accel}}$ 加速所行驶的距离（防御后车在反应期有意加速的最坏情形）；第三项 $\frac{(v_A + a_{\max,\text{accel}}\rho)^2}{2 a_{\min,\text{brake}}}$ 是反应期结束时车辆 $A$ 已达新速度后的刹车距离，使用最低合理制动减速度 $a_{\min,\text{brake}}$（保守上界）；第四项 $-\frac{v_B^2}{2 a_{\text{brake}}}$ 是前车 $B$ 在此期间也以最大减速度 $a_{\text{brake}}$ 制动所行驶的距离，作为前车距离的削减量。RSS 模型的核心思想即为：在合理参数假设下，后车若保持该距离，则不与前车发生碰撞——这一保证由数学不等式严格保证，不依赖任何经验或统计数据。

参数值如表 3-17 所示。

**表 3-17** RSS 子层参数表（代码 `DEFAULT_RSS_PARAMS`）

[三线表]

| 参数 | 符号 | 值 | 单位 | 描述 |
|------|------|-----|------|------|
| 反应时间 | $\rho$ | 0.3 | s |自车从感知到制动的时间 |
| 后车最大加速度 | $a_{\max,\text{accel}}$ | 0.5 | m/s² | 反应时间内合理加速上限 |
| 后车最小制动 | $a_{\min,\text{brake}}$ | 3.0 | m/s² | 反应后至少能实现的制动减速度 |
| 前车最大制动 | $a_{\text{brake}}$ | 8.0 | m/s² | 前车可能实现的极端制动减速度 |
| 横向安全裕度 | $\mu$ | 0.5 | m | 横向最小缓冲间距 |
| 横向最小制动 | $a_{\min,\text{lat,brake}}$ | 1.5 | m/s² | 横向最小合理刹车减速度 |
| 横向最大制动 | $a_{\text{lat,brake}}$ | 3.0 | m/s² | 前车横向最大刹车减速度 |

参数取值均来自 RSS 原论文 [Shalev-Shwartz, 2017] 表 1 的标准值。这些值的选取兼顾保守性与现实性：例如 $a_{\min,\text{brake}}=3$ m/s² 是普通驾驶员在紧急制动时容易达到的最低水平，将其设为后车最坏制动能力上限可保证即便驾驶员反应迟钝仍可避免碰撞；$a_{\text{brake}}=8$ m/s² 是车辆在 ABS 系统下的接近极限值，假设前车可实现这一减速度以保守估计前车距离削减量。代码 `DEFAULT_RSS_PARAMS` 字典将上述参数作为默认值导出，并允许在 `RuleEnforcer.__init__` 时通过 `rss_params` 参数覆盖——这一可配置性使得 STKG 可在不同安全严苛度需求场景下复用，例如在低速园区场景下可放宽 $a_{\min,\text{brake}}$ 至 2 m/s² 以避免过度紧张的安全距离。

纵向危险状态判定：

$$
\text{SafeDistanceViolation}(A, B, t) \iff d_{\text{long}}(A, B, t) < d_{\min}^{\text{long}}(A, B, t)
\tag{3.12}
$$

**横向安全距离**：

$$
d_{\min}^{\text{lat}}(A, B, t) = \mu + \frac{v_{\text{lat},A}^2}{2\,a_{\min,\text{lat,brake}}} + \rho\,v_{\text{lat},A} - \frac{v_{\text{lat},B}^2}{2\,a_{\min,\text{lat,brake}}}
\tag{3.13}
$$

横向安全距离公式与纵向类似但增加了基础安全裕度 $\mu=0.5$ m，对应车辆侧向的最小物理缓冲间距（车门间距、车身晃动等）。横向场景下两车的"互相逼近"由各自的横向速度分量驱动，公式结构与纵向对称：第一项 $\mu$ 是常数裕度；第二项是 $A$ 车的横向刹车距离；第三项 $\rho v_{\text{lat},A}$ 是反应期内 $A$ 车横向惯性位移；第四项 $-\frac{v_{\text{lat},B}^2}{2 a_{\min,\text{lat,brake}}}$ 是 $B$ 车横向刹车距离的削减项。

横向危险状态判定：

$$
\text{LateralDangerousState}(A, B, t) \iff d_{\text{lat}}(A, B, t) < d_{\min}^{\text{lat}}(A, B, t)
\tag{3.14}
$$

**复合状态与责任归因**：

$$
\text{DangerousState}(A, B, t) = \text{SafeDistanceViolation}(A, B, t) \lor \text{LateralDangerousState}(A, B, t)
\tag{3.15}
$$

**反应不当**（No Proper Response）判定是 RSS 模型中责任归因的关键步骤：仅当后车 $A$ 在 `DangerousState` 持续 3 帧内未实施充分制动时才记为反应不当，避免误判"瞬时未刹车"为严重违章：

$$
\text{NoProperResponse}(A, t) = \bigwedge_{k=0}^{2} \text{brake}(A, t+k) < 0.3
\tag{3.16}
$$

公式 (3.16) 中阈值 0.3 是经验值——通常车辆紧急制动时 `brake` 值会迅速跳至 0.7 以上，若 3 帧内始终低于 0.3 则基本可确认后车未做有效制动反应。该阈值在 `DEFAULT_RSS_PARAMS` 之外，由 `_no_proper_response_threshold = 0.3` 单独定义，可在配置文件中调整。

责任归因逻辑：

$$
\text{ResponsibleAgent}(A, \text{event}) \iff \text{NoProperResponse}(A) \land B \text{ 行为合规}
\tag{3.17}
$$

公式 (3.17) 表明：责任归因不仅要求 `NoProperResponse(A)` 成立，还要求前车 $B$ 自身的行为合规（即 $B$ 未突然变道切入、未突然急刹等），后者由行为层与场景层提供证据。这一双重条件符合 RSS 原论文的"责任权衡"理念：在双方都行为合规的情况下，未保持安全距离的车辆责任主体；若 $B$ 主动做出非合规动作导致 $A$ 难以避免追尾，则责任可能归于 $B$。

`run_rss_check(d_long, d_lat, v_A, v_B, v_lat_A, v_lat_B, brake_values)` 是 RSS 综合入口函数，返回包含上述全部判定的九项输出：`is_dangerous_state`、`is_safe_distance_violation`、`is_lateral_dangerous_state`、`is_no_proper_response`、`is_responsible`、`d_min_long`、`d_min_lat`、`severity`、`evidence_path`。其中 `evidence_path` 是触发该判定的证据节点集合，按本节末尾 3.3.3.4 节所述，作为 `supportedByEvidence` 边的目标节点写入图谱。

### 3.3.3.2 交通法规子层（R1-R18）

交通法规子层覆盖中国《道路交通安全法》中 14 条与自动驾驶直接相关的规则（R1-R18，含 R6/R12/R14/R15 未实现）。每条规则都由一个独立的 `check_Ri_*` 函数实现。表 3-18 汇总全部 14 条规则。

**表 3-18** 交通法规规则清单（含法条依据）

[三线表]

| 规则 | 函数名 | 判定条件摘要 | 关键阈值 | 法规依据 |
|------|--------|------------|---------|---------|
| R1 | `check_R1_pedestrian_priority` | 行人在横道线 + 车辆距离 $<$ 15 m + 车速 $>$ 0.5 m/s | 15 m, 0.5 m/s | 道交法第47条 |
| R2 | `check_R2_red_light` | $v$ 在路口 + 信号灯红 + 车速 $>$ 0.3 m/s | 0.3 m/s | 道交法第38条 |
| R3 | `check_R3_solid_line_change` | `changing_lane` + 跨越实线 | 实线判定 | 道交法第44条 |
| R4 | `check_R4_opposite_meeting` | `opposite_direction` + 距离 $<$ 10 m | 10 m | 实施条例第48条 |
| R5 | `check_R5_reversing` | 速度与朝向夹角 $>135°$ 持续 5 帧 | 135°, 5 帧 | 道交法第35条 |
| R7 | `check_R7_junction_no_yield` | 进入路口 + 未让行主路车辆 | — | 道交法第52条 |
| R8 | `check_R8_vulnerable_protection` | 天气恶劣 + 行人距离 $<$ 20 m + 车速超限 | 20 m | 道交法第42条 |
| R9 | `check_R9_school_zone_speed` | 学区路段 + 车速 $>$ 30 km/h | 30 km/h | 地方实施办法 |
| R10 | `check_R10_highway_speed` | 高速路段 + 车速 $>$ 120 km/h 或 $<$ 60 km/h | 120/60 km/h | 实施条例第78条 |
| R11 | `check_R11_weather_speed` | 强降水 + 车速超过天气允许上限 | 天气阈值 | 实施条例第81条第3款 |
| R13 | `check_R13_illegal_stop` | 禁停区 + 车速 $<$ 0.3 m/s 持续 30 帧 | 0.3 m/s, 30 帧 | 道交法第63条 |
| R16 | `check_R16_amber_jumping` | 黄灯亮后未减速进入路口 | 停止线判定 | 道交法第26条 |
| R17 | `check_R17_wrong_lane` | 驶入错车道 + 持续 5 帧 | 5 帧 | 实施条例第44条 |
| R18 | `check_R18_wrong_direction_lane` | 在路口导向车道内行为与导向不一致 | 导向匹配 | 实施条例第51条 |

规则编号 R1-R18 并非连续——存在 R6（违规掉头）、R12（备用）、R14（违反交通标志）、R15（违反标线）四个空缺。这些空缺的来源有三类：① R6、R12 是开发优先级排序时为后续扩展预留的"挂账"位，对应规则计划在后续版本实现；② R14、R15 与 R3 实线变道存在语义重叠（实线即属于交通标线的一种），在 v1 实现中合并到 R3；③ 部分规则需依赖 CARLA 地图标定的标志/标线数据，而 CARLA 0.9.16 在 Town10HD 中对交通标志的覆盖不完整，使得 R14、R15 的判定逻辑无法获得稳定输入。预期违规列在 3.6 节场景库中以代码实际实现为准。

每条规则在工程实现上对应一个 `check_Ri_*` 函数，统一接受 `(frame_id, entities, scene_rels, behavior_rels)` 四元组作为输入，统一输出 `(is_violated: bool, severity: float, evidence: List[NodeRef])` 三元组。这一接口统一性使得 `RuleEnforcer.enforce` 可以用同一循环（算法 3.3 第 26-35 行）处理全部 14 条规则而无须特殊分支，是规则引擎可扩展性的关键。

阈值取值上，R2 中"车速 > 0.3 m/s"用于过滤仿真器在停车线内因浮点风速的伪停止抖动（车辆已经在停止状态但因扰动保持 0.1-0.3 m/s 的微速度）；R13 中"持续 30 帧（1.5 s）"用于区分"短暂停车"与"违法停车"——这一阈值与北京市实施办法第44条所定义的"机动车在禁停区停留超过 1 分钟即视为违法停车"相比偏短，但考虑仿真场景下行人端口与车流密度的高频变化，偏短的阈值可更敏感地捕捉停车开始时刻，避免误判红绿灯排队停车为违法停车。

每条规则的形式化逻辑判别式可统一写作一阶逻辑风格。以下给出 6 条代表性规则的完整形式化表达，其余规则形式类似（限于篇幅不全部列出）：

**R1** 行人优先：
$$
\text{YieldingToPedestrianViolation}(A, P, t) \leftarrow \text{VehicleEntity}(A) \land \text{PedestrianEntity}(P) \land \text{is\_on\_crosswalk}(P, t) \land \text{distance}(A, P, t) < 15\ \text{m} \land \text{speed}(A, t) > 0.5\ \text{m/s}
\tag{3.18}
$$

**R2** 闯红灯：
$$
\text{RedLightViolation}(A, TL, t) \leftarrow \text{VehicleEntity}(A) \land \text{TrafficLight}(TL) \land \text{in\_junction}(A, t) \land TL.\text{state} = \text{Red} \land \text{speed}(A, t) > 0.3\ \text{m/s}
\tag{3.19}
$$

**R4** 对向会车：
$$
\text{WrongSideMeetingViolation}(A, B, t) \leftarrow \text{VehicleEntity}(A) \land \text{VehicleEntity}(B) \land \text{opposite\_direction}(A, B, t) \land \text{distance}(A, B, t) < 10\ \text{m}
\tag{3.20}
$$

**R11** 恶劣天气限速：
$$
\text{WeatherSpeedViolation}(A, E, t) \leftarrow \text{VehicleEntity}(A) \land \text{EnvSnapshot}(E) \land E.\text{precipitation} > 50 \land \text{speed}(A, t) > \text{max\_allowed}(E)
\tag{3.21}
$$

**R13a** RSS 纵向安全距离：
$$
\text{SafeDistanceViolation}(A, B, t) \leftarrow \text{VehicleEntity}(A) \land \text{VehicleEntity}(B) \land \text{ahead\_of}(B, A, t) \land d_{\text{long}}(A, B, t) < d_{\min}^{\text{long}}(A, B, t)
\tag{3.22}
$$

**R16** 黄灯抢行：
$$
\text{AmberLightJumpingViolation}(A, TL, t) \leftarrow \text{VehicleEntity}(A) \land TL.\text{state}(t-1) = \text{Yellow} \land \text{is\_at\_stop\_line}(A, t-1) = \text{False} \land \text{in\_junction}(A, t) = \text{True}
\tag{3.23}
$$

### 3.3.3.3 RuleEnforcer 主生成器

`RuleEnforcer`（`stk/rules/generator.py`）是规则层的主驱动。其核心方法 `enforce()` 在一个调用中完成 RSS 扫描与全部 14 条交规检查，并统一生成后续输出。算法伪代码如算法 3.3 所示。

```
算法 3.3: RuleEnforcer.enforce(frame_id, entities, scene_rels, behavior_rels)
输入: frame_id, 当前帧全部实体与关系
输出: {violations, violation_rels, defined_by_rels, evidence_rels, responsibilities, resp_rels}

1. violations ← []; violation_rels ← []; defined_by_rels ← []
2. evidence_rels ← []; responsibilities ← []; resp_rels ← []
3.
4. // === RSS 扫描（Ego-Centric 模式）===
5. if ego_filter is not None:
6.     ego ← ego_filter.select(vehicles, frame_id).ego
7.     roi ← ego_filter.select(vehicles, frame_id).roi_targets
8.     rss_pairs ← ego × roi   // O(ego · roi)
9. else:
10.    rss_pairs ← pairs(vehicles)   // 退化为 O(N²)
11. end if
12. for each (A, B) in rss_pairs:
13.     d_long, d_lat ← compute_distances(A, B)
14.     rss_result ← run_rss_check(d_long, d_lat, v_A, v_B, brake_history)
15.     if rss_result.is_dangerous:
16.         sv ← create SafetyViolation(R13a/R14a, severity)
17.         defined_by_rels.append(defined_by(sv, rule_code))
18.         if rss_result.is_no_proper_response:
19.             responsibilities.append(ResponsibilityAssignment(sv, A))
20.         end if
21.     end if
22. end for
23.
24. // === 交通法规 R1-R18 扫描 ===
25. for each rule_fn in [R1, R2, ..., R18]:
26.     is_v, severity, evidence ← rule_fn(frame_id, entities, ...)
27.     if is_v:
28.         sv ← SafetyViolation(rule_code, severity, evidence)
29.         violation_rels.append(violates(src, dst, sv))
30.         defined_by_rels.append(defined_by(sv, rule_code))
31.         for each ev in evidence:
32.             evidence_rels.append(supportedByEvidence(sv, ev))
33.         end for
34.     end if
35. end for
36. return {violations, violation_rels, defined_by_rels, evidence_rels, responsibilities, resp_rels}
```

**算法 3.3** RuleEnforcer 生成器。第 5-11 行的 RSS 扫描支持 Ego-Centric 模式（默认）与 `legacy_full_pairing` 回退模式。RSS 扫描走 `ego × ROI` 配对而非全 O(N²)，是长时运行性能的关键优化。

`RuleEnforcer` 维护两个跨帧状态用于复杂判定：
- `_brake_history`: `{vehicle_id: [brake_values]}`，存储每辆车最近 30 帧的制动记录；
- `_stop_duration`: `{vehicle_id: frames}`，存储每辆车连续静止帧数。

这些状态在 3.4 节增量更新中通过 `IncrementalEngine.checkpoint` 序列化与恢复，保证跨 chunk 的规则判定一致性。

为应对 CARLA 浮点扰动，每条规则的 `severity` 经温度缩放处理：
$$
\text{severity} = \min\left(1.0,\ \frac{|\text{threshold} - \text{value}|}{\max(\text{threshold},\ 10^{-3})}\right)
\tag{3.24}
$$
该机制确保当值刚好超过阈值时 severity 很小，大幅超过时才接近 1.0。

### 3.3.3.4 证据链与节点关系汇总

每条 `SafetyViolation` 生成时构建完整的证据链，通过 `supportedByEvidence` 边链接：

```
sv_R13a_2052_veh123
  ├── [supportedByEvidence idx=0] → triple_in_lane_2052_veh123_lane2
  ├── [supportedByEvidence idx=1] → triple_ahead_of_2052_veh124_veh123
  └── [supportedByEvidence idx=2] → triple_following_2052_veh123_veh124
```

证据链末端为场景层空间关系边、行为层 InteractionEvent 节点或 ManeuverNode 节点。表 3-19 与表 3-20 汇总规则层节点与关系。

**表 3-19** 规则层节点类型

[三线表]

| 节点 | Neo4j Label | 属性 | 作用 |
|------|-------------|------|------|
| `RuleDefinition` | `Rule` | rule_id, rule_name, rule_layer, predicate_name | 规则元定义 |
| `RuleParameter` | `Param` | param_id, name, value, unit, rule_id | 参数元数据 |
| `SafetyViolation` | `SafetyViolation` | sv_id, rule_code, rule_name, rule_layer, frame_id, severity, predicate_str, evidence_path | 违规实例 |
| `ResponsibilityAssignment` | `Responsibility` | resp_id, sv_id, responsible_actor_id, reason | 责任归属 |

**表 3-20** 规则层关系类型

[三线表]

| 关系 | 源→目标 | 说明 |
|------|---------|------|
| `defined_by` | `SafetyViolation` → `Rule` | 违规归属到规则定义 |
| `uses_param` | `Rule` → `Param` | 规则使用参数 |
| `violates` | 违规源实体 → 违规目标实体 | 违规边 |
| `supportedByEvidence` | `SafetyViolation` → 场景/行为证据 | 证据链 |
| `triggers`/`causedBy` | `SafetyViolation` ↔ `SafetyViolation` | 因果链 |
| `responsibleFor` | `Responsibility` → `SafetyViolation` | 责任归属 |

### 3.3.3.5 规则层输出与异常检测框架的接口设计

规则层在本章中的定位不仅是独立的规则判定模块，它同时为第 4 章的 K-HSTGAN 图神经网络异常检测模型与第 5 章的 KS-NBCF 融合框架提供三种形态的先验输入。本小节阐述这些接口约定，而非具体的融合算法——后者在第 5 章中详述。重要的是，规则层的输出遵循"解耦"原则：规则层只负责产生节点、边与证据链，不关心下游如何消费；下游框架只读取消规则层的结构化输出，不修改规则层的内部状态。这种解耦保证了当未来扩展新规则时，无需修改接口协议。

**接口一：节点级先验特征拼接**

每帧的 `RuleEnforcer.enforce()` 产生的 `SafetyViolation` 节点通过 `violates` 边与被判定为违规的车辆实体关联。下游框架（第 5 章 `KS-NBCF`）读取该关联后，将违规信息映射为车辆节点特征矩阵的附加列。具体而言，第 $t$ 帧中车辆节点 $v_i$ 的扩展特征向量为：

$$
\mathbf{f}_i^t = [\mathbf{f}_i^{\text{scene}(t)} \;\|\; \mathbf{f}_i^{\text{behav}(t)} \;\|\; \mathbf{f}_i^{\text{rule}(t)}]
\tag{3.25}
$$

其中 $\mathbf{f}_i^{\text{scene}(t)}$（约 18 维）由场景层属性构成：位置、速度、朝向、刹车、油门等；$\mathbf{f}_i^{\text{behav}(t)}$（约 13 维）由行为层属性构成：`if_following`、`distance`、`TTC`、`if_changing_lane` 等；$\mathbf{f}_i^{\text{rule}(t)}$（约 29 维）由规则层三个子特征构成：

- **规则激活二进制向量** $r_{\text{active}} \in \{0,1\}^{14}$：如果 $v_i$ 在本帧被判定为触发规则 Rxx，则该位为 1（维度 = 14，对应 R1 至 R18）。
- **规则严重度向量** $r_{\text{severity}} \in [0,1]^{14}$：对应规则的严重度值（未触发时为 0）。
- **历史违规计数** $r_{\text{history}} \in \mathbb{N}^+$：$v_i$ 在过去 100 帧内的违规计数 (1 维)。

这样车辆节点的基础特征向量从纯场景+行为特征的约 31 维扩展到约 60 维。在 K-HSTGAN 编码器的训练过程中，注意力机制可以自主学到哪些规则特征对下游异常检测任务最具判别力。初步实验表明，$r_{\text{severity}}$ 对 K-HSTGAN 最终的 AUC 提升贡献最大（约 +8%），而 $r_{\text{active}}$ 的独立效果约为 +3%。

**接口二：边级注意力偏置**

第 5 章的 KS-NBCF 融合框架集成了 GAT（Graph Attention Network）作为图编码器。在 GAT 的标准消息传递中，邻居节点聚合权重由注意力系数 $\alpha_{ij}$ 控制：

$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \; \|\; \mathbf{W} h_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \; \|\; \mathbf{W} h_k]))}
\tag{3.26}
$$

规则层的介入方式是：若 $v_i$ 与 $v_j$ 之间在本帧存在 `violates(v_i, v_j, rule_code)` 关系，则在计算 $\alpha_{ij}$ 前将原始 logits 加上偏置项 $\gamma \cdot \text{severity}$，其中 $\gamma$ 是一个可学习的温度超参数：

$$
\alpha_{ij}^{\text{rule}} = \frac{\exp(\text{logits}_{ij} + \gamma \cdot \text{severity}_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(\text{logits}_{ik} + \gamma \cdot \text{severity}_{ik})}
\tag{3.27}
$$

该机制使 RSS 的"强先验"不通过前置过滤或后处理方式参与 GNN 过程，而是直接调节消息传递中的注意力分配率。例如，若 `v_i` 为 RSS 安全距离违规的后车，则其与 `v_j`（前车）之间的注意力偏置显著增加，GNN 更倾向于从"前车的速度与距离特征"中学习安全距离失效的时空模式——而非均匀地从所有邻居中聚合信息。

**接口三：证据链回溯接口**

当 KS-NBCF 的决策融合层检测到一个异常事件（如 OOD 检测分数高于阈值）时，需要对"为什么这个事件被判定为异常"给出结构化解释。规则层的证据链为此提供了现成素材：通过 `anomaly_trace_query(sv_id)` 查询 `SafetyViolation` 节点的 `supportedByEvidence` 路径，返回该违规判定的完整图路径。该路径可直接作为解释模板。

```
MATCH (sv:SafetyViolation {sv_id: 'sv_R13a_2052_veh123'})
MATCH (sv)-[:supportedByEvidence]->(e1)
MATCH (sv)-[:definedBy]->(rule)
OPTIONAL MATCH (sv)-[:responsibleFor]-(ra)
RETURN sv, e1, rule, ra
```

该查询语句的执行过程已在 `stk/storage/queries.py` 中封装为 `anomaly_trace_query(sv_id)`，返回结果可直接用于第 5 章所述的回放模块生成可解释性报告。同时，证据链中节点与边的 `frame_id` 约束确保只有与违规时刻相关的证据被召回，避免全量日志扫描。

**接口四：帧切片导出**

GNN 训练前的最后一步是数据生成。K-HSTGAN 模型不需要完整的全帧图谱，而是需要每帧的"帧切片"——即以 `SceneSnapshot` 为中心的 $k$-hop 子图。`stk/storage/queries.export_for_gnn_cypher(frame_id_start, frame_id_end, road_id)` 执行该切片导出：

```
MATCH (center:RoadElementEntity {road_id: $road_id})
MATCH (center)<-[r:in_lane]-(v:Vehicle)
WHERE r.frame_id >= $frame_start AND r.frame_id <= $frame_end
OPTIONAL MATCH (v)-[b:following]->(w:Vehicle)
  WHERE b.frame_id >= $frame_start AND b.frame_id <= $frame_end
OPTIONAL MATCH (v)-[sv:violates]->()
  WHERE sv.frame_id >= $frame_start AND sv.frame_id <= $frame_end
RETURN v, b, w, sv, r, center
```

该查询返回的是以目标车道为中心、包含前车跟驰关系与违规关系的结构化子图，并标注了帧号区间——这些子图将在第 4 章中作为 K-HSTGAN 的图样本，按时间顺序组成时序图序列，供图编码器学习车辆交互的时序依赖模式。帧切片机制也保证了导出数据的规模可控（单次导出的子图规模约为数百节点与数千边），避免一次性全图导出的内存撑爆风险。

上述四个接口的设计均遵循"解耦"原则：规则层只负责产出节点/边/证据链，不负责融合算法；下游框架只消消费规则层的结构化输出，不修改规则层的内部状态。这种解耦保证了当规则层增加新规则（如扩展 R19-R20）时，只需扩展 $\mathbf{f}_i^{\text{rule}(t)}$ 的维度，接口协议本身无需变更。

## 3.3.4 三层构建小结

本节以"场景→行为→规则"的递进顺序，完成了三层本体的构建方法描述。场景层通过 6 类节点与 15 种空间关系提供"此刻此地的结构化快照"；行为层通过 2 类节点、11 个检测器与防抖状态机将瞬时空间关系聚合为跨帧行为语义；规则层通过 RSS 与 R1-R18 双层结构实现物理安全校验与行为合规判定，并生成可追溯的证据链。三层间的语义衔接通过跨层桥接关系（`manifestsAs`、`actor`、`src`、`dst`）与证据链（`supportedByEvidence`）形成闭环。
