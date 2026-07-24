# 3.8 实验场景库

为统一验证 STKG 框架在自动驾驶模拟环境中的有效性，本节设计并实现了一个预置场景库，供后续第 6 章实验直接使用。该场景库基于 CARLA 0.9.16 真值数据，覆盖从简单到复杂、从单因素到多因素耦合的 14 个典型交通场景，并按图谱价值的来源划分为四个档次。

## 3.8.1 场景分类体系

场景库按风险复杂程度分为 A、B、C、D 四个档次，与自动驾驶安全测试中的"OOD 边缘场景越来越稀有"的逻辑相呼应。

**表 3-14** 预置场景分类总表
[三线表]

| 档位 | 场景 | 函数 | 类型 | 预期违规 |
|------|------|------|------|---------|
| A 基线 | S00 | `make_S00_baseline_following()` | 直行跟车基线 | 无 |
| | S01 | `make_S01_normal_signalized_intersection()` | 信号路口正常通行 | 无 |
| | S02 | `make_S02_pedestrian_far_avoidance()` | 行人远距避让 | 无 |
| B 单点异常 | S10 | `make_S10_pedestrian_sudden_crossing()` | 行人鬼探头 | R1 违规 |
| | S11 | `make_S11_unprotected_left_turn_conflict()` | 无信号左转冲突 | R7 违规 |
| | S12 | `make_S12_red_light_running()` | 红灯抢行 | R2 违规 |
| | S13 | `make_S13_too_close_following()` | 跟车过近 | R13a 违规 |
| C 多车冲突 | S20 | `make_S20_merging_conflict()` | 汇入主路冲突 | R7 + R13a |
| | S21 | `make_S21_three_way_unsignalized()` | 三车交叉无信号 | R7 + R3 |
| | S22 | `make_S22_emergency_vehicle_yielding()` | 应急车辆通行权 | R8 |
| D 跨层联动 | S30 | `make_S30_night_pedestrian_sudden()` | 夜间 + 鬼探头 | R1 + R11 |
| | S31 | `make_S31_rainy_lane_change_blind()` | 雨天跨线盲变 | R3 + R11 |
| | S32 | `make_S32_construction_detour()` | 施工路段绕行 | R14 + R17 |
| | S33 | `make_S33_glare_multi_pedestrian()` | 路口逆光 + 多行人 | R1 + R2 + R8 |

A 档 3 个基线场景完全不触发规则；B 档 4 个单点异常场景各触发一条规则；C 档 3 个多车冲突场景触发 2–3 条规则，且存在违规链式传播的可能；D 档 4 个跨层联动场景在环境因素（夜间、雨天、逆光等）的基础上叠加行为异常，验证规则在恶劣条件下的鲁棒性。

每个场景由 `stk/scenario/scenario_library.py` 中的工厂函数生成，产出 `List[FrameData]`，每帧以 400 ms（0.4 s）为间隔，每个场景固定 6 帧。`SCENARIO_REGISTRY` 注册表维护场景列表，`list_scenarios()` 列出全部注册场景，`get_scenario(name)` 获取指定场景的数据。

## 3.9 本章小结

本章详细描述了时空动态知识图谱（STKG）的完整构建方法。现将全章内容总结如下：

**3.2 节** 给出 STKG 的形式化定义 $\mathcal{STKG} := \langle \mathcal{E}, \mathcal{R}, \mathcal{A}, \mathcal{T}, \mathcal{P}, \{G_t\} \rangle$，建立时态三元组 $\tau$、节点-边-三元组类化表达、节点生命周期四状态机的符号基础；

**3.3 节** 设计四层本体总体架构，定义 14 种实体类型与 4 大类 42 种关系类型（场景层 15 种、行为层 13 种、规则层 7 种、跨层桥接 7 种），确立"节点+边双轨表达"原则与七条核心公理 $A_1$–$A_7$；

**3.4 节** 实现场景层 6 类节点（Vehicle/Pedestrian/TrafficLight/RoadElement/EnvSnapshot/ScenarioSnapshot）共 68 个属性字段，15 种空间关系通过纯函数计算实现"几何确定性"（`in_lane` 车道匹配、`ahead_of` 同车道纵距、`beside` 横向并排、`controlled_by` 灯道映射等），每帧平均生成约 $4.2 \times 10^3$ 条空间关系边；

**3.5 节** 实现行为层 2 类节点（Maneuver 单实体持续状态、Interaction 多实体交互事件）、13 种行为检测器与防抖状态机，覆盖跟车、接近、让行、超车、对向行驶、横穿等 11 种独立行为，防抖阈值在 1–5 帧之间按关系类型差异化设定；

**3.6 节** 实现规则层的 RSS 子层与交规子层双层架构：RSS 子层以 7 个参数实现纵向/横向安全距离公式与责任归因；交规子层实现 14 条法规规则（R1–R18，含跳号），与规则引擎 `RuleEnforcer` 统一管理所有规则的触发逻辑、跨帧状态维护与证据链生成；

**3.7 节** 实现动态更新机制：差分图 $\Delta g_t$ 覆盖四类更新动作（实体/属性/关系/规则事件），增量引擎采用五步流程（recv→diff→patch→eval→writeback），属性版本化 `VersionManager` 支持任意时点的属性时间旅行查询，滑动窗口聚合器支持跨帧复杂行为分析；

**3.8 节** 实现流式长时采集系统：分块采集（chunk=2000 帧）+ checkpoint 恢复 + 跨 chunk 状态持续，异常注入调度器支持 7 种异常类型泊松过程调度，图谱持久化支持 Neo4j 批量 MERGE 与 JSON 分片双后端。

**3.9 节**（本节）构建 14 个预置测试场景，按 A/B/C/D 四档覆盖从基线到跨层联动的风险梯度，为后续第 6 章实验提供标准化的验证输入。

全章实现了一个从 CARLA 仿真器原始数据输入到 4 层图图谱输出的完整端到端 pipeline。该 pipeline 在 70 个任务（14 场景 × 5 地图）的批量化测试中通过率达 100%（`tests/` 中的 25 个测试文件对其进行了覆盖验证）。下一章将在此基础上引入 K-HSTGAN 模型，利用 STKG 的时序特性进行深度异常检测。