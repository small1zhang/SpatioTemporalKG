# 3.5 规则层：RSS 与交通法规推理

规则层是 STKG 语义抽象的最高层，回答"这些行为合不合规、危不危险"。它的输入是场景层实体属性与行为层检测结果，输出是 `SafetyViolation` 节点、`violates` 边、`supportedByEvidence` 证据链以及 `ResponsibilityAssignment` 责任归因。规则层在架构上分为两个互补子层——**RSS 子层**（负责车辆间纵横向物理安全距离校验）和 **交通法规子层**（负责行为合规性的法规判别），二者共同构成完整的符号推理系统。

## 3.5.1 规则层双层架构

规则层的双层设计背后有两个动机。第一，RSS 提供的是**可证明安全的物理约束**（responsibility-sensitive safety），它在每对同车道跟驰车辆之间定义一个明确的最小安全距离，小于该距离即判定为危险状态——这种判定不依赖任何学习或场景知识，是"强先验"的。第二，交规子层覆盖了 RSS 未涉及的规则维度，如闯红灯、实线变道、不按规定车道等，但交规子层的判定需要场景层信号灯状态与行为层变道检测结果的协同，是"弱先验"的。二者的结合使得 STKG 的规则推理既可覆盖"物理碰撞风险"，也可覆盖"行为合规违规"。

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

## 3.5.2 RSS 子层

RSS（Responsibility-Sensitive Safety）由 Shalev-Shwartz 等[Shalev-Shwartz, 2017] 提出，为自动驾驶设计了一套形式化的安全模型。核心思想是：若每辆车始终保持一个**随其速度动态变化的最小安全距离**，则所有碰撞的责任均可归咎于未保持该距离的车辆。本文实现 RSS 子层的三个核心算子——纵向安全距离、横向安全距离与复合状态判定——并基于此进行责任归因。

### 3.5.2.1 纵向安全距离

给定前车 $B$ 与跟驰后车 $A$，两车同车道且 $\text{ahead\_of}(B, A)$ 成立，则后车 $A$ 必须与 $B$ 保持的纵向最小安全距离 $d_{\min}^{\text{long}}(A, B, t)$ 定义为：

\$$
d_{\min}^{\text{long}}(A, B, t) = \max\!\left(0,\ v_A \rho + \frac12 a_{\max,\text{accel}} \rho^2 + \frac{(v_A + a_{\max,\text{accel}}\rho)^2}{2\,a_{\min,\text{brake}}} - \frac{v_B^2}{2\,a_{\text{brake}}}\right)
\tag{3.6}
\$$

其中各参数的含义与取值如表 3-8 所示。

**表 3-8** RSS 子层参数表（`DEFAULT_RSS_PARAMS`）
[三线表]

| 参数 | 符号 | 值 | 单位 | 描述 |
|------|------|-----|------|------|
| $\rho$ | 反应时间 | 0.3 | s | 自车从感知到路面信号到做出反应的时间 |
| $a_{\max,\text{accel}}$ | 后车最大加速度 | 0.5 | m/s² | 后车在反应时间内合理加速的上限 |
| $a_{\min,\text{brake}}$ | 后车最小制动 | 3.0 | m/s² | 后车在反应后至少能实现的制动减速度 |
| $a_{\text{brake}}$ | 前车最大制动 | 8.0 | m/s² | 前车可能实现的极端制动减速度 |

纵向危险状态判定：

\$$
\text{SafeDistanceViolation}(A, B, t) \iff d_{\text{long}}(A, B, t) < d_{\min}^{\text{long}}(A, B, t)
\tag{3.7}
\$$

式中 $d_{\text{long}}(A, B, t)$ 为后车 $A$ 前端到前车 $B$ 后端的实测纵向距离，由 `check_safe_distance_longitudinal(d_actual, v_A, v_B, params)` 实现。

### 3.5.2.2 横向安全距离

当两车横向间距过近时可能产生刮擦或碰撞。横向最小安全距离定义为：

\$$
d_{\min}^{\text{lat}}(A, B, t) = \mu + \frac{v_{\text{lat},A}^2}{2\,a_{\min,\text{lat,brake}}} + \rho\,v_{\text{lat},A} - \frac{v_{\text{lat},B}^2}{2\,a_{\min,\text{lat,brake},B}}
\tag{3.8}
\$$

其中 $\mu = 0.5$ m 为横向安全裕度，$a_{\min,\text{lat,brake}} = 1.5$ m/s² 为最小横向制动减速度，$v_{\text{lat}}$ 为横向速度。

横向危险状态判定：

\$$
\text{LateralDangerousState}(A, B, t) \iff d_{\text{lat}}(A, B, t) < d_{\min}^{\text{lat}}(A, B, t)
\tag{3.9}
\$$

### 3.5.2.3 复合状态与责任归因

RSS 将车道并线、横穿等交互场景视为纵向与横向危险状态的组合：

\$$
\text{DangerousState}(A, B, t) = \text{SafeDistanceViolation}(A, B, t) \lor \text{LateralDangerousState}(A, B, t)
\tag{3.10}
\$$

RSS 还定义"反应不当"判定：若 $\text{DangerousState}(A, B, t)$ 在连续 3 帧内，$A$ 的制动踏板 $< 0.3$（即 $A$ 未采取充分制动），则触发：

\$$
\text{NoProperResponse}(A, t) = \bigwedge_{k=0}^{2} \text{brake}(A, t+k) < 0.3
\tag{3.11}
\$$

责任归因逻辑：

\$$
\text{ResponsibleAgent}(A, \text{event}) \iff \text{NoProperResponse}(A) \land B \text{ 行为合规}
\tag{3.12}
\$$

此处"行为合规"定义为 $B$ 在相关帧未触发任何交规违规。

`run_rss_check(d_long, d_lat, v_A, v_B, v_lat_A, v_lat_B, brake_values)` 是 RSS 综合入口函数，返回包含 `is_long_violation`、`d_min_long`、`is_lat_violation`、`d_min_lat`、`is_dangerous`、`is_no_proper_response`、`is_responsible` 等九项输出。

## 3.5.3 交通法规子层（R1-R18）

交通法规子层覆盖中国《道路交通安全法》中 14 条与自动驾驶直接相关的规则（R1-R18，含 R6/R12/R14 未实现）。每条规则都由一个独立的 `check_Ri_*` 函数实现，输入为场景层实体与行为层输出，输出为 `(is_violation, severity, evidence)` 三元组。表 3-9 汇总全部 14 条规则。

**表 3-9** 交通法规规则清单
[三线表]

| 规则 | 函数名 | 谓词名 | 判定条件摘要 |
|------|--------|--------|-------------|
| R1 | `check_R1_pedestrian_priority` | `YieldingToPedestrianViolation` | 行人在横道线 + 车辆距离 $<$ 15 m + 车速 $>$ 0.5 m/s |
| R2 | `check_R2_red_light` | `RedLightViolation` | $v$ 在路口 + 信号灯变为红 + 车速 $>$ 0.3 m/s |
| R3 | `check_R3_solid_line_change` | `IllegalLaneCrossing` | `changing_lane` + 跨越的车道标线类型为实线 |
| R4 | `check_R4_opposite_meeting` | `WrongSideMeetingViolation` | `opposite_direction` + 距离 $<$ 10 m |
| R5 | `check_R5_reversing` | `IllegalReversing` | 速度方向与车头朝向夹角 $> 135°$ 持续 5 帧 |
| R7 | `check_R7_junction_no_yield` | `JunctionNoYieldViolation` | 进入路口 + 未让行主路车辆 |
| R8 | `check_R8_vulnerable_protection` | `VulnerableUserProtectionViolation` | 天气恶劣 + 行人距离 $<$ 20 m + 车速超限 |
| R9 | `check_R9_school_zone_speed` | `SchoolZoneSpeedViolation` | 在学区路段 + 车速 $>$ 30 km/h |
| R10 | `check_R10_highway_speed` | `HighwaySpeedViolation` | 在高速路段 + 车速 $>$ 120 km/h 或 $<$ 60 km/h |
| R11 | `check_R11_weather_speed` | `WeatherSpeedViolation` | 强降水 + 车速超过天气允许上限 |
| R13 | `check_R13_illegal_stop` | `IllegalStopViolation` | 在禁停区 + 车速 $<$ 0.3 m/s 持续 30 帧 |
| R16 | `check_R16_amber_jumping` | `AmberLightJumpingViolation` | 黄灯亮时 $v$ 已在路口外 + 未减速进入路口 |
| R17 | `check_R17_wrong_lane` | `WrongLaneViolation` | 驶入错车道 + 持续 5 帧 |
| R18 | `check_R18_wrong_direction_lane` | `WrongDirectionLaneViolation` | 在路口导向车道内 + 车辆行为与导向方向不一致 |

## 3.5.4 RuleEnforcer 主生成器

`RuleEnforcer`（`stk/rules/generator.py`）是规则层的主驱动。其核心方法 `enforce()` 在一个调用中完成 RSS 扫描与全部 14 条交规检查，并统一生成后续输出。算法伪代码如下：

```
算法 3.3: RuleEnforcer.enforce(frame_id, entities, scene_rels, behavior_rels)
输入: frame_id, 当前帧全部实体与关系
输出: {violations, violation_rels, defined_by_rels, evidence_rels, responsibilities, resp_rels}

1. violations ← []
2. violation_rels ← [] // violates 边
3. defined_by_rels ← []
4. evidence_rels ← []  // supportedByEvidence 边
5. responsibilities ← []
6. resp_rels ← []      // responsibleFor 边
7.
8. // === RSS 扫描 ===
9. rss_pairs ← rss_pair_scan(vehicles, scene_rels)
10. for each (A, B, d_long, d_lat, v_A, v_B, brake_history) in rss_pairs:
11.    rss_result ← run_rss_check(d_long, d_lat, v_A, v_B, ...)
12.    if rss_result.is_dangerous:
13.        sv ← create SafetyViolation(R13a/R14a, severity)
14.        violations.append(sv)
15.        violation_rels.append(violates(src, dst, ...))
16.        defined_by_rels.append(defined_by(sv, R13a, ...))
17.        if rss_result.is_no_proper_response && rss_result.is_responsible:
18.            resp ← create ResponsibilityAssignment(sv, A)
19.            responsibilities.append(resp)
20.            resp_rels.append(responsibleFor(resp, sv))
21.        end if
22.    end if
23. end for
24.
25. // === 交通法规扫描 ===
26. for each rule_fn in [R1, R2, ..., R18]:
27.    is_violation, severity, evidence ← rule_fn(frame_id, entities, ...)
28.    if is_violation:
29.        sv ← create SafetyViolation(rule_code, severity)
30.        sv.evidence_path ← evidence
31.        violations.append(sv)
32.        violation_rels.append(violates(src, dst, ...))
33.        defined_by_rels.append(defined_by(sv, rule_code))
34.        for each evidence_item in evidence:
35.            evidence_rels.append(supportedByEvidence(sv, evidence_item))
36.        end for
37.    end if
38. end for
39.
40. _sv_counter += len(violations)
41. return {violations, violation_rels, defined_by_rels, evidence_rels, responsibilities, resp_rels}
```

`RuleEnforcer` 维护两个跨帧状态用于复杂判定：
- `_brake_history`: `{vehicle_id: [brake_values_list]}`，存储每辆车最近 30 帧的制动记录，用于 RSS `NoProperResponse` 判定所需的 3 帧制动历史检查；
- `_stop_duration`: `{vehicle_id: frames}`，存储每辆车连续静止帧数，用于 R13 禁停判定。

这些状态在 3.6 节增量更新中通过 `IncrementalEngine.checkpoint` 序列化与恢复，保证跨 chunk 和跨 restart 的规则判定一致性。

## 3.5.5 证据链生成

每条 `SafetyViolation` 生成时，`RuleEnforcer` 会构建一条完整的证据链，记录导致该违规判定的事件源。证据链结构为：

```
sv_R13a_2052_veh123
  ├── [supportedByEvidence idx=0] → triple_in_lane_2052_veh123_lane2
  ├── [supportedByEvidence idx=1] → triple_ahead_of_2052_veh124_veh123
  └── [supportedByEvidence idx=2] → triple_following_2052_veh123_veh124
```

每条 `supportedByEvidence` 边的属性包括 `evidence_idx`（证据在链中的位置索引）与 `evidence_type`（如 `scene_rel`、`behavior_rel`、`rule_evt`）。证据链的末端可以是：
- 场景层空间关系边 ID（`triple_<type>_<frame>_<src>_<dst>`）
- 行为层 InteractionEvent 节点 ID（`int_<src>_<dst>_<type>_<frame>`）
- 行为层 ManeuverNode 节点 ID（`man_<actor>_<frame>`）

证据链的存在使得规则的判定结果对下游完全可解释——任何违规都可以通过 `MATCH (sv)-[:supportedByEvidence]->(e)` 的图查询追踪到触发该违规的具体场景/行为事实。这一机制是 KS-NBCF 融合框架（大论文第 5 章）中冲突消解路径回溯的依赖基础。

## 3.5.6 规则参数配置与温度缩放

RSS 参数与交规阈值可通过 YAML 配置文件动态调整，无需修改代码。`config/rss_rules.yaml` 定义 7 个 RSS 参数，`config/traffic_rules.yaml` 定义 18 条规则的触发阈值（如 R2 闯红灯车速阈值 0.3 m/s、R13 禁停持续帧数 30 等）。此外代码 `DEFAULT_RSS_PARAMS` 中保留了与 RSS 论文默认参数一致的基准值，两者之间的差异（如 `rho`: 代码 0.3 s vs 配置 0.1 s）可通过实验对比分析不同参数集对检测率的影响。

为应对 CARLA 物理仿真的微小扰动（如浮点噪声、碰撞检测波动），每条规则的 severity 输出经过 `severity = min(1.0, (threshold - value) / threshold)` 的"温度缩放"处理——当值刚好超过阈值时 severity 很小，大幅超过时才接近 1.0。这避免了"帧帧触发大量接近阈值的低严重度违规"的问题。

## 3.5.7 节点与关系类型汇总

规则层涉及 4 类节点与 7 种关系，表 3-10 与 3-11 列出了全部。

**表 3-10** 规则层节点类型
[三线表]

| 节点 | Neo4j Label | 属性 | 作用 |
|------|-------------|------|------|
| `RuleDefinition` | `Rule` | rule_id, rule_name, rule_layer, predicate_name | 规则元定义 |
| `RuleParameter` | `Param` | param_id, name, value, unit, rule_id | 参数元数据 |
| `SafetyViolation` | `SafetyViolation` | sv_id, rule_code, rule_name, rule_layer, frame_id, severity, predicate_str, evidence_path | 违规实例 |
| `ResponsibilityAssignment` | `Responsibility` | resp_id, sv_id, responsible_actor_id, reason | 责任归属 |

**表 3-11** 规则层关系类型
[三线表]

| 关系 | RuleRelationType | 源→目标 | 说明 |
|------|-----------------|---------|------|
| `defined_by` | `definedBy` | `SafetyViolation` → `Rule` | 违规归属到规则定义 |
| `uses_param` | `usesParam` | `Rule` → `Param` | 规则使用参数 |
| `violates` | `violates` | 违规源实体 → 违规目标实体 | 违规边 |
| `supportedByEvidence` | `supportedByEvidence` | `SafetyViolation` → 场景/行为证据 | 证据链 |
| `triggers` | `triggers` | `SafetyViolation` → `SafetyViolation` | 因果链触发 |
| `responsibleFor` | `responsibleFor` | `Responsibility` → `SafetyViolation` | 责任归属 |
| `causedBy` | `causedBy` | `SafetyViolation` → `SafetyViolation` | 因果链导致 |

其中 `triggers` 与 `causedBy` 支持复杂场景下的违规链式传播——如"跟车过近导致紧急变道→变道过程实线违规"的双违规连锁关系。

## 3.5.8 小结

本节设计了 STKG 规则层的双层结构：RSS 子层提供物理安全的可证明校验（3 个核心算子、7 个参数），交通法规子层覆盖 14 条驾驶行为合规规则。`RuleEnforcer` 统一管理所有规则的触发逻辑、证据链生成与跨帧状态维护。规则层的输出 `SafetyViolation` 节点与 `supportedByEvidence` 边构成后续框架中可解释性分析的基础。
