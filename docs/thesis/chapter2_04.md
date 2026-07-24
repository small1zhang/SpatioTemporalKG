# 2.4 自动驾驶仿真器与 RSS 安全模型

## 2.4.1 CARLA 仿真器

CARLA（Car Learning to Act）是 Intel 实验室与巴塞罗那计算机视觉中心（CVC）于 2017 年 CoRL 会议上发布的开源自动驾驶仿真平台 [6]。截至本文写作时，CARLA 已发布至 0.9.16 版本，是学术界自动驾驶研究中应用最广泛的开源仿真器之一。

CARLA 的核心架构与本文工作相关的特性如下：

**（1）地图与场景。** CARLA 内置 8 张地图（Town01–Town07、Town10HD），覆盖村镇、城市高速、环岛、隧道、桥梁、施工路段等多种道路拓扑。每张地图的 OpenDRIVE 文件描述车道线、车道宽度、交叉口等结构化道路信息，可通过 Python API 直接获取。

**（2）传感器。** CARLA 提供摄像头、激光雷达、雷达、IMU、GPS、GNSS、语义分割相机、深度相机、实例分割相机等 16 类传感器，可挂在任意车辆上。本论文主要使用：（i）RGB 相机（可视化与人工评审证据链）；（ii）激光雷达（用于车辆位置验证）；（iii）语义分割相机（用于场景层 `in_lane` 等几何关系的辅助验证）。

**（3）Actor 与 API。** CARLA 中的实体（vehicle、walker、traffic light、traffic sign、speed limit sign、static obstacle）均为 Actor，通过 `carla.World.get_actors()` 可获取所有 Actor 的引用。每个 Actor 提供 `get_location()`、`get_transform()`、`get_velocity()`、`get_angular_velocity()` 等 API 直接返回地面真值（Ground Truth）——无需经过感知管线即可获取高精度物理量。这是本文选用 CARLA 作为 STKG 构建平台的核心理由：地面真值的高精度使得第 3 章场景层 15 种空间关系可作为"几何确定性关系"通过纯函数计算实现，无需感知学习过程。

**（4）天气与时间。** CARLA 提供 6 种基础天气（晴天、阴天、雨天、雾天、雨夹雪、夜晚）与若干混合天气，可通过 `world.set_weather(...)` 切换。`world.set_timeofday(...)` 可控制仿真时间。本文将天气状态作为场景层的 `EnvSnapshot` 节点属性，与车辆、行人等共享场景层。

**（5）Python 客户端 SDK 与同步帧机制。** CARLA 客户端通过 `carla.Client(host, port)` 连接服务器，以 `world.tick()` 推进一帧（同步模式，20 Hz 默认），这一帧推进机制与本论文第 3 章 §3.7 流式采集的 chunk=2000 帧 + checkpoint 恢复机制直接对接。

## 2.4.2 SceneRunner 与 ASAM OpenSCENARIO

针对自动驾驶仿真场景的标准化描述，工业界已建立 ASAM OpenSCENARIO [25] 标准。OpenSCENARIO 描述语言以场景的"开始条件 → 事件触发 → 动作序列"为核心抽象，支持车辆行为脚本化、信号灯时序控制、行人运动等。CARLA 官方维护 ScenarioRunner [24] 工具集作为 OpenSCENARIO 标准的开源实现。

OpenSCENARIO 侧重于场景"剧本式"执行规则，不天然提供场景的知识图谱化表示。本论文工作在 ScenarioRunner 之上引入 STKG 作为"剧本到知识图谱"的中间层，将"车辆 A 在 t3 时刻变道进入车道 B"这一动作转化为 STKG 中的 `ManeuverEvent` 节点与 `performs` 关系。

## 2.4.3 RSS 安全模型

Shalev-Shwartz 等人在 2017 年提出 RSS（Responsibility-Sensitive Safety）模型 [10]，是自动驾驶形式化安全理论的代表性工作。RSS 定义了纵向与横向安全距离的封闭性公式与责任归因规则。

**纵向安全距离**：同车道后车 $A$ 与前车 $B$，纵向距离 $d_{\text{long}}$ 应满足：

\$$
d_{\text{long}}^{\min} = \max\left( 0,\ \frac{v_A^2}{2 a_{\min,\text{brake},A}} - \frac{v_B^2}{2 a_{\max,\text{brake},B}} + v_A \rho \right)
\tag{2.26}
\$$

其中各参数含义如表 2-5 所示。

**表 2-5** RSS 纵向公式参数
[三线表]
| 参数 | 含义 | RSS 默认值 |
|------|------|-----------|
| $v_A$ | 后车速度 | — |
| $v_B$ | 前车速度 | — |
| $a_{\min,\text{brake},A}$ | 后车最小合理刹车加速度 | 2 m/s² |
| $a_{\max,\text{brake},B}$ | 前车最大合理刹车加速度 | 4 m/s² |
| $\rho$ | 后车反应时间 | 0.1 s |

**横向安全距离**：相邻车道车 $A$ 与 $B$，横向距离 $d_{\text{lat}}$ 应满足：

\$$
d_{\text{lat}}^{\min} = \mu + \max\left( 0,\ \frac{w_A (v_{A,\perp} + v_{AB,\perp})}{2 a_{\min,\text{lat},A}} + \frac{w_B v_{AB,\perp}}{2 a_{\min,\text{lat},B}} \right)
\tag{2.27}
\$$

其中 $w_A, w_B$ 为车宽，$\mu$ 为横向最小安全余量，$a_{\min,\text{lat}}$ 为横向最小合理刹车加速度（默认 1 m/s²），$v_{\perp}$ 为横向速度。

公式 (2.26) 与 (2.27) 是 RSS 模型的核心闭式表达。其物理逻辑为：**"在我车以当前速度刹车 → 走到完全停止" 过程中，与对方车的距离始终不小于对方车在最差情况下同时刹车的影响范围**。参数 $\rho$ 表示我车的反应滞后——我车在感知到危险后到开始刹车之间经过 $\rho$ 秒内的速度仍维持 $v_A$。

**纵向安全距离残差**：

\$$
\Delta d_{\text{long}} = d_{\text{long}} - d_{\text{long}}^{\min}
\tag{2.28}
\$$

$\Delta d_{\text{long}} > 0$ 表示安全；$< 0$ 表示违规，触发违规事件 $\text{SafetyViolation}$（详见 §3.5 RSS 子层）。

**责任归因**：当两车发生碰撞或同时发生 $\Delta d_{\text{long}} < 0$，规则引擎通过对比各自是否遵守 RSS 距离确定责任：

- 后车 $A$ 满足 $d \geq d^{\min}_{\text{long}}$ → 责任在前车
- 后车 $A$ 不满足 $d \geq d^{\min}_{\text{long}}$ 但前车不正常急刹 → 责任在前车
- 后车 $A$ 不满足 $d \geq d^{\min}_{\text{long}}$ 而前车正常减速 → 责任在后车

## 2.4.4 中国交通法规规则

RSS 模型覆盖车辆间物理安全距离，但不涉及行为合规性（如闯红灯、不按导向车道行驶、违章停车等）。中国《道路交通安全法》及配套实施条例是行为合规性判别的法律依据。本文第 3 章 §3.5.3 交通法规子层从中提炼 14 条与自动驾驶直接相关的规则，按 R1–R18 编号（含 R6/R12/R14 未实现跳号），包括：

- R1${}^\dagger$ 不按规定车道行驶（车道非"专用车道"+未保持车道居中）
- R2 闯红灯（红灯时仍越过停止线）
- R3${}^\dagger$ 不按导向车道行驶（路口范围内、转向方向与导向车道不一致）
- R4 违反禁止标线（压实线/双黄线变道）
- R5${}^\dagger$ 违章超车（弯道/路口/陡坡超车）
- R7 不按规定跟车（跟车距离<31m 且车速>32m/s）
- R8${}^\dagger$ 不让行（无信号路口、未让右侧来车）
- R10${}^\dagger$ 不按停车规定停车（在禁停标志处停车）
- R11${}^\dagger$ 未保持侧向安全距离
- R13 违反禁停（持续帧数 > 30）
- R14a${}^\dagger$ 违反让行规则（会车时未让右车）
- R15–R18 高级规则（暂仅占位）

（${}^\dagger$ 表示在 RSS 物理安全的"硬规则"之外补充的"软规则"判定。）

## 2.4.5 与本文第 3 章设计的接续关系

**表 2-6** RSS 与交规理论工具与第 3 章设计对应
[三线表]
| 本节理论工具 | 第 3 章对应设计 | 节号 |
|------------|---------------|------|
| RSS 公式 (2.26)(2.27) | 规则层 RSS 子层 3 个核心算子（纵向、横向、责任归因）| §3.5.2 |
| RSS 7 个参数 | `DEFAULT_RSS_PARAMS` 参数表 | §3.5.2, 表 3-8 |
| 责任归因规则 | `ResponsibilityAssignment` 节点与 `assigns` 边 | §3.5.4 |
| 中国交规 14 条规则 | 规则层交通法规子层 `check_Ri_*` 函数与 R1–R18 编号 | §3.5.3 |
| CARLA 仿真器 Actor API | 场景层 Vehicle/Pedestrian/TrafficLight 节点的属性获取接口 | §3.3 §3.7 |
| CARLA 同步帧 `world.tick()` + ScenarioRunner | §3.7 长时流式采集 chunk=2000 帧 + checkpoint 恢复机制 | §3.7 |
| EnvSnapshot 节点 | CARLA 天气状态与时间戳映射为 STKG 中的环境节点 | §3.3 |

本节内容提供了第 3 章规则层设计与仿真器接口设计的理论基线。RSS 公式的封闭性使得第 3 章规则层可用纯函数计算实现"几何确定性违规检出"，而 CARLA 的地面真值 API 保证了场景层 15 种空间关系同样可作为"几何确定性关系"由纯函数计算实现。这两条几何确定性链路共同确保了 STKG 第 3 章规则层与场景层的高可靠性，是本文后续在第 4 章基于数据驱动 GNN 异常检测模型上的"可信基础"。