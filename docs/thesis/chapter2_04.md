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

**纵向安全距离**：同车道后车 $A$ 与前车 $B$，纵向距离 $d_{\text{long}}$ 应满足（采用 Shalev-Shwartz 等 [Shalev-Shwartz et al., 2017] 与 Mobileye RSS v3.0 完整定义 [Mobileye, 2018]，含反应期加速项）：

\$$
d_{\text{long}}^{\min} = \max\left( 0,\ v_A \rho + \tfrac{1}{2} a_{\max,\text{accel},A} \rho^2 + \frac{\left(v_A + a_{\max,\text{accel},A} \rho\right)^2}{2\, a_{\min,\text{brake},A}} - \frac{v_B^2}{2\, a_{\max,\text{brake},B}} \right)
\tag{2.26}
\$$

其中各参数含义如表 2-5 所示。

**表 2-5** RSS 纵向公式参数
[三线表]
| 参数 | 含义 | RSS 默认值 |
|------|------|-----------|
| $v_A$ | 后车速度 | — |
| $v_B$ | 前车速度 | — |
| $\rho$ | 后车反应时间 | 0.3 s |
| $a_{\max,\text{accel},A}$ | 反应期内后车最大合理加速度 | 0.5 m/s² |
| $a_{\min,\text{brake},A}$ | 后车最小合理刹车加速度 | 3 m/s² |
| $a_{\max,\text{brake},B}$ | 前车最大合理刹车加速度 | 8 m/s² |

**横向安全距离**：相邻车道车 $A$ 与 $B$，横向距离 $d_{\text{lat}}$ 应满足（采用 Mobileye RSS v3.0 [Mobileye, 2018] Section 8.4 形式）：

\$$
d_{\text{lat}}^{\min} = \max\left( 0,\ \mu + \frac{v_{\text{lat},A}^2}{2\, a_{\min,\text{lat,brake},A}} + \rho\, v_{\text{lat},A} - \frac{v_{\text{lat},B}^2}{2\, a_{\min,\text{lat,brake},B}} \right)
\tag{2.27}
\$$

其中 $\mu$ 为横向最小安全余量（默认 0.5 m），$a_{\min,\text{lat,brake}}$ 为横向最小合理刹车加速度（默认 1.5 m/s²），$v_{\text{lat},A}$、$v_{\text{lat},B}$ 分别为车 $A$、$B$ 的横向速度分量，$\rho$ 同纵向定义。横向公式不含车宽项——车宽的影响在 §3.3.3 给出物理解释：横向距离是相对运动投影，$w_A,w_B$ 已隐含在车道级相对几何中，不应再以车宽乘子重复计入。

公式 (2.26) 与 (2.27) 是 RSS 模型的核心闭式表达。其物理逻辑为：**"在我车以当前速度刹车 → 走到完全停止" 过程中，与对方车的距离始终不小于对方车在最差情况下同时刹车的影响范围**。参数 $\rho$ 表示我车的反应滞后——我车在感知到危险后到开始刹车之间经过 $\rho$ 秒内的速度仍维持 $v_A$。

**纵向安全距离残差**：

\$$
\Delta d_{\text{long}} = d_{\min}^{\text{long}} - d_{\text{long}}
\tag{2.28}
\$$

$\Delta d_{\text{long}} < 0$ 表示安全（实际距离超出最小要求）；$\Delta d_{\text{long}} > 0$ 表示实际距离低于最小安全距离，触发违规事件 $\text{SafetyViolation}$（详见 §3.3.3 RSS 子层）。该残差方向与知识注入层 §4.4 公式 (4.24) 以及代码 `compute_kappa_rss`（`stk/gnn/exporter.py`）的工程实现一致——正向残差表示"危险程度"，负向残差表示"安全程度"。

**责任归因**：当两车的 RSS 距离残差超出安全阈值（即 $\Delta d_{\text{long}} > 0$ 或 $\Delta d_{\text{lat}} > 0$，表示实际距离低于最小安全距离），规则引擎通过对比各自是否遵守 RSS 距离确定责任：

- 后车 $A$ 满足 $d_{\text{long}} \geq d^{\min}_{\text{long}}$（即 $A$ 已保持安全距离） → 责任在前车
- 后车 $A$ 不满足 $d_{\text{long}} \geq d^{\min}_{\text{long}}$ 但前车不正常急刹（前车减速度超过 $a_{\max,\text{brake},B}$ 的合理上界） → 责任在前车
- 后车 $A$ 不满足 $d_{\text{long}} \geq d^{\min}_{\text{long}}$ 而前车正常减速 → 责任在后车

**RSS 扩充场景规则**：基本 RSS 模型仅覆盖同车道匀速跟驰与横向并列场景，Mobileye RSS v3.0 [Mobileye, 2018] 在基本模型外针对动态交互与道路条件定义了四项扩充规则：（1）**Cut-in 切入安全距离**——旁车 $B$ 完成变道切入后纵向距离需满足 $1.5 \times d_{\min}^{\text{long}}$；（2）**Cut-out 驶离缓冲**——前车 $B$ 变道驶离后 $3$ s 缓冲期内新跟驰关系需保持原 RSS 距离；（3）**反应不当增强判定**——在 NoProperResponse 基础上加入制动速率学约束 $\text{brake}_{jerk}$ 与反应时间-速度因变阈值；（4）**施工路段参数自适应**——进入 `construction_zone` 路段时 RSS 参数集 $\Theta_{\text{RSS}}$ 替换为 $\Theta_{\text{RSS}}^{\text{cz}}$（$\rho + 0.1$ s、$2\,a_{\max,\text{accel}}$、$0.75\,a_{\min,\text{brake}}$ 等）。四项扩充规则的形式化定义、判别公式及本文工程状态见 §3.3.3.1a 表 3-17a。本文将这四项作为框架性描述纳入论文（代码暂未实现，可在后续版本中逐步扩展），保留与 Mobileye v3.0 标准的兼容性。

## 2.4.4 中国交通法规规则

RSS 模型覆盖车辆间物理安全距离，但不涉及行为合规性（如闯红灯、不按导向车道行驶、违章停车等）。中国《道路交通安全法》及配套实施条例是行为合规性判别的法律依据。本文第 3 章 §3.3.3.2 交通法规子层从中提炼 11 条与自动驾驶直接相关的规则（编号 R1–R18，含 R6/R12/R14/R15 跳号空缺），与 §3.3.3 RSS 子层的 3 项物理先验规则（R13a 纵向、R14a 横向、R15a 横向危险状态）共同构成规则层。各规则的判定条件、关键阈值与法规依据详见 §3.3.3 表 3-18，此处仅作概览列举：

- R1 行人优先（道交法第47条）
- R2 闯红灯（道交法第38条）
- R3 实线变道（道交法第44条）
- R4 对向会车违规（实施条例第48条）
- R5 违章倒车（道交法第35条）
- R7 路口未让行（道交法第52条）
- R8 弱势参与者保护（道交法第42条）
- R9 学区限速（地方实施办法）
- R10 高速限速（实施条例第78条）
- R11 恶劣天气限速（实施条例第81条第3款）
- R13 违法停车（道交法第63条）
- R16 黄灯抢行（道交法第26条）
- R17 不按规定车道行驶（实施条例第44条）
- R18 路口导向车道违规（实施条例第51条）

R6（违规掉头）、R12（备用）为开发优先级预留空缺，R14（违反交通标志）、R15（违反标线）在 v1 实现中因与 R3 实线变道存在语义重叠且依赖的 CARLA 标志数据不全而暂未实现。RSS 子层的 R13a/R14a/R15a 后缀 `a` 表示 RSS 派生形式，与不带后缀的交规条文规则区分。

（${}^\dagger$ 表示在 RSS 物理安全的"硬规则"之外补充的"软规则"判定。）

## 2.4.5 与本文第 3 章设计的接续关系

**表 2-6** RSS 与交规理论工具与第 3 章设计对应
[三线表]
| 本节理论工具 | 第 3 章对应设计 | 节号 |
|------------|---------------|------|
| RSS 公式 (2.26)(2.27) | 规则层 RSS 子层 3 个核心算子（纵向、横向、责任归因）| §3.3.3.1 |
| RSS 7 个参数 | `DEFAULT_RSS_PARAMS` 参数表 | §3.3.3.1, 表 3-17 |
| 责任归因规则 | `ResponsibilityAssignment` 节点与 `responsibleFor` 边 | §3.3.3.1 Eq.(3.16)-(3.17) |
| 中国交规 11 条规则 | 规则层交通法规子层 `check_Ri_*` 函数与 R1–R18 编号 | §3.3.3.2, 表 3-18 |
| CARLA 仿真器 Actor API | 场景层 Vehicle/Pedestrian/TrafficLight 节点的属性获取接口 | §3.3.1 |
| CARLA 同步帧 `world.tick()` + ScenarioRunner | §3.7 长时流式采集 chunk=2000 帧 + checkpoint 恢复机制 | §3.7 |
| EnvSnapshot 节点 | CARLA 天气状态与时间戳映射为 STKG 中的环境节点 | §3.3.1 |

本节内容提供了第 3 章规则层设计与仿真器接口设计的理论基线。RSS 公式的封闭性使得第 3 章规则层可用纯函数计算实现"几何确定性违规检出"，而 CARLA 的地面真值 API 保证了场景层 15 种空间关系同样可作为"几何确定性关系"由纯函数计算实现。这两条几何确定性链路共同确保了 STKG 第 3 章规则层与场景层的高可靠性，是本文后续在第 4 章基于数据驱动 GNN 异常检测模型上的"可信基础"。