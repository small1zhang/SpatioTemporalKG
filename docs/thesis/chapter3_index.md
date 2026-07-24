# 第 3 章 时空动态知识图谱构建（全文索引）

> 对应设计文档：`docs/v3_paragraphs.txt` + 参考素材（方案A v3 / v8.3 文档）
> 对应代码：`stk/ontology/`, `stk/scenario/`, `stk/behavior/`, `stk/rules/`, `stk/dynamic/`, `stk/storage/`, `stk/filter/`, `scripts/long_run/`
> 对应测试：`tests/`（25 个测试文件，覆盖全部核心模块）
> 重构方案：见 `docs/thesis/SDD_chapter3_restructure.md`

---

## 章节结构（方案 B：减少标题层级，3.1做深、3.3做大）

| 序号 | 内容 | 对应代码模块 | 文件 | 字符数 |
|------|------|-------------|------|-------|
| 3.1 | 问题定义与形式化（含文献对比） | `stk/ontology/` | `chapter3_01.md` | ~10.5k |
| 3.2 | 四层本体总体设计 | `stk/ontology/`, `stk/ontology/axioms.py` | `chapter3_02.md` | ~12.8k |
| 3.3 | 三层构建（场景+行为+规则） | `stk/scenario/`, `stk/behavior/`, `stk/rules/` | `chapter3_03.md` | ~35.4k |
| 3.4 | 动态更新机制 | `stk/dynamic/` | `chapter3_04.md` | ~10.3k |
| 3.5 | 流式长时采集与存储 | `scripts/long_run/`, `stk/storage/`, `stk/filter/` | `chapter3_05.md` | ~12.6k |
| 3.6 | 实验场景库 | `stk/scenario/scenario_library.py` | `chapter3_06.md` | ~7.2k |
| 3.7 | 本章小结 | — | `chapter3_07.md` | ~4.5k |
| **合计** | **全章** | — | — | **~93.3k 字符** |

---

## 核心数据统计（已与代码对齐）

| 维度 | 数值 | 来源 |
|------|------|------|
| 实体类型 | 14 类 | `stk/ontology/types.py` |
| 场景层关系 | 15 种 | `SceneRelationType` |
| 行为层关系 | 13 种 | `BehaviorRelationType` |
| 规则层关系 | 7 种 | `RuleRelationType` |
| 跨层桥接关系 | 7 种 | `CrossLayerRelationType` |
| 关系总数 | 42 种 | 4 大类之和 |
| VehicleEntity 属性数 | 28 | `actor_extractor.py` |
| PedestrianEntity 属性数 | 13 | `actor_extractor.py` |
| 场景节点 6 类属性字段总数 | 77 | 各 entity 属性和 |
| 行为检测器 | 11 个 | `stk/behavior/detectors.py` |
| 行为节点 | 2 类（Maneuver + Interaction）| `stk/behavior/nodes.py` |
| 防抖关系 | 13 种，阈值 1–5 帧（代码值） | `stk/behavior/debouncer.py` |
| RSS 参数 | 7 个（含横向 μ、a_min_lat_brake）| `stk/rules/rss/model.py` |
| 交规规则 | 14 条（R1-R18 含跳号） | `stk/rules/traffic/rules.py` |
| 核心公理 | 7 条（A1-A7） | `stk/ontology/axioms.py` |
| 预置场景 | 14 个（A/B/C/D 四类） | `stk/scenario/scenario_library.py` |
| 异常注入类型 | 7 种（λ=0.005/帧）| `scripts/long_run/anomaly_scheduler.py` |
| 流式分块 | 2000 帧/块，24000 帧最大 | `scripts/long_run/collect.py` |
| 节点生命周期参数 | $N_{\text{stale}}=3$, $N_{\text{forget}}=30$ | `ThresholdConfig` |
| 全栈压缩体系 | 19 个 FE 优化、3 道裁剪、压缩比>95% | FE-1 ~ FE-18 |

---

## 写作约定

- 公式编号：`(3.1)`, `(3.2)` 等，按章内连续编号
- 表编号：`表 3-1`, `表 3-2` 至 `表 3-26` 已分配
- 图编号：`图 3-1`（四层架构）, `图 3-2`（数据流）, `图 3-3`（规则层双层）
- 伪代码：算法 3.1（防抖状态机）, 算法 3.2（行为关系生成器）, 算法 3.3（RuleEnforcer）, 算法 3.4（增量引擎）, 算法 3.5（事件反向注入）, 算法 3.6（pipeline）
- 引文格式：`[Author, Year]`
- 术语统一：自车（ego）、Ego-Centric（模式名）、chunk（数据块）、帧（frame）

---

## 重构关键变化点

1. **结构简化**：原 9 个二级节缩减为 7 个，多层嵌套标题（X.X.X）减少约 30%；场景/行为/规则合并为 3.3 单一节，节标题层级清晰。
2. **3.1 节做深**：新增文献对比表（表 3-1）+ 属性版本化选型论证表（表 3-2）+ 研究目标定位表（表 3-4），形成完整的"动机→定义→选型→目标→约定"叙述链；新增"为何选 STKG"研究范式定位与"现有方法不足升维"两段。
3. **3.2 节做深**：3.2.1 增加"为何三层而非四层独立"的论证；3.2.3 双轨表达原则增加理论/工程双重论证；3.2.4 七条公理每条均补充"设计动机（针对的失败模式）"段；3.2.5 借鉴表后增加"借项适配性"升华。
4. **3.3 节做大**：场景/行为/规则三子节统一编号 3.3.1-3.3.3，VehicleEntity 28 字段补充设计原则；PedestrianEntity/TrafficLight/RoadElement 各增加字段深度叙述；六类提取器后增加执行流文字描述；行为层增加"承上启下"作用说明；防抖状态机增加两大工程观察论证；规则层首段增加双层结构论证；RSS 子层每个公式补充物理含义推导；交规子层增加规则编号缺失的三类来源说明 + 阈值取值理由；GNN 接口扩展为四个独立接口（节点级特征拼接、边级注意力偏置、证据链回溯、帧切片导出）。
5. **3.4 节做深**：开头增加"为何需要动态更新"动机段；属性阈值表后增加三阶段实测调优叙述；防污染机制增加"渐进式类型混淆"防御说明。
6. **3.5 节做新**：新增 3.5.3 节"Ego-Centric 全栈图谱压缩"独立段，集成 FE-1 至 FE-18 的全部优化（椭圆 ROI 生成、ego×ROI 配对、三道裁剪、coalesce_containment）；新增压缩效果对比表（表 3-23a）+ 公理兼容性论证 + 异常注入泊松过程推导 + 跨 chunk 状态一致性论证。
7. **3.6 节做实**：场景表新增车辆数/行人数/关键参数三列，新增 3.6.2 节场景-规则触发对照矩阵（表 3-26），新增 3.6.4 节"14 场景设计原理"逐个文字叙述；预期违规以代码 `scenario_library.py` 实际输出为准。
8. **3.7 节做厚**：从 800 字扩展到约 4500 字符，含五大贡献点 + 三大创新点 + 五条局限性 + 与前后章衔接 + 算法/表/图索引附录。
9. **数值对齐**：VehicleEntity 属性 28 个（旧值 18）、属性阈值按代码值（location 0.1 m / speed 0.5 m/s / heading 0.035 rad / brake 0.05）、生命周期参数 $N_{\text{stale}}=3$ / $N_{\text{forget}}=30$。
10. **格式合规**：所有表格使用 `[三线表]` 标记，公式用 `\tag{3.x}` 编号，引文用 `[Author, Year]` 格式，遵守 `FORMAT_GUIDE.md` 知网硕士论文要求。
11. **术语统一**：全章 ego/Ego/自车/ego 车辆等混用形式统一为"自车"（中文）+ "Ego-Centric"（模式名）+ "ego"作为字段代码引用。