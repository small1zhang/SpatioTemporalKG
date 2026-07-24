# 第 3 章 时空动态知识图谱构建（全文索引）

> 对应设计文档：`docs/v3_paragraphs.txt`（7 章完整设计）
> 对应代码：`stk/ontology/`, `stk/scenario/`, `stk/behavior/`, `stk/rules/`, `stk/dynamic/`, `stk/storage/`, `scripts/long_run/`
> 对应测试：`tests/`（25 个测试文件，覆盖全部核心模块）

---

## 章节结构

| 序号 | 内容 | 对应设计文档 | 对应代码模块 | 文件 |
|------|------|-------------|-------------|------|
| 3.1 | 问题定义与形式化 | §1.7–1.11 | `stk/ontology/` | `chapter3_01.md` |
| 3.2 | 四层本体总体设计 | §1.4–1.6 | `stk/ontology/`, `stk/ontology/axioms.py` | `chapter3_01.md` |
| 3.3 | 场景层：实体与空间关系提取 | §2.2–2.11 | `stk/scenario/` | `chapter3_02.md` |
| 3.4 | 行为层：行为检测与防抖 | §3.1–3.7 | `stk/behavior/` | `chapter3_03.md` |
| 3.5 | 规则层：RSS 与交通法规推理 | §4.7–4.18 | `stk/rules/` | `chapter3_04.md` |
| 3.6 | 动态更新机制 | §5.1–5.5 | `stk/dynamic/` | `chapter3_05.md` |
| 3.7 | 流式长时采集与存储 | — | `scripts/long_run/`, `stk/storage/` | `chapter3_06.md` |
| 3.8 | 实验设计与场景库 | §2.14 | `stk/scenario/scenario_library.py` | `chapter3_07.md` |
| 3.9 | 本章小结 | — | — | `chapter3_07.md` |

---

## 核心数据统计

| 维度 | 数值 | 来源 |
|------|------|------|
| 实体类型 | 14 类 | `stk/ontology/types.py` |
| 场景层关系 | 15 种 | `SceneRelationType` |
| 行为层关系 | 13 种 | `BehaviorRelationType` |
| 规则层关系 | 7 种 | `RuleRelationType` |
| 跨层桥接关系 | 7 种 | `CrossLayerRelationType` |
| 关系总数 | 42 种 | 4 大类之和 |
| 场景节点属性 | 6 类 × 4~18 个字段 | `stk/scenario/nodes.py` |
| 行为检测器 | 11 个 | `stk/behavior/detectors.py` |
| 行为节点 | 2 类（Maneuver + Interaction）| `stk/behavior/nodes.py` |
| 防抖关系 | 13 种，阈值 1~5 帧 | `stk/behavior/debouncer.py` |
| RSS 参数 | 7 个 | `stk/rules/rss/model.py` |
| 交规规则 | 14 条（R1-R18 含跳号）| `stk/rules/traffic/rules.py` |
| 核心公理 | 7 条（A1-A7）| `stk/ontology/axioms.py` |
| 预置场景 | 14 个（A/B/C/D 四类）| `stk/scenario/scenario_library.py` |
| 配置文件 | 6 个 | `config/*.yaml` |
| 预置图表 | 5 张 | `config/map_configs/` |
| 异常注入类型 | 7 种 | `scripts/long_run/anomaly_scheduler.py` |
| 流式分块 | 2000 帧/块，24000 帧最大 | `scripts/long_run/collect.py` |

---

## 写作约定

- 公式编号：`(3.1)`, `(3.2)` 等
- 表编号：`表 3-1`, `表 3-2` 等
- 图编号：`图 3-1`, `图 3-2` 等
- 伪代码：算法 3.1, 算法 3.2 等
- 引文格式：[Author, Year]

---

## 各文件字数规划

| 文件 | 内容 | 预计字数 |
|------|------|---------|
| `chapter3_01.md` | 形式化 + 本体总体设计（含公理体系） | ~4000 字 |
| `chapter3_02.md` | 场景层 | ~3000 字 |
| `chapter3_03.md` | 行为层 | ~3000 字 |
| `chapter3_04.md` | 规则层 | ~4000 字 |
| `chapter3_05.md` | 动态更新 | ~2500 字 |
| `chapter3_06.md` | 流式采集与存储 | ~2000 字 |
| `chapter3_07.md` | 场景库 + 本章小结 | ~1500 字 |
| **合计** | **全章** | **~20000 字** |

> 注：中文字数按实际汉字数计，不含公式、代码、表格。
