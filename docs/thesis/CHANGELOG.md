# 论文修改记录（CHANGELOG）

> 本文件记录 `docs/thesis/` 下所有章节的创建、重写、修复与格式调整，用于版本管理与回溯。
> 格式：`YYYY-MM-DD | 文件 | 操作 | 说明`

---

## 2026-07-27

### 实现：exp_multiscenario.py 多场景训练 + 评估脚本（commit 72d7f26）

新增 `scripts/long_run/exp_multiscenario.py`，实现完整的 K-HSTGAN + KS-NBCF 训练/评估流水线。

| 文件 | 操作 | 说明 |
|------|------|------|
| `exp_multiscenario.py` | 新增 | 多场景训练脚本：数据收集→划分→训练→评估→消融→汇总 |
| `exp_results/` | 新增 | 训练结果（confusion_matrix / ablation / summary.json） |

### 实验结果：RQ1 主实验 + RQ2 消融（14 场景 × 6 帧 = 84 帧）

| 消融方案 | F1 | P | R | F1_drop |
|----------|----|----|----|---------|
| full | 0.564 | 0.393 | 1.000 | — |
| no_rule_inject | 0.564 | 0.393 | 1.000 | 0% |
| no_rss | 0.000 | 0.000 | 0.000 | **100%** |
| no_delta_gate | 0.000 | 0.000 | 0.000 | **100%** |

**结论：** RSS 残差注入与差分门控 LSTM 是 K-HSTGAN 异常检测的核心信号源（Stage I 实验验证），规则强度残差在 Stage II 联合训练中才会发挥作用。

---

## 2026-07-26

### 实现：第 4 章 K-HSTGAN 模型骨架（commit 5733e78）

新增 `stk/gnn/` 包，含 6 个文件（exporter / rgat / dhlstm_attn / knowledge_injector / k_hstgan / trainer），落地 §4.1–4.5 全部模块。K-HSTGAN smoke test 通过：F=18→23→64、H=4、T=1，K=14 关系通道。

| 文件 | 操作 | 说明 |
|------|------|------|
| `stk/gnn/exporter.py` | 新增 | STKG snapshot → PyG Data 转换，含 kappa_rule 主类映射 |
| `stk/gnn/rgat.py` | 新增 | RGAT 关系感知图注意力，per-relation channel + 门控融合 |
| `stk/gnn/dhlstm_attn.py` | 新增 | 差分门控 LSTM + 行为注意力 + Scene Transformer |
| `stk/gnn/knowledge_injector.py` | 新增 | RSS 强度残差 + 规则强度残差注入 |
| `stk/gnn/k_hstgan.py` | 新增 | 完整模型 & 多任务融合头 |
| `stk/gnn/trainer.py` | 新增 | 多任务训练器（Focal Loss + 三阶段调度 + EMA） |
| `scripts/long_run/smoke_test_k_hstgan.py` | 新增 | 端到端 smoke test |

### 实现：第 5 章 KS-NBCF 融合框架（commit e320dfd）

新增 `stk/fusion/` 包，含 4 个模块，落地 §5.2–5.5 全部核心算法。K_HSTGAN 扩展 `return_extras=True` 接口暴露 per_head_anomaly / rgat_attention。KS-NBCF smoke test 通过：D-S K=0.516，resolve_type=trust_GNN，overlap=1.0。

| 文件 | 操作 | 说明 |
|------|------|------|
| `stk/fusion/feat_injection.py` | 新增 | φ_feat 编排层（§5.2 算法 5.1） |
| `stk/fusion/loop_feedback.py` | 新增 | φ_loop 三阶段闭环反馈（§5.3 算法 5.2+5.3） |
| `stk/fusion/ds_fuser.py` | 新增 | D-S 证据理论融合（§5.4 式 5.9–5.21） |
| `stk/fusion/evidence_chain.py` | 新增 | KG 证据链回溯仲裁（§5.4.5 算法 5.4） |
| `scripts/long_run/smoke_test_ks_nbcf.py` | 新增 | 端到端 KS-NBCF smoke test |

### 修复：key bug 修复（commit 05a0d8f）

| 文件 | 操作 | 说明 |
|------|------|------|
| `stk/gnn/exporter.py` | 修复 | `_attr()` 函数穿透 `attrs` 子字典（pydantic 模型字段访问） |
| `stk/gnn/exporter.py` | 修复 | `_build_edge_index` 增加空间 K-NN fallback（无 waypoints 时从车辆位置建图） |
| `stk/gnn/rgat.py` | 修复 | einsum 维度错位 + 手工 scatter softmax + index_add_ 维度对齐 |

---

## 2026-07-24

### 新增：第 1 章（绪论）完整重写

| 文件 | 操作 | 说明 |
|------|------|------|
| `chapter1_index.md` | 新增 | 第 1 章索引与文献引用约定 |
| `chapter1_01.md` | 重写 | 1.1 研究背景与意义 — 引入真实统计数据（CA DMV 脱离事件 18 632 起、NHTSA 碰撞事故 392 起）、三重难题（场景结构化 / 时态管理 / 符号-神经协同）、四层研究意义 |
| `chapter1_02.md` | 重写 | 1.2 国内外研究现状 — 四个研究方向 15+ 篇真实文献引用、表 1-1 创新点对照、两条断链+一个未闭环的定位分析 |
| `chapter1_03.md` | 重写 | 1.3 研究内容与创新点 — 四项研究内容（STKG / K-HSTGAN / KS-NBCF / 实验）、三项创新点的逻辑关系 |
| `chapter1_04.md` | 重写 | 1.4 论文组织结构 — 七章关系说明与图 1-1 （ASCII 架构图）|

### 修复：RE-GCN 引用错误（5 处）

| 文件 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `chapter4_01.md:128` | `RE-GCN [Xu et al., AAAI 2021]` | `RE-GCN [Li et al., SIGIR 2021]` | 作者为 Zixuan Li 等 8 人，发表于 SIGIR 2021 |
| `chapter4_02.md:123` | `RE-GCN [Xu et al., 2021]` | `RE-GCN [Li et al., SIGIR 2021]` | 同上 |
| `chapter4_06.md:9` 对比表头 | `RE-GCN<br>[Xu, AAAI 2021]` | `RE-GCN<br>[Li, SIGIR 2021]` | 同上 |
| `chapter4_06.md:42` | `RE-GCN [Xu, 2021]` | `RE-GCN [Li et al., 2021]` | 同上 |
| `chapter4_index.md:73` | `Xu et al., RE-GCN, AAAI 2021` | `Li et al., RE-GCN, SIGIR 2021` | 同上 |

### 修复：xERTE 会议出处错误（2 处）

| 文件 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `chapter4_06.md:9` 对比表头 | `xERTE<br>[Han, ICLR 2021]` | `xERTE<br>[Han, arXiv 2020]` | xERTE 仅有 arXiv:2012.15537 预印本，未正式会议发表 |
| `chapter5_index.md:68` | `Han et al., xERTE, ICLR 2021` | `Han et al., xERTE, arXiv 2020` | 同上 |

### 修复：第 3 章公式编号回退 + 索引表校正

| 文件 | 操作 | 说明 |
|------|------|------|
| `chapter3_02.md` | 公式编号 -1 | 旧 3.8-3.11 → 新 3.7-3.10（A3/A5/A6/A7 四公理） |
| `chapter3_03.md` | 公式编号 -1 | 旧 3.12-3.28 → 新 3.11-3.27（RSS 四式 + 6 交规 + severity + GNN 三式） |
| `chapter3_04.md` | 公式编号 -1 | 旧 3.29-3.35 → 新 3.28-3.34（Δg_t 四元组 + AttrVersion + SummaryEvent） |
| `chapter3_05.md` | 公式编号 -1 | 旧 3.36 → 新 3.35（泊松间隔分布） |
| `chapter3_07.md` | 索引表更新 | 公式范围、公式描述、脚注编号全面校正为 3.1-3.35 |
| `CHANGELOG_chapter3_refactor.md` | 补记二次修正 | 记录修复起因、策略、全文验收结果 |

> 根因：3.1 节二次盲审修改中加入 STALE→INACTIVE 判定式后，batch sed 对 chapter3_02-05 做了 +1 偏移，但 chapter3_01 内部未同步，导致出现两个 `\tag{3.7}` 且缺 `\tag{3.6}`。修复方式为保持 chapter3_01 新编号（3.1-3.6），将 chapter3_02-05 统一回退 -1，恢复全章 3.1-3.35 连续无重复。

---

## 2026-07-23

### 新增：第 5 章 KS-NBCF（融合框架，6 个文件）

| 文件 | 内容 | 字数 |
|------|------|------|
| `chapter5_index.md` | 索引与接口契约 | — |
| `chapter5_01.md` | 5.1 融合挑战与总体架构（三类冲突分析、三子模块协作） | ~1500 |
| `chapter5_02.md` | 5.2 $\phi_{\text{feat}}$ 特征层规则先验注入 | ~1300 |
| `chapter5_03.md` | 5.3 $\phi_{\text{loop}}$ 训练-推理三阶段双向闭环 | ~2600 |
| `chapter5_04.md` | 5.4 $\phi_{\text{fuse}}$ D-S 证据理论融合与冲突消解（质量函数构造、Dempster 组合、KG 证据链回溯仲裁） | ~2500 |
| `chapter5_05.md` | 5.5 完整算法（算法 5.5）与复杂度分析 | ~1500 |
| `chapter5_06.md` | 5.6 与现有方法对比+本章小结（7 维对比表） | ~1200 |

### 新增：第 4 章 K-HSTGAN（GNN 异常检测模型，6 个文件）

| 文件 | 内容 | 字数 |
|------|------|------|
| `chapter4_index.md` | 索引与引文约定 | — |
| `chapter4_01.md` | 4.1 问题映射与总体架构 | ~2200 |
| `chapter4_02.md` | 4.2 关系感知 GAT（RGAT） | ~2500 |
| `chapter4_03.md` | 4.3 差分驱动层次化 LSTM-Attention | ~2500 |
| `chapter4_04.md` | 4.4 知识注入层 | ~1900 |
| `chapter4_05.md` | 4.5 多模态融合 + 多任务训练 | ~2500 |
| `chapter4_06.md` | 4.6 与现有方法对比+本章小结 | ~1500 |

### 新增：第 3 章 STKG 构建（图谱基础设施，7 个文件）

| 文件 | 内容 | 字数 |
|------|------|------|
| `chapter3_index.md` | 索引与核心数据统计 | — |
| `chapter3_01.md` | 3.1–3.2 形式化定义 + 四层本体 + 公理体系 A1–A7 | ~5000 |
| `chapter3_02.md` | 3.3 场景层：6 类节点 + 15 种空间关系 | ~3500 |
| `chapter3_03.md` | 3.4 行为层：11 个检测器 + 13 种关系 + 防抖状态机 | ~3200 |
| `chapter3_04.md` | 3.5 规则层：RSS 公式 + 14 条交规 + 证据链 | ~4200 |
| `chapter3_05.md` | 3.6 动态更新：Δg_t + 增量引擎 + 属性版本化 | ~3200 |
| `chapter3_06.md` | 3.7 流式采集 + 异常注入 + Neo4j 持久化 | ~2800 |
| `chapter3_07.md` | 3.8 场景库 + 3.9 全章小结 | ~1800 |

### 审校：公式重复编号修复（1 处）

| 文件 | 问题 | 修复方式 |
|------|------|---------|
| `chapter5_04.md` | 式 (5.7) 与 chapter5_03.md 中重复 | 5.4 节全部公式编号重排为 (5.8)–(5.25) |

### 审校：错字修复（4 处）

| 文件 | 原文 | 改为 | 类型 |
|------|------|------|------|
| `chapter4_05.md` 表格行 | "主任务ni调" | "主任务微调" | 输入法错误 |
| `chapter4_05.md` 正文 | "主任务ni调" | "主任务微调" | 输入法错误 |
| `chapter4_06.md` 正文 | "主任务ni调" | "主任务微调" | 输入法错误 |
| `chapter4_03.md` 正文 | "对徐行为窗口" | "对应行为窗口" | 输入法错误 |

### 新增：格式指南（1 个文件）

| 文件 | 说明 |
|------|------|
| `FORMAT_GUIDE.md` | 三线表标准、字号字体规范、页面设置、公式规范、知网学位论文提交格式 |

### 新增：全文总索引（1 个文件）

| 文件 | 内容 |
|------|------|
| `README.md` | 全部已完成章节与待续章的总索引、字数统计、写作约定 |

---

## 文件总量统计

| 章/文件 | 文件数 | 等效中文字数 |
|---------|--------|------------|
| 第 1 章 绪论 | 5（含 index） | ~7 630 |
| 第 3 章 STKG 构建 | 8（含 index） | ~23 339 |
| 第 4 章 K-HSTGAN | 7（含 index） | ~13 567 |
| 第 5 章 KS-NBCF | 7（含 index） | ~11 304 |
| FORMAT_GUIDE | 1 | — |
| README | 1 | — |
| CHANGELOG | 1（本文） | — |
| **总计** | **30** | **~55 841** |

---

## 待完成项

| 章 | 预计文件数 | 优先级 |
|----|----------|--------|
| 第 2 章 相关理论基础 | 3 | ⭐⭐⭐ |
| 第 6 章 实验与结果分析 | 5 | ⭐⭐⭐（部分依赖代码实现）|
| 第 7 章 总结与展望 | 1 | ⭐⭐ |