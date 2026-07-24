# 第 5 章 知识引导的符号-神经双向闭环融合框架 KS-NBCF（全文索引）

> 对应规划文档：`docs/PLAN_thesis_and_paper.md` §3.1–3.7
> 对应代码模块（待实现）：`stk/fusion/` 下的 6 个核心文件
> 设计依据：第 3 章 STKG 证据链机制 + 第 4 章 K-HSTGAN 多任务输出

---

## 章节结构

| 序号 | 内容 | 文件 | 预计字数 |
|------|------|------|---------|
| 5.1 | 融合动机与 KS-NBCF 总体架构 | `chapter5_01.md` | 2500 |
| 5.2 | 子模块①：特征层规则先验注入 $\phi_{\text{feat}}$ | `chapter5_02.md` | 2500 |
| 5.3 | 子模块②：训练-推理三阶段双向闭环 $\phi_{\text{loop}}$ | `chapter5_03.md` | 3000 |
| 5.4 | 子模块③：D-S 证据理论融合与冲突消解 $\phi_{\text{fuse}}$ | `chapter5_04.md` | 3500 |
| 5.5 | 完整算法与复杂度分析 | `chapter5_05.md` | 2000 |
| 5.6 | 与现有融合方法对比+本章小结 | `chapter5_06.md` | 1500 |
| **合计** | — | — | **15000** |

---

## 核心创新关键词

- **KS-NBCF**（Knowledge-guided Symbolic-Neural Bi-directional Closed-loop Fusion）
- 3 项子模块：
  1. **特征层规则先验注入** $\phi_{\text{feat}}$——RSS 残差与交规强度以连续值进入 GNN 初始特征
  2. **三阶段双向闭环** $\phi_{\text{loop}}$——训练前弱监督 / 训练中反馈调整规则置信度 / 推理时规则模板动态更新
  3. **D-S 证据理论融合与冲突消解** $\phi_{\text{fuse}}$——质量函数构造 + Dempster 组合 + KG 证据链路径回溯仲裁

---

## 与第 3/4 章的接口契约

**输入**：

| 输入 | 来源 | 第 5 章用途 |
|------|------|-----------|
| `K-HSTGAN 输出` | 第 4 章 | 4 头多任务预测作为 GNN 质量函数 |
| `SafetyViolation 集合` | §3.5 RuleEnforcer | 规则引擎质量函数 + 证据链输入 |
| `STKG 证据链边` | §3.5.5 supportedByEvidence | 冲突消解路径回溯 |
| `GNN 注意力权重` | §4.2 RGAT | 子图提取用于冲突仲裁 |

**输出**：

| 输出 | 格式 | 用途 |
|------|------|------|
| `y_fused` | `[0,1]` | 融合后最终异常判断 |
| `K (conflict coefficient)` | `[0,1]` | 冲突系数，供消融实验分析 |
| `explanation_path` | Cypher 路径 | 可解释性证据链 |
| `resolve_type` | `"consistent" | "trust_GNN" | "trust_rule" | "needs_review"` | 仲裁类型 |

---

## 公式编号

第 5 章公式从 (5.1) 开始，算法从 5.1 开始。

---

## 引文约定

| 引文 | 用于 |
|------|------|
| Dempster, 1968; Shafer, 1976 | D-S 证据理论基础 |
| Shalev-Shwartz et al., RSS, 2017 | 符号规则 × 神经融合对比 |
| Deng & Hooi, GDN, AAAI 2021 | 仅 GNN 方法对比 |
| Han et al., xERTE, arXiv 2020 | 时序图可解释性对比 |