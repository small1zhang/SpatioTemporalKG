# 第 4 章 知识引导的层次化时空图注意力网络 K-HSTGAN（全文索引）

> 对应规划文档：`docs/PLAN_thesis_and_paper.md` 第 4.2 节
> 对应代码模块（待实现）：`stk/gnn/` 下的 9 个文件
> 设计依据：第 3 章构建的 STKG 提供 14 类实体类型、42 种关系类型、属性版本化与差分图作为模型输入

---

## 章节结构

| 序号 | 内容 | 文件 | 预计字数 |
|------|------|------|---------|
| 4.1 | 问题映射与 K-HSTGAN 总体架构 | `chapter4_01.md` | 3000 |
| 4.2 | 空间编码层：关系感知图注意力 GAT | `chapter4_02.md` | 3500 |
| 4.3 | 时序编码层：差分驱动的层次化 LSTM-Attention | `chapter4_03.md` | 3500 |
| 4.4 | 知识注入层：RSS 公式与交规规则的嵌入 | `chapter4_04.md` | 3000 |
| 4.5 | 多模态融合头与多任务训练策略 | `chapter4_05.md` | 3000 |
| 4.6 | 与现有方法对比与本章小结 | `chapter4_06.md` | 1500 |
| **合计** | — | — | **17500** |

---

## 核心创新关键词

- **K-HSTGAN**（Knowledge-guided Hierarchical Spatio-Temporal Graph Attention Network）
- 4 项核心组件：
  1. **关系感知图注意力**（Relation-aware GAT）——15 种场景关系作为先验注意力
  2. **差分驱动门控层次化 LSTM-Attention**（Delta-gated Hierarchical LSTM-Attention）——三层时序编码（帧级+行为级+场景级）
  3. **规则知识编码与注入**（Rule Knowledge Encoding）——RSS 公式 + 交规 Embedding + 弱监督三策略
  4. **多模态多任务融合头**（Multi-modal Multi-task Fusion Head）——场景层+行为层+规则层三任务输出

---

## 输入输出契约

**输入**：

| 输入张量 | 形状 | 来源 |
|---------|------|------|
| `X_t` | $N \times F$（$F = 18+\text{派生}$） | `VehicleEntity.attrs` 等 |
| `A_t` | $N \times N \times 16$（15 类关系 + 自环） | 场景层 15 种空间关系 |
| `B_t` | $N \times N \times 14$ | 行为层 13 种行为关系 |
| `Δg_t` | 四元组（实体/属性/关系/规则事件） | `stk/dynamic/diff.py:DeltaGraph` |
| `κ_rss` | $N \times 5$ | RSS 残差向量（5 维） |
| `κ_rule` | $N \times 14$ | 交规触发强度（14 维） |
| `env_t` | $1 \times 12$ | 环境快照特征 |

**输出**：

| 输出 | 形状 | 任务 |
|------|------|------|
| `p_anomaly` | $[0,1]$ 标量 | 主任务：异常二分类 |
| `p_scene` | $[0,1]^3$ | 辅助任务 1：场景层异常类别 |
| `p_behavior` | $[0,1]^7$ | 辅助任务 2：行为层异常类别 |
| `p_rule` | $[0,1]^{24}$ | 辅助任务 3：规则层触发类别 |
| `attention_weights` | 多层注意力 | 可解释性支撑 |

---

## 公式编号

第 4 章公式从 (4.1) 开始，按节顺序连续编号。
算法从 4.1 开始，按节顺序连续编号。
表从 4-1 开始，图从 4-1 开始。

---

## 引文约定

| 引文 | 用于 |
|------|------|
| Veličković et al., GAT, ICLR 2018 | 关系感知 GAT 的基础 |
| Li et al., RE-GCN, SIGIR 2021 | TKG 推理基线 |
| Deng & Hooi, GDN, AAAI 2021 | 异常检测基线 |
| Shalev-Shwartz et al., RSS, 2017 | 知识注入来源 |
| Lipton et al., LSTM for Anomaly, 2018 | 时序编码基线 |
| Vaswani et al., Transformer, NeurIPS 2017 | 自注意力机制 |
