# 论文整体框架图 — 详细文字描述（绘图 AI 用，顶会范式精简版）

> **本文件用途**：交给专门的绘图 AI 生成一张**论文方法框图**（framework/architecture figure）。
> **结构**：单条横向方法流水线 —— CARLA → STKG → K-HSTGAN → KS-NBCF → 输出。
> **范式**：借鉴 CVPR/ICCV/NeurIPS 顶会论文的架构图惯例 —— **去色、克制、低信息密度、模块封装修饰为单块**、结果不画进图。
> 该图作为论文图 1 / 第 1 章介绍性框图（Overview），不承载任何定量实验结果。

---

## 一、总体原则（顶会范式对照）

| 顶会惯例 | 本文图的做法 | 理由 |
|---------|------------|------|
| **Duotone + 灰色基调** | 主色蓝 + 灰白基底，唯一强调色用于"核心贡献模块" | NaiNet 系/CenterPoint/BEVFusion 统一风格 |
| **模块框 ≤ 10~12 个** | 全图 8 个主模块框（见下方流水线） | 超过 15 个即"地图化"、审稿人判 overloaded |
| **框内 1~3 个词** | 模块名只写词组，不长句 | 长描述放图注 |
| **张量形状只标首尾** | 只在输入起点和最终输出各标一次 | 每个中间都标 = 业余感 |
| **L→R 单向单调流** | 一条主横线，单向箭头 | 方向单调、清晰 |
| **结果不进架构图** | 全图零数字：无 F1、无 PR 曲线、无表 | 顶会架构图绝不混入定量结果 |
| **符号标准化** | ⊕ / σ / ⊗/ 损失 ℒ | 融入块/箭头，不独立悬空 |
| **训练时反馈虚线** | 训练循环用虚线 + 灰色/紫罗兰，标 "training only" | 不误导推理流 |

**结果呈现位置**（不进框架图）：第 6 章用独立结果图（PR 曲线、F1 直方图）+ 结果表呈现。

---

## 二、画布规格（建议直接喂给绘图 AI）

```
画布尺寸：16:10 或 21:9 横向（宽幅），矢量 SVG/PDF
分辨率：300 dpi（印刷友好）
配色基调：灰白基底 #F7F9FB + 一种主蓝 accent（#2E6FCO / #1F5EA8）
辅助色：紫罗兰 #6C5CE7（仅用于训练反馈虚线），暖橙 #FF7A45（**仅用于"核心贡献模块"高亮**）
背景：纯白或极浅灰，无大渐变光晕（克制）
边框：细 1px，模块浅灰填充 #EFF3F7，圆角 4px
字体层级：模块内文字 14px 深灰 #1A2233；图注（不在图中，在 caption）
连线规范：
   ── 实线（主数据流）蓝或深灰，箭头三角，2pt
   ┄┄ 虚线（训练反馈回路）紫罗兰，1.5pt，标注 "training"
  中空小圆 ⊕ / ⊗ / ⊙ / σ 符号融入箭头或模块边缘
无面板标签 a/b/c/d（本图是单一方法 Overview，非多实验）
```

---

## 三、整体流水线（8 个主模块，横向单链）

```
┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌─────────┐
│ ①      │→│ ②        │→│ ③           │→│ ④            │→│ ⑤       │
│ CARLA  │ │ STKG     │ │ K-HSTGAN   │ │ KS-NBCF     │ │ 检测     │
│ 仿真    │ │ 图谱     │ │ 时空图注意力 │ │ 双向闭环融合  │ │ 输出     │
└────────┘ └──────────┘ └────────────┘ └─────────────┘ └─────────┘
   数据源      图谱表示      神经网络           深度融合       异常判定
  (输入)      (载体)      (特征/编码)        (识别决策)      (符号+神经结果)
```

- **5 个主模块**（① CARLA → ② STKG → ③ K-HSTGAN → ④ KS-NBCF → ⑤ 检测输出）
- ③ K-HSTGAN 内部展开 4 个串联子块（§4），④ KS-NBCF 内部展开 3 个串联子块（§5）——内部展开仅在贡献模块处。
- 全图核心贡献模块 = **④ KS-NBCF**（创新点③），用暖橙轮廓/填充高亮；其余灰蓝统一。

---

## 四、模块 ① CARLA 仿真（输入，最简单）

```
┌──────────────────┐
│ CARLA 0.9.16 仿真 │   ← 深灰框，浅灰填充
│ 5 地图 · 20min    │   ← 一行小字即可，不展开
└──────────────────┘
```

- 输入：原始帧数据（actor / traffic light / weather / waypoint）
- 输出：逐帧快照序列 {frame_t}
- 无任何数字、无地图名列表。
- 样式：灰白框，一行描述。

---

## 五、模块 ② STKG 时空图谱构建（§3）

```
┌─────────────────────────────────┐
│ STKG 四层本体（O = ⟨E, A, R, T⟩）  │  ← 浅灰填充 + 蓝细边框
│                                 │
│  [ 场景层  行为层  规则层  动态层 ] │  ← 4 个短语并排
│        │      │      │     │     │
│  [ 节点 · 关系 · 实体 · 属性 ]     │  ← 4 个短语竖排
│                                 │
│  ↺ 增量更新：Δg_t 差分图（虚线小箭头）│  ← 动态更新用虚线标在内部，不展开
└─────────────────────────────────┘
```

- **内部 4 层并排**：场景层 / 行为层 / 规则层 / 动态层 —— 每层一个短语框（灰蓝小框）
- 右侧以**一个虚线小箭头**标注 `Δg_t 差分图 · 增量更新`（动态更新作为一个"概念"，不画五步流程）
- 输出：STKG 图快照序列 + Δg_t 序列 + 规则向量 κ
- 样式：浅灰底 + 蓝细边，内部 4 个同级小框。

> **说明**：动态更新算法（IncrementalEngine 五步、Δg_t 四元组）属于 §3.4 正文细节，**不在 Overview 图**中展开（顶会 Overview 只画"概念级"）。

---

## 六、模块 ③ K-HSTGAN 时空图注意力网络（§4，贡献点之二，内部展开）

```
┌──────────────────────────────────────────────────────────┐
│ K-HSTGAN（$h$ = RGAT → DH-LSTM-Attn → 知识注入 → 融合头）   │
│ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐       │
│ │ 空间编码 │→│ 时序编码 │→│ 知识注入  │→│   多任务融合头  │      │
│ │ RGAT   │ │ DH-LSTM│ │ ⊕规则先验 │ │ 场景/行为/规则 │     │
│ └────────┘ └────────┘ └──────────┘ └──────────────┘       │
│     h^s       h^t       κ_rss,κ_rule      p_scene        │
│                                              p_behavior   │
│                                              p_rule       │
└──────────────────────────────────────────────────────────┘
```

- **4 个串联子块**（左→右）：
  1. **空间编码 RGAT** —— §4.2，`h^s = GAT(X_t, A_t)`；下方小字标关系先验 γ
  2. **时序编码 DH-LSTM-Attn** —— §4.3，`h^t`；门控符号 σ 融入
  3. **知识注入** —— §4.4，⊕ κ_rss 与 κ_rule（⊕ 符号画在进入融合头的分支上）
  4. **多任务融合头** —— §4.5，输出 p_scene / p_behavior / p_rule 三个短语分支
- 张量标注：仅输入侧标 `X_t, A_t, B_t` 一次，输出侧 `p_anomaly` 一次；中间 `h^s/h^t` 用小字即可，不全标形状。
- 样式：灰蓝主调；**只有"知识注入 ⊕"子块用暖橙细边**（此为本模型相对 GNN 基线的差异化贡献）。

---

## 七、模块 ④ KS-NBCF 双向闭环融合（§5，**核心贡献模块**，唯一高亮）

```
┌────────────────────────────────────────────────────────────┐
│ KS-NBCF（创新点 ③ · 高亮模块）                             │
│ ┌─────────┐ ┌──────────┐ ┌────────────┐                   │
│ │ φ_feat  │→│ φ_loop   │→│ φ_fuse     │                   │
│ │特征注入   │ │双向闭环   │ │ D-S证据融合 │                  │
│ └─────────┘ └──────────┘ └────────────┘                   │
│     ⊕规则     ↺ 虚线↑train  ⊕ Dempster                    │
│    先验                 ⊕ K (冲突系数)                    │
│ ┌──────────────────────────────┐                          │
│ │ 冲突消解：K > τ → 证据链仲裁    │  ← 决策菱形               │
│ └──────────────────────────────┘                          │
└────────────────────────────────────────────────────────────┘
```

- **3 个串联子块 + 冲突消解**：
  1. **φ_feat 特征注入** —— §5.2，⊕ 规则先验进入初始特征（⊕ 符号画在进入 K-HSTGAN 的分支）
  2. **φ_loop 双向闭环** —— §5.3，虚线反馈回 STKG 规则层 与 K-HSTGAN，标 "training"（训练时）
  3. **φ_fuse D-S 证据融合** —— §5.4，两个信源 m_rule ⊕ m_GNN → m_fused + K
  4. **冲突消解决策** —— 菱形 `K > τ`，分支到"证据链仲裁"或"直接判定"
- 输出：`ŷ（异常判定）, m_fused, ExpPath（可解释证据链）`
- 样式：**整体暖橙高亮**（轮廓 + 轻填充），与其他灰蓝模块形成唯一对比，突出这是三大创新点中最具差异化的一项。

---

## 八、模块 ⑤ 检测输出（最终判定）

```
┌──────────────────┐
│  异常判定 ŷ        │  ← 深灰/蓝核心框
│  推理路径 ExpPath  │  ← 可解释输出
└──────────────────┘
```

- 输出：二分类异常判定 ŷ + 可解释证据链路径 ExpPath
- 无数字，无 F1，无 PR。
- 样式：深蓝/深灰强调框。

---

## 九、连接与反馈（跨模块）

### 9.1 主数据流（实线，单向 L→R）

```
CARLA ──→ STKG ──→ K-HSTGAN ──→ KS-NBCF ──→ 输出
```

标注一次输入（`actor/weather/waypoint → frame_t`），一次全图末端输出（`ŷ`），中间不标。

### 9.2 训练反馈（紫色虚线，标 "training only"）

- **KS-NBCF φ_loop → STKG 规则层**（紫色虚线弯回左侧）—— 标注 `training`，表达"训练中规则置信度调整"
- **KS-NBCF φ_loop → K-HSTGAN 知识注入**（紫色虚线垂直/弯回）—— 标注 `training`
- 这两条是"双向闭环"的唯一可视化，**必须画成虚线 + 明显 "training" 标注**，避免被误读为推理流向。
- 推理时（inference）主线纯左→右，无此环。

---

## 十、配色对照表（克制，Duotone）

| 色彩角色 | 用途 | HEX |
|----------|------|-----|
| **基底灰白** | 画布背景、模块浅填充 | `#F7F9FB` / `#EFF3F7` |
| **主蓝** | 模块边框、主数据流箭头、模块标题 | `#2E6FCO`（或 `#1F5EA8`）|
| **深灰/黑** | 模块内文字、框线 | `#1A2233` |
| **紫罗兰** | 训练反馈虚线（唯一虚线用色）| `#6C5CE7` |
| **暖橙**（**严格限量**）| **仅 KS-NBCF 核心贡献模块高亮** | `#FF7A45` |
| **中灰** | 次要分隔、弱化元素 | `#8A94A6` |

> 强调：**暖橙只在 KS-NBCF 一处**。若绘图 AI 在别处大量加橙色/彩色，请一律退回灰蓝基底 —— 这是顶会"lit-up"惯例：只有一处被点亮。

---

## 十一、绘图 AI 直接提示词模板（建议复制粘贴）

```
Generate a publication-grade method overview figure for a CVPR/ICCV-style paper.
Single horizontal pipeline, left to right, monotone flow.

PALETTE (critical, duotone-with-grays):
  - Background: near-white #F7F9FB
  - Most blocks: light gray fill #EFF3F7, thin dark-gray border, dark text #1A2233
  - ONE accent blue #2E6FCO used sparingly for borders and main flow arrows
  - ONE violet #6C5CE7 for training-phase dashed feedback loops ONLY
  - ONE warm-orange #FF7A45 to highlight ONE single block (the core contribution)
  - NO rainbow, NO polychrome, NO heavy gradients, NO 3D blocks

BOX DENSITY: exactly 5 main blocks in a row, total internal ~8-10 small boxes.
TEXT: 1-3 words per box, no sentences. Long descriptions go to the caption.
TENSOR SHAPES: annotate only the input at far-left once and the output at far-right once.
DO NOT put any quantitative result (no F1, no PR curve, no numbers, no table) in this figure.

DIRECTION: strictly left-to-right, single chain, monotone.
SYMBOLS (integrated into blocks/arrows, not floating):
  ⊕ on arrows entering fusion, σ gate inside temporal module,
  K conflict coefficient near D-S fusion. Summon only where meaningful.

The 5 modules:
  ① CARLA simulation (gray box, one line "5 maps · 20 min")
  ② STKG knowledge graph — light-gray box, blue thin border,
     inside 4 small sub-boxes side-by-side:
     Scene Layer / Behavior Layer / Rule Layer / Dynamic Layer
     one small dashed arrow inside labeled "Δg_t incremental update" (concept, not expanded)
  ③ K-HSTGAN (§4) — light-gray, expanded into 4 chained small boxes:
     Spatial RGAT → Temporal DH-LSTM-Attn (σ gate) → Knowledge Inject (⊕ κ_rss, κ_rule)
       → Multi-task Fusion Head (p_scene / p_behavior / p_rule)
     annotate input side once: "X_t, A_t, B_t"; output side "p_anomaly"
     ONLY the "Knowledge Inject ⊕" sub-block may get a thin orange border
  ④ KS-NBCF (§5) — THE ONE highlighted block (warm orange outline + light fill):
     inside 3 chained small boxes + a decision diamond:
       φ_feat Feature Inject (⊕ rule prior)
       φ_loop Bi-directional Closed-loop — two violet DASHED arcs leave here:
         one back to STKG rule layer, one back to K-HSTGAN knowledge inject
         each labeled "training"
       φ_fuse D-S Evidence Fusion — two sources m_rule ⊕ m_GNN → m_fused + K
       decision diamond: K > τ → evidence-chain arbitration / direct decision
       final output: ŷ, m_fused, ExpPath
  ⑤ Detection Output (dark navy core box): ŷ anomaly decision, ExpPath explanation

MAIN FLOW (solid blue arrows, left→right):
  CARLA → STKG → K-HSTGAN → KS-NBCF → Output

TRAINING FEEDBACK (only dashed, violet, labeled "training"):

Canvas: wide horizontal (16:10 or 21:9), vector, 300dpi.
Aesthetic: restrained, high-quality, minimal; a single lit contribution;
referee-friendly; suitable for the overview figure in an academic thesis chapter 1.
```

---

## 十二、关键视觉约定（给绘图 AI 的"不要做 / 要做"清单）

### 必须做到（No-miss list）

| # | 视觉元素 | 说明 |
|---|---------|------|
| 1 | 单条 L→R 横向链 | 方向单调，无 zig-zag |
| 2 | Duotone 配色 | 灰白基底 + 单蓝 + 单橙高亮 + 单紫虚线 |
| 3 | 全图零数字 | 无 F1 / 无 PR / 无任何实验量 |
| 4 | 模块框 ≤ 5 主框 + ≤10 内盒 | 不 overloaded |
| 5 | 框内 1-3 个词 | 短语，不是句子 |
| 6 | 张量只标首尾 | 输入 `X_t,A_t,B_t` + 输出 `p_anomaly` |
| 7 | KS-NBCF 唯一橙高亮 | 3 个虚线反馈弧 → STKG + K-HSTGAN，标 "training" |
| 8 | ⊕ 符号 | 知识注入 / D-S 融合两处 |
| 9 | σ 门控符号 | 时序模块内 |
| 10 | 决策菱形 K > τ | KS-NBCF 内 |
| 11 | STKG 内 Δg_t 虚线小箭头 | 概念级，不展开五步流程 |

### 绝对不做（Must-not）

| ✗ 不要做 | 对应顶会反模式 |
|---------|--------------|
| 把 PR 曲线 / F1 值 / 数据集统计画进图 | 结果混入架构图 |
| 彩虹多色 / 高饱和渐变 / 3D 透视块 | 配色杂乱、视觉不一致 |
| 每个张量都标形状 | 业余特征 |
| 框内写完整长句/公式推导 | 信息过载 |
| >15 个框塞满 | 地图化 |
| 训练反馈画成实线或混入推理流 | 误导 |
| 装饰性 feature map / heatmap | 非承载贡献不应出现 |

---

## 十三、结果呈现位置（不在此图内）

- **PR 曲线、F1 直方图**：第 6 章单独结果图（Fig 6-x）
- **RQ1-RQ5 全表**：第 6 章结果表（表 6-x）
- **消融、跨 Town OOD、冲突消解矩阵**：第 6 章正文表格 + 结果图
- 框架图仅承担"方法概览"，与结果图分离，符合顶会惯例。

---

## 十四、交付说明

将第十一节"绘图 AI 提示词模板"提交给专门绘图 AI 即可生成一张专业、克制的论文方法框图。
若绘图 AI 输出效果良好，可将 PNG/PDF 直接插入论文（图 1 或第 1 章 Overview）。
如需我把 prompt 导出为纯文本 .txt 方便复制，或再做一版英文图标版（去除所有中文，模块名全英文），告诉我即可。
