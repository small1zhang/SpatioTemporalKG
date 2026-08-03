## 一、整体画布规格（适合直接喂给绘图 AI）

```
画布尺寸：16:9 横向，1920×1080 像素（或矢量 SVG/PDF）
分辨率：300 dpi（印刷友好）
配色基调：深蓝主调（70%）+ 紫罗兰辅助（20%）+ 暖橙强调（10%）
背景：冰白浅底 #F4F8FB，浅蓝渐变光晕从中心向四周淡淡弥散
边框：无外框线，仅靠区块间留白与背景色分区
点缀技法：节点带轻微外发光（gaussian blur shadow），玻璃质感（80% 不透明填色 + 高光）
字体层级：标题 32px 粗体白字 / 区块副题 22px 粗体白字 / 节点 14-16px 黑字 / 标签 12px 斜体深蓝
连线规范：
   ── 实线深蓝（主数据流），箭头三角，2pt 粗
   ┄┄ 紫色虚线（反馈/迭代回路），1.5pt 粗
   ━━ 暖橙粗箭头（决策路径/最终输出），3pt 粗
```

---

## 二、主图三区块横向布局（左 → 右管道）

```
┌────────────────┐      ┌──────────────────────────┐      ┌────────────────┐
│  ① 知识图谱    │  ──> │  ② 算法框架（双子块）      │  ──> │  ③ 实验验证    │
│  构建（STKG）  │      │  K-HSTGAN ⟷ KS-NBCF       │      │  与结果分析    │
│  占 25% 宽度   │      │  占 50% 宽度               │      │  占 25% 宽度   │
└────────────────┘      └──────────────────────────┘      └────────────────┘
        ↑                          ↑                              ↑
     输入数据              模型推理 + 融合判定                 输出结论
   (CARLA 仿真)            (三大创新点主战场)              (RQ1-RQ5 五组实验)
```

**顶部主标题**（横跨整个画布顶部）：

> **面向自动驾驶安全验证的时空动态知识图谱构建与图神经网络异常检测方法研究**
> *— 论文整体框架 —*

样式：深蓝底色长条 #0B1F3A + 白色粗体 32px 字，居中，宽 100%。

**三大创新点编号徽章**：在三个区块顶部各放一枚圆形徽章（直径 60px）：
- ① 金色填充 + 黑字
- ② 蓝色填充 + 白字
- ③ 紫色填充 + 白字
徽章下方一行小字：`Innovation #1 / Innovation #2 / Innovation #3`

---


## 三、左区块 — 创新点①「STKG 四层本体构建」子图

```
位置：画布左 25% 宽度
区块标题栏：深蓝长条 #0B1F3A + 白字"① STKG 时空知识图谱构建"
副标题：浅蓝底 #74B9FF 半透明 + 深蓝字"四层本体 O = (E, A, R, T, P) · 节点生命周期 · 增量引擎"
```

### 3.1 子图内部结构（自上而下分层）

```
┌────────────────────────────────────────────────────┐
│ 顶部：CARLA 数据输入（橙色小框，暖色强调）            │
└──────────────────┬─────────────────────────────────┘
                   ↓ (实线箭头)
┌────────────────────────────────────────────────────┐
│ 中部：四层叠层（自上而下）                            │
│ ─────────────────────────────────────────────────  │
│ [深蓝大块边界框，圆角 8px，对角线深蓝渐变背景]          │
│                                                   │
│   ┌─────────────────────────────────┐             │
│   │ ① 场景层  6 类节点 · 15 种关系   │ ← 天蓝105% │
│   └─────────────────────────────────┘             │
│                  ↓                                 │
│   ┌─────────────────────────────────┐             │
│   │ ② 行为层  11 检测 · 13 行为关系  │ ← 天蓝90% │
│   └─────────────────────────────────┘             │
│                  ↓                                 │
│   ┌─────────────────────────────────┐             │
│   │ ③ 规则层  RSS + 14 交规 + 4 扩充 │ ← 天蓝75% │
│   └─────────────────────────────────┘             │
│                  ↓                                 │
│   ┌─────────────────────────────────┐             │
│   │ ④ 动态层  增量引擎 · 版本化      │ ← 天蓝60% │
│   └─────────────────────────────────┘             │
└────────────────────────────────────────────────────┘
                   ↓ (实线箭头)
┌────────────────────────────────────────────────────┐
│ 底部：双输出                                       │
│   ┌────────────┐    ┌────────────┐                │
│   │ Neo4j 持久化│    │ PyG 张量化  │               │
│   │ (橙色小框) │───>│ (橙色小框)  │               │
│   └────────────┘    └────────────┘                │
│                          ↓                         │
│                  → 喂入算法框架中区块               │
└────────────────────────────────────────────────────┘
```

### 3.2 横向机制侧柱（位于四层右侧）

四层本体右边贴一条**紫色窄柱**（紫罗兰 #5A4FCF，宽 50px），代表"横向机制"：
- *manifestsAs*（场景 → 行为桥接）
- *violates*（行为 → 规则桥接）
- *definedBy*（规则定义回指）
- *supportedByEvidence*（证据链回溯）

每个标签是浅色小药丸（pill 形状），紫色虚线箭头从场景层指向规则层（跨三层桥接）。

### 3.3 节点内文字密度（三段档）

| 模块节点 | 节点内显示文字（一行 + 一行副标题） |
|---------|-------------------------------------|
| 场景层 | **场景层 Scene Layer** / 6 类节点 · 15 种空间关系 |
| 行为层 | **行为层 Behavior Layer** / 11 检测器 · 13 行为关系 |
| 规则层 | **规则层 Rule Layer** / RSS + 14 交规 + 4 扩充 |
| 动态层 | **动态层 Dynamic Layer** / Δg_t 差分图 · 增量引擎五步 · 版本化 |

### 3.4 颜色分配

- **四层叠层**：由下至上深蓝渐变（深→浅），每层用 `azure #2E86DE` → `sky #74B9FF` 递变
- **输入框**：暖橙 `emberSoft #FFB088` 边框 + 浅橙填充
- **输出框**：暖橙（同上，呼应数据流向）
- **桥接柱**：紫罗兰 `#5A4FCF`

### 3.5 底部铭牌（小块）

浅蓝药丸标签位于子图最底部：

> *基于 7 条核心公理 · 节点生命周期 · Ego-Centric 全栈压缩*

---


## 三'、动态更新算法子图 — 创新点①核心机制（§3.4）

> **重要**：动态更新算法是 STKG 区别于静态知识图谱的关键，也是 §3.4 的核心创新。
> 在主图中**作为左区块下方的展开子图**呈现，由"动态层"节点下方伸出一条虚线引出**展开框**（callout），
> 详细展示 IncrementalEngine 五步流程与 Δg_t 四元组结构。
> 类似论文正文里常见的"局部放大示意"——主体节点简洁，下方展开算法细节。

### 3'.1 子图位置与连接关系

```
左区块中部（四层本体）            左区块下方展开子图
─────────────────                ───────────────────
[场景层]                              ┌──────────────┐
   ↓                                  │ ① Frame t    │ ← 输入帧快照
[行为层]                              │ (橙色小框)  │
   ↓                                  └──────┬───────┘
[规则层]                                     ↓
   ↓                                  ┌─────┴─────┐
[动态层] ─┐                            │  Step1    │ ─┐
          │                            │  recv     │  │
          └──── 虚线引出 ─────>        │  接收/校验 │  │
                                       └─────┬─────┘  │
                                             ↓        │ 五步
                                       ┌─────┴─────┐  │ 主流程
                                       │  Step2    │  │ 自上而下
                                       │  diff     │  │
                                       │  计算差分  │  │
                                       └─────┬─────┘  │
                                             ↓        │
                                       ┌─────┴─────┐  │
                                       │  Step3    │  │
                                       │  patch    │  │
                                       │  打补丁   │  │
                                       └─────┬─────┘  │
                                             ↓        │
                                       ┌─────┴─────┐  │
                                       │  Step4    │  │
                                       │  eval     │  │
                                       │  规则评估  │  │
                                       └─────┬─────┘  │
                                             ↓        │
                                       ┌─────┴─────┐  │
                                       │  Step5    │  │
                                       │writeback │  │
                                       │  写回保存  │  │
                                       └─────┬─────┘ ─┘
                                             ↓
                                       ┌─────────────┐
                                       │ Δg_t 四元组 │ ← 输出差分图
                                       │ (橙色发光)  │
                                       └─────────────┘
```

### 3'.2 子图标题与样式

- **位置**：左区块下方，与四层本体共享左对齐，宽度约等于左区块的 110%
- **标题栏**：深蓝长条 + 白字 "IncrementalEngine 动态更新算法 (§3.4) · 算法 3.4"
- **副标题**：浅蓝半透明 + 深蓝字 "Δg_t = ⟨Δ_ℰ, Δ_𝒜, Δ_ℛ, ℰ_rule⟩ · 五步主流程 · 属性版本化"
- **整体边框**：紫色虚线圆角框（与横向机制侧柱同色），表示"算法子图，与本体四层是放大关系"

### 3'.3 子图内部结构（左→右流水线布局）

子图内部分**两个并排区域**：

#### 3'.3.1 左半：五步主流程纵向流水线（深蓝主调）

5 个步骤节点纵向堆叠（由上至下），每个节点用**天蓝玻璃质感矩形**：
- 节点宽度 100px，高度 50px
- 主标签深蓝粗体，副标签小字
- 节点间用细深蓝实线箭头连接
- 右侧标注"五步主流程"竖向标签

| 步骤节点 | 主标签 | 副标签（小字斜体） |
|---------|--------|-------------------|
| Step 1 | **recv** 接收与校验 | 数值属性拒绝字符串污染（公理 A3 防御） |
| Step 2 | **diff** 计算差分 | compute_delta_entities / attrs / relations |
| Step 3 | **patch** 打补丁 | 实体生命周期迁移 + 创建 AttrVersion |
| Step 4 | **eval** 规则评估 | RuleEnforcer.enforce() → SafetyViolation |
| Step 5 | **writeback** 写回保存 | _prev_frame ← frame · _delta_history.append |

**首帧/重置快速路径**（用紫色虚线绕过 Step 2）：
- 一条紫色虚线箭头从 Step 1 直接指向 Step 3，标注 "首帧或 reset() 后跳过 diff"
- 该虚线展示"帧跳跃检测"特征：当 `|frame_id - prev_frame_id| > 1` 触发 reset，按首帧走 added=all / removed=∅ / unchanged=∅

#### 3'.3.2 右半：Δg_t 四元组结构卡（橙色强调，输出端）

位于五步流程下方的"输出"端，用一个**橙色发光卡片**可视化 Δg_t 的四元组结构：

```
┌───────────────────────────────────────────────┐
│  Δg_t  := ⟨ Δ_ℰ,  Δ_𝒜,  Δ_ℛ,  ℰ_rule ⟩    │  ← 公式药丸（圆角，深蓝边）
│  (公式 3.28)                                   │
└───────────────────────────────────────────────┘
                  ↓
4 个分量卡片（横向 4 列，每列 1 个）
──────────────────────────────────────────────
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Δ_ℰ       │ │ Δ_𝒜       │ │ Δ_ℛ       │ │ ℰ_rule   │
│ 实体差分  │ │ 属性差分  │ │ 关系差分  │ │ 规则事件 │
│ (azure)  │ │ (azure)  │ │ (azure)  │ │ (violet) │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

每个分量卡片内部含 1-2 行关键内容：

| 卡片 | 内部小标签 |
|------|----------|
| **Δ_ℰ 实体差分** | DiffSet 三集合：added / removed / unchanged |
| **Δ_𝒜 属性差分** | (e, a, val_{t-1}, val_t)；按属性差异化阈值 ε_thresh |
| **Δ_ℛ 关系差分** | (src_id, dst_id, type, frame_id) 唯一性判定 |
| **ℰ_rule 规则事件** | SafetyViolation + ResponsibilityAssignment 列表（紫色） |

下方加一行小字药丸：
> *对应代码 dataclass：DeltaGraph.{delta_entities, delta_attrs, delta_relations, rule_events}*

### 3'.4 关键创新点视觉强调（绘图 AI 必看）

| # | 视觉元素 | 强调技法 |
|---|---------|---------|
| 1 | **Δg_t 四元组公式药丸** | 橙色外发光 + 公式 3.28 标号 |
| 2 | **5 步流程的 recv→diff→patch→eval→writeback** | 节点用逐步渐深天蓝（强调"流程性"） |
| 3 | **首帧/重置快速路径紫色虚线** | Step 1 → Step 3 直连，标注"首帧或 reset 后跳过 diff" |
| 4 | **属性版本化小药丸** | 在 Step 3 旁边引出"创建 AttrVersion"小气泡 |
| 5 | **生命周期状态箭头** | 在 Step 3 右侧引出 `CREATED → ACTIVE → STALE → FORGOTTEN` 四态流程小卡片 |
| 6 | **帧跳跃检测警示徽章** | Step 1 顶部一个小红色三角"!"图标，标注"frame_id jump → reset" |
| 7 | **五步流程左侧标注** | 竖向小字 "增量引擎 IncrementalEngine (算法 3.4)" |
| 8 | **Δ_ℰ DiffSet 三集合可视化** | 三个小子框横排：added (绿+) / removed (红×) / unchanged (蓝=) |

### 3'.5 颜色分配

- **五步节点**：天蓝 `azure #2E86DE` 主框 + 玻璃质感
- **公式药丸**：暖橙 `ember #FF7A45` 边框（突出输出）
- **四分量卡片**：实体/属性/关系用 azure 天蓝，规则事件用 violet 紫罗兰（呼应"规则层"配色）
- **首帧快速路径**：紫色虚线（与反馈回路同语义）
- **帧跳跃警示徽章**：红色三角 `#E74C3C` 小图标
- **生命周期小卡片**：4 状态用浅蓝→浅灰渐变背景

### 3'.6 与主图左区块的连接关系

- **展开连线**：从左区块中部"动态层"节点引出一条紫色虚线（点状）向下延伸至本子图标题栏**
- 视觉示意 "动态层下方有放大图"——类似论文中 "见下方算法展开示意"
- **数据流入**：橙色"Frame t 帧快照"输入框从五步流程顶部进入 Step 1
- **数据流出**：Δg_t 四元组结构卡从五步流程底部 Step 5 输出
- **跨子图输出**：Δg_t 通过一条**深蓝粗实线箭头**横跨主图，连到中区块 K-HSTGAN 子块的"输入层"的 `Δg_t` 橙色小框（与主图已有连接呼应）

### 3'.7 与右区块实验验证的连接

- 增量引擎的"压缩比"指标通过一条紫色虚线连到右区块的 **RQ2 流式性能** 节点的"内存/长时"小药丸
- 标注："Δg_t 增量压缩 82%-88%（实测）；若全量模式 100% 则 48 万帧需数百 GB"
- 该连线呼应论文 §3.4 开篇的工程论断："动态更新是 STKG 在 20Hz×24000 帧长时运行下可行的关键"

### 3'.8 子图底部铭牌

浅紫药丸标签位于本子图最底部：

> *基于差分图 Δg_t · 增量引擎五步流程 · 属性版本化 · 时间窗口聚合 · 规则事件反向注入*


## 四、中区块 — 创新点②+③「算法框架」双子块互连（高级 CNN 式多图形拼合）

```
位置：画布中 50% 宽度（核心区块）
区块标题栏：深蓝长条 + 白字"② ③ 算法框架 — 检测与融合"
副标题：浅蓝半透明 + 深蓝字"K-HSTGAN 层次化时空图注意力 ⟷ KS-NBCF 双向闭环融合"
```

> **设计哲学**：本区块采用 **CNN 论文风格的多面板拼合**（multi-panel collage），
> 类似 CVPR/ICCV 论文中展示卷积核、特征图张量、attention heatmap、FPN 金字塔、
> 多尺度特征融合的排版方式——用**几何图形拼合 + 张量形状标注 + 渐变流**替代文字描述，
> 体现"高级感"。K-HSTGAN 与 KS-NBCF 两个子块上下排列，用粗紫色双向箭头连接成 H 形。

### 4.1 上子块 — 「K-HSTGAN 模型」(创新点②)

**子块标题栏**：深蓝 #1A1B3A 半透明 + 白字"K-HSTGAN (§4)"
**子块背景**：浅蓝 70% 不透明 + 圆角 12px + 深蓝细实线边框

#### 4.1.1 整体布局（CNN 式多面板拼合）

K-HSTGAN 子块内部采用 **5 列横向面板 + 1 列输出**的拼合布局，
类似 CNN 论文中"输入 → 卷积层 → 特征图 → 分类头"的多图排版：

```
┌────────────────────────────────────────────────────────────────────────────┐
│  K-HSTGAN 面板拼合（5 列 + 输出）                                          │
│                                                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───┐ │
│  │  Panel  │  │  Panel  │  │  Panel  │  │  Panel  │  │  Panel  │  │   │ │
│  │  ①      │→ │  ②      │→ │  ③      │→ │  ④      │→ │  ⑤      │→ │⑥  │ │
│  │  Input  │  │  Spatial│  │ Temporal│  │ Knowledge│  │  Fusion │  │Out│ │
│  │  Tensor │  │  RGAT   │  │  LSTM   │  │  Inject │  │  Head   │  │   │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───┘ │
│       │             │             │             │             │            │
│  ┌────┴─────────────┴─────────────┴─────────────┴─────────────┴────┐     │
│  │                    渐变流背景（浅蓝 → 天蓝 → 深蓝）                  │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                            │
│  ⑤ → ⑥ 之间：三路辅助分支（紫色细线）→ 汇聚到青色发光圆（异常分数输出）     │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 各面板详细内容（CNN 风格）

**Panel ① — 输入张量面板（橙色边框 + 浅橙填充）**

类似 CNN 论文中展示 `Input: [B, C, H, W]` 的面板，标注张量形状：

```
┌─────────────────────────┐
│  🚗 Input Tensors       │
│  ┌───────────────────┐  │
│  │ X_t  [N × 18]     │  │
│  │ A_t  [N × N × 16] │  │
│  │ B_t  [N × N × 14] │  │
│  │ Δg_t  (四元组)     │  │
│  │ κ_rss, κ_rule     │  │
│  └───────────────────┘  │
│  5 路输入 → 横向箭头汇聚  │
└─────────────────────────┘
```

视觉：5 个小橙色张量卡横排，箭头汇聚到面板中央。

**Panel ② — 空间 RGAT 面板（天蓝渐变 + 注意力热力图暗示）**

类似 CNN 论文中展示"Attention Map / Feature Map"的面板：

```
┌─────────────────────────┐
│  🔵 Spatial RGAT (§4.2) │
│  ┌───────────────────┐  │
│  │  ┌───┐ ┌───┐     │  │
│  │  │ α │ │ α │ ... │  │  ← 注意力权重矩阵暗示（小方块网格）
│  │  ├───┤ ├───┤     │  │
│  │  │ α │ │ α │     │  │
│  │  └───┘ └───┘     │  │
│  │  15 relations × H=4 heads  │
│  └───────────────────┘  │
│  输出: h_spatial [N × F'] │
└─────────────────────────┘
```

视觉：15×4 小方块网格（暗示注意力矩阵），右上角标注输出张量形状。

**Panel ③ — 时序 LSTM-Attention 面板（天蓝渐变 + 门控符号）**

类似 CNN 论文中展示"LSTM cell / GRU gate"的面板：

```
┌─────────────────────────┐
│  🔵 Temporal Enc (§4.3) │
│  ┌───────────────────┐  │
│  │  ┌───┐   ┌───┐   │  │
│  │  │ g │──▶│ h │   │  │  ← 差分门控 g_t = σ(·) 符号
│  │  └───┘   └───┘   │  │
│  │  Δg_t ──▶ LSTM ──▶│  │
│  │  Scene Transformer │  │
│  │  (self-attn)      │  │
│  └───────────────────┘  │
│  输出: h_temporal [N × F']│
└─────────────────────────┘
```

视觉：门控符号 Δg_t → LSTM → Scene Transformer 自注意力，箭头流动。

**Panel ④ — 知识注入面板（天蓝渐变 + 残差符号）**

类似 CNN 论文中展示"Residual Connection / Skip Connection"的面板：

```
┌─────────────────────────┐
│  🔵 Knowledge Inj (§4.4)│
│  ┌───────────────────┐  │
│  │  RSS 公式残差      │  │
│  │  κ_rss = d_min - d │  │
│  │       ⊕ κ_rule    │  │  ← ⊕ 符号表示拼接/残差注入
│  │  交规 Embedding    │  │
│  │  (14 维强度)       │  │
│  │  弱监督 ŷ_rule     │  │
│  └───────────────────┘  │
│  输出: h_knowledge [N × 5+14]│
└─────────────────────────┘
```

视觉：⊕ 拼接符号 + RSS 公式残差可视化（d_min - d 箭头）。

**Panel ⑤ — 多模态融合头面板（紫色主框）**

类似 CNN 论文中展示"Multi-head Prediction"的面板：

```
┌─────────────────────────┐
│  🟣 Fusion Head (§4.5)  │
│  ┌───────────────────┐  │
│  │  ⊕ 汇聚            │  │
│  │  h = h_spatial     │  │
│  │    + h_temporal    │  │
│  │    + h_knowledge   │  │
│  │       ↓            │  │
│  │  ┌───┐ ┌───┐ ┌───┐│  │
│  │  │p_s│ │p_b│ │p_r││  │  ← 三路辅助输出（小方框）
│  │  │3  │ │7  │ │24 ││  │
│  │  └───┘ └───┘ └───┘│  │
│  │       ↓            │  │
│  │  ● p_anomaly [0,1] │  │  ← 青色发光圆（主输出）
│  └───────────────────┘  │
└─────────────────────────┘
```

视觉：⊕ 汇聚符号 + 三路辅助小方框 + 青色发光圆（异常分数）。

**Panel ⑥ — 输出面板（青色发光圆）**

```
┌─────────┐
│  ●       │
│ p_anomaly│
│  [0,1]   │
└─────────┘
```

视觉：青色发光圆 + 暖橙粗箭头从 Panel ⑤ 指向 Panel ⑥。

#### 4.1.3 子块底部铭牌

> *Focal Loss + 三阶段调度 + EMA · 41K 帧 F1 = 1.000（实证）*

### 4.2 下子块 — 「KS-NBCF 融合框架」(创新点③)

**子块标题栏**：紫色 #5A4FCF + 白字"KS-NBCF (§5)"
**子块背景**：浅紫 50% 不透明 + 圆角 12px + 紫罗兰细实线边框

#### 4.2.1 整体布局（CNN 式多面板拼合 + 证据体叠层）

KS-NBCF 子块采用 **"信源叠层 + 融合几何 + 决策流"** 三段式 CNN 风格布局：

```
┌────────────────────────────────────────────────────────────────────────────┐
│  KS-NBCF 面板拼合（三段式）                                                │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐ │
│  │  Panel   │  │  Panel   │  │  Panel   │  │     Panel                │ │
│  │  ① φ_feat│  │  ② φ_loop│  │  ③ φ_fuse│  │     ④ 冲突消解 + 输出    │ │
│  │  特征注入 │  │  双向闭环 │  │  D-S 融合 │  │     (几何拼合)           │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────────────┘ │
│       │             │             │                   │                  │
│  ┌────┴─────────────┴─────────────┴───────────────────┴──────────────┐   │
│  │                    渐变流背景（浅紫 → 紫罗兰 → 深紫）                  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ① → ② → ③ → ④ 横向箭头流 +  ② ↔ 规则引擎 紫色虚线反馈回路               │
│  ③ → ④ 决策菱形分支 → 最终输出（深蓝核心框）                               │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 各面板详细内容（CNN 风格）

**Panel ① — φ_feat 特征注入面板（紫色主框）**

类似 CNN 论文中展示"Feature Concatenation / Channel Expansion"的面板：

```
┌─────────────────────────┐
│  🟣 φ_feat (§5.2)       │
│  ┌───────────────────┐  │
│  │  h_v^(0) =        │  │
│  │  [x_v ‖ κ_rss ‖   │  │  ← 公式药丸（圆角，深蓝边）
│  │   κ_rule]         │  │
│  │                   │  │
│  │  37 维 =           │  │
│  │  18 物理 +         │  │
│  │  5 RSS 残差 +      │  │
│  │  14 交规强度       │  │
│  └───────────────────┘  │
│  输入：K-HSTGAN 输出 + STKG 规则引擎
│  输出：h_feat [N × 37]  │
└─────────────────────────┘
```

视觉：公式药丸居中 + 37 维分解条（3 个彩色小段：18/5/14）+ 上下箭头标注输入输出。

**Panel ② — φ_loop 三阶段闭环面板（紫色主框 + 反馈弧线）**

类似 CNN 论文中展示"Training Loop / Iterative Refinement"的面板：

```
┌─────────────────────────┐
│  🟣 φ_loop (§5.3)       │
│  ┌───────────────────┐  │
│  │  ┌─────┐ ┌─────┐ │  │
│  │  │ I   │→│ II  │ │  │  ← 三阶段横向排列
│  │  │弱监督│ │反馈 │ │  │
│  │  └─────┘ └─────┘ │  │
│  │       ↓     ↑    │  │
│  │  ┌────────────┐   │  │
│  │  │   III      │   │  │  ← 第三阶段（推理时模板更新）
│  │  │  模板更新   │   │  │
│  │  └────────────┘   │  │
│  │                   │  │
│  │  紫色虚线反馈弧线：  │  │
│  │  Stage II → 规则引擎 │  │  ← 置信度调整（关键创新点）
│  │  Stage II → K-HSTGAN│  │  ← GNN 反馈
│  └───────────────────┘  │
│  输出：更新后的规则置信度  │
└─────────────────────────┘
```

视觉：三阶段横向排列 + 两条紫色虚线弧形反馈弧线（分别指向规则引擎和 K-HSTGAN）。

**Panel ③ — φ_fuse D-S 证据融合面板（紫色主框 + 几何拼合）**

类似 CNN 论文中展示"Feature Fusion / Attention Fusion"的面板，用**几何拼合**体现 D-S 组合：

```
┌─────────────────────────┐
│  🟣 φ_fuse (§5.4)       │
│  ┌───────────────────┐  │
│  │  ┌───────────┐    │  │
│  │  │ m_rule    │    │  │  ← 左侧信源（来自 STKG 规则引擎）
│  │  │ (s_v)     │    │  │
│  │  └───────────┘    │  │
│  │       ⊕          │  │  ← Dempster 组合 ⊕（齿轮符号）
│  │  ┌───────────┐    │  │
│  │  │ m_GNN     │    │  │  ← 右侧信源（来自 K-HSTGAN）
│  │  │ (p_v,ε_v) │    │  │
│  │  └───────────┘    │  │
│  │       ↓            │  │
│  │  m_fused + K      │  │  ← 融合质量 + 冲突系数
│  └───────────────────┘  │
│  输出：m_fused, K, conflict?  │
└─────────────────────────┘
```

视觉：两个信源框（m_rule / m_GNN）并排 + ⊕ 齿轮符号居中 + 下方输出 m_fused + K。

**Panel ④ — 冲突消解 + 最终输出（几何拼合 + 决策分支）**

类似 CNN 论文中展示"Decision Boundary / Classification Head"的面板：

```
┌─────────────────────────┐
│  🟣 冲突消解 + 输出      │
│  ┌───────────────────┐  │
│  │   K > τ=0.3 ?     │  │  ← 菱形决策框（紫色）
│  │   ┌───────┐       │  │
│  │  是│       │否     │  │
│  │   ↓       ↓        │  │
│  │  ┌────┐  ┌────┐   │  │
│  │  │KG  │  │ 直接 │   │  │  ← 两条分支
│  │  │证据│  │ 融合 │   │  │
│  │  │链  │  │ 判定 │   │  │
│  │  └────┘  └────┘   │  │
│  │       ↓            │  │
│  │  ┌─────────────┐   │  │
│  │  │  最终判定     │   │  │  ← 深蓝核心输出框
│  │  │  ŷ, m_fused, │   │  │
│  │  │  ExpPath     │   │  │
│  │  └─────────────┘   │  │
│  └───────────────────┘  │
│  输出 → 喂入右区块 RQ3  │
└─────────────────────────┘
```

视觉：菱形决策框 + 两条分支 + 深蓝核心输出框（暖橙外发光）+ 箭头指向右区块。

#### 4.2.3 双子块之间的互连（必须强调）

两个子块通过 **2 条双向紫色虚线箭头** 互相连接，构成 H 形：

1. **K-HSTGAN 多任务输出 → KS-NBCF φ_fuse**（实线深蓝向下方）
   - 标注：`p_anomaly, ε_v, attention_weights`
2. **KS-NBCF φ_loop 反馈 → K-HSTGAN 知识注入层**（紫色虚线返回上方）
   - 标注：`置信度调整 / 阈值软化`

#### 4.2.4 中区块整体外框（轻装饰）

建议在中区块整体外套一个深蓝渐变虚线大框（细线，圆角 16px），仅在右上角标 `§4-§5 算法框架`小字标签，视觉上把两个子块圈为一个整体以区别于左右两侧。

---

## 五、右区块 — 「实验验证与结果」子图

```
位置：画布右 25% 宽度
区块标题栏：深蓝长条 + 白字"⑥ 实验与结果分析 (§6)"
副标题：浅蓝半透明 + 深蓝字"RQ1 → RQ5 五组验证链 · 41,150 帧真实数据"
```

### 5.1 子图内部结构（自上而下）

#### 5.1.1 顶部数据基础条（橙色窄长条）

| 标签 | 副文 |
|------|------|
| **数据基础设施** | 41,150 帧 · 5 地图 · 20 min · 24K 帧 |

#### 5.1.2 五个 RQ 节点纵列（深蓝核心节点，竖向堆叠）

5 个深蓝矩形节点等距竖排，节点间用细实线箭头连接成 RQ1 → RQ2 → RQ3 → RQ4 → RQ5 链：

| 节点 | 标题 | 子项小药丸（每个 RQ 旁边 1-3 个）|
|------|------|--------------------------------|
| RQ1 | **图谱构建质量评测** | 场景关系 F1 · 行为检测 F1 · 规则 DR/FAR |
| RQ2 | **流式处理性能评测** | 吞吐/延迟 · 内存/长时 · 增量 vs 全量 |
| RQ3 | **K-HSTGAN 异常检测效果** | **主结果 F1 = 1.000** · PR 曲线 · 跨 Town OOD |
| RQ4 | **消融实验** | 架构消融 · 融合消融 · 系统级 |
| RQ5 | **融合框架 + 冲突消解** | 冲突消解矩阵 · Case Study 证据链可视化 |

RQ3 节点用**暖橙高亮边框**（4pt），强调"已真实化 F1 = 1.000"。

#### 5.1.3 三个状态色块（底部并排小药丸）

| 色块 | 标签 | 列表 |
|------|------|------|
| 🟢 绿色 | **已真实化** | 表 6-13 主结果 F1=1.000 · 表 6-18 跨 Town OOD · 表 6-3 规则码分布 |
| 🟠 橙色 | **预估待跑** | 表 6-4/5/6 + 表 6-8 → 6-11（性能/内存/长时） |
| 🟣 紫色 | **需人工评审** | 表 6-16/17 可解释性人工评审 |

### 5.2 最底部输出框

午夜深蓝矩形 + 暖橙外发光：

> **→ 第 7 章 总结与展望**
> 三大创新点闭环验证 · 投稿 Engineering Applications of AI

---

## 六、跨区块连线（关键视觉元素）

### 6.1 左 → 中 主流（黑色实线，粗 2pt）

- **STKG → K-HSTGAN**：STKG 输出 `PyG Data 张量` → K-HSTGAN 输入层
- **STKG → KS-NBCF φ_feat**：STKG 规则层 → 注入 κ_rss, κ_rule
- **STKG → KS-NBCF φ_fuse**：STKG RuleEnforcer → m_rule 质量函数

### 6.2 中 → 反 馈（紫色虚线回路，关键创新点）

- **KS-NBCF φ_loop → STKG 规则层**（紫色虚线弧形大箭头回流到左区块）—— 表达"训练中置信度调整反向修正规则引擎"
- **KS-NBCF φ_loop → K-HSTGAN**（紫色虚线垂直返回）—— 表达"GNN 预测反馈调整阈值"
- 这条紫色回路**必须在视觉上明显**，因为它是双向闭环融合的核心创新点

### 6.3 中 → 右 输出

- 多条暖橙粗箭头从 KS-NBCF `ŷ, m_fused, ExpPath` 汇入右区块的 RQ3 主结果节点
- 从 K-HSTGAN `p_anomaly` 直接连线到 RQ3 的"主结果 F1=1.000"

### 6.4 右 → （跨越 → 总结）

- 一条紫色虚线大弧**从 RQ3 / RQ5 的输出**绕过画布顶部（或底部）回流到左区块 STKG 输入端
- 标注：`实验结果 → 反馈指导图谱构建（迭代闭环）`

---

## 七、整体配色对照表（绘图 AI 直接参考）

| 色彩角色 | 用途 | HEX |
|----------|------|-----|
| **deepNavy 深海军蓝** | 区块标题栏、核心输出框、外框装饰 | `#0B1F3A` |
| **midnight 午夜蓝** | 主要深色块、最终判定 | `#1A1B3A` |
| **midBlue 中蓝** | 边框、主连线 | `#1B4F8C` |
| **azure 天蓝** | 模块级主框（K-HSTGAN 三层） | `#2E86DE` |
| **sky 亮天蓝** | 子级节点（药丸/小框） | `#74B9FF` |
| **cyan30 青雾** | 浅底装饰、子项 | `#A8D8EA` |
| **iceWhite 冰白** | 画布背景底色 | `#F4F8FB` |
| **ember 暖橙** | 数据输入框、强调标题、决策箭头 | `#FF7A45` |
| **emberSoft 浅橙** | 输入小框填充 | `#FFB088` |
| **violet 紫罗兰** | KS-NBCF 子块、双向闭环、反馈虚线 | `#5A4FCF` |
| **moss 苔绿** | "已真实化"状态色块 | `#3FA17A` |

---

## 八、绘图 AI 直接提示词模板（建议复制粘贴）

```
Generate a high-quality, publication-grade framework diagram for an academic thesis
illustration. Use horizontal pipeline layout left→right with three big sections.

Layout:
  Left 25%: "① STKG Spatio-Temporal Knowledge Graph Construction (§3)"
  Center 50%: "② ③ Algorithm Framework — K-HSTGAN ⟷ KS-NBCF (§4-§5)"
  Right 25%: "⑥ Experimental Validation & Results (§6)"

Style:
  - Deep navy primary palette (#0B1F3A, #1B4F8C, #2E86DE, #74B9FF)
  - Violet secondary (#5A4FCF) for feedback / KS-NBCF fusion
  - Warm orange (#FF7A45, #FFB088) only for input/output data and innovation labels
  - Moss green (#3FA17A) accent for "verified-real" status
  - Ice white background (#F4F8FB), soft gradient glow
  - Glass morphism effect (semi-transparent + soft blur shadow) on all nodes
  - Thin deep-navy borders on major blocks, rounded corners
  - Solid dark-blue arrows for main data flow (2pt)
  - Dashed violet arrows for feedback loops (1.5pt)
  - Thick orange arrows for decision outputs (3pt)
  - Subtle outer glow on key nodes (K-HSTGAN final, KS-NBCF final, RQ3 result)
  - Top banner: thesis full title
  - Each of 3 sections has a colored circular badge: ① gold, ② blue, ③ violet

=== LEFT SUB-GRAPH: STKG ===
Top: orange "CARLA 0.9.16 / 5 maps / 20min / 24K frames" input box
Center: stacked 4-layer ontology in deep→light blue gradient:
  Scene Layer (6 node types / 15 spatial relations)
  Behavior Layer (11 detectors / 13 behavior relations)
  Rule Layer (RSS + 14 traffic rules + 4 extended)
  Dynamic Layer (Δg_t diff graph / incremental engine / versioning)
Right side of stack: violet vertical column with 4 cross-layer bridge labels:
  manifestsAs / violates / definedBy / supportedByEvidence
Bottom: twin orange outputs (Neo4j storage + PyG tensor) feeding right
Small badge at bottom: "7 axioms · node lifecycle · Ego-Centric compression"

*** BELOW STKG: Dynamic Update Algorithm callout ***
A "Dynamic Update Algorithm" expanded sub-graph is shown BELOW the STKG
4-layer stack, connected by a violet dashed callout line from the
"Dynamic Layer" node downward.

Inside the sub-graph (titled "IncrementalEngine 动态更新算法 · Algorithm 3.4"):
  Left half — 5-step vertical pipeline, top→bottom, azure glass cards:
    Step 1 "recv 接收/校验" — defends Axiom A3 (no string into numeric attrs)
      + small red triangle warning badge: "frame_id jump → reset"
    Step 2 "diff 计算差分" — compute_delta_entities / attrs / relations
    Step 3 "patch 打补丁" — entity lifecycle transition + create AttrVersion
      + side-bubble: lifecycle CREATED → ACTIVE → STALE → FORGOTTEN (4 small cards)
    Step 4 "eval 规则评估" — RuleEnforcer.enforce() → SafetyViolation list
    Step 5 "writeback 写回保存" — _prev_frame ← frame · _delta_history.append
    Each step connected by thin dark-blue arrows; "5-step main flow" vertical label on left
    First-frame bypass — PURPLE DASHED arrow from Step 1 directly to Step 3
        label: "first frame or post-reset() skip diff"

  Right half — Δg_t 4-tuple structure card (orange glowing formula pill at top):
    "Δg_t := ⟨ Δ_ℰ, Δ_𝒜, Δ_ℛ, ℰ_rule ⟩  (Eq.3.28)"
  Below the formula, 4 small component cards in a row:
    Δ_ℰ entities — DiffSet: added (green +) / removed (red ×) / unchanged (blue =)
    Δ_𝒜 attrs — (e, a, val_{t-1}, val_t); per-attr threshold ε_thresh
    Δ_ℛ relations — (src_id, dst_id, type, frame_id) uniqueness
    ℰ_rule events — SafetyViolation + ResponsibilityAssignment (violet)

Sub-graph outputs:
  Orange "Δg_t 4-tuple" card → solid deep-blue ARROW crosses to Center
      K-HSTGAN Input Layer "Δg_t" orange box (echoing main graph)
  Violet dashed line to right RQ2 "memory/longtime" pill
      label: "incremental compression 82-88% vs full 100%"

Sub-graph bottom violet pill badge:
  "diff graph Δg_t · 5-step engine · attribute versioning · time-window aggregation · rule event backward injection"

=== CENTER SUB-GRAPH: Algorithm (CNN-style multi-panel collage) ===
Upper sub-block "K-HSTGAN (§4)" — light-blue background, navy border
  Layout: 5-column horizontal panel collage + 1 output column, similar to CVPR/ICCV
    feature-map / attention-heatmap / tensor-shape style

  Panel ① Input Tensor (orange border, light orange fill):
    5 input tensors stacked as small orange cards:
      X_t [N×18], A_t [N×N×16], B_t [N×N×14], Δg_t, κ_rss/κ_rule
    Arrows converge to center

  Panel ② Spatial RGAT (azure gradient + attention matrix grid):
    15×4 small square grid暗示 attention weight matrix
    Output label: h_spatial [N × F']

  Panel ③ Temporal LSTM-Attention (azure gradient + gate symbol):
    Δg_t → LSTM → Scene Transformer self-attention flow arrows
    Gate symbol g_t = σ(·) prominently shown
    Output label: h_temporal [N × F']

  Panel ④ Knowledge Injection (azure gradient + residual ⊕ symbol):
    RSS formula residual d_min - d + rule embedding ⊕ concatenation symbol
    Output label: h_knowledge [N × 19]

  Panel ⑤ Fusion Head (violet main box):
    ⊕汇聚 symbol → 3 auxiliary small boxes (p_scene/3, p_behavior/7, p_rule/24)
    → cyan glowing circle "p_anomaly ∈ [0,1]" (main output)

  Panel ⑥ Output (cyan glowing circle):
    p_anomaly [0,1] — final anomaly score

  Thick warm orange arrow from Panel ⑤ → Panel ⑥

Lower sub-block "KS-NBCF (§5)" — light-violet background, violet border
  Layout: 4-panel horizontal collage + decision branch + deep navy output

  Panel ① φ_feat Feature Injection (purple main box):
    Formula pill: h_v^(0) = [x_v ‖ κ_rss ‖ κ_rule]
    37-dim decomposition bar (3 colored segments: 18/5/14)
    Input arrows from K-HSTGAN + STKG rule engine
    Output: h_feat [N × 37]

  Panel ② φ_loop Three-Stage Closed-Loop (purple main box):
    3 horizontal pills: Stage I / Stage II / Stage III
    Two purple dashed curved feedback arcs:
      Stage II → STKG rule engine (label: "confidence adjustment")
      Stage II → K-HSTGAN knowledge injector (label: "threshold softening")

  Panel ③ φ_fuse D-S Evidence Fusion (purple main box):
    Two source boxes side by side: m_rule (left) + m_GNN (right)
    ⊕ Dempster combination gear symbol in center
    Output: m_fused + conflict coefficient K

  Panel ④ Conflict Resolution + Final Output (geometric collage):
    Purple diamond decision: K > τ=0.3?
      Yes branch → KG evidence chain trace-back arbitration box
      No branch → direct fused decision
    Both converge → deep navy core output box with orange glow:
      "ŷ, m_fused, ExpPath"

  Inter-block connection (CRITICAL — violet dashed double arrow):
    K-HSTGAN Panel ⑤ outputs → KS-NBCF Panel ③ φ_fuse
    KS-NBCF Panel ② φ_loop feedback → K-HSTGAN Panel ④ knowledge injector

  Center sub-block bottom small sign:
    "three-stage closed-loop → D-S evidence fusion → KG evidence chain trace-back"

=== RIGHT SUB-GRAPH: Experiments ===
Top orange bar: "Data Infrastructure — 41,150 frames / 5 maps / 20min"
Five deep-navy RQ nodes stacked vertically, connected by thin arrows:
  RQ1 "Knowledge Graph Quality Evaluation" — pills: scene F1, behavior F1, rule DR/FAR
  RQ2 "Streaming Performance" — pills: throughput/latency, memory/longtime, incremental
  RQ3 "K-HSTGAN Detection Effect" — orange highlighted border — pills:
      "Main Result F1=1.000", "PR Curve", "Cross-Town OOD"
  RQ4 "Ablation Study" — pills: architecture, fusion, system level
  RQ5 "Fusion + Conflict Resolution" — pills: conflict matrix, case study
Bottom three status pills in row:
  Green "Verified-Real: Table 6-13 F1=1.000, Table 6-18 Cross-Town, Table 6-3 distribution"
  Orange "Estimated Pending: Table 6-4/5/6 + 6-8~11"
  Violet "Manual Review Needed: Table 6-16/17 Interpretability"
Final midnight box with orange glow:
  "→ Chapter 7 Conclusion and Future Work · Submit to Engineering Applications of AI"

=== Cross-block lines ===
  ① STKG PyG output → K-HSTGAN input layer (solid dark blue)
  ① STKG rule layer → KS-NBCF φ_feat injection (solid violet)
  ① STKG RuleEnforcer → KS-NBCF φ_fuse m_rule (solid violet)
  ① STKG IncrementalEngine Δg_t output → K-HSTGAN Input Layer Δg_t box (solid dark blue)
  ① STKG IncrementalEngine compression → right RQ2 "memory/longtime" pill (violet dashed)
      label: "incremental compression 82-88% vs full 100%"
  ② K-HSTGAN Panel ⑤ output → KS-NBCF Panel ③ φ_fuse (solid dark blue, downward)
  ③ KS-NBCF Panel ② φ_loop → ① STKG rule layer (DASHED VIOLET arc curving back to left)
      label: "confidence adjustment"  ← critical visual for closed-loop innovation
  ③ KS-NBCF Panel ② φ_loop → ② K-HSTGAN Panel ④ knowledge injector (dashed violet upward)
      label: "threshold softening"
  ③ KS-NBCF Panel ④ final output → RQ3 main result (THICK ORANGE ARROW)
  ② K-HSTGAN Panel ⑥ p_anomaly → RQ3 main result (thick orange arrow)
  ⑥ RQ3/RQ5 → back to ① STKG input (faint dashed violet wide arc)
      label: "experimental results feedback to guide graph construction (iterative closed loop)"

=== Canvas ===
16:9 horizontal, 1920×1080 @ 300dpi
High school/academic publication-grade aesthetic with soft glass-morphism
Information density: medium (three-tier)
Overall vibe: deep navy primary + violet accent + restrained warm orange points
Highly professional, suitable for printing in a thesis.
```

---

## 九、关键创新点视觉强调清单（绘图 AI 不要遗漏）

| # | 视觉元素 | 强调技法 |
|---|---------|---------|
| 1 | **创新点① 徽章** (STKG 子图) | 金色圆形 + 黑字"①" |
| 2 | **四层本体的渐变叠层** | 从下到上深→浅蓝渐变，显示"递进抽象" |
| 3 | **紫色跨层桥接柱** | manifestsAs/violates 等 4 个药丸 + 虚线箭头跨层 |
| 4 | **创新点② 徽章** (K-HSTGAN) | 蓝色圆形 + 白字"②" |
| 5 | **5 个橙色输入小框** | 体现 X_t/A_t/B_t/Δg_t/κ 多模态输入 |
| 6 | **Panel ② RGAT 注意力矩阵暗示** | 15×4 小方块网格（CNN 论文风格） |
| 7 | **Panel ③ 门控符号 g_t = σ(·)** | 关键创新——门控符号必须可见 |
| 8 | **Panel ④ ⊕ 残差注入符号** | RSS 公式残差 + 拼接符号 |
| 9 | **Panel ⑤ 三路辅助 + 青色发光输出圆** | 体现 p_anomaly 的核心预测输出 |
| 10 | **创新点③ 徽章** (KS-NBCF) | 紫色圆形 + 白字"③" |
| 11 | **Panel ① 公式药丸** | h_v^(0) = [x_v ‖ κ_rss ‖ κ_rule] |
| 12 | **Panel ② 三阶段闭环横排** | Stage I/II/III 横向排列 |
| 13 | **Panel ③ ⊕ 齿轮符号** | Dempster 组合的核心融合定义符 |
| 14 | **Panel ④ 菱形 K > τ 决策框** | 体现冲突消解分支逻辑 |
| 15 | **KS-NBCF → STKG 紫色虚线大弧** | **最关键创新点**——双向闭环反向反馈 |
| 16 | **KS-NBCF → K-HSTGAN 紫色虚线垂直** | GNN 反馈调整阈值的反向回路 |
| 17 | **RQ3 暖橙边框高亮 + F1=1.000 字样** | 体现"已完成且有真实实验数据" |
| 18 | **三色状态药丸** (绿/橙/紫) | 已真实化/待跑/待评审 三类实验状态 |
| 19 | **→ 第 7 章 总结与展望** | 最底部最终输出节点 |
| 20 | **整体画布浅蓝渐变光晕背景** | 营造高级科技氛围 |
| 21 | **动态更新子图展开框** | 由"动态层"节点引出紫色虚线 callout |
| 22 | **五步流水线渐深天蓝节点** | recv→diff→patch→eval→writeback |
| 23 | **首帧/重置快速路径紫色虚线** | Step 1 → Step 3 直连 |
| 24 | **Δg_t 公式药丸（暖橙发光）** | 橙色外发光强调公式 Eq.3.28 |
| 25 | **Δg_t 四分量卡片** | 4 个横排卡片：Δ_ℰ/Δ_𝒜/Δ_ℛ（azure）+ ℰ_rule（violet） |
| 26 | **DiffSet 三子集可视化** | 3 个小药丸：added (绿+) / removed (红×) / unchanged (蓝=) |
| 27 | **帧跳跃检测红色三角警示徽章** | Step 1 顶部小红三角"!"图标 |
| 28 | **K-HSTGAN CNN 式 5 面板拼合** | 5 列横向面板 + 1 输出列，类似 CVPR 论文 feature map 排版 |
| 29 | **KS-NBCF CNN 式 4 面板拼合** | 4 列横向面板 + 决策分支 + 深蓝输出，类似证据体几何拼合 |
| 30 | **⊕ 拼接/残差符号** | Panel ④ 知识注入 + Panel ③ D-S 融合 |
| 31 | **注意力矩阵暗示网格** | Panel ② 15×4 小方块暗示 RGAT 注意力权重 |
| 32 | **门控符号 g_t** | Panel ③ 差分门控可视化 |
| 33 | **Panel ⑤ 三路辅助分支汇聚** | 紫色细分支 → 青色发光圆 |
| 34 | **Panel ④ 菱形决策 + 双分支** | K > τ → KG 证据链 / K ≤ τ → 直接融合 |
| 35 | **Δg_t → K-HSTGAN Δg_t 输入框** | 跨区块实线深蓝箭头连接动态更新子图与中区块 K-HSTGAN 子块 |
| 36 | **Δg_t 压缩比 → RQ2 内存指标** | 紫色虚线连至右区块 RQ2 "内存/长时"小药丸，标注"82-88% 压缩" |
| 37 | **CARLA 数据输入橙色框** | 顶部暖橙强调 |
| 38 | **四层叠层渐变** | 由下至上深蓝→浅蓝递变 |
| 39 | **Neo4j + PyG 双输出** | 底部橙色双输出框 |
| 40 | **三大创新点编号徽章** | ①金 ②蓝 ③紫 |
| 41 | **顶部论文全标题** | 横跨整个画布顶部，深蓝底 + 白字 |
| 42 | **中区块外框虚线** | 深蓝渐变虚线大框，圆角 16px，右上角 §4-§5 标签 |
| 43 | **紫色反馈回路** | KS-NBCF ↔ STKG 规则层 / KS-NBCF ↔ K-HSTGAN 知识注入层 |
| 44 | **→ 第 7 章 总结与展望** | 最底部深蓝核心输出框 + 暖橙外发光 |

---

## 十、交付说明

将以上文字描述（推荐直接复制第八节"绘图 AI 提示词模板"）提交给专门绘图 AI 即可生成图片。
若绘图 AI 输出效果良好，可裁剪 PDF 直接插入论文：
- 论文首页/绪论部分作为整体框架图（图 1-1）
- 或拆分单独投稿期刊配图使用

如果需要我额外生成纯中文版的精简 prompt（去掉 LaTeX 公式细节）、或额外导出一份 PDF/SVG 矢量版的预览图，告诉我即可。
