# 方案：改造 dashboard.html 支持 KG 分片 + 子图查看

## 总体策略

在原 `viz_output/dashboard.html` H1 基础上覆盖升级，保留原有的 5 个标签页框架，对知识图谱（KG）标签页做深度改造，并更新总览标签页以适配 5min 分片数据。

**核心设计**：检测当前目录是否存在 `phase5_kg_summary.json` + `graph_XXXX_*.json` 文件。若存在，进入「分片模式」；否则保留旧有 `viz_data.json` 加载逻辑（向后兼容）。

---

## 数据模型

分片模式下，页面加载所需文件：
| 文件 | 作用 |
|---|---|
| `phase5_kg_summary.json` | 分片索引、总节点/边数、engine 指标 |
| `graph_0001_0_1999.json` | 第 1 分片完整 {nodes, edges} |
| `graph_0002_2000_3999.json` | 第 2 分片 |
| ... | 其余分片（按需按用户选 loading） |

HTML 中的 JS 对象：
```js
// 分片模式
const KG_MODE = 'shard';
const KGS = {
  summary: { ... },           // 从 phase5_kg_summary.json 加载
  shards: {
    'frames 0-1999': { nodes: [...], edges: [...] },  // 按需 fetch
    'frames 2000-3999': { nodes: [...], edges: [...] },
    ...
  },
  currentShard: 'frames 0-1999',  // 默认第 1 片
};
```

---

## 改造内容

### 1. 总览标签页 (Overview) — 更新为分片摘要

**现有现状**：显示 70 个 batch 任务的汇总卡片+图表（需要 `viz_data.json`）。

**改造**：
- 检测分片模式时，卡片显示：
  - **总帧数**：6000（5min 数据）
  - **总节点数**：sum(shards.graph_nodes)
  - **总边数**：sum(shards.graph_edges)
  - **分片数**：3
  - **引擎增量数**：engine_n_deltas
- 柱状图改为「各分片的节点/边数」Plotly 图
- 移除热力图、批量对比分析等不相关图表（仅分片模式下隐藏）

### 2. 知识图谱标签页 (KG) — 核心改造

#### 2a. 分片选择器

在 controls 区域（`#kg-map` / `#kg-scenario` 下拉）所在行，改为：
```html
<label>分片 <select id="kg-shard-select"></select></label>
<label>按类型 <select id="kg-mode-select"><option value="all">全部视图</option><option value="shard">单分片</option></select></label>
```
- `kg-shard-select`：从 summary.shards 填充 `frames {start}-{end}`，加上第一个选项「全部（汇总）」
- 选择「全部（汇总）」时显示分片对比信息 + 各分片类型分布
- 选择单一分片时加载对应 JSON 并渲染 D3 力导向图

#### 2b. 时间范围滑块

在 filter sidebar 中添加帧范围过滤：
```html
<h4>⏱️ 帧范围</h4>
<div style="display:flex;gap:6px;align-items:center">
  <input type="range" id="kg-frame-from" min="0" max="1999" value="0" style="flex:1">
  <span id="kg-frame-label">0-1999</span>
</div>
```
- 初始化 min/max 为当前分片的 `frame_start` / `frame_end`
- 滑块调节 → `filterKG()` 中额外检查 `n.first_frame <= maxFrame && n.last_frame >= minFrame`

#### 2c. 子图过滤

保留原有功能：
- ✅ 节点类型复选框过滤（VISIBLE_NODE_TYPES）
- ✅ 边类型复选框过滤（VISIBLE_EDGE_TYPES）
- ✅ 搜索框（节点 ID / 标签）
- ✅ 悬停高亮 + 点击属性面板
- ✅ 子图导出 JSON
- ✅ 子图统计（可见节点/边数）

#### 2d. 视图布局调整

原有 `kg-layout` 的 3 列布局（240px 筛选器 + 中央图 + 300px 详情面板）保留。但在筛选器侧边栏顶部新增：
- 分片切换下拉（取代原有的「地图+场景」选择）
- 帧范围滑块

### 3. 数据适配层 — 重写 `loadKG()`

```js
async function loadKG() {
  const shardKey = document.getElementById('kg-shard-select').value;
  
  if (shardKey === 'ALL') {
    // 显示全部分片的汇总统计（不加载各分片图）
    renderAllShardsSummary();
    return;
  }
  
  // 从 KGS.shards 取数据（若未加载则 fetch）
  if (!KGS.shards[shardKey]) {
    const shardInfo = KGS.summary.shards.find(s => `frames ${s.frame_start}-${s.frame_end}` === shardKey);
    const resp = await fetch(`graph_${String(shardInfo.shard_idx).padStart(4,'0')}_${shardInfo.frame_start}_${shardInfo.frame_end}.json`);
    KGS.shards[shardKey] = await resp.json();
  }
  
  kgGraph = KGS.shards[shardKey];
  // ... 复用现有 filter/render 逻辑
}
```

### 4. 向后兼容

- 首次加载时检测 `phase5_kg_summary.json` 是否存在（`fetch` 带 `.catch`）
- 若不存在 + `viz_data.json` 存在 → 原 batch 模式（不改 tab 行为）
- 若两者都不存在 → 显示提示"未找到图数据文件"

### 5. 文件操作

| 操作 | 说明 |
|---|---|
| 复制分片文件到 viz_output/ | 从 `data/long_run/test_5min_v2/run_20260721_220016_6000f/phase5/` 复制 `graph_*.json` + `phase5_kg_summary.json` 到 `viz_output/` |
| 修改 `viz_output/dashboard.html` | 覆盖更新文件（保留原始样式和框架，改造 JS 部分） |

---

## 实施步骤

1. 复制分片数据到 `viz_output/` 
2. 修改 `dashboard.html` 的 JS 加载逻辑（分片模式检测 + 分片选择器 + 全部摘要视图）
3. 修改 KG 标签页的 loadKG() 函数 + 时间范围滑块
4. 更新总览标签页在分片模式下的内容
5. 验证：在浏览器打开 `viz_output/dashboard.html`（通过 http-server 启动，因为需要 fetch 本地 JSON）

---

## 预期最终效果

在浏览器中打开后：
1. **总览标签页**：看到 5min 数据——6000 帧、3 个分片、总计 13,607 节点 / 386,107 边
2. **知识图谱标签页**：
   - 下拉选择「全部（汇总）」→ 看到各分片节点/边数对比表
   - 下拉选择「frames 0-1999」→ 加载第 1 分片 D3 力导向图渲染
   - 左侧筛选器：节点类型复选框、边类型复选框、搜索框
   - 帧范围滑块：调节只显示 first_frame/last_frame 在区间内的节点
   - 悬停高亮 + 点击属性面板
   - 右侧详情面板：节点属性、关联边列表
3. **向后兼容**：如果目录里只有老的 `viz_data.json`，自动回退到原 batch 模式

准备实施了吗？
