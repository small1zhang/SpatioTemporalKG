# 📊 STKG 多页可视化系统

> SpatioTemporal Knowledge Graph · CARLA 0.9.16 自动驾驶知识图谱交互式可视化

---

## 一、系统概览

系统包含 **3 个交互式页面** + **1 个共享库**，基于 D3.js v7 构建，无需任何外部框架。

```
viz_output/
├── graph_view.html      ← 图谱视图（D3 力导向 + 增强交互）
├── data_table.html      ← 数据表格（节点/边排序/过滤/导出）
├── dashboard.html       ← 统计仪表盘（全宽图表 6 个子页）
├── shared.js            ← 公共数据加载（KG 全局对象）
├── shared.css           ← 公共暗色主题 + 动画样式
│
├── Town01_20min/        ← 各地图数据（含上述5个文件 + JSON）
├── Town02_20min/
├── Town04_20min/
├── Town05_20min/
└── Town10HD_20min/
```

每个 Town 目录内的数据文件：

| 文件 | 说明 |
|------|------|
| `graph_0001_0_1999.json` … `graph_0012_22000_23999.json` | 分片帧级图数据 |
| `phase5_kg_summary.json` | 分片元信息 |
| `anomaly_log.json` | 异常事件日志（4500+ 条） |
| `viz_stats.json` | 预聚合统计（14 个字段） |
| `metadata.json` | 采集元数据 |
| `frame_snapshot_kg.html` | 旧版单页（保留，未删除） |

---

## 二、启动方式

### 方式 A：SSH 端口转发（推荐）

```bash
# 1. 在你的本地电脑执行（保持会话不断）
ssh -L 8080:localhost:8080 aisecurity@172.18.42.56

# 2. 浏览器打开任一地址（见下方 URL 列表）
```

### 方式 B：在服务器上手动启动

```bash
# 进入 viz_output 目录启动 HTTP 服务
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/viz_output
python3 -m http.server 8080 --bind 0.0.0.0

# 然后通过 SSH 端口转发访问（方式 A）
# 或者在服务器桌面环境直接用浏览器打开
```

### 方式 C：后台运行

```bash
nohup python3 -m http.server 8080 \
  --directory /home/aisecurity/01_ZHB/SpatioTemporalKG/viz_output \
  > /tmp/viz_server.log 2>&1 &

# 检查是否运行
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/Town01_20min/graph_view.html
# 返回 200 表示正常
```

### 停止服务器

```bash
# 找到并 kill 进程
lsof -ti:8080 | xargs kill
```

---

## 三、访问地址

> ⚠️ 请确认服务器已在 8080 端口启动。若用 8081 端口，请将下方 `8080` 替换为 `8081`。

### 🗺️ 图谱视图（主页面）

| 地图 | 地址 |
|------|------|
| Town01 | `http://localhost:8080/Town01_20min/graph_view.html` |
| Town02 | `http://localhost:8080/Town02_20min/graph_view.html` |
| Town04 | `http://localhost:8080/Town04_20min/graph_view.html` |
| Town05 | `http://localhost:8080/Town05_20min/graph_view.html` |
| Town10HD | `http://localhost:8080/Town10HD_20min/graph_view.html` |

### 📋 数据表格

| 地图 | 地址 |
|------|------|
| Town01 | `http://localhost:8080/Town01_20min/data_table.html` |
| Town02 | `http://localhost:8080/Town02_20min/data_table.html` |
| Town04 | `http://localhost:8080/Town04_20min/data_table.html` |
| Town05 | `http://localhost:8080/Town05_20min/data_table.html` |
| Town10HD | `http://localhost:8080/Town10HD_20min/data_table.html` |

### 📊 统计仪表盘

| 地图 | 地址 |
|------|------|
| Town01 | `http://localhost:8080/Town01_20min/dashboard.html` |
| Town02 | `http://localhost:8080/Town02_20min/dashboard.html` |
| Town04 | `http://localhost:8080/Town04_20min/dashboard.html` |
| Town05 | `http://localhost:8080/Town05_20min/dashboard.html` |
| Town10HD | `http://localhost:8080/Town10HD_20min/dashboard.html` |

---

## 四、页面功能说明

### 4.1 🗺️ 图谱视图 `graph_view.html`

D3 力导向交互式图谱，显示当前帧的实体节点与关系边。

#### 顶部控制栏

| 控件 | 功能 |
|------|------|
| **分片选择** ▼ | 多选下拉框，勾选多个分片合并加载 |
| **帧滑块** | 拖动查看任意帧（24000 帧范围） |
| **⏮100 / ⏭100** | 快速前进/后退 100 帧 |
| **⚠ Next Anom** | 跳到下一个有异常事件的帧 |
| **🎯 EGO** | 仅显示与 EGO 直连的节点 |
| **2跳** | 显示 EGO 的 2 跳邻居 |
| **🌤 环境** | 包含 EnvironmentSnapshot 节点 |
| **🏷 标签** | 显示边标签 |
| **🔴 异常高亮** | ⭐ **新增**：本帧相关异常节点显示红色脉冲边框 |
| **🟠 EGO轨迹** | ⭐ **新增**：橙色虚线显示 EGO 最近 200 帧轨迹 |
| **sev ≥ X** | ⭐ **新增**：severity 滑块（0.0-1.0），隐藏低严重度违规 |
| **布局** | Force / Circular / Radial 三种布局切换 |
| **▶ 播放** | 自动播放帧序列，可设置 fps |
| **📷 截图** | 导出当前视图为 PNG |
| **↺ 重置** | 重置视图到默认状态 |

#### 左侧边栏（6 个可折叠面板）

| 面板 | 内容 |
|------|------|
| 📋 EGO 自车信息 | 车辆 ID、位置、速度、朝向、控制状态 |
| ⚠️ 异常事件 | 当前帧的 anomaly 列表（急刹/急停/逆向/横穿等） |
| 🎯 过滤 | 节点类型 + 关系类型 + severity 滑块 + 交互类型（含全选/全清） |
| 🔍 搜索 | 输入节点 ID 或 entity_id 快速定位 |
| 🌤️ 环境/天气 | 当前帧的天气、光照、路面状态 |
| 🗺️ Mini-map | 所有车辆的二维缩略图（橙色=EGO） |

#### 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` `→` | 前/后一帧 |
| `Shift + ←/→` | 前/后 100 帧 |
| `Space` | 播放 / 暂停 |
| `F` | 全屏（隐藏侧边栏） |
| `R` | 重置视图 |
| `A` | 切换异常高亮 |

---

### 4.2 📋 数据表格 `data_table.html`

支持排序、过滤、分页、导出的结构化数据查看器。

#### 顶部工具栏

| 控件 | 功能 |
|------|------|
| **节点表 / 边表** | 切换显示节点或边数据 |
| **搜索** | 按 ID / entity_id 模糊搜索 |
| **类型下拉** | 按节点类型（Vehicle / SafetyViolation 等）或边类型过滤 |
| **排序** | ID ↑↓ / first_frame ↑↓ / severity ↑↓ |
| **每页** | 25 / 50 / 100 条 |
| **📥 CSV** | 导出当前过滤结果为 CSV 文件 |

#### 节点表列

| 列名 | 说明 |
|------|------|
| ID | 节点 ID |
| 类型 | Vehicle / Pedestrian / SafetyViolation / ... |
| first_frame | 首次出现帧 |
| last_frame | 最后出现帧 |
| entity_type | 车辆类型 / 行人类型 |
| severity_max | 最大严重度（仅 SafetyViolation） |
| entity_id | 原始实体 ID |

#### 边表列

| 列名 | 说明 |
|------|------|
| src_id | 源节点 ID |
| dst_id | 目标节点 ID |
| 关系类型 | violates / following / opposite_direction / ... |
| first_frame / last_frame | 帧范围 |
| confidence | 边置信度 |

#### 交互
- 点击列头排序
- 点击行展开详情（JSON）
- 详情弹窗中有「跳转图谱」按钮 → 切回 graph_view 聚焦该节点

---

### 4.3 📊 统计仪表盘 `dashboard.html`

全宽统计图表，6 个子 Tab。

#### 子 Tab

| Tab | 内容 |
|-----|------|
| **概览** | 节点类型饼图 + 边类型柱状图 + 图谱元信息卡片（总帧数/节点/边/分片数） |
| **RSS规则** | RSS + TrafficLaw 规则分布卡片（含 severity / fired 总数）+ by_layer 饼图 + fired_bins 时序折线 |
| **异常分布** | anom_bins 堆叠柱状图（按 200 帧窗口）+ anomaly_event_log 事件表 |
| **交互类型** | InteractionEvent 类型分布 + pair_interactions 热力图（Top 30 配对） |
| **责任分析** | resp_reasons 分布 + resp_top 柱状图（Top 30 责任方） |
| **趋势** | frame_trend 双折线（节点/边密度随时间） |

---

## 五、数据字段说明

### viz_stats.json（预聚合统计）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_trend` | `[{frame, nodes, edges, anomalies}]` | 每帧的节点/边/异常数量 |
| `anom_bins` | `{bin_idx: {anomaly_type: count}}` | 按 200 帧窗口的异常密度 |
| `anomaly_event_log` | `[{frame, event_id, type, target_actor_id}]` | 每条异常事件（4500+ 条） |
| `ie_dist` | `{type: count}` | InteractionEvent 类型分布 |
| `vehicle_lifetimes` | `[{id, lifetime, is_ego, vehicle_type}]` | 车辆存活帧数 Top 30 |
| `resp_reasons` | `{reason: count}` | 责任归属原因分布 |
| `rss_dist` | `{by_code, by_layer, resp_top, fired_bins, pair_top, total_sv, total_resp}` | RSS/规则详细统计 |
| `node_type_dist` | `{type: count}` | 全局节点类型分布 |
| `edge_type_dist` | `{type: count}` | 全局边类型分布 |
| `severity_hist` | `{range: count}` | severity_max 分布直方图 |
| `shard_summary` | `{total_shards, shards: [{idx, frames, nodes, edges}]}` | 分片概览 |
| `ego_tail` | `[{frame, x, y, heading_rad, speed_ms}]` | EGO 轨迹点（120 点，每 200 帧采样） |
| `pair_interact` | `{pair_key: count}` | 交互对 Top 30 |

### anomaly_log.json

| 字段 | 说明 |
|------|------|
| `frame_id` | 事件发生帧 |
| `event_id` | 事件编号（E0001, E0002, ...） |
| `anomaly_type` | sudd_brk / sudd_stp / avd_col / jun_ny / rev_drive / ped_crs / obs_blk |
| `target_actor_id` | 被影响的 actor ID |
| `log` | 原始日志文本 |

### graph_XXXX_*.json（分片图数据）

```json
{
  "nodes": [
    {
      "id": "971",
      "type": "Vehicle",
      "first_frame": 0,
      "last_frame": 23999,
      "attrs": {
        "entity_id": "971",
        "is_ego": true,
        "location_x": 338.7,
        "location_y": 216.4,
        "heading_rad": -1.57,
        "speed_kmh": 17.1,
        ...
      }
    }
  ],
  "edges": [
    {
      "src_id": "971",
      "dst_id": "987",
      "type": "opposite_direction",
      "first_frame": 45,
      "last_frame": 1666,
      "attrs": {}
    }
  ]
}
```

---

## 六、图谱节点类型速查

| 类型 | 颜色 | 说明 |
|------|------|------|
| `Vehicle` | 🔵 蓝 | 车辆（EGO 标橙色） |
| `Pedestrian` | 🟡 黄 | 行人 |
| `TrafficLight` | 🔴 红 | 交通信号灯 |
| `RoadElement` | ⚪ 灰 | 道路/车道元素 |
| `Junction` | 🔷 青 | 路口 |
| `Maneuver` | 🟠 橙 | 驾驶动作（变道/转弯等） |
| `InteractionEvent` | 🟣 紫 | 车辆间交互 |
| `BehaviorRelation` | 🩷 粉 | 行为关系 |
| `EnvironmentSnapshot` | 🟢 绿 | 环境快照（天气/光照） |
| `SafetyViolation` | 🔴 红 | RSS/交通法规违规 |
| `ResponsibilityAssignment` | 🟠 深橙 | 责任归属 |
| `Rule` | 🟡 金 | 规则定义 |
| `ScenarioSnapshot` | 💜 淡紫 | 场景快照 |

## 图谱边类型速查

| 类型 | 颜色 | 说明 |
|------|------|------|
| `violates` | 🔴 | 违反（SV→实体） |
| `responsibleFor` | 🟠 | 责任归属（RA→实体） |
| `opposite_direction` | 🩷 | 对向行驶 |
| `following` | 🟢 | 跟车 |
| `overtaking` | 💜 | 超车 |
| `changing_lane` | 💜 | 变道 |
| `blocked_view` | 🟣 | 视线遮挡 |
| `approaching` | 🔴 | 逼近 |
| `yielding_to` | 🟢 | 让行 |
| `approaching_pedestrian` | 🔴 | 逼近行人 |
| `nearby_pedestrian` | 🟡 | 附近行人 |

---

## 七、异常类型说明

| 类型 | 中文 | 说明 |
|------|------|------|
| `sudd_brk` | 急刹 | 前车突然制动 |
| `sudd_stp` | 急停 | 前车突然停止 |
| `avd_col` | 紧急变道 | 紧急避碰变道 |
| `jun_ny` | 路口不让行 | 路口未让行 |
| `rev_drive` | 逆向行驶 | 逆向行驶 |
| `ped_crs` | 行人横穿 | 行人突然横穿 |
| `obs_blk` | 视线遮挡 | 前方障碍物遮挡 |

---

## 八、常见问题

### Q：打开页面后图表不显示 / 控制台报 CORS 错误？

**原因**：直接用 `file://` 打开 HTML，浏览器安全策略阻止了 `fetch()` 请求。
**解决**：必须通过 HTTP 服务器访问（用上面的启动方式）。

### Q：页面显示「加载失败」？

1. 检查服务器是否运行：`curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/Town01_20min/graph_view.html`
2. 返回 200 表示正常，其他码表示服务器未启动或路径错误
3. 检查 viz_stats.json 是否存在：`ls viz_output/Town01_20min/viz_stats.json`

### Q：8080 端口被占用？

```bash
# 查看占用进程
lsof -ti:8080
# 杀掉后重启
kill $(lsof -ti:8080)
python3 -m http.server 8080 --directory viz_output
```

### Q：新增了数据后如何更新统计？

```bash
# 重新生成 RSS 统计
python3 scripts/viz/build_rss_stats.py --all

# 重新生成新增字段（node_type_dist / ego_tail / severity_hist 等）
python3 scripts/viz/augment_viz_stats.py
```

### Q：如何删除旧版 `frame_snapshot_kg.html`？

旧版不影响新版功能，可按需删除：
```bash
rm viz_output/Town*/frame_snapshot_kg.html
```

---

*最后更新：2026-07-27*
