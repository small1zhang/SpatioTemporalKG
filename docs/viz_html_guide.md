# 可视化页面启动指南

本文档说明如何在本地查看项目中所有 HTML 可视化页面。

---

## 一、所有 HTML 文件清单

### 1. 5 地图 20 分钟知识图谱（主要）

| 页面 | 路径 | 说明 |
|------|------|------|
| Town01 20min | `viz_output/Town01_20min/frame_snapshot_kg.html` | 24000帧·12分片·RSS+异常分析 |
| Town02 20min | `viz_output/Town02_20min/frame_snapshot_kg.html` | 24000帧·12分片·小图(0.7x) |
| Town04 20min | `viz_output/Town04_20min/frame_snapshot_kg.html` | 24000帧·12分片·大图(1.3x) |
| Town05 20min | `viz_output/Town05_20min/frame_snapshot_kg.html` | 24000帧·12分片·中等图 |
| Town10HD 20min | `viz_output/Town10HD_20min/frame_snapshot_kg.html` | 24000帧·12分片·高清复杂路口 |

每个 `Town??_20min/` 目录包含：
```
frame_snapshot_kg.html   ← 主可视化页面
graph_0001_0_1999.json   ← 分片1 (帧0-1999)
graph_0002_2000_3999.json ← 分片2
...
graph_0012_22000_23999.json ← 分片12
phase5_kg_summary.json   ← 分片索引
anomaly_log.json         ← 异常事件日志
metadata.json            ← 采集元数据
viz_stats.json           ← 预处理统计（含RSS）
```

### 2. 根目录 Dashboard 系列

| 页面 | 路径 | 说明 |
|------|------|------|
| frame_snapshot_kg.html | `viz_output/frame_snapshot_kg.html` | 基线版图谱（无 viz_stats） |
| dashboard.html | `viz_output/dashboard.html` | 完整 Dashboard（多图联控） |
| dashboard_lite.html | `viz_output/dashboard_lite.html` | 精简版 Dashboard |
| dashboard_standalone.html | `viz_output/dashboard_standalone.html` | 单页独立 Dashboard |
| ego_kg.html | `viz_output/ego_kg.html` | 自车视角知识图谱 |

### 3. 测试数据

| 页面 | 路径 | 说明 |
|------|------|------|
| frame_snapshot_kg.html | `data/runs/test_3min_observe_v1/phases_20260723_183240_3600f/frame_snapshot_kg.html` | 3分钟测试数据 |

---

## 二、启动本地 HTTP 服务器

> ⚠️ 不能直接双击 `.html` 文件打开，浏览器会因 CORS 安全策略阻断 `fetch()` 请求，
> 所有面板都会显示"无数据"。必须通过 HTTP 服务访问。

### 方法一：Python 内置服务器（推荐）

```bash
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/viz_output
python3 -m http.server 8080 --bind 0.0.0.0
```

### 方法二：后台运行（关闭终端不中断）

```bash
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/viz_output
nohup python3 -m http.server 8080 --bind 0.0.0.0 > /tmp/viz_server.log 2>&1 &
echo "服务器已启动，PID: $!"
```

### 方法三：测试数据目录

```bash
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/data/runs/test_3min_observe_v1/phases_20260723_183240_3600f
python3 -m http.server 8082 --bind 0.0.0.0
```

### 方法四：如果 8080 端口被占用

```bash
# 查看占用端口的进程
lsof -i:8080

# 换一个端口
python3 -m http.server 8081 --bind 0.0.0.0
```

---

## 三、SSH 端口转发（远程机器 → 本地浏览器）

如果你是通过 SSH 连接到服务器（172.18.42.56），需要在**本地电脑**另开一个终端：

```bash
# 本地电脑执行（保持窗口开着）
ssh -L 8080:localhost:8080 aisecurity@172.18.42.56
```

如果用的是 8081 端口：
```bash
ssh -L 8081:localhost:8081 aisecurity@172.18.42.56
```

端口转发就绪后，本地浏览器即可访问。

---

## 四、本地浏览器访问地址

### 20分钟图谱（主页面）

```
http://localhost:8080/Town01_20min/frame_snapshot_kg.html
http://localhost:8080/Town02_20min/frame_snapshot_kg.html
http://localhost:8080/Town04_20min/frame_snapshot_kg.html
http://localhost:8080/Town05_20min/frame_snapshot_kg.html
http://localhost:8080/Town10HD_20min/frame_snapshot_kg.html
```

### Dashboard 系列

```
http://localhost:8080/dashboard.html
http://localhost:8080/dashboard_lite.html
http://localhost:8080/dashboard_standalone.html
http://localhost:8080/ego_kg.html
```

### 测试数据（需单独启动 8082 端口）

```
http://localhost:8082/frame_snapshot_kg.html
```

---

## 五、各页面功能速览

### frame_snapshot_kg.html（帧时刻快照图谱）

**右侧面板（共 15 个）：**

| 面板 | 功能 | 数据源 |
|------|------|--------|
| EGO 自车信息 | 选中节点时显示属性详情 | 帧内节点 |
| 当前帧可见节点类型 | 勾选显示/隐藏节点类型 | 帧内节点 |
| EGO 关联边 | 与 EGO 直连的边列表 | 帧内边 |
| 节点类型图例 | 颜色说明 | 全局 |
| KG 规则统计 | 违反规则分布（旧版） | SafetyViolation |
| 图谱动态趋势 | 节点/边密度折线图 | viz_stats.json |
| 异常事件时间分布 | 7类异常堆叠柱状图 | viz_stats.json |
| 行为交互类型 | InteractionEvent 分布 | viz_stats.json |
| 车辆活跃时长排行 | Top15 车辆存活帧数 | viz_stats.json |
| 责任归属原因分布 | ResponsibilityAssignment 分布 | viz_stats.json |
| **RSS 规则分布** | 按规则码/层级分类，severity统计 | viz_stats.json |
| **RSS 触发时序** | 每规则码按200帧bin的触发密度 | viz_stats.json |
| **责任方 Top10** | 违反次数最多的 actor 排行 | viz_stats.json |
| **当前帧违规** | ⚠ 实时帧联动，显示本帧触发的违规 | 帧内SafetyViolation |

**操作说明：**

| 操作 | 说明 |
|------|------|
| 选择分片 | 顶部下拉框选择帧范围 |
| 帧滑块 | 拖动查看每一帧的图谱快照 |
| 🎯EGO 中心 | 仅显示与 EGO 直连的节点 |
| 2跳 | 扩展显示 2 跳邻居 |
| 🌤️环境 | 是否包含 EnvironmentSnapshot 节点 |
| 🏷️边 | 显示/隐藏边标签 |
| 节点类型过滤 | 左侧面板勾选框 |
| 关系类型过滤 | 左侧面板勾选框 |
| 节点搜索 | 输入 ID 搜索节点 |
| ⟳重新加载 | 刷新当前分片数据 |

### dashboard.html / dashboard_lite.html

多视图联控 Dashboard，包含：
- 动态力导向图（D3.js）
- 时间轴
- 异常面板
- 统计面板

### dashboard_standalone.html

单页完整版，包含图谱+轨迹+规则所有信息的独立展示。

### ego_kg.html

自车视角知识图谱可视化。

---

## 六、服务器管理

### 启动服务

```bash
# 推荐后台运行
nohup python3 -m http.server 8080 --bind 0.0.0.0 > /tmp/viz_server.log 2>&1 &
echo "PID: $!"
```

### 查看日志

```bash
tail -f /tmp/viz_server.log
```

### 健康检查

```bash
# 检查服务是否响应
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/Town01_20min/phase5_kg_summary.json
# 应返回 200
```

### 关闭服务

```bash
# 方法1：通过 PID 关闭
kill 4100753

# 方法2：关闭所有 python http.server
pkill -f "python3 -m http.server"
```

---

## 七、常见问题

### 问题1：页面打开后所有面板显示"无数据"

**原因**：直接双击 HTML 文件打开，浏览器 CORS 策略阻断了 `fetch()` 请求。

**解决**：必须通过 HTTP 服务器访问（参见第二节）。

### 问题2：端口 8080 被占用

```bash
# 查看谁占用
lsof -i:8080

# 关闭占用进程（PID 3173474 为例）
kill 3173474

# 或使用其他端口
python3 -m http.server 8081 --bind 0.0.0.0
```

### 问题3：SSH 端口转发后仍无法访问

确认本地终端窗口保持 SSH 连接不断开，检查转发参数：
```bash
# 本地终端执行
ssh -L 8080:localhost:8080 aisecurity@172.18.42.56

# 然后在本地浏览器访问
http://localhost:8080/Town01_20min/frame_snapshot_kg.html
```

### 问题4：viz_stats.json 缺失导致 RSS 板块无数据

```bash
# 重新生成 RSS 统计
python3 scripts/viz/build_rss_stats.py --all
```

### 问题5：服务器启动后无响应

检查日志：
```bash
tail /tmp/viz_server.log
```

如果是端口冲突，换端口重启即可。

---

## 八、端口对照表（快速参考）

| 端口 | 用途 |
|------|------|
| 8080 | viz_output 主目录（5 地图 + dashboard） |
| 8081 | 备用端口 |
| 8082 | test_3min 测试数据 |
| 8888 | 备用 |

---

## 九、完整流程（从零开始）

```bash
# 步骤1：SSH 连接到服务器
ssh aisecurity@172.18.42.56

# 步骤2：启动 HTTP 服务器（后台）
cd /home/aisecurity/01_ZHB/SpatioTemporalKG/viz_output
nohup python3 -m http.server 8080 --bind 0.0.0.0 > /tmp/viz_server.log 2>&1 &
echo "服务器 PID: $!"

# 步骤3：在本地电脑另开终端，SSH 端口转发
ssh -L 8080:localhost:8080 aisecurity@172.18.42.56

# 步骤4：本地浏览器打开任意页面
# 推荐从 Town01_20min 开始：
http://localhost:8080/Town01_20min/frame_snapshot_kg.html
```
