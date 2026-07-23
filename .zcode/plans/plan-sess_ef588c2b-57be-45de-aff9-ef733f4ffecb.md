# 异常检测数据集构建方案

## 一、采集总规划
```
3 张 CARLA 地图（Town10HD, Town01, Town05）×（短场景覆盖 + 长跑采集）
= 42 短场景（3地图×14场景, 每7.5s/150帧） + 3 长跑（每地图×1次20min/24000帧）
≈ 65 min 总数据量, 约 3-5 GB JSON
```

## 二、需要创建/修改的脚本

### 新建: `scripts/long_run/build_anomaly_dataset.py`
核心数据集构建脚本，功能：
- **三种输入模式**: `--run-dir`（单长跑目录）/ `--batch-dir`（批量场景目录）/ `--all`（自动扫描）
- **处理流程**: 读 chunk*.json + anomaly_log.json + phase5_graph.json → 对齐标签 → 输出带标签的特征数据
- **输出文件**（4个）:
  - `frame_labels.csv`: 帧级标签（frame_id, is_anomaly, anomaly_type, max_severity, rule_codes, split）
  - `frame_actors.csv`: 帧×actor 运动学特征（每帧每个 actor 一行：位置/速度/朝向/控制/所在道路）
  - `event_labels.json`: 事件级异常记录（event_id, type, trigger/end_frame, target_actor, intensity）
  - `dataset_index.json`: 元数据（地图→帧数/事件数/分类分布/划分统计）

### 修改: `scripts/long_run/collect.py`
为 collect.py 增加三个多样性增强选项：
1. `--weather-cycle`：20min 内分 4 段（clear→cloudy→rain→night），依时间插值天气参数
2. `--density-ramp`：分 3 段渐变交通密度（15→25→35 veh, 6→10→15 walkers）
3. `--spawn-offset`：不同跑通过不同 seed 选择不同起始位置

### 修改: `scripts/long_run/run_long_run.sh`
传递新增的参数到 collect.py

## 三、数据划分策略
**按时间窗分层 70/15/15**（每条长跑内部按时间顺序切分）：
```
┌──── 前 70% ─────┬── 15% ──┬─ 15% ──┐
│      train       │   val   │  test   │
└──────────────────┴─────────┴─────────┘
帧号: 0          16800     20400    24000
```
- 短场景：S00-S02（基线）进 train，有异常场景进 val/test
- 划分写入 frame_labels.csv 的 split 列，无需重组织文件

## 四、标签覆盖矩阵
| 异常类型 | 短场景 | 长跑 |
|---------|--------|------|
| 基线正常 | S00,S01,S02 | 无事件帧 |
| 行人穿行 (ped_crs) | S10 | ✓泊松注入 |
| 路口不讓行 (jun_ny) | S11 | ✓泊松注入 |
| 闯红灯 | S12 | TL自然触 |
| 急刹/跟车 (sudd_brk) | S13 | ✓泊松注入 |
| 急停 (sudd_stp) | — | ✓泊松注入 |
| 强行变道 (avd_col) | S20 | ✓泊松注入 |
| 逆行 (rev_drive) | — | ✓泊松注入 |
| 障碍物 (obs_blk) | — | ✓泊松注入 |
| 三岔冲突 | S21 | 自然交互 |
| 紧急车避让 | S22 | 自然交互 |
| 夜间+行人 | S30 | 天气循环 |
| 雨+变道 | S31 | 天气循环 |
| 施工区 | S32 | 道路拓扑 |

## 五、实施步骤
1. **修改 collect.py**：加天气循环 + 密度渐变 + 起位置轮换
2. **编写 build_anomaly_dataset.py**：核心数据集构建
3. ✅ **用已有数据验证**（data/long_run/ 已有 24000f+2400f 数据）
4. **批量采集短场景**：用 batch_collect.py 跑 3地图×14场景
5. **长跑采集**：用增强版 collect.py 跑 3地图×1次
6. **跑 pipeline.py**：将 chunk 数据转为 phase5_graph.json
7. **运行 build_anomaly_dataset.py**：生成最终数据集
8. **输出统计报告**：帧数/事件数/地图覆盖/类别分布/时间跨度

## 六、改动范围摘要
| 文件 | 操作 | 改动量 |
|------|------|--------|
| `scripts/long_run/collect.py` | 修改 | +~50行（天气/密度/起位置） |
| `scripts/long_run/build_anomaly_dataset.py` | 新建 | ~350行 |
| `scripts/long_run/run_long_run.sh` | 修改 | +~10行（参数传递） |
