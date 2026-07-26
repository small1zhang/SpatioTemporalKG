# 工作记录: 5 地图 × 20 分钟场景采集 + 知识图谱构建

## 任务清单

| 序号 | 任务 | 状态 | 产出 |
|------|------|------|------|
| 1 | 5 地图各 10 分钟数据采集 | ✅ 完成 | 各 6 chunks, 总计 ~4.5 GB |
| 2 | P0/P1 代码修改 (场景合理性与异常注入) | ✅ 完成 | 3 commits (FE-20/21/22) |
| 3 | 5 地图各 20 分钟数据采集 (含丰富性) | ✅ 完成 | 各 12 chunks, 总计 ~9.4 GB |
| 4 | Town02/Town04 同步模式 SIGSEGV 修复 | ✅ 完成 | 独立 wrapper (collect_town02_04.py) + 问题记录文档 |
| 5 | pipeline.py args 未定义 bug 修复 | ✅ 完成 | 3 处修改 (函数签名/函数体/调用处) |
| 6 | 5 地图图谱构建 (Phase2→5) | ✅ 完成 | 223k nodes, 2.4M edges, 5 × phase5_graph.json |
| 7 | 5 地图可视化分片 + 增强版 HTML | ✅ 完成 | 12 shards × 5 + 11 个交互版块 |

## 1. 采集参数与策略

### 10 分钟数据集

```
命令: collect.py --total-frames 12000 --chunk-frames 2000 --vehicles 25 --walkers 12 --density 2.0 --fps 20.0 --no-spectator
```

| 地图 | 时间 | 结果 |
|------|------|------|
| Town01 | 22:55~22:57 | 6 chunks, 844 MB ✅ |
| Town02 | 09:30~09:41 | 6 chunks, 794 MB ✅ |
| Town04 | 09:41~09:44 | 6 chunks, 748 MB ✅ |
| Town05 | 09:44~09:48 | 6 chunks, 1087 MB ✅ |
| Town10HD | 09:48~09:52 | 6 chunks, 1026 MB ✅ |

### 20 分钟数据集 (带场景丰富性)

```
差异参数: --total-frames 24000 --weather-cycle --density-ramp --spawn-offset
```

| 地图 | 车辆/行人 | 丰富性 | 结果 |
|------|----------|--------|------|
| Town01 | 30v/15w | weather + density-ramp + offset=0 | 12 chunks, 1.8G, 4570 anom ✅ |
| Town02 | 25v/12w | weather + density-ramp + offset=5 | 12 chunks, 1.8G, 5250 anom ✅ |
| Town04 | 25v/8w | weather (无 density-ramp, 防过载) | 12 chunks, 1.9G, 3800 anom ✅ |
| Town05 | 30v/15w | weather + density-ramp + offset=10 | 12 chunks, 2.3G, 5290 anom ✅ |
| Town10HD | 25v/15w | weather + density-ramp + offset=0 | 12 chunks, 2.0G, 4950 anom ✅ |

小计: **60 chunks, 9.4 GB, 23860 anomaly events**

## 2. 代码修改摘要

### P0 修复 (FE-20, commit 60fe2b7)

| 问题 | 原因 | 修法 |
|------|------|------|
| smoke_test.py 行人在车道上 spawn | 从车辆 spawn_points 取位置 + 5m 扰动 | 改用 get_random_location_from_navigation() |
| yaml 缺 vehicle/pedestrian count | 5 yaml × 14 场景多数缺失 | 按场景语义 × 地图缩放比补齐 |
| walker 重试不足 | 10 次重试大地图不够 | 30 次 + under-spawn 告警 |
| 三表字段名不一致 | batch/smoke 用 vehicles/walkers，yaml 用 vehicle_count/pedestrian_count | 全部统一为 vehicle_count/pedestrian_count |

### P1 修复 (FE-21, commit 2084e78)

| 问题 | 原因 | 修法 |
|------|------|------|
| 异常结束不恢复 autopilot | tick() 不返完成事件，collect 拿不到 | tick() 改返回 (active, completed) 二元组；车辆用 set_autopilot, prop 用 destroy |
| 椭圆朝向用 spawn_point yaw | 与 bind_targets 实时 yaw 不一致 | 加 ego_yaw_deg 参数，取 ego.get_transform().rotation.yaw |
| ped_crs 空挂 | bind_targets 当车辆绑 + apply_anomaly else [skip] | walker 绑定 + AI controller go_to_location(ego) |
| obs_blk 空挂 | 从未实现 | 懒 spawn static.prop + 事件结束 destroy |

### P1 修复 (FE-22, commit 61cd67e)

| 问题 | 原因 | 修法 |
|------|------|------|
| S22 无紧急车辆 | spawn 全用 vehicle.* 随机选 | 加 --emergency-vehicles，优先选 ambulance/police/firetruck |
| bind_targets 常降级 rng.choice | 阈值过严 + 无 waypoint 硬返回 False | 三级阈值 (strict→relaxed→nearest→rng)；无 waypoint 退化用 lat 距离 |
| crosswalk 硬编码 False | build_actor_dict 不接收 map_/carla_module | 真值化为 LaneType.Any 查询 + 修 f-string bug × 2 + PYEOF 残留 |

## 3. 采集异常与修复

### 3.1 Town02/Town04 同步模式 spawn walker → SIGSEGV

**根因**: CARLA 0.9.16 在 synchronous_mode=True 下 spawn AI walker controller (attach_to=walker)
触发 UE4 FlushRoutingGraph() → UNavigationSystemV1 NULL pointer 解引用。
Town02(1505 waypoints 小路网) / Town04(16919 waypoints 大郊区) 的 NavMesh 恰好触发。

**修法**: 写独立 wrapper `scripts/long_run/collect_town02_04.py`。
- spawn walker 阶段临时切异步模式 (异步下 controller spawn 正常)
- density-ramp 补充 walker 时同理临时切异步
- 保持 collect.py 不动，Town01/05/10HD 继续用 collect.py

**详见**: `docs/incident_town02_04_sigsegv.md`

### 3.2 pipeline.py args 未定义

**根因**: `process_chunks()` 函数体内直接引用 `args` (在 main() 中定义)，
导致 NameError: name 'args' is not defined。新增 4 个 CLI 参数 (ego_id, importance_threshold, exclude_lanes, prune_edges) 时未补到函数签名。

**修法**: 3 处改动
- L86-97: 函数签名加 4 个参数
- L138-153: `args.xxx` → 参数变量
- L682-690: 调用处补传 4 个参数

## 4. 文件清单

### 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/long_run/collect_town02_04.py` | Town02/Town04 专用采集 wrapper (异步安全) |
| `docs/incident_town02_04_sigsegv.md` | SIGSEGV 根因与修复记录 |
| `docs/work_record.md` | **本文件** — 全部任务汇总 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/pipeline/smoke_test.py` | P0-1: walker 导航点 |
| `scripts/long_run/collect.py` | P0-6: 重试 30+警告; P1-1: tick() 返回 (active, completed); P1-2: emergency-vehicles; P1-1b: ego_yaw_live |
| `scripts/long_run/anomaly_scheduler.py` | P1-1: tick() 二元组; P1-3: ped_crs/obs_blk 实体化; P1-4: 三级阈值 |
| `scripts/pipeline/run_phases_1_5.py` | P0-6: walker 重试 30; P1-2: emergency-vehicles; P1-1b: 椭圆实时 yaw |
| `scripts/pipeline/batch_collect.py` | P0-2: 字段名统一; P1-2: emergency-vehicles 透传 |
| `scripts/carla/spawn_traffic.py` | P1-5: is_on_crosswalk 真值化; fix f-string bug × 2 + PYEOF |
| `map_configs/Town01~10HD.yaml` | P0-2: 补齐 vehicle_count/pedestrian_count |
| `scripts/long_run/pipeline.py` | process_chunks args bug 修复 (3 处)

### 数据产出目录

```
data/long_run/
├── Town01_10min/  (6 chunks, 844 MB)
├── Town02_10min/  (6 chunks, 794 MB)
├── Town04_10min/  (6 chunks, 748 MB)
├── Town05_10min/  (6 chunks, 1087 MB)
├── Town10HD_10min/ (6 chunks, 1026 MB)
├── Town01_20min/  (12 chunks, 1.8 GB)  ← 正在跑 kg
├── Town02_20min/  (12 chunks, 1.8 GB)  ← 等待 kg
├── Town04_20min/  (12 chunks, 1.9 GB)  ← 等待 kg
├── Town05_20min/  (12 chunks, 2.3 GB)  ← 等待 kg
└── Town10HD_20min/ (12 chunks, 2.0 GB) ← 等待 kg
```
---

## 8. 论文第 4/5 章 GNN 模型代码实现 (2026-07-26)

### 8.1 背景

论文第 3 章 STKG 图谱基础设施完成后，进入第 4 章（K-HSTGAN 异常检测模型）和第 5 章（KS-NBCF 融合框架）的代码实现阶段。目标：先搭建最小可运行骨架（skeleton-first），跑通全链路后再补全训练/实验。

---

### 8.2 第 4 章：K-HSTGAN 模型（commit 5733e78）

**新建 `stk/gnn/` 包**，包含 6 个文件：

| 文件 | 功能 | 论文章节 | 行数 |
|------|------|---------|------|
| `__init__.py` | 模块导出 | — | 15 |
| `exporter.py` | STKG snapshot → PyG Data 转换 | §4.1 输入层 | 480+ |
| `rgat.py` | 关系感知图注意力（per-relation channel + 门控融合） | §4.2 式 4.4–4.11 | 290+ |
| `dhlstm_attn.py` | 差分门控 LSTM + 行为注意力 + Scene Transformer | §4.3 式 4.13–4.23 | 260+ |
| `knowledge_injector.py` | RSS 残差注入 + 规则强度残差注入 | §4.4 式 4.24–4.28 | 100+ |
| `k_hstgan.py` | 完整模型（串联所有模块 + 多任务融合头） | §4.1–4.5 | 210+ |
| `trainer.py` | 多任务训练器（Focal Loss + 三阶段调度 + EMA） | §4.5 式 4.40–4.49 | 250+ |

**关键设计决策：**
- 输入维度 F=18（基础）→ 23（+5 RSS 残差）→ 64（隐藏层 F'）
- 关系先验 γ_k 按 Table 4-2 初始化（ahead_of=ln(4), nearby_pedestrian=ln(4), contains*=0 排除）
- 多头注意力 H=4，每头输出 head_dim=16，最终**平均**而非拼接（与论文一致）
- **单帧模式**：输入 [N, 1, F']，T 维留作后续多帧窗口

---

### 8.3 第 5 章：KS-NBCF 融合框架（commit e320dfd）

**新建 `stk/fusion/` 包**，包含 5 个文件：

| 文件 | 功能 | 论文章节 | 行数 |
|------|------|---------|------|
| `__init__.py` | 包导出 | — | 23 |
| `feat_injection.py` | φ_feat 编排层（调度 K-HSTGAN） | §5.2 算法 5.1 | 81 |
| `loop_feedback.py` | φ_loop 三阶段闭环反馈 | §5.3 算法 5.2+5.3 | 364 |
| `ds_fuser.py` | D-S 证据理论融合核心 | §5.4 式 5.9–5.21 | 195 |
| `evidence_chain.py` | KG 证据链回溯仲裁 | §5.4.5 算法 5.4 | 290 |

**三阶段闭环关键参数：**
- Stage I 弱监督：γ₃(epoch) = max(0, 0.5·(1−epoch/10))
- Stage II η 更新：β=0.001, η_floor=0.3, EMA 0.9/0.1
- Stage III 动态规则：top-K 注意力映射到 6 种模板
- τ_K = 0.3（§5.4.4 表 5-3，论文正文默认值）

---

### 8.4 K-HSTGAN 接口扩展（支持 KS-NBCF）

为支持融合模块对多头方差 ε_t 和注意力权重的需求，扩展了两个接口：

**`k_hstgan.py forward(data, return_extras=False)`：**
- `return_extras=True` 时额外返回 dict，含：
  - `per_head_anomaly` [N, H]：逐头异常概率（§5.4.2.2 用于 ε_t）
  - `rgat_attention` Dict[int, [H, E_k]]：RGAT 注意力权重（§5.3.4 Stage III + 仲裁）
  - `h_spatial` / `h_temporal` [N, F']：空间编码 / 时序编码
  - `edge_index` / `edge_type` / `delta_feat`
- 不传 `return_extras` 时行为完全向后兼容

**`rgat.py forward(..., return_attention=False)`：**
- 返回 `(h_out, attentions, per_head_per_node)` 或仅 `h_out`

---

### 8.5 关键 bug 修复（commit 05a0d8f）

#### (a) `_attr()` attrs 子字典穿透

**问题**：SafetyViolation pydantic 模型的 src_id/dst_id/severity 存于 `attrs` 子字典中，`model_dump().get("src_id")` 返回 None → exporter / loop_feedback / evidence_chain 无法读取违规信息。

**修复**：三处 `_attr()` 函数统一先查顶层 key，再 fallback 到 `attrs` 子字典。

#### (b) exporter 空间 K-NN fallback

**问题**：scenario_library 不提供 waypoints，`build_lane_topology` 返回空列表 → 场景层 `scene_rels` 永远为空 → GNN 无法构建图（#e=0 对所有场景）。

**修复**：在 `_build_edge_index` 中增加 fallback：当 `scene_rels` 为空时，从车辆/行人 location 构建 K-NN 图（K=5），边类型根据相对方向判定（ahead_of / in_lane / nearby_pedestrian / beside）。验证：S00 两车产生 2 条边，S33 四车产生 12 条边。

#### (c) RGAT softmax / scatter 维度 bug

**问题**：commit 5733e78 中，PyG `softmax` 调用传入2D `idx_dst` [H, E_k]，PyG softmax 要求1D index → 报错。此外 `scatter_add_` 中 source transpose 后维度不匹配。

**修复**：
1. 替换 PyG softmax 为手工 scatter softmax（数值稳定，减 max 后 exp + 分母累加）
2. `node_h.index_add_(1, dst_k, weighted)` 直接传 [H, E_k, head_dim]，source.size(1)=E_k==len(dst_k) ✓

---

### 8.6 Smoke Test 结果

#### K-HSTGAN（commit 5733e78 + 后续修复）
```
✅ PASSED 0.16s (CPU)
场景 S00, max_frames=6
模型参数: 132,607
forward: y_a=(2,1) mean=0.478, y_s=(2,3), y_b=(2,7), y_r=(2,14)
backward loss: 4.59, grad_norm: 3.70 (44 params)
1 epoch trainer: L_total=2.03, stage=I, lr=1e-3
```

#### KS-NBCF（commit e320dfd + 后续修复）
```
✅ PASSED 0.01s (CPU)
场景 S00, max_frames=4
K=0.516, m_fused_anomaly=0.783
evidence_strength=0.800, rule_fires=2
resolve_type=trust_GNN (overlap=1.00)
explanation: (veh_010)-[:in_lane α=1.00]->(veh_011)
```

#### 14 场景全量验证（随机权重）

| 场景 | #v | #e | #vi | resolve | overlap | 说明 |
|------|----|----|-----|---------|---------|------|
| S00  | 2  | 2  | 1   | trust_GNN | 1.00 | in_lane + ahead_of, K>τ_K |
| S01  | 1  | 0  | 0   | consistent | 0.00 | 单车无边 |
| S02  | 2  | 2  | 0   | consistent | 0.00 | 无违规 |
| S10  | 3  | 6  | 0   | consistent | 0.00 | 3车6边，无违规 |
| S11  | 2  | 2  | 0   | consistent | 0.00 | — |
| S12  | 1  | 0  | 0   | consistent | 0.00 | 单车无边 |
| S13  | 2  | 2  | 3   | trust_GNN | 1.00 | 跟车过近 |
| S20  | 3  | 6  | 2   | trust_GNN | 1.00 | 合流冲突 |
| S21  | 3  | 6  | 2   | trust_GNN | 1.00 | 三路冲突 |
| S22  | 2  | 2  | 2   | trust_GNN | 1.00 | 紧急车辆 |
| S30  | 3  | 6  | 0   | consistent | 0.00 | 夜间行人 |
| S31  | 2  | 2  | 2   | trust_GNN | 1.00 | 雨天变道 |
| S32  | 2  | 2  | 2   | trust_GNN | 1.00 | 施工路段 |
| S33  | 4  | 12 | 0   | consistent | 0.00 | 眩光多行人 |

**关键发现：**
- 有 violation 且有边的场景（S00/S13/S20–S22/S31/S32）→ D-S K>τ_K → trust_GNN（overlap=1.0，因为两车互相为对方的唯一邻居）
- 无 violation 的场景 → K=0.0 → consistent，无需回溯
- 注意力权重因随机权重（未训练），所有 α≈1.0（归一化到唯一邻居），训练后会分化
- ε_t=0.0（随机权重各头输出相同），训练后各头分化，ε_t 才有区分度

---

### 8.7 回归测试

```
tests/{ontology,debouncer,extraction,dynamic}：100 passed, 0 new failures ✅
smoke_test_k_hstgan.py（return_extras=False 路径）：PASSED ✅
smoke_test_ks_nbcf.py：PASSED ✅
```

---

### 8.8 Git 提交记录

| Commit | 内容 |
|--------|------|
| `5733e78` | feat(gnn): K-HSTGAN 骨架实现 + smoke test 通过 |
| `e320dfd` | feat(fusion): KS-NBCF 融合框架实现 + smoke test 通过 |
| `05a0d8f` | fix(gnn): exporter K-NN 图构建 + rgat softmax/dim bug 修复 |

---

### 8.9 待办（下一步）

| 优先级 | 任务 |
|--------|------|
| ⭐⭐⭐ | `exp_multiscenario.py`：多场景训练脚本，收集第 6 章 F1/P/R 数据 |
| ⭐⭐⭐ | 训练后评估：KS-NBCF D-S 融合 vs 纯 K-HSTGAN 对比 |
| ⭐⭐ | 第 7 章总结与展望（不依赖实验数据） |
| ⭐⭐ | 参考文献 GB/T 7714 整理 + 中英文摘要 |

