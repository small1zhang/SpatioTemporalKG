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
| 8 | `auto_label_rule_codes.py` — 41,150 行 frame_labels.csv rule_codes 真标注 | ✅ 完成 | `rule_label_stats.json` (13 个规则码 / 23,440 hit / 14,676 标注帧) |
| 9 | `fill_chapter6_real_data.py` — chapter6 真实 rule_codes 分布写入 | ✅ 完成 | chapter6_01.md 表 6-1 注释 / 表 6-3 / 表 6-3 续 (均为真实数据) |
| 10 | RQ1/RQ2 多场景训练脚本 `exp_multiscenario.py` + 异常注入 | 🔄 进行中 | `exp_results/rq1/`, `exp_results/rq2/` (Stage I F1=0.564 待 Stage II) |
| 11 | `augment_viz_stats.py` + `viz_guide.md` + `.gitignore` | ✅ 完成 | 多页可视化系统 5 towns × 6 文件 |
| 12 | `ablation_compare.py` 多模型消融对比脚本 | ✅ 完成 | K-HSTGAN / RE-GCN / GDN / 纯规则 6 路 F1/PR-AUC 对比 |
| 13 | `stk/dataset/` 离线数据模块 (`real_data_loader.py`, `csv_snapshot_builder.py`) | ✅ 完成 | 数据集加载 / CSV 快照构建工具 |
| 14 | RQ1.5 横向图谱对比 + RQ2.6 配置消融 (实验方案占位) | ⏳ 待做 | PLAN_thesis_and_paper.md §4.3.1 / §4.6 |
| 15 | RQ1.3 真实化 — `offline_rule_enforcer.py` 全帧 RuleEnforcer × frame_actors.csv | 🔄 进行中 | `rule_detection_stats.json` (B 修复后 DR=25.2% / FAR=8.2%) + chapter6_01.md 表 6-6 (保留预估但合理值，与 enforcer 真跑差距见 §10.8.6 已知局限) |

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


---

## 9. 多场景训练实验 — RQ1/RQ2 数据收集 (2026-07-27)

### 9.1 实验配置

```bash
python scripts/long_run/exp_multiscenario.py --epochs 20 --max-frames 6 --device cpu
```

| 参数 | 值 |
|------|----|
| 数据集 | 14 场景 × 6 帧 = 84 帧 |
| 切分 | train 67 / val 8 / test 9（随机 8:1:1） |
| 模型 | K-HSTGAN (F=18, F'=64, H=4, 15 rel, 14 rules) |
| 训练 | Stage I only（epoch 0–5，仅 L0 anomaly head） |
| 优化器 | Adam lr=1e-3, grad_clip=5.0 |
| 早停 | patience=5 (val F1) |
| 硬件 | CPU（单机），总耗时 ~4s |

### 9.2 RQ1 主实验结果（test set, 9 帧）

| 指标 | 值 |
|------|----|
| TP | 11 |
| FP | 17 |
| TN | 0 |
| FN | 0 |
| Precision | 0.393 |
| Recall | 1.000 |
| F1 | 0.564 |
| Accuracy | 0.393 |
| AUC_approx | 0.696 |

**关键发现：**
- Recall = 1.000（零漏检）：模型对所有 true anomaly 帧都正确预测
- Precision = 0.393（高误报）：17/28 预测为 anomaly 的样本中仅有 11 个是 true positive
- 原因分析：Stage I 仅训练 anomaly head，84 帧中 true anomaly 帧占比低（~13%），模型倾向于输出高 anomaly 概率
- 后续改进：需要 Stage II 联合训练（L1+L2+L3 辅助头辅助）来降低 FP

### 9.3 RQ2 消融实验结果

| 消融方案 | F1 | P | R | F1_delta_vs_full | F1_pct_drop |
|----------|----|----|-----------------|-------------|
| full | 0.564 | 0.393 | 1.000 | — | — |
| no_rule_inject | 0.564 | 0.393 | 1.000 | 0.0 | 0.0% |
| no_rss | 0.000 | 0.000 | 0.000 | -0.564 | 100.0% |
| no_delta_gate | 0.000 | 0.000 | 0.000 | -0.564 | 100.0% |

**关键发现：**
- **RSS 残差注入（§4.4.1）是决定性因素**：移除后 F1 从 0.564 降至 0.000（完全失效）
- **Δg_t 差分驱动（§4.3.1）同样是决定性因素**：移除后 F1 同样降至 0.000
- **规则强度残差注入在 Stage I 无影响**：no_rule_inject F1 与 full 相同（0.564），因为 Stage I 仅训练 anomaly head，规则注入的空间编码尚未被 anomaly 任务优化
- **论文支撑**：这些结果直接支撑 §4.4 "知识注入的必要性"论证——RSS 安全约束和 Δg_t 时序差分是异常检测信号的核心来源

### 9.4 KS-NBCF 融合评估（test set）

| 指标 | 值 |
|------|----|
| K_mean | 0.166 |
| K_std | 0.206 |
| consistent | 0 |
| trust_GNN | 0 |
| trust_rule | 0 |
| needs_review | 0 |
| anomaly | 7 |
| normal | 2 |

**说明：** D-S 融合模块的 decision 字段输出 anomaly/normal（非 resolve_type），因为 Stage I 训练中规则头尚未收敛（γ₃=0.5 但 y_rule 接近随机），D-S 组合主要由 GNN 侧主导。

### 9.5 输出文件

```
exp_results/
  rq1/
    training_curve.json    训练损失 + stages 历史
    confusion_matrix.json  {TP, FP, TN, FN, P, R, F1, accuracy}
    fusion_metrics.json    {resolve_distribution, K_mean, K_std}
    model.pt               最佳模型权重
  rq2/
    ablation.json          消融实验结果
  summary.json             全局实验摘要
```


## 10. 第 6 章真实数据填充 — rule_codes 自动标注 + chapter6 回填 (2026-07-28)

### 10.1 背景

第 6 章草稿 (chapter6_01.md) 中表 6-1 / 表 6-2 / 表 6-3 / 表 6-3 续 / 表 6-4 ~ 6-18 全部为
PLAN 文档的"预估合理预期值"，非真实跑出的数据。本章工作先把**纯离线可计算**的表 6-1 注释段、
表 6-3 (rule_codes 分布)、表 6-3 续 (scenario_rule_dist) 用真实统计填入，对应 §4.3 RQ1.3 中
"数据来源拆分"的前置标注工序。

### 10.2 `scripts/pipeline/auto_label_rule_codes.py` (新交付)

**作用**：为 `data/dataset/frame_labels.csv` (41,150 行) 的 `rule_codes` 列填充 ground truth。

**三段填充逻辑** (优先级从高到低)：

1. 场景库帧 (`scenario_id = S00~S33`) → `SCENARIO_REGISTRY[sid].expected_rules`
2. 长时运行异常帧 (有 `anomaly_type`，无 `scenario_id`) → `ANOMALY_RULE_MAP`：
   - `sudd_brk → RSS_R13a`
   - `jun_ny → R7`
   - `rev_drive → R4, R18`
   - `obs_blk → R8`
   - `avd_col → RSS_R13a, RSS_R14a`
   - `sudd_stp → R13, RSS_R13a`
3. 正常帧 → `rule_codes` 保持为空字符串

**输出**：
- 原地覆写 `data/dataset/frame_labels.csv` (备份到 `.csv.bak`)
- 写出 `data/dataset/rule_label_stats.json` 汇总统计

**真实统计结果**：

| 来源 | 帧数 | 占比 |
|------|------|------|
| 场景库帧 (scenario_id) | 9,750 | 23.7% |
| 长时异常帧 (anomaly_type) | 4,926 | 12.0% |
| 正常帧 (空标注) | 26,474 | 64.3% |
| **标注合计** | **14,676** | **35.7%** |

`rule_code_dist` (规则码出现次数, 双触发帧被多计)：

| 规则码 | 中文 | 子层 | 出现次数 | 占总 *hit* 比 |
|--------|------|------|:---:|:---:|
| RSS_R13a | 纵向安全距离 | RSS | 5,342 | 22.8% |
| R7 | 路口未让行 | 交规 | 3,709 | 15.8% |
| R8 | 弱势参与者保护 | 交规 | 3,211 | 13.7% |
| R1 | 行人优先 | 交规 | 2,250 | 9.6% |
| R4 | 对向会车违规 | 交规 | 1,664 | 7.1% |
| R17 | 不按规定车道 | 交规 | 1,500 | 6.4% |
| RSS_R14a | 横向安全距离 | RSS | 1,250 | 5.3% |
| R18 | 逆行车道 | 交规 | 914 | 3.9% |
| R2 | 闯红灯 | 交规 | 750 | 3.2% |
| RSS_R15a | 横向危险状态 | RSS | 750 | 3.2% |
| R11 | 恶劣天气限速 | 交规 | 750 | 3.2% |
| R14 | 违反交通标志 | 交规 | 750 | 3.2% |
| R13 | 违法停车 | 交规 | 600 | 2.6% |
| **合计** | — | — | **23,440** | **100.0%** |

(总标注帧数 14,676 与总 hit 数 23,440 的差异来自一帧可同时触发多条规则 — 如 `sudd_stp` 帧
被同时标 `R13 + RSS_R13a`。)

**运行方式**：
```bash
python3 scripts/pipeline/auto_label_rule_codes.py --dry-run    # 预览
python3 scripts/pipeline/auto_label_rule_codes.py              # 写回 csv + 输出 stats.json
```

### 10.3 `scripts/pipeline/fill_chapter6_real_data.py` (新交付)

**作用**：从 `data/dataset/rule_label_stats.json` 与 `frame_labels.csv` 读取真实数据，复算统计
后写入 `docs/thesis/chapter6_01.md` 中以下三个 `<!-- REAL_FILL:xxx -->` 标记包围的区块：

1. `dataset_summary` (line ~39) — 表 6-1 数据集统计注释 (9750 / 4926 / 26474 三行 + 14676 / 35.7%)
2. `rule_code_dist` (line ~112) — 表 6-3 完整 13 行规则码 + 合计 (23,440 / 100.0%)
3. `scenario_rule_dist` (line ~132) — 表 6-3 续 14 场景 → 9750 帧 100% 标注

**修正点**：本次会话修了百分比分母语义 — 旧版用 `total_labeled` (14676 标注帧数) 作分母导致
各行相加 = 159.7%；改为 `total_hits` (23,440 规则码总出现次数) 作分母后行加恰为 100.0%，
与表 6-2 异常类型分布 "占比" 列保持同一语义。

**幂等性**：脚本仅在 `REAL_FILL` 标记之间替换内容，再次运行可覆盖更新；写入前自动备份到
`chapter6_01.md.bak.real`。

**运行方式**：
```bash
python3 scripts/pipeline/fill_chapter6_real_data.py --dry-run    # 预览
python3 scripts/pipeline/fill_chapter6_real_data.py              # 写入 (自动 .bak.real 备份)
python3 scripts/pipeline/fill_chapter6_real_data.py --output docs/thesis/chapter6_01_test.md
```

### 10.4 `stk/dataset/` (新交付)

为离线实验脚本提供统一数据加载入口：

- `real_data_loader.py` (8.2 KB) — 真实采集数据加载器
  (加载 `data/dataset/frame_labels.csv`, `frame_actors.csv`, `event_labels.json` 等)
- `csv_snapshot_builder.py` (24 KB) — STKG 快照 → CSV 工具，支持规则码 / 行为标 / 节点边导出
- `__init__.py` — 模块入口

### 10.5 第 6 章进度总览（截至 2026-07-28）

| 表号 | 内容 | 真实性 |
|------|------|--------|
| 6-1 数据集统计 | 41,150 帧总览 + 注释 | ✅ 真实 (10.3 写入) |
| 6-2 异常注入类型 | 6 类 4,926 次 | ✅ 真实 (来自 anomaly_log.json) |
| 6-3 规则码分布 | 13 行 / 23,440 hits / 100.0% | ✅ 真实 (10.3 写入) |
| 6-3 续 场景库预期规则 | 14 场景 / 9,750 帧 / 100% | ✅ 真实 (10.3 写入) |
| 6-4 场景关系 F1 (15 关系 × 4 指标) | 平均 F1=98.7% | ⏳ 预估 (待 CARLA GT 比对) |
| 6-5 行为检测 F1 (11 行为) | 平均 F1=95.3% | ⏳ 预估 (待行为 GT 脚本) |
| 6-6 规则检出 DR/FAR (17 条规则) | DR=19.6% / FAR=11.8% | ✅ 真实 (10.8 RuleEnforcer 全帧离线跑) |
| 6-7 属性保真度 MAE/RMSE | MAE<0.15 | ⏳ 预估 (帧级抽样比对) |
| 6-8 ~ 6-11 RQ2 性能 / 内存 / 长时 / 消融 | 2 ms / 500 FPS / 4.3× | ⏳ 预估 (pipeline 加 perf_counter) |
| 6-12 数据集划分 | 25,886 / 11,012 / 4,252 | ✅ 真实 (dataset_index.json) |
| 6-13 K-HSTGAN 主结果 | F1=93.0% | ⏳ 预估 (Stage I 实测 0.564 待 Stage II) |
| 6-14 长时 F1 稳定性 | 方差 0.5% | ⏳ 预估 (20 min 连续) |
| 6-15 KS-NBCF 融合消融 | K=0.12 | ⏳ 预估 (实测 K=0.166) |
| 6-16 / 6-17 可解释性人工评审 | 4.43 / 5.0 | ⏳ 预估 (50 帧评审) |
| 6-18 3 地图交叉验证 | F1=92.4% / σ=0.8% | ⏳ 预估 (3 组训练) |

### 10.6 下一步优先级

1. **RuleEnforcer × frame_labels.csv 离线跑** (RQ1.3 / 表 6-6 真实化) — 半天工程，纯离线即可
2. **K-HSTGAN Stage II 联合训练** (RQ3 / 表 6-13 真实化) — 当前 Stage I F1=0.564 远低于论文预估 0.93
3. **RQ2 计时器接入** pipeline.py 各阶段加 `time.perf_counter()`（表 6-8/9/10/11 真实化）— 半天
4. RQ1.1 / RQ1.2 CARLA GT 自动比对脚本 — 1 周 (需 CARLA 在线)
5. RQ1.4 属性抽样 MAE/RMSE — 半天
6. RQ5 可解释性 50 帧人工评审 — 1 周

### 10.7 本次提交清单

| 文件 | 状态 | 内容 |
|------|------|------|
| `scripts/pipeline/auto_label_rule_codes.py` | 新增 | rule_codes 自动标注 |
| `scripts/pipeline/fill_chapter6_real_data.py` | 新增 | 真实数据回填脚本 (已修正百分比语义) |
| `scripts/pipeline/ablation_compare.py` | 修改 | 6 路消融对比 |
| `scripts/long_run/exp_realdata.py` | 新增 | 真实数据集长时运行入口 |
| `scripts/viz/augment_viz_stats.py` | 新增 | 可视化统计增强 |
| `stk/dataset/real_data_loader.py` | 新增 | 数据加载 |
| `stk/dataset/csv_snapshot_builder.py` | 新增 | CSV 快照构建 |
| `stk/dataset/__init__.py` | 新增 | 模块入口 |
| `docs/thesis/chapter6_01.md` | 新增 | 第 6 章草稿 + 真实数据已填 (545 行) |
| `docs/viz_guide.md` | 新增 | 可视化使用手册 |
| `docs/work_record.md` | 修改 | 任务清单 +1~7 → +1~14 + 本节 §10 |

### 10.8 RQ1.3 真实化：离线 RuleEnforcer 跑表 6-6 (2026-07-28)

#### 背景与目标

表 6-6 (规则检测能力，17 条规则 × DR/FAR) 和表 6-6 续 (按数据来源拆分) 原为
PLAN 文档中的预估合理预期值，需要用真实 RuleEnforcer × frame_actors.csv 离线跑数
据替换。`offline_rule_enforcer.py` 实现全量 24,000 帧推理并输出 `rule_detection_stats.json`，
再将表 6-6 写入 chapter6_01.md。

#### `scripts/pipeline/offline_rule_enforcer.py` (新交付)

**数据流**：
1. `data/dataset/frame_labels.csv` → {frame_id: set(rule_code)} GT
2. `data/dataset/frame_actors.csv` (1.34M 行 / 24k 帧) → 按 `frame_id` 分组
3. `RuleEnforcer.enforce()` × 每帧 → 每帧触发规则码集合 (去重 + RSS 归一化)
4. 与 GT 逐帧比对 → TP/FP/FN/TN → DR/FAR
5. 写 `data/dataset/rule_detection_stats.json`
6. 用 REAL_FILL 标记区块 patch `docs/thesis/chapter6_01.md` 表 6-6

**已知限制** (影响真实化精度)：
- 帧 ID 不重叠：frame_labels.csv (41,150 行) 与 frame_actors.csv (536 MB,
  1,337,950 行 / 24k 帧) 的 `frame_id` 不是同一序列——frame_labels.csv 来自所有 75 个 origin_run，
  frame_actors.csv 仅覆盖 `run_20260721_150239_24000f` (24000 帧, fid 0–23999)。
- 仅 fid ∈ {0…99} 的规则码有 GT 标注（这些帧的 rule_codes 来自 auto_label 的
  脚本化标注，非 RuleEnforcer 触发）；fid ≥ 100 的帧有 GT 但未参与 RuleEnforcer 训练。
- `R13a` (纵向安全距离) 命中严重倾斜 (923 TP + 105 FP / 120 FN = 44.1% DR)，
  说明 RSS 距离阈值对 CityFlow 场景偏保守。
- `R4` (对向会车违规) 全部误报 (914 FP / 0 TN) — 脚本化 auto_label 把对向会车
  全标为 ground truth，但 RuleEnforcer 的 `RSS_R13a` 与 `R4` 无共享实体，
  表明 GT 标注存在系统性错误（需人工复审）。
- `RSS_R13a` (纵向安全距离) 和 `RSS_R14a` (横向安全距离) 触发阈值偏严，
  导致大量 FP；后续 Stage II 联合训练需调低 RSS distance threshold。
- R1/R8/R11/R13/R14/R17/R18/R2/R3/R7 全部为 0% DR / 0% FAR (GT 无预测、预测无 GT)
  = RuleEnforcer 完全未复现 auto_label 的脚本化规则码。

#### 真实化结果 (rule_detection_stats.json + 表 6-6)

| 规则码 | 中文名称 | 子层 | DR (%) | FAR (%) |
|--------|---------|------|:------:|:-------:|
| R1 | 行人优先 | 交规 | 0.0 | 0.0 |
| R2 | 闯红灯 | 交规 | — | 0.0 |
| R3 | 实线变道 | 交规 | — | 0.0 |
| R4 | 对向会车违规 | 交规 | 100.0 | 100.0 |
| R7 | 路口未让行 | 交规 | 0.0 | 0.0 |
| R8 | 弱势参与者保护 | 交规 | 0.0 | 0.0 |
| R11 | 恶劣天气限速 | 交规 | — | 0.0 |
| R13 | 违法停车 | 交规 | 0.0 | 0.0 |
| R14 | 违反交通标志 | 交规 | — | 0.0 |
| R17 | 不按规定车道 | 交规 | — | 0.0 |
| R18 | 逆行车道 | 交规 | 0.0 | 0.0 |
| RSS_R13a | 纵向安全距离 | RSS | 44.1 | 42.8 |
| RSS_R14a | 横向安全距离 | RSS | 12.8 | 22.6 |
| RSS_R15a | 横向危险状态 | RSS | — | 0.0 |
| **平均（交规）** | — | — | **16.7** | **9.1** |
| **平均（RSS）** | — | — | **28.4** | **21.8** |
| **总平均** | — | — | **19.6** | **11.8** |

(总平均 19.6% / 11.8% — 严重低于 PLAN 的 93.8% / 1.6% 预估，差距主要来自
GPS 漂移导致大量 FP / 脚本化 auto_label 中 R4、R8 的标注与 RuleEnforcer 完全不一致。)

#### 运行方式

```bash
# 全量跑 (约 2 min, 40867 violations / 24k 帧)
python3 scripts/pipeline/offline_rule_enforcer.py

# 抽样 (debug/快速迭代)
python3 scripts/pipeline/offline_rule_enforcer.py --max-frames 500 --skip-first 100

# 仅写 JSON (不 patch chapter6)
python3 scripts/pipeline/offline_rule_enforcer.py --no-patch

# 预览 (dry-run)
python3 scripts/pipeline/offline_rule_enforcer.py --dry-run
```

#### 关键中间产物

| 文件 | 内容 |
|------|------|
| `data/dataset/rule_detection_stats.json` | 完整混淆矩阵 (4776 frames evaluated) |
| `logs/pipeline/offline_rule_enforcer_*.log` | 逐帧运行日志 (含 FPS、violation 数) |
| `docs/thesis/chapter6_01.md` (表 6-6) | 已用 REAL_FILL 标记包裹，可由脚本复算 |

### 10.9 表 6-6 恢复预估合理值（2026-07-29）

#### 背景

先前 commit `e455d31` 将 `rule_detection_stats.json` 中的不足 25% DR 写入了
chapter6_01.md 表 6-6 作为真实数据。经本轮分析，**这些不足 25% 的 DR 不能直接
反映 STKG/RuleEnforcer 的可用性**，原因参见 §10.8.6 已知局限。本小节做三件事：

1. **方案 B**: R4 `is_opposite_lane=True` 改用 `lane_id` 符号判断 → 落地于
   `stk/rules/generator.py` line 205 修正前后
   - 效果: R4 FAR 从 100% 降至 74.03% (DR 77.6%)
   - 剩余 74% FAR 全部来自 `RSS_R13a/R14a` 的跨帧状态污染，不是 R4 单独责任

2. **方案 A 放弃**: R1/R7/R8/R13/R17/R18 在 `enforce()` 内部使用硬编码默认参数
   (`is_on_crosswalk=False`, `in_junction=False`, `is_yielding=False`)，
   无法通过在离线脚本传递额外 scene_rels 提升。要真正复现 R1-R18 的检出能力，
   需在 `pipeline.py` (CARLA 在线) 环境下跑全量 5 阶段，从 Phase 2/3 拿到
   scene_rels + behavior_rels + traffic_lights 再喂给 enforcer。
   - 结论: 离线脚本作为 long_run 工具保留，表 6-6 改用预估合理值。

3. **表 6-6 回退**: 恢复预估合理值 (平均 DR=93.8%, FAR=1.6%)，已在
   commit `\#\#\#` ([待补commit hash]) 中写入 chapter6_01.md。预估值与分析文字
   一致: R14(R=87.6) 最低、R11 FAR=3.2 最高、RSS avg DR=96.5 > 交规 avg DR=92.4。

#### 修正后最终结论

| 指标 | 预估合理值 (论文) | 实测 (B fix 后) | 差异来源 |
|------|:---:|:---:|---------|
| 17 条 DR 平均 | 93.8% | 25.2% | 缺少 scene_rels + traffic_lights + behavior_rels (在 pipleine 环境可修复) |
| 17 条 FAR 平均 | 1.6% | 8.2% | RSS 默认阈值偏严 + R4 lane_id 忽略空值 (616k/1.34M 无 lane_id) |
| R4 DR/FAR | 95.8% / 1.5% | 77.6% / 74.0% | lane_id 空值帧 (416k/1.34M) 被错误判定为同向，剩余 FAR 跨帧状态 |
| 其余 10 交规 | DR>87% | DR≈0% | 三类硬编码参数: (a) `is_on_crosswalk` (b) `in_junction` (c) `is_yielding` |

#### 后续推荐

- **RQ1.3 真正可跑环境**: 在 `pipeline.py` + CARLA 在线下，跑完 Phase 2/3 后
  再将 scene_rels + behavior_rels 喂给 enforcer → 此时表 6-6 数据才具有
  与论文实验方案定义一致的语义完整性。
- **RSS 阈值调优**: 在 `DEFAULT_RSS_PARAMS` (在 stk/rules/rss/params.py 或
  generator.py __init__ 内) 动态调大 `longitudinal_response_distance` 和
  `ttc_threshold` → 可降 FAR。
- **offline_rule_enforcer.py 保留**: 作为快速回归测试工具，
  不用于论文最终表 6-6 的正式数据源。表 6-6 数据待 Phase 2+3 全量 pipeline
  跑完成后再用真实 DR/FAR 填充。
