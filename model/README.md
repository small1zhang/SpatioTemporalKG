# Model Folder — 模型复现与实验代码

本文件夹存放论文实验中所有模型的复现代码，包括：

- **K-HSTGAN** — 论文核心骨架（已有 `stk/gnn/k_hstgan.py`，此处提供实时检测集成）
- **KS-NBCF** — 本文完整模型（融合 K-HSTGAN + φ_loop + φ_fuse + φ_arb）
- **RE-GCN** — 基线模型 1（关系增强图卷积网络）
- **GDN** — 基线模型 2（Graph Domain Network，领域自适应）
- **GeneralDyG** — 基线模型 3（通用动态图网络）

---

## 目录结构

```
model/
├── test_models.py           # 统一 smoke test（所有模型验证入口）
├── k_hstgan/
│   └── realtime_detection.py  # CARLA → STKG → K-HSTGAN → KS-NBCF 实时检测引擎
├── ks_nbcf/
│   └── model.py             # KS-NBCF 完整实现（φ_fuse + φ_loop + φ_arb）
├── re_gcn/
│   └── re_gcn.py            # RE-GCN 复现
├── gdn/
│   └── gdn.py               # GDN 复现
└── general_dyg/
    └── general_dyg.py       # GeneralDyG 复现
```

## 快速使用

```bash
# 运行所有模型的 smoke test
python model/test_models.py

# 单独测试某模型
python model/re_gcn/re_gcn.py
python model/gdn/gdn.py
python model/general_dyg/general_dyg.py
```

## 与 CARLA 数据管道联通

```python
from model.k_hstgan.realtime_detection import RealtimeDetectionEngine
from stk.pipeline.orchestrator import PipelineOrchestrator

engine = RealtimeDetectionEngine(device="cpu")

# 使用现有 pipeline 获取 snapshot
orchestrator = PipelineOrchestrator()
summary = orchestrator.run_scenario("S00", max_frames=4)
snapshot = orchestrator.snapshot_store.get(summary["results"][0]["frame_id"])

# 实时推理
result = engine.process_frame_from_carla(carla_frame=None, snapshot=snapshot)
print(f"Decision: {result['d_s_fusion']['decision']}")
```

## 模型输出说明

| 字段 | 形状 | 含义 |
|------|------|------|
| `y_anomaly` | [N, 1] | 节点级异常概率（sigmoid） |
| `y_scene` | [N, 3] | 场景分类（softmax） |
| `y_behavior` | [N, 7] | 行为分类（softmax） |
| `y_rule` | [N, 14] | 交规触发 multi-label |
| `fusion_decision` | str | D-S 融合决策："anomaly"/"normal"/"uncertain" |
| `K` | float | D-S 冲突系数 |
| `resolve_type` | str | 仲裁结果：consistent / trust_GNN / trust_rule / needs_review |

## 实验表格目标结果

| 模型 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | A10 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Rule-only | 82.8 | 92.2 | 87.2 | 91.3 | 88.5 | 79.9 | 89.4 | 84.4 | 88.4 | 85.3 |
| RE-GCN | 87.1 | 84.6 | 85.8 | 91.8 | 87.6 | 84.1 | 81.5 | 82.8 | 89.2 | 84.9 |
| GDN | 89.0 | 86.4 | 87.7 | 93.0 | 89.5 | 86.3 | 83.0 | 84.6 | 90.8 | 87.2 |
| GeneralDyG | 90.0 | 88.6 | 89.3 | 94.1 | 91.2 | 87.8 | 85.7 | 86.7 | 92.0 | 88.8 |
| **Ours (KS-NBCF)** | **91.0** | **89.4** | **90.2** | **95.0** | **92.2** | **88.7** | **86.8** | **87.7** | **92.9** | **89.9** |

> 注：以上为论文报告的目标值，实际实验结果以训练后报告为准。