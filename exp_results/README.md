# K-HSTGAN 实验结果目录

本目录是第 6 章全部实验的**唯一权威源**。结构整洁、可追溯、可一键审查。

---

## 一、目录结构

```
exp_results/
├── README.md                       # 本文档（全局索引 + 一键审查指引）
├── verify_data.py                  # 评审一键核查脚本（自动校验所有结果）
├── main_v6/                        # §6.4.2  主实验 (F1 = 1.000)
│   ├── README.md                    #   实验配置 + 论文表 6-13 字段映射
│   ├── checkpoint/
│   │   ├── model_41K_f1_1.000.pt    #   模型权重（540 KB，重命名含 F1）
│   │   ├── results.json              #   最终 P/R/F1/TP/FP/FN/TN
│   │   └── history.json              #   每 epoch 训练曲线
│   └── logs/                        #   9 份训练日志
│       ├── realdata_all_20e_gpu0.log             # 41K 全量训练主日志
│       ├── realdata_2000f_20e_oversample_v6.log    # 2000 帧子集 + oversample
│       └── realdata_2000f_*_gpu1*.log             # 早期调试（v2/v3/v4/v5）
│
├── ablation/                       # §6.4.3  4 组消融
│   ├── README.md                    #   消融对比总览表
│   ├── A_no_oversample/             #   F1 = 0.842（去掉 oversample）
│   │   ├── checkpoint/model_A_no_oversample_f1_0.842.pt
│   │   ├── results.json
│   │   ├── history.json
│   │   └── logs/ablation_A_no_oversample.log
│   ├── B_no_skipconn/               #   F1 = 0.021（去掉 skip connection — 崩溃）
│   ├── C_gamma2/                    #   F1 = 0.842（focal γ=3 → 2）
│   └── D_alpha100/                  #   F1 = 1.000（α_cap=500 → 100）
│
├── pr_curve/                       # §6.4.4  PR 曲线 38 阈值扫描
│   ├── README.md
│   ├── scan_v6_41K.json             #   38 阈值对应的 P/R/F1
│   └── logs/pr_curve_v6_41K.log
│
├── cross_town/                     # §6.7    跨 Town OOD 泛化
│   ├── README.md
│   ├── eval_full.json               #   完整评估（1200 OOD + 4110 ID 帧）
│   └── eval_subset.json             #   早期 100 帧/split 快速评估
│
├── legacy_rq1/                     # 已过时（被 main_v6 + ablation 取代）
│   └── [4 个旧结果文件 + model.pt]
└── legacy_rq2/                     # 已过时
    └── [ablation.json, summary.json]
```

---

## 二、数据 ↔ 论文表/图对应

| 论文位置 | 表/图 | 数据来源（相对路径） | 关键指标 |
|---------|------|---------------------|---------|
| §6.4.2 | 表 6-13 | `main_v6/checkpoint/results.json` | F1 = 1.000，TP=1050，FP=0，FN=0 |
| §6.4.2 | 图 6-1 训练曲线 | `main_v6/checkpoint/history.json` | 20 epoch 收敛轨迹 |
| §6.4.3 | 表 6-13 续 1 | `ablation/{A,B,C,D}/results.json` | 4 组消融 P/R/F1 对比 |
| §6.4.4 | 表 6-13 续 2 + 图 6-2 PR 曲线 | `pr_curve/scan_v6_41K.json` | 21 阈值 P/R/F1 (覆盖 [0.01, 0.46]，16 个阈值达到 F1=1.000) |
| §6.7   | 表 6-18 | `cross_town/eval_full.json` | OOD 合计 FPR = 4.43% |

---

## 三、文件命名规范

**模型权重** 一律形如：

```
model_{规模}_{settings_hash}_f1_{score}.pt
```

| 文件名 | 含义 |
|--------|------|
| `model_41K_f1_1.000.pt` | main_v6 主模型，41K 帧训练，F1 完美 |
| `model_A_no_oversample_f1_0.842.pt` | 消融 A，去 oversample，F1=0.842 |
| `model_B_no_skipconn_f1_0.021.pt` | 消融 B，去 skip connection（崩溃现象，F1=0.021）|
| `model_C_gamma2_f1_0.842.pt` | 消融 C，focal γ=3→2 |
| `model_D_alpha100_f1_1.000.pt` | 消融 D，α_cap=500→100 |

文件名直接含 F1，**评审一眼即可定位主模型、无需打开 json**。

---

## 四、一键审查流程（评审使用）

### 步骤 1：环境准备

```bash
cd /path/to/SpatioTemporalKG
conda activate base   # 或 source .venv/bin/activate
python -c "import torch, json, pandas; print('ok')"   # 确认依赖
```

### 步骤 2：核查全部数据

```bash
python exp_results/verify_data.py
```

脚本会做 5 件事：

1. **存在性校验**：每个 README 中列出的关键文件是否存在
2. **可读性校验**：每个 JSON 是否能被 `json.load` 正常解析
3. **指标一致性校验**：results.json 中的 F1 与文件名中的 F1 后缀是否一致
4. **论文表数据回填校验**：抽取 §6.4.2/§6.4.3/§6.7 中关键数字与 JSON 比对
5. **生成报告**：`exp_results/_verify_report.txt` + 终端彩色打印 PASS/FAIL

期望输出（健康状态）：

```
[PASS] exp_results/main_v6/checkpoint/model_41K_f1_1.000.pt  (540 KB)
[PASS]   └─ results.json: F1=1.000 (filename match ✓)
[PASS] exp_results/ablation/A_no_oversample/checkpoint/model_A_no_oversample_f1_0.842.pt
[PASS]   └─ results.json: F1=0.842 (filename match ✓)
[PASS] exp_results/ablation/B_no_skipconn/checkpoint/model_B_no_skipconn_f1_0.021.pt
[PASS]   └─ results.json: F1=0.021 (filename match ✓)
...
[PASS] cross_town/eval_full.json: OOD FPR=0.04434 (论文 §6.7 表 6-18: FPR=4.43%)
[PASS] pr_curve/scan_v6_41K.json: 38 阈值扫描完整
----------------------------------------------
Summary: 36/36 PASS, 0 FAIL  →  实验数据全部可追溯
Report:  exp_results/_verify_report.txt
```

### 步骤 3：评审报告

报告含 3 张数据出处表 + 1 张 fail 列表（如有）。评审只看 `_verify_report.txt` 即可：

```bash
cat exp_results/_verify_report.txt
```

---

## 五、复现命令（开发用）

```bash
# 主训练（~15 min）
python scripts/long_run/exp_realdata.py \
    --max-frames 41150 --epochs 20 \
    --oversample-pos 5 --focal-gamma 3.0 --alpha-cap 500 --threshold 0.15 \
    --out-dir exp_results/main_v6/checkpoint/

# 4 组消融（每组 ~15 min）
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 --focal-gamma 3.0 \
    --out-dir exp_results/ablation/A_no_oversample/checkpoint/
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 --focal-gamma 2.0 \
    --out-dir exp_results/ablation/C_gamma2/checkpoint/
python scripts/long_run/exp_realdata.py --max-frames 2000 --epochs 20 --alpha-cap 100 \
    --out-dir exp_results/ablation/D_alpha100/checkpoint/
# B (no skip conn) 需修改 stk/gnn/k_hstgan.py 移除 h_temporal += h_spatial

# PR 曲线（~1 min）
python scripts/long_run/pr_curve_scan.py --all \
    --output exp_results/pr_curve/scan_v6_41K.json

# 跨 Town OOD（~2.5 min）
python scripts/long_run/eval_cross_town.py --threshold 0.15 --max-actors 30 \
    --output exp_results/cross_town/eval_full.json
```

---

## 六、备注

- **大模型权重（.pt）不入版本控制**（受 GitHub 100 MB 限制与仓库膨胀考量）
  - `.gitignore` 排除 `exp_results/**/*.pt`，可用脚本一键重训复现
- **JSON + 日志 + README 入版本**，约 200 KB，是论文表/图的直接来源
- **legacy_rq1 / legacy_rq2** 为早期 RQ1/RQ2 探索性实验，已被 main_v6 + ablation 取代，
  保留以备历史回溯，**评审可跳过**
