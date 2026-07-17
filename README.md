# SpatioTemporalKG — 时空动态知识图谱

> 基于 CARLA 0.9.16 真值与 Neo4j 的自动驾驶场景时空动态知识图谱
> 设计文档：`../时空动态知识图谱实验设计_方案A定版_v3.docx`

## 项目结构

```
SpatioTemporalKG/
├── stk/                 # 主包
│   ├── ontology/        # §1：本体层（实体/关系/属性/公理）
│   ├── scenario/        # §2：场景层（6 类节点 + 5 类关系）
│   ├── behavior/        # §3：行为层（节点+边双轨 + 防抖）
│   ├── rules/           # §4：规则层（RSS 子层 + 交规 R1-R18）
│   ├── dynamic/         # §5：动态更新（Δg_t + 版本管理）
│   ├── storage/         # §6：Neo4j 存储与查询
│   ├── extraction/      # §7：CARLA 真值提取
│   ├── pipeline/        # 主流水线编排
│   ├── viz/            # 可视化与回放
│   └── cli.py          # CLI 入口
├── config/             # YAML 配置文件
├── tests/              # 单元测试
├── scripts/            # 运维脚本
├── docs/               # 设计文档
├── notebooks/          # Jupyter 探索
├── data/               # 录放数据（gitignore）
├── logs/               # 运行日志（gitignore）
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n stk python=3.10 -y
conda activate stk

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .

# 3. 验证安装
pytest tests/test_smoke.py -v
stk --help
```

## 开发阶段

| 阶段 | 名称 | 对应 v3 章节 |
|---|---|---|
| 0 | 项目骨架与环境就绪 | — |
| 1 | 本体层 | §1 |
| 2 | 场景层 | §2 |
| 3 | 行为层 | §3 |
| 4 | 规则层（RSS + 交规 R1-R18） | §4 |
| 5 | 动态更新 | §5 |
| 6 | Neo4j 存储 | §6 |
| 7 | CARLA 提取 | §7 |
| 8 | 集成流水线 | §1-§7 |
