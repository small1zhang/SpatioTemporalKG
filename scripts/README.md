# scripts/ 脚本目录

## 目录分区  (v2.0 — 2026-07-16)

```
scripts/
├── carla/          CARLA 连接、数据提取、交通流生成
│   ├── spawn_traffic.py         生成车流+人流,自动清理
│   ├── run_extraction.py        简单帧提取入口
│   └── ingest_carla_recording.py (待实现)
├── pipeline/       一键全流程
│   └── run_phases_1_5.py        1-5阶段: CARLA→场景→行为→规则→动态→KG
├── replay/         场景回放、可视化
│   ├── build_replay_from_scenario.py  14场景→scene_graph JSON
│   ├── render_scenario_gif.py         GIF渲染
│   ├── render_kg_dashboard_gif.py     KG Dashboard GIF
│   └── build_graph_from_frames.py     帧→图
├── query/          查询分析
│   └── query_anomaly.py
├── remote/         远程连接辅助
│   └── connect.*, server_scripts/
└── archive/        备份
```

用法: python scripts/pipeline/run_phases_1_5.py --frames 60
