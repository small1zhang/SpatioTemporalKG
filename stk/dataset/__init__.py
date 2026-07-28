# -*- coding: utf-8 -*-
"""stk/dataset — 真实 CARLA 数据加载与适配

将 data/dataset/ 中的 CSV + event_labels.json 转换为
与 extract_stkg_tensors() 兼容的 snapshot dict 格式，
实现零改动复用现有 GNN + Fusion 链路。

公共接口:
  build_snapshot_from_csv(frame_id, actors_df, labels_df, events) -> Dict
  RealDataDataset(torch.utils.data.Dataset)
  load_realdata_splits() -> (train_ds, val_ds, test_ds)
"""
from .csv_snapshot_builder import (
    build_snapshot_from_csv,
    build_snapshots_from_csv,
    FRAME_LABEL_COLS,
    ACTOR_COLS,
    EVENT_COLS,
    ANOMALY_TYPE_TO_RULE,
)
from .real_data_loader import (
    RealDataDataset,
    load_realdata_splits,
)

__all__ = [
    "build_snapshot_from_csv",
    "build_snapshots_from_csv",
    "FRAME_LABEL_COLS",
    "ACTOR_COLS",
    "EVENT_COLS",
    "ANOMALY_TYPE_TO_RULE",
    "RealDataDataset",
    "load_realdata_splits",
]
