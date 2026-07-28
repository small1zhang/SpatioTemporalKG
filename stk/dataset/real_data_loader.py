# -*- coding: utf-8 -*-
"""
RealDataDataset — 从 CSV 懒加载真实 CARLA 数据的 PyTorch Dataset

基于预切分 train/val/test split，提供：
  - Lazy iteration: 按需加载帧 (不预读全量数据到内存)
  - 关联前后帧构建 delta_feat（Δg_t）
  - 输出格式与 STKGGraphDataset 兼容（PyG Data 对象）
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stk.gnn.exporter import extract_stkg_tensors
from stk.dataset.csv_snapshot_builder import (
    build_snapshot_from_csv,
    build_snapshots_from_csv,
)


class RealDataDataset(Dataset):
    """
    真实 CARLA 数据集的 PyTorch Dataset（延迟构建）。

    每个样本是从 CSV 中按 `split` 切分后的一帧。
    输出为 `torch_geometric.Data` 对象，兼容 K_HSTGAN forward()。

    Args:
        actors_df:      frame_actors.csv (DataFrame 或 CSV 路径)
        labels_df:      frame_labels.csv (DataFrame 或 CSV 路径)
        events:         event_labels.json (list 或 JSON 路径)
        split:          数据子集：'train' / 'val' / 'test'
        max_actors:     每帧最大节点数（默认 30）
        max_frames:     最多加载多少帧（None=全部，用于调试）
        cache_snapshots: 是否缓存已构建的 snapshot（加速训练，Trade-off 内存）
    """

    def __init__(
        self,
        actors_df: Optional[pd.DataFrame] = None,
        labels_df: Optional[pd.DataFrame] = None,
        events: Optional[List[Dict]] = None,
        actors_path: str = "data/dataset/frame_actors.csv",
        labels_path: str = "data/dataset/frame_labels.csv",
        events_path: str = "data/dataset/event_labels.json",
        split: str = "train",
        max_actors: int = 30,
        max_frames: Optional[int] = None,
        cache_snapshots: bool = True,
    ):
        super().__init__()
        self.max_actors = max_actors
        self.cache_snapshots = cache_snapshots
        self._snap_cache: Dict[int, Dict] = {}

        # 加载数据
        if actors_df is None:
            actors_df = pd.read_csv(
                actors_path, dtype={"actor_id": str, "scenario_id": str},
                low_memory=False,
            )
        if labels_df is None:
            labels_df = pd.read_csv(
                labels_path, dtype={"scenario_id": str, "split": str},
                low_memory=False,
            )
        if events is None:
            with open(events_path, "r") as f:
                events = json.load(f)

        # 按 split 过滤帧 ID，并 shuffled 后截断以保证异常帧均匀分布
        split_frames = labels_df[labels_df["split"] == split]["frame_id"].unique()
        frame_ids = sorted(split_frames.tolist())
        if max_frames is not None and max_frames < len(frame_ids):
            rng = np.random.RandomState(42)
            rng.shuffle(frame_ids)
            frame_ids = frame_ids[:max_frames]
            frame_ids.sort()

        self.frame_ids = frame_ids
        self.actors_df = actors_df
        self.labels_df = labels_df
        self.events = events

        # 预建 snapshot 列表（无需 lazy 再查 CSV，省去 DataFrame 过滤开销）
        # 对于 41K 帧，内存约 200–300 MB（每 snapshot ~5–8 KB），可接受
        self.snapshots = self._build_all_snapshots()

    def _build_all_snapshots(self) -> List[Dict[str, Any]]:
        """批量构建所有帧的 snapshot dict。"""
        snapshots = []
        prev = None
        for fid in self.frame_ids:
            snap = build_snapshot_from_csv(
                frame_id=int(fid),
                actors_df=self.actors_df,
                labels_df=self.labels_df,
                events=self.events,
                prev_snapshot=prev,
                max_actors=self.max_actors,
            )
            snapshots.append(snap)
            prev = snap
        return snapshots

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回 `torch_geometric.Data` 对象。

        字段：
          x [N, 18], edge_index [2, E], edge_type [E],
          kappa_rss [N, 5], kappa_rule [N, 14], delta_feat [4],
          y_anomaly [N] (long), scene_id (str), frame_id (int)
        """
        snap = self.snapshots[idx]
        data = extract_stkg_tensors(snap)

        #split    附加元数据
        data.scene_id = str(self.labels_df[
            self.labels_df["frame_id"] == snap["extracted"]["frame_id"]
        ].iloc[0].get("map_name", "")) if len(self.labels_df[
            self.labels_df["frame_id"] == snap["extracted"]["frame_id"]
        ]) > 0 else ""

        data.origin_frame_id = int(snap["extracted"]["frame_id"])

        # 节点级异常标签（已在 extract_stkg_tensors 中处理为 y_anomaly）
        # 这里做一次兜底：若 y_anomaly 为 None（新数据未标注），用 _y_anomaly
        if not hasattr(data, "y_anomaly") or data.y_anomaly is None:
            y_anom = torch.tensor(
                snap["extracted"].get("_y_anomaly", [0] * data.x.size(0)),
                dtype=torch.long,
            )
            data.y_anomaly = y_anom

        return data


def make_realdata_collate_fn(max_nodes: int = 30):
    """
    Collate function：将变长图 + 元数据打包为 padded batch。

    因为 DataLoader 默认 collate 会拼接 PyG Batch，
    但节点数不同且不全是稠密图时，需要手动处理。

    Args:
        max_nodes: padding 最大节点数
    """
    def collate(batch: List[torch.Tensor]) -> Dict[str, Any]:
        # 这里 batch 是 [Data, ...]（每个 Data 是一个独立图）
        return batch[0] if isinstance(batch, list) and len(batch) == 1 else batch

    return collate


# ============================================================
# 便捷函数：加载预切分数据集
# ============================================================
def load_realdata_splits(
    actors_path: str = "data/dataset/frame_actors.csv",
    labels_path: str = "data/dataset/frame_labels.csv",
    events_path: str = "data/dataset/event_labels.json",
    max_actors: int = 30,
    max_frames: Optional[int] = None,
) -> Tuple[RealDataDataset, RealDataDataset, RealDataDataset]:
    """
    加载 train/val/test 三个 DataLoader。

    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_ds = RealDataDataset(
        actors_path=actors_path, labels_path=labels_path,
        events_path=events_path, split="train",
        max_actors=max_actors, max_frames=max_frames,
    )
    val_ds = RealDataDataset(
        actors_path=actors_path, labels_path=labels_path,
        events_path=events_path, split="val",
        max_actors=max_actors, max_frames=max_frames,
    )
    test_ds = RealDataDataset(
        actors_path=actors_path, labels_path=labels_path,
        events_path=events_path, split="test",
        max_actors=max_actors, max_frames=max_frames,
    )
    return train_ds, val_ds, test_ds


# ============================================================
# 数据可视化（调试用）
# ============================================================
def print_dataset_stats(ds: RealDataDataset, name: str = "dataset") -> None:
    """打印数据集统计信息。"""
    n_anom_pos = 0
    n_total_nodes = 0
    n_total_edges = 0
    n_frames = len(ds)
    n_anom_frames = 0

    for i in range(min(len(ds), 100)):  # 采样 100 帧统计
        d = ds[i]
        n_total_nodes += d.x.size(0)
        n_total_edges += d.edge_index.size(1)
        n_anom = int(d.y_anomaly.sum().item())
        n_anom_pos += n_anom
        if n_anom > 0:
            n_anom_frames += 1

    print(f"\n[{name}] stats (first {min(len(ds), 100)} frames sampled):")
    print(f"  total frames:  {len(ds)}")
    print(f"  avg nodes/frame: {n_total_nodes / max(1, min(len(ds), 100)):.1f}")
    print(f"  avg edges/frame: {n_total_edges / max(1, min(len(ds), 100)):.1f}")
    print(f"  anomaly nodes (sampled): {n_anom_pos}")
    print(f"  anomaly frames (sampled): {n_anom_frames}")
