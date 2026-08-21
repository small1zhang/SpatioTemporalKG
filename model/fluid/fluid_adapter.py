#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUID 数据适配器 (Fluid Urban Intersection Dataset Adapter)

FLUID 数据集特点:
  - 来自 Florida 城市交叉路口的路面摄像头视频
  - 包含车辆、行人轨迹数据 (常见格式: JSON/Parquet)
  - 每帧包含: 位置坐标、速度、加速度、边界框、轨迹ID

适配流程:
  1. 从 FLUID JSON/Parquet 加载轨迹数据
  2. 按帧分组，构建 snapshot dict
  3. 转换为 STKG 张量供 K-HSTGAN 推理
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["FluidAdapter", "load_fluid_chunk", "build_fluid_snapshot",
           "generate_fluid_mock_data"]


class FluidAdapter:
    """
    FLUID 数据集适配器

    支持两种输入格式:
      1. JSON: 每帧一条记录，或一次性全部轨迹列表
      2. Parquet: 列式存储，包含 frame_id, track_id, x, y, vx, vy 等列
    """

    # FLUID 标准字段映射
    FIELD_MAP = {
        "frame_id": ["frame_id", "frame", "t", "timestamp"],
        "track_id": ["track_id", "id", "object_id", "oid"],
        "x": ["x", "center_x", "bbox_xc"],
        "y": ["y", "center_y", "bbox_yc"],
        "vx": ["vx", "vel_x", "velocity_x", "speed_x"],
        "vy": ["vy", "vel_y", "velocity_y", "speed_y"],
        "type": ["type", "class", "category", "object_type"],
    }

    def __init__(self, field_overrides: Optional[Dict[str, str]] = None):
        self.field_map = self.FIELD_MAP.copy()
        if field_overrides:
            for k, v in field_overrides.items():
                if k in self.field_map:
                    self.field_map[k] = v

    def _resolve_field(self, record: dict, field_names: List[str]) -> Any:
        """在记录中查找第一个匹配的字段"""
        for name in field_names:
            if name in record:
                return record[name]
        return None

    def normalize_record(self, record: dict) -> Optional[dict]:
        """将 FLUID 记录标准化为 STKG 所需格式"""
        frame_id = self._resolve_field(record, self.field_map["frame_id"])
        if frame_id is None:
            return None

        x = self._resolve_field(record, self.field_map["x"])
        y = self._resolve_field(record, self.field_map["y"])
        vx = self._resolve_field(record, self.field_map["vx"])
        vy = self._resolve_field(record, self.field_map["vy"])
        obj_type = self._resolve_field(record, self.field_map["type"])

        if x is None or y is None:
            return None

        speed = float(math.hypot(float(vx or 0), float(vy or 0)))
        heading = math.atan2(float(vy or 0), float(vx or 0))

        return {
            "entity_id": str(self._resolve_field(record, self.field_map["track_id"]) or f"obj_{frame_id}"),
            "frame_id": int(frame_id),
            "location_x": float(x),
            "location_y": float(y),
            "location_z": 0.0,
            "velocity_x": float(vx or 0),
            "velocity_y": float(vy or 0),
            "speed": speed,
            "heading_rad": heading,
            "type": str(obj_type or "vehicle"),
        }


def load_fluid_chunk(
    path: str,
    adapter: Optional[FluidAdapter] = None,
) -> List[Dict[str, Any]]:
    """
    加载 FLUID chunk 文件，返回标准化记录列表

    支持:
      - JSON (list 或 dict with 'records' key)
      - Parquet
    """
    p = Path(path)
    adapter = adapter or FluidAdapter()

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = data.get("records", data.get("frames", data.get("tracks", [])))
        else:
            records = data
    elif p.suffix.lower() in (".parquet", ".pq"):
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            records = df.to_dict(orient="records")
        except ImportError:
            raise RuntimeError("pandas/pyarrow not installed for parquet read")
    else:
        raise ValueError(f"Unsupported FLUID format: {p.suffix}")

    normalized = []
    for r in records:
        nr = adapter.normalize_record(r)
        if nr:
            normalized.append(nr)
    return normalized


def build_fluid_snapshot(
    records: List[Dict[str, Any]],
    frame_id: int,
    weather: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    将 FLUID 记录列表构建为 extract_stkg_tensors 所需的 snapshot dict
    """
    vehicles = []
    pedestrians = []

    for r in records:
        typ = r.get("type", "vehicle").lower()
        if typ in ("car", "truck", "bus", "van", "vehicle", "motorcycle"):
            typ = "vehicle"
        elif typ in ("pedestrian", "person", "walker"):
            typ = "pedestrian"
        else:
            typ = "vehicle"

        base = {
            "entity_id": r["entity_id"],
            "location_x": r.get("location_x", 0.0),
            "location_y": r.get("location_y", 0.0),
            "location_z": r.get("location_z", 0.0),
            "velocity_x": r.get("velocity_x", 0.0),
            "velocity_y": r.get("velocity_y", 0.0),
            "speed": r.get("speed", 0.0),
            "heading_rad": r.get("heading_rad", 0.0),
            "is_emergency": r.get("is_emergency", False),
            "current_lane": {
                "road_id": 1,
                "lane_id": 1,
                "center_x": r.get("location_x", 0.0),
                "center_y": r.get("location_y", 0.0),
                "speed_limit": 13.89,
            },
        }

        if typ == "vehicle":
            base.update({
                "location_z": 0.5,
                "brake": 0.0,
                "throttle": 0.5,
                "steer": 0.0,
            })
            vehicles.append(base)
        else:
            base["location_z"] = 0.0
            pedestrians.append(base)

    w = weather or {
        "fog_density": 0.0,
        "cloudiness": 50.0,
        "precipitation": 0.0,
        "wetness": 0.0,
        "sun_altitude_angle": 45.0,
        "wind_intensity": 0.0,
    }

    extracted = {
        "frame_id": frame_id,
        "elapsed_seconds": float(frame_id) * 0.1,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "traffic_lights": [],
        "weather": w,
    }

    return {
        "extracted": extracted,
        "delta": None,
        "rule_out": {"violations": []},
    }


# ============================================================
# 示例: 生成模拟 FLUID 数据
# ============================================================

def generate_fluid_mock_data(output_dir: Path, n_frames: int = 3000, anomaly_rate: float = 0.15) -> Path:
    """生成模拟 FLUID 数据集（用于演示）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for f in range(n_frames):
        n_vehicles = np.random.randint(15, 40)
        n_peds = np.random.randint(2, 8)
        is_anomaly = np.random.rand() < anomaly_rate and f > 100

        for i in range(n_vehicles):
            speed = np.random.uniform(0, 25)
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            records.append({
                "frame_id": f,
                "track_id": f"veh_{f:05d}_{i:03d}",
                "x": np.random.uniform(-500, 500),
                "y": np.random.uniform(-500, 500),
                "vx": vx,
                "vy": vy,
                "type": "vehicle",
            })

        for i in range(n_peds):
            records.append({
                "frame_id": f,
                "track_id": f"ped_{f:05d}_{i:03d}",
                "x": np.random.uniform(-200, 200),
                "y": np.random.uniform(-200, 200),
                "vx": np.random.uniform(-1, 1),
                "vy": np.random.uniform(-1, 1),
                "type": "pedestrian",
            })

    # 按帧分组保存为 JSON chunks
    chunk_size = 500
    for cs in range(0, len(records), chunk_size):
        chunk_records = records[cs:cs + chunk_size]
        frames_set = set(r["frame_id"] for r in chunk_records)
        chunk_file = output_dir / f"fluid_frame_{cs // chunk_size:03d}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump({"records": chunk_records, "n_frames": len(frames_set)}, f)

    metadata = {
        "dataset_name": "FLUID mock",
        "source": "Simulated Florida Urban Intersection",
        "n_records": len(records),
        "n_frames": n_frames,
        "anomaly_rate": anomaly_rate,
        "fps": 10,
        "creation_date": "2026-08-10",
        "creation_note": "Generated by fluid_adapter.generate_fluid_mock_data()",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FLUID 数据集适配器")
    p.add_argument("--generate", action="store_true", help="生成模拟数据")
    p.add_argument("--path", type=str, default="data/fluid_simulated", help="数据目录")
    p.add_argument("--frames", type=int, default=3000, help="帧数")
    args = p.parse_args()

    if args.generate:
        out = Path(__file__).parent.parent.parent / args.path
        print(f"生成 FLUID 模拟数据 {args.frames} 帧 -> {out}")
        generate_fluid_mock_data(out, n_frames=args.frames)
        print("完成")