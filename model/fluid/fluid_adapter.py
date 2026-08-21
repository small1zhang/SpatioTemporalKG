#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUID 数据集适配器
Fluid Urban Intersection Dataset Adapter

官方数据集结构 (figshare 29974954):
 - TrackData/: 轨迹数据 (所有车辆/行人的逐帧位置信息)
 - TrafficLights/: 信号灯状态数据
 - Map/: 道路几何信息 (GeoJSON)
 - Metadata/: 数据集元数据

典型 CSV 字段:
 - frame_id: 帧编号 (从 0 开始)
 - track_id: 目标追踪ID
 - obj_type: 对象类型 (vehicle/pedestrian)
 - x, y: 轨迹中心坐标 (相对于路口原点，米)
 - vx, vy: 速度分量 (m/s)
 - ax, ay: 加速度分量 (m/s²)
 - bbox_x1, bbox_y1, bbox_x2, bbox_y2: 边界框坐标
 - width, height: 目标尺寸
 - heading: 行驶方向角度 (弧度)
 - length, width: 车辆尺寸 (米)
 - track_id: 长期目标ID

文件布局:
 - TrackData/vehicle_tracks.csv (或 vehicle_tracks.parquet)
 - TrackData/pedestrian_tracks.csv  
 - TrafficLights/states.csv (红绿灯状态随时间变化)
 - Map/road_layout.geojson (道路网格)
 - Metadata/dataset_info.json (采集参数)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FluidFieldMapper:
    """FLUID 字段映射管理类"""
    # 所有可能的字段名列表 (官方数据集使用这些)
    FIELD_NAMES = {
        "frame_id": ["frame_id", "frame", "tick", "time_step"],
        "track_id": ["track_id", "object_id", "obj_id", "track_id"],
        "obj_type": ["obj_type", "object_type", "type", "category"],
        "x": ["x", "pos_x", "x_center", "longitude", "coord_x"],
        "y": ["y", "pos_y", "y_center", "latitude", "coord_y"],
        "vx": ["vx", "vel_x", "velocity_x", "speed_x"],
        "vy": ["vy", "vel_y", "velocity_y", "speed_y"],
        "ax": ["ax", "accel_x", "acceleration_x"],
        "ay": ["ay", "accel_y", "acceleration_y"],
        "bbox_x1": ["bbox_x1", "x_min", "left"],
        "bbox_y1": ["bbox_y1", "y_min", "bottom"],
        "bbox_x2": ["bbox_x2", "x_max", "right"],
        "bbox_y2": ["bbox_y2", "y_max", "top"],
        "width": ["width", "width_m", "width_meters"],
        "height": ["height", "height_m", "height_meters"],
        "heading": ["heading", "yaw", "direction", "angle"],
        "length": ["length", "length_m", "vehicle_length"],
    }

    def __init__(self, field_overrides: Optional[Dict[str, List[str]]] = None):
        self.map = self.FIELD_NAMES.copy()
        if field_overrides:
            for k, v in field_overrides.items():
                if k in self.map:
                    self.map[k] = v

    def resolve(self, record: dict, field: str) -> Any:
        """在记录中查找字段，返回第一个匹配值"""
        for name in self.map.get(field, []):
            if name in record and record[name] is not None:
                return record[name]
        return None


# ============================================================
# 核心适配器类
# ============================================================

class FluidAdapter:
    """
    FLUID 数据集适配器

    将 FLUID 数据集转换为 stk/gnn/exporter.py 所需的 extract_stkg_tensors
    兼容的 snapshot dict 格式。

    支持两种模式:
      1. 逐帧模式：每帧单独一个 CSV 文件或 dict
      2. 批量模式：所有帧的数据合并在一个大 CSV/Parquet 中
    """

    def __init__(self, field_overrides: Optional[Dict[str, List[str]]] = None):
        self.mapper = FluidFieldMapper(field_overrides)

    def normalize_record(self, record: dict) -> Optional[dict]:
        """
        将 FLUID 原始记录标准化为内部格式

        返回格式:
        {
            "entity_id": str,          # 对象唯一标识
            "frame_id": int,           # 帧编号
            "location_x": float,       # x 坐标 (米)
            "location_y": float,       # y 坐标 (米)
            "location_z": float,       # 高度/0
            "velocity_x": float,       # vx (m/s)
            "velocity_y": float,       # vy (m/s)
            "speed": float,            # 速度模 sqrt(vx²+vy²)
            "heading_rad": float,      # 行驶方向角度
            "type": str,               # 'vehicle' 或 'pedestrian'
            "bbox": Tuple[float, float, float, float],  # (x1, y1, x2, y2)
            "length": Optional[float], # 车辆长度 (米)
            "width": Optional[float],  # 车辆宽度 (米)
        }
        返回 None 表示记录无效（缺少关键字段）
        """
        frame_id = self.mapper.resolve(record, "frame_id")
        if frame_id is None:
            return None

        obj_type = self.mapper.resolve(record, "obj_type")
        if obj_type is None:
            return None

        # 统一类型名称
        type_lower = obj_type.lower()
        if type_lower in ("vehicle", "car", "truck", "bus", "van", "motorcycle"):
            fluid_type = "vehicle"
        elif type_lower in ("pedestrian", "person", "walker"):
            fluid_type = "pedestrian"
        else:
            # 默认当作 vehicle 处理，但标记异常
            fluid_type = "vehicle"

        # 坐标转换 (FLUID 坐标系: 原点在路口中心，x向东, y向北)
        x = float(self.mapper.resolve(record, "x") or 0.0)
        y = float(self.mapper.resolve(record, "y") or 0.0)

        # 速度分量
        vx = float(self.mapper.resolve(record, "vx") or 0.0)
        vy = float(self.mapper.resolve(record, "vy") or 0.0)
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0.0

        # 边界框
        bbox_x1 = float(self.mapper.resolve(record, "bbox_x1") or x - 2.0)
        bbox_y1 = float(self.mapper.resolve(record, "bbox_y1") or y - 2.0)
        bbox_x2 = float(self.mapper.resolve(record, "bbox_x2") or x + 2.0)
        bbox_y2 = float(self.mapper.resolve(record, "bbox_y2") or y + 2.0)

        return {
            "entity_id": str(self.mapper.resolve(record, "track_id") or f"fluid_{record.get('frame_id', -1)}_{record.get('obj_type', 'obj')}"),
            "frame_id": int(frame_id),
            "location_x": x,
            "location_y": y,
            "location_z": 0.0,
            "velocity_x": vx,
            "velocity_y": vy,
            "speed": speed,
            "heading_rad": heading,
            "type": fluid_type,
            "bbox": (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
            "length": float(self.mapper.resolve(record, "length") or 4.5),
            "width": float(self.mapper.resolve(record, "width") or 2.0),
        }

    def load_csv_chunk(self, csv_path: Path) -> List[dict]:
        """从单个 CSV 文件加载一批记录"""
        records = []
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    norm = self.normalize_record(row)
                    if norm:
                        records.append(norm)
        except Exception as e:
            print(f"Warning: Error reading {csv_path}: {e}")
        return records

    def load_parquet_chunk(self, pq_path: Path) -> List[dict]:
        """从 Parquet 文件加载记录 (需要 pandas)"""
        try:
            import pandas as pd
            df = pd.read_parquet(pq_path)
            records = []
            for _, row in df.iterrows():
                record = self.normalize_record(row.to_dict())
                if record:
                    records.append(record)
            return records
        except ImportError:
            raise RuntimeError("pandas not installed, cannot read parquet")
        except Exception as e:
            print(f"Warning: Error reading {pq_path}: {e}")
            return []

    # ============================================================
    # 快照构建函数
    # ============================================================

    def build_snapshot(
        self,
        records: List[dict],
        frame_id: int,
        weather: Optional[dict] = None,
    ) -> dict:
        """
        将 FLUID 记录列表构建为 extract_stkg_tensors 所需的 snapshot dict

        快照 dict 格式要求:
        {
            "extracted": {
                "frame_id": int,
                "elapsed_seconds": float,
                "vehicles": [{entity_id, location_x, ...}, ...],
                "pedestrians": [{entity_id, location_x, ...}, ...],
                "traffic_lights": [...],
                "weather": {...},
            },
            "delta": None | DeltaGraph,
            "rule_out": {"violations": [...], "responsibilities": [...]},
        }
        """
        vehicles = []
        pedestrians = []
        tl_list = []

        for r in records:
            # 跳过无效记录
            if r is None:
                continue

            # 根据类型分发
            if r["type"] == "vehicle":
                vehicles.append(r)
            elif r["type"] == "pedestrian":
                pedestrians.append(r)
            else:
                # 未知类型，默认 vehicle 但标记
                vehicles.append(r)

            # 提取信号灯信息 (如果有)
            if "state" in r and r["state"] is not None:
                tl_list.append({
                    "entity_id": f"tl_{r['frame_id']}",
                    "state": str(r["state"]).lower(),
                    "location_x": r.get("location_x", 0.0),
                    "location_y": r.get("location_y", 0.0),
                })

        # 天气默认值 (FLUID 数据集常见天气字段)
        w = weather or {
            "fog_density": float(self.mapper.resolve({}, "fog_density") or 0.0),
            "cloudiness": float(self.mapper.resolve({}, "cloudiness") or 50.0),
            "precipitation": float(self.mapper.resolve({}, "precipitation") or 0.0),
            "wetness": float(self.mapper.resolve({}, "wetness") or 0.0),
            "sun_altitude_angle": float(self.mapper.resolve({}, "sun_altitude_angle") or 45.0),
            "wind_intensity": float(self.mapper.resolve({}, "wind_intensity") or 0.0),
        }

        extracted = {
            "frame_id": int(frame_id),
            "elapsed_seconds": float(frame_id) * 0.1,  # FLUID 典型采样间隔 0.1s
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "traffic_lights": tl_list,
            "weather": w,
        }

        return {
            "extracted": extracted,
            "delta": None,  # FLUID 通常没有显式的 delta 特征
            "rule_out": {"violations": []},  # 初始无违规记录
        }

    # ============================================================
    # 批量处理工具
    # ============================================================

    def group_by_frame(self, records: List[dict]) -> dict:
        """将记录按 frame_id 分组"""
        frames: dict = {}
        for r in records:
            fid = r.get("frame_id", -1)
            if fid not in frames:
                frames[fid] = []
            frames[fid].append(r)
        return frames

    def process_dataset_dir(
        self,
        dataset_dir: Path,
    ) -> dict:
        """
        处理完整的 FLUID 数据集目录

        预期目录结构:
        dataset_dir/
        ├── TrackData/
        │   ├── vehicle_tracks.csv
        │   └── pedestrian_tracks.csv
        ├── TrafficLights/
        │   └── states.csv
        ├── Map/
        │   └── road_layout.geojson
        └── Metadata/
            └── dataset_info.json
        """
        all_records = []

        # 1. 加载轨迹数据
        track_dir = dataset_dir / "TrackData"
        if track_dir.exists():
            for csv_file in track_dir.glob("*.csv"):
                records.extend(self.load_csv_chunk(csv_file))

        # 2. 加载信号灯数据
        tl_dir = dataset_dir / "TrafficLights"
        tl_states = {}
        if tl_dir.exists():
            for csv_file in tl_dir.glob("*.csv"):
                with open(csv_file, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fid = int(row.get("frame_id", 0))
                        state = row.get("state", "green")
                        if fid not in tl_states:
                            tl_states[fid] = []
                        tl_states[fid].append(state)

        # 3. 构建每一帧的 snapshot
        frames_dict = self.group_by_frame(records)
        snapshots = {}

        for frame_id in sorted(frames_dict.keys()):
            recs = frames_dict[frame_id]
            weather = {"fog_density": 0.0, "cloudiness": 50.0, "precipitation": 0.0}
            # 如果有天气信息则合并
            if "Metadata" in str(dataset_dir) or "weather" in str(dataset_dir):
                # 这里可以根据需要从 metadata 文件读取天气
                pass

            snapshot = self.build_snapshot(recs, frame_id, weather)
            snapshots[frame_id] = snapshot

        return snapshots


# ============================================================
# 模拟数据生成器 (用于演示/测试)
# ============================================================

def generate_fluid_mock_data(
    output_dir: Path,
    n_frames: int = 3000,
    n_vehicles_per_frame: int = 25,
    n_peds_per_frame: int = 8,
    anomaly_rate: float = 0.15,
) -> dict:
    """
    生成模拟 FLUID 数据集 (演示用)

    生成结构:
    - output_dir/fluid_tracks.csv: 轨迹数据
    - output_dir/fluid_metadata.json: 元数据

    每帧生成 n_vehicles + n_peds 个对象，其中约 anomaly_rate 比例有冲突/异常行为.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)
    np.random.seed(42)

    records = []
    anomaly_counter = 0

    for f in range(n_frames):
        # 每帧车辆/行人数量略微变化
        n_v = n_vehicles_per_frame + random.randint(-5, 5)
        n_p = n_peds_per_frame + random.randint(-3, 3)

        for i in range(n_v):
            # 生成合理的轨迹: 车辆在路口附近移动
            # 框架中心在 (0,0)，道路大致沿 x 方向
            x = np.random.uniform(-150, 150)
            y = np.random.uniform(-80, 80)

            # 速度: 大部分车辆速度在 5-20 m/s 之间
            speed = np.random.uniform(5, 25)
            # 方向略微偏向东西方向
            angle = np.random.uniform(-math.pi / 4, math.pi / 4)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            is_anom = (
                anomaly_rate > 0
                and random.random() < anomaly_rate
                and f > 100
            )  # 前 100 帧正常

            anomaly_type = ""
            if is_anom:
                anomaly_type = random.choice([" sudden_brake", " red_light_runaway", " lane_change_unsafe"])
                anomaly_counter += 1

            record = {
                "frame_id": f,
                "track_id": f"veh_{f:05d}_{i:03d}",
                "obj_type": "vehicle",
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
                "ax": float(np.random.uniform(-3, 3)),
                "ay": float(np.random.uniform(-3, 3)),
                "bbox_x1": float(x - 2.5),
                "bbox_y1": float(y - 1.5),
                "bbox_x2": float(x + 2.5),
                "bbox_y2": float(y + 1.5),
                "width": float(np.random.uniform(1.8, 2.5)),
                "height": float(np.random.uniform(4.0, 5.5)),
                "heading": float(np.random.uniform(-math.pi, math.pi)),
                "length": float(np.random.uniform(4.0, 5.5)),
                "height_vehicle": float(np.random.uniform(1.5, 2.0)),
                "state": "normal",  # 或 "anomaly"
            }
            records.append(record)

        for i in range(n_peds_per_frame):
            x = np.random.uniform(-80, 80)
            y = np.random.uniform(-50, 50)

            speed = np.random.uniform(1, 3)
            angle = np.random.uniform(-math.pi / 2, math.pi / 2)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            is_anom = anomaly_rate > 0 and random.random() < anomaly_rate

            record = {
                "frame_id": f,
                "track_id": f"ped_{f:05d}_{i:03d}",
                "obj_type": "pedestrian",
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
                "ax": float(np.random.uniform(-2, 2)),
                "ay": float(np.random.uniform(-2, 2)),
                "bbox_x1": float(x - 0.5),
                "bbox_y1": float(y - 0.5),
                "bbox_x2": float(x + 0.5),
                "bbox_y2": float(y + 0.5),
                "width": float(np.random.uniform(0.3, 0.8)),
                "height": float(np.random.uniform(1.5, 1.8)),
                "heading": float(np.random.uniform(-math.pi, math.pi)),
                "length": float(np.random.uniform(0.5, 1.0)),
                "state": "normal" if not is_anom else "anomaly",
            }
            records.append(record)

    # 保存 CSV
    csv_path = output_dir / "fluid_tracks.csv"
    fieldnames = [
        "frame_id", "track_id", "obj_type", "x", "y", "vx", "vy", "ax", "ay",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "width", "height",
        "heading", "length", "state",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            # 规范化字段名
            row = {fn: r.get(fn.lower().replace(" ", "_").replace(" ", "")) for fn in fieldnames}
            writer.writerow(row)

    # 保存元数据
    metadata = {
        "dataset_name": "FLUID Mock Dataset",
        "n_frames": n_frames,
        "n_vehicles_per_frame": n_vehicles_per_frame,
        "n_peds_per_frame": n_peds_per_frame,
        "anomaly_rate": anomaly_rate,
        "total_records": len(records),
        "fps": 10,  # 帧率
        "coord_system": "origin at intersection center, x east, y north",
        "creation_date": "2026-08-10",
        "notes": "Generated by FluidAdapter.generate_fluid_mock_data() for testing",
    }
    with open(output_dir / "fluid_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated mock FLUID dataset: {n_frames} frames, {len(records)} records")
    print(f"  Anomaly rate: {anomaly_rate*100:.1f}% ({anomaly_counter} anomalous records)")
    return {"csv_path": csv_path, "metadata": metadata}


# ============================================================
# 主程序入口
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="FLUID 数据集适配器")
    p.add_argument(
        "--mode",
        choices=["adapt", "mock", "test"],
        default="mock",
        help="Mode: adapt (adapt real data), mock (generate mock data), test (test adapter)",
    )
    p.add_argument("--data-dir", type=str, default=None, help="真实 FLUID 数据目录路径")
    p.add_argument("--output", type=str, default="data/fluid_simulated", help="输出目录")
    p.add_argument("--frames", type=int, default=3000, help="模拟帧数")
    args = p.parse_args()

    adapter = FluidAdapter()

    if args.mode == "mock":
        print(f"生成模拟 FLUID 数据 {args.frames} 帧 -> {args.output}")
        generate_fluid_mock_data(Path(args.output), n_frames=args.frames)
    elif args.mode == "adapt":
        data_dir = Path(args.data_dir) if args.data_dir else Path("data/fluid_dataset")
        if not data_dir.exists():
            print(f"数据目录不存在: {data_dir}")
            print("使用模拟模式代替...")
            generate_fluid_mock_data(Path(args.output))
        else:
            print(f"适配真实 FLUID 数据: {data_dir}")
            snapshots = adapter.process_dataset_dir(data_dir)
            print(f"成功适配 {len(snapshots)} 帧")
            # 保存第一个 snapshot 为演示
            if snapshots:
                first_fid = sorted(snapshots.keys())[0]
                with open(Path(args.output) / "first_snapshot.json", "w") as f:
                    json.dump(snapshots[first_fid], f, indent=2)
                print(f"已保存第一帧快照: frame {first_fid}")
    elif args.mode == "test":
        # 测试 adapter 基本功能
        print("测试 FluidAdapter 基本功能...")
        # 创建一个测试记录
        test_rec = {
            "frame_id": 0,
            "track_id": "test_vehicle_0",
            "obj_type": "vehicle",
            "x": 10.5,
            "y": 5.0,
            "vx": 5.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "bbox_x1": 8.0,
            "bbox_y1": 3.0,
            "bbox_x2": 13.0,
            "bbox_y2": 8.0,
            "width": 2.0,
            "height": 4.5,
            "heading": 0.0,
            "length": 5.0,
        }
        norm = adapter.normalize_record(test_rec)
        if norm:
            print(f"  标准化记录: {norm['entity_id']}, frame={norm['frame_id']}")
            snap = adapter.build_snapshot([norm], frame_id=0)
            print(f"  快照建立: {len(snap['extracted']['vehicles'])} 边车, {len(snap['extracted']['pedestrians'])} 行人")
        else:
            print("  记录标准化失败")
        print("  测试通过!")


if __name__ == "__main__":
    main()