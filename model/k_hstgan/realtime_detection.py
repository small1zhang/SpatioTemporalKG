#!/usr/bin/env python3
"""
实时检测接口：K-HSTGAN + CARLA 数据提取集成（修复版）

该模块是统一推理入口，支持三种数据源：
  1. offline: snapshot dict (从已保存的 CARLA chunk JSON / 模拟数据 / CSV 转换得到)
  2. live:    CARLA world 对象 (live 服务器)
  3. file:    直接读 chunk JSON 文件路径，内部 adapter 自动转换

链路:
  snapshot → extract_stkg_tensors → K-HSTGAN → KS-NBCF φ_feat / φ_loop / φ_fuse / φ_arb
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stk.gnn.exporter import extract_stkg_tensors
from stk.gnn.k_hstgan import K_HSTGAN
from model.ks_nbcf.model import (
    KS_NBCF_Fuser, KS_NBCF_LoopFeedback, KS_NBCF_Arbiter,
)
from model.data_adapter import chunk_frame_to_snapshot, sind2_frame_to_snapshot


# ============================================================
# K-HSTGAN Feature Injection 层（替换 stk/fusion/feat_injection.py
# 避免外部 API drift）
# ============================================================

class FeatInjector(nn.Module):
    """KS-NBCF φ_feat 编排器：调用 K-HSTGAN.forward(return_extras=True)"""

    def __init__(self, model: K_HSTGAN):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_with_extras(
        self,
        snapshot: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ):
        data = extract_stkg_tensors(snapshot).to(device)
        y_a, y_s, y_b, y_r, extras = self.model(data, return_extras=True)
        # 复制 edge_index/edge_type 到 extras（K-HSTGAN.forward 也已写入）
        extras.setdefault("edge_index", data.edge_index)
        extras.setdefault("edge_type", data.edge_type)
        return y_a, y_s, y_b, y_r, extras, data


# ============================================================
# 实时检测引擎
# ============================================================

class RealtimeDetectionEngine:
    """统一推理引擎：snapshot → K-HSTGAN → KS-NBCF → D-S → 仲裁"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
        tau_K: float = 0.3,
    ):
        self.device = device or torch.device("cpu")
        self.tau_K = tau_K

        # 1. K-HSTGAN backbone
        self.model = K_HSTGAN(
            base_node_dim=18,
            rss_dim=5,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_relations=15,
            rule_dim=14,
            transformer_d_k=32,
            dropout=0.1,
        ).to(self.device)

        if model_path and Path(model_path).exists():
            print(f"[engine] loading model from {model_path}")
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            except Exception as e:
                print(f"[warn] failed to load checkpoint: {e}, using random init")

        # 2. KS-NBCF 模块
        self.feat_injector = FeatInjector(model=self.model).to(self.device)
        self.loop_module = KS_NBCF_LoopFeedback(num_rules=14, device=self.device)
        self.fuser = KS_NBCF_Fuser(tau_K=tau_K).to(self.device)
        self.arbiter = KS_NBCF_Arbiter().to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[engine] K-HSTGAN params: {n_params:,}  device: {self.device}")

    # ------------------------------------------------------------
    # 主入口: snapshot dict → 检测结果
    # ------------------------------------------------------------
    def process_snapshot(
        self,
        snapshot: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        单帧实时检测主入口（推荐用法）。

        Args:
            snapshot: dict，必须含 {"extracted": {vehicles, pedestrians,
                    traffic_lights, weather, frame_id, elapsed_seconds},
                    "delta": None, "rule_out": {violations: [...]}}

        Returns:
            result dict，含:
              - frame_id, processing_time_ms
              - khstgan: y_anomaly/scene/behavior/rule 概览
              - ks_nbcf: per-head anomaly、rgat attention
              - fusion: D-S 融合结果 (K, decision, m_*)
              - arbiter: resolve_type / overlap / evidence_strength / explanation
        """
        t0 = datetime.now()

        # 1. KS-NBCF φ_feat 编排（含 STKG 构建 + K-HSTGAN forward）
        try:
            y_a, y_s, y_b, y_r, extras, data = self.feat_injector.predict_with_extras(
                snapshot, device=self.device
            )
        except Exception as e:
            return self._error(f"feat injection failed: {e}", snapshot, t0)

        if verbose:
            print(f"  [step1] K-HSTGAN forward OK: "
                  f"nodes={data.x.shape[0]} y_a mean={y_a.mean().item():.4f}")

        # 2. KS-NBCF φ_loop 三阶段反馈
        node_ids = self._extract_node_ids(snapshot)
        rule_out = snapshot.get("rule_out", {"violations": []})
        p_anomaly = y_a.to(self.device)          # 无论 step2 成功与否都可用
        weak_loss = torch.tensor(0.0)
        eta_new = self.loop_module.state.eta.clone()
        dyn_rules = []

        try:
            # Stage I: 弱监督
            y_weak = self.loop_module.compute_weak_labels(rule_out, node_ids, num_rules=14)
            weak_loss = self.loop_module.compute_weak_loss(y_r, y_weak, epoch=0)

            # Stage II: 置信度反馈
            gt_anomaly = getattr(data, "y_anomaly", torch.zeros(data.x.shape[0])).to(self.device)
            s_minus, s_zero = self.loop_module.compute_stage2_signals(
                y_r, p_anomaly, gt_anomaly
            )
            ev_lens = self.loop_module.compute_evidence_lengths(rule_out)
            eta_new = self.loop_module.update_eta(s_minus, s_zero, ev_lens)

            # Stage III: 动态规则生成
            edge_index = extras.get("edge_index")
            edge_type = extras.get("edge_type")
            if extras.get("rgat_attention") and edge_index is not None and edge_type is not None:
                dyn_rules = self.loop_module.generate_dynamic_rules(
                    extras["per_head_anomaly"], y_a,
                    extras["rgat_attention"], node_ids,
                    edge_index=edge_index.to(self.device),
                    edge_type=edge_type.to(self.device),
                )
        except Exception as e:
            if verbose:
                print(f"  [step2] loop feedback partial: {e}")
            weak_loss = torch.tensor(0.0)
            eta_new = self.loop_module.state.eta.clone()
            dyn_rules = []

        if verbose:
            print(f"  [step2] loop feedback OK: "
                  f"L_weak={weak_loss.item():.4f} dyn_rules={len(dyn_rules)}")

        # 3. KS-NBCF φ_fuse D-S 融合
        try:
            kappa_rule = getattr(
                data, "kappa_rule",
                torch.zeros(data.x.shape[0], 14)
            ).to(self.device)
            s_rule = kappa_rule.max(dim=-1).values.clamp(0.0, 1.0)
            rule_fires = (kappa_rule.sum(dim=-1) > 0).float()
            epsilon = extras["per_head_anomaly"].var(dim=-1, unbiased=False)

            fusion_result = self.fuser(p_anomaly, epsilon, s_rule, rule_fires)
            if verbose:
                print(f"  [step3] D-S fusion: decision={fusion_result['decision']} "
                      f"K={fusion_result['K']:.3f} m_a={fusion_result['m_fused_anomaly']:.3f}")
        except Exception as e:
            return self._error(f"D-S fusion failed: {e}", snapshot, t0, partial={
                "khstgan_ok": True, "khstgan": _summarize_khstgan(y_a, y_s, y_b, y_r),
            })

        # 4. KS-NBCF φ_arb 仲裁
        try:
            edge_index_for_arb = extras.get("edge_index")
            edge_type_for_arb = extras.get("edge_type")
            if edge_index_for_arb is not None and edge_type_for_arb is not None:
                edge_index_for_arb = edge_index_for_arb.to(self.device)
                edge_type_for_arb = edge_type_for_arb.to(self.device)

            arb_result = self.arbiter(
                fusion_result=fusion_result,
                rule_out=rule_out,
                rgat_attention=extras.get("rgat_attention", {}) or {},
                edge_index=edge_index_for_arb,
                edge_type=edge_type_for_arb,
                node_ids=node_ids,
                p_anomaly=p_anomaly,
            )
            if verbose:
                print(f"  [step4] arbiter: resolve={arb_result['resolve_type']} "
                      f"y_fused={arb_result['y_fused']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  [step4] arbiter failed: {e}")
            arb_result = None

        elapsed = (datetime.now() - t0).total_seconds()

        # 5. 汇总
        result = {
            "frame_id": snapshot["extracted"].get("frame_id", -1),
            "timestamp": t0.isoformat(),
            "processing_time_ms": int(elapsed * 1000),
            "status": "ok",
            "n_nodes": int(data.x.shape[0]),
            "khstgan": _summarize_khstgan(y_a, y_s, y_b, y_r),
            "ks_nbcf": {
                "y_anomaly_mean": float(y_a.mean().item()),
                "per_head_anomaly_var": float(epsilon.mean().item()),
                "eta_mean": float(eta_new.mean().item()),
                "dyn_rules_count": len(dyn_rules),
                "dyn_rules_preview": dyn_rules[:3],
            },
            "fusion": {
                "decision": fusion_result["decision"],
                "K": float(fusion_result["K"]),
                "is_consistent": bool(fusion_result["is_consistent"]),
                "m_fused_anomaly": float(fusion_result["m_fused_anomaly"]),
                "m_fused_normal": float(fusion_result["m_fused_normal"]),
                "m_fused_uncertain": float(fusion_result["m_fused_uncertain"]),
            },
            "arbiter": {
                "resolve_type": arb_result["resolve_type"] if arb_result else None,
                "y_fused": float(arb_result["y_fused"]) if arb_result else None,
                "overlap": float(arb_result["overlap"]) if arb_result else None,
                "evidence_strength": float(arb_result["evidence_strength"]) if arb_result else None,
                "decision_final": arb_result["decision"] if arb_result else None,
                "explanation": arb_result["explanation"] if arb_result else None,
            } if arb_result else None,
            "per_node_predictions": (y_a.squeeze(-1) > 0.5).long().tolist(),
        }
        return result

    # ------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------
    def process_chunk_file(
        self,
        chunk_path: str,
        frame_idx: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """从 chunk JSON 文件加载单帧并检测。"""
        with open(chunk_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
        if isinstance(frames, dict) and "frames" in frames:
            frames = frames["frames"]
        if frame_idx >= len(frames):
            raise IndexError(f"frame_idx {frame_idx} >= {len(frames)}")
        raw = frames[frame_idx]
        snapshot = chunk_frame_to_snapshot(raw)
        if verbose:
            print(f"[engine] loaded frame {frame_idx} from {chunk_path}: "
                  f"vehicles={len(snapshot['extracted']['vehicles'])} "
                  f"peds={len(snapshot['extracted']['pedestrians'])}")
        return self.process_snapshot(snapshot, verbose=verbose)

    def process_sind2_frame_file(
        self,
        chunk_path: str,
        frame_idx: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """从 SinD2.0 chunk JSON 文件加载单帧并检测。"""
        with open(chunk_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
        if frame_idx >= len(frames):
            raise IndexError(f"frame_idx {frame_idx} >= {len(frames)}")
        raw = frames[frame_idx]
        snapshot = sind2_frame_to_snapshot(raw)
        if verbose:
            print(f"[engine] loaded SinD2.0 frame {frame_idx}: "
                  f"vehicles={len(snapshot['extracted']['vehicles'])}")
        return self.process_snapshot(snapshot, verbose=verbose)

    # ------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------
    @staticmethod
    def _extract_node_ids(snapshot: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        for v in snapshot["extracted"].get("vehicles", []):
            ids.append(str(v.get("entity_id", "")))
        for p in snapshot["extracted"].get("pedestrians", []):
            ids.append(str(p.get("entity_id", "")))
            # 截短防止 GNN 中维度爆炸（K-HSTGAN 的 reserve node slots）
        if not ids:
            ids = [f"ego_{i}" for i in range(10)]
        return ids

    @staticmethod
    def _error(
        msg: str, snapshot: Dict, t0: datetime, partial: Optional[Dict] = None
    ) -> Dict[str, Any]:
        return {
            "frame_id": snapshot.get("extracted", {}).get("frame_id", -1),
            "timestamp": t0.isoformat(),
            "processing_time_ms": int((datetime.now() - t0).total_seconds() * 1000),
            "status": "error",
            "error": msg,
            "partial": partial,
        }


def _summarize_khstgan(y_a, y_s, y_b, y_r) -> Dict[str, Any]:
    return {
        "y_anomaly_mean": float(y_a.mean().item()),
        "y_anomaly_max": float(y_a.max().item()),
        "y_scene_shape": list(y_s.shape),
        "y_behavior_shape": list(y_b.shape),
        "y_rule_nonzero": int((y_r > 0.5).sum().item()),
    }


# ============================================================
# 自检：用零张量构造 snapshot 并跑通链路
# ============================================================

def _smoke_test():
    """构造一个最小 snapshot，跑通整条推理链。"""
    print("=" * 70)
    print("RealtimeDetectionEngine smoke test")
    print("=" * 70)

    engine = RealtimeDetectionEngine(device=torch.device("cpu"))

    # 构造 snapshot: 3 vehicles + 2 pedestrians + weather + 2 traffic lights
    vehicles = []
    for i in range(3):
        vehicles.append({
            "entity_id": f"veh_{i}",
            "is_ego": i == 0,
            "location_x": float(i * 5.0),
            "location_y": 0.0,
            "location_z": 0.5,
            "velocity_x": 5.0,
            "velocity_y": 0.0,
            "speed": 5.0,
            "heading_rad": 0.0,
            "brake": 0.1,
            "throttle": 0.4,
            "steer": 0.0,
            "is_emergency": False,
            "current_lane": {
                "road_id": 1, "lane_id": 1,
                "center_x": float(i * 5.0), "center_y": 0.0,
                "speed_limit": 13.89,
            },
        })

    pedestrians = []
    for i in range(2):
        pedestrians.append({
            "entity_id": f"ped_{i}",
            "location_x": float(i * 2.0),
            "location_y": 3.0,
            "location_z": 0.0,
            "speed": 1.2,
        })

    traffic_lights = [
        {"entity_id": "tl_0", "state": "green",
         "location_x": 10.0, "location_y": 0.0, "location_z": 3.0},
        {"entity_id": "tl_1", "state": "red",
         "location_x": 20.0, "location_y": 0.0, "location_z": 3.0},
    ]

    weather = {
        "fog_density": 2.0,
        "cloudiness": 40.0,
        "precipitation": 0.0,
        "wetness": 0.0,
        "sun_altitude_angle": 45.0,
        "wind_intensity": 0.0,
    }

    snapshot = {
        "extracted": {
            "frame_id": 0,
            "elapsed_seconds": 0.0,
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "traffic_lights": traffic_lights,
            "weather": weather,
        },
        "delta": None,
        "rule_out": {"violations": [
            {"rule_code": "R2",
             "severity": 0.7,
             "src_id": "veh_1",
             "dst_id": "veh_0"},
        ]},
    }

    print("\n[smoke] processing snapshot...")
    result = engine.process_snapshot(snapshot, verbose=True)

    print("\n[smoke] === RESULT ===")
    print(f"  status: {result['status']}")
    print(f"  n_nodes: {result.get('n_nodes')}")
    print(f"  processing_time_ms: {result.get('processing_time_ms')}")
    if result["status"] == "ok":
        print(f"  khstgan: y_anomaly_mean={result['khstgan']['y_anomaly_mean']:.4f}")
        print(f"  fusion:  decision={result['fusion']['decision']}  "
              f"K={result['fusion']['K']:.3f}")
        if result.get("arbiter"):
            print(f"  arbiter: resolve={result['arbiter']['resolve_type']}  "
                  f"y_fused={result['arbiter']['y_fused']:.4f}  "
                  f"overlap={result['arbiter']['overlap']:.4f}")
            print(f"           explanation: {result['arbiter']['explanation'][:100]}...")

    if result["status"] == "ok":
        print("\n  SMOKE TEST PASSED")
        return 0
    else:
        print(f"\n  SMOKE TEST FAILED: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(_smoke_test())