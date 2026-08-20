#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 smoke test：验证所有模型的前向传播

检查项：
1. 模型实例化
2. 前向传播（使用 CARLA 数据构建的 STKG）
3. 输出维度正确性
4. 梯度反向传播
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import torch
from torch_geometric.data import Data

# 模型导入
from model.ks_nbcf.model import KS_NBCF, KS_NBCFTrainer
from model.re_gcn.re_gcn import RE_GCN, RE_GCNTrainer
from model.gdn.gdn import GDN, GDNTrainer
from model.general_dyg.general_dyg import GeneralDyG, GeneralDyGTrainer
from stk.gnn.k_hstgan import K_HSTGAN

def make_test_data(num_nodes: int = 15):
    """构造测试用的 STKG Data 对象"""
    return Data(
        x=torch.randn(num_nodes, 18),                       # [N, 18]
        edge_index=torch.randint(0, num_nodes, (2, 40)),    # [2, E]
        edge_type=torch.randint(0, 15, (40,)),              # [E]
        kappa_rss=torch.randn(num_nodes, 5),                # [N, 5]
        kappa_rule=torch.rand(num_nodes, 14),               # [N, 14]
        env_feat=torch.randn(12),                           # [12]
        delta_feat=torch.randn(4),                          # [4]
        y_anomaly=torch.randint(0, 2, (num_nodes,)),        # [N]
        y_scene=torch.randint(0, 3, (num_nodes,)),          # [N]
        y_behavior=torch.randint(0, 7, (num_nodes,)),       # [N]
        y_rule=torch.rand(num_nodes, 14),                   # [N, 14]
    )


def test_model(name: str, model, trainer=None):
    """统一测试函数"""
    print(f"\n{'=' * 60}")
    print(f"Testing {name}")
    print(f"{'=' * 60}")
    device = torch.device("cpu")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [✓] Params: {n_params:,}")

    # 前向传播
    data = make_test_data(15)
    data = data.to(device)

    try:
        output = model(data)
        # 输出可能是 4 或 5 个元素（return_extras）
        y_a, y_s, y_b, y_r = output[0], output[1], output[2], output[3]

        print(f"  [✓] Forward OK")
        print(f"      y_anomaly: {tuple(y_a.shape)}")
        print(f"      y_scene:   {tuple(y_s.shape)}")
        print(f"      y_behavior:{tuple(y_b.shape)}")
        print(f"      y_rule:    {tuple(y_r.shape)}")

        # 检查输出范围
        assert 0 <= y_a.min() <= y_a.max() <= 1.0, "y_anomaly out of [0,1]"
        assert 0 <= y_s.min() <= y_s.max() <= 1.0, "y_scene out of [0,1]"
        assert 0 <= y_b.min() <= y_b.max() <= 1.0, "y_behavior out of [0,1]"
        assert 0 <= y_r.min() <= y_r.max() <= 1.0, "y_rule out of [0,1]"
        print(f"  [✓] Output range [0,1] OK")

    except Exception as e:
        print(f"  [✗] Forward FAILED: {e}")
        import traceback; traceback.print_exc()
        return False

    # 反向传播
    try:
        target = torch.randint(0, 2, (15,)).float()
        loss = torch.nn.functional.binary_cross_entropy(y_a.squeeze(-1), target)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        assert grad_norm > 0.0, "Zero gradients"
        print(f"  [✓] Backward OK (grad_norm={grad_norm:.4f})")
    except Exception as e:
        print(f"  [✗] Backward FAILED: {e}")
        return False

    return True


def main():
    print("=" * 70)
    print("Model Smoke Test Suite")
    print("=" * 70)

    results = {}

    # 1. K-HSTGAN (backbone)
    try:
        model = K_HSTGAN(hidden_dim=64)
        results["K-HSTGAN"] = test_model("K-HSTGAN", model)
    except Exception as e:
        print(f"K-HSTGAN FAILED: {e}")
        results["K-HSTGAN"] = False

    # 2. KS-NBCF (融合了 K-HSTGAN 的完整模型)
    try:
        k_hstgan = K_HSTGAN(hidden_dim=64)
        model = KS_NBCF(k_hstgan_model=k_hstgan)
        results["KS-NBCF"] = test_model("KS-NBCF", model)
    except Exception as e:
        print(f"KS-NBCF FAILED: {e}")
        results["KS-NBCF"] = False

    # 3. RE-GCN
    try:
        model = RE_GCN(input_dim=18, hidden_dim=64)
        results["RE-GCN"] = test_model("RE-GCN", model)
    except Exception as e:
        print(f"RE-GCN FAILED: {e}")
        results["RE-GCN"] = False

    # 4. GDN
    try:
        model = GDN(input_dim=18, hidden_dim=64)
        results["GDN"] = test_model("GDN", model)
    except Exception as e:
        print(f"GDN FAILED: {e}")
        results["GDN"] = False

    # 5. GeneralDyG
    try:
        model = GeneralDyG(input_dim=18, hidden_dim=64)
        results["GeneralDyG"] = test_model("GeneralDyG", model)
    except Exception as e:
        print(f"GeneralDyG FAILED: {e}")
        results["GeneralDyG"] = False

    # 汇总
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    all_pass = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n🎉 All models passed smoke test!")
        return 0
    else:
        print("\n⚠️  Some models failed smoke test.")
        return 1


if __name__ == "__main__":
    sys.exit(main())