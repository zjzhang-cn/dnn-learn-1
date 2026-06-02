"""
DNN 符号分类器 - 模型剪枝
=========================
加载已训练模型，通过 L1 非结构化剪枝移除不重要的权重，
对比剪枝前后的准确率和稀疏度。

用法：
    python prune.py                # 默认剪掉 50% 权重
    python prune.py --amount 0.3   # 剪掉 30%
    python prune.py --amount 0.9   # 剪掉 90%，测试极端情况
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

from model import DEVICE, BitChecker, int_to_bits
from train import evaluate, generate_validation_data

MODEL_PATH = "sign_classifier.pth"
PRUNED_PATH = "sign_classifier_pruned.pth"


# ============================================================
#  剪枝
# ============================================================

def apply_pruning(model: nn.Module, amount: float):
    """
    对模型所有 Linear 层应用 L1 非结构化剪枝。

    每个权重矩阵中，绝对值最小的 `amount` 比例的权重被置零。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def remove_pruning_reparam(model: nn.Module):
    """
    将剪枝后的 mask 固化到权重中，移除 hook。

    这样模型可以正常保存/加载，不再依赖 prune 的 reparameterization。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # 该层没有被剪枝，跳过
    return model


# ============================================================
#  稀疏度统计
# ============================================================

def sparsity_report(model: nn.Module) -> dict:
    """统计模型每层和整体的稀疏度（零权重比例）。"""
    total_zeros = 0
    total_params = 0
    per_layer = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.cpu().numpy()
            zeros = int((w == 0).sum())
            params = int(w.size)
            sparsity = zeros / params
            per_layer[name] = {
                "shape": list(w.shape),
                "zeros": zeros,
                "params": params,
                "sparsity": sparsity,
            }
            total_zeros += zeros
            total_params += params

    return {
        "total_zeros": total_zeros,
        "total_params": total_params,
        "overall_sparsity": total_zeros / total_params if total_params > 0 else 0,
        "per_layer": per_layer,
    }


def print_sparsity(report: dict, title: str = "稀疏度报告"):
    """格式化打印稀疏度。"""
    print(f"\n--- {title} ---")
    for layer_name, info in report["per_layer"].items():
        print(
            f"  {layer_name:<12} shape={str(info['shape']):<18} "
            f"零权重: {info['zeros']:>5}/{info['params']:<5} "
            f"稀疏度: {info['sparsity']:.2%}"
        )
    print(
        f"  {'整体':<12} "
        f"零权重: {report['total_zeros']:>5}/{report['total_params']:<5} "
        f"稀疏度: {report['overall_sparsity']:.2%}"
    )


# ============================================================
#  主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="对训练好的符号分类器进行剪枝")
    parser.add_argument(
        "--amount", type=float, default=0.5,
        help="剪枝比例 (0~1)，默认 0.5 即剪掉 50%% 权重",
    )
    args = parser.parse_args()

    if not 0 < args.amount < 1:
        print("⚠️  剪枝比例必须在 (0, 1) 之间")
        return

    print(f"使用设备: {DEVICE}")
    print(f"剪枝比例: {args.amount:.0%}（保留 {(1 - args.amount):.0%} 权重）")

    # 1. 加载原始模型
    print("\n[1/4] 加载原始模型...")
    model = BitChecker()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    original_model = copy.deepcopy(model)

    # 原始模型稀疏度
    orig_report = sparsity_report(model)
    print_sparsity(orig_report, "剪枝前")

    # 2. 准备验证数据
    print("\n[2/4] 准备验证数据...")
    X_val, y_val = generate_validation_data(1000)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    # 原始准确率
    orig_acc = evaluate(model, X_val_t, y_val_t)["acc"]
    print(f"  剪枝前验证准确率: {orig_acc:.2%}")

    # 3. 执行剪枝
    print(f"\n[3/4] 执行 L1 非结构化剪枝 (amount={args.amount})...")
    apply_pruning(model, args.amount)

    # 剪枝后稀疏度
    pruned_report = sparsity_report(model)
    print_sparsity(pruned_report, "剪枝后（mask 未固化）")

    # 剪枝后准确率
    pruned_acc = evaluate(model, X_val_t, y_val_t)["acc"]
    print(f"\n  剪枝后验证准确率: {pruned_acc:.2%}")
    print(f"  准确率损失:       {(orig_acc - pruned_acc) * 100:.4f} 个百分点")

    # 4. 固化 mask 并保存
    print("\n[4/4] 固化剪枝 mask 并保存...")
    remove_pruning_reparam(model)

    # 再次检查固化后准确率
    final_acc = evaluate(model, X_val_t, y_val_t)["acc"]
    print(f"  固化后验证准确率: {final_acc:.2%}")

    torch.save(model.state_dict(), PRUNED_PATH)

    # 对比模型文件大小
    import os
    orig_size = os.path.getsize(MODEL_PATH)
    pruned_size = os.path.getsize(PRUNED_PATH)
    print(f"\n  原始模型大小: {orig_size:,} bytes")
    print(f"  剪枝模型大小: {pruned_size:,} bytes")
    print(f"  已保存到 {PRUNED_PATH}")

    print("\n" + "=" * 60)
    print(f"剪枝完成！在剪掉 {args.amount:.0%} 权重的情况下，准确率保持 {final_acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
