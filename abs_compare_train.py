"""
DNN 符号分类器 - 绝对值比较训练脚本
================================
判断 |x| > threshold（默认 1000）。

这是第一个非线性决策边界的任务——|x| > 1000 在两个方向上
（x > 1000 和 x < -1000）都是正样本，网络需要学会忽略 MSB。

用法：
    python abs_compare_train.py
    python abs_compare_train.py --threshold 5000
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import DEVICE, print_model, BitClassifier
from train import generate_abs_compare_dataset, evaluate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "abs_compare_classifier.pth"


def train(model, train_loader, X_val, y_val, epochs=40, lr=0.001):
    """训练绝对值比较模型。"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(DEVICE)
    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, X_val, y_val)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_metrics['loss']:.4f}  "
                f"Val Acc: {val_metrics['acc']:.2%}"
            )


def show_learned_weights(model):
    """分析权重——MSB 是否被压制（因为需要忽略符号）。"""
    with np.printoptions(precision=4, linewidth=120, suppress=True):
        first_layer = model.net[0]
        weights = first_layer.weight.data.cpu().numpy()

        bit_weights = np.abs(weights).mean(axis=0)
        msb_w = bit_weights[0]   # MSB（bit31）
        others_mean = bit_weights[1:].mean()

        print(f"\n--- 权重分析 ---")
        print(f"MSB (bit31) 权重: {msb_w:.4f}")
        print(f"其余 31 位均值:  {others_mean:.4f}")

        if msb_w < others_mean * 0.7:
            print("→ MSB 权重被压制——网络学会了忽略符号位！")
        elif msb_w > others_mean * 1.5:
            print("→ MSB 权重偏高——网络过度依赖符号位")
        else:
            print("→ MSB 权重与其他位相当")

        # 高位（bit30-bit24）vs 低位
        high_bits = bit_weights[1:8].mean()   # bit30-bit24
        low_bits = bit_weights[24:].mean()    # bit7-bit0
        print(f"高位 (bit30-24) 均值: {high_bits:.4f}")
        print(f"低位 (bit7-0) 均值:   {low_bits:.4f}")
        if high_bits > low_bits * 1.3:
            print("→ 高位权重大于低位——网络关注数值大小！")


def main():
    parser = argparse.ArgumentParser(description="绝对值比较训练")
    parser.add_argument("--threshold", type=int, default=1000,
                        help="比较阈值（默认 1000）")
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")
    print(f"阈值: |x| > {args.threshold}")

    print("\n[1/3] 正在生成绝对值比较数据...")
    X_train, y_train, X_val, y_val = generate_abs_compare_dataset(
        20000, threshold=args.threshold
    )
    print(f"  训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")
    print(f"  正样本比例 (|x| > {args.threshold}): {y_train.mean():.1%}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    print("\n[2/3] 创建模型...")
    model = BitClassifier()
    print_model(model)

    print("\n[3/3] 开始训练...")
    train(
        model,
        train_loader,
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        epochs=40,
        lr=0.001,
    )

    final = evaluate(
        model,
        torch.from_numpy(X_val).to(DEVICE),
        torch.from_numpy(y_val).to(DEVICE),
    )
    print(f"\n训练完成！验证集准确率: {final['acc']:.2%}")

    show_learned_weights(model)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n模型已保存到 {MODEL_PATH}")


if __name__ == "__main__":
    main()
