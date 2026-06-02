"""
DNN 符号分类器 - 2 的幂判断训练脚本
==================================
判断 32 位有符号整数是否为 2 的正整数次幂（1, 2, 4, 8, ...）。

与符号/奇偶判断不同，这不是单 bit 问题——网络需要学会计数：
32 位输入中恰好只有 1 个 bit 为 1，其余全为 0。

用法：
    python power_of_two_train.py
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import DEVICE, print_model, BitClassifier
from train import generate_power_of_two_dataset, evaluate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "power_of_two_classifier.pth"


def train(model, train_loader, X_val, y_val, epochs=30, lr=0.001):
    """训练 2 的幂判断模型。"""
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
    """分析第一层权重——是否所有位都被均等关注（因为需要计数）。"""
    with np.printoptions(precision=4, linewidth=120, suppress=True):
        first_layer = model.net[0]
        weights = first_layer.weight.data.cpu().numpy()

        bit_weights = np.abs(weights).mean(axis=0)  # 每个输入位的平均权重
        top_indices = np.argsort(bit_weights)[::-1]

        print(f"\n--- 权重分析 ---")
        print(f"各 bit 位平均权重绝对值:")
        print(f"  最高 5 位: {[(f'bit{31-i}', bit_weights[i]) for i in top_indices[:5]]}")
        print(f"  最低 5 位: {[(f'bit{31-i}', bit_weights[i]) for i in top_indices[-5:]]}")
        print(f"  均值: {bit_weights.mean():.4f}  标准差: {bit_weights.std():.4f}")

        # 符号/奇偶是单 bit 问题，std 很大；2 的幂需要均等关注所有位，std 应该小
        if bit_weights.std() / bit_weights.mean() < 0.5:
            print("→ 权重分布均匀——网络学会了对所有位'计数'，而不是只看某一位！")
        else:
            print(f"→ 权重最高位是其他位的 {bit_weights.max() / bit_weights.mean():.1f} 倍")


def main():
    print(f"使用设备: {DEVICE}")

    # 1. 生成数据
    print("\n[1/3] 正在生成 2 的幂判断数据...")
    X_train, y_train, X_val, y_val = generate_power_of_two_dataset(20000)
    print(f"  训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")
    print(f"  正样本比例: {y_train.mean():.1%}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 2. 创建模型
    print("\n[2/3] 创建模型...")
    model = BitClassifier()
    print_model(model)

    # 3. 训练
    print("\n[3/3] 开始训练...")
    train(
        model,
        train_loader,
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        epochs=30,
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
