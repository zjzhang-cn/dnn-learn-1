"""
DNN 符号分类器 - 奇偶判断训练脚本
================================
与符号判断不同，奇偶性需要学习看 LSB（最低位），
32 位都要参与处理——不再是单 bit 问题，网络需要真正"学会"计算。

用法：
    python parity_train.py
"""

import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import DEVICE, print_model, BitChecker
from train import generate_parity_dataset, evaluate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "parity_classifier.pth"


def train(model, train_loader, X_val, y_val, epochs=30, lr=0.001):
    """训练奇偶判断模型。"""
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
    """分析第一层权重分布——LSB（bit0，即第 32 位）是否被重点关注。"""
    with np.printoptions(precision=4, linewidth=120, suppress=True):
        first_layer = model.net[0]
        weights = first_layer.weight.data.cpu().numpy()

        # bit31 是 MSB（输入第 0 位），bit0 是 LSB（输入第 31 位）
        msb_mean_abs = np.abs(weights[:, 0]).mean()   # MSB
        lsb_mean_abs = np.abs(weights[:, 31]).mean()  # LSB
        others_mean_abs = np.abs(weights[:, 1:31]).mean()

        print(f"\n--- 权重分析 ---")
        print(f"MSB (bit31, 输入位0) 平均权重绝对值:  {msb_mean_abs:.4f}")
        print(f"LSB (bit0,  输入位31) 平均权重绝对值: {lsb_mean_abs:.4f}")
        print(f"其余 30 位平均权重绝对值:             {others_mean_abs:.4f}")

        if lsb_mean_abs > msb_mean_abs * 1.5:
            print("→ 网络学会：最低位是判断奇偶的关键特征！")
        else:
            # 打印每位的权重以帮助诊断
            bit_weights = np.abs(weights).mean(axis=0)
            top_indices = np.argsort(bit_weights)[::-1][:5]
            print(f"→ 权重最高的 5 个输入位: {[(i, bit_weights[i]) for i in top_indices]}")


def main():
    print(f"使用设备: {DEVICE}")

    # 1. 生成数据
    print("\n[1/3] 正在生成奇偶判断数据...")
    X_train, y_train, X_val, y_val = generate_parity_dataset(20000)
    print(f"  训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")
    print(f"  训练集偶数比例: {y_train.mean():.1%}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 2. 创建模型
    print("\n[2/3] 创建模型...")
    model = BitChecker()
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
