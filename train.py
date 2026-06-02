"""
DNN 符号分类器 - 训练脚本
=========================
生成训练数据，训练模型，保存权重到 sign_classifier.pth，导出 ONNX 模型。

用法：
    python train.py
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import DEVICE, SignClassifier, int_to_bits, print_model, export_model_to_onnx

# ============================================================
#  随机种子
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "sign_classifier.pth"


# ============================================================
#  数据生成
# ============================================================

def generate_dataset(n_samples: int = 10000):
    """
    生成训练/验证数据集。

    从 [-2^31, 2^31-1] 范围内均匀采样，转换为 32 位二进制特征。
    标签: 1 表示 >=0（正/零），0 表示 <0（负）。

    Returns:
        (X_train, y_train, X_val, y_val) — 80/20 分割
    """
    # 修正范围为 [-2^31, 2^31-1]，即 [-2147483648, 2147483647]
    values = np.random.randint(-2147483648, 2147483647, size=n_samples)

    X = np.stack([int_to_bits(v) for v in values])          # (N, 32)
    y = (values >= 0).astype(np.float32).reshape(-1, 1)     # (N, 1)

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def generate_validation_data(n_samples: int = 1000):
    """
    仅生成验证数据，不产生训练集浪费。

    Returns:
        (X_val, y_val)
    """
    # 修正范围为 [-2^31, 2^31-1]，即 [-2147483648, 2147483647]
    values = np.random.randint(-2147483648, 2147483647, size=n_samples)

    X = np.stack([int_to_bits(v) for v in values])
    y = (values >= 0).astype(np.float32).reshape(-1, 1)

    return X, y


def generate_parity_dataset(n_samples: int = 10000):
    """
    生成奇偶判断数据集。

    标签: 1 表示偶数，0 表示奇数。
    与符号判断不同，奇偶性需要看最低位（LSB），
    所有位都参与处理——不再是单 bit 问题。

    Returns:
        (X_train, y_train, X_val, y_val) — 80/20 分割
    """
    values = np.random.randint(-2147483648, 2147483648, size=n_samples)

    X = np.stack([int_to_bits(v) for v in values])
    y = ((values % 2 == 0)).astype(np.float32).reshape(-1, 1)

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ============================================================
#  评估
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    """在给定数据上计算损失和准确率。"""
    model.eval()
    outputs = model(X)
    loss = nn.BCELoss()(outputs, y).item()
    preds = (outputs >= 0.5).float()
    acc = (preds == y).float().mean().item()
    return {"loss": loss, "acc": acc}


# ============================================================
#  训练
# ============================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.001,
):
    """训练模型并输出进度。"""
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


# ============================================================
#  权重分析
# ============================================================

def show_learned_weights(model: nn.Module):
    """打印网络所有层的权重绝对值和 MSB 分析。"""
    with np.printoptions(precision=6, linewidth=120, suppress=True, threshold=np.inf):

        for i, module in enumerate(model.net):
            if isinstance(module, nn.Linear):
                w = module.weight.data.cpu().numpy()
                b = module.bias.data.cpu().numpy()
                print(f"\n--- 第 {i} 层: Linear(in={module.in_features}, out={module.out_features}) ---")
                print(f"权重绝对值 shape={w.shape}:\n{np.abs(w)}")
                print(f"偏置绝对值 shape={b.shape}:\n{np.abs(b)}")

        first_layer = model.net[0]
        weights = first_layer.weight.data.cpu().numpy()
        msb_mean_abs = np.abs(weights[:, 0]).mean()
        others_mean_abs = np.abs(weights[:, 1:]).mean()
        print(f"\n--- MSB 分析 ---")
        print(f"bit31 (MSB) 平均权重绝对值: {msb_mean_abs:.4f}")
        print(f"其余 31 位平均权重绝对值:   {others_mean_abs:.4f}")
        print(f"MSB 权重是其余位平均的 {msb_mean_abs / (others_mean_abs + 1e-8):.1f} 倍")
        print("→ 网络已经学会：最高位（符号位）是判断正负的关键特征！")


# ============================================================
#  主程序
# ============================================================

def main():
    print(f"使用设备: {DEVICE}")

    # 1. 生成数据
    print("\n[1/3] 正在生成数据...")
    X_train, y_train, X_val, y_val = generate_dataset(20000)
    print(f"  训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 2. 创建模型
    print("\n[2/3] 创建模型...")
    model = SignClassifier()
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

    # 4. 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n模型已保存到 {MODEL_PATH}")

    # 5. 导出 ONNX 模型
    print("\n[5/5] 导出 ONNX 模型...")
    export_model_to_onnx("sign_classifier.onnx", model=model.to("cpu"))


if __name__ == "__main__":
    main()
