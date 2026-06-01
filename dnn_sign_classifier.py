"""
DNN 示例：符号分类器
====================
通过深度神经网络判断一个 16 位有符号整数是正数还是负数。

核心思路：
  - 将整数的 16 位二进制表示（补码）作为网络输入
  - 网络通过训练自动学习到"最高位（MSB）决定符号"这一规律

这是一个经典的"简单问题 + 深层网络"教学示例，展示 DNN 的基本组件：
  数据准备 → 模型定义 → 训练循环 → 验证评估 → 推理预测
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
#  全局设置
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"使用设备: {DEVICE}")


# ============================================================
#  1. 数据准备
# ============================================================

def int_to_bits(value: int) -> np.ndarray:
    """
    将 16 位有符号整数转换为 16 个二进制特征（MSB 在前）。

    使用 two's complement 的位模式：
      - 非负数: MSB = 0，其余位为数值的二进制表示
      - 负数:   MSB = 1，其余位为补码表示

    Returns:
        shape (16,) 的 float32 数组，每个元素为 0.0 或 1.0
    """
    # 取低 16 位，保留补码位模式
    unsigned = value & 0xFFFF
    # MSB 在前 (bit15, bit14, ..., bit0)
    bits = [(unsigned >> (15 - i)) & 1 for i in range(16)]
    return np.array(bits, dtype=np.float32)


def generate_dataset(n_samples: int = 10000):
    """
    生成训练/验证数据集。

    从 [-32768, 32767] 范围内均匀采样，转换为 16 位二进制特征。
    标签: 1 表示 >=0（正/零），0 表示 <0（负）。
    """
    values = np.random.randint(-32768, 32768, size=n_samples)

    X = np.stack([int_to_bits(v) for v in values])          # (N, 16)
    y = (values >= 0).astype(np.float32).reshape(-1, 1)     # (N, 1)

    # 80% 训练 / 20% 验证
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ============================================================
#  2. 模型定义
# ============================================================

class SignClassifier(nn.Module):
    """
    三层全连接网络：16 → 32 → 16 → 1

    虽然这个问题理论上一个神经元就能解决（只看 MSB），
    但多层结构可以演示 DNN 的典型设计模式。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),   # 输入层: 16 位 → 32 维。
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),   # 隐藏层: 32 → 16
            nn.ReLU(),

            nn.Linear(16, 1),    # 输出层: 16 → 1
            nn.Sigmoid(),        # 映射到 [0, 1]，表示 P(正数)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
#  3. 训练
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
        # ---- 训练阶段 ----
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        # ---- 验证阶段 ----
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
#  4. 推理
# ============================================================

def predict(model: nn.Module, value: int) -> tuple[bool, float]:
    """
    对用户输入的整数进行预测。

    Returns:
        (is_positive, confidence): 是否为正数，以及置信度 [0, 1]
    """
    model.eval()
    bits = torch.from_numpy(int_to_bits(value)).unsqueeze(
        0).to(DEVICE)  # (1, 16)
    prob = model(bits).item()
    is_positive = prob >= 0.5
    confidence = prob if is_positive else (1.0 - prob)
    return is_positive, confidence


# ============================================================
#  5. 权重可视化
# ============================================================

def print_model(model: nn.Module):
    """打印模型结构概览，逐层显示输入输出形状和参数量。"""
    print("\n" + "=" * 62)
    print(f"{'层名':<12} {'类型':<18} {'输出形状':<16} {'参数量':>8}")
    print("-" * 62)

    total = 0
    # 构造一个 dummy 输入来计算每层的输出形状
    dummy = torch.zeros(1, 16)

    for name, module in model.named_children():
        if name == "net" and isinstance(module, nn.Sequential):
            # 展开 Sequential 内部的子层
            x = dummy
            for i, child in enumerate(module):
                class_name = child.__class__.__name__
                x = child(x)
                params = sum(p.numel() for p in child.parameters())
                total += params
                print(
                    f"  [{i:<2}]      {class_name:<18} {str(list(x.shape)):<16} {params:>8,}")
        else:
            class_name = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters())
            total += params
            print(f"  {name:<10} {class_name:<18} {'-':<16} {params:>8,}")

    print("-" * 62)
    print(f"{'总计':>46} {total:>8,}")
    print("=" * 62)


def show_learned_weights(model: nn.Module):
    """打印网络所有层的权重和偏置数据。"""
    np.set_printoptions(precision=6, linewidth=120,
                        suppress=True, threshold=np.inf)

    for i, module in enumerate(model.net):
        if isinstance(module, nn.Linear):
            w = module.weight.data.cpu().numpy()
            b = module.bias.data.cpu().numpy()
            in_feat, out_feat = module.in_features, module.out_features
            # print(f"\n--- 第 {i} 层: Linear(in={in_feat}, out={out_feat}) ---")
            # print(f"权重绝对值 shape={w.shape}:\n{np.abs(w)}")
            # print(f"偏置绝对值 shape={b.shape}:\n{np.abs(b)}")

    # 分析第一层 MSB 连接强度
    first_layer = model.net[0]  # nn.Linear(16, 32)
    weights = first_layer.weight.data.cpu().numpy()  # shape (32, 16)
    msb_mean_abs = np.abs(weights[:, 0]).mean()  # 第0列是 MSB（最高位），我们关注它的权重绝对值平均值
    others_mean_abs = np.abs(weights[:, 1:]).mean()  # 第0列是 MSB，剩下 15 列是其他位
    print(f"\n--- MSB 分析 ---")
    print(f"bit15 (MSB) 平均权重绝对值: {msb_mean_abs:.4f}")
    print(f"其余 15 位平均权重绝对值:   {others_mean_abs:.4f}")
    print(f"MSB 权重是其余位平均的 {msb_mean_abs / (others_mean_abs + 1e-8):.1f} 倍")
    print("→ 网络已经学会：最高位（符号位）是判断正负的关键特征！")


# ============================================================
#  主程序
# ============================================================

def main():
    # ---- 生成数据 ----
    print("\n[1/3] 正在生成数据...")
    X_train, y_train, X_val, y_val = generate_dataset(20000)
    print(f"  训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # ---- 创建模型 ----
    print("\n[2/3] 创建模型...")
    model = SignClassifier()
    print_model(model)

    # ---- 训练 ----
    print("\n[3/3] 开始训练...")
    train(
        model,
        train_loader,
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        epochs=30,
        lr=0.001,
    )

    # 最终评估
    final = evaluate(
        model,
        torch.from_numpy(X_val).to(DEVICE),
        torch.from_numpy(y_val).to(DEVICE),
    )
    print(f"\n训练完成！验证集准确率: {final['acc']:.2%}")

    # 权重分析
    show_learned_weights(model)

    # ---- 交互式推理 ----
    print("\n" + "=" * 50)
    print("交互式推理 — 输入 16 位有符号整数，模型判断正负")
    print("范围: [-32768, 32767]，输入 q 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n请输入一个整数: ").strip()
            if user_input.lower() == "q":
                print("再见！")
                break

            value = int(user_input)
            if value < -32768 or value > 32767:
                print("⚠️  超出 16 位有符号整数范围，请重试")
                continue

            is_positive, conf = predict(model, value)
            actual = "正数（>=0）" if value >= 0 else "负数（<0）"
            prediction = "正数" if is_positive else "负数"

            correct = (value >= 0) == is_positive
            mark = "✅" if correct else "❌"
            print(
                f"  预测: {prediction}  |  实际: {actual}  |  置信度: {conf:.2%}  {mark}")

        except ValueError:
            print("⚠️  请输入有效的整数")
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    main()
