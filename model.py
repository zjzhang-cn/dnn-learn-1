"""
DNN 符号分类器 - 模型定义
=========================
三层全连接网络：16 → 32 → 16 → 1
判断 16 位有符号整数是正数还是负数。
"""

import numpy as np
import torch
import torch.nn as nn

# ============================================================
#  设备选择：CUDA > Apple MPS > CPU
# ============================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ============================================================
#  数据编码
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
    unsigned = value & 0xFFFF
    bits = [(unsigned >> (15 - i)) & 1 for i in range(16)]
    return np.array(bits, dtype=np.float32)


# ============================================================
#  模型
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
            nn.Linear(16, 32),   # 输入层: 16 位 → 32 维
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
#  模型结构打印
# ============================================================

def print_model(model: nn.Module):
    """打印模型结构概览，逐层显示输入输出形状和参数量。"""
    print("\n" + "=" * 62)
    print(f"{'层名':<12} {'类型':<18} {'输出形状':<16} {'参数量':>8}")
    print("-" * 62)

    total = 0
    dummy = torch.zeros(1, 16)

    for name, module in model.named_children():
        if name == "net" and isinstance(module, nn.Sequential):
            x = dummy
            for i, child in enumerate(module):
                class_name = child.__class__.__name__
                x = child(x)
                params = sum(p.numel() for p in child.parameters())
                total += params
                print(f"  [{i:<2}]      {class_name:<18} {str(list(x.shape)):<16} {params:>8,}")
        else:
            class_name = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters())
            total += params
            print(f"  {name:<10} {class_name:<18} {'-':<16} {params:>8,}")

    print("-" * 62)
    print(f"{'总计':>46} {total:>8,}")
    print("=" * 62)
