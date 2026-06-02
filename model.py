"""
DNN 符号分类器 - 模型定义
=========================
三层全连接网络：32 → 64 → 32 → 1
判断 32 位有符号整数是正数还是负数。
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
    将 32 位有符号整数转换为 32 个二进制特征（MSB 在前）。

    使用 two's complement 的位模式：
      - 非负数: MSB = 0，其余位为数值的二进制表示
      - 负数:   MSB = 1，其余位为补码表示

    Returns:
        shape (32,) 的 float32 数组，每个元素为 0.0 或 1.0
    """
    # 将输入限制在32位有符号整数范围内
    # 通过掩码操作确保只保留低32位，模拟32位整数行为
    if value > 0x7FFFFFFF:  # 如果大于最大32位有符号整数
        # 对于超出范围的正数，将其转换为对应的32位有符号整数表示
        value = ((value + 0x80000000) % 0x100000000) - 0x80000000
    elif value < -0x80000000:  # 如果小于最小32位有符号整数
        # 对于超出范围的负数，将其转换为对应的32位有符号整数表示
        value = ((value + 0x80000000) % 0x100000000) - 0x80000000
    
    # 确保值在32位有符号整数范围内
    value = int(np.int32(value))
    
    unsigned = value & 0xFFFFFFFF
    bits = [(unsigned >> (31 - i)) & 1 for i in range(32)]
    return np.array(bits, dtype=np.float32)


# ============================================================
#  模型
# ============================================================

class SignClassifier(nn.Module):
    """
    三层全连接网络：32 → 64 → 32 → 1

    虽然这个问题理论上一个神经元就能解决（只看 MSB），
    但多层结构可以演示 DNN 的典型设计模式。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),   # 输入层: 32 位 → 64 维
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),   # 隐藏层: 64 → 32
            nn.ReLU(),

            nn.Linear(32, 1),    # 输出层: 32 → 1
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
    dummy = torch.zeros(1, 32)

    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            x = dummy
            for i, child in enumerate(module):
                class_name = child.__class__.__name__
                try:
                    x = child(x)
                    shape_str = str(list(x.shape))
                except Exception:
                    shape_str = "-"
                params = sum(p.numel() for p in child.parameters())
                total += params
                print(f"  [{i:<2}]      {class_name:<18} {shape_str:<16} {params:>8,}")
            break  # Sequential 已处理，不再递归
        elif isinstance(module, nn.Linear) and name:
            # 非 Sequential 包裹的独立 Linear 层
            params = sum(p.numel() for p in module.parameters())
            total += params
            print(f"  {name:<10} {module.__class__.__name__:<18} {'-':<16} {params:>8,}")

    print("-" * 62)
    print(f"{'总计':>46} {total:>8,}")
    print("=" * 62)


# ============================================================
#  使用 torch.fx 分析网络结构
# ============================================================

def analyze_with_torch_fx():
    """
    使用 torch.fx 分析网络结构，打印图 IR、生成代码和操作统计。
    """
    try:
        import torch.fx

        model = SignClassifier()
        traced_model = torch.fx.symbolic_trace(model)

        print("\n" + "=" * 60)
        print("使用 torch.fx 分析的网络结构:")
        print("=" * 60)
        print(traced_model.graph)

        print("\n" + "-" * 60)
        print("生成的代码表示:")
        print("-" * 60)
        print(traced_model.code)

        print("\n" + "-" * 60)
        print("操作类型统计:")
        print("-" * 60)
        op_count = {}
        for node in traced_model.graph.nodes:
            op_type = node.op
            op_count[op_type] = op_count.get(op_type, 0) + 1
            if op_type == 'call_module':
                module_type = type(model.get_submodule(node.target)).__name__
                print(f"  {node.name}: {op_type} -> {module_type}")

        print(f"\n操作统计: {op_count}")
        return traced_model

    except ImportError:
        print("⚠️ torch.fx 不可用，请升级 PyTorch 版本")
        return None
    except Exception as e:
        print(f"❌ torch.fx 分析失败: {e}")
        return None


# ============================================================
#  ONNX 导出功能
# ============================================================

def export_model_to_onnx(model_path="sign_classifier.onnx", input_shape=(1, 32)):
    """
    将模型导出为 ONNX 格式

    Args:
        model_path: ONNX 模型保存路径
        input_shape: 输入张量的形状 (batch_size, features)
    """
    import torch.onnx

    model = SignClassifier()
    model.eval()

    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        dynamo=False,  # 使用旧版 TorchScript 导出路径，兼容 OpenCV
    )

    print(f"✅ 模型已成功导出到: {model_path}")

    try:
        import onnx
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型验证通过")
    except ImportError:
        print("⚠️ 未安装 onnx，跳过模型验证")
    except Exception as e:
        print(f"❌ ONNX 模型验证失败: {e}")


def test_onnx_model(onnx_path="sign_classifier.onnx"):
    """
    测试导出的 ONNX 模型
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        # 创建测试数据
        test_input = np.random.randn(1, 32).astype(np.float32)
        
        # 加载 ONNX 模型
        session = ort.InferenceSession(onnx_path)
        
        # 运行推理
        result = session.run(None, {'input': test_input})
        
        print(f"✅ ONNX 模型推理成功")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {result[0].shape}")
        print(f"   输出值: {result[0][0][0]:.4f} (概率)")
        
        # 测试几个特定值
        print("\n   测试特殊值:")
        for val in [0x7FFFFFFF, 0x00000000, 0x80000000]:  # 正数、零、负数
            bits = int_to_bits(val)
            result = session.run(None, {'input': bits.reshape(1, -1)})
            sign_pred = "正数" if result[0][0][0] > 0.5 else "负数"
            actual_sign = "正数" if val >= 0 else "负数"
            print(f"   值: {val:08X} -> 预测: {sign_pred}, 实际: {actual_sign}")
        
    except ImportError:
        print("⚠️ 未安装 onnxruntime，跳过 ONNX 模型测试")
    except Exception as e:
        print(f"❌ ONNX 模型测试失败: {e}")


# ============================================================
#  主函数
# ============================================================

if __name__ == "__main__":
    print("开始导出模型为 ONNX 格式...")
    export_model_to_onnx()
    test_onnx_model()

    print("\n正在分析网络结构...")
    analyze_with_torch_fx()