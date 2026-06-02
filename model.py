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
#  网络结构可视化
# ============================================================

def visualize_model_structure(model_path="sign_classifier_graph.png"):
    """
    使用 torchviz 生成网络结构图
    
    Args:
        model_path: 图片保存路径
    """
    try:
        from torchviz import make_dot
        
        # 创建模型实例
        model = SignClassifier()
        
        # 创建示例输入
        x = torch.randn(1, 32)  # 批次大小为1，输入32维
        
        # 前向传播获取输出
        y = model(x)
        
        # 生成可视化图
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # 保存为PNG格式
        dot.format = 'png'
        dot.render(model_path.replace('.png', ''), cleanup=True)
        
        print(f"✅ 网络结构图已保存到: {model_path}")
        
    except ImportError:
        print("⚠️ 未安装 torchviz 或 graphviz")
        print("请运行: pip install torchviz graphviz")
        print("还需要安装 Graphviz 软件: https://graphviz.org/download/")
    except Exception as e:
        print(f"❌ 生成网络结构图失败: {e}")


def draw_network_diagram_matplotlib():
    """
    使用 matplotlib 手动绘制网络结构图
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 定义各层的神经元数量
        layers = [32, 64, 32, 1]  # 32 -> 64 -> 32 -> 1
        layer_names = ['Input\n(32)', 'Hidden1\n(64)', 'Hidden2\n(32)', 'Output\n(1)']
        
        # 绘制各层的神经元
        layer_positions = []
        for layer_idx, (n_neurons, name) in enumerate(zip(layers, layer_names)):
            x_pos = layer_idx * 2.5  # 每层之间的水平距离
            
            # 计算垂直位置
            if n_neurons == 1:  # 输出层只有一个神经元
                y_positions = [0]
            else:
                y_start = -(n_neurons-1)*0.2
                y_positions = [y_start + i*0.4 for i in range(n_neurons)]
            
            # 绘制神经元
            neuron_circles = []
            for y_pos in y_positions[:min(n_neurons, 10)]:  # 限制显示的神经元数量
                circle = patches.Circle((x_pos, y_pos), 0.08, linewidth=2, 
                                      edgecolor='blue', facecolor='lightblue')
                ax.add_patch(circle)
                neuron_circles.append((x_pos, y_pos))
            
            # 如果神经元太多，用省略号表示
            if n_neurons > 10:
                dots_y = y_positions[9] + 0.4
                dots_circle = patches.Circle((x_pos, dots_y), 0.08, 
                                           edgecolor='gray', facecolor='lightgray')
                ax.add_patch(dots_circle)
                ax.text(x_pos, dots_y, '...', ha='center', va='center', fontsize=10)
            
            # 添加层名称
            ax.text(x_pos, max(y_positions[:min(n_neurons, 10)]) + 0.6, 
                   name, ha='center', va='center', fontweight='bold')
            
            layer_positions.append(neuron_circles)
        
        # 绘制连接线（仅显示部分连接作为示意）
        for i in range(len(layer_positions)-1):
            for j, (src_x, src_y) in enumerate(layer_positions[i][:3]):  # 只显示前3个连接
                for k, (tgt_x, tgt_y) in enumerate(layer_positions[i+1][:3]):
                    ax.plot([src_x, tgt_x], [src_y, tgt_y], 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-1, 3*len(layers)-1)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('SignClassifier Network Architecture\n32 → 64 → 32 → 1', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        
        print("✅ 手绘网络结构图已保存为: network_architecture.png")
        
    except ImportError:
        print("⚠️ 未安装 matplotlib，跳过手绘结构图")


def visualize_simple_model():
    """
    创建一个更简洁的模型可视化，突出显示主要结构
    """
    try:
        from torchviz import make_dot
        
        # 创建简化版的模型结构展示
        model = SignClassifier()
        model.eval()  # 设置为评估模式
        
        # 创建虚拟输入
        dummy_input = torch.randn(1, 32)
        
        # 获取模型输出
        output = model(dummy_input)
        
        # 生成模型图
        model_graph = make_dot(output, 
                              params=dict(model.named_parameters()),
                              show_attrs=True,
                              show_saved=True)
        
        model_graph.format = 'png'
        model_graph.render('detailed_network_structure', cleanup=True)
        
        print("✅ 详细网络结构图已保存为: detailed_network_structure.png")
        
    except Exception as e:
        print(f"❌ 生成详细结构图失败: {e}")


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
    
    # 创建模型实例
    model = SignClassifier()
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 导出为 ONNX
    torch.onnx.export(
        model,                              # 要导出的模型
        dummy_input,                        # 模型的示例输入
        model_path,                         # 输出文件路径
        export_params=True,                 # 存储训练后的参数权重
        opset_version=11,                   # ONNX 操作集版本 (兼容性较好)
        do_constant_folding=True,           # 是否执行常量折叠优化
        input_names=['input'],              # 输入名称
        output_names=['output'],            # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},    # 动态批次大小
            'output': {0: 'batch_size'},
        }
    )
    
    print(f"✅ 模型已成功导出到: {model_path}")
    
    # 验证导出的模型
    try:
        import onnx
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型验证通过")
    except ImportError:
        print("⚠️ 未安装 onnx，跳过模型验证")
    except Exception as e:
        print(f"❌ ONNX 模型验证失败: {e}")


# ============================================================
#  使用 torch.fx 分析网络结构
# ============================================================

def analyze_with_torch_fx(visualize=True, output_path="fx_model_graph.png"):
    """
    使用 torch.fx 分析和可视化网络结构
    
    Args:
        visualize: 是否生成可视化图片
        output_path: 图片输出路径
    """
    try:
        import torch.fx
        
        # 创建模型实例
        model = SignClassifier()
        
        # 使用 torch.fx 跟踪模型
        traced_model = torch.fx.symbolic_trace(model)
        
        print("\n" + "=" * 60)
        print("使用 torch.fx 分析的网络结构:")
        print("=" * 60)
        
        # 打印模型的图形表示
        print(traced_model.graph)
        
        # 也可以直接打印模型的代码表示
        print("\n" + "-" * 60)
        print("生成的代码表示:")
        print("-" * 60)
        print(traced_model.code)
        
        # 统计模型中的操作类型
        print("\n" + "-" * 60)
        print("操作类型统计:")
        print("-" * 60)
        op_count = {}
        for node in traced_model.graph.nodes:
            op_type = node.op  # 'call_module', 'call_function', 'placeholder', 'output', etc.
            op_count[op_type] = op_count.get(op_type, 0) + 1
            
            # 如果是模块调用，也记录具体的模块类型
            if op_type == 'call_module':
                module_type = type(model.get_submodule(node.target)).__name__
                print(f"  {node.name}: {op_type} -> {module_type}")
        
        print(f"\n操作统计: {op_count}")
        
        # 如果需要可视化，则使用torchviz生成图片
        if visualize:
            try:
                from torchviz import make_dot
                import subprocess
                
                # 检查是否能找到 dot 命令
                try:
                    subprocess.run(['dot', '-V'], capture_output=True, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("⚠️ 未找到 Graphviz 的 dot 命令")
                    print("请安装 Graphviz 应用程序并确保其 bin 目录在系统 PATH 中")
                    print("下载地址: https://graphviz.org/download/")
                    print("Windows 用户可以使用: choco install graphviz 或手动安装")
                    return traced_model
                
                # 使用traced_model的输出进行可视化
                x = torch.randn(1, 32)
                y = model(x)
                
                # 使用make_dot生成图形
                dot = make_dot(y, params=dict(model.named_parameters()))
                
                # 保存为指定路径
                dot.format = 'png'
                dot.render(output_path.replace('.png', ''), cleanup=True)
                
                print(f"✅ torch.fx网络结构图已保存到: {output_path}")
                
            except ImportError:
                print("⚠️ 未安装 torchviz，跳过图片生成")
                print("请运行: pip install torchviz")
            except subprocess.CalledProcessError:
                print("⚠️ Graphviz 命令执行失败")
                print("请确保 Graphviz 已正确安装且在系统 PATH 中")
            except Exception as e:
                print(f"❌ 生成图片失败: {e}")
        
        return traced_model
        
    except ImportError:
        print("⚠️ torch.fx 不可用，请升级 PyTorch 版本")
        return None
    except Exception as e:
        print(f"❌ torch.fx 分析失败: {e}")
        return None


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
#  主函数 - 用于测试和可视化
# ============================================================

if __name__ == "__main__":
    print("开始导出模型为 ONNX 格式...")
    
    # 导出模型
    export_model_to_onnx()
    
    # 测试导出的模型
    test_onnx_model()
    
    # 生成网络结构图
    print("\n正在生成网络结构图...")
    # visualize_model_structure()
    # draw_network_diagram_matplotlib()
    # visualize_simple_model()
    
    analyze_with_torch_fx()