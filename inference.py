"""
DNN 符号分类器 - 推理脚本
=========================
加载已训练模型，交互式判断 32 位有符号整数的正负。

用法：
    python inference.py                        # 加载原始模型
    python inference.py --model sign_classifier_pruned.pth   # 加载剪枝模型
    python inference.py -m sign_classifier_pruned.pth         # 简写
"""

import argparse
import zipfile

import torch

from model import DEVICE, SignClassifier, int_to_bits

DEFAULT_MODEL = "sign_classifier.pth"


# ============================================================
#  量化引擎初始化
# ============================================================

def _init_quant_engine():
    """初始化量化后端，加载 INT8 TorchScript 模型前必须调用。"""
    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    current = getattr(torch.backends.quantized, "engine", "none")
    if current != "none":
        return
    for candidate in ("qnnpack", "fbgemm", "x86"):
        if candidate in engines:
            torch.backends.quantized.engine = candidate
            return


# ============================================================
#  INT4 解包（与 quantize.py 保持一致）
# ============================================================

def _unpack_int4(packed_info: dict) -> torch.Tensor:
    """从打包的 INT4 数据还原 float32 权重。"""
    scale = packed_info["scale"]
    packed = packed_info["packed"]
    shape = packed_info["shape"]
    numel = packed_info["numel"]

    high = (packed >> 4).to(torch.uint8)
    low = (packed & 0x0F).to(torch.uint8)
    flat = torch.stack([high, low], dim=1).flatten()[:numel]
    q = flat.to(torch.float32) - 7.0
    return (q * scale).reshape(shape)


def _load_int4_state(filepath: str, map_location) -> dict:
    """加载 INT4 文件并还原为普通 state_dict。"""
    raw = torch.load(filepath, map_location="cpu", weights_only=False)
    state_dict = {}
    for key, value in raw.items():
        if key == "__format__":
            continue
        if isinstance(value, dict) and "packed" in value:
            state_dict[key] = _unpack_int4(value)
        else:
            state_dict[key] = value
    return state_dict


# ============================================================
#  预测
# ============================================================

def predict(
    model: torch.nn.Module,
    value: int,
    device: torch.device,
    input_dtype: torch.dtype,
) -> tuple[bool, float]:
    """
    对用户输入的整数进行预测。

    Returns:
        (is_positive, confidence): 是否为正数，以及置信度 [0, 1]
    """
    model.eval()
    bits = torch.from_numpy(int_to_bits(value)).unsqueeze(0).to(device=device, dtype=input_dtype)
    prob = model(bits).item()
    is_positive = prob >= 0.5
    confidence = prob if is_positive else (1.0 - prob)
    return is_positive, confidence


# ============================================================
#  主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="加载模型并交互式判断整数正负")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型文件路径（默认: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--fp16-infer",
        action="store_true",
        help="对 .pth 模型启用 FP16 推理（CPU 上会自动回退到 FP32）",
    )
    parser.add_argument(
        "--bf16-infer",
        action="store_true",
        help="对 .pth 模型启用 BF16 推理（需要设备支持 bfloat16）",
    )
    args = parser.parse_args()

    # 通过文件头判断：TorchScript 是 zip 格式，state_dict 是 pickle
    # 通过 zip 内容判断：TorchScript 包含 "code/__torch__" 目录
    is_jit = False
    if zipfile.is_zipfile(args.model):
        with zipfile.ZipFile(args.model) as zf:
            names = zf.namelist()
        is_jit = any("code/" in n for n in names)

    if is_jit:
        # TorchScript 量化模型，CPU 推理更稳定
        _init_quant_engine()
        runtime_device = torch.device("cpu")
        input_dtype = torch.float32
        model = torch.jit.load(args.model, map_location=runtime_device)
        model.eval()
        print(f"使用设备: {runtime_device}")
        print(f"模型类型: TorchScript（常用于 INT8 动态量化导出）")
        print(f"模型已从 {args.model} 加载\n")
    else:
        runtime_device = DEVICE
        input_dtype = torch.float32
        model = SignClassifier()

        # 检测是否为 INT4 打包格式
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("__format__") == "int4":
            state_dict = _load_int4_state(args.model, runtime_device)
            print(f"模型类型: INT4 对称量化（解包后推理）")
        else:
            state_dict = checkpoint
            print(f"模型类型: PyTorch state_dict")

        model.load_state_dict(state_dict)
        model.to(runtime_device)

        if args.fp16_infer:
            if runtime_device.type == "cpu":
                print("⚠️  CPU 对 FP16 推理支持有限，已自动回退 FP32")
            else:
                model = model.half()
                input_dtype = torch.float16

        if args.bf16_infer:
            if not hasattr(torch, "bfloat16"):
                print("⚠️  当前 PyTorch 版本不支持 bfloat16，已自动回退 FP32")
            elif runtime_device.type == "cpu":
                print("⚠️  CPU 对 BF16 推理支持有限，已自动回退 FP32")
            elif runtime_device.type == "mps":
                print("⚠️  MPS 可能不支持 bfloat16，尝试降级；若报错请用 CPU FP32")
                model = model.bfloat16()
                input_dtype = torch.bfloat16
            else:
                model = model.bfloat16()
                input_dtype = torch.bfloat16

        model.eval()
        print(f"使用设备: {runtime_device}")
        print(f"模型已从 {args.model} 加载\n")

    # 交互式推理
    print("=" * 50)
    print("交互式推理 — 输入 32 位有符号整数，模型判断正负")
    print("范围: [-2147483648, 2147483647]，输入 q 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n请输入一个整数: ").strip()
            if user_input.lower() == "q":
                print("再见！")
                break

            value = int(user_input)
            if value < -2147483648 or value > 2147483647:
                print("⚠️  超出 32 位有符号整数范围，请重试")
                continue

            is_positive, conf = predict(model, value, runtime_device, input_dtype)
            actual = "正数（>=0）" if value >= 0 else "负数（<0）"
            prediction = "正数" if is_positive else "负数"

            correct = (value >= 0) == is_positive
            mark = "✅" if correct else "❌"
            print(f"  预测: {prediction}  |  实际: {actual}  |  置信度: {conf:.2%}  {mark}")

        except ValueError:
            print("⚠️  请输入有效的整数")
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    main()
