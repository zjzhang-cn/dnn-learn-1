"""
DNN 符号分类器 - 低精度/量化导出
====================================
导出四种更小的模型文件：
1) FP16 权重（state_dict）
2) BF16 权重（state_dict）
3) INT8 动态量化（TorchScript）
4) INT4 对称量化（自定义打包，2 个 4 位值 → 1 个 uint8）

用法：
    python quantize.py
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn

from model import DEVICE, SignClassifier
from train import evaluate, generate_dataset

MODEL_PATH = "sign_classifier.pth"
FP16_PATH = "sign_classifier_fp16.pth"
BF16_PATH = "sign_classifier_bf16.pth"
INT8_PATH = "sign_classifier_int8.pth"
INT4_PATH = "sign_classifier_int4.pth"


def setup_quant_engine() -> tuple[bool, str]:
    """检查并配置动态量化后端。"""
    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    if not engines:
        return False, "当前 PyTorch 未编译量化引擎"

    current = getattr(torch.backends.quantized, "engine", "none")
    if current != "none":
        return True, f"量化引擎: {current}"

    for candidate in ("qnnpack", "fbgemm", "x86"):
        if candidate in engines:
            torch.backends.quantized.engine = candidate
            return True, f"量化引擎: {candidate}"

    return False, f"可用引擎: {engines}，但未找到可用候选"


def evaluate_jit(model: torch.jit.ScriptModule, X: torch.Tensor, y: torch.Tensor) -> dict:
    """评估 TorchScript 模型准确率和损失。"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = nn.BCELoss()(outputs, y).item()
        preds = (outputs >= 0.5).float()
        acc = (preds == y).float().mean().item()
    return {"loss": loss, "acc": acc}


def file_size(path: str) -> int:
    return os.path.getsize(path)


# ============================================================
#  INT4 对称量化（自定义打包）
# ============================================================

def pack_int4(weight: torch.Tensor) -> dict:
    """
    对称 INT4 量化并打包。

    量化公式: q = round(clip(w / scale, -7, 7))
              scale = max(|w|) / 7

    打包: 两个 int4 值拼成一个 uint8（高 4 位在前）
    如果元素数为奇数，末尾补 0。

    Returns:
        {"scale": float, "packed": uint8 tensor, "shape": tuple, "numel": int}
    """
    w = weight.detach().cpu().float()
    max_abs = w.abs().max().item()
    if max_abs == 0:
        max_abs = 1e-8
    scale = max_abs / 7.0

    # 量化到 [-7, 7]
    q = torch.clamp(torch.round(w / scale), -7, 7).to(torch.int8)

    # 转为无符号偏移: [-7,7] → [0,14]
    q_unsigned = (q + 7).to(torch.uint8)

    # 展平并补齐到偶数
    flat = q_unsigned.flatten()
    numel = flat.numel()
    if numel % 2 != 0:
        flat = torch.cat([flat, torch.tensor([7], dtype=torch.uint8)])  # 补一个 7(=0)
        numel += 1

    # 打包: 两个 uint8→一个 uint8 (高4位|低4位)
    packed = torch.zeros(numel // 2, dtype=torch.uint8)
    packed = (flat[0::2] << 4) | flat[1::2]

    return {
        "scale": scale,
        "packed": packed,
        "shape": tuple(w.shape),
        "numel": w.numel(),
    }


def unpack_int4(packed_info: dict) -> torch.Tensor:
    """从打包的 INT4 数据还原 float32 权重。"""
    scale = packed_info["scale"]
    packed = packed_info["packed"]
    shape = packed_info["shape"]
    numel = packed_info["numel"]

    # 解包: 每个 uint8 拆成高 4 位和低 4 位
    high = (packed >> 4).to(torch.uint8)
    low = (packed & 0x0F).to(torch.uint8)
    flat = torch.stack([high, low], dim=1).flatten()[:numel]

    # 偏移还原: [0,14] → [-7,7]
    q = flat.to(torch.float32) - 7.0

    # 反量化
    w = q * scale
    return w.reshape(shape)


def evaluate_int4(model: nn.Module, int4_state: dict, X: torch.Tensor, y: torch.Tensor) -> dict:
    """用 INT4 解包权重评估模型准确率。"""
    state_dict = {}
    for key, value in int4_state.items():
        if key == "__format__":
            continue
        if isinstance(value, dict) and "packed" in value:
            state_dict[key] = unpack_int4(value)
        else:
            state_dict[key] = value

    temp_model = SignClassifier().to(X.device)
    temp_model.load_state_dict(state_dict)
    temp_model.eval()
    return evaluate(temp_model, X, y)


def main():
    print(f"训练/评估设备: {DEVICE}")

    # 1) 加载原始模型
    print("\n[1/6] 加载原始模型...")
    base_model = SignClassifier().to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base_model.eval()

    # 2) 准备验证集并评估基线准确率
    print("[2/6] 准备验证集并评估基线...")
    _, _, X_val, y_val = generate_dataset(4000)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)
    base_metrics = evaluate(base_model, X_val_t, y_val_t)
    print(f"  基线准确率: {base_metrics['acc']:.2%}")

    # 3) 导出 FP16 state_dict
    print("\n[3/6] 导出 FP16 权重...")
    fp16_model = copy.deepcopy(base_model).to("cpu").half().eval()
    torch.save(fp16_model.state_dict(), FP16_PATH)

    # 为了可比性，按原路径加载并在 float32 上评估精度
    fp16_reload = SignClassifier().to(DEVICE)
    fp16_reload.load_state_dict(torch.load(FP16_PATH, map_location=DEVICE))
    fp16_reload.eval()
    fp16_metrics = evaluate(fp16_reload, X_val_t, y_val_t)
    print(f"  FP16 文件已保存: {FP16_PATH}")
    print(f"  FP16(重载后)准确率: {fp16_metrics['acc']:.2%}")

    # 3.5) 导出 BF16 state_dict
    print("\n[4/6] 导出 BF16 权重...")
    bf16_exported = False
    bf16_metrics = None
    try:
        bf16_model = copy.deepcopy(base_model).to("cpu").bfloat16().eval()
        torch.save(bf16_model.state_dict(), BF16_PATH)

        bf16_reload = SignClassifier().to(DEVICE)
        bf16_reload.load_state_dict(torch.load(BF16_PATH, map_location=DEVICE))
        bf16_reload.eval()
        bf16_metrics = evaluate(bf16_reload, X_val_t, y_val_t)
        bf16_exported = True
        print(f"  BF16 文件已保存: {BF16_PATH}")
        print(f"  BF16(重载后)准确率: {bf16_metrics['acc']:.2%}")
    except Exception as e:
        print(f"  ⚠️  跳过 BF16 导出：{e}")

    # 5) 导出 INT8 动态量化 TorchScript
    print("\n[5/7] 导出 INT8 动态量化模型...")
    int8_exported = False
    int8_metrics = None
    ok, engine_msg = setup_quant_engine()
    if not ok:
        print(f"  ⚠️  跳过 INT8 导出：{engine_msg}")
    else:
        print(f"  {engine_msg}")
        try:
            cpu_model = copy.deepcopy(base_model).to("cpu").eval()
            quantized_model = torch.ao.quantization.quantize_dynamic(
                cpu_model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            scripted = torch.jit.script(quantized_model)
            scripted.save(INT8_PATH)

            X_val_cpu = X_val_t.to("cpu")
            y_val_cpu = y_val_t.to("cpu")
            int8_metrics = evaluate_jit(scripted, X_val_cpu, y_val_cpu)
            int8_exported = True
            print(f"  INT8 文件已保存: {INT8_PATH}")
            print(f"  INT8 准确率: {int8_metrics['acc']:.2%}")
        except Exception as e:
            print(f"  ⚠️  跳过 INT8 导出：{e}")

    # 6) 导出 INT4 对称量化（自定义打包）
    print("\n[6/7] 导出 INT4 对称量化权重...")
    int4_exported = False
    int4_metrics = None
    try:
        int4_state = {"__format__": "int4"}
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                int4_state[f"{name}.weight"] = pack_int4(module.weight.data)
                if module.bias is not None:
                    int4_state[f"{name}.bias"] = module.bias.data.cpu()
        torch.save(int4_state, INT4_PATH)

        int4_metrics = evaluate_int4(base_model, int4_state, X_val_t, y_val_t)
        int4_exported = True
        print(f"  INT4 文件已保存: {INT4_PATH}")
        print(f"  INT4 准确率: {int4_metrics['acc']:.2%}")
    except Exception as e:
        print(f"  ⚠️  跳过 INT4 导出：{e}")

    # 7) 对比文件大小
    print("\n[7/7] 文件大小对比...")
    base_size = file_size(MODEL_PATH)
    fp16_size = file_size(FP16_PATH)

    print(f"  原始 FP32: {base_size:,} bytes")
    print(f"  FP16:      {fp16_size:,} bytes  (压缩比 {base_size / fp16_size:.2f}x)")
    if bf16_exported:
        bf16_size = file_size(BF16_PATH)
        print(f"  BF16:      {bf16_size:,} bytes  (压缩比 {base_size / bf16_size:.2f}x)")
    else:
        print("  BF16:      未导出")
    if int8_exported:
        int8_size = file_size(INT8_PATH)
        print(f"  INT8:      {int8_size:,} bytes  (压缩比 {base_size / int8_size:.2f}x)")
    else:
        print("  INT8:      未导出（当前环境不支持动态量化后端）")
    if int4_exported:
        int4_size = file_size(INT4_PATH)
        print(f"  INT4:      {int4_size:,} bytes  (压缩比 {base_size / int4_size:.2f}x)")
    else:
        print("  INT4:      未导出")

    print("\n" + "=" * 60)
    print("低精度/量化导出完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
