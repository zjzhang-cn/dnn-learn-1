"""
DNN 符号分类器 - OpenCV ONNX 推理
================================
使用 OpenCV DNN 模块加载 ONNX 模型进行推理。

用法：
    python opencv_inference.py                        # 默认模型
    python opencv_inference.py --model sign_classifier.onnx
"""

import argparse

import numpy as np
import cv2

from model import int_to_bits

DEFAULT_ONNX = "sign_classifier.onnx"


def load_onnx_model(model_path: str) -> cv2.dnn.Net:
    """加载 ONNX 模型，返回 cv2.dnn.Net。"""
    net = cv2.dnn.readNetFromONNX(model_path)
    print(f"✅ ONNX 模型已加载: {model_path}")
    return net


def predict(net: cv2.dnn.Net, value: int) -> tuple[bool, float]:
    """
    对单个 32 位有符号整数进行预测。

    Returns:
        (is_positive, confidence): 是否为正数，置信度 [0, 1]
    """
    bits = int_to_bits(value).reshape(1, 32).astype(np.float32)

    net.setInput(bits)
    output = net.forward()
    prob = float(output[0, 0])

    is_positive = prob >= 0.5
    confidence = prob if is_positive else (1.0 - prob)
    return is_positive, confidence


def batch_predict(net: cv2.dnn.Net, values: list[int]) -> list[tuple[int, bool, float]]:
    """
    批量预测。

    Returns:
        [(value, is_positive, confidence), ...]
    """
    batch = np.stack([int_to_bits(v) for v in values]).astype(np.float32)
    net.setInput(batch)
    outputs = net.forward()
    results = []
    for i, value in enumerate(values):
        prob = float(outputs[i, 0])
        is_positive = prob >= 0.5
        confidence = prob if is_positive else (1.0 - prob)
        results.append((value, is_positive, confidence))
    return results


def main():
    parser = argparse.ArgumentParser(description="OpenCV ONNX 模型推理")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_ONNX,
                        help=f"ONNX 模型路径（默认: {DEFAULT_ONNX}）")
    args = parser.parse_args()

    net = load_onnx_model(args.model)

    # 测试固定样本
    print("\n" + "=" * 50)
    print("批量测试")
    print("=" * 50)
    test_values = [
        0,           # 零
        1,           # 正数
        -1,          # 负数
        2147483647,  # 最大正数
        -2147483648, # 最小负数
        100,         # 小正数
        -255,        # 小负数
    ]
    results = batch_predict(net, test_values)
    for value, is_positive, conf in results:
        actual = "正数" if value >= 0 else "负数"
        prediction = "正数" if is_positive else "负数"
        correct = (value >= 0) == is_positive
        mark = "✅" if correct else "❌"
        print(f"  {value:>12} → 预测: {prediction}  实际: {actual}  置信度: {conf:.2%}  {mark}")

    # 交互式推理
    print("\n" + "=" * 50)
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

            is_positive, conf = predict(net, value)
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
