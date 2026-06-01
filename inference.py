"""
DNN 符号分类器 - 推理脚本
=========================
加载已训练模型，交互式判断 16 位有符号整数的正负。

用法：
    python inference.py
"""

import torch

from model import DEVICE, SignClassifier, int_to_bits

MODEL_PATH = "sign_classifier.pth"


# ============================================================
#  预测
# ============================================================

def predict(model: torch.nn.Module, value: int) -> tuple[bool, float]:
    """
    对用户输入的整数进行预测。

    Returns:
        (is_positive, confidence): 是否为正数，以及置信度 [0, 1]
    """
    model.eval()
    bits = torch.from_numpy(int_to_bits(value)).unsqueeze(0).to(DEVICE)
    prob = model(bits).item()
    is_positive = prob >= 0.5
    confidence = prob if is_positive else (1.0 - prob)
    return is_positive, confidence


# ============================================================
#  主程序
# ============================================================

def main():
    print(f"使用设备: {DEVICE}")

    # 加载模型
    model = SignClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"模型已从 {MODEL_PATH} 加载\n")

    # 交互式推理
    print("=" * 50)
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
            print(f"  预测: {prediction}  |  实际: {actual}  |  置信度: {conf:.2%}  {mark}")

        except ValueError:
            print("⚠️  请输入有效的整数")
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    main()
