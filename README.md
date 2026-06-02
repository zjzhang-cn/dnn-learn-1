# DNN 深度学习入门示例

通过 PyTorch 实现的深度神经网络教学项目，用小型实验展示神经网络的核心概念。统一使用 `BitClassifier`（32→64→32→1）模型，通过不同数据集训练解决多个任务。

## 项目简介

本项目通过四个难度递进的二分类任务，展示 DNN 从"单 bit 问题"到"位模式识别"的学习过程：

| 任务 | 脚本 | 关键位 | 难度 | 核心概念 |
|------|------|--------|------|---------|
| 符号判断 | `train.py` | MSB（bit31） | ★ | 入门：单 bit 线性可分 |
| 奇偶判断 | `parity_train.py` | LSB（bit0） | ★ | 同上，换个 bit |
| 2 的幂判断 | `power_of_two_train.py` | 所有位 | ★★ | 位计数、困难负样本 |
| 绝对值比较 | `abs_compare_train.py` | 高位 | ★★ | 非线性决策边界 |

此外还涵盖：
- 数据预处理（整数 → 32 位二进制特征）
- 权重分析（验证网络学到了哪个 bit）
- 交互式推理（PyTorch / OpenCV）
- 模型剪枝（L1 非结构化，稀疏度对比）
- 低精度量化导出（FP16 / BF16 / INT8 / INT4）
- ONNX 导出与跨平台部署

## 环境要求

- Python 3.12+
- PyTorch >= 2.0
- NumPy >= 1.24
- 可选：`opencv-python`（OpenCV ONNX 推理）

## 快速开始

```bash
pip install -r requirements.txt

# 符号判断（最基础的入门任务）
python train.py

# 奇偶判断
python parity_train.py

# 2 的幂判断（位计数）
python power_of_two_train.py

# 绝对值比较（非线性决策边界）
python abs_compare_train.py --threshold 1000
```

每个训练脚本会自动：
1. 生成 20000 条训练数据
2. 构建 `BitClassifier` 并训练
3. 打印权重分析（验证网络学到了什么）
4. 保存模型到 `.pth`

## 模型结构

```
输入层 (32)               ← 32 位二进制表示（补码，MSB 在前）
    ↓
全连接层 (64) + ReLU      ← 扩展到 64 维，学习特征组合
    ↓
Dropout (0.2)             ← 防止过拟合
    ↓
全连接层 (32) + ReLU      ← 压缩到 32 维
    ↓
输出层 (1) + Sigmoid      ← 输出概率，范围 [0, 1]
```

- **模型**：`BitClassifier`（统一模型，所有任务共用）
- **可训练参数**：4,225
- **损失函数**：BCELoss（二元交叉熵）
- **优化器**：Adam（lr=0.001）
- **批次大小**：128

## 核心知识点

### 1. 为什么单 bit 任务能 100% 准确？

32 位有符号整数使用**补码（two's complement）**表示：

| 十进制 | 32 位补码表示 | 标签（正负） |
|--------|--------------|-------------|
| 5 | `00000000 00000000 ... 00000101` | 正数 |
| -5 | `11111111 11111111 ... 11111011` | 负数 |
| 2147483647 | `01111111 11111111 ... 11111111` | 正数 |
| -2147483648 | `10000000 00000000 ... 00000000` | 负数 |

网络通过训练自动发现关键位：符号任务中 MSB 权重是其余位的 5~7 倍，奇偶任务中 LSB 权重是其余位的 5~7 倍。

### 2. 权重分析：不同任务学到不同的位

| 任务 | 关键位权重 | 其余位 | 结论 |
|------|-----------|--------|------|
| 符号判断 | MSB = 0.34 | 0.07 | 只看 bit31 |
| 奇偶判断 | LSB = 0.37 | 0.07 | 只看 bit0 |
| 2 的幂 | 均匀 ~0.12 | 0.12 | 对所有位"计数" |
| 绝对值 | 高位 > 低位 | — | 关注数值大小 |

### 3. 数据表示

```python
def int_to_bits(value):
    """将 32 位有符号整数转为 32 位二进制特征"""
    unsigned = value & 0xFFFFFFFF     # 取低 32 位，保留补码位模式
    bits = [(unsigned >> (31 - i)) & 1 for i in range(32)]
    return np.array(bits, dtype=np.float32)
```

### 4. 训练流程

```
for epoch in range(epochs):
    for batch in data_loader:
        outputs = model(batch_X)       # 前向传播
        loss = criterion(outputs, y)   # 计算损失
        optimizer.zero_grad()          # 清零梯度
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数
```

## 项目结构

```
dnn-learn-1/
├── model.py                    # BitClassifier 模型、ONNX 导出、torch.fx 分析
├── train.py                    # 数据生成（4 种数据集）、训练、评估、权重分析
├── parity_train.py             # 奇偶判断训练
├── power_of_two_train.py       # 2 的幂判断训练
├── abs_compare_train.py        # 绝对值比较训练
├── prune.py                    # L1 非结构化剪枝
├── quantize.py                 # 低精度/量化导出（FP16/BF16/INT8/INT4）
├── inference.py                # PyTorch 推理（含量化模型）
├── opencv_inference.py         # OpenCV ONNX 推理
├── requirements.txt            # Python 依赖
├── CLAUDE.md                   # Claude Code 项目指引
├── ROADMAP.md                  # DNN 学习路线图
├── COMPARISON.md               # 16 位 vs 32 位模型对比
└── README.md                   # 本文件
```

各文件职责：

| 文件 | 职责 |
|------|------|
| `model.py` | 共享核心：`BitClassifier` 模型、`int_to_bits` 编码、设备检测、ONNX 导出、torch.fx 分析、`print_model` |
| `train.py` | 数据生成（4 种数据集）、`evaluate`、`show_learned_weights`、符号判断训练主程序 |
| `parity_train.py` | 奇偶判断训练（导入 `BitClassifier` + `generate_parity_dataset`） |
| `power_of_two_train.py` | 2 的幂判断训练（首个非单 bit 任务） |
| `abs_compare_train.py` | 绝对值比较训练（非线性决策边界） |
| `prune.py` | L1 非结构化剪枝，加载任意 `.pth`，稀疏度对比 |
| `quantize.py` | 低精度导出：FP16/BF16/INT8/INT4 |
| `inference.py` | 交互式推理，支持 PyTorch 模型 + 量化模型 + FP16/BF16 |
| `opencv_inference.py` | 使用 `cv2.dnn.readNetFromONNX` 加载 ONNX 进行推理 |

## 交互推理

```bash
# PyTorch 推理
python inference.py -m sign_classifier.pth
python inference.py -m parity_classifier.pth
python inference.py -m power_of_two_classifier.pth

# OpenCV ONNX 推理
python opencv_inference.py --model sign_classifier.onnx
```

```
请输入一个整数: 100
  预测: 正数  |  实际: 正数（>=0）  |  置信度: 99.84%  ✅

请输入一个整数: -255
  预测: 负数  |  实际: 负数（<0）  |  置信度: 100.00%  ✅

请输入一个整数: q
再见！
```

## 模型剪枝

支持对训练好的模型进行 L1 非结构化剪枝：

```bash
python prune.py --amount 0.3    # 剪掉 30%
python prune.py --amount 0.5    # 剪掉 50%
python prune.py --amount 0.9    # 极端测试
```

| 剪枝比例 | 准确率 | 稀疏度 |
|----------|--------|--------|
| 0% | 100.00% | 0% |
| 30% | 100.00% | 30% |
| 50% | 95.25% | 50% |
| 90% | 51.00% | 90% |

## 低精度与量化保存

```bash
python quantize.py
```

导出：`sign_classifier_fp16.pth` / `_bf16.pth` / `_int8.pth` / `_int4.pth`

> 注：INT8 动态量化依赖 PyTorch 量化后端（qnnpack/fbgemm），不支持时自动跳过。

## ONNX 导出与 OpenCV 推理

训练完成后自动导出 ONNX（`train.py` 内置），也可用 OpenCV 推理：

```bash
pip install opencv-python
python opencv_inference.py --model sign_classifier.onnx
```

```
             0 → 预测: 正数  实际: 正数  置信度: 99.69%  ✅
             1 → 预测: 正数  实际: 正数  置信度: 99.81%  ✅
            -1 → 预测: 负数  实际: 负数  置信度: 100.00%  ✅
```

> ONNX 模型可跨平台部署，支持 C++/Java/JavaScript/C# 等语言的 OpenCV、ONNX Runtime 推理。

## 下一步

- 更多任务和优化技巧见 [ROADMAP.md](ROADMAP.md)
- 16 位 vs 32 位模型剪枝对比见 [COMPARISON.md](COMPARISON.md)

## 许可

MIT License
