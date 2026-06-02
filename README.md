# DNN 深度学习入门示例

通过 PyTorch 实现的深度神经网络（DNN）教学项目，用小型实验展示神经网络的核心概念。

## 项目简介

本项目包含一个 **DNN 符号分类器**，通过三层全连接网络判断 32 位有符号整数是正数还是负数。虽然问题本身很简单（只需看最高位），但作为 DNN 入门示例，它完整展示了：

- 数据预处理（整数 → 32 位二进制特征）
- 模型定义（`nn.Module` 子类化）
- 训练循环（前向传播 → 损失计算 → 反向传播 → 参数更新）
- 验证评估（准确率、损失曲线）
- 权重分析（验证网络学到了什么）
- 交互式推理（PyTorch / OpenCV）
- 模型剪枝（L1 非结构化，稀疏度对比）
- 低精度量化导出（FP16 / BF16 / INT8 / INT4）
- ONNX 导出与 OpenCV 推理

## 环境要求

- Python 3.12+
- PyTorch >= 2.0
- NumPy >= 1.24
- 可选：`opencv-python`（OpenCV ONNX 推理）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python train.py

# 3. 运行推理（交互式）
python inference.py
```

程序会自动：
1. 生成 20000 条训练数据
2. 构建并训练神经网络，保存模型到 `sign_classifier.pth`
3. 导出 ONNX 模型到 `sign_classifier.onnx`
4. 打印权重分析（展示网络学会了 MSB 是关键特征）

## 交互示例

```
请输入一个整数: 100
  预测: 正数  |  实际: 正数（>=0）  |  置信度: 99.84%  ✅

请输入一个整数: -255
  预测: 负数  |  实际: 负数（<0）  |  置信度: 100.00%  ✅

请输入一个整数: q
再见！
```

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
输出层 (1) + Sigmoid      ← 输出 P(正数)，范围 [0, 1]
```

- **可训练参数**：4225
- **损失函数**：BCELoss（二元交叉熵）
- **优化器**：Adam（lr=0.001）
- **训练轮数**：30 epochs
- **批次大小**：128

## 核心知识点

### 1. 为什么网络能 100% 准确？

32 位有符号整数使用**补码（two's complement）**表示：

- 最高位（MSB，bit31）= 0 → 非负数（0 ~ 2147483647）
- 最高位（MSB，bit31）= 1 → 负数（-2147483648 ~ -1）

网络通过训练自动发现：**第一层中连接 MSB 的权重远大于其他位**（约 5-7 倍），说明它学会了"符号位决定正负"这一规律。

### 2. 数据表示

```python
def int_to_bits(value):
    """将 32 位有符号整数转为 32 位二进制特征"""
    unsigned = value & 0xFFFFFFFF     # 取低 32 位，保留补码位模式
    bits = [(unsigned >> (31 - i)) & 1 for i in range(32)]
    return np.array(bits, dtype=np.float32)
```

| 十进制 | 32 位补码表示（MSB 前 16 位...） | 标签 |
|--------|----------------------------------|------|
| 5 | `00000000 00000000 ... 00000101` | 正数 (1) |
| -5 | `11111111 11111111 ... 11111011` | 负数 (0) |
| 2147483647 | `01111111 11111111 ... 11111111` | 正数 (1) |
| -2147483648 | `10000000 00000000 ... 00000000` | 负数 (0) |

### 3. 训练流程

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
├── model.py                    # 模型定义、设备选择、数据编码、ONNX 导出、torch.fx 分析
├── train.py                    # 数据生成、训练循环、权重分析、自动导出 ONNX
├── prune.py                    # L1 非结构化剪枝、稀疏度对比
├── inference.py                # PyTorch 模型加载、预测、交互式推理（含量化模型）
├── quantize.py                 # 低精度/量化导出（FP16/BF16/INT8/INT4）
├── opencv_inference.py         # OpenCV DNN 模块加载 ONNX 进行推理
├── requirements.txt            # Python 依赖
├── CLAUDE.md                   # Claude Code 项目指引
├── ROADMAP.md                  # DNN 学习路线图
└── README.md                   # 本文件
```

各文件各司其职，体现 **模型 / 训练 / 剪枝 / 量化 / 推理** 分离的设计思想：
- `model.py` 是共享核心，提供设备检测、编码函数、`BitChecker` 模型、ONNX 导出、torch.fx 分析
- `train.py` 从 `model` 导入，训练后保存 `.pth` 权重并自动导出 `.onnx`
- `prune.py` 加载 `.pth`，执行 L1 非结构化剪枝，对比剪枝前后效果
- `inference.py` 从 `model` + `quantize` 导入，支持 PyTorch 原生模型和量化模型的交互式推理
- `quantize.py` 从 `model` + `train` 导入，导出 FP16/BF16/INT8/INT4 低精度模型
- `opencv_inference.py` 使用 OpenCV DNN 加载 ONNX 模型，批量/交互式推理

## 模型剪枝

支持对训练好的模型进行 L1 非结构化剪枝，移除不重要的权重：

```bash
# 剪掉 30% 权重（推荐，无精度损失）
python prune.py --amount 0.3

# 剪掉 50% 权重
python prune.py --amount 0.5

# 剪掉 90% 权重（极端测试）
python prune.py --amount 0.9
```

| 剪枝比例 | 准确率 | 稀疏度 |
|----------|--------|--------|
| 0% | 100.00% | 0% |
| 30% | 100.00% | 30% |
| 50% | 95.25% | 50% |
| 90% | 51.00% | 90% |

剪枝后的模型保存为 `sign_classifier_pruned.pth`。

## 低精度与量化保存

支持将模型导出为更小文件：

```bash
# 导出 FP16 权重 + INT8 动态量化模型
python quantize.py
```

导出后会生成：
- `sign_classifier_fp16.pth`：FP16 权重（体积通常约为 FP32 的 1/2）
- `sign_classifier_bf16.pth`：BF16 权重（与 FP16 同体积，但数值范围更宽，精度损失更小）
- `sign_classifier_int8.pth`：INT8 动态量化 TorchScript（体积通常更小）
- `sign_classifier_int4.pth`：INT4 对称量化（自定义打包，2 个 4 位值 → 1 个字节，约 1/4 体积）

> 注：INT8 动态量化依赖 PyTorch 量化后端（如 qnnpack/fbgemm）。若当前环境不支持，脚本会自动跳过 INT8，仅导出 FP16/BF16。

并打印基线/低精度模型准确率和文件大小对比。

## ONNX 导出与 OpenCV 推理

训练完成后会自动导出 ONNX 模型，可以使用 PyTorch 或 OpenCV 进行推理：

### PyTorch 推理

```bash
# 原始模型
python inference.py -m sign_classifier.pth

# 剪枝模型
python inference.py -m sign_classifier_pruned.pth

# 量化模型
python inference.py -m sign_classifier_fp16.pth --fp16-infer
python inference.py -m sign_classifier_int4.pth
```

### OpenCV 推理

使用 OpenCV DNN 模块加载 ONNX 模型，无需 PyTorch 依赖：

```bash
# 安装 opencv-python
pip install opencv-python

# 运行推理
python opencv_inference.py --model sign_classifier.onnx
```

```
             0 → 预测: 正数  实际: 正数  置信度: 99.69%  ✅
             1 → 预测: 正数  实际: 正数  置信度: 99.81%  ✅
            -1 → 预测: 负数  实际: 负数  置信度: 100.00%  ✅
    2147483647 → 预测: 正数  实际: 正数  置信度: 100.00%  ✅
   -2147483648 → 预测: 负数  实际: 负数  置信度: 99.99%  ✅
```

> 提示：ONNX 模型可跨平台部署，支持 C++/Java/JavaScript/C# 等语言的 OpenCV、ONNX Runtime 推理。

## 扩展方向

理解本示例后，可以尝试：

1. **简化网络**：去掉隐藏层，用单层感知机（Logistic Regression）解决，观察效果
2. **增加噪声**：在输入位上随机翻转，测试网络鲁棒性
3. **更复杂的任务**：判断数字的奇偶性、是否为 2 的幂、绝对值是否大于某个阈值等
4. **CNN 视角**：将 16 位视为 1D 序列，用 Conv1d 处理
5. **直接输入数值**：改为输入单个归一化后的整数值，对比两种输入方式的区别

更多学习建议见 [ROADMAP.md](ROADMAP.md)。

## 许可

MIT License
