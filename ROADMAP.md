# DNN 学习路线图

基于当前项目（符号分类器）的扩展建议。现有代码已覆盖：架构定义、训练循环、权重分析、剪枝、量化、ONNX 导出、OpenCV 推理。

---

## 一、已完成

| 任务 | 文件 | 说明 |
|------|------|------|
| ✅ 符号判断 | `train.py` | 看 MSB，单 bit 问题 |
| ✅ 奇偶判断 | `parity_train.py` | 看 LSB，同样是单 bit 问题 |
| ✅ 模型剪枝 | `prune.py` | L1 非结构化剪枝 |
| ✅ 低精度量化 | `quantize.py` | FP16 / BF16 / INT8 / INT4 |
| ✅ ONNX 导出 | `model.py` → `export_model_to_onnx` | 训练后自动导出 |
| ✅ OpenCV 推理 | `opencv_inference.py` | cv2.dnn 加载 ONNX |

---

## 二、下一步计划（优先级排序）

### ⭐⭐⭐ 任务 A：2 的幂判断（`power_of_two_train.py`）

判断 32 位有符号整数是否为 2 的幂（1, 2, 4, 8, 16, ...）。**不再是单 bit 问题**——网络需要识别位模式：恰好只有一个 1，其余全是 0。

| 维度 | 说明 |
|------|------|
| 输入 | 32 位二进制 |
| 输出 | 二分类（是 2 的幂 / 不是） |
| 难度 | 中——需要"数 bit 中有几个 1" |
| 数据 | `generate_power_of_two_dataset` 均匀采样 + 正样本专用生成 |
| 模型 | `BitClassifier`（32→64→32→1） |
| 看点 | 网络能否学会"只有一个 1"这个全局约束 |

### ⭐⭐⭐ 任务 B：绝对值比较（`abs_compare_train.py`）

判断 |x| > 1000？这是一个**非线性决策边界**，单层感知机无法完美解决。需要网络学会：忽略符号位、提取数值大小。

| 维度 | 说明 |
|------|------|
| 输入 | 32 位二进制 |
| 输出 | 二分类（|x| > 1000 / |x| ≤ 1000） |
| 难度 | 中——需要忽略 MSB，关注高位数值 |
| 数据 | `generate_abs_compare_dataset` + 阈值参数 |
| 模型 | `BitClassifier`（32→64→32→1） |
| 看点 | 决策边界可视化、对比单层 vs 多层 |

### ⭐⭐ 任务 C：多分类（`multiclass_train.py`）

正 / 负 / 零 三分类，引入 `CrossEntropyLoss` + softmax 输出，真正展示 DNN 的表达能力。

| 维度 | 说明 |
|------|------|
| 输入 | 32 位二进制 |
| 输出 | 三分类（正 / 负 / 零） |
| 难度 | 低——仍然是单 bit 问题（正负看 MSB，零判断所有位） |
| 模型 | 输出改为 3 维 |
| 看点 | 从 `BCELoss` + Sigmoid 到 `CrossEntropyLoss` + Softmax |

---

## 三、后续扩展（待排期）

| 优先级 | 内容 | 理由 |
|--------|------|------|
| ⭐⭐⭐ | 过拟合+正则化 | DNN 最核心概念，现有代码零覆盖 |
| ⭐⭐⭐ | 优化器对比 | 一行 `optim.SGD` 换 `optim.Adam` 就能演示 |
| ⭐⭐ | BatchNorm | 理解归一化在深度学习中的地位 |
| ⭐ | LR 调度 | 理解训练策略选择 |
| ⭐ | 激活函数对比 | 理解梯度消失的根源 |

### 过拟合 + 正则化对比

用小数据集（100~200 条）训练，展示过拟合现象（训练准确率 100%、验证只有 70%），然后用 L1/L2 正则化、增大 Dropout 来修复。

| 方案 | 实现方式 |
|------|---------|
| L1 正则 | `nn.L1Loss()` 叠加到总损失上 |
| L2 正则（Weight Decay） | `Adam(weight_decay=...)` |
| 增大 Dropout | `nn.Dropout(0.2)` → `nn.Dropout(0.5)` |
| 减少模型容量 | 隐藏层 64→32→1 改为 32→16→1 |

### 优化器对比

| 优化器 | 关键参数 | 特点 |
|--------|---------|------|
| SGD | lr=0.01 | 收敛慢，对学习率敏感 |
| SGD + Momentum | lr=0.01, momentum=0.9 | 加速收敛，减少震荡 |
| Adam | lr=0.001 | 自适应学习率，收敛最快 |
| RMSprop | lr=0.001 | 处理非平稳目标 |

### Batch Normalization

```
原结构: Linear → ReLU → Linear → ReLU → Linear → Sigmoid
BN 结构: Linear → BN → ReLU → Linear → BN → ReLU → Linear → Sigmoid
```

### 学习率调度

| 调度器 | 特点 |
|--------|------|
| Constant | 基准对照 |
| StepLR | 每隔 N 轮衰减 γ 倍 |
| CosineAnnealingLR | 余弦衰减，末期平滑 |
| ReduceLROnPlateau | 验证 loss 停滞时自动降 |

### 激活函数对比

| 激活函数 | 深层网络表现 |
|----------|-------------|
| ReLU | 收敛快，无梯度消失 |
| LeakyReLU | 对负值区域也保留梯度 |
| Tanh | 中心对称，深层仍有梯度消失 |
| Sigmoid | 深层几乎不收敛 |

---

## 四、当前文件结构

```
dnn-learn-1/
├── model.py                    # BitClassifier 统一模型、ONNX 导出、torch.fx 分析
├── train.py                    # 符号判断训练 + ONNX 自动导出
├── parity_train.py             # 奇偶判断训练
├── prune.py                    # L1 非结构化剪枝
├── quantize.py                 # 低精度量化导出
├── inference.py                # PyTorch 推理（含量化模型）
├── opencv_inference.py         # OpenCV ONNX 推理
├── power_of_two_train.py       # TODO: 2 的幂判断
├── abs_compare_train.py        # TODO: 绝对值比较
├── multiclass_train.py         # TODO: 多分类
├── requirements.txt            # Python 依赖
├── CLAUDE.md                   # Claude Code 项目指引
├── README.md                   # 项目说明
└── ROADMAP.md                  # 本文件
```
