# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DNN (Deep Neural Network) learning project using PyTorch. Contains example programs that demonstrate deep neural network concepts through small, focused experiments.

## Environment

- Python 3.12+ with PyTorch and NumPy
- Install: `pip install -r requirements.txt`
- Train: `python train.py`
- Prune: `python prune.py --amount 0.3`
- Inference: `python inference.py`

## Project Structure

```
model.py      — 模型定义、设备选择、数据编码（int_to_bits）
train.py      — 数据生成、训练循环、评估、权重分析
prune.py      — L1 非结构化剪枝，稀疏度统计，剪枝前后对比
inference.py  — 模型加载、预测、交互式推理
```

The project follows a **model / train / prune / inference** separation:
- `model.py` is the shared core (no dependencies beyond torch + numpy)
- `train.py` imports from `model`, trains, and saves `sign_classifier.pth`
- `prune.py` imports from `model` + `train`, loads `.pth`, applies L1 unstructured pruning, saves `sign_classifier_pruned.pth`
- `inference.py` imports from `model`, loads the `.pth` file, and runs interactive prediction

## Code Conventions

- Models inherit `nn.Module`, use `nn.Sequential` for simple feedforward architectures
- Training uses `BCELoss` + `Adam`, data wrapped in `TensorDataset` + `DataLoader`
- Random seeds (42) are set in `train.py` for reproducibility
- Device priority: CUDA → Apple MPS → CPU (set once in `model.py`)

## Key Technical Notes

- 16-bit signed integers are represented as 16 binary bits in two's complement (MSB first), not as a single normalized scalar. This lets the network "discover" the sign bit.
- The first-layer weight magnitude on bit 15 (MSB) vs other bits is used to verify the network learned correctly.
- `*.pth` is gitignored — run `python train.py` to regenerate before inference.
