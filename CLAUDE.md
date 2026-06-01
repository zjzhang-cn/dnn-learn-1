# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DNN (Deep Neural Network) learning project using PyTorch. Contains self-contained example programs that demonstrate deep neural network concepts through small, focused experiments.

## Environment

- Python 3.12+ with PyTorch and NumPy
- Install: `pip install -r requirements.txt`
- Run: `python dnn_sign_classifier.py`
- Quick test (no interactive prompt): `python -c "from dnn_sign_classifier import *; ..."`

## Code Conventions

Each `.py` file is a self-contained experiment following this structure:
1. Imports + random seed + device selection
2. Data generation (usually as standalone functions)
3. Model class inheriting `nn.Module` (with `nn.Sequential` for simple architectures)
4. Training function with per-epoch validation
5. `@torch.no_grad()` evaluation helper
6. Inference function for interactive use
7. `main()` entry point orchestrating all steps

Models use `nn.Sequential` for simple feedforward architectures. Training uses `BCELoss` + `Adam`. Data is wrapped in `TensorDataset` + `DataLoader`.

## Files

| File | Purpose |
|------|---------|
| `dnn_sign_classifier.py` | 16-bit signed integer → positive/negative classifier. Input: 16 binary bits (two's complement). Architecture: 16→32→16→1 with ReLU + Sigmoid. Includes weight analysis showing the network learns MSB matters most. |
| `README.md` | Chinese documentation for the project. |

## Key Technical Notes

- 16-bit signed integers are represented as 16 binary bits in two's complement (MSB first), not as a single normalized scalar. This lets the network "discover" the sign bit.
- The first-layer weight magnitude on bit 15 (MSB) vs other bits is used to verify the network learned correctly.
- All scripts set random seeds (42) for reproducibility.
