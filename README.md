# Smart Botanist TinyML Project

This repository contains the model training and quantization pipeline for the **Smart Botanist** project, which aims to deploy highly-accurate PyTorch models to an ESP32 microcontroller with a strict **<1.5MB** file size constraint.

The pipeline utilizes **MobileNetV2 (width 0.35)** and **Post-Training Static Quantization (PTQ)** to achieve high precision INT8 `.ptl` (PyTorch Lite) edge models.

## Dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Requires Python 3.10+ due to recent PyTorch Mobile `torch.jit` optimizations)*

## Scripts Overview

### 1. Training (`train.py`)
Trains a small-footprint `MobileNetV2` from scratch. Because the baseline mobile architecture `weights='DEFAULT'` does not match a customized sub-1MB scaled width, training from scratch is required to generate the underlying `FP32` `.pt` model file.
```bash
# Example usage
python train.py --dataset "Data/flowers" --output "flower_mobilenet.pt" --epochs 30
```

### 2. Quantization (`quantize.py`)
Applies **Post-Training Static Quantization (PTQ)** using `torchvision.models.quantization`. 
1. Absorbs the original `.pt` model into a QNNPACK-ready framework.
2. Fuses layers (`Conv2d` + `BatchNorm` + `ReLU`) to minimize edge inference calculation time.
3. Automatically calibrates the INT8 scalar parameters dynamically using a requested dataset prior to compression to avoid precision loss resulting in randomly guessed 0% accuracy traces.
4. Traces the calibrated INT8 map out as an optimized `.ptl` (PyTorch Mobile Lite).
```bash
# Example usage (needs the dataset to run the calibration pass)
python quantize.py --model "flower_mobilenet.pt" --dataset "Data/flowers" --output "flower_mobilenet.ptl"
```

### 3. Evaluation (`evaluate.py`)
Validates the INT8 precision loss using an evaluation dataset. 
Passes the dataset using `torch.jit.load` (instead of standard `torch.load` which cannot read `.ptl`). Automatically shuffles the dataset into subsets so evaluation time isn't blocked on the entire 4,000+ dataset during testing loops.
```bash
# Example usage
python evaluate.py --model "flower_mobilenet.ptl" --dataset "Data/flowers"
```

## Produced Edge Models
* **`flower_mobilenet.ptl`** (0.85 MB) - 83.00% Validation Accuracy on 5 Flower Classes.
* **`disease_model.ptl`** (0.89 MB) - 94.80% Validation Accuracy on 38 Leaf Pathologies.
