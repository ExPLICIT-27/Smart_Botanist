import os
import argparse
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models.quantization as qmodels
from torch.utils.data import DataLoader, Subset
from torch.utils.mobile_optimizer import optimize_for_mobile

def load_and_calibrate(model_path, data_dir, arch):
    print(f"Loading checkpoint from {model_path} into Quantizable Architecture...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    num_classes = checkpoint['num_classes']
    
    # Must use the quantizable version of MobileNetV2 for proper static quantization
    model = qmodels.mobilenet_v2(width_mult=0.35)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Fuse Conv2d + BatchNorm2d + ReLU for better accuracy and speed
    print("Fusing modules to prevent precision loss...")
    model.fuse_model()

    # Set up config for the standard QNNPACK mobile backend
    model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare(model, inplace=True)

    print("Calibrating the model using a representative dataset...")
    # Use the same transforms as validation
    transform_val = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform_val)
    
    # Randomly select a subset of images for calibration
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    calib_subset = Subset(full_dataset, indices[:200]) # 200 random images
    calib_loader = DataLoader(calib_subset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, _ in calib_loader:
            model(inputs)

    # Convert calibrated architecture back to quantized INT8 weights
    print("Applying INT8 Conversions...")
    torch.ao.quantization.convert(model, inplace=True)
    return model

def optimize_for_esp32(model, ptl_path):
    print("Tracing model with TorchScript (INT8)...")
    dummy_input = torch.randn(1, 3, 96, 96)
    traced_model = torch.jit.trace(model, dummy_input)
    
    print("Applying PyTorch Mobile optimizations...")
    optimized_model = optimize_for_mobile(traced_model)
    
    print(f"Exporting to PyTorch Lite format (.ptl) at {ptl_path}...")
    optimized_model._save_for_lite_interpreter(ptl_path)
    
    size_mb = os.path.getsize(ptl_path) / (1024 * 1024)
    print(f"\nOptimized .ptl model saved successfully!")
    print(f"Model File Size: {size_mb:.2f} MB")
    
    if size_mb < 1.5:
        print("SUCCESS! Model is under 1.5 MB requirement.")
    else:
        print("WARNING! Model exceeds 1.5 MB requirement.")

def main():
    parser = argparse.ArgumentParser(description="Optimize PyTorch model for Edge (.ptl)")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained .pt model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset for calibration")
    parser.add_argument('--output', type=str, default="model.ptl", help="Path to save the final .ptl model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' does not exist.")
        return
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return
        
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    arch = checkpoint.get('arch', 'mobilenet')
    
    quantized_model = load_and_calibrate(args.model, args.dataset, arch)
    optimize_for_esp32(quantized_model, args.output)

if __name__ == "__main__":
    main()
