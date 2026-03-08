import os
import argparse
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def evaluate_ptl(model_path, data_dir, sample_size=500):
    print(f"Loading PyTorch Lite (.ptl) model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()
    
    print(f"Preparing a random subset of {sample_size} test images for evaluation...")
    transform_val = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(data_dir, transform=transform_val)
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:sample_size]
    subset = Subset(full_dataset, subset_indices)
    
    test_loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    print("Evaluating model...")
    device = torch.device('cpu')
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
    final_acc = 100. * correct / total
    print(f"\nFinal PyTorch Lite Model Accuracy: {final_acc:.2f}% on {total} test images.")
    return final_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate Edge .ptl on Test Data")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained .ptl model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the test dataset directory")
    parser.add_argument('--sample_size', type=int, default=500, help="Number of random samples to evaluate (0 for all)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' does not exist.")
        return
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return
        
    sample_size = args.sample_size
    if sample_size == 0:
        # Evaluate all
        sample_size = len(datasets.ImageFolder(args.dataset))
        
    evaluate_ptl(args.model, args.dataset, sample_size)

if __name__ == "__main__":
    main()
