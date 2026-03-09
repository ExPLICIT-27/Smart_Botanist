import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, validation_split=0.2):
    # Enhanced Transforms for 96x96 input resolution to prevent overfitting
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
    
    # Calculate lengths
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    
    # Split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply correct transform to val dataset
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, full_dataset.classes

def build_model(num_classes):
    # MobileNetV2 with smaller width to stay under 1.5MB 
    # (must train from scratch because ImageNet weights don't fit width 0.35)
    model = models.mobilenet_v2(width_mult=0.35)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
    return model

def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    # Added weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Added Learning Rate Scheduler to drop LR when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    model.to(device)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Step the scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model!")
            
    # Load best model and return
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    return model

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on Plant/Flower dataset")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--output', type=str, default="flower_mobilenet.pt", help="Path to save the final model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading data from {args.dataset}...")
    train_loader, val_loader, classes = get_data_loaders(args.dataset, args.batch_size)
    print(f"Found {len(classes)} classes: {classes}")
    
    print("Building mobilenet model...")
    model = build_model(len(classes))
    
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, args.epochs, device)
    
    # Save the final model
    torch.save({
        'state_dict': model.state_dict(),
        'num_classes': len(classes),
        'classes': classes,
        'arch': 'mobilenet'
    }, args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
