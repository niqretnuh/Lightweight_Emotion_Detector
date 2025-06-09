import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models, transforms, datasets
import timm
from torch.utils.data import DataLoader
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import argparse
from get_dataset import Four4All
from cnn import resnet56, resnet20
from baseline import BaselineModel

# Main train logic
def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, save_path='/home/qinh3/MCNC_pretrain/cvhw/models/res20'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    os.makedirs(save_path, exist_ok=True)

    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    save_file = os.path.join(save_path, f"model.pth")
    torch.save(model.state_dict(), save_file)
    
    # Save files
    with open(os.path.join(save_path, "train_loss.txt"), 'w') as f:
        for val in train_losses:
            f.write(f"{val:.6f}\n")
    with open(os.path.join(save_path, "train_accuracy.txt"), 'w') as f:
        for val in train_accuracies:
            f.write(f"{val:.6f}\n")
    with open(os.path.join(save_path, "val_accuracy.txt"), 'w') as f:
        for val in val_accuracies:
            f.write(f"{val:.6f}\n")

# Evaluate
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Main loop
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test various models')
    parser.add_argument('--model-type', type=str, required=True, choices=["pretrained_res18", "res18", "res56", "deit", "pretrained_deit", "ResEmoteNet", "baseline"],
                        help="Choose the model to train.")
    parser.add_argument('--save-dir', type=str, required=True, help="Directory to save model checkpoints and results.")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training.")
    args = parser.parse_args()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Paths to data
    '''
    base_path = 'data/rafdb'
    train_csv = 'data/train_labels.csv'
    test_csv = 'data/test_labels.csv'
    '''
    
    base_path = 'data/rafdb_augmented'
    train_csv = 'data/train_labels.csv'
    test_csv = 'data/test_labels.csv'

    model_type = args.model_type 

    # Preprocess using our predefined dataloaders
    train_dataset = Four4All(csv_file=train_csv, img_dir=base_path, transform=transform)
    test_dataset = Four4All(csv_file=test_csv, img_dir=base_path, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=True)

    if model_type == "pretrained_res18":
        model = models.resnet18(pretrained=True) 
        model.fc = nn.Linear(model.fc.in_features, 7) 
    elif model_type == "res18":
        model = models.resnet18(pretrained=False) 
        model.fc = nn.Linear(model.fc.in_features, 7)
    elif model_type == "res56":
        model = resnet56(num_classes=7) 
    elif model_type == "deit":
        model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=7).cuda()
    elif model_type == "pretrained_deit":
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=7).cuda() 
    elif model_type == "res5620":
        model = resnet56(num_classes=7)
    elif model_type == "baseline":
        model = BaselineModel().cuda()
    else:
        raise ValueError("Invalid model type")

    train_model(model, train_loader, test_loader, num_epochs=args.epochs, lr=args.lr, save_path=args.save_dir)
