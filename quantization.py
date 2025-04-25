import torch
import torch.quantization
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from get_dataset import Four4All 
import os

# Load pretrained DeiT
def load_deit_model():
    teacher_model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=7)  # Change num_classes if needed
    teacher_model.eval()  # Set the model to evaluation mode
    teacher_model.load_state_dict(torch.load('/home/qinh3/MCNC_pretrain/cvhw/models/pretrained_deit/model.pth'))
    return teacher_model

# Function to apply dynamic quantization
def apply_dynamic_quantization(model):
    # Quantize Lin. and Conv. layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d},  
        dtype=torch.qint8 
    )
    return quantized_model

# Load DeiT and Quantize
teacher_model = load_deit_model()
quantized_teacher_model = apply_dynamic_quantization(teacher_model)

def print_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.2f} million")

def check_memory_usage(model):
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_memory / (1024 ** 2)


print("Original Model Size:")
print_model_size(teacher_model)
original_memory = check_memory_usage(teacher_model)
print(f"Original Model Memory Usage: {original_memory:.2f} MB")

print("Quantized Model Size:")
print_model_size(quantized_teacher_model)
quantized_memory = check_memory_usage(quantized_teacher_model)
print(f"Quantized Model Memory Usage: {quantized_memory:.2f} MB")

def get_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) 
    ])
    
    base_path = '/home/qinh3/MCNC_pretrain/cvhw/rafdb_augmented'
    test_csv = '/home/qinh3/MCNC_pretrain/cvhw/rafdb_augmented/test_labels.csv'
    
    test_dataset = Four4All(csv_file=test_csv, img_dir=base_path, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

def evaluate_quantized_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

test_loader = get_test_loader(batch_size=64)

device = torch.device("cpu")
quantized_teacher_model = quantized_teacher_model.to(device)

torch.save(quantized_teacher_model.state_dict(), '/home/qinh3/MCNC_pretrain/cvhw/models/quantized/quantized_deit_0.2M.pth')
torch.save(quantized_teacher_model, '/home/qinh3/MCNC_pretrain/cvhw/models/quantized/quantized_deit_full.pth')

accuracy = evaluate_quantized_model(quantized_teacher_model, test_loader, device=device)
print(f"Quantized Model Accuracy: {accuracy:.4f}")
