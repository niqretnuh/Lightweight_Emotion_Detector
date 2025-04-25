import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import models
from tqdm import tqdm
import os
from get_dataset import Four4All
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from cnn import resnet20, resnet10, resnet14

# Define Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs, labels):
        # CE
        ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        # KL divergence
        soft_loss = nn.KLDivLoss()(F.log_softmax(student_outputs / self.temperature, dim=1),
                                   F.softmax(teacher_outputs / self.temperature, dim=1)) * (self.temperature ** 2)
        
        loss = self.alpha * ce_loss + (1. - self.alpha) * soft_loss
        return loss

# Function to train distillation model
def train_distilled_model(student_model, teacher_model, train_loader, val_loader, num_epochs=30, lr=0.001, save_path='/path/to/save'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # Set teacher model to evaluation mode
    teacher_model.eval()

    # Loss and optimizer
    criterion = DistillationLoss(temperature=2.0, alpha=0.5)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    os.makedirs(save_path, exist_ok=True)

    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        student_model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)  # Get teacher's predictions

            loss = criterion(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(student_outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / total
        train_acc = correct / total
        val_acc = evaluate(student_model, val_loader, device)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), os.path.join(save_path, f"best_student_model.pth"))

    # Save training losses and accuracies
    with open(os.path.join(save_path, "train_loss.txt"), 'w') as f:
        for val in train_losses:
            f.write(f"{val:.6f}\n")
    with open(os.path.join(save_path, "train_accuracy.txt"), 'w') as f:
        for val in train_accuracies:
            f.write(f"{val:.6f}\n")
    with open(os.path.join(save_path, "val_accuracy.txt"), 'w') as f:
        for val in val_accuracies:
            f.write(f"{val:.6f}\n")


# Function to evaluate the model
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


# Main code to distill ResNet-18 teacher into smaller models
def distill_models():
    # Load the teacher model (ResNet-18)
    teacher_model = models.resnet18(pretrained=True).cuda()
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 7)  # Adjust for 7 classes
    pretrained_model_path = '/home/qinh3/MCNC_pretrain/cvhw/models/custom_augmented/pretrained_res18/model.pth'
    state_dict = torch.load(pretrained_model_path)
    teacher_model.load_state_dict(state_dict)
    '''
    student_model_res20 = resnet20(num_classes=7)  # Define a custom ResNet14 model or adjust the layers
    print(f"Number of parameters in ResNet20: {sum(p.numel() for p in student_model_res20.parameters()) / 1e6} million")
    '''
    # Define smaller variants (compression rates) of ResNet
    student_model_res14 = resnet14(num_classes=7)  # Define a custom ResNet14 model or adjust the layers
    print(f"Number of parameters in ResNet14: {sum(p.numel() for p in student_model_res14.parameters()) / 1e6} million")
    student_model_res10 = resnet10(num_classes=7)  # Define a custom ResNet10 model with fewer layers/filters
    print(f"Number of parameters in ResNet10: {sum(p.numel() for p in student_model_res10.parameters()) / 1e6} million")

    # Prepare data (example paths)
    base_path = '/home/qinh3/MCNC_pretrain/cvhw/rafdb_augmented'
    train_csv = '/home/qinh3/MCNC_pretrain/cvhw/rafdb_augmented/train_labels.csv'
    test_csv = '/home/qinh3/MCNC_pretrain/cvhw/rafdb_augmented/test_labels.csv'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = Four4All(csv_file=train_csv, img_dir=base_path, transform=transform)
    test_dataset = Four4All(csv_file=test_csv, img_dir=base_path, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    '''
    print("Training ResNet-20 distillation model...")
    train_distilled_model(student_model_res20, teacher_model, train_loader, test_loader, num_epochs=40, lr=0.001, save_path='/home/qinh3/MCNC_pretrain/cvhw/distilled/res20')
    '''
    # Distill ResNet-14 model
    print("Training ResNet-14 distillation model...")
    train_distilled_model(student_model_res14, teacher_model, train_loader, test_loader, num_epochs=40, lr=0.0001, save_path='/home/qinh3/MCNC_pretrain/cvhw/distilled/res14')

    # Distill ResNet-10 model
    print("Training ResNet-10 distillation model...")
    train_distilled_model(student_model_res10, teacher_model, train_loader, test_loader, num_epochs=40, lr=0.001, save_path='/home/qinh3/MCNC_pretrain/cvhw/distilled/res10')


# Run distillation
if __name__ == '__main__':
    distill_models()
