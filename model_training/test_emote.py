import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from ResEmoteNet import ResEmoteNet
from get_dataset import Four4All
import os

'''
The original model from ResEmoteNet is trained on dataset with different label encoding.
This file converts the label encodings to that used by the authors to train ResEmoteNet
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

base_path = 'data/rafdb_augmented'
train_csv = 'data/rafdb_augmented/train_labels.csv'
test_csv = 'data/rafdb_augmented/test_labels.csv'

fer_dataset_train = Four4All(csv_file='data/rafdb_augmented/train_labels.csv',
                            img_dir='data/rafdb_augmented', transform=transform)

fer_dataset_test = Four4All(csv_file='data/rafdb_augmented/test_labels.csv',
                            img_dir='data/rafdb_augmented', split="test", transform=transform)

train_loader = DataLoader(fer_dataset_train, batch_size=16, shuffle=True)
test_loader = DataLoader(fer_dataset_test, batch_size=16, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
model = ResEmoteNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=80):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {train_accuracy:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'checkpoint_epoch_{epoch+1}.pth')

def evaluate_model(model, test_loader):
    model.eval()
    final_test_loss = 0.0
    final_test_correct = 0
    final_test_total = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            final_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            final_test_total += labels.size(0)
            final_test_correct += (predicted == labels).sum().item()

    final_test_loss /= len(test_loader)
    final_test_acc = final_test_correct / final_test_total
    print(f"Test Loss: {final_test_loss:.4f} | Test Accuracy: {final_test_acc:.4f}")

def evaluate(model, dataloader, device='cuda'):
    # ResEmoteNet's output labels map to your labels
    res_emote_to_test_labels = {
        0: 1, 
        1: 5, 
        2: 4, 
        3: 0, 
        4: 2, 
        5: 3, 
        6: 6 
    }

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds_mapped = torch.tensor([res_emote_to_test_labels[p.item()] for p in preds]).to(device)

            correct += (preds_mapped == labels).sum().item()
            total += labels.size(0)

    return correct / total
# Main execution
if __name__ == '__main__':
    '''
    train_model(model, train_loader, num_epochs=80)
    '''
    checkpoint = torch.load('/data/rafdb_model (1).pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Model loaded for final evaluation.")
    print(evaluate(model, train_loader))
    '''
    evaluate_model(model, train_loader)
    evaluate_model(model, test_loader)
    '''
