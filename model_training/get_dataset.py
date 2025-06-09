import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

# Label encoder
id2emotion = {
    1: 'Suprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Sad',
    6: 'Angry',
    7: 'Neural'
}

# Acknowledgement: Code Available at: https://github.com/ArnabKumarRoy02/ResEmoteNet/blob/main/get_dataset.py
class Four4All(Dataset):
    def __init__(self, csv_file, img_dir, split="train", transform=None):
        self.labels = pd.read_csv(csv_file, header=None, names=["image_name", "emotion_id"])
        self.img_dir = img_dir
        self.split = split  # 'train' or 'test'
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.labels.iloc[idx, 0]  
        emotion_id = self.labels.iloc[idx, 1]  
        img_path = os.path.join(self.img_dir, self.split, img_name)
        
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, emotion_id - 1