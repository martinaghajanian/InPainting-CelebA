import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

class SmilingInpaintDataset(Dataset):
    def __init__(self, real_root, masked_root, transform=None):
        self.real_root = real_root
        self.masked_root = masked_root
        self.transform = transform
        self.real_images = sorted(glob.glob(os.path.join(real_root, '*.jpg')))
        
    def __len__(self):
        return len(self.real_images)
    
    def __getitem__(self, idx):
        real_path = self.real_images[idx]
        filename = os.path.basename(real_path)
        masked_path = os.path.join(self.masked_root, filename)
        
        real_img = Image.open(real_path).convert('RGB')
        masked_img = Image.open(masked_path).convert('RGB')
        
        if self.transform:
            real_img = self.transform(real_img)
            masked_img = self.transform(masked_img)
        
        mask = (torch.mean(masked_img, dim=0, keepdim=True) < 0.001).float()

        return masked_img, mask, real_img