# src/dataset.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random

class SimpleCOCODataset(Dataset):
    """Simple dataset for COCO images"""
    
    def __init__(self, data_dir, split='train', image_size=256, max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        
        # Get image paths
        split_dir = 'train2017' if split == 'train' else 'val2017'
        img_dir = os.path.join(data_dir, split_dir)
        
        self.image_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        
        # Limit samples if specified
        if max_samples:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"Loaded {len(self.image_paths)} images for {split}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image with OpenCV
            img_path = self.image_paths[idx]
            img = cv2.imread(img_path)
            
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Convert to Lab
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
            
            # Normalize
            L = lab[:, :, 0] / 100.0  # [0, 100] -> [0, 1]
            a = (lab[:, :, 1] + 128) / 255.0  # [-128, 127] -> [0, 1]
            b = (lab[:, :, 2] + 128) / 255.0  # [-128, 127] -> [0, 1]
            
            # Convert to tensors
            L_tensor = torch.FloatTensor(L).unsqueeze(0)  # [1, H, W]
            ab_tensor = torch.stack([torch.FloatTensor(a), torch.FloatTensor(b)], dim=0)  # [2, H, W]
            
            # RGB tensor for perceptual loss
            rgb_tensor = torch.FloatTensor(img.transpose(2, 0, 1) / 255.0)  # [3, H, W]
            
            return {
                'L': L_tensor,
                'ab': ab_tensor,
                'rgb': rgb_tensor,
                'path': img_path
            }
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a different image
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def get_data_loaders(data_dir, batch_size=8, image_size=256, 
                     train_samples=5000, val_samples=500, num_workers=0):
    """Create data loaders"""
    
    train_dataset = SimpleCOCODataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        max_samples=train_samples
    )
    
    val_dataset = SimpleCOCODataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        max_samples=val_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader