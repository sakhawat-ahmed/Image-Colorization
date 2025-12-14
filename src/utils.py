# src/utils.py (CORRECTED VERSION)
import cv2
import numpy as np
import torch

def lab_to_rgb_simple(L, ab):
    """Simple, robust Lab to RGB conversion"""
    # Convert to numpy
    if isinstance(L, torch.Tensor):
        L = L.detach().cpu().numpy()
    if isinstance(ab, torch.Tensor):
        ab = ab.detach().cpu().numpy()
    
    # Debug shapes
    # print(f"DEBUG: L shape = {L.shape}, ab shape = {ab.shape}")
    
    # Handle batch dimension
    if L.ndim == 4:  # [B, 1, H, W]
        batch_size = L.shape[0]
        rgb_images = []
        
        for i in range(batch_size):
            # Get single image
            L_img = L[i, 0]  # [H, W]
            ab_img = ab[i]   # [2, H, W]
            
            # Denormalize
            L_img = L_img * 100.0
            ab_img = ab_img * 255.0 - 128
            
            # Transpose ab to [H, W, 2]
            ab_img = ab_img.transpose(1, 2, 0)
            
            # Combine channels
            lab = np.stack([L_img, ab_img[:, :, 0], ab_img[:, :, 1]], axis=-1)
            
            # Convert to uint8
            lab = lab.astype(np.uint8)
            
            # Convert to RGB
            rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
            rgb_images.append(rgb)
        
        return np.array(rgb_images)
    
    elif L.ndim == 3:  # [1, H, W] or [H, W]
        if L.shape[0] == 1:  # [1, H, W]
            L_img = L[0]  # [H, W]
            ab_img = ab[0]  # [2, H, W]
        else:  # [H, W]
            L_img = L
            ab_img = ab
        
        # Denormalize
        L_img = L_img * 100.0
        ab_img = ab_img * 255.0 - 128
        
        # Transpose ab to [H, W, 2]
        if ab_img.ndim == 3 and ab_img.shape[0] == 2:
            ab_img = ab_img.transpose(1, 2, 0)
        
        # Combine channels
        lab = np.stack([L_img, ab_img[:, :, 0], ab_img[:, :, 1]], axis=-1)
        
        # Convert to uint8
        lab = lab.astype(np.uint8)
        
        # Convert to RGB
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        
        return rgb
    
    else:
        raise ValueError(f"Unsupported L shape: {L.shape}")

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# Alias for backward compatibility
lab_to_rgb = lab_to_rgb_simple