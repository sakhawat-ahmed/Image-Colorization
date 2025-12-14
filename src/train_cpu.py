import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .model_advanced import ColorFormer, PerceptualLoss
from .utils import lab_to_rgb, save_comparison, calculate_metrics

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                             for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
            
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class ColorConsistencyLoss(nn.Module):
    """Loss to enforce color consistency"""
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        
    def forward(self, pred, target):
        # Convert to HSV and compare hue/saturation
        pred_hsv = self.rgb_to_hsv(pred)
        target_hsv = self.rgb_to_hsv(target)
        
        # Hue difference (circular)
        hue_diff = torch.min(
            torch.abs(pred_hsv[:, 0, :, :] - target_hsv[:, 0, :, :]),
            1 - torch.abs(pred_hsv[:, 0, :, :] - target_hsv[:, 0, :, :])
        )
        
        # Saturation difference
        sat_diff = torch.abs(pred_hsv[:, 1, :, :] - target_hsv[:, 1, :, :])
        
        return hue_diff.mean() + sat_diff.mean()
    
    def rgb_to_hsv(self, rgb):
        # Simplified RGB to HSV conversion
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        max_val, _ = torch.max(rgb, dim=1)
        min_val, _ = torch.min(rgb, dim=1)
        diff = max_val - min_val
        
        # Hue
        hue = torch.zeros_like(max_val)
        hue[max_val == r] = (g[max_val == r] - b[max_val == r]) / (diff[max_val == r] + 1e-7)
        hue[max_val == g] = 2.0 + (b[max_val == g] - r[max_val == g]) / (diff[max_val == g] + 1e-7)
        hue[max_val == b] = 4.0 + (r[max_val == b] - g[max_val == b]) / (diff[max_val == b] + 1e-7)
        hue = (hue / 6.0) % 1.0
        
        # Saturation
        saturation = diff / (max_val + 1e-7)
        
        # Value
        value = max_val
        
        return torch.stack([hue, saturation, value], dim=1)


class Trainer:
    """Advanced trainer for CPU training"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = ColorFormer(
            in_channels=1,
            out_channels=2,
            base_channels=64,
            num_blocks=4,  # More blocks for better accuracy
            use_attention=config.use_attention
        ).to(device)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
        if config.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None
            
        if config.use_color_loss:
            self.color_loss = ColorConsistencyLoss()
        else:
            self.color_loss = None
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True
        )
        
        # Trackers
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.best_psnr = 0.0
        self.best_epoch = 0
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training on: {device}")
    
    def compute_loss(self, pred_ab, target_ab, pred_rgb, target_rgb):
        """Compute combined loss"""
        # Base L1 loss
        l1 = self.l1_loss(pred_ab, target_ab)
        
        # SSIM loss on RGB
        ssim = self.ssim_loss(pred_rgb, target_rgb)
        
        # Combined loss
        total_loss = l1 + 0.5 * ssim
        
        # Add perceptual loss if enabled
        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred_rgb, target_rgb)
            total_loss += 0.1 * perceptual
            
        # Add color consistency loss if enabled
        if self.color_loss is not None:
            color = self.color_loss(pred_rgb, target_rgb)
            total_loss += 0.05 * color
            
        return total_loss, {'l1': l1.item(), 'ssim': ssim.item()}
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0}
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            L = batch['L'].to(self.device)
            ab = batch['ab'].to(self.device)
            rgb = batch['rgb'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_ab = self.model(L)
            
            # Convert to RGB for additional losses
            pred_rgb = torch.zeros_like(rgb)
            for i in range(L.size(0)):
                pred_rgb_i = lab_to_rgb(L[i:i+1].cpu(), pred_ab[i:i+1].cpu())
                pred_rgb[i] = torch.FloatTensor(pred_rgb_i.transpose(2, 0, 1))
            pred_rgb = pred_rgb.to(self.device)
            
            # Compute loss
            loss, components = self.compute_loss(pred_ab, ab, pred_rgb, rgb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update trackers
            total_loss += loss.item()
            for k in components:
                loss_components[k] += components[k]
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'L1': f'{components["l1"]:.4f}',
                    'SSIM': f'{components["ssim"]:.4f}'
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v/len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                L = batch['L'].to(self.device)
                ab = batch['ab'].to(self.device)
                rgb = batch['rgb'].to(self.device)
                
                pred_ab = self.model(L)
                
                # Convert to RGB for metrics
                pred_rgb_list = []
                for i in range(L.size(0)):
                    pred_rgb_i = lab_to_rgb(L[i:i+1].cpu(), pred_ab[i:i+1].cpu())
                    pred_rgb_list.append(torch.FloatTensor(pred_rgb_i.transpose(2, 0, 1)))
                pred_rgb = torch.stack(pred_rgb_list).to(self.device)
                
                # Compute loss
                loss, _ = self.compute_loss(pred_ab, ab, pred_rgb, rgb)
                total_loss += loss.item()
                
                # Compute metrics
                metrics = calculate_metrics(pred_rgb.cpu(), rgb.cpu())
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                
                # Save sample images every 5 epochs
                if epoch % 5 == 0 and batch_idx == 0:
                    save_comparison(
                        L[:4].cpu(),
                        pred_ab[:4].cpu(),
                        ab[:4].cpu(),
                        epoch,
                        self.config.results_dir
                    )
        
        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_psnrs': self.val_psnrs,
            'best_psnr': self.best_psnr,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        # Keep only recent checkpoints
        if epoch % 20 != 0 and not is_best:
            old_checkpoint = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch-20}.pth")
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.config.num_epochs} epochs")
        
        early_stop_counter = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.config.val_interval == 0:
                val_loss, val_psnr, val_ssim = self.validate(val_loader, epoch)
                self.val_losses.append(val_loss)
                self.val_psnrs.append(val_psnr)
                
                print(f"\nValidation Results:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  PSNR: {val_psnr:.2f} dB")
                print(f"  SSIM: {val_ssim:.4f}")
                
                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Save best model
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  🎉 New best model! PSNR: {val_psnr:.2f} dB")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                # Save regular checkpoint
                if epoch % self.config.save_interval == 0:
                    self.save_checkpoint(epoch)
            
            # Early stopping
            if early_stop_counter >= self.config.early_stop_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
                break
        
        # Save final model
        self.save_checkpoint(self.config.num_epochs)
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
        print(f"{'='*50}")
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(np.arange(0, len(self.val_losses) * self.config.val_interval, 
                          self.config.val_interval), 
                self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # PSNR plot
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(0, len(self.val_psnrs) * self.config.val_interval, 
                          self.config.val_interval), 
                self.val_psnrs, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR')
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 3, 3)
        lr_values = []
        for i, lr in enumerate([self.optimizer.param_groups[0]['lr']] * len(self.train_losses)):
            if i % 10 == 0:  # Sample every 10 epochs
                lr_values.append(lr)
        plt.plot(lr_values, 'purple-', linewidth=2)
        plt.xlabel('Epoch (sampled)')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'training_history.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()