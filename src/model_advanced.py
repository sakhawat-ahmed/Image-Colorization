# src/model_advanced.py (COMPLETE)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==================== MODEL COMPONENTS ====================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return self.relu(out)


class ColorFormer(nn.Module):
    """Main colorization model"""
    
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super(ColorFormer, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x


# ==================== LOSS FUNCTIONS ====================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, device='cpu'):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG (features only)
        vgg = models.vgg16(pretrained=True).features[:16]  # First few layers
        
        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.to(device)
        self.device = device
        
        # Normalization for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
    def normalize(self, x):
        return (x - self.mean) / self.std
        
    def forward(self, pred, target):
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        # Calculate perceptual loss
        loss = F.l1_loss(pred_features, target_features)
                
        return loss


class SSIMLoss(nn.Module):
    """Simplified SSIM Loss"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        
    def forward(self, img1, img2):
        # Simple approximation of SSIM loss
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1*img1, self.window_size, stride=1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2*img2, self.window_size, stride=1, padding=self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1*img2, self.window_size, stride=1, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()