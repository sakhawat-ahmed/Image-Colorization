# train_final.py
#!/usr/bin/env python3
"""
Final working training script
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_data_loaders
from src.model_advanced import ColorFormer

class WorkingTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        
        # Create model
        self.model = ColorFormer(
            in_channels=1,
            out_channels=2,
            base_channels=32
        ).to(self.device)
        
        # Simple L1 loss
        self.criterion = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Trackers
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            L = batch['L'].to(self.device)
            ab = batch['ab'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_ab = self.model(L)
            
            # Loss
            loss = self.criterion(pred_ab, ab)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate_simple(self, val_loader):
        """Simple validation - just loss, no PSNR"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                L = batch['L'].to(self.device)
                ab = batch['ab'].to(self.device)
                
                pred_ab = self.model(L)
                loss = self.criterion(pred_ab, ab)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
        }
        
        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"  Saved: {filename}")
    
    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if self.val_losses:
            improvement = [self.train_losses[i] - self.val_losses[i//2] 
                          for i in range(0, len(self.train_losses), 2)]
            plt.plot(improvement, 'g-', linewidth=2)
            plt.xlabel('Epoch (every 2nd)')
            plt.ylabel('Train-Val Difference')
            plt.title('Generalization Gap')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'training_history.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*40}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            
            # Validate every 2 epochs (simple loss only)
            if epoch % 2 == 0:
                val_loss = self.validate_simple(val_loader)
                self.val_losses.append(val_loss)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"\nValidation:")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  🎉 New best loss: {val_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping if loss stops improving
            if epoch > 10 and len(self.val_losses) > 2:
                if abs(self.val_losses[-1] - self.val_losses[-2]) < 0.001:
                    print(f"\n⚠️  Early stopping - loss converged")
                    break
        
        # Plot history
        self.plot_history()
        
        # Save final model
        self.save_checkpoint(num_epochs)
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Model saved to: {os.path.join(self.config.checkpoint_dir, 'best_model.pth')}")
        print(f"{'='*50}")


def main():
    """Main function"""
    print("="*60)
    print("WORKING IMAGE COLORIZATION TRAINING")
    print("="*60)
    
    # Configuration
    class Config:
        data_dir = "./data/coco2017"
        checkpoint_dir = "./checkpoints"
        results_dir = "./results"
        batch_size = 8
        num_epochs = 30
        learning_rate = 0.0002
        weight_decay = 1e-5
        image_size = 256
    
    config = Config()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LR: {config.learning_rate}")
    
    # Get data loaders
    print("\nLoading dataset...")
    try:
        from src.dataset import get_data_loaders
        
        train_loader, val_loader = get_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            train_samples=2000,
            val_samples=200,
            num_workers=0
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create and run trainer
    trainer = WorkingTrainer(config)
    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
    
    print("\n✅ Training complete!")
    print("\nTo test the model, run:")
    print("  python test_model.py --image your_image.jpg")


if __name__ == "__main__":
    main()