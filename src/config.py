import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Paths
    data_dir: str = "./data/coco2017"
    train_dir: str = "train2017"
    val_dir: str = "val2017"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    # Model
    model_name: str = "ColorFormer"  # Our advanced model
    input_size: Tuple[int, int] = (256, 256)
    use_attention: bool = True
    use_residual: bool = True
    normalization: str = "instance"  # "batch", "instance", or "group"
    
    # Training (CPU Optimized)
    batch_size: int = 4  # Smaller for CPU
    num_epochs: int = 200  # More epochs for better accuracy
    num_workers: int = 0  # 0 for Windows, 2-4 for Linux/Mac
    
    # Optimization
    learning_rate: float = 0.0001  # Lower for stability
    weight_decay: float = 1e-5  # Regularization
    scheduler_patience: int = 10  # More patience
    scheduler_factor: float = 0.5
    
    # Loss functions (we'll use multiple)
    use_perceptual_loss: bool = True
    use_ssim_loss: bool = True
    use_color_loss: bool = True
    
    # Data augmentation
    use_heavy_augmentation: bool = True
    augment_prob: float = 0.7
    
    # Checkpointing
    save_interval: int = 10
    early_stop_patience: int = 20
    
    # Validation
    val_interval: int = 2  # Validate every 2 epochs
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "comparisons"), exist_ok=True)