# test_quick.py
import torch
import sys
import os

print("Testing imports...")

try:
    from src.model_advanced import ColorFormer
    print("✅ Model imports successful")
    
    # Test model creation
    model = ColorFormer(in_channels=1, out_channels=2, base_channels=32)
    print(f"✅ Model created, parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print(f"✅ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTesting dataset...")
try:
    from src.dataset import SimpleCOCODataset
    
    # Test with small dataset
    dataset = SimpleCOCODataset(
        data_dir="./data/coco2017",
        split='train',
        image_size=256,
        max_samples=10
    )
    
    sample = dataset[0]
    print(f"✅ Dataset loaded, sample shapes:")
    print(f"   L: {sample['L'].shape}")
    print(f"   ab: {sample['ab'].shape}")
    print(f"   rgb: {sample['rgb'].shape}")
    
except Exception as e:
    print(f"❌ Dataset error: {e}")

print("\n" + "="*50)
print("TEST COMPLETE")
print("="*50)
print("\nIf all tests pass, run:")
print("  python train.py")