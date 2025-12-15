# inference.py
import torch
import argparse
from PIL import Image
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_advanced import ColorFormer
from src.utils import lab_to_rgb

def colorize_image(model, image_path, output_path=None, device='cpu'):
    """Colorize a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to model input size
    image_resized = image.resize((256, 256))
    image_np = np.array(image_resized)
    
    # Convert to Lab
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    
    # Extract L channel
    L = lab[:, :, 0] / 100.0  # Normalize
    L_tensor = torch.FloatTensor(L).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred_ab = model(L_tensor)
    
    # Convert to RGB
    pred_rgb = lab_to_rgb(L_tensor.cpu(), pred_ab.cpu())
    
    # Resize back to original size
    pred_pil = Image.fromarray(pred_rgb)
    pred_pil = pred_pil.resize(original_size, Image.Resampling.LANCZOS)
    
    # Save or return
    if output_path:
        pred_pil.save(output_path)
        print(f"Colorized image saved to: {output_path}")
    
    return pred_pil

def main():
    parser = argparse.ArgumentParser(description='Colorize images using trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device)
    config = checkpoint['config']
    
    model = ColorFormer(
        in_channels=1,
        out_channels=2,
        base_channels=64,
        num_blocks=4,
        use_attention=config.use_attention
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Colorize
    output_path = args.output or f"colorized_{os.path.basename(args.image)}"
    result = colorize_image(model, args.image, output_path, args.device)
    
    print(f"\nDone! Check {output_path} for the colorized image.")

if __name__ == "__main__":
    main()