"""
Script to quantize a trained model for faster inference.
Reduces model size and increases inference speed by 2-4x with minimal quality loss.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(parent_dir))

from utils.quantization import quantize_model, save_quantized_model
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main(args):
    """Quantize a trained model."""
    print("=" * 50)
    print("Model Quantization for Faster Inference")
    print("=" * 50)
    
    # Load model and tokenizer
    print(f"\nLoading model from {args.model_path}...")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    
    print(f"Original model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Quantize model
    print(f"\nQuantizing model with {args.quantization_type} quantization...")
    quantized_model = quantize_model(model, quantization_type=args.quantization_type)
    
    # Save quantized model
    print(f"\nSaving quantized model to {args.output_dir}...")
    save_quantized_model(quantized_model, tokenizer, args.output_dir)
    
    print("\n" + "=" * 50)
    print("Quantization completed!")
    print(f"Quantized model saved to: {args.output_dir}")
    print("\nExpected speedup: 2-4x faster inference")
    print("Expected quality loss: <2% on most tasks")
    print("=" * 50)
    
    print("\nTo use the quantized model for inference:")
    print(f"  python inference.py --model_path {args.output_dir} --interactive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize trained model for faster inference")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_quantized",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--quantization_type",
        type=str,
        choices=["int8", "dynamic"],
        default="dynamic",
        help="Type of quantization (dynamic is recommended)"
    )
    
    args = parser.parse_args()
    main(args)
