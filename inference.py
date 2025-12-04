"""
Inference script for text-to-LLVM IR model.
Optimized for best quality and speed.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(parent_dir))

from model import create_model


def main(args):
    """Run inference on input text with optimizations."""
    print("Loading model...")
    model = create_model(
        model_name=args.model_name,
        max_length=args.max_length,
        use_gradient_checkpointing=False,  # Not needed for inference
        compile_model=args.compile_model  # Use compilation for faster inference
    )
    
    # Load trained model if checkpoint exists
    if os.path.exists(args.model_path):
        model.load(args.model_path)
    else:
        print(f"Warning: Model path {args.model_path} not found. Using base model.")
    
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Enter text descriptions (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                text = input("\nEnter description: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                print("\nGenerating LLVM IR...")
                llvm_ir = model.generate(
                    text,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    use_cache=args.use_cache,
                    repetition_penalty=args.repetition_penalty
                )
                
                print("\nGenerated LLVM IR:")
                print("-" * 50)
                print(llvm_ir)
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    else:
        # Single text mode
        if not args.text:
            print("Error: --text argument required in non-interactive mode")
            return
        
        print(f"Input: {args.text}\n")
        print("Generating LLVM IR...")
        
        llvm_ir = model.generate(
            args.text,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=args.use_cache,
            repetition_penalty=args.repetition_penalty
        )
        
        print("\nGenerated LLVM IR:")
        print("-" * 50)
        print(llvm_ir)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLVM IR from text")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-small",
        help="Base model name"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text to translate"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search (higher = better quality)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Penalty for repeating tokens"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use caching for repeated inputs"
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=False,
        help="Compile model with torch.compile for faster inference (PyTorch 2.0+)"
    )
    
    args = parser.parse_args()
    main(args)
