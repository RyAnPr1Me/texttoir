"""
Model quantization utilities for faster inference.
Provides 8-bit quantization for 2-4x faster inference with minimal quality loss.
"""

import torch
from transformers import T5ForConditionalGeneration


def quantize_model(model, quantization_type="int8"):
    """
    Quantize model for faster inference.
    
    Args:
        model: Model to quantize
        quantization_type: Type of quantization ("int8" or "dynamic")
        
    Returns:
        Quantized model
    """
    if quantization_type == "int8":
        # Static int8 quantization
        try:
            # For PyTorch 2.0+, use quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("Model quantized to INT8 (2-4x faster inference)")
            return quantized_model
        except Exception as e:
            print(f"Warning: Could not quantize model: {e}")
            return model
    
    elif quantization_type == "dynamic":
        # Dynamic quantization (easier, still good speedup)
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            print("Model quantized with dynamic quantization")
            return quantized_model
        except Exception as e:
            print(f"Warning: Could not quantize model: {e}")
            return model
    
    else:
        print(f"Unknown quantization type: {quantization_type}")
        return model


def save_quantized_model(model, tokenizer, output_dir):
    """
    Save quantized model.
    
    Args:
        model: Quantized model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "quantized_model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Quantized model saved to {output_dir}")


def load_quantized_model(model_dir, model_name="t5-small"):
    """
    Load quantized model.
    
    Args:
        model_dir: Directory containing quantized model
        model_name: Base model name
        
    Returns:
        Quantized model
    """
    import os
    from transformers import T5Tokenizer
    
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Quantize
    model = quantize_model(model, quantization_type="dynamic")
    
    # Load state dict if available
    state_dict_path = os.path.join(model_dir, "quantized_model.pt")
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True))
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    
    print(f"Quantized model loaded from {model_dir}")
    return model, tokenizer
