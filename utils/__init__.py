"""Utility package for text-to-LLVM IR project."""

from .data_utils import LLVMDataset, create_dataloaders, load_jsonl, dynamic_collate_fn
from .training_utils import save_checkpoint, load_checkpoint, calculate_metrics

__all__ = [
    'LLVMDataset',
    'create_dataloaders',
    'load_jsonl',
    'dynamic_collate_fn',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_metrics'
]

# Quantization utilities (optional import)
try:
    from .quantization import quantize_model, save_quantized_model, load_quantized_model
    __all__.extend(['quantize_model', 'save_quantized_model', 'load_quantized_model'])
except ImportError:
    pass
