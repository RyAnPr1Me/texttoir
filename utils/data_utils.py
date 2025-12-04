"""
Utility functions for data loading and processing.
Optimized for speed with:
- Dynamic padding to reduce computation on padding tokens
- Multi-worker data loading
- Memory-efficient data structures
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict


class LLVMDataset(Dataset):
    """Dataset class for text-to-LLVM IR pairs with optimizations."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data efficiently
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single example with dynamic length (no padding yet)."""
        example = self.data[idx]
        text = example['text']
        llvm_ir = example['llvm_ir']
        
        # Add task prefix
        input_text = f"translate to llvm: {text}"
        
        # Tokenize without padding (will be padded in collate_fn)
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target without padding
        targets = self.tokenizer(
            llvm_ir,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


def dynamic_collate_fn(batch, pad_token_id=0):
    """
    Collate function with dynamic padding for efficiency.
    Only pads to the longest sequence in the batch, not max_length.
    """
    # Extract sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad to longest in batch (not max_length)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "labels": labels_padded
    }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 4,
    use_dynamic_padding: bool = True
):
    """
    Create optimized train and validation dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading (0 = main process only)
        use_dynamic_padding: Use dynamic padding for efficiency
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LLVMDataset(train_path, tokenizer, max_length)
    val_dataset = LLVMDataset(val_path, tokenizer, max_length)
    
    # Only use pinned memory if CUDA is available
    use_pinned_memory = torch.cuda.is_available()
    
    # Setup collate function for dynamic padding
    collate_fn = None
    if use_dynamic_padding:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        collate_fn = lambda batch: dynamic_collate_fn(batch, pad_token_id=pad_token_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pinned_memory,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch for faster loading
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pinned_memory,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
