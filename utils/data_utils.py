"""
Utility functions for data loading and processing.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class LLVMDataset(Dataset):
    """Dataset class for text-to-LLVM IR pairs."""
    
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
        
        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single example."""
        example = self.data[idx]
        text = example['text']
        llvm_ir = example['llvm_ir']
        
        # Add task prefix
        input_text = f"translate to llvm: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        targets = self.tokenizer(
            llvm_ir,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LLVMDataset(train_path, tokenizer, max_length)
    val_dataset = LLVMDataset(val_path, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
