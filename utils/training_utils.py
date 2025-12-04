"""
Training utilities and helper functions.
"""

import os
import torch
from tqdm import tqdm


def save_checkpoint(model, optimizer, epoch, loss, output_dir):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, loss


def calculate_metrics(predictions, references):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        Dictionary of metrics
    """
    # Exact match accuracy
    exact_matches = sum(pred.strip() == ref.strip() 
                       for pred, ref in zip(predictions, references))
    accuracy = exact_matches / len(predictions) if predictions else 0
    
    return {
        'accuracy': accuracy,
        'exact_matches': exact_matches,
        'total': len(predictions)
    }
