"""
Training script for text-to-LLVM IR model.
"""

import argparse
import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from utils import create_dataloaders, save_checkpoint


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.get_model().train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model.forward(input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    """Validate the model."""
    model.get_model().eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model.forward(input_ids, attention_mask, labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main(args):
    """Main training function."""
    print("=" * 50)
    print("Text-to-LLVM IR Model Training")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"\nInitializing model: {args.model_name}")
    model = create_model(model_name=args.model_name, max_length=args.max_length)
    print(f"Model parameters: {sum(p.numel() for p in model.get_model().parameters()):,}")
    
    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=model.get_tokenizer(),
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.get_model().parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss! Saving model...")
            model.save(args.output_dir)
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            model.save(checkpoint_dir)
    
    print(f"\n{'=' * 50}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text-to-LLVM IR model")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-small",
        help="Base model name (default: t5-small for optimal speed/accuracy)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Directory containing train.jsonl and val.jsonl"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    
    args = parser.parse_args()
    main(args)
