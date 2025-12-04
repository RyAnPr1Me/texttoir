"""
Training script for text-to-LLVM IR model.
Optimized for best quality AI in least time with:
- Mixed precision training (FP16/BF16)
- Gradient accumulation for larger effective batch sizes
- Cosine annealing learning rate schedule
- Early stopping to prevent overfitting
- Dynamic padding for efficiency
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))

from model import create_model
from utils import create_dataloaders, save_checkpoint


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, 
                gradient_accumulation_steps=1, use_amp=True, scaler=None):
    """
    Train for one epoch with optimization features.
    
    Args:
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_amp: Use automatic mixed precision
        scaler: GradScaler for mixed precision training
    """
    model.get_model().train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        if use_amp:
            # Use FP16 on CUDA, BF16 on CPU or if preferred
            # BF16 is better for training stability but requires newer hardware
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            with autocast(enabled=True, dtype=dtype):
                outputs = model.forward(input_ids, attention_mask, labels)
                loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            outputs = model.forward(input_ids, attention_mask, labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        # Update weights after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=1.0)
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device, use_amp=True):
    """Validate the model with mixed precision support."""
    model.get_model().eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                # Use FP16 on CUDA for validation
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                with autocast(enabled=True, dtype=dtype):
                    outputs = model.forward(input_ids, attention_mask, labels)
                    loss = outputs.loss
            else:
                outputs = model.forward(input_ids, attention_mask, labels)
                loss = outputs.loss
                
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main(args):
    """Main training function with optimizations."""
    print("=" * 50)
    print("Text-to-LLVM IR Model Training (OPTIMIZED)")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster training on compatible GPUs")
    
    # Create model with optimizations
    print(f"\nInitializing model: {args.model_name}")
    model = create_model(
        model_name=args.model_name,
        max_length=args.max_length,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        compile_model=False  # Don't compile during training
    )
    print(f"Model parameters: {sum(p.numel() for p in model.get_model().parameters()):,}")
    
    # Create dataloaders with optimizations
    print(f"\nLoading data from {args.data_dir}")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=model.get_tokenizer(),
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers  # Use multiple workers for faster data loading
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Setup optimizer with better hyperparameters
    optimizer = AdamW(
        model.get_model().parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate training steps accounting for gradient accumulation
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    # Use cosine annealing schedule for better convergence
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup mixed precision training
    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LR scheduler: Cosine annealing with warmup")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Mixed precision (AMP): {use_amp}")
    print(f"  Gradient checkpointing: {args.use_gradient_checkpointing}")
    print(f"  Data workers: {args.num_workers}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'=' * 50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=use_amp,
            scaler=scaler
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device, use_amp=use_amp)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss! Saving model...")
            model.save(args.output_dir)
            # Also save training state for resumption
            save_checkpoint(
                model.get_model(),
                optimizer,
                epoch,
                val_loss,
                os.path.join(args.output_dir, "training_state")
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Save periodic checkpoint with full training state
        if epoch % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            model.save(checkpoint_dir)
            save_checkpoint(
                model.get_model(),
                optimizer,
                epoch,
                val_loss,
                checkpoint_dir
            )
    
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
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (effective batch size = batch_size * this)"
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
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training"
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without improvement)"
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
