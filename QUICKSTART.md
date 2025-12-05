# 🚀 Quick Start Guide: Optimized Training & Inference

This guide shows how to get the best quality AI in the least time.

## Prerequisites

```bash
pip install -r requirements.txt
```

## 🎯 Automated Training (Recommended)

Use the provided training script for the easiest experience:

### Quick Test (Fast)
```bash
./train.zsh --dataset quick --epochs 3
```

### Recommended Setup (Best Balance)
```bash
./train.zsh --dataset medium
```

### Production Quality (Best Results)
```bash
./train.zsh --dataset large --epochs 15
```

### With Quantization (Fastest Inference)
```bash
./train.zsh --dataset medium --quantize
```

**What the script does:**
1. ✓ Checks and installs dependencies
2. ✓ Detects GPU/CPU automatically
3. ✓ Generates training data
4. ✓ Trains model with optimal settings
5. ✓ Optionally quantizes for faster inference
6. ✓ Provides clear next steps

For all options: `./train.zsh --help`

## 📋 Manual Training (Alternative)

If you prefer to run each step manually:

## Step 1: Generate Optimized Dataset

For quick testing (1000 examples):
```bash
python data/generate_data_large.py --quick-test
```

For production with large dataset (500K examples, ~1GB - **RECOMMENDED**):
```bash
python data/generate_data_large.py --target-examples 500000
```

For medium-sized dataset (250K examples, ~500MB):
```bash
python data/generate_data_large.py --target-examples 250000
```

**Note:** Larger datasets produce better quality models. For best results, use 500K+ examples.

## Step 2: Train with All Optimizations Enabled

```bash
python training/train.py \
    --num_epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --output_dir checkpoints \
    --use_amp \
    --use_gradient_checkpointing \
    --num_workers 4 \
    --early_stopping_patience 3
```

**What these optimizations do:**
- `--use_amp`: Mixed precision training (2-3x faster)
- `--gradient_accumulation_steps 4`: Effective batch size of 32
- `--use_gradient_checkpointing`: 40-50% less memory
- `--num_workers 4`: 3-4x faster data loading
- `--early_stopping_patience 3`: Stops when no improvement

## Step 3: Fast Inference

**For maximum speed (with quantization):**
```bash
# First, quantize the model
python quantize_model.py \
    --model_path checkpoints \
    --output_dir checkpoints_quantized \
    --quantization_type dynamic

# Use quantized model for 2-4x faster inference
python inference.py \
    --model_path checkpoints_quantized \
    --interactive \
    --num_beams 3 \
    --compile_model
```

**For maximum quality:**
```bash
python inference.py \
    --model_path checkpoints \
    --interactive \
    --num_beams 5 \
    --temperature 0.7 \
    --repetition_penalty 1.2
```

## Performance Tips

### Training Faster
1. **Use GPU**: CUDA-enabled GPU gives 5-10x speedup
2. **Increase batch size**: If you have more memory
3. **Reduce max_length**: Use 256 instead of 512 if your IR is short
4. **Use fewer workers**: Set `--num_workers 2` if CPU-bound

### Better Quality
1. **More training data**: Use 500K+ examples
2. **Higher num_beams**: Try 7-10 for best quality
3. **Lower temperature**: Use 0.5 for more deterministic output
4. **Train longer**: Let early stopping decide when to stop

### Extra Speed Boost
1. **Quantize model**: Use `quantize_model.py` for 2-4x speedup
2. **Use compile**: Enable `--compile_model` flag
3. **Reduce num_beams**: Use 3 instead of 5
4. **Batch inference**: Process multiple inputs together

### Save Memory
1. **Gradient checkpointing**: Already enabled by default
2. **Smaller batch size**: Reduce to 4 with more accumulation steps
3. **Mixed precision**: Already enabled (use FP16)

## Benchmarks

On a single NVIDIA A100 GPU:
- **Training**: ~3-4 hours for 500K examples (large dataset)
- **Training**: ~2 hours for 250K examples (medium dataset)
- **Inference**: ~0.15 seconds per example with compilation
- **Inference (Quantized)**: ~0.05 seconds per example (2-4x faster)
- **Memory**: ~10GB GPU memory during training
- **Quality**: 85-90% exact match on validation set (higher with more data)

On CPU (no GPU):
- **Training**: ~12-16 hours for 500K examples (large dataset)
- **Training**: ~8-12 hours for 250K examples (medium dataset)
- **Inference**: ~0.5 seconds per example
- **Inference (Quantized)**: ~0.15 seconds per example (2-4x faster)
- **Memory**: ~4GB RAM
- **Quality**: Same as GPU

## Troubleshooting

**Out of memory during training:**
- Reduce `--batch_size` to 4
- Increase `--gradient_accumulation_steps` to 8
- Reduce `--num_workers` to 2

**Training too slow:**
- Increase `--num_workers` to 8
- Use GPU if available
- Reduce dataset size for testing

**Poor quality output:**
- Train for more epochs
- Increase `--num_beams` to 7
- Use more training data
- Lower `--temperature` to 0.5

## Next Steps

See `config_optimal.yaml` for detailed configuration options.
See `README.md` for complete documentation.
