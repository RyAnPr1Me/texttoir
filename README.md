# Text-to-LLVM IR

An AI-powered project that converts natural language text descriptions into LLVM Intermediate Representation (IR) code. This project uses a T5-based sequence-to-sequence model optimized for translation accuracy and training/inference speed.

## Features

- **Modular Architecture**: Clean separation of model, data, training, and utility components
- **High-Quality Data Generation**: Automated generation of diverse text-to-LLVM IR training pairs
- **Optimal Model**: Uses T5-small for the best balance of accuracy and speed
- **CI/CD Integration**: GitHub Actions workflow for automated data generation and training
- **Interactive Inference**: Easy-to-use script for generating LLVM IR from text

## Project Structure

```
texttoir/
├── model/                  # Model architecture
│   ├── __init__.py
│   └── text_to_llvm_model.py
├── data/                   # Data generation
│   └── generate_data.py
├── training/               # Training scripts
│   └── train.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_utils.py
│   └── training_utils.py
├── inference.py            # Inference script
├── requirements.txt
└── .github/workflows/
    └── train.yml          # CI/CD workflow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RyAnPr1Me/texttoir.git
cd texttoir
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Training Data

Generate diverse text-to-LLVM IR training pairs:

```bash
python data/generate_data.py
```

This creates a `dataset/` directory with train, validation, and test splits.

### Train the Model

Train the text-to-LLVM IR translation model:

```bash
python training/train.py \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --output_dir checkpoints
```

**Training arguments:**
- `--model_name`: Base model (default: `t5-small`)
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--output_dir`: Output directory for checkpoints (default: `checkpoints`)

### Generate LLVM IR (Inference)

**Interactive mode:**
```bash
python inference.py --model_path checkpoints --interactive
```

**Single text mode:**
```bash
python inference.py \
    --model_path checkpoints \
    --text "Write a function that adds two integers"
```

## Model Architecture

This project uses **T5-small** (60M parameters) as the optimal model architecture for several reasons:

1. **Translation Accuracy**: T5 is pre-trained on text-to-text tasks, making it ideal for code translation
2. **Training Speed**: Small enough to train quickly on CPU or single GPU
3. **Inference Speed**: Fast generation with beam search optimization
4. **Resource Efficiency**: Reasonable memory footprint (~250MB)

The model is fine-tuned on text-to-LLVM IR pairs with a custom prefix: `"translate to llvm: {text}"`.

## Training Data

The data generation script creates diverse examples covering:

- **Arithmetic Operations**: add, subtract, multiply, divide
- **Conditionals**: if-else, comparisons, branching
- **Loops**: for loops, while loops, phi nodes
- **Arrays**: array operations, pointer arithmetic
- **Functions**: function calls, composition
- **Floating Point**: float and double operations
- **Logical Operations**: AND, OR, XOR, comparisons
- **Data Augmentation**: Variations in text descriptions

The dataset is split into:
- Training: 80%
- Validation: 10%
- Test: 10%

## GitHub Actions Workflow

The included workflow automatically:

1. **Generates training data** when code is pushed
2. **Trains the model** with configurable parameters
3. **Uploads artifacts** (dataset and checkpoints)

Trigger manually with custom parameters:
- Go to Actions → Train Text-to-LLVM IR Model → Run workflow
- Set custom `num_epochs` and `batch_size`

## Examples

**Input:**
```
Write a function that adds two integers
```

**Output:**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
```

**Input:**
```
Create a function that returns the maximum of two integers
```

**Output:**
```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret i32 %a

if.else:
  ret i32 %b
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 4GB+ RAM (8GB recommended for training)
- Optional: CUDA-capable GPU for faster training

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.