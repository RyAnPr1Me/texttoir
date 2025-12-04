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

**Quick Test Dataset** (for rapid prototyping):

```bash
python data/generate_data.py
```

This creates a small `dataset/` directory (~100KB) with train, validation, and test splits for quick testing.

**Production Dataset** (1GB+ with extreme diversity):

```bash
python data/generate_data_large.py
```

This generates a comprehensive dataset (~1GB, 500K+ examples) with:
- **Extreme diversity**: All integer types (i8-i128), float types, and operation combinations
- **Edge cases**: Overflow detection, null checks, boundary conditions
- **Good code examples**: Properly validated, safe operations
- **Bad code examples**: Marked with "BAD CODE:" prefix, demonstrating common bugs like:
  - Division by zero vulnerabilities
  - Missing null pointer checks
  - Buffer overflow risks
  - Stack overflow in recursion
  - Undefined behavior (e.g., shift by >= bit width)
  - Integer overflow without checks
- **Quality markers**: Each example tagged as "GOOD" or "BAD" in the dataset

The large dataset covers:
- Arithmetic (all ops × all int/float types)
- Comparisons (signed, unsigned, ordered, unordered)
- Conditionals (if/else, clamp, abs, min/max)
- Loops (for, while, phi nodes)
- Arrays (sum, max, reverse, with/without bounds checking)
- Bitwise operations (AND, OR, XOR, shifts, bit manipulation)
- Functions (calls, recursion, GCD, Fibonacci)
- Structs and pointers
- Select operations
- Edge cases and overflow handling

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

### Quick Dataset (`generate_data.py`)
Small dataset (~50 examples) for rapid testing and prototyping.

### Large Production Dataset (`generate_data_large.py`)

The large-scale data generator creates **500K+ extremely diverse examples** (~1GB) covering:

**Core Operations:**
- **Arithmetic**: All operations (add, sub, mul, div, rem) × all types (i8, i16, i32, i64, i128, float, double, x86_fp80)
- **Comparisons**: Signed/unsigned integer comparisons, ordered/unordered float comparisons
- **Conditionals**: if-else, max/min, absolute value, clamp, sign function
- **Loops**: sum, factorial, power, with proper phi nodes
- **Arrays**: sum, max, reverse, with boundary checking
- **Bitwise**: AND, OR, XOR, shifts (logical/arithmetic), bit manipulation, popcount
- **Functions**: composition, recursion (GCD, Fibonacci)
- **Structs**: field access, pointer arithmetic
- **Select**: conditional select operations

**Quality Markers:**
- **GOOD examples**: Properly validated, safe code with edge case handling
- **BAD examples**: Explicitly marked bugs and vulnerabilities:
  - Division by zero (no checks)
  - Null pointer dereference
  - Buffer overflows (no bounds checking)
  - Integer overflow (esp. INT_MIN absolute value)
  - Stack overflow in recursion (missing base case)
  - Undefined behavior (shift by >= bit width)
  - Float NaN/infinity mishandling

**Uniqueness Features:**
- Hash-based deduplication ensures every example is unique
- Text variation generator creates diverse phrasings
- Comprehensive type coverage (all LLVM integer and float types)
- Real-world edge cases and corner conditions

Dataset split:
- **Training**: 80% (400K examples)
- **Validation**: 10% (50K examples)
- **Test**: 10% (50K examples)

Each example includes:
```json
{
  "text": "Description of what to generate",
  "llvm_ir": "Valid or intentionally buggy LLVM IR code",
  "quality": "GOOD" or "BAD"
}
```

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