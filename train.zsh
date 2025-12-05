#!/usr/bin/env bash

# Text-to-LLVM IR Model Training Script
# This script automates the complete workflow for training the model on your device
# Compatible with both bash and zsh (tested with bash 4.0+, zsh 5.0+)
#
# Assumes this script is in a freshly cloned texttoir repository
# Run from the repository root directory: ./train.zsh [options]
#
# For zsh users: You can also run directly with: zsh train.zsh [options]
# For bash users: Run with: bash train.zsh [options] or ./train.zsh [options]

set -e  # Exit on error

# Ensure we're in the repository root directory
# Compatible with both bash and zsh
if [[ -n "$BASH_SOURCE" ]]; then
    # bash
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "$ZSH_VERSION" ]]; then
    # zsh
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
    # fallback
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$SCRIPT_DIR"

# Verify we're in the texttoir repository
if [[ ! -f "requirements.txt" ]] || [[ ! -d "training" ]] || [[ ! -d "model" ]]; then
    echo "ERROR: This script must be run from the texttoir repository root directory"
    echo "Expected files/directories: requirements.txt, training/, model/"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DATASET_SIZE="large"  # Default to large for maximum quality
NUM_EPOCHS=10
BATCH_SIZE=8  # Will be auto-tuned for GPU
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
OUTPUT_DIR="checkpoints"
DATA_DIR="dataset"
USE_AMP=true  # Essential for GPU speed
USE_GRADIENT_CHECKPOINTING=true
NUM_WORKERS=4  # Will be auto-tuned for GPU
EARLY_STOPPING_PATIENCE=3
QUANTIZE=false
SKIP_DATA_GENERATION=false

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Text-to-LLVM IR Model Training Script
Automates data generation and model training on your device.

OPTIONS:
    -h, --help                  Show this help message
    
    Dataset Options:
    -d, --dataset SIZE          Dataset size: quick, small, medium, large, xlarge (default: large)
                                  quick:  1,000 examples (~2MB) - for testing only
                                  small:  50,000 examples (~100MB) - minimal quality
                                  medium: 250,000 examples (~500MB) - good quality
                                  large:  500,000 examples (~1GB) - high quality (RECOMMENDED)
                                  xlarge: 1,000,000 examples (~2GB) - maximum quality
    --skip-data                 Skip data generation (use existing dataset)
    --data-dir DIR              Data directory (default: dataset)
    
    Training Options:
    -e, --epochs NUM            Number of training epochs (default: 10)
    -b, --batch-size NUM        Batch size per device (default: 8, auto-tuned for GPU)
    -g, --grad-accum NUM        Gradient accumulation steps (default: 4)
    -l, --learning-rate RATE    Learning rate (default: 5e-5)
    -o, --output-dir DIR        Output directory for checkpoints (default: checkpoints)
    -w, --workers NUM           Number of data loading workers (default: 4, auto-tuned for GPU)
    -p, --patience NUM          Early stopping patience (default: 3)
    
    Optimization Options:
    --no-amp                    Disable mixed precision training (NOT recommended for GPU)
    --no-checkpointing          Disable gradient checkpointing
    --quantize                  Quantize model after training for faster inference
    
EXAMPLES:
    # Quick test run (1K examples, fast training)
    $0 --dataset quick --epochs 3
    
    # Large dataset with GPU auto-tuning (RECOMMENDED)
    $0
    
    # Maximum quality with extra-large dataset
    $0 --dataset xlarge --epochs 15
    
    # Custom configuration with quantization
    $0 --dataset large --epochs 20 --batch-size 16 --quantize
    
    # Use existing dataset and train with custom settings
    $0 --skip-data --epochs 5 --learning-rate 3e-5

GPU ACCELERATION FEATURES:
    - Auto-tunes batch size and workers based on GPU memory
    - Enables TF32 for Ampere+ GPUs (automatic)
    - Mixed precision (FP16) training for 2-3x speedup
    - Larger datasets recommended for GPU (large or xlarge)
    - Detects and optimizes for multi-GPU setups
    
PERFORMANCE TIPS:
    - Use GPU for 5-10x speedup (CUDA-enabled GPU recommended)
    - Larger datasets (large/xlarge) produce better quality models
    - Use --quantize for 2-4x faster inference after training
    - Mixed precision (AMP) is enabled by default for GPU speed
    - Use --no-amp to disable mixed precision (not recommended for GPU)
    - Script auto-tunes settings based on your GPU
    
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--dataset)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --dataset requires a value"
                exit 1
            fi
            DATASET_SIZE="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA_GENERATION=true
            shift
            ;;
        --data-dir)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --data-dir requires a value"
                exit 1
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        -e|--epochs)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --epochs requires a value"
                exit 1
            fi
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --batch-size requires a value"
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--grad-accum)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --grad-accum requires a value"
                exit 1
            fi
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --learning-rate requires a value"
                exit 1
            fi
            LEARNING_RATE="$2"
            shift 2
            ;;
        -o|--output-dir)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --output-dir requires a value"
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --workers requires a value"
                exit 1
            fi
            NUM_WORKERS="$2"
            shift 2
            ;;
        -p|--patience)
            if [[ -z "$2" || "$2" == -* ]]; then
                print_error "Option --patience requires a value"
                exit 1
            fi
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --no-amp)
            USE_AMP=false
            shift
            ;;
        --no-checkpointing)
            USE_GRADIENT_CHECKPOINTING=false
            shift
            ;;
        --quantize)
            QUANTIZE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate dataset size
case $DATASET_SIZE in
    quick|small|medium|large|xlarge)
        ;;
    *)
        print_error "Invalid dataset size: $DATASET_SIZE"
        print_error "Valid options: quick, small, medium, large, xlarge"
        exit 1
        ;;
esac

# Print banner
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Text-to-LLVM IR Model Training Script                  ║"
echo "║         Optimized for Best Quality AI in Least Time            ║"
echo "║         GPU-Accelerated with Auto-Tuning                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
print_info "Welcome! This script will guide you through training the model."
print_info "Perfect for a freshly cloned repository - all setup is automated."
echo ""

# Print initial configuration
print_info "Repository: texttoir (freshly cloned)"
print_info "Working directory: $SCRIPT_DIR"
echo ""
print_info "Initial Configuration:"
echo "  Dataset size:              $DATASET_SIZE"
echo "  Data directory:            $DATA_DIR"
echo "  Skip data generation:      $SKIP_DATA_GENERATION"
echo "  Number of epochs:          $NUM_EPOCHS"
echo "  Batch size:                $BATCH_SIZE (may be auto-tuned for GPU)"
echo "  Gradient accumulation:     $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size:      $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning rate:             $LEARNING_RATE"
echo "  Output directory:          $OUTPUT_DIR"
echo "  Data workers:              $NUM_WORKERS (may be auto-tuned for GPU)"
echo "  Early stopping patience:   $EARLY_STOPPING_PATIENCE"
echo "  Mixed precision (AMP):     $USE_AMP"
echo "  Gradient checkpointing:    $USE_GRADIENT_CHECKPOINTING"
echo "  Quantize after training:   $QUANTIZE"
echo ""

# Check if Python is available
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_error "Please install Python 3.8 or higher and try again"
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python3"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_CMD="python3"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
print_success "Found Python $PYTHON_VERSION"

# Verify Python version is 3.8 or higher
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    print_error "Python version $PYTHON_VERSION is too old"
    print_error "This project requires Python 3.8 or higher"
    echo ""
    echo "Please upgrade Python and try again"
    exit 1
fi

# Check if requirements are installed
print_info "Checking dependencies from requirements.txt..."
if ! $PYTHON_CMD -c "import torch; import transformers" 2>/dev/null; then
    print_warning "Required packages not found in fresh clone"
    print_info "Installing dependencies (this may take several minutes)..."
    echo ""
    
    # Install dependencies with progress
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        print_success "Dependencies installed successfully from requirements.txt"
    else
        print_error "Failed to install dependencies"
        echo ""
        echo "Try installing manually with:"
        echo "  python3 -m pip install -r requirements.txt"
        exit 1
    fi
else
    print_success "Dependencies already installed"
fi

# Detect device (GPU or CPU)
print_info "Detecting compute device..."
DEVICE=$($PYTHON_CMD -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
if [[ "$DEVICE" == "CUDA" ]]; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEMORY=$($PYTHON_CMD -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')" 2>/dev/null)
    GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    print_success "GPU detected: $GPU_NAME (${GPU_MEMORY}GB VRAM)"
    print_info "Available GPUs: $GPU_COUNT"
    print_info "Training will use GPU acceleration (5-10x faster)"
    
    # GPU-specific optimizations for maximum speed
    print_info "Applying GPU-specific optimizations for maximum speed..."
    
    # Auto-tune batch size based on GPU memory
    GPU_MEMORY_INT=${GPU_MEMORY%%.*}  # Remove everything after first dot for integer comparison
    if [[ $GPU_MEMORY_INT -ge 24 ]]; then
        # High-end GPU (24GB+): Use larger batch sizes for maximum speed
        if [[ $BATCH_SIZE -eq 8 ]]; then
            BATCH_SIZE=16
            print_info "Auto-tuned batch size to 16 for high-end GPU"
        fi
        if [[ $NUM_WORKERS -eq 4 ]]; then
            NUM_WORKERS=8
            print_info "Auto-tuned data workers to 8 for maximum throughput"
        fi
    elif [[ $GPU_MEMORY_INT -ge 16 ]]; then
        # Mid-range GPU (16-24GB): Use moderate batch sizes
        if [[ $BATCH_SIZE -eq 8 ]]; then
            BATCH_SIZE=12
            print_info "Auto-tuned batch size to 12 for mid-range GPU"
        fi
        if [[ $NUM_WORKERS -eq 4 ]]; then
            NUM_WORKERS=6
            print_info "Auto-tuned data workers to 6 for better throughput"
        fi
    else
        # Lower-end GPU (<16GB): Keep reasonable settings
        print_info "Using default settings for GPU with ${GPU_MEMORY}GB VRAM"
    fi
    
    # Enable TF32 for Ampere+ GPUs (automatic in training script)
    print_success "GPU optimizations applied - maximum speed enabled!"
    
    # Display optimized configuration
    echo ""
    print_success "Final Optimized Configuration (GPU-Tuned):"
    echo "  Batch size:                $BATCH_SIZE"
    echo "  Effective batch size:      $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
    echo "  Data workers:              $NUM_WORKERS"
    echo "  Mixed precision:           $USE_AMP (FP16 on GPU for 2-3x speedup)"
    echo "  Expected speedup:          5-10x faster than CPU"
    
else
    print_warning "No GPU detected. Training will use CPU (slower)"
    print_info "For faster training, consider using a CUDA-enabled GPU"
    print_warning "CPU training may take significantly longer (5-10x slower)"
    
    # Reduce workers for CPU to avoid overhead
    if [[ $NUM_WORKERS -gt 2 ]]; then
        NUM_WORKERS=2
        print_info "Reduced data workers to 2 for CPU training"
    fi
    
    # Display optimized configuration for CPU
    echo ""
    print_info "Final Configuration (CPU-Optimized):"
    echo "  Batch size:                $BATCH_SIZE"
    echo "  Effective batch size:      $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
    echo "  Data workers:              $NUM_WORKERS"
fi
echo ""

# Step 1: Generate training data
if [[ "$SKIP_DATA_GENERATION" == true ]]; then
    print_info "Skipping data generation (using existing dataset)"
    if [[ ! -d "$DATA_DIR" ]]; then
        print_error "Data directory '$DATA_DIR' does not exist"
        exit 1
    fi
    if [[ ! -f "$DATA_DIR/train.jsonl" ]] || [[ ! -f "$DATA_DIR/val.jsonl" ]]; then
        print_error "Required files (train.jsonl, val.jsonl) not found in '$DATA_DIR'"
        exit 1
    fi
else
    echo "════════════════════════════════════════════════════════════════"
    echo "Step 1: Generating Training Data"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Determine target examples based on dataset size
    case $DATASET_SIZE in
        quick)
            TARGET_EXAMPLES=1000
            print_info "Generating quick test dataset (1,000 examples, ~2MB)"
            ;;
        small)
            TARGET_EXAMPLES=50000
            print_info "Generating small dataset (50,000 examples, ~100MB)"
            ;;
        medium)
            TARGET_EXAMPLES=250000
            print_info "Generating medium dataset (250,000 examples, ~500MB)"
            ;;
        large)
            TARGET_EXAMPLES=500000
            print_info "Generating large dataset (500,000 examples, ~1GB)"
            ;;
        xlarge)
            TARGET_EXAMPLES=1000000
            print_info "Generating extra-large dataset (1,000,000 examples, ~2GB)"
            ;;
    esac
    
    print_info "This may take several minutes..."
    echo ""
    
    if [[ "$DATASET_SIZE" == "quick" ]]; then
        $PYTHON_CMD data/generate_data_large.py --quick-test --output-dir "$DATA_DIR"
    else
        $PYTHON_CMD data/generate_data_large.py --target-examples $TARGET_EXAMPLES --output-dir "$DATA_DIR"
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Data generation completed successfully"
        print_info "Dataset saved to: $DATA_DIR"
    else
        print_error "Data generation failed"
        exit 1
    fi
    echo ""
fi

# Step 2: Train the model
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Training the Model"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Provide performance expectations
print_info "Performance Expectations:"
if [[ "$DEVICE" == "CUDA" ]]; then
    case $DATASET_SIZE in
        quick)
            print_info "  Estimated time: ~5-10 minutes on GPU"
            ;;
        small)
            print_info "  Estimated time: ~30-60 minutes on GPU"
            ;;
        medium)
            print_info "  Estimated time: ~2-3 hours on GPU"
            ;;
        large)
            print_info "  Estimated time: ~3-5 hours on GPU"
            ;;
        xlarge)
            print_info "  Estimated time: ~6-10 hours on GPU"
            ;;
    esac
    print_success "GPU acceleration will provide 5-10x speedup vs CPU"
else
    case $DATASET_SIZE in
        quick)
            print_warning "  Estimated time: ~30-60 minutes on CPU"
            ;;
        small)
            print_warning "  Estimated time: ~3-6 hours on CPU"
            ;;
        medium)
            print_warning "  Estimated time: ~12-16 hours on CPU"
            ;;
        large)
            print_warning "  Estimated time: ~20-30 hours on CPU"
            ;;
        xlarge)
            print_warning "  Estimated time: ~40-60 hours on CPU"
            ;;
    esac
fi
echo ""

print_info "Starting model training with optimized settings..."
print_info "Training progress will be displayed below"
echo ""

# Build training command as array for safe execution
TRAIN_ARGS=(
    "$PYTHON_CMD" "training/train.py"
    "--data_dir" "$DATA_DIR"
    "--num_epochs" "$NUM_EPOCHS"
    "--batch_size" "$BATCH_SIZE"
    "--gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS"
    "--learning_rate" "$LEARNING_RATE"
    "--output_dir" "$OUTPUT_DIR"
    "--num_workers" "$NUM_WORKERS"
    "--early_stopping_patience" "$EARLY_STOPPING_PATIENCE"
)

if [[ "$USE_AMP" == true ]]; then
    TRAIN_ARGS+=("--use_amp")
fi

if [[ "$USE_GRADIENT_CHECKPOINTING" == true ]]; then
    TRAIN_ARGS+=("--use_gradient_checkpointing")
fi

# Execute training safely with array expansion
"${TRAIN_ARGS[@]}"

if [[ $? -eq 0 ]]; then
    print_success "Model training completed successfully"
    print_info "Model saved to: $OUTPUT_DIR"
else
    print_error "Model training failed"
    exit 1
fi
echo ""

# Step 3: Quantize model (optional)
if [[ "$QUANTIZE" == true ]]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "Step 3: Quantizing Model for Faster Inference"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    QUANTIZED_DIR="${OUTPUT_DIR}_quantized"
    print_info "Quantizing model to INT8 for 2-4x faster inference..."
    print_info "Output directory: $QUANTIZED_DIR"
    echo ""
    
    # Build quantization command as array for safe execution
    QUANTIZE_ARGS=(
        "$PYTHON_CMD" "quantize_model.py"
        "--model_path" "$OUTPUT_DIR"
        "--output_dir" "$QUANTIZED_DIR"
        "--quantization_type" "dynamic"
    )
    
    # Execute quantization safely with array expansion
    "${QUANTIZE_ARGS[@]}"
    
    if [[ $? -eq 0 ]]; then
        print_success "Model quantization completed successfully"
        print_info "Quantized model saved to: $QUANTIZED_DIR"
    else
        print_error "Model quantization failed"
        exit 1
    fi
    echo ""
fi

# Print summary
echo "════════════════════════════════════════════════════════════════"
echo "Training Complete! 🎉"
echo "════════════════════════════════════════════════════════════════"
echo ""
print_success "All steps completed successfully!"
echo ""
echo "Summary:"
echo "  ✓ Dataset generated:     $DATA_DIR"
echo "  ✓ Model trained:         $OUTPUT_DIR"
if [[ "$QUANTIZE" == true ]]; then
    echo "  ✓ Model quantized:       ${OUTPUT_DIR}_quantized"
fi
echo ""
echo "Next Steps:"
echo ""
echo "1. Test the model with interactive inference:"
echo "   ${BLUE}$PYTHON_CMD inference.py --model_path $OUTPUT_DIR --interactive${NC}"
echo ""
echo "2. Generate LLVM IR from text:"
echo "   ${BLUE}$PYTHON_CMD inference.py --model_path $OUTPUT_DIR --text \"Write a function that adds two integers\"${NC}"
echo ""
if [[ "$QUANTIZE" == true ]]; then
    echo "3. Use quantized model for faster inference:"
    echo "   ${BLUE}$PYTHON_CMD inference.py --model_path ${OUTPUT_DIR}_quantized --interactive${NC}"
    echo ""
fi
echo "For more options, see: ${BLUE}$PYTHON_CMD inference.py --help${NC}"
echo ""
print_info "Happy coding! 🚀"
echo ""
