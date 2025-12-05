#!/usr/bin/env bash
# =====================================================================
#  Accelerated Training → LoRA → Quantization → GGUF Export
#  - GPU detection + auto batch sizing
#  - HuggingFace Accelerate for distributed training
#  - LoRA fine-tuning
#  - Auto-resume from latest checkpoint
#  - Quantization (int8 or 4bit GPTQ)
#  - GGUF export (llama.cpp-compatible)
#  - Calls the two scripts your repo requires:
#       1) data/generate_data_large.py
#       2) training/train.py (Accelerate)
# =====================================================================

set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------- Defaults ------------------------------
DATASET_SIZE="large"
DATA_DIR="dataset"
OUTPUT_DIR="checkpoints"
EPOCHS=5
BATCH_SIZE=4
GRAD_ACC=4
LR="2e-4"
WORKERS=4
SKIP_DATA=0

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

QUANTIZE=1
QUANT_MODE="int8"   # int8 | 4bit

EXPORT_GGUF=1       # << NEW
CHECKPOINT_PATH=""  # << New: Will store auto-resume checkpoint path

# ----------------------------- Colors --------------------------------
BLUE="\033[1;34m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; NC="\033[0m"
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ----------------------------- Args ----------------------------------
usage() {
cat <<EOF
Usage: ./train.sh [options]

--dataset {quick|small|medium|large|xlarge}
--skip-data
--data-dir PATH
--epochs N
--batch-size N
--grad-accum N
--learning-rate LR
--output-dir PATH
--workers N

--lora-r N
--lora-alpha N
--lora-dropout FLOAT

--quantize
--quant-mode {int8|4bit}
--no-gguf          Disable GGUF export

EOF
exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET_SIZE="$2"; shift 2;;
    --skip-data) SKIP_DATA=1; shift;;
    --data-dir) DATA_DIR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --grad-accum) GRAD_ACC="$2"; shift 2;;
    --learning-rate) LR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;

    --lora-r) LORA_R="$2"; shift 2;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2;;
    --lora-dropout) LORA_DROPOUT="$2"; shift 2;;

    --quantize) QUANTIZE=1; shift;;
    --quant-mode) QUANT_MODE="$2"; shift 2;;
    --no-gguf) EXPORT_GGUF=0; shift;;

    -h|--help) usage;;
    *) err "Unknown option: $1";;
  esac
done

# ------------------------- Python & pip -------------------------------
info "Checking Python3…"

command -v python3 >/dev/null || err "Python3 missing"

PY=python3
PIP="$PY -m pip"

if ! $PIP --version >/dev/null 2>&1; then
    info "pip missing — installing…"
    $PY -m ensurepip
fi

info "Installing required packages…"
$PIP install -q --upgrade pip
$PIP install -q accelerate bitsandbytes peft transformers datasets torch sentencepiece

# llama.cpp-compatible exporter:
$PIP install -q llama-cpp-python transformers-gguf

ok "Environment ready."

# ------------------------- GPU Detection ------------------------------
info "Detecting GPU…"

HAS_GPU="$($PY -c 'import torch; print(torch.cuda.is_available())')"

if [[ "$HAS_GPU" == "True" ]]; then
    GPU_MEM="$($PY -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory//(1024**3))')"
    GPU_NAME="$($PY -c 'import torch; print(torch.cuda.get_device_name(0))')"
    ok "GPU: $GPU_NAME (${GPU_MEM}GB)"

    # auto-tune memory settings:
    if (( GPU_MEM >= 24 )); then
        BATCH_SIZE=8
    elif (( GPU_MEM >= 16 )); then
        BATCH_SIZE=6
    fi
else
    warn "No GPU detected — CPU will be slow"
    WORKERS=2
fi

# ------------------------- Auto-Resume Check --------------------------
info "Checking for existing checkpoints…"

LATEST_CKPT=$(ls -dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n 1 || true)

if [[ -n "$LATEST_CKPT" ]]; then
    info "Found previous checkpoint: $LATEST_CKPT"
    CHECKPOINT_PATH="$LATEST_CKPT"
else
    info "No previous checkpoint found – starting fresh."
fi

# ------------------------- Dataset Generation -------------------------
if (( SKIP_DATA == 0 )); then
    info "Generating dataset ($DATASET_SIZE)"

    case "$DATASET_SIZE" in
        quick) N=1000;;
        small) N=50000;;
        medium) N=250000;;
        large) N=500000;;
        xlarge) N=1000000;;
        *) err "Invalid dataset size";;
    esac

    $PY data/generate_data_large.py \
        --target-examples "$N" \
        --output-dir "$DATA_DIR" \
        || err "Dataset generation failed"
fi

# ------------------------- Accelerate Config --------------------------
accelerate config default --mixed_precision fp16 --dynamo_backend no

# ------------------------- TRAINING ----------------------------------
info "Launching Accelerate fine-tuning…"

CMD="accelerate launch training/train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --num_workers $WORKERS \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT"

# Auto resume logic
if [[ -n "$CHECKPOINT_PATH" ]]; then
    CMD+=" --resume_from_checkpoint $CHECKPOINT_PATH"
    info "Training will auto-resume from: $CHECKPOINT_PATH"
fi

eval "$CMD" || err "Training failed."

ok "Training complete → $OUTPUT_DIR"

# ------------------------- QUANTIZATION -------------------------------
if (( QUANTIZE == 1 )); then
    info "Applying $QUANT_MODE quantization…"

    case "$QUANT_MODE" in
        int8)
            $PY quantize_model.py \
                --model_path "$OUTPUT_DIR" \
                --output_dir "${OUTPUT_DIR}_int8" \
                --quantization_type dynamic

            ok "INT8 model saved → ${OUTPUT_DIR}_int8"
            FINAL_MODEL="${OUTPUT_DIR}_int8"
            ;;
        4bit)
            $PY quantize_model.py \
                --model_path "$OUTPUT_DIR" \
                --output_dir "${OUTPUT_DIR}_4bit" \
                --quantization_type gptq

            ok "4-bit GPTQ model saved → ${OUTPUT_DIR}_4bit"
            FINAL_MODEL="${OUTPUT_DIR}_4bit"
            ;;
        *)
            err "Invalid quantization mode: $QUANT_MODE"
            ;;
    esac
else
    FINAL_MODEL="$OUTPUT_DIR"
fi

# ------------------------- GGUF EXPORT -------------------------------
if (( EXPORT_GGUF == 1 )); then
    info "Exporting model to GGUF…"

    GGUF_OUT="${FINAL_MODEL}_gguf"
    mkdir -p "$GGUF_OUT"

    $PY - <<EOF
from transformers_gguf import convert_llama

convert_llama(
    input_dir="$FINAL_MODEL",
    output_dir="$GGUF_OUT",
    dtype="q8_0",  # Change to q4_0, q5_k_m, q8_0, etc.
)
EOF

    ok "GGUF export complete → $GGUF_OUT"
fi

# ------------------------- DONE ---------------------------------------
ok "Pipeline finished successfully! 🎉"
echo -e "\nRun inference:\n  ${BLUE}python3 inference.py --model_path $FINAL_MODEL --interactive${NC}"
echo -e "GGUF available at: ${BLUE}${FINAL_MODEL}_gguf/${NC}"
