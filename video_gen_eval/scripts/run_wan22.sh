#!/bin/bash
# =============================================================================
# Wan 2.2 (Alibaba) - Local GPU Video Generation
# =============================================================================
# Model: Wan-AI/Wan2.2-T2V-{1.3B,5B,14B}
# Requires: CUDA GPU (RTX 3090+ for 14B with quantization)
# Duration: configurable
# Cost: Free (local GPU)
#
# For RTX 3090 (24GB VRAM):
#   - 1.3B: native, 720p
#   - 14B: needs quantization (Q6_K), 480p recommended
#   - Consider Wan2GP for GPU-poor optimization
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

# =============================================================================
# Configuration
# =============================================================================
MODE="${1:-generate}"              # generate, edu, batch
PROMPT="${2:-Create an educational animation explaining the Pythagorean theorem}"
DURATION=8                         # Video duration in seconds
RESOLUTION="720p"                  # 480p, 720p (1080p needs >24GB VRAM)
ASPECT_RATIO="16:9"                # 16:9, 9:16, 4:3, 1:1
OUTPUT_DIR="./outputs"
LABEL=""

# Batch mode
INPUT_FILE="examples/batch_input.json"

# Edu mode
SUBJECT="math"
GRADE="middle school"

# =============================================================================
# GPU Configuration
# =============================================================================
export ENABLE_GPU_MODELS=true
# export WAN_MODEL_PATH="/path/to/local/model"    # Optional: local model path
# export HUGGINGFACE_TOKEN="your_token"            # Optional: for downloading

# =============================================================================
# Environment
# =============================================================================
cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

if [ -f "$ROOT_DIR/.env" ]; then
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
fi

if [ -n "$FAL_AI_API_KEY" ]; then
    export FAL_KEY="$FAL_AI_API_KEY"
fi

# Check GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null || {
    echo "ERROR: CUDA GPU not available."
    echo "Wan 2.2 requires a CUDA-compatible GPU."
    exit 1
}

# =============================================================================
# Run
# =============================================================================
echo "=============================================="
echo "Wan 2.2 Video Generation (Local GPU)"
echo "=============================================="
echo "Mode: $MODE"
echo "Resolution: $RESOLUTION"
echo "Duration: ${DURATION}s"

# Show GPU info
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
" 2>/dev/null

echo "=============================================="
echo ""

case $MODE in
    generate)
        python main.py generate \
            --prompt "$PROMPT" \
            --model wan22 \
            --duration "$DURATION" \
            --resolution "$RESOLUTION" \
            --aspect-ratio "$ASPECT_RATIO" \
            --output "$OUTPUT_DIR" \
            ${LABEL:+--label "$LABEL"}
        ;;
    edu)
        python main.py edu \
            --problem "$PROMPT" \
            --subject "$SUBJECT" \
            --grade "$GRADE" \
            --model wan22 \
            --duration "$DURATION" \
            --resolution "$RESOLUTION" \
            --output "$OUTPUT_DIR" \
            ${LABEL:+--label "$LABEL"}
        ;;
    batch)
        python main.py batch \
            --input "$INPUT_FILE" \
            --model wan22 \
            --duration "$DURATION" \
            --resolution "$RESOLUTION" \
            --output "$OUTPUT_DIR"
        ;;
    *)
        echo "Usage: $0 [generate|edu|batch] [prompt]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
