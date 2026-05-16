#!/bin/bash
# =============================================================================
# Veo 3.1 (Google) - Video Generation via fal.ai
# =============================================================================
# Endpoint: fal-ai/veo3.1
# Resolution: 720p, 1080p, 4k
# Duration: 4s, 6s, 8s
# Pricing: $0.20-0.60/s (depends on resolution & audio)
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
DURATION=8                         # 4, 6, or 8 seconds
RESOLUTION="1080p"                 # 720p, 1080p, 4k
ASPECT_RATIO="16:9"                # 16:9, 9:16
OUTPUT_DIR="./outputs"
LABEL=""

# Batch mode
INPUT_FILE="examples/batch_input.json"

# Edu mode
SUBJECT="math"
GRADE="middle school"

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

if [ -z "$FAL_KEY" ]; then
    echo "ERROR: FAL_AI_API_KEY is not set in $ROOT_DIR/.env"
    exit 1
fi

# =============================================================================
# Run
# =============================================================================
echo "=============================================="
echo "Veo 3.1 Video Generation"
echo "=============================================="
echo "Mode: $MODE"
echo "Resolution: $RESOLUTION"
echo "Duration: ${DURATION}s"
echo "=============================================="
echo ""

case $MODE in
    generate)
        python main.py generate \
            --prompt "$PROMPT" \
            --model veo31 \
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
            --model veo31 \
            --duration "$DURATION" \
            --resolution "$RESOLUTION" \
            --output "$OUTPUT_DIR" \
            ${LABEL:+--label "$LABEL"}
        ;;
    batch)
        python main.py batch \
            --input "$INPUT_FILE" \
            --model veo31 \
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
