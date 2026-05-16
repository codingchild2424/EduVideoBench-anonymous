#!/bin/bash
# =============================================================================
# Kling 3.0 (Kuaishou) - Video Generation via fal.ai
# =============================================================================
# Endpoint: fal-ai/kling-video/v3/standard/text-to-video
# Resolution: up to 1080p
# Duration: 3-15 seconds (flexible)
# Pricing: $0.168/s (no audio), $0.252/s (with audio)
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
DURATION=8                         # 3-15 seconds
RESOLUTION="1080p"                 # 720p, 1080p
ASPECT_RATIO="16:9"                # 16:9, 9:16, 1:1
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
echo "Kling 3.0 Video Generation"
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
            --model kling3 \
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
            --model kling3 \
            --duration "$DURATION" \
            --resolution "$RESOLUTION" \
            --output "$OUTPUT_DIR" \
            ${LABEL:+--label "$LABEL"}
        ;;
    batch)
        python main.py batch \
            --input "$INPUT_FILE" \
            --model kling3 \
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
