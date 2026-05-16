#!/bin/bash
# =============================================================================
# API Test Script - fal.ai 모델 5종 각 1건씩 테스트
# =============================================================================
# Veo 3.1, Sora 2, Kling 3.0, Wan 2.2, Wan 2.6 각각 1개 프롬프트로 테스트
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Load .env from project root
if [ -f "$ROOT_DIR/.env" ]; then
    export $(grep -v '^#' "$ROOT_DIR/.env" | xargs)
fi

# Map FAL_AI_API_KEY -> FAL_KEY for fal_client
if [ -n "$FAL_AI_API_KEY" ]; then
    export FAL_KEY="$FAL_AI_API_KEY"
fi

if [ -z "$FAL_KEY" ]; then
    echo "ERROR: FAL_AI_API_KEY is not set."
    echo "Set it in $ROOT_DIR/.env"
    exit 1
fi

# =============================================================================
# Test Configuration
# =============================================================================
TEST_PROMPT="A teacher draws a right triangle on a whiteboard, labels the sides 3, 4, and 5, and explains the Pythagorean theorem step by step with clear handwriting and annotations."
OUTPUT_DIR="./outputs/api_test"
DURATION=8          # 8초 통일 (Wan 2.2는 최대 ~6.7초, Wan 2.6은 10초로 스냅)

echo "============================================================"
echo "  fal.ai API Test - 5 Models x 1 Prompt"
echo "============================================================"
echo "Prompt: ${TEST_PROMPT:0:80}..."
echo "Duration: ${DURATION}s"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

PASS=0
FAIL=0
TOTAL=5

# --- Veo 3.1 ---
echo "[1/5] Testing Veo 3.1 (fal-ai/veo3.1)..."
if python3 main.py generate \
    --prompt "$TEST_PROMPT" \
    --model veo31 \
    --duration $DURATION \
    --resolution 720p \
    --output "$OUTPUT_DIR" \
    --label "test_veo31"; then
    echo "[1/5] Veo 3.1: PASS"
    PASS=$((PASS + 1))
else
    echo "[1/5] Veo 3.1: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Sora 2 ---
echo "[2/5] Testing Sora 2 (fal-ai/sora-2/text-to-video/pro)..."
if python3 main.py generate \
    --prompt "$TEST_PROMPT" \
    --model sora2 \
    --duration $DURATION \
    --resolution 720p \
    --output "$OUTPUT_DIR" \
    --label "test_sora2"; then
    echo "[2/5] Sora 2: PASS"
    PASS=$((PASS + 1))
else
    echo "[2/5] Sora 2: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Kling 3.0 ---
echo "[3/5] Testing Kling 3.0 (fal-ai/kling-video/v3/standard)..."
if python3 main.py generate \
    --prompt "$TEST_PROMPT" \
    --model kling3 \
    --duration $DURATION \
    --resolution 720p \
    --output "$OUTPUT_DIR" \
    --label "test_kling3"; then
    echo "[3/5] Kling 3.0: PASS"
    PASS=$((PASS + 1))
else
    echo "[3/5] Kling 3.0: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Wan 2.2 ---
echo "[4/5] Testing Wan 2.2 (fal-ai/wan/v2.2-5b)..."
if python3 main.py generate \
    --prompt "$TEST_PROMPT" \
    --model wan22 \
    --duration $DURATION \
    --resolution 720p \
    --output "$OUTPUT_DIR" \
    --label "test_wan22"; then
    echo "[4/5] Wan 2.2: PASS"
    PASS=$((PASS + 1))
else
    echo "[4/5] Wan 2.2: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Wan 2.6 ---
echo "[5/5] Testing Wan 2.6 (wan/v2.6)..."
if python3 main.py generate \
    --prompt "$TEST_PROMPT" \
    --model wan26 \
    --duration $DURATION \
    --resolution 720p \
    --output "$OUTPUT_DIR" \
    --label "test_wan26"; then
    echo "[5/5] Wan 2.6: PASS"
    PASS=$((PASS + 1))
else
    echo "[5/5] Wan 2.6: FAIL"
    FAIL=$((FAIL + 1))
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "============================================================"
echo "  Test Results: $PASS/$TOTAL passed, $FAIL failed"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "All API tests passed!"
    exit 0
else
    echo "Some tests failed. Check logs above."
    exit 1
fi
