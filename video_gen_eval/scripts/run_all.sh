#!/bin/bash
# =============================================================================
# EduVBench - Full Benchmark Run (Generation + Evaluation)
# =============================================================================
# Usage:
#   ./scripts/run_all.sh                          # generate + evaluate all
#   ./scripts/run_all.sh --models veo31 sora2     # specific models only
#   ./scripts/run_all.sh --dimensions knowledge   # specific dimension only
#   ./scripts/run_all.sh --resume                 # resume interrupted run
#   ./scripts/run_all.sh --parallel               # run models in parallel
#   ./scripts/run_all.sh --dry-run                # show plan without running
#   ./scripts/run_all.sh --skip-gen               # evaluate only (skip generation)
#   ./scripts/run_all.sh --skip-eval              # generate only (skip evaluation)
#
# Output structure:
#   outputs/{model}/{timestamp}/
#     ├── videos/
#     ├── *.json          (paired metadata)
#     └── run_summary.json
#   results/run_{timestamp}/
#     ├── {model}_scorecard.json
#     ├── comparative_report.json
#     ├── comparative_report.md
#     └── leaderboard.json
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$PROJECT_DIR")"
DATASET_DIR="$ROOT_DIR/eduvbench-dataset"

# =============================================================================
# Defaults
# =============================================================================
ALL_MODELS=(veo31 sora2 kling3 wan22 wan26)
ALL_DIMENSIONS=(knowledge skills attitude)
DURATION=8
RESOLUTION="720p"
OUTPUT_DIR="./outputs"
RESULTS_DIR="./results"
DRY_RUN=false
RESUME=false
PARALLEL=false
SKIP_GEN=false
SKIP_EVAL=false

MODELS=()
DIMENSIONS=()

# =============================================================================
# Parse Arguments
# =============================================================================
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --models MODEL [MODEL ...]       Models to run (default: all 5)"
    echo "  --dimensions DIM [DIM ...]       Dimensions: knowledge, skills, attitude (default: all)"
    echo "  --resume                         Resume from latest run (skip completed prompts)"
    echo "  --parallel                       Run models in parallel (one process per model)"
    echo "  -d, --duration SEC               Duration in seconds (default: $DURATION)"
    echo "  -r, --resolution RES             Resolution: 480p, 720p, 1080p (default: $RESOLUTION)"
    echo "  -o, --output DIR                 Output directory (default: $OUTPUT_DIR)"
    echo "  --results-dir DIR                Evaluation results directory (default: $RESULTS_DIR)"
    echo "  --skip-gen                       Skip video generation (evaluate only)"
    echo "  --skip-eval                      Skip evaluation (generate only)"
    echo "  --eval-only                      Alias for --skip-gen"
    echo "  --dry-run                        Show plan without running"
    echo "  -h, --help                       Show this help"
    echo ""
    echo "Models: veo31, sora2, kling3, wan22, wan26"
    echo ""
    echo "Examples:"
    echo "  $0                                       # Full run (generate + evaluate)"
    echo "  $0 --parallel                            # Full run (models in parallel)"
    echo "  $0 --parallel --resume                   # Resume all models in parallel"
    echo "  $0 --models veo31 sora2 --parallel       # 2 models in parallel"
    echo "  $0 --dimensions knowledge --dry-run      # Preview knowledge only"
    echo "  $0 --skip-gen                            # Evaluate existing videos"
    echo "  $0 --skip-eval                           # Generate only, no evaluation"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --dimensions)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DIMENSIONS+=("$1")
                shift
            done
            ;;
        -d|--duration) DURATION="$2"; shift 2 ;;
        -r|--resolution) RESOLUTION="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --parallel) PARALLEL=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-gen) SKIP_GEN=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --eval-only) SKIP_GEN=true; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

# Apply defaults
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
fi

if [ ${#DIMENSIONS[@]} -eq 0 ]; then
    DIMENSIONS=("${ALL_DIMENSIONS[@]}")
fi

# =============================================================================
# Map dimensions to JSON files (bash 3.x compatible — no associative arrays)
# =============================================================================
dim_file() {
    case "$1" in
        knowledge) echo "knowledge_prompts.json" ;;
        skills)    echo "skills_prompts.json" ;;
        attitude)  echo "attitude_prompts.json" ;;
        *)         echo "" ;;
    esac
}

dim_count() {
    case "$1" in
        knowledge) echo 82 ;;
        skills)    echo 86 ;;
        attitude)  echo 47 ;;
        *)         echo 0 ;;
    esac
}

# Build file list
INPUT_FILES=()
TOTAL_PROMPTS=0
for dim in "${DIMENSIONS[@]}"; do
    file="$(dim_file "$dim")"
    if [ -z "$file" ]; then
        echo "ERROR: Unknown dimension '$dim'. Use: knowledge, skills, attitude"
        exit 1
    fi
    filepath="$DATASET_DIR/$file"
    if [ ! -f "$filepath" ]; then
        echo "ERROR: Dataset file not found: $filepath"
        exit 1
    fi
    INPUT_FILES+=("$filepath")
    TOTAL_PROMPTS=$((TOTAL_PROMPTS + $(dim_count "$dim")))
done

TOTAL_VIDEOS=$((TOTAL_PROMPTS * ${#MODELS[@]}))

# =============================================================================
# Environment
# =============================================================================
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

if [ -n "$FAL_AI_API_KEY" ]; then
    export FAL_KEY="$FAL_AI_API_KEY"
fi

# =============================================================================
# Plan
# =============================================================================
MODE="sequential"
if [ "$PARALLEL" = true ]; then
    MODE="parallel (${#MODELS[@]} processes)"
fi

PHASES=""
if [ "$SKIP_GEN" = true ] && [ "$SKIP_EVAL" = true ]; then
    echo "ERROR: Cannot skip both generation and evaluation."
    exit 1
elif [ "$SKIP_GEN" = true ]; then
    PHASES="evaluation only"
elif [ "$SKIP_EVAL" = true ]; then
    PHASES="generation only"
else
    PHASES="generation + evaluation"
fi

echo "============================================================"
echo "  EduVBench - Full Benchmark Run"
echo "============================================================"
echo ""
echo "  Pipeline:    $PHASES"
echo "  Models:      ${MODELS[*]}"
echo "  Dimensions:  ${DIMENSIONS[*]}"
echo "  Prompts:     $TOTAL_PROMPTS"
echo "  Videos:      $TOTAL_VIDEOS (${#MODELS[@]} models × $TOTAL_PROMPTS prompts)"
if [ "$SKIP_GEN" = false ]; then
    echo "  Duration:    ${DURATION}s"
    echo "  Resolution:  $RESOLUTION"
fi
echo "  Output:      $OUTPUT_DIR"
echo "  Results:     $RESULTS_DIR"
echo "  Mode:        $MODE"
if [ "$RESUME" = true ]; then
    echo "  Resume:      ON (skip completed prompts)"
fi
echo ""

for dim in "${DIMENSIONS[@]}"; do
    echo "  - ${dim}: $(dim_count "$dim") prompts ($(dim_file "$dim"))"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    if [ "$SKIP_GEN" = false ]; then
        echo "[DRY RUN] Phase 1: Would generate $TOTAL_VIDEOS videos."
    fi
    if [ "$SKIP_EVAL" = false ]; then
        echo "[DRY RUN] Phase 2: Would evaluate ${#MODELS[@]} models with dual VLM judges."
        echo ""
        echo "Running evaluation dry-run..."
        echo ""
        (cd "$ROOT_DIR" && python3 -m video_gen_eval.eval \
            --dataset-dir "$DATASET_DIR" \
            evaluate \
            --videos-dir "$PROJECT_DIR/$OUTPUT_DIR" \
            --results-dir "$PROJECT_DIR/$RESULTS_DIR" \
            --models "${MODELS[@]}" \
            --dimensions "${DIMENSIONS[@]}" \
            --no-auto-metrics \
            --dry-run \
            --verbose) || true
    fi
    exit 0
fi

# =============================================================================
# Phase 1: Video Generation
# =============================================================================
GEN_FAIL=0

if [ "$SKIP_GEN" = false ]; then
    if [ -z "$FAL_KEY" ]; then
        echo "ERROR: FAL_KEY / FAL_AI_API_KEY is not set."
        echo "Set it in $ROOT_DIR/.env"
        exit 1
    fi

    STARTED_AT=$(date +%s)

    echo ""
    echo "============================================================"
    echo "  Phase 1: Video Generation"
    echo "============================================================"

    if [ "$PARALLEL" = true ]; then
        # ---------------------------------------------------------------
        # Parallel: one background process per model
        # ---------------------------------------------------------------
        echo "Starting ${#MODELS[@]} models in parallel..."
        echo ""

        PIDS=()
        LOG_DIR="$OUTPUT_DIR/.logs"
        mkdir -p "$LOG_DIR"

        for model in "${MODELS[@]}"; do
            log_file="$LOG_DIR/${model}.log"

            (
                resume_flag=""
                if [ "$RESUME" = true ]; then
                    resume_flag="--resume"
                fi

                fail=0
                for filepath in "${INPUT_FILES[@]}"; do
                    filename=$(basename "$filepath")
                    dim_name="${filename%%_prompts.json}"

                    echo "[$(date '+%H:%M:%S')] [$model] Starting $dim_name..."

                    if ! python3 main.py batch \
                        --input "$filepath" \
                        --model "$model" \
                        --duration "$DURATION" \
                        --resolution "$RESOLUTION" \
                        --output "$OUTPUT_DIR" \
                        $resume_flag; then
                        echo "[$(date '+%H:%M:%S')] [$model] $dim_name: FAILED"
                        fail=$((fail + 1))
                    else
                        echo "[$(date '+%H:%M:%S')] [$model] $dim_name: DONE"
                    fi
                done

                exit $fail
            ) > "$log_file" 2>&1 &

            PIDS+=($!)
            echo "  $model (PID $!) → log: $log_file"
        done

        echo ""
        echo "All models launched. Waiting for completion..."
        echo "  Monitor logs: tail -f $LOG_DIR/*.log"
        echo ""

        # Wait for all and collect results
        GEN_PASS=0
        for idx in "${!MODELS[@]}"; do
            model="${MODELS[$idx]}"
            pid="${PIDS[$idx]}"

            if wait "$pid"; then
                echo "  $model: DONE"
                GEN_PASS=$((GEN_PASS + 1))
            else
                echo "  $model: FAILED (check $LOG_DIR/${model}.log)"
                GEN_FAIL=$((GEN_FAIL + 1))
            fi
        done

        GEN_TOTAL=${#MODELS[@]}

    else
        # ---------------------------------------------------------------
        # Sequential: one model at a time
        # ---------------------------------------------------------------
        GEN_PASS=0
        RUN_INDEX=0
        GEN_TOTAL=$((${#MODELS[@]} * ${#INPUT_FILES[@]}))

        RESUME_FLAG=""
        if [ "$RESUME" = true ]; then
            RESUME_FLAG="--resume"
        fi

        for model in "${MODELS[@]}"; do
            for filepath in "${INPUT_FILES[@]}"; do
                RUN_INDEX=$((RUN_INDEX + 1))
                filename=$(basename "$filepath")
                dim_name="${filename%%_prompts.json}"

                echo ""
                echo "------------------------------------------------------------"
                echo "  [$RUN_INDEX/$GEN_TOTAL] $model × $dim_name"
                echo "------------------------------------------------------------"

                if python3 main.py batch \
                    --input "$filepath" \
                    --model "$model" \
                    --duration "$DURATION" \
                    --resolution "$RESOLUTION" \
                    --output "$OUTPUT_DIR" \
                    $RESUME_FLAG; then
                    echo "[$RUN_INDEX/$GEN_TOTAL] $model × $dim_name: DONE"
                    GEN_PASS=$((GEN_PASS + 1))
                else
                    echo "[$RUN_INDEX/$GEN_TOTAL] $model × $dim_name: FAILED"
                    GEN_FAIL=$((GEN_FAIL + 1))
                fi
            done
        done
    fi

    GEN_ENDED_AT=$(date +%s)
    GEN_ELAPSED=$((GEN_ENDED_AT - STARTED_AT))
    GEN_MIN=$((GEN_ELAPSED / 60))
    GEN_SEC=$((GEN_ELAPSED % 60))

    echo ""
    echo "  Phase 1 complete: $GEN_PASS/$GEN_TOTAL succeeded (${GEN_MIN}m ${GEN_SEC}s)"
fi

# =============================================================================
# Phase 2: Evaluation
# =============================================================================
EVAL_EXIT=0

if [ "$SKIP_EVAL" = false ]; then
    # Check for OPENROUTER_API_KEY (needed for VLM judges)
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo ""
        echo "WARNING: OPENROUTER_API_KEY is not set. Evaluation requires VLM judge access."
        echo "Set it in $ROOT_DIR/.env"
        echo "Skipping evaluation."
        EVAL_EXIT=1
    else
        echo ""
        echo "============================================================"
        echo "  Phase 2: Evaluation (Dual VLM Judges)"
        echo "============================================================"
        echo ""

        EVAL_STARTED_AT=$(date +%s)

        # Build dimension flags
        DIM_FLAGS=()
        for dim in "${DIMENSIONS[@]}"; do
            DIM_FLAGS+=("$dim")
        done

        if (cd "$ROOT_DIR" && python3 -m video_gen_eval.eval \
            --dataset-dir "$DATASET_DIR" \
            evaluate \
            --videos-dir "$PROJECT_DIR/$OUTPUT_DIR" \
            --results-dir "$PROJECT_DIR/$RESULTS_DIR" \
            --models "${MODELS[@]}" \
            --dimensions "${DIM_FLAGS[@]}" \
            --no-auto-metrics \
            --verbose); then
            EVAL_EXIT=0
        else
            EVAL_EXIT=1
        fi

        EVAL_ENDED_AT=$(date +%s)
        EVAL_ELAPSED=$((EVAL_ENDED_AT - EVAL_STARTED_AT))
        EVAL_MIN=$((EVAL_ELAPSED / 60))
        EVAL_SEC=$((EVAL_ELAPSED % 60))

        if [ $EVAL_EXIT -eq 0 ]; then
            echo ""
            echo "  Phase 2 complete: evaluation succeeded (${EVAL_MIN}m ${EVAL_SEC}s)"
        else
            echo ""
            echo "  Phase 2 complete: evaluation FAILED (${EVAL_MIN}m ${EVAL_SEC}s)"
        fi
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  EduVBench Benchmark Complete"
echo "============================================================"
echo ""
echo "  Pipeline:    $PHASES"

if [ "$SKIP_GEN" = false ]; then
    echo "  Generation:  $GEN_PASS/$GEN_TOTAL succeeded (${GEN_MIN}m ${GEN_SEC}s)"
    echo "  Videos:      $OUTPUT_DIR"
fi

if [ "$SKIP_EVAL" = false ]; then
    if [ $EVAL_EXIT -eq 0 ]; then
        echo "  Evaluation:  PASSED"
    else
        echo "  Evaluation:  FAILED"
    fi
    echo "  Results:     $RESULTS_DIR"
fi

echo ""

TOTAL_FAIL=$((GEN_FAIL + EVAL_EXIT))

if [ $TOTAL_FAIL -eq 0 ]; then
    echo "  All phases completed successfully!"
else
    echo "  WARNING: Some phases had failures. Check logs above."
fi
echo "============================================================"

exit $TOTAL_FAIL
