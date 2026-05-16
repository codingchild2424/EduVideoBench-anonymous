"""Command-line interface for the EduVBench evaluation engine.

Provides subcommands for running evaluations, checking system status,
and generating reports from saved results.

Usage:
    python -m video_gen_eval.eval evaluate --models sora veo3 --verbose
    python -m video_gen_eval.eval status
    python -m video_gen_eval.eval report --results-dir ./results/run_20260217_120000
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluation pipeline.

    Args:
        args: Parsed CLI arguments for the 'evaluate' subcommand.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from .config import EvalConfig, load_judges_from_models_json
    from .pipeline import EvalPipeline

    config = EvalConfig(
        models=args.models or [],
        videos_dir=args.videos_dir,
        results_dir=args.results_dir,
        dataset_dir=args.dataset_dir,
        dimensions=args.dimensions,
        categories=args.categories,
        prompt_ids=args.prompt_ids,
        enable_auto_metrics=not args.no_auto_metrics,
        dry_run=args.dry_run,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
    )

    # Load judges from models.json
    config.judges = load_judges_from_models_json(config.dataset_dir)

    logger.info("Evaluation configuration: models=%s, judges=%s",
                config.models, [j.id for j in config.judges])

    try:
        pipeline = EvalPipeline(config)
    except FileNotFoundError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        logger.error("Initialisation error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        score_cards = asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        print(f"Error: Evaluation failed: {exc}", file=sys.stderr)
        return 1

    if score_cards:
        _print_summary(score_cards)
        print(f"\nResults saved to: {pipeline._reporter.run_dir}")
    else:
        print("No models were evaluated.")

    return 0


def _handle_status(args: argparse.Namespace) -> int:
    """Check system status and print dependency/configuration information.

    Args:
        args: Parsed CLI arguments for the 'status' subcommand.

    Returns:
        Exit code (always 0).
    """
    _print_status(dataset_dir=args.dataset_dir)
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    """Generate or display a report from saved results.

    Args:
        args: Parsed CLI arguments for the 'report' subcommand.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    output_format = args.format

    # Look for comparative_report or reconstruct from individual score cards
    comparative_path = results_dir / "comparative_report.json"
    leaderboard_path = results_dir / "leaderboard.json"
    md_path = results_dir / "comparative_report.md"

    if output_format == "markdown":
        if md_path.exists():
            print(md_path.read_text(encoding="utf-8"))
            return 0
        else:
            print(
                f"Error: Markdown report not found at {md_path}",
                file=sys.stderr,
            )
            # Fall through to try JSON
            if not comparative_path.exists():
                return 1

    if output_format == "json" or not md_path.exists():
        target = comparative_path if comparative_path.exists() else leaderboard_path
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        else:
            print(
                f"Error: No report files found in {results_dir}",
                file=sys.stderr,
            )
            return 1

    return 0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_summary(score_cards: list) -> None:
    """Print a formatted summary table of evaluation results.

    Args:
        score_cards: List of ModelScoreCard instances.
    """
    from .models import ModelScoreCard

    if not score_cards:
        print("No results to display.")
        return

    # Sort by final score descending
    sorted_cards = sorted(
        score_cards, key=lambda c: c.final_score, reverse=True
    )

    # Column widths
    name_width = max(len(c.model_name) for c in sorted_cards)
    name_width = max(name_width, 10)  # minimum width

    # Header
    header = (
        f"{'Rank':<6}"
        f"{'Model':<{name_width + 2}}"
        f"{'Final':>8}"
        f"{'K':>8}"
        f"{'S':>8}"
        f"{'A':>8}"
        f"{'Safety':>8}"
    )
    separator = "-" * len(header)

    print()
    print("EduVBench-KSA Evaluation Results")
    print(separator)
    print(header)
    print(separator)

    for rank, card in enumerate(sorted_cards, start=1):
        k = _dim_score(card, "knowledge")
        s = _dim_score(card, "skills")
        a = _dim_score(card, "attitude")
        safety = "PASS" if card.safety_gate_passed else "FAIL"

        print(
            f"{rank:<6}"
            f"{card.model_name:<{name_width + 2}}"
            f"{card.final_score:>8.4f}"
            f"{k:>8.4f}"
            f"{s:>8.4f}"
            f"{a:>8.4f}"
            f"{safety:>8}"
        )

    print(separator)
    print(f"Models evaluated: {len(sorted_cards)}")
    print()


def _dim_score(card: object, dimension: str) -> float:
    """Extract a dimension score from a score card.

    Args:
        card: A ModelScoreCard instance.
        dimension: Dimension name.

    Returns:
        The score, or 0.0 if not present.
    """
    dim = card.dimension_scores.get(dimension)  # type: ignore[attr-defined]
    return dim.score if dim else 0.0


def _print_status(dataset_dir: str = "./eduvbench-dataset") -> None:
    """Print system status: dependencies, API keys, and dataset summary.

    Args:
        dataset_dir: Path to the dataset directory.
    """
    print()
    print("EduVBench Evaluation Engine - System Status")
    print("=" * 50)

    # Python version
    print(f"\nPython: {sys.version.split()[0]}")

    # Check dependencies
    print("\nDependencies:")
    deps = {
        "openai": "OpenAI SDK (for OpenRouter VLM access)",
        "cv2": "OpenCV (for frame extraction & auto metrics)",
        "whisper": "Whisper (for speech-to-text in CL metrics)",
        "numpy": "NumPy (numerical operations)",
        "PIL": "Pillow (image processing)",
    }

    for module, description in deps.items():
        try:
            __import__(module)
            status = "AVAILABLE"
        except ImportError:
            status = "NOT FOUND"
        print(f"  {module:<12} {status:<14} {description}")

    # Check ffmpeg
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  {'ffmpeg':<12} {'AVAILABLE':<14} FFmpeg (video processing) at {ffmpeg_path}")
    else:
        print(f"  {'ffmpeg':<12} {'NOT FOUND':<14} FFmpeg (video processing)")

    # API key status
    print("\nAPI Keys:")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  OPENROUTER_API_KEY: SET ({masked})")
    else:
        print("  OPENROUTER_API_KEY: NOT SET")

    # Dataset summary
    print("\nDataset:")
    dataset_path = Path(dataset_dir)
    if dataset_path.is_dir():
        prompt_files = list(dataset_path.glob("*_prompts.json"))
        print(f"  Directory: {dataset_path.resolve()}")
        print(f"  Prompt files: {len(prompt_files)}")

        total_prompts = 0
        for pf in sorted(prompt_files):
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                count = len(data.get("prompts", []))
                total_prompts += count
                dimension = data.get("dimension", "Unknown")
                print(f"    {pf.name}: {count} prompts ({dimension})")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"    {pf.name}: ERROR ({exc})")

        print(f"  Total prompts: {total_prompts}")

        # Scoring config
        config_path = dataset_path / "scoring_config.json"
        if config_path.exists():
            print(f"  Scoring config: {config_path.name} (present)")
        else:
            print("  Scoring config: NOT FOUND")
    else:
        print(f"  Directory not found: {dataset_path}")

    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="video-gen-eval",
        description=(
            "EduVBench Evaluation Engine: evaluate AI-generated educational "
            "videos across Knowledge, Skills, and Attitude dimensions."
        ),
    )

    # Global options
    parser.add_argument(
        "--dataset-dir",
        default="./eduvbench-dataset",
        help="Path to the EduVBench dataset directory (default: ./eduvbench-dataset)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run the evaluation pipeline on generated videos",
    )
    eval_parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model IDs to evaluate (space-separated). "
            "If omitted, auto-discovers from --videos-dir."
        ),
    )
    eval_parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        choices=["knowledge", "skills", "attitude"],
        help="Evaluate only specified dimensions",
    )
    eval_parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Evaluate only specified categories (e.g. K-CK S-PF A-NE)",
    )
    eval_parser.add_argument(
        "--prompt-ids",
        nargs="*",
        default=None,
        help="Evaluate only specified prompt IDs",
    )
    eval_parser.add_argument(
        "--videos-dir",
        default="./outputs",
        help="Root directory for generated videos (default: ./outputs)",
    )
    eval_parser.add_argument(
        "--results-dir",
        default="./results",
        help="Root directory for saving results (default: ./results)",
    )
    eval_parser.add_argument(
        "--cache-dir",
        default=".cache/eval",
        help="Directory for caching VLM responses (default: .cache/eval)",
    )
    eval_parser.add_argument(
        "--no-auto-metrics",
        action="store_true",
        default=False,
        help="Skip auto-metric scorers (cv2/whisper dependent)",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Discover videos and show what would be evaluated without scoring",
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Check system dependencies, API keys, and dataset status",
    )
    status_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging",
    )

    # --- report ---
    report_parser = subparsers.add_parser(
        "report",
        help="Display or regenerate a report from saved results",
    )
    report_parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the results run directory (e.g. ./results/run_20260217_120000)",
    )
    report_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    report_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging",
    )

    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stderr,
    )

    # Reduce noise from third-party libraries
    if not verbose:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the EduVBench evaluation CLI.

    Parses arguments, dispatches to the appropriate subcommand handler,
    and exits with the handler's return code.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    verbose = getattr(args, "verbose", False)
    _setup_logging(verbose=verbose)

    # Dispatch to subcommand handler
    handlers = {
        "evaluate": _handle_evaluate,
        "status": _handle_status,
        "report": _handle_report,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    exit_code = handler(args)
    sys.exit(exit_code)
