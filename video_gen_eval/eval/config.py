"""Evaluation configuration for the EduVBench engine.

Provides the :class:`EvalConfig` dataclass that centralises all tunable
parameters, and helpers to hydrate it from on-disk JSON files or CLI
arguments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    """Configuration for a single VLM judge model.

    Attributes:
        id:               Short identifier (e.g. ``gemini-3-pro``).
        model_id:         OpenRouter model string.
        input_method:     ``video_url`` (native) or ``frame_images``.
        reasoning_effort: Reasoning level for thinking models (``low``/``medium``/``high``).
        frames_per_video: Frames to extract when input_method is ``frame_images``.
        temperature:      Sampling temperature.
        max_retries:      Retry attempts on transient errors.
        rate_limit_rpm:   Maximum requests per minute.
        timeout:          Request timeout in seconds.
    """

    id: str = "gemini-3-pro"
    model_id: str = "google/gemini-3-pro-preview"
    input_method: str = "video_url"
    reasoning_effort: Optional[str] = "low"
    frames_per_video: int = 8
    temperature: float = 0.0
    max_retries: int = 3
    rate_limit_rpm: int = 60
    timeout: int = 120


# Default dual-judge configuration
DEFAULT_JUDGES = [
    JudgeConfig(
        id="gemini-3-flash",
        model_id="google/gemini-3-flash-preview",
        input_method="frame_images",
        reasoning_effort=None,
        frames_per_video=8,
    ),
    JudgeConfig(
        id="gpt-4o",
        model_id="openai/gpt-4o",
        input_method="frame_images",
        reasoning_effort=None,
        frames_per_video=8,
    ),
]


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Complete configuration for a single evaluation run.

    Attributes:
        models:              List of video generation model IDs to evaluate.
        dimensions:          Optional subset of dimensions to evaluate.
        categories:          Optional subset of category codes.
        prompt_ids:          Optional explicit list of prompt IDs.
        dataset_dir:         Path to the EduVBench dataset directory.
        videos_dir:          Root directory of generated video files.
        results_dir:         Directory where evaluation results will be written.
        cache_dir:           Directory for caching VLM responses.
        judges:              List of VLM judge configurations (dual-judge).
        score_merge_method:  How to merge dual-judge scores (``mean_with_flag``).
        score_flag_threshold: Max score diff before flagging for review.
        enable_auto_metrics: Whether to compute automated frame-level metrics.
        frames_per_video:    Default frames to sample per video.
        viu_trials:          Number of VIU block trials per prompt.
        dry_run:             Walk the pipeline without VLM calls.
        verbose:             Emit debug-level logging.
    """

    # -- Scope ---------------------------------------------------------------
    models: List[str] = field(default_factory=list)
    dimensions: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    prompt_ids: Optional[List[str]] = None

    # -- Paths ---------------------------------------------------------------
    dataset_dir: str = "eduvbench-dataset"
    videos_dir: str = "videos"
    results_dir: str = "results"
    cache_dir: str = ".cache/eval"

    # -- VLM judges ----------------------------------------------------------
    judges: List[JudgeConfig] = field(default_factory=lambda: list(DEFAULT_JUDGES))
    score_merge_method: str = "mean_with_flag"
    score_flag_threshold: float = 2.0

    # -- Metrics & sampling --------------------------------------------------
    enable_auto_metrics: bool = True
    frames_per_video: int = 8
    viu_trials: int = 30

    # -- Run behaviour -------------------------------------------------------
    dry_run: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

def load_scoring_config(dataset_dir: str) -> Dict[str, Any]:
    """Load the three JSON configuration files from the dataset directory.

    Reads ``scoring_config.json``, ``rubrics.json``, and ``models.json``
    from *dataset_dir* and returns them as a single dictionary.

    Args:
        dataset_dir: Path to the EduVBench dataset directory.

    Returns:
        A dictionary with three top-level keys::

            {
                "scoring_config": { ... },
                "rubrics": { ... },
                "models": { ... }
            }

    Raises:
        FileNotFoundError: If any of the three expected files is missing.
        json.JSONDecodeError: If any file contains invalid JSON.
    """
    base = Path(dataset_dir)
    result: Dict[str, Any] = {}

    file_map = {
        "scoring_config": "scoring_config.json",
        "rubrics": "rubrics.json",
        "models": "models.json",
    }

    for key, filename in file_map.items():
        filepath = base / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required config file not found: {filepath}"
            )
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            result[key] = data
            logger.debug("Loaded %s (%d top-level keys)", filepath, len(data))

    return result


def load_judges_from_models_json(dataset_dir: str) -> List[JudgeConfig]:
    """Load judge configurations from models.json.

    Args:
        dataset_dir: Path to the EduVBench dataset directory.

    Returns:
        List of :class:`JudgeConfig` instances.
    """
    filepath = Path(dataset_dir) / "models.json"
    if not filepath.exists():
        logger.warning("models.json not found, using default judges")
        return list(DEFAULT_JUDGES)

    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    eval_vlm = data.get("evaluation_vlm", {})
    judges_data = eval_vlm.get("judges", [])

    if not judges_data:
        logger.warning("No judges found in models.json, using defaults")
        return list(DEFAULT_JUDGES)

    judges = []
    for j in judges_data:
        judges.append(JudgeConfig(
            id=j.get("id", "unknown"),
            model_id=j.get("model_id", ""),
            input_method=j.get("input_method", "frame_images"),
            reasoning_effort=j.get("reasoning_effort"),
            frames_per_video=j.get("frames_per_video", 8),
        ))

    logger.info("Loaded %d judges from models.json: %s",
                len(judges), [j.id for j in judges])
    return judges


# ---------------------------------------------------------------------------
# CLI argument conversion
# ---------------------------------------------------------------------------

def load_eval_config_from_args(args: Any) -> EvalConfig:
    """Convert an :mod:`argparse` ``Namespace`` into an :class:`EvalConfig`.

    This function maps CLI flag names to ``EvalConfig`` field names using a
    straightforward attribute lookup.  Unknown attributes on *args* are
    silently ignored so that callers can extend the CLI without updating
    this function.

    Args:
        args: An :class:`argparse.Namespace` (or any object with matching
              attributes).

    Returns:
        A fully populated :class:`EvalConfig` instance.
    """
    field_names = {f.name for f in EvalConfig.__dataclass_fields__.values()}
    kwargs: Dict[str, Any] = {}

    for name in field_names:
        if name == "judges":
            continue  # loaded separately from models.json
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value

    config = EvalConfig(**kwargs)

    # Load judges from models.json
    config.judges = load_judges_from_models_json(config.dataset_dir)

    if config.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("EvalConfig created from CLI args: %s", kwargs)

    return config
