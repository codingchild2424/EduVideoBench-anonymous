"""Core data models for the EduVBench evaluation engine.

Defines the dataclasses and enums used throughout the evaluation pipeline,
from individual prompt items through to final model scorecards.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoringMethod(Enum):
    """Evaluation method used to score a single prompt item.

    Each prompt in the EduVBench dataset specifies one of these methods,
    which determines how the generated video is evaluated against the
    ground truth.

    Members:
        EXACT_MATCH:   Binary check -- the video must display a specific
                       factual element (formula, label, sequence).
        RUBRIC_10PT:   VLM-based 10-point rubric assessment with criteria
                       defined per prompt or per subcategory.
        AUTO_METRIC:   Automated frame-level metrics (e.g. cognitive load
                       proxy, visual distraction score).
        VLM_COMPARISON: Side-by-side VLM comparison between generated video
                       and a reference video or specification.
        BLOCK_TEST:    Safety gate -- checks whether the model refused to
                       generate harmful content (A-NE category).
        CONSISTENCY:   Prompt-variation robustness test measuring coefficient
                       of variation across a group of semantically equivalent
                       prompts (A-DD category).
    """

    EXACT_MATCH = "exact_match"
    RUBRIC_10PT = "rubric_10pt"
    AUTO_METRIC = "auto_metric"
    VLM_COMPARISON = "vlm_comparison"
    BLOCK_TEST = "block_test"
    CONSISTENCY = "consistency"


# ---------------------------------------------------------------------------
# Prompt & Video Reference
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A single evaluation prompt from the EduVBench dataset.

    Each prompt describes an educational video to be generated and contains
    the ground-truth expectations plus scoring configuration needed to
    evaluate the result.

    Attributes:
        id:               Unique prompt identifier (e.g. ``EVB-Sci-ELow-K-CK-F1-001``).
        subject:          Subject area (``science``, ``math``, ``social``, ``art``, ``language``).
        grade_level:       Target grade band (``elementary_low``, ``elementary_high``,
                          ``middle``, ``high``, ``none``).
        category:         KSA category code (``K-CK``, ``K-PK``, ``S-PF``, ``S-UC``,
                          ``S-VIU``, ``A-ES``, ``A-IS``, ``A-NE``, ``A-DD``).
        subcategory:      Fine-grained subcategory (e.g. ``K-CK-F1``, ``S-PF-V1``).
        prompt_text:      The text prompt to send to the video generation model.
        ground_truth:     Expected outcome details (structure varies by category).
        rubric_criteria:  Human-readable rubric description for VLM judge prompts.
        scoring_method:   The :class:`ScoringMethod` to use when evaluating.
        metadata:         Additional per-prompt metadata (cross-axes, eval hints, etc.).
        bloom_questions:  Optional higher-order comprehension questions for VIU tests.
        reference_video_id: Optional ID of a reference video for VLM comparison scoring.
    """

    id: str
    subject: str
    grade_level: str
    category: str
    subcategory: str
    prompt_text: str
    ground_truth: Dict[str, Any]
    rubric_criteria: str
    scoring_method: ScoringMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
    bloom_questions: Optional[List[str]] = None
    reference_video_id: Optional[str] = None


@dataclass
class VideoRef:
    """Reference to a generated video file and its associated metadata.

    Attributes:
        video_path:    Absolute or relative path to the video file.
        model_id:      Identifier of the model that generated the video.
        prompt_id:     Identifier of the prompt used to generate the video.
        metadata_path: Optional path to a sidecar JSON with generation metadata
                       (e.g. seed, resolution, actual duration).
    """

    video_path: str
    model_id: str
    prompt_id: str
    metadata_path: Optional[str] = None
    video_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation Results (leaf -> aggregate)
# ---------------------------------------------------------------------------

@dataclass
class ItemEvalResult:
    """Result of evaluating a single prompt--model pair.

    Scores are stored in two forms:
    - ``score``: normalised to [0, 1] for cross-method aggregation.
    - ``raw_score``: the native score (e.g. integer 1--10 for rubric, bool
      for block test, float metric value for auto metrics).

    Attributes:
        prompt_id:         The prompt that was evaluated.
        model_id:          The model whose output was evaluated.
        judge_id:          Identifier of the VLM judge that scored this item.
        scoring_method:    Method used for this evaluation.
        score:             Normalised score in [0.0, 1.0].
        raw_score:         Native score value before normalisation.
        reasoning:         VLM judge's explanation for the score.
        key_observations:  Specific visual/audio evidence observed by the judge.
        rubric_alignment:  Which rubric criteria were met/missed.
        details:           Scoring breakdown, evidence, or intermediate values.
        vlm_response:      Raw VLM judge response text (if applicable).
        error:             Error message if evaluation failed for this item.
        timestamp:         When the evaluation was performed.
    """

    prompt_id: str
    model_id: str
    judge_id: str = ""
    scoring_method: ScoringMethod = ScoringMethod.RUBRIC_10PT
    score: float = 0.0
    raw_score: Any = None
    reasoning: str = ""
    key_observations: List[str] = field(default_factory=list)
    rubric_alignment: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    vlm_response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate that normalised score is within [0, 1]."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"Normalised score must be in [0, 1], got {self.score}"
            )


@dataclass
class CategoryScore:
    """Aggregated score for one KSA category (e.g. K-CK, S-PF, A-NE).

    Attributes:
        category:     Category code.
        model_id:     Model being scored.
        score:        Weighted or averaged normalised score for the category.
        item_count:   Number of items evaluated in this category.
        item_results: Individual :class:`ItemEvalResult` instances.
        breakdown:    Per-subcategory or per-method score breakdown.
    """

    category: str
    model_id: str
    score: float
    item_count: int
    item_results: List[ItemEvalResult] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Aggregated score for one KSA dimension (Knowledge, Skills, or Attitude).

    Attributes:
        dimension:       Dimension name (``knowledge``, ``skills``, ``attitude``).
        model_id:        Model being scored.
        score:           Weighted normalised score for the dimension.
        category_scores: Mapping of category code to :class:`CategoryScore`.
    """

    dimension: str
    model_id: str
    score: float
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)


@dataclass
class ModelScoreCard:
    """Final evaluation scorecard for one video generation model.

    Combines all dimension scores into the composite EduVBench-KSA score
    and records whether the model passed the safety gate (A-NE block rate).

    Attributes:
        model_id:           Unique model identifier.
        model_name:         Human-readable model name.
        final_score:        Composite EduVBench-KSA score in [0, 1].
        safety_gate_passed: ``True`` if A-NE block rate >= 0.90.
        dimension_scores:   Mapping of dimension name to :class:`DimensionScore`.
        timestamp:          When the scorecard was generated.
        metadata:           Arbitrary metadata (config snapshot, run ID, etc.).
    """

    model_id: str
    model_name: str
    final_score: float
    safety_gate_passed: bool
    dimension_scores: Dict[str, DimensionScore] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
