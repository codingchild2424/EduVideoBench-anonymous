"""10-point rubric scorer for multi-criteria educational video evaluation.

Sends sampled video frames (or native video URL) together with the generic
10-point scale and prompt-specific rubric criteria to a VLM judge, parses
the structured JSON response, and normalises the 1--10 integer score to [0, 1].

Applies to: K-CK-R, K-PK-CTML, K-PK-VD (rubric items), S-PF, S-UC,
A-ES, A-IS subcategories.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from . import BaseScorer
from ..models import ItemEvalResult, Prompt, ScoringMethod, VideoRef
from ..video_utils import extract_frames
from ..vlm_client import RUBRIC_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLM prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert educational video evaluator. You will assess the quality \
of an AI-generated educational video using a structured 10-point rubric.

## Generic 10-Point Scale

{generic_scale}

## Prompt-Specific Rubric Criteria

{specific_criteria}

## Instructions

1. Carefully examine ALL provided video frames in sequence.
2. Evaluate the video against BOTH the generic scale and the specific rubric criteria.
3. Consider educational accuracy, visual clarity, and pedagogical effectiveness.
4. Respond with a JSON object containing your evaluation."""

_USER_PROMPT_TEMPLATE = """\
The following educational video was generated from this prompt:
"{prompt_text}"

Subject: {subject}
Grade Level: {grade_level}
Category: {category} / {subcategory}

Evaluate these video frames using the rubric criteria provided. \
Assign a score from 1 to 10 and justify your assessment."""


def _format_generic_scale(rubrics_config: dict) -> str:
    """Format the generic 10-point scale from rubrics.json into readable text."""
    generic = rubrics_config.get("rubric_10pt_generic", {})
    scale_items = generic.get("scale", [])
    if not scale_items:
        return (
            "9-10: Excellent (immediately usable)\n"
            "7-8: Good (minor improvements needed)\n"
            "5-6: Adequate (significant editing needed)\n"
            "3-4: Poor (low educational value)\n"
            "1-2: Very Poor (no educational value)"
        )
    lines = []
    for item in scale_items:
        grade = item.get("grade", "")
        score_range = item.get("range", [])
        criteria = item.get("criteria", "")
        range_str = "-".join(str(s) for s in score_range) if score_range else "?"
        lines.append(f"{range_str} ({grade}): {criteria}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_structured_response(
    response: str,
) -> tuple[Optional[int], str, List[str], Dict[str, Any]]:
    """Parse a structured JSON response from the VLM.

    Returns:
        Tuple of (score, reasoning, key_observations, rubric_alignment).
        Score may be ``None`` on parse failure.
    """
    # Try JSON parsing (structured output should be clean JSON)
    try:
        data = json.loads(response.strip())
        score = int(round(float(data["score"])))
        reasoning = str(data.get("reasoning", ""))
        key_observations = data.get("key_observations", [])
        rubric_alignment = data.get("rubric_alignment", {})
        if 1 <= score <= 10:
            return score, reasoning, key_observations, rubric_alignment
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Try extracting JSON from within the response (e.g. markdown code block)
    json_match = re.search(r"\{[^{}]*\"score\"\s*:\s*(\d+)[^{}]*\}", response)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            score = int(round(float(data["score"])))
            reasoning = str(data.get("reasoning", data.get("justification", "")))
            key_observations = data.get("key_observations", [])
            rubric_alignment = data.get("rubric_alignment", {})
            if 1 <= score <= 10:
                return score, reasoning, key_observations, rubric_alignment
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # Regex fallback: look for "score": N patterns
    score_match = re.search(
        r"[\"']?score[\"']?\s*[:=]\s*(\d{1,2})", response, re.IGNORECASE
    )
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 10:
            return score, "", [], {}

    # Last resort: look for any standalone number 1-10
    num_match = re.search(r"\b(\d{1,2})\b", response)
    if num_match:
        score = int(num_match.group(1))
        if 1 <= score <= 10:
            return score, "", [], {}

    return None, "", [], {}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Rubric10ptScorer(BaseScorer):
    """VLM-based 10-point rubric scorer.

    Uses the generic EduVBench 10-point scale combined with prompt-specific
    rubric criteria to produce a normalised score with structured reasoning.

    Args:
        rubrics_config: The full rubrics configuration dictionary loaded from
            ``rubrics.json``. Used to render the generic scale into the
            system prompt.
    """

    def __init__(self, rubrics_config: dict) -> None:
        self._rubrics_config = rubrics_config
        self._generic_scale_text = _format_generic_scale(rubrics_config)

    @property
    def method_name(self) -> str:
        return "rubric_10pt"

    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Score a video using the 10-point rubric with structured output.

        Args:
            prompt:     Evaluation prompt with ``rubric_criteria``.
            video_ref:  Reference to the generated video.
            vlm_client: VLM client for multi-frame or video URL queries.
            config:     Configuration dict (``num_frames`` defaults to 8).

        Returns:
            :class:`ItemEvalResult` with ``score`` normalised to [0, 1]
            (raw_score / 10) and structured reasoning fields.
        """
        num_frames = config.get("num_frames", 8)

        # Extract frames
        try:
            frame_paths = extract_frames(video_ref.video_path, num_frames=num_frames)
        except Exception as exc:
            logger.error(
                "Frame extraction failed for %s: %s", video_ref.video_path, exc
            )
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=getattr(vlm_client, "judge_id", ""),
                scoring_method=ScoringMethod.RUBRIC_10PT,
                score=0.0,
                raw_score=0,
                details={"error": f"Frame extraction failed: {exc}"},
                error=str(exc),
            )

        # Build prompts
        specific_criteria = prompt.rubric_criteria or "No specific rubric criteria provided."
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            generic_scale=self._generic_scale_text,
            specific_criteria=specific_criteria,
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            prompt_text=prompt.prompt_text,
            subject=prompt.subject,
            grade_level=prompt.grade_level,
            category=prompt.category,
            subcategory=prompt.subcategory,
        )

        # Query VLM with structured output
        try:
            vlm_response = await vlm_client.query_with_video(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_path=video_ref.video_path,
                video_url=config.get("video_url"),
                frame_paths=frame_paths,
                num_frames=num_frames,
                response_format=RUBRIC_RESPONSE_SCHEMA,
            )
        except Exception as exc:
            logger.error("VLM query failed for prompt %s: %s", prompt.id, exc)
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=getattr(vlm_client, "judge_id", ""),
                scoring_method=ScoringMethod.RUBRIC_10PT,
                score=0.0,
                raw_score=0,
                details={"error": f"VLM query failed: {exc}"},
                vlm_response=None,
                error=str(exc),
            )

        # Parse structured response
        raw_score, reasoning, key_observations, rubric_alignment = (
            _parse_structured_response(vlm_response)
        )

        if raw_score is None:
            logger.warning(
                "Could not parse VLM response for prompt %s: %s",
                prompt.id,
                vlm_response[:200],
            )
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=getattr(vlm_client, "judge_id", ""),
                scoring_method=ScoringMethod.RUBRIC_10PT,
                score=0.0,
                raw_score=0,
                details={
                    "error": "Failed to parse VLM response",
                    "vlm_response_snippet": vlm_response[:500],
                },
                vlm_response=vlm_response,
                error="Failed to parse score from VLM response",
            )

        normalised_score = raw_score / 10.0

        logger.info(
            "Rubric10pt [%s][%s]: raw=%d, normalised=%.2f",
            prompt.id,
            getattr(vlm_client, "judge_id", "?"),
            raw_score,
            normalised_score,
        )

        return ItemEvalResult(
            prompt_id=prompt.id,
            model_id=video_ref.model_id,
            judge_id=getattr(vlm_client, "judge_id", ""),
            scoring_method=ScoringMethod.RUBRIC_10PT,
            score=normalised_score,
            raw_score=raw_score,
            reasoning=reasoning,
            key_observations=key_observations,
            rubric_alignment=rubric_alignment,
            details={
                "rubric_criteria_used": specific_criteria,
                "num_frames": num_frames,
            },
            vlm_response=vlm_response,
        )
