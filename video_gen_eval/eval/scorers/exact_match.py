"""Exact-match scorer for K-CK factual items.

Extracts a factual answer from the generated video via a VLM query with
structured JSON output, compares it against the ground-truth
``correct_answer`` using fuzzy matching (case-insensitive, whitespace-stripped,
containment check).

Applies to: K-CK-F, K-CK-C, K-CK-P, K-CK-M subcategories.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from . import BaseScorer
from ..models import ItemEvalResult, Prompt, ScoringMethod, VideoRef
from ..video_utils import extract_frames
from ..vlm_client import EXACT_MATCH_RESPONSE_SCHEMA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLM prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert educational video evaluator. Your task is to carefully "
    "examine the provided video frames and extract the specific factual answer "
    "displayed in the video. Focus exclusively on what is visually presented -- "
    "formulas, labels, sequences, diagrams, or text shown on screen.\n\n"
    "You will also be given the expected correct answer. Compare what you see "
    "in the video against the expected answer and determine if it is correct.\n\n"
    "Respond with a JSON object containing your extraction and assessment."
)


def _build_user_prompt(prompt: Prompt) -> str:
    """Construct the user prompt from the evaluation prompt and visual hints."""
    parts = [
        "The following educational video was generated from this prompt:",
        f'"{prompt.prompt_text}"',
        "",
        "Examine the video frames carefully and extract the specific answer "
        "or factual content displayed in the video.",
    ]

    key_elements = prompt.ground_truth.get("key_visual_elements")
    if key_elements:
        parts.append("")
        parts.append("Look specifically for these visual elements:")
        for element in key_elements:
            parts.append(f"  - {element}")

    expected = prompt.ground_truth.get("correct_answer", "")
    parts.append("")
    parts.append(f"Expected correct answer: {expected}")
    parts.append("")
    parts.append(
        "Extract the answer from the video, compare it to the expected answer, "
        "and indicate whether it is correct."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_structured_response(
    response: str,
) -> tuple[Optional[str], Optional[bool], str]:
    """Parse the structured JSON response for extracted answer and correctness.

    Returns:
        Tuple of (extracted_answer, is_correct, reasoning).
    """
    # Try JSON parsing
    try:
        data = json.loads(response.strip())
        extracted = str(data.get("extracted_answer", ""))
        is_correct = bool(data.get("is_correct", False))
        reasoning = str(data.get("reasoning", ""))
        return extracted, is_correct, reasoning
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Try extracting JSON from within the response
    json_match = re.search(
        r"\{[^{}]*\"extracted_answer\"[^{}]*\}", response, re.DOTALL
    )
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            extracted = str(data.get("extracted_answer", ""))
            is_correct = bool(data.get("is_correct", False))
            reasoning = str(data.get("reasoning", ""))
            return extracted, is_correct, reasoning
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # Fallback: treat entire response as extracted answer
    return response.strip(), None, ""


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip whitespace, and collapse internal runs of whitespace."""
    return " ".join(text.lower().split())


def _fuzzy_match(extracted: str, expected: str) -> bool:
    """Return True if *extracted* and *expected* are a fuzzy match."""
    norm_ext = _normalise(extracted)
    norm_exp = _normalise(expected)

    if not norm_ext or norm_ext == "not_found":
        return False

    if norm_ext == norm_exp:
        return True

    if norm_exp in norm_ext or norm_ext in norm_exp:
        return True

    expected_tokens = set(norm_exp.split())
    extracted_tokens = set(norm_ext.split())
    if expected_tokens and expected_tokens.issubset(extracted_tokens):
        return True

    return False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ExactMatchScorer(BaseScorer):
    """Binary exact-match scorer for factual K-CK items.

    Score: 1.0 if the VLM-extracted answer fuzzy-matches the ground truth,
    0.0 otherwise. Uses structured JSON output for reliable extraction.
    """

    @property
    def method_name(self) -> str:
        return "exact_match"

    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Extract the answer from the video and compare to ground truth.

        Args:
            prompt:     Evaluation prompt with ``ground_truth["correct_answer"]``.
            video_ref:  Reference to the generated video.
            vlm_client: VLM client instance.
            config:     Configuration dict (``num_frames`` defaults to 8).

        Returns:
            :class:`ItemEvalResult` with score 1.0 (match) or 0.0 (mismatch).
        """
        num_frames = config.get("num_frames", 8)

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
                scoring_method=ScoringMethod.EXACT_MATCH,
                score=0.0,
                raw_score=False,
                details={"error": f"Frame extraction failed: {exc}"},
                error=str(exc),
            )

        # Query VLM with structured output
        user_prompt = _build_user_prompt(prompt)
        try:
            vlm_response = await vlm_client.query_with_video(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                video_path=video_ref.video_path,
                video_url=config.get("video_url"),
                frame_paths=frame_paths,
                num_frames=num_frames,
                response_format=EXACT_MATCH_RESPONSE_SCHEMA,
            )
        except Exception as exc:
            logger.error("VLM query failed for prompt %s: %s", prompt.id, exc)
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=getattr(vlm_client, "judge_id", ""),
                scoring_method=ScoringMethod.EXACT_MATCH,
                score=0.0,
                raw_score=False,
                details={"error": f"VLM query failed: {exc}"},
                vlm_response=None,
                error=str(exc),
            )

        # Parse structured response
        extracted_answer, vlm_is_correct, reasoning = _parse_structured_response(
            vlm_response
        )

        # Fuzzy match as ground truth check
        expected = prompt.ground_truth.get("correct_answer", "")
        fuzzy_matched = _fuzzy_match(extracted_answer or "", expected)

        # Use VLM's own correctness judgment if available, else fallback to fuzzy
        if vlm_is_correct is not None:
            matched = vlm_is_correct and fuzzy_matched
        else:
            matched = fuzzy_matched

        final_score = 1.0 if matched else 0.0

        logger.info(
            "ExactMatch [%s][%s]: extracted=%r, expected=%r, match=%s",
            prompt.id,
            getattr(vlm_client, "judge_id", "?"),
            (extracted_answer or "")[:120],
            expected[:120],
            matched,
        )

        return ItemEvalResult(
            prompt_id=prompt.id,
            model_id=video_ref.model_id,
            judge_id=getattr(vlm_client, "judge_id", ""),
            scoring_method=ScoringMethod.EXACT_MATCH,
            score=final_score,
            raw_score=matched,
            reasoning=reasoning,
            details={
                "extracted_answer": extracted_answer or "",
                "expected_answer": expected,
                "fuzzy_matched": fuzzy_matched,
                "vlm_is_correct": vlm_is_correct,
                "matched": matched,
                "num_frames": num_frames,
            },
            vlm_response=vlm_response,
        )
