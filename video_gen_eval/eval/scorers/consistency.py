"""Consistency scorer for A-DD prompt-variation robustness items.

Evaluates whether a video generation model produces consistent quality
across semantically equivalent prompts (different phrasings of the same
educational content).  Each prompt in a variation group receives an
individual rubric-10pt score, then the group's coefficient of variation
(CV = std / mean) is used to derive a consistency score.

Applies to: A-DD subcategories (variation groups and context tests).
"""

import logging
import math
from typing import Any, Dict, List

from . import BaseScorer
from .rubric_10pt import Rubric10ptScorer
from ..models import ItemEvalResult, Prompt, ScoringMethod, VideoRef
from ..video_utils import extract_frames

logger = logging.getLogger(__name__)


class ConsistencyScorer(BaseScorer):
    """Group-level consistency scorer for A-DD items.

    Individual ``score()`` calls raise :class:`NotImplementedError` because
    consistency is inherently a group-level metric.  Use :meth:`score_group`
    instead, which evaluates all prompts in a variation group together.

    Args:
        rubrics_config: Rubrics configuration dictionary (passed through to
            the internal :class:`Rubric10ptScorer`).
    """

    def __init__(self, rubrics_config: dict) -> None:
        self._rubric_scorer = Rubric10ptScorer(rubrics_config)

    @property
    def method_name(self) -> str:
        return "consistency"

    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Not supported for individual items.

        Raises:
            NotImplementedError: Always.  Use :meth:`score_group` instead.
        """
        raise NotImplementedError(
            "ConsistencyScorer requires group-level evaluation. "
            "Call score_group() with all prompts in the variation group."
        )

    async def score_group(
        self,
        prompts: List[Prompt],
        video_refs_dict: Dict[str, VideoRef],
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> List[ItemEvalResult]:
        """Evaluate a group of variation prompts for output consistency.

        Steps:
        1. Score each prompt individually using the 10-point rubric.
        2. Compute the coefficient of variation (CV) across the group.
        3. Derive ``consistency_score = 1.0 - min(cv, 1.0)``.
        4. Return an :class:`ItemEvalResult` for every prompt in the group,
           all carrying the same consistency score.

        Args:
            prompts:          All prompts in the variation group.
            video_refs_dict:  Mapping of prompt ID to :class:`VideoRef`.
            vlm_client:       VLM client instance.
            config:           Configuration dict passed to the rubric scorer.

        Returns:
            A list of :class:`ItemEvalResult` -- one per prompt -- each with
            the same group-level consistency score.
        """
        if not prompts:
            return []

        judge_id = getattr(vlm_client, "judge_id", "")

        # --- Step 1: Individual rubric scores ---
        individual_scores: List[float] = []
        individual_results: List[ItemEvalResult] = []
        errored_ids: List[str] = []

        for prompt in prompts:
            video_ref = video_refs_dict.get(prompt.id)
            if video_ref is None:
                logger.warning(
                    "No video ref for prompt %s in consistency group", prompt.id
                )
                errored_ids.append(prompt.id)
                individual_results.append(
                    ItemEvalResult(
                        prompt_id=prompt.id,
                        model_id="unknown",
                        judge_id=judge_id,
                        scoring_method=ScoringMethod.CONSISTENCY,
                        score=0.0,
                        raw_score=0,
                        details={"error": "No video reference found"},
                        error="Missing video reference",
                    )
                )
                continue

            # Re-use the rubric scorer for the individual quality assessment
            try:
                result = await self._rubric_scorer.score(
                    prompt, video_ref, vlm_client, config
                )
                individual_scores.append(result.score)
                individual_results.append(result)
            except Exception as exc:
                logger.error(
                    "Rubric scoring failed for %s in consistency group: %s",
                    prompt.id,
                    exc,
                )
                errored_ids.append(prompt.id)
                individual_results.append(
                    ItemEvalResult(
                        prompt_id=prompt.id,
                        model_id=video_ref.model_id,
                        judge_id=judge_id,
                        scoring_method=ScoringMethod.CONSISTENCY,
                        score=0.0,
                        raw_score=0,
                        details={"error": f"Rubric scoring failed: {exc}"},
                        error=str(exc),
                    )
                )

        # --- Step 2: Compute CV ---
        if len(individual_scores) < 2:
            # Cannot compute CV with fewer than 2 valid scores
            logger.warning(
                "Consistency group has fewer than 2 valid scores (%d). "
                "Returning 0.0 for all items.",
                len(individual_scores),
            )
            cv = 1.0  # worst case
            mean_score = individual_scores[0] if individual_scores else 0.0
            std_score = 0.0
        else:
            n = len(individual_scores)
            mean_score = sum(individual_scores) / n
            variance = sum((s - mean_score) ** 2 for s in individual_scores) / n
            std_score = math.sqrt(variance)

            if mean_score > 0:
                cv = std_score / mean_score
            else:
                # All scores are zero -- maximum inconsistency is not meaningful
                cv = 1.0

        # --- Step 3: Consistency score ---
        consistency_score = 1.0 - min(cv, 1.0)

        # Clamp to valid range
        consistency_score = max(0.0, min(1.0, consistency_score))

        logger.info(
            "Consistency group (%d prompts): mean=%.3f, std=%.3f, cv=%.3f, "
            "consistency=%.3f, errors=%d",
            len(prompts),
            mean_score,
            std_score,
            cv,
            consistency_score,
            len(errored_ids),
        )

        # --- Step 4: Build group results ---
        group_details = {
            "group_size": len(prompts),
            "valid_scores": len(individual_scores),
            "mean_rubric_score": round(mean_score, 4),
            "std_rubric_score": round(std_score, 4),
            "coefficient_of_variation": round(cv, 4),
            "individual_scores": [round(s, 4) for s in individual_scores],
            "errored_prompt_ids": errored_ids,
        }

        results: List[ItemEvalResult] = []
        for i, prompt in enumerate(prompts):
            video_ref = video_refs_dict.get(prompt.id)
            model_id = video_ref.model_id if video_ref else "unknown"

            # Merge individual rubric details with group details
            item_details = dict(group_details)
            if i < len(individual_results):
                ind_result = individual_results[i]
                item_details["individual_rubric_score"] = round(ind_result.score, 4)
                item_details["individual_justification"] = ind_result.details.get(
                    "justification", ""
                )

            results.append(
                ItemEvalResult(
                    prompt_id=prompt.id,
                    model_id=model_id,
                    judge_id=judge_id,
                    scoring_method=ScoringMethod.CONSISTENCY,
                    score=consistency_score,
                    raw_score=round(cv, 4),
                    details=item_details,
                    vlm_response=None,
                )
            )

        return results
