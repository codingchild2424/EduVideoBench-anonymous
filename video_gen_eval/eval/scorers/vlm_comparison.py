"""VLM comparison scorer for S-VIU video information utility items.

Measures the educational value-add of a generated video by comparing VLM
accuracy on comprehension questions under two conditions:
  * **Baseline** -- text-only query (no video frames).
  * **With video** -- query includes sampled video frames.

Each condition is run for multiple trials.  The effect size
(video_accuracy - baseline_accuracy) is the primary metric.  Statistical
significance is assessed with McNemar's test on the paired 2x2 contingency
table.

Applies to: S-VIU subcategories.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from . import BaseScorer
from ..models import ItemEvalResult, Prompt, ScoringMethod, VideoRef
from ..video_utils import extract_frames

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLM prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an educational assessment system. You will be asked a "
    "comprehension question about an educational topic. Answer with ONLY "
    "the correct answer -- no explanations, no qualifications, no extra text."
)

_USER_PROMPT_TEXT_ONLY = (
    "Answer the following question based on your knowledge:\n\n"
    "Question: {question}\n\n"
    "Provide only the answer."
)

_USER_PROMPT_WITH_VIDEO = (
    "You are shown frames from an educational video. Use the visual "
    "information to answer the following question:\n\n"
    "Question: {question}\n\n"
    "Provide only the answer."
)


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return " ".join(text.lower().split())


def _answers_match(response: str, expected: str) -> bool:
    """Fuzzy comparison between VLM response and expected answer."""
    norm_resp = _normalise(response)
    norm_exp = _normalise(expected)

    if not norm_resp:
        return False

    if norm_resp == norm_exp:
        return True

    # Containment (either direction)
    if norm_exp in norm_resp or norm_resp in norm_exp:
        return True

    # Token overlap
    exp_tokens = set(norm_exp.split())
    resp_tokens = set(norm_resp.split())
    if exp_tokens and exp_tokens.issubset(resp_tokens):
        return True

    return False


# ---------------------------------------------------------------------------
# Chi-squared p-value approximation (avoids scipy dependency)
# ---------------------------------------------------------------------------

def _chi2_sf_1df(x: float) -> float:
    """Survival function (1 - CDF) for chi-squared distribution with 1 df.

    Uses the complementary error function relationship:
        P(chi2 > x | df=1) = erfc(sqrt(x / 2))

    Falls back to a simple normal-tail approximation when ``math.erfc``
    is not available (should not happen in CPython >= 3.2).
    """
    if x <= 0:
        return 1.0
    try:
        return math.erfc(math.sqrt(x / 2.0))
    except (ValueError, OverflowError):
        return 0.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class VLMComparisonScorer(BaseScorer):
    """Video information utility scorer using paired VLM comparison trials.

    For each Bloom-level question attached to the prompt, runs *N* trials
    in both text-only (baseline) and with-video conditions, builds a McNemar
    contingency table, and computes effect size.

    Args:
        default_trials: Default number of trials per question (overridden by
            ``config["trials_count"]`` at scoring time).  Defaults to 30.
    """

    def __init__(self, default_trials: int = 30) -> None:
        self._default_trials = default_trials

    @property
    def method_name(self) -> str:
        return "vlm_comparison"

    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Run paired comparison trials and compute effect size.

        Args:
            prompt:     Evaluation prompt with ``bloom_questions``.
            video_ref:  Reference to the generated (or reference) video.
            vlm_client: VLM client supporting ``query_text`` and
                ``query_with_frames``.
            config:     Configuration dict.  Recognised keys:
                * ``trials_count`` (int): trials per question.
                * ``num_frames`` (int): frames to sample from video.

        Returns:
            :class:`ItemEvalResult` with ``score = max(0, effect_size)``
            normalised to [0, 1].
        """
        trials_count = config.get("trials_count", self._default_trials)
        num_frames = config.get("num_frames", 8)

        judge_id = getattr(vlm_client, "judge_id", "")

        bloom_questions = prompt.bloom_questions or []
        if not bloom_questions:
            logger.warning(
                "No bloom_questions for VLM comparison prompt %s", prompt.id
            )
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=judge_id,
                scoring_method=ScoringMethod.VLM_COMPARISON,
                score=0.0,
                raw_score=0.0,
                details={"error": "No bloom_questions defined"},
                error="No bloom_questions defined for vlm_comparison",
            )

        # Extract frames once for all questions / trials
        try:
            frame_paths = extract_frames(
                video_ref.video_path, num_frames=num_frames
            )
        except Exception as exc:
            logger.error(
                "Frame extraction failed for %s: %s", video_ref.video_path, exc
            )
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=judge_id,
                scoring_method=ScoringMethod.VLM_COMPARISON,
                score=0.0,
                raw_score=0.0,
                details={"error": f"Frame extraction failed: {exc}"},
                error=str(exc),
            )

        # Aggregate McNemar table across all questions
        total_a = 0  # both_correct
        total_b = 0  # baseline_only_correct
        total_c = 0  # video_only_correct
        total_d = 0  # both_wrong
        total_baseline_correct = 0
        total_video_correct = 0
        total_trials = 0

        question_details: List[Dict[str, Any]] = []

        for q_item in bloom_questions:
            # Support both dict-style bloom_questions and plain strings
            if isinstance(q_item, dict):
                question_text = q_item.get("question_text", "")
                expected_answer = q_item.get("expected_answer", "")
                bloom_level = q_item.get("bloom_level", "unknown")
            else:
                question_text = str(q_item)
                expected_answer = prompt.ground_truth.get("correct_answer", "")
                bloom_level = "unknown"

            if not question_text:
                continue

            q_a, q_b, q_c, q_d = 0, 0, 0, 0

            for trial in range(trials_count):
                # --- Baseline (text-only) ---
                try:
                    baseline_resp = await vlm_client.query_text(
                        system_prompt=_SYSTEM_PROMPT,
                        user_prompt=_USER_PROMPT_TEXT_ONLY.format(
                            question=question_text
                        ),
                    )
                    baseline_correct = _answers_match(
                        baseline_resp, expected_answer
                    )
                except Exception as exc:
                    logger.debug(
                        "Baseline trial %d failed for %s: %s",
                        trial,
                        prompt.id,
                        exc,
                    )
                    baseline_correct = False

                # --- With video ---
                try:
                    video_resp = await vlm_client.query_with_video(
                        system_prompt=_SYSTEM_PROMPT,
                        user_prompt=_USER_PROMPT_WITH_VIDEO.format(
                            question=question_text
                        ),
                        video_path=video_ref.video_path,
                        video_url=config.get("video_url"),
                        frame_paths=frame_paths,
                        num_frames=num_frames,
                    )
                    video_correct = _answers_match(video_resp, expected_answer)
                except Exception as exc:
                    logger.debug(
                        "Video trial %d failed for %s: %s",
                        trial,
                        prompt.id,
                        exc,
                    )
                    video_correct = False

                # Classify into contingency cell
                if baseline_correct and video_correct:
                    q_a += 1
                elif baseline_correct and not video_correct:
                    q_b += 1
                elif not baseline_correct and video_correct:
                    q_c += 1
                else:
                    q_d += 1

            q_trials = trials_count
            q_baseline_acc = (q_a + q_b) / q_trials if q_trials > 0 else 0.0
            q_video_acc = (q_a + q_c) / q_trials if q_trials > 0 else 0.0

            question_details.append({
                "bloom_level": bloom_level,
                "question": question_text[:200],
                "trials": q_trials,
                "contingency": {
                    "both_correct": q_a,
                    "baseline_only": q_b,
                    "video_only": q_c,
                    "both_wrong": q_d,
                },
                "baseline_accuracy": round(q_baseline_acc, 4),
                "video_accuracy": round(q_video_acc, 4),
                "effect_size": round(q_video_acc - q_baseline_acc, 4),
            })

            total_a += q_a
            total_b += q_b
            total_c += q_c
            total_d += q_d
            total_baseline_correct += q_a + q_b
            total_video_correct += q_a + q_c
            total_trials += q_trials

        # --- Aggregate statistics ---
        if total_trials == 0:
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=judge_id,
                scoring_method=ScoringMethod.VLM_COMPARISON,
                score=0.0,
                raw_score=0.0,
                details={"error": "No valid trials completed"},
                error="No valid trials completed",
            )

        baseline_accuracy = total_baseline_correct / total_trials
        video_accuracy = total_video_correct / total_trials
        effect_size = video_accuracy - baseline_accuracy

        # McNemar's test (with continuity correction)
        b_plus_c = total_b + total_c
        if b_plus_c > 0:
            chi2 = ((abs(total_b - total_c) - 1) ** 2) / b_plus_c
            p_value = _chi2_sf_1df(chi2)
        else:
            chi2 = 0.0
            p_value = 1.0

        # Final score: clamp effect size to [0, 1]
        final_score = max(0.0, min(1.0, effect_size))

        logger.info(
            "VLMComparison [%s]: baseline=%.3f, video=%.3f, effect=%.3f, "
            "chi2=%.3f, p=%.4f, score=%.3f",
            prompt.id,
            baseline_accuracy,
            video_accuracy,
            effect_size,
            chi2,
            p_value,
            final_score,
        )

        return ItemEvalResult(
            prompt_id=prompt.id,
            model_id=video_ref.model_id,
            judge_id=judge_id,
            scoring_method=ScoringMethod.VLM_COMPARISON,
            score=final_score,
            raw_score=round(effect_size, 4),
            details={
                "baseline_accuracy": round(baseline_accuracy, 4),
                "video_accuracy": round(video_accuracy, 4),
                "effect_size": round(effect_size, 4),
                "total_trials": total_trials,
                "contingency": {
                    "both_correct": total_a,
                    "baseline_only": total_b,
                    "video_only": total_c,
                    "both_wrong": total_d,
                },
                "mcnemar_chi2": round(chi2, 4),
                "mcnemar_p_value": round(p_value, 6),
                "significant_at_005": p_value < 0.05,
                "num_questions": len(question_details),
                "question_details": question_details,
                "num_frames": num_frames,
            },
        )
