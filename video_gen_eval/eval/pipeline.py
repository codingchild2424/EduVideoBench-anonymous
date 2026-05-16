"""Main evaluation orchestrator for the EduVBench pipeline.

Coordinates the end-to-end evaluation flow with dual VLM judge support:
loading the dataset and configuration, discovering generated videos,
dispatching scoring tasks to each judge, merging dual-judge scores,
aggregating results into model scorecards, and persisting reports.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregator import ScoreAggregator
from .config import EvalConfig, JudgeConfig
from .models import (
    ItemEvalResult,
    ModelScoreCard,
    Prompt,
    ScoringMethod,
    VideoRef,
)
from .reliability import compute_reliability_metrics, generate_reliability_report
from .reporter import Reporter
from .vlm_client import VLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loader helpers
# ---------------------------------------------------------------------------

def load_prompts(dataset_dir: str) -> List[Prompt]:
    """Load all prompts from the EduVBench dataset JSON files.

    Scans for ``*_prompts.json`` files in the dataset directory and
    parses each prompt entry into a :class:`Prompt` dataclass.

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        List of all parsed Prompt objects.
    """
    prompts: List[Prompt] = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        logger.error("Dataset directory not found: %s", dataset_dir)
        return prompts

    prompt_files = sorted(dataset_path.glob("*_prompts.json"))
    if not prompt_files:
        logger.warning("No prompt files found in %s", dataset_dir)
        return prompts

    for pf in prompt_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            for item in data.get("prompts", []):
                prompt = _parse_prompt(item)
                if prompt is not None:
                    prompts.append(prompt)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load prompt file %s: %s", pf, exc)

    logger.info(
        "Loaded %d prompts from %d files in %s",
        len(prompts), len(prompt_files), dataset_dir,
    )
    return prompts


def load_scoring_config(dataset_dir: str) -> dict:
    """Load the scoring configuration from scoring_config.json."""
    config_path = Path(dataset_dir) / "scoring_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Scoring config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_rubrics_config(dataset_dir: str) -> dict:
    """Load the rubrics configuration from rubrics.json."""
    config_path = Path(dataset_dir) / "rubrics.json"
    if not config_path.exists():
        logger.warning("rubrics.json not found, using empty rubrics config")
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _parse_prompt(data: dict) -> Optional[Prompt]:
    """Parse a single prompt dictionary into a Prompt dataclass."""
    try:
        scoring_method = ScoringMethod(data["scoring_method"])
    except (KeyError, ValueError) as exc:
        logger.warning(
            "Invalid scoring_method in prompt %s: %s",
            data.get("id", "UNKNOWN"), exc,
        )
        return None

    try:
        return Prompt(
            id=data["id"],
            subject=data.get("subject", ""),
            grade_level=data.get("grade_level", ""),
            category=data.get("category", ""),
            subcategory=data.get("subcategory", ""),
            prompt_text=data.get("prompt_text", ""),
            ground_truth=data.get("ground_truth", {}),
            rubric_criteria=data.get("rubric_criteria", ""),
            scoring_method=scoring_method,
            metadata=data.get("metadata", {}),
            bloom_questions=data.get("bloom_questions"),
            reference_video_id=data.get("reference_video_id"),
        )
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse prompt %s: %s", data.get("id"), exc)
        return None


# ---------------------------------------------------------------------------
# Score merging for dual-judge results
# ---------------------------------------------------------------------------

def merge_dual_judge_scores(
    results_by_judge: Dict[str, ItemEvalResult],
    method: str = "mean_with_flag",
    flag_threshold: float = 2.0,
) -> ItemEvalResult:
    """Merge scores from two judges into a single result.

    Args:
        results_by_judge: Dict mapping judge_id to ItemEvalResult.
        method: Merging method (currently only ``mean_with_flag``).
        flag_threshold: Max score difference (on 10pt scale) before flagging.

    Returns:
        A merged ItemEvalResult with averaged scores and combined metadata.
    """
    judges = list(results_by_judge.keys())
    results = list(results_by_judge.values())

    if len(results) == 1:
        return results[0]

    if len(results) == 0:
        raise ValueError("No results to merge")

    # Use the first result as the template
    base = results[0]

    # Compute merged score (average of normalised scores)
    scores = [r.score for r in results]
    merged_score = sum(scores) / len(scores)

    # Check for raw score diff (on 10-point scale for rubric)
    raw_scores = []
    for r in results:
        if isinstance(r.raw_score, (int, float)):
            raw_scores.append(float(r.raw_score))

    flagged = False
    raw_diff = None
    if len(raw_scores) >= 2:
        raw_diff = abs(raw_scores[0] - raw_scores[1])
        if raw_diff > flag_threshold:
            flagged = True

    # Combine reasoning from all judges
    reasonings = {r.judge_id: r.reasoning for r in results if r.reasoning}
    all_observations = []
    for r in results:
        for obs in r.key_observations:
            if obs not in all_observations:
                all_observations.append(obs)

    # Build per-judge detail breakdown
    judge_details = {}
    for r in results:
        judge_details[r.judge_id] = {
            "score": r.score,
            "raw_score": r.raw_score,
            "reasoning": r.reasoning,
            "key_observations": r.key_observations,
            "rubric_alignment": r.rubric_alignment,
        }

    merged_details = dict(base.details)
    merged_details["judge_scores"] = judge_details
    merged_details["merge_method"] = method
    merged_details["flagged_for_review"] = flagged
    if raw_diff is not None:
        merged_details["raw_score_diff"] = round(raw_diff, 2)

    return ItemEvalResult(
        prompt_id=base.prompt_id,
        model_id=base.model_id,
        judge_id="merged",
        scoring_method=base.scoring_method,
        score=max(0.0, min(1.0, merged_score)),
        raw_score=round(sum(raw_scores) / len(raw_scores), 2) if raw_scores else base.raw_score,
        reasoning="; ".join(f"[{jid}] {r}" for jid, r in reasonings.items()),
        key_observations=all_observations,
        rubric_alignment=base.rubric_alignment,
        details=merged_details,
        vlm_response=None,
        error=base.error,
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class EvalPipeline:
    """Orchestrates the full EduVBench evaluation pipeline with dual judges.

    Lifecycle:
        1. Load dataset and scoring configuration.
        2. Initialise VLM clients for each judge, scorers, and aggregator.
        3. For each model, discover videos, score with each judge, merge, aggregate.
        4. Generate comparative reports.

    Args:
        config: An EvalConfig instance describing the evaluation run.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

        # Load dataset
        self._prompts = load_prompts(config.dataset_dir)
        self._scoring_config = load_scoring_config(config.dataset_dir)
        self._rubrics_config = load_rubrics_config(config.dataset_dir)

        # Build prompt lookup by ID
        self._prompt_map: Dict[str, Prompt] = {p.id: p for p in self._prompts}

        # VLM clients keyed by judge_id (lazy initialization)
        self._vlm_clients: Dict[str, VLMClient] = {}

        # Initialise scorers
        self._init_scorers()

        # Aggregator
        self._aggregator = ScoreAggregator(self._scoring_config)

        # Reporter
        self._reporter = Reporter(config.results_dir)

        # Concurrency control
        self._semaphore = asyncio.Semaphore(5)

        logger.info(
            "EvalPipeline initialised: %d prompts, %d models, %d judges, "
            "dry_run=%s",
            len(self._prompts),
            len(config.models),
            len(config.judges),
            config.dry_run,
        )

    # ------------------------------------------------------------------
    # Scorer initialisation
    # ------------------------------------------------------------------

    def _init_scorers(self) -> None:
        """Import and instantiate scorer classes."""
        from .scorers.exact_match import ExactMatchScorer
        from .scorers.rubric_10pt import Rubric10ptScorer
        from .scorers.block_test import BlockTestScorer
        from .scorers.vlm_comparison import VLMComparisonScorer
        from .scorers.consistency import ConsistencyScorer
        from .scorers.auto_metrics import AutoMetricScorer

        self._exact_match_scorer = ExactMatchScorer()
        self._rubric_scorer = Rubric10ptScorer(self._rubrics_config)
        self._block_test_scorer = BlockTestScorer()
        self._vlm_comparison_scorer = VLMComparisonScorer()
        self._consistency_scorer = ConsistencyScorer(self._rubrics_config)
        self._auto_metric_scorer = AutoMetricScorer(self._scoring_config)

        self._scorer_map = {
            ScoringMethod.EXACT_MATCH: self._exact_match_scorer,
            ScoringMethod.RUBRIC_10PT: self._rubric_scorer,
            ScoringMethod.BLOCK_TEST: self._block_test_scorer,
            ScoringMethod.VLM_COMPARISON: self._vlm_comparison_scorer,
            ScoringMethod.AUTO_METRIC: self._auto_metric_scorer,
            # CONSISTENCY is handled separately via score_group()
        }

    def _get_vlm_client(self, judge_config: JudgeConfig) -> VLMClient:
        """Lazily initialise and return a VLM client for a judge.

        Args:
            judge_config: Judge configuration.

        Returns:
            The VLMClient instance for this judge.
        """
        if judge_config.id not in self._vlm_clients:
            cache_dir = os.path.join(
                self.config.cache_dir, judge_config.id
            )
            self._vlm_clients[judge_config.id] = VLMClient.from_judge_config(
                judge_config, cache_dir=cache_dir,
            )
        return self._vlm_clients[judge_config.id]

    # ------------------------------------------------------------------
    # Video discovery
    # ------------------------------------------------------------------

    def discover_videos(self, model_id: str) -> Dict[str, VideoRef]:
        """Discover generated videos for a model and map them to prompt IDs.

        Scans ``{videos_dir}/{model_id}/`` recursively for metadata JSON
        files which contain prompt_id, video_file path, and video_url.
        Also handles failure stubs (success=false, generation_refused=true).

        Args:
            model_id: The model identifier to scan for.

        Returns:
            Dictionary mapping prompt_id to VideoRef.
        """
        model_dir = Path(self.config.videos_dir) / model_id
        if not model_dir.is_dir():
            logger.warning(
                "Video directory not found for model %s: %s",
                model_id, model_dir,
            )
            return {}

        videos: Dict[str, VideoRef] = {}

        # Build reverse lookup from prompt_text to prompt_id
        text_to_id: Dict[str, str] = {
            p.prompt_text.strip().lower(): p.id for p in self._prompts
        }

        # Strategy: scan metadata JSON files (not video files)
        for json_file in sorted(model_dir.rglob("*.json")):
            if json_file.name == "run_summary.json":
                continue

            try:
                meta = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read metadata %s: %s", json_file, exc)
                continue

            # Extract prompt_id
            prompt_id = meta.get("prompt_id") or meta.get("title")
            if not prompt_id:
                prompt_text = (
                    meta.get("prompt_text") or meta.get("prompt", "")
                ).strip().lower()
                if prompt_text:
                    prompt_id = text_to_id.get(prompt_text)

            if not prompt_id:
                continue

            # Resolve video file path
            video_file_name = meta.get("video_file")
            video_path = None
            if video_file_name:
                # Try relative to json_file parent, then in videos/ subdir
                candidate = json_file.parent / video_file_name
                if candidate.exists():
                    video_path = str(candidate)
                else:
                    candidate = json_file.parent / "videos" / video_file_name
                    if candidate.exists():
                        video_path = str(candidate)

            # For failure stubs (generation_refused), video_path may be None
            is_refused = meta.get("generation_refused", False) or not meta.get("success", True)

            video_url = meta.get("video_url")

            videos[prompt_id] = VideoRef(
                video_path=video_path or "",
                model_id=model_id,
                prompt_id=prompt_id,
                metadata_path=str(json_file),
                video_url=video_url,
            )

            if is_refused:
                videos[prompt_id].metadata = {"generation_refused": True}

        logger.info(
            "Discovered %d videos for model %s in %s",
            len(videos), model_id, model_dir,
        )
        return videos

    @staticmethod
    def _extract_prompt_id_from_path(video_path: Path) -> Optional[str]:
        """Attempt to extract a prompt ID from the video file path."""
        import re
        pattern = r"(EVB-[A-Za-z]+-[A-Za-z]+-[A-Z]-[A-Z]{2}(?:-[A-Z0-9]+)*-\d{3})"
        match = re.search(pattern, video_path.stem)
        if match:
            return match.group(1)
        match = re.search(pattern, video_path.parent.name)
        if match:
            return match.group(1)
        return None

    # ------------------------------------------------------------------
    # Prompt filtering
    # ------------------------------------------------------------------

    def _filter_prompts(self) -> List[Prompt]:
        """Apply dimension, category, and prompt_id filters."""
        prompts = list(self._prompts)

        dimension_prefixes = {
            "knowledge": "K-",
            "skills": "S-",
            "attitude": "A-",
        }

        if self.config.dimensions:
            allowed_prefixes = [
                dimension_prefixes[d.lower()]
                for d in self.config.dimensions
                if d.lower() in dimension_prefixes
            ]
            if allowed_prefixes:
                prompts = [
                    p for p in prompts
                    if any(p.category.startswith(pf) for pf in allowed_prefixes)
                ]

        if self.config.categories:
            allowed_cats = set(self.config.categories)
            prompts = [p for p in prompts if p.category in allowed_cats]

        if self.config.prompt_ids:
            allowed_ids = set(self.config.prompt_ids)
            prompts = [p for p in prompts if p.id in allowed_ids]

        logger.info(
            "Filtered prompts: %d / %d active",
            len(prompts), len(self._prompts),
        )
        return prompts

    # ------------------------------------------------------------------
    # Scoring (single judge)
    # ------------------------------------------------------------------

    async def _score_item(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: VLMClient,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Score a single prompt--video pair with one judge.

        Args:
            prompt: The evaluation prompt.
            video_ref: Reference to the generated video.
            vlm_client: The VLM judge client.
            config: Scorer configuration dict.

        Returns:
            An ItemEvalResult.
        """
        scorer = self._scorer_map.get(prompt.scoring_method)
        if scorer is None:
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=vlm_client.judge_id,
                scoring_method=prompt.scoring_method,
                score=0.0,
                error=f"No scorer for {prompt.scoring_method.value}",
            )

        try:
            # Per-item config with video_url from VideoRef (for Gemini native input)
            item_config = {**config}
            if video_ref.video_url:
                item_config["video_url"] = video_ref.video_url

            async with self._semaphore:
                result = await scorer.score(prompt, video_ref, vlm_client, item_config)

            if "category" not in result.details:
                result.details["category"] = prompt.category
            if "subcategory" not in result.details:
                result.details["subcategory"] = prompt.subcategory

            return result

        except Exception as exc:
            logger.error(
                "Scorer failed for %s [%s]: %s",
                prompt.id, vlm_client.judge_id, exc,
                exc_info=True,
            )
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                judge_id=vlm_client.judge_id,
                scoring_method=prompt.scoring_method,
                score=0.0,
                details={
                    "category": prompt.category,
                    "subcategory": prompt.subcategory,
                },
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Dual-judge scoring loop
    # ------------------------------------------------------------------

    async def _score_with_all_judges(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        config: Dict[str, Any],
        per_judge_collector: Optional[Dict[str, List[ItemEvalResult]]] = None,
    ) -> ItemEvalResult:
        """Score a prompt--video pair with all configured judges and merge.

        For auto_metric scoring, judges are not used (runs once).
        For other methods, each judge scores independently, then results
        are merged.

        Args:
            prompt: Evaluation prompt.
            video_ref: Reference to the generated video.
            config: Scorer configuration dict.
            per_judge_collector: If provided, per-judge results are appended
                here (keyed by judge_id) for reliability analysis.

        Returns:
            Merged ItemEvalResult from all judges.
        """
        # Auto metrics don't need VLM judges
        if prompt.scoring_method == ScoringMethod.AUTO_METRIC:
            scorer = self._scorer_map[ScoringMethod.AUTO_METRIC]
            return await scorer.score(prompt, video_ref, None, config)

        # Score with each judge
        results_by_judge: Dict[str, ItemEvalResult] = {}

        for judge_config in self.config.judges:
            vlm_client = self._get_vlm_client(judge_config)
            result = await self._score_item(
                prompt, video_ref, vlm_client, config
            )
            results_by_judge[judge_config.id] = result

            # Collect for reliability analysis
            if per_judge_collector is not None:
                per_judge_collector[judge_config.id].append(result)

        # Merge dual-judge results
        if len(results_by_judge) >= 2:
            return merge_dual_judge_scores(
                results_by_judge,
                method=self.config.score_merge_method,
                flag_threshold=self.config.score_flag_threshold,
            )
        elif results_by_judge:
            return next(iter(results_by_judge.values()))
        else:
            return ItemEvalResult(
                prompt_id=prompt.id,
                model_id=video_ref.model_id,
                scoring_method=prompt.scoring_method,
                score=0.0,
                error="No judges configured",
            )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def run(self) -> List[ModelScoreCard]:
        """Execute the full evaluation pipeline with dual-judge scoring.

        Returns:
            List of ModelScoreCard instances, one per model.
        """
        run_start = datetime.now()
        logger.info("Starting evaluation run at %s", run_start.isoformat())
        logger.info(
            "Judges: %s",
            [j.id for j in self.config.judges],
        )

        prompts = self._filter_prompts()
        if not prompts:
            logger.error("No prompts to evaluate after filtering")
            return []

        if not self.config.models:
            videos_root = Path(self.config.videos_dir)
            if videos_root.is_dir():
                self.config.models = sorted([
                    d.name for d in videos_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ])
            logger.info(
                "Auto-discovered %d models: %s",
                len(self.config.models), self.config.models,
            )

        if not self.config.models:
            logger.error("No models to evaluate")
            return []

        model_names = self._load_model_names()
        score_cards: List[ModelScoreCard] = []

        for model_id in self.config.models:
            logger.info("=" * 60)
            logger.info("Evaluating model: %s", model_id)
            logger.info("=" * 60)

            video_map = self.discover_videos(model_id)
            if not video_map and not self.config.dry_run:
                logger.warning("No videos found for model %s", model_id)
                continue

            regular_prompts = [
                p for p in prompts
                if p.scoring_method != ScoringMethod.CONSISTENCY
            ]
            consistency_prompts = [
                p for p in prompts
                if p.scoring_method == ScoringMethod.CONSISTENCY
            ]

            if self.config.dry_run:
                self._print_dry_run_summary(
                    model_id, video_map, regular_prompts, consistency_prompts
                )
                continue

            scorer_config: Dict[str, Any] = {
                "num_frames": self.config.frames_per_video,
                "trials_count": self.config.viu_trials,
            }

            # Per-judge result collector for reliability analysis
            per_judge_results: Dict[str, List[ItemEvalResult]] = defaultdict(list)

            # Phase 1: Score regular items with dual judges
            logger.info(
                "Phase 1: Scoring %d regular items for %s",
                len(regular_prompts), model_id,
            )
            phase1_tasks = []
            for prompt in regular_prompts:
                video_ref = video_map.get(prompt.id)
                if video_ref is None:
                    continue
                phase1_tasks.append(
                    self._score_with_all_judges(
                        prompt, video_ref, scorer_config, per_judge_results,
                    )
                )

            phase1_results: List[ItemEvalResult] = []
            if phase1_tasks:
                phase1_results = list(await asyncio.gather(*phase1_tasks))

            # Phase 2: Score consistency groups with dual judges
            logger.info(
                "Phase 2: Scoring %d consistency items for %s",
                len(consistency_prompts), model_id,
            )
            phase2_results: List[ItemEvalResult] = []
            if consistency_prompts:
                for judge_config in self.config.judges:
                    vlm_client = self._get_vlm_client(judge_config)
                    group_results = await self._consistency_scorer.score_group(
                        consistency_prompts,
                        {p.id: video_map[p.id] for p in consistency_prompts if p.id in video_map},
                        vlm_client,
                        scorer_config,
                    )
                    # For consistency, results from first judge are used
                    # (consistency is about variation, not absolute score)
                    if not phase2_results:
                        phase2_results = group_results
                    break  # Use first judge for consistency

            # Combine and collect per-judge raw results
            all_results = phase1_results + phase2_results

            # Phase 3: Aggregate
            logger.info(
                "Phase 3: Aggregating %d results for %s",
                len(all_results), model_id,
            )
            model_name = model_names.get(model_id, model_id)
            score_card = self._aggregator.aggregate(
                model_id=model_id,
                model_name=model_name,
                results=all_results,
            )
            score_cards.append(score_card)

            # Phase 4: Inter-rater reliability (dual-judge)
            if len(self.config.judges) >= 2 and per_judge_results:
                logger.info(
                    "Phase 4: Computing inter-rater reliability for %s",
                    model_id,
                )
                prompt_model_map = {
                    p.id: model_id for p in prompts if p.id in video_map
                }
                reliability = compute_reliability_metrics(
                    all_judge_results=dict(per_judge_results),
                    prompt_model_map=prompt_model_map,
                )
                score_card.metadata["reliability"] = reliability

                # Save reliability report
                reliability_md = generate_reliability_report(reliability)
                self._reporter.save_reliability_report(
                    model_id, reliability, reliability_md,
                )

            # Save per-model results
            self._reporter.save_model_results(
                model_id=model_id,
                results=all_results,
                score_card=score_card,
            )

            logger.info(
                "Model %s: final=%.4f, safety=%s",
                model_id, score_card.final_score, score_card.safety_gate_passed,
            )

        if score_cards:
            self._reporter.save_comparative_report(score_cards)

        run_duration = (datetime.now() - run_start).total_seconds()
        logger.info(
            "Evaluation complete in %.1fs: %d models",
            run_duration, len(score_cards),
        )

        return score_cards

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_names(self) -> Dict[str, str]:
        """Load human-readable model names from models.json."""
        models_file = Path(self.config.dataset_dir) / "models.json"
        if not models_file.exists():
            return {}

        try:
            data = json.loads(models_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                models_list = data.get("models", [])
            elif isinstance(data, list):
                models_list = data
            else:
                return {}
            return {
                m.get("id", ""): m.get("name", "")
                for m in models_list if isinstance(m, dict)
            }
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load model names: %s", exc)
        return {}

    def _print_dry_run_summary(
        self,
        model_id: str,
        video_map: Dict[str, VideoRef],
        regular_prompts: List[Prompt],
        consistency_prompts: List[Prompt],
    ) -> None:
        """Print a summary of what would be evaluated in a dry run."""
        matched_regular = sum(1 for p in regular_prompts if p.id in video_map)
        matched_consistency = sum(1 for p in consistency_prompts if p.id in video_map)

        logger.info("[DRY RUN] Model: %s", model_id)
        logger.info("[DRY RUN]   Judges: %s", [j.id for j in self.config.judges])
        logger.info("[DRY RUN]   Videos: %d", len(video_map))
        logger.info("[DRY RUN]   Regular: %d / %d matched", matched_regular, len(regular_prompts))
        logger.info("[DRY RUN]   Consistency: %d / %d matched", matched_consistency, len(consistency_prompts))
        logger.info(
            "[DRY RUN]   Total VLM calls: ~%d",
            matched_regular * len(self.config.judges) + matched_consistency,
        )
