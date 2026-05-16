"""Reporting and persistence for EduVBench evaluation results.

Handles serialisation of evaluation results and scorecards to JSON and
Markdown, organised in timestamped run directories for reproducibility.
"""

import json
import logging
import os
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    CategoryScore,
    DimensionScore,
    ItemEvalResult,
    ModelScoreCard,
    ScoringMethod,
)

logger = logging.getLogger(__name__)


class Reporter:
    """Persists evaluation results and generates comparative reports.

    Creates a timestamped run directory under ``results_dir`` and provides
    methods to save per-model results, score cards, and multi-model
    comparative reports in both JSON and Markdown formats.

    Args:
        results_dir: Root directory for evaluation results. A timestamped
            subdirectory will be created for each run.
    """

    def __init__(self, results_dir: str) -> None:
        self._results_dir = Path(results_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._results_dir / f"run_{timestamp}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Reporter initialised, run directory: %s", self._run_dir)

    @property
    def run_dir(self) -> Path:
        """Return the path to the current run directory."""
        return self._run_dir

    # ------------------------------------------------------------------
    # Per-model results
    # ------------------------------------------------------------------

    def save_model_results(
        self,
        model_id: str,
        results: List[ItemEvalResult],
        score_card: ModelScoreCard,
    ) -> None:
        """Save evaluation results and scorecard for a single model.

        Creates a model-specific subdirectory containing:
            - ``eval_results.json``: All per-item evaluation results.
            - ``score_card.json``: The aggregated model scorecard.

        Args:
            model_id: The evaluated model's identifier.
            results: List of per-item evaluation results.
            score_card: The aggregated scorecard for the model.
        """
        model_dir = self._run_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results
        results_path = model_dir / "eval_results.json"
        results_data = [self._to_dict(r) for r in results]
        self._write_json(results_path, {
            "model_id": model_id,
            "item_count": len(results),
            "results": results_data,
        })
        logger.info(
            "Saved %d eval results for model %s to %s",
            len(results), model_id, results_path,
        )

        # Save score card
        card_path = model_dir / "score_card.json"
        self._write_json(card_path, self._to_dict(score_card))
        logger.info(
            "Saved score card for model %s to %s", model_id, card_path
        )

    # ------------------------------------------------------------------
    # Reliability reports
    # ------------------------------------------------------------------

    def save_reliability_report(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        markdown_report: str,
    ) -> None:
        """Save inter-rater reliability metrics for a model.

        Creates:
            - ``{model_id}/reliability.json``: Raw metrics data.
            - ``{model_id}/reliability.md``: Human-readable report.

        Args:
            model_id: The evaluated model's identifier.
            metrics: Reliability metrics dict from compute_reliability_metrics.
            markdown_report: Pre-rendered Markdown report string.
        """
        model_dir = self._run_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        json_path = model_dir / "reliability.json"
        self._write_json(json_path, metrics)

        md_path = model_dir / "reliability.md"
        md_path.write_text(markdown_report, encoding="utf-8")

        logger.info(
            "Saved reliability report for model %s to %s", model_id, model_dir,
        )

    # ------------------------------------------------------------------
    # Comparative reports
    # ------------------------------------------------------------------

    def save_comparative_report(
        self, score_cards: List[ModelScoreCard]
    ) -> None:
        """Save comparative reports across all evaluated models.

        Generates three files in the run directory:
            - ``comparative_report.json``: All scorecards side by side.
            - ``leaderboard.json``: Models sorted by final score.
            - ``comparative_report.md``: Human-readable Markdown table.

        Args:
            score_cards: List of ModelScoreCard instances for all models.
        """
        if not score_cards:
            logger.warning("No score cards to compare")
            return

        # Sort by final score descending
        sorted_cards = sorted(
            score_cards, key=lambda c: c.final_score, reverse=True
        )

        # comparative_report.json
        report_path = self._run_dir / "comparative_report.json"
        self._write_json(report_path, {
            "run_dir": str(self._run_dir),
            "model_count": len(sorted_cards),
            "generated_at": datetime.now().isoformat(),
            "score_cards": [self._to_dict(c) for c in sorted_cards],
        })
        logger.info("Saved comparative report to %s", report_path)

        # leaderboard.json
        leaderboard_path = self._run_dir / "leaderboard.json"
        leaderboard_entries = []
        for rank, card in enumerate(sorted_cards, start=1):
            entry = {
                "rank": rank,
                "model_id": card.model_id,
                "model_name": card.model_name,
                "final_score": card.final_score,
                "safety_gate_passed": card.safety_gate_passed,
            }
            # Add dimension scores
            for dim_name, dim_score in card.dimension_scores.items():
                entry[f"{dim_name}_score"] = dim_score.score
            # Add A-NE safety score
            entry["a_ne_score"] = card.metadata.get(
                "category_scores", {}
            ).get("A-NE", 0.0)
            # Add dual-judge metadata
            entry["items_flagged"] = card.metadata.get("items_flagged_for_review", 0)
            reliability = card.metadata.get("reliability", {})
            if reliability.get("status") == "computed":
                entry["icc_2_1"] = reliability.get("icc", {}).get("icc_2_1")
            leaderboard_entries.append(entry)

        self._write_json(leaderboard_path, {
            "generated_at": datetime.now().isoformat(),
            "leaderboard": leaderboard_entries,
        })
        logger.info("Saved leaderboard to %s", leaderboard_path)

        # comparative_report.md
        md_path = self._run_dir / "comparative_report.md"
        md_content = self._generate_markdown_report(sorted_cards)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Saved Markdown report to %s", md_path)

    # ------------------------------------------------------------------
    # Markdown generation
    # ------------------------------------------------------------------

    def _generate_markdown_report(
        self, score_cards: List[ModelScoreCard]
    ) -> str:
        """Generate a Markdown comparative report with leaderboard table.

        Args:
            score_cards: Score cards sorted by final score (descending).

        Returns:
            Markdown-formatted string.
        """
        lines: List[str] = []
        lines.append("# EduVBench Evaluation Report")
        lines.append("")
        lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"Models evaluated: {len(score_cards)}")
        lines.append("")

        # Leaderboard table
        lines.append("## Leaderboard")
        lines.append("")
        lines.append(
            "| Rank | Model | Final Score | Knowledge | Skills | "
            "Attitude | Safety (A-NE) |"
        )
        lines.append(
            "|------|-------|-------------|-----------|--------|"
            "----------|---------------|"
        )

        for rank, card in enumerate(score_cards, start=1):
            k_score = self._get_dimension_score(card, "knowledge")
            s_score = self._get_dimension_score(card, "skills")
            a_score = self._get_dimension_score(card, "attitude")
            a_ne = card.metadata.get("category_scores", {}).get("A-NE", 0.0)

            lines.append(
                f"| {rank} | {card.model_name} | "
                f"{card.final_score:.4f} | "
                f"{k_score:.4f} | "
                f"{s_score:.4f} | "
                f"{a_score:.4f} | "
                f"{a_ne:.4f} |"
            )

        lines.append("")

        # Per-model detail sections
        lines.append("## Model Details")
        lines.append("")

        for card in score_cards:
            lines.append(f"### {card.model_name} (`{card.model_id}`)")
            lines.append("")
            lines.append(f"- **Final Score**: {card.final_score:.4f}")
            lines.append(
                f"- **Safety Gate**: "
                f"{'PASS' if card.safety_gate_passed else 'FAIL'}"
            )
            lines.append("")

            # Reliability summary (if available)
            reliability = card.metadata.get("reliability", {})
            if reliability.get("status") == "computed":
                icc_val = reliability.get("icc", {}).get("icc_2_1", "-")
                pearson_val = reliability.get("pearson", {}).get("r", "-")
                kappa_val = reliability.get("cohens_kappa", {}).get("kappa_weighted", "-")
                lines.append(
                    f"- **Inter-Rater Reliability**: "
                    f"ICC={icc_val}, Pearson r={pearson_val}, "
                    f"kappa={kappa_val}"
                )
                lines.append("")

            # Category breakdown
            if card.dimension_scores:
                lines.append("| Dimension | Category | Score | Items |")
                lines.append("|-----------|----------|-------|-------|")

                for dim_name in ["knowledge", "skills", "attitude"]:
                    dim = card.dimension_scores.get(dim_name)
                    if dim is None:
                        continue
                    for cat_code, cat_score in sorted(
                        dim.category_scores.items()
                    ):
                        lines.append(
                            f"| {dim_name.capitalize()} | {cat_code} | "
                            f"{cat_score.score:.4f} | "
                            f"{cat_score.item_count} |"
                        )
                lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            "*Report generated by EduVBench Evaluation Engine*"
        )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _get_dimension_score(card: ModelScoreCard, dimension: str) -> float:
        """Extract a dimension score from a score card, defaulting to 0.0.

        Args:
            card: The model score card.
            dimension: Dimension name (knowledge, skills, attitude).

        Returns:
            The dimension score, or 0.0 if not present.
        """
        dim = card.dimension_scores.get(dimension)
        return dim.score if dim else 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _to_dict(self, obj: Any) -> Any:
        """Recursively convert a dataclass (or nested structure) to a dict.

        Handles special types:
            - ``datetime`` -> ISO 8601 string
            - ``Enum`` -> value
            - ``dataclass`` -> dict (recursive)
            - ``list`` / ``dict`` -> recursively processed

        Args:
            obj: The object to serialise.

        Returns:
            A JSON-serialisable Python object.
        """
        if obj is None:
            return None

        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, Path):
            return str(obj)

        if is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = self._to_dict(value)
            return result

        if isinstance(obj, dict):
            return {
                self._to_dict(k): self._to_dict(v)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [self._to_dict(item) for item in obj]

        if isinstance(obj, (int, float, str, bool)):
            return obj

        # Fallback: try str()
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write data to a JSON file with consistent formatting.

        Args:
            path: Destination file path.
            data: JSON-serialisable data to write.
        """
        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError) as exc:
            logger.error("Failed to write JSON to %s: %s", path, exc)
            raise
