#!/usr/bin/env python3
"""Re-aggregate existing eval results with updated scoring config.

Reads eval_results.json from each model's results directory and re-computes
scorecards using the current aggregator logic and scoring_config.json.
This avoids re-running expensive VLM calls.

Usage:
    cd <project_root>
    python3 -m video_gen_eval.scripts.reaggregate
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_gen_eval.eval.aggregator import ScoreAggregator
from video_gen_eval.eval.models import ItemEvalResult, ModelScoreCard, ScoringMethod

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "veo31": "Veo 3.1",
    "sora2": "Sora 2",
    "kling3": "Kling 3.0",
    "wan22": "Wan 2.2",
    "wan26": "Wan 2.6",
}

RESULTS_BASE = Path("results")
DATASET_DIR = Path("eduvbench-dataset")
OUTPUT_DIR = Path("results/merged")


def load_scoring_config() -> Dict[str, Any]:
    path = DATASET_DIR / "scoring_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dict_to_item_eval_result(d: Dict[str, Any]) -> ItemEvalResult:
    """Convert a dict from eval_results.json back to ItemEvalResult."""
    method_str = d.get("scoring_method", "rubric_10pt")
    try:
        method = ScoringMethod(method_str)
    except ValueError:
        method = ScoringMethod.RUBRIC_10PT

    return ItemEvalResult(
        prompt_id=d.get("prompt_id", ""),
        model_id=d.get("model_id", ""),
        judge_id=d.get("judge_id", "merged"),
        scoring_method=method,
        score=d.get("score", 0.0),
        raw_score=d.get("raw_score"),
        reasoning=d.get("reasoning", ""),
        key_observations=d.get("key_observations", []),
        rubric_alignment=d.get("rubric_alignment", {}),
        details=d.get("details", {}),
        vlm_response=d.get("vlm_response"),
        error=d.get("error"),
        timestamp=d.get("timestamp"),
    )


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
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
            result[f.name] = to_dict(value)
        return result
    if isinstance(obj, dict):
        return {to_dict(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_dim_score(card_dict: Dict, dim: str) -> float:
    d = card_dict.get("dimension_scores", {}).get(dim)
    if d is None:
        return 0.0
    if isinstance(d, dict):
        return d.get("score", 0.0)
    return 0.0


def main():
    scoring_config = load_scoring_config()
    aggregator = ScoreAggregator(scoring_config)

    score_cards: List[ModelScoreCard] = []

    for model_id, model_name in MODELS.items():
        model_dir = RESULTS_BASE / model_id
        er_path = None
        for p in sorted(model_dir.rglob("eval_results.json")):
            er_path = p
            break

        if er_path is None:
            logger.warning("No eval_results.json for %s", model_id)
            continue

        with open(er_path, "r", encoding="utf-8") as f:
            er_data = json.load(f)

        raw_results = er_data.get("results", [])
        results = [dict_to_item_eval_result(r) for r in raw_results]

        logger.info("Loaded %d results for %s", len(results), model_id)

        card = aggregator.aggregate(model_id, model_name, results)
        score_cards.append(card)

    # Sort by final score
    sorted_cards = sorted(score_cards, key=lambda c: c.final_score, reverse=True)

    # Convert to dicts for serialization
    card_dicts = [to_dict(c) for c in sorted_cards]

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # comparative_report.json
    write_json(OUTPUT_DIR / "comparative_report.json", {
        "run_dir": str(OUTPUT_DIR),
        "model_count": len(card_dicts),
        "generated_at": datetime.now().isoformat(),
        "score_cards": card_dicts,
    })

    # leaderboard.json
    leaderboard = []
    for rank, cd in enumerate(card_dicts, start=1):
        entry = {
            "rank": rank,
            "model_id": cd.get("model_id"),
            "model_name": cd.get("model_name"),
            "final_score": cd.get("final_score", 0),
            "safety_gate_passed": cd.get("safety_gate_passed", False),
            "a_ne_block_rate": cd.get("metadata", {}).get("category_scores", {}).get("A-NE", 0),
        }
        for dim in ["knowledge", "skills", "attitude"]:
            entry[f"{dim}_score"] = get_dim_score(cd, dim)
        meta = cd.get("metadata", {})
        entry["items_flagged"] = meta.get("items_flagged_for_review", 0)
        leaderboard.append(entry)

    write_json(OUTPUT_DIR / "leaderboard.json", {
        "generated_at": datetime.now().isoformat(),
        "leaderboard": leaderboard,
    })

    # Per-model score cards
    for cd in card_dicts:
        mid = cd.get("model_id", "unknown")
        write_json(OUTPUT_DIR / mid / "score_card.json", cd)

    # Markdown report
    lines = ["# EduVBench Evaluation Report", ""]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Models evaluated: {len(card_dicts)}")
    lines.append("")
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

    for entry in leaderboard:
        lines.append(
            f"| {entry['rank']} | {entry['model_name']} | "
            f"{entry['final_score']:.4f} | "
            f"{entry['knowledge_score']:.4f} | "
            f"{entry['skills_score']:.4f} | "
            f"{entry['attitude_score']:.4f} | "
            f"{'PASS' if entry['safety_gate_passed'] else 'FAIL'} |"
        )

    lines.append("")
    lines.append("## Model Details")
    lines.append("")

    for cd in card_dicts:
        name = cd.get("model_name", cd.get("model_id"))
        mid = cd.get("model_id")
        lines.append(f"### {name} (`{mid}`)")
        lines.append("")
        lines.append(f"- **Final Score**: {cd.get('final_score', 0):.4f}")
        safety = "PASS" if cd.get("safety_gate_passed") else "FAIL"
        lines.append(f"- **Safety Gate**: {safety}")
        lines.append("")

        dims = cd.get("dimension_scores", {})
        if dims:
            lines.append("| Dimension | Category | Score | Items |")
            lines.append("|-----------|----------|-------|-------|")
            for dim_name in ["knowledge", "skills", "attitude"]:
                dim = dims.get(dim_name)
                if not dim:
                    continue
                cats = dim.get("category_scores", {})
                for cat_code in sorted(cats.keys()):
                    cat = cats[cat_code]
                    if isinstance(cat, dict):
                        lines.append(
                            f"| {dim_name.capitalize()} | {cat_code} | "
                            f"{cat.get('score', 0):.4f} | "
                            f"{cat.get('item_count', 0)} |"
                        )
            lines.append("")

    lines.append("---")
    lines.append("*Report generated by EduVBench Evaluation Engine*")

    (OUTPUT_DIR / "comparative_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Print summary
    print()
    print("=" * 70)
    print("  EduVBench v1 — Final Scores (Re-aggregated)")
    print("=" * 70)
    print()
    print(f"  {'Rank':<5} {'Model':<15} {'Final':>7} {'K':>7} {'S':>7} {'A':>7} {'Safety':>8}")
    print(f"  {'-'*5} {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for entry in leaderboard:
        safety = "PASS" if entry["safety_gate_passed"] else "FAIL"
        print(
            f"  {entry['rank']:<5} {entry['model_name']:<15} "
            f"{entry['final_score']:>7.4f} "
            f"{entry['knowledge_score']:>7.4f} "
            f"{entry['skills_score']:>7.4f} "
            f"{entry['attitude_score']:>7.4f} "
            f"{safety:>8}"
        )
    print()
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
