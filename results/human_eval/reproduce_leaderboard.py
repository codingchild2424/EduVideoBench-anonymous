"""Reproduce the Human-Center rows of leaderboard_ksa.csv from the released files.

Inputs
  results/human_eval/human_eval_215_scores.csv  expert scores per (prompt, model)
  results/merged_v4/<model>/eval_results.json   algorithmic items (K-PK auto-metric,
                                                S-VIU comparison, A-NE block test,
                                                A-DD consistency) and item metadata

Rules
  - The expert mean of the available raters is the item score; rubric scores are
    normalized as (s - 1) / 4 and exact-match scores are kept on their 0 / 0.5 / 1 scale.
  - Cells with no expert score (model refusals of benign prompts) score 0.
  - Algorithmic items keep their pipeline score.
  - Aggregation is the official ScoreAggregator with eduvbench-dataset/scoring_config.json.

Usage (from the repository root)
  python results/human_eval/reproduce_leaderboard.py
"""
import csv, copy, dataclasses, json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "video_gen_eval"))
logging.disable(logging.WARNING)

from eval.aggregator import ScoreAggregator  # noqa: E402
from eval.models import ItemEvalResult, ScoringMethod  # noqa: E402

MODELS = {"veo31": "Veo 3.1", "sora2": "Sora 2", "kling3": "Kling 3.0", "wan22": "Wan 2.2", "wan26": "Wan 2.6"}
ALGORITHMIC = {"auto_metric", "vlm_comparison", "block_test", "consistency"}
CATEGORIES = ["K-CK", "K-PK", "S-PF", "S-UC", "S-VIU", "A-ES", "A-IS", "A-NE", "A-DD"]


def main():
    config = json.loads((ROOT / "eduvbench-dataset" / "scoring_config.json").read_text())
    aggregator = ScoreAggregator(config)
    fields = {f.name for f in dataclasses.fields(ItemEvalResult)}
    with open(ROOT / "results" / "human_eval" / "human_eval_215_scores.csv") as fh:
        human = {row["prompt_id"]: row for row in csv.DictReader(fh)}
    published = {}
    with open(ROOT / "results" / "human_eval" / "leaderboard_ksa.csv") as fh:
        for row in csv.DictReader(fh):
            if row["signal"] == "Human-Center":
                published[row["model"]] = row

    mismatches = 0
    for model_id, name in MODELS.items():
        items = json.loads((ROOT / "results" / "merged_v4" / model_id / "eval_results.json").read_text())["results"]
        objs = []
        for item in items:
            item = copy.deepcopy(item)
            if item["scoring_method"] not in ALGORITHMIC:
                row = human[item["prompt_id"]]
                mean = row[f"{model_id}_mean"]
                if mean == "":
                    score = 0.0
                elif row["scoring_method"] == "exact_match":
                    score = float(mean)
                else:
                    score = (float(mean) - 1) / 4
                item["score"] = min(1.0, max(0.0, score))
            item["scoring_method"] = ScoringMethod(item["scoring_method"])
            objs.append(ItemEvalResult(**{k: v for k, v in item.items() if k in fields}))
        card = aggregator.aggregate(model_id, name, objs)
        cats = card.metadata["category_scores"]
        dims = {key: card.dimension_scores[dim].score for key, dim in (("K_total", "knowledge"), ("S_total", "skills"), ("A_total", "attitude"))}
        values = {c.replace("-", "_"): cats[c] for c in CATEGORIES} | dims | {"KSA": card.final_score, "block_rate": cats["A-NE"]}
        row_bad = [f"{k} {v:.4f} != {published[name][k]}" for k, v in values.items() if f"{v:.4f}" != published[name][k]]
        mismatches += len(row_bad)
        print(f"{name:10s} KSA {card.final_score:.4f}  {'OK' if not row_bad else row_bad}")
    print("all Human-Center values reproduced" if mismatches == 0 else f"{mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
