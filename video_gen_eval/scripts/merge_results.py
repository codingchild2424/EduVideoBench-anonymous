#!/usr/bin/env python3
"""Merge per-model evaluation results into a unified comparative report.

When models are evaluated in parallel (separate processes), each writes to
its own results directory.  This script collects the individual score cards
and generates the unified outputs:

    results/merged/
        comparative_report.json
        leaderboard.json
        comparative_report.md

Usage:
    python3 -m video_gen_eval.scripts.merge_results \
        --results-dirs results/veo31 results/sora2 results/kling3 results/wan22 results/wan26 \
        --output results/merged
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def find_score_card(results_dir: Path) -> Path | None:
    """Find score_card.json in a model's results directory."""
    for p in sorted(results_dir.rglob("score_card.json")):
        return p
    return None


def find_eval_results(results_dir: Path) -> Path | None:
    """Find eval_results.json in a model's results directory."""
    for p in sorted(results_dir.rglob("eval_results.json")):
        return p
    return None


def find_reliability(results_dir: Path) -> Path | None:
    """Find reliability.json in a model's results directory."""
    for p in sorted(results_dir.rglob("reliability.json")):
        return p
    return None


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def get_dim_score(card: Dict, dimension: str) -> float:
    dim = card.get("dimension_scores", {}).get(dimension)
    if dim is None:
        return 0.0
    if isinstance(dim, dict):
        return dim.get("score", 0.0)
    return 0.0


def generate_markdown(sorted_cards: List[Dict]) -> str:
    lines = []
    lines.append("# EduVBench Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Models evaluated: {len(sorted_cards)}")
    lines.append("")

    # Leaderboard table
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(
        "| Rank | Model | Final Score | Knowledge | Skills | "
        "Attitude | Safety Gate |"
    )
    lines.append(
        "|------|-------|-------------|-----------|--------|"
        "----------|-------------|"
    )

    for rank, card in enumerate(sorted_cards, start=1):
        k = get_dim_score(card, "knowledge")
        s = get_dim_score(card, "skills")
        a = get_dim_score(card, "attitude")
        safety = "PASS" if card.get("safety_gate_passed") else "FAIL"
        name = card.get("model_name", card.get("model_id", "?"))
        lines.append(
            f"| {rank} | {name} | "
            f"{card.get('final_score', 0):.4f} | "
            f"{k:.4f} | {s:.4f} | {a:.4f} | {safety} |"
        )

    lines.append("")

    # Per-model detail sections
    lines.append("## Model Details")
    lines.append("")

    for card in sorted_cards:
        name = card.get("model_name", card.get("model_id", "?"))
        mid = card.get("model_id", "?")
        lines.append(f"### {name} (`{mid}`)")
        lines.append("")
        lines.append(f"- **Final Score**: {card.get('final_score', 0):.4f}")
        safety = "PASS" if card.get("safety_gate_passed") else "FAIL"
        lines.append(f"- **Safety Gate**: {safety}")
        lines.append("")

        # Reliability
        reliability = card.get("metadata", {}).get("reliability", {})
        if reliability.get("status") == "computed":
            icc_val = reliability.get("icc", {}).get("icc_2_1", "-")
            pearson_val = reliability.get("pearson", {}).get("r", "-")
            kappa_val = reliability.get("cohens_kappa", {}).get(
                "kappa_weighted", "-"
            )
            lines.append(
                f"- **Inter-Rater Reliability**: "
                f"ICC={icc_val}, Pearson r={pearson_val}, kappa={kappa_val}"
            )
            lines.append("")

        # Category breakdown
        dim_scores = card.get("dimension_scores", {})
        if dim_scores:
            lines.append("| Dimension | Category | Score | Items |")
            lines.append("|-----------|----------|-------|-------|")

            for dim_name in ["knowledge", "skills", "attitude"]:
                dim = dim_scores.get(dim_name)
                if dim is None:
                    continue
                cat_scores = dim.get("category_scores", {})
                for cat_code in sorted(cat_scores.keys()):
                    cat = cat_scores[cat_code]
                    if isinstance(cat, dict):
                        score = cat.get("score", 0)
                        items = cat.get("item_count", 0)
                    else:
                        score = 0
                        items = 0
                    lines.append(
                        f"| {dim_name.capitalize()} | {cat_code} | "
                        f"{score:.4f} | {items} |"
                    )

            lines.append("")

    lines.append("---")
    lines.append("*Report generated by EduVBench Evaluation Engine*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-model evaluation results into unified report"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="Paths to individual model results directories",
    )
    parser.add_argument(
        "--output",
        default="results/merged",
        help="Output directory for merged report (default: results/merged)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Collect score cards
    score_cards = []
    all_eval_results = []

    for dir_path_str in args.results_dirs:
        dir_path = Path(dir_path_str)
        if not dir_path.is_dir():
            print(f"WARNING: {dir_path} is not a directory, skipping")
            continue

        sc_path = find_score_card(dir_path)
        if sc_path is None:
            print(f"WARNING: No score_card.json found in {dir_path}")
            continue

        card = load_json(sc_path)
        model_id = card.get("model_id", dir_path.name)
        print(f"  Loaded: {model_id} from {sc_path}")
        score_cards.append(card)

        # Also collect eval_results
        er_path = find_eval_results(dir_path)
        if er_path:
            er = load_json(er_path)
            all_eval_results.append(er)

        # Copy reliability if exists
        rel_path = find_reliability(dir_path)
        if rel_path:
            rel = load_json(rel_path)
            # Store in output per model
            model_out = output_dir / model_id
            write_json(model_out / "reliability.json", rel)

    if not score_cards:
        print("ERROR: No score cards found. Nothing to merge.")
        sys.exit(1)

    # Sort by final_score descending
    sorted_cards = sorted(
        score_cards, key=lambda c: c.get("final_score", 0), reverse=True
    )

    print(f"\nMerging {len(sorted_cards)} models...")

    # 1. comparative_report.json
    write_json(
        output_dir / "comparative_report.json",
        {
            "run_dir": str(output_dir),
            "model_count": len(sorted_cards),
            "generated_at": datetime.now().isoformat(),
            "score_cards": sorted_cards,
        },
    )

    # 2. leaderboard.json
    leaderboard = []
    for rank, card in enumerate(sorted_cards, start=1):
        entry = {
            "rank": rank,
            "model_id": card.get("model_id"),
            "model_name": card.get("model_name", card.get("model_id")),
            "final_score": card.get("final_score", 0),
            "safety_gate_passed": card.get("safety_gate_passed", False),
        }
        for dim in ["knowledge", "skills", "attitude"]:
            entry[f"{dim}_score"] = get_dim_score(card, dim)

        meta = card.get("metadata", {})
        entry["items_flagged"] = meta.get("items_flagged_for_review", 0)
        reliability = meta.get("reliability", {})
        if reliability.get("status") == "computed":
            entry["icc_2_1"] = reliability.get("icc", {}).get("icc_2_1")

        leaderboard.append(entry)

    write_json(
        output_dir / "leaderboard.json",
        {
            "generated_at": datetime.now().isoformat(),
            "leaderboard": leaderboard,
        },
    )

    # 3. comparative_report.md
    md_content = generate_markdown(sorted_cards)
    md_path = output_dir / "comparative_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Written: {md_path}")

    # 4. Copy individual score_cards and eval_results
    for card in sorted_cards:
        mid = card.get("model_id", "unknown")
        write_json(output_dir / mid / "score_card.json", card)

    for er in all_eval_results:
        mid = er.get("model_id", "unknown")
        write_json(output_dir / mid / "eval_results.json", er)

    # Summary
    print("\n" + "=" * 60)
    print("  EduVBench Merged Results")
    print("=" * 60)
    print()
    for entry in leaderboard:
        safety = "PASS" if entry["safety_gate_passed"] else "FAIL"
        print(
            f"  #{entry['rank']} {entry['model_name']:20s} "
            f"Final={entry['final_score']:.4f}  "
            f"K={entry['knowledge_score']:.4f}  "
            f"S={entry['skills_score']:.4f}  "
            f"A={entry['attitude_score']:.4f}  "
            f"Safety={safety}"
        )
    print()
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
