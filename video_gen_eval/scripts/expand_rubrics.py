#!/usr/bin/env python3
"""Expand rubric_criteria for rubric_10pt items to include all 5 score ranges.

Currently rubric_criteria only have 9-10 and 3-4 ranges (Skills) or
descriptive text without ranges (Knowledge, Attitude). This script
uses an LLM to generate the missing 7-8, 5-6, and 1-2 ranges.

Uses OpenRouter API (Claude Sonnet) to generate expansions.

Usage:
    python3 scripts/expand_rubrics.py                    # expand all 3 JSONs
    python3 scripts/expand_rubrics.py --dry-run           # preview without writing
    python3 scripts/expand_rubrics.py --file knowledge    # single file only
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests required. Install with: pip install requests")

# ── Config ────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "eduvbench-dataset"

FILES = {
    "knowledge": DATASET_DIR / "knowledge_prompts.json",
    "skills": DATASET_DIR / "skills_prompts.json",
    "attitude": DATASET_DIR / "attitude_prompts.json",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o"


# ── LLM Call ──────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call OpenRouter API and return the response text."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Expansion Logic ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert educational assessment designer. You expand rubric criteria for \
evaluating AI-generated educational videos.

Given existing rubric criteria (which may only have 9-10 and 3-4 ranges, or may be \
purely descriptive text), generate a COMPLETE 5-level rubric with ALL score ranges.

RULES:
1. Output EXACTLY this format (one continuous string, no newlines):
   9-10: [criteria]. 7-8: [criteria]. 5-6: [criteria]. 3-4: [criteria]. 1-2: [criteria].
2. Each range description should be 1-2 sentences.
3. Criteria should form a clear GRADIENT from excellent (9-10) to unacceptable (1-2).
4. Keep the original 9-10 and 3-4 text if provided, but you may slightly edit for consistency.
5. Focus on EDUCATIONAL quality, not just visual/technical quality.
6. Be specific to the prompt topic — do NOT use generic phrases.
7. Output ONLY the rubric string. No preamble, no explanation, no markdown."""

def build_user_prompt(prompt_data: dict) -> str:
    """Build the user prompt for rubric expansion."""
    parts = []
    parts.append(f"Prompt ID: {prompt_data.get('id', 'unknown')}")
    parts.append(f"Category: {prompt_data.get('category', 'unknown')}")
    parts.append(f"Grade Level: {prompt_data.get('grade_level', 'unknown')}")
    parts.append(f"Prompt Text: {prompt_data.get('prompt_text', '')[:500]}")

    gt = prompt_data.get("ground_truth", {})
    if isinstance(gt, dict):
        if "correct_answer" in gt:
            parts.append(f"Correct Answer: {gt['correct_answer']}")
        if "misconception" in gt:
            parts.append(f"Misconception: {gt['misconception']}")
        if "correct_concept" in gt:
            parts.append(f"Correct Concept: {gt['correct_concept']}")
        if "key_visual_elements" in gt:
            parts.append(f"Key Visuals: {', '.join(gt['key_visual_elements'])}")

    rubric = prompt_data.get("rubric_criteria", "")
    if rubric:
        parts.append(f"Current Rubric Criteria: {rubric}")

    return "\n".join(parts)


def has_all_ranges(rubric: str) -> bool:
    """Check if rubric already has all 5 score ranges."""
    ranges = ["9-10:", "7-8:", "5-6:", "3-4:", "1-2:"]
    return all(r in rubric for r in ranges)


def validate_expansion(expanded: str) -> bool:
    """Validate that the expansion has all required ranges."""
    ranges = ["9-10:", "7-8:", "5-6:", "3-4:", "1-2:"]
    return all(r in expanded for r in ranges)


# ── Main ──────────────────────────────────────────────────────────────

def process_file(filepath: Path, dry_run: bool = False) -> tuple[int, int]:
    """Process one JSON file. Returns (processed, failed)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data["prompts"]
    rubric_items = [
        p for p in prompts
        if p.get("scoring_method") == "rubric_10pt"
    ]

    print(f"\n  {filepath.name}: {len(rubric_items)} rubric_10pt items")

    processed = 0
    failed = 0
    skipped = 0

    for i, prompt in enumerate(rubric_items):
        pid = prompt.get("id", "?")
        current = prompt.get("rubric_criteria", "")

        if has_all_ranges(current):
            skipped += 1
            print(f"    [{i+1}/{len(rubric_items)}] {pid}: SKIP (already has 5 ranges)")
            continue

        if dry_run:
            print(f"    [{i+1}/{len(rubric_items)}] {pid}: WOULD EXPAND")
            print(f"      Current: {current[:100]}...")
            processed += 1
            continue

        try:
            user_prompt = build_user_prompt(prompt)
            expanded = call_llm(SYSTEM_PROMPT, user_prompt)

            # Clean up any markdown formatting the LLM might add
            expanded = expanded.strip("`").strip()
            if expanded.startswith("rubric"):
                expanded = expanded.split("\n", 1)[-1].strip()

            if validate_expansion(expanded):
                prompt["rubric_criteria"] = expanded
                processed += 1
                print(f"    [{i+1}/{len(rubric_items)}] {pid}: OK")
                preview = expanded[:120].replace("\n", " ")
                print(f"      → {preview}...")
            else:
                failed += 1
                print(f"    [{i+1}/{len(rubric_items)}] {pid}: INVALID (missing ranges)")
                print(f"      Got: {expanded[:120]}...")

            # Rate limit: ~2 requests/sec
            time.sleep(0.5)

        except Exception as e:
            failed += 1
            print(f"    [{i+1}/{len(rubric_items)}] {pid}: ERROR - {e}")
            time.sleep(1)

    if skipped:
        print(f"  Skipped {skipped} items (already complete)")

    if not dry_run and processed > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  Saved {filepath.name}")

    return processed, failed


def main():
    parser = argparse.ArgumentParser(description="Expand rubric_criteria to 5 score ranges")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--file", choices=["knowledge", "skills", "attitude"],
                        help="Process single file only")
    args = parser.parse_args()

    # Load env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    print("=" * 60)
    print("  EduVBench Rubric Criteria Expansion")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Mode:  {'DRY RUN' if args.dry_run else 'LIVE'}")

    files_to_process = FILES
    if args.file:
        files_to_process = {args.file: FILES[args.file]}

    total_processed = 0
    total_failed = 0

    for name, filepath in files_to_process.items():
        if not filepath.exists():
            print(f"\n  WARNING: {filepath} not found, skipping")
            continue
        processed, failed = process_file(filepath, args.dry_run)
        total_processed += processed
        total_failed += failed

    print(f"\n{'=' * 60}")
    print(f"  Done. Processed: {total_processed}, Failed: {total_failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
