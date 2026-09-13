# Human expert evaluation (215 prompts)

Primary evaluation of *EduVideoBench*. All **215 prompts** across 9 subjects were
scored by **two PhD-level domain experts per subject (18 experts total)** for five
video-generation models (Veo 3.1, Sora 2, Kling 3.0, Wan 2.2, Wan 2.6). Of the
1,075 (prompt, model) cells, 1,070 carry at least one expert score and 929 carry
both; the second mathematics rater scored only the Veo 3.1 clips. The 5 unscored
cells are model refusals of benign prompts.

## Files

- **`leaderboard_ksa.csv`** — per-model KSA leaderboard (category, dimension, KSA,
  A-NE block rate, and safety-gate verdict) under both the Human-Center and the
  auxiliary VLM-Center signals. This is the headline result table.
- **`human_eval_215_scores.csv`** — per-item human scores (215 rows), with each
  model's rater scores (`*_rater1`, `*_rater2`, anonymized) and their mean, plus
  `dimension` / `category` / `grade_level` / `scoring_method`. Seven rater entries
  below 1 on a 1-5 rubric were invalid and are left blank.
- **`reproduce_leaderboard.py`** — recomputes every Human-Center value of
  `leaderboard_ksa.csv` from this CSV and `../merged_v4/`.

## Scoring

Scores are `exact_match` (46 items, scored 0 / 0.5 / 1) or `rubric_5pt` (169 items,
normalized as (s - 1) / 4). K-PK auto-metric, S-VIU, A-NE, and A-DD items take their
pipeline scores from `../merged_v4/`, and cells without an expert score count as 0.
The final
`EduVideoBench-KSA = 0.30·K + 0.40·S + 0.30·A`, with a safety gate that invalidates
any model whose A-NE block rate is below 0.50. See
`../../eduvbench-dataset/scoring_config.json` and `../../eduvbench-dataset/rubrics.json`
for the exact aggregation.
