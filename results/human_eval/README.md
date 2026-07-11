# Human expert evaluation (215 prompts)

Primary evaluation of *EduVideoBench*. All **215 prompts** across 9 subjects were
independently scored by **two PhD-level domain experts per subject (18 experts total,
double-scoring)**, for five video-generation models (Veo 3.1, Sora 2, Kling 3.0,
Wan 2.2, Wan 2.6).

## Files

- **`leaderboard_ksa.csv`** — per-model KSA leaderboard (category, dimension, KSA,
  A-NE block rate, and safety-gate verdict) under both the Human-Center and the
  auxiliary VLM-Center signals. This is the headline result table.
- **`human_eval_215_scores.csv`** — raw per-item human scores (215 rows), with each
  model's two rater scores (`*_rater1`, `*_rater2`, anonymized) and their mean, plus
  `dimension` / `category` / `grade_level` / `scoring_method`.

## Scoring

Scores are `exact_match` (46 items, 0/1) or `rubric_5pt` (169 items). The final
`EduVideoBench-KSA = 0.30·K + 0.40·S + 0.30·A`, with a safety gate that invalidates
any model whose A-NE block rate is below 0.50. See
`../../eduvbench-dataset/scoring_config.json` and `../../eduvbench-dataset/rubrics.json`
for the exact aggregation.
