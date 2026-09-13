# merged_v4 (VLM-Center results used in the paper)

Per-item results of the dual-judge VLM pipeline (Gemini 3 Flash + GPT-4o, merged)
for all 215 prompts and five models, after re-running the K-PK auto-metric and S-VIU
comparison items. `master_sheet.json` holds the category, dimension, and KSA scores
reported as the VLM-Center block of the paper's main table. Per-judge scores are in
`details.judge_scores`. The algorithmic items here also feed the Human-Center
leaderboard (see `../human_eval/reproduce_leaderboard.py`).
