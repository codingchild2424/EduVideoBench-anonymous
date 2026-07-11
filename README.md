# EduVideoBench-anonymous

Anonymous repository for the EduVideoBench submission. A KSA-grounded benchmark for evaluating video generation models (VGMs) in the education domain across 9 categories and 215 prompts.

## Generated videos

All 1,070 videos used in the evaluation are available here (view-only): **https://drive.google.com/drive/folders/1caQFcX2bSGN0n4Z_XmSx0gwSJy33voxr?usp=sharing**

During review the videos can also be regenerated from the prompts in `eduvbench-dataset/` using the scripts described below.

## Reviewer quick start

```bash
# 1. install
cd video_gen_eval && pip install -r requirements.txt

# 2. set API keys
cp ../.env.example ../.env
# then edit ../.env and fill:
#   FAL_AI_API_KEY=<your fal.ai key>           # for Sora 2, Veo 3.1, Kling 3.0
#   OPENROUTER_API_KEY=<your openrouter key>   # for VLM judges (GPT-4o, Gemini 3 Flash)

# 3a. (optional) regenerate videos from the prompt suite
python main.py batch --input ../eduvbench-dataset/knowledge_prompts.json --model sora2

# 3b. run the VLM evaluation pipeline
python -m video_gen_eval.eval evaluate --models sora2 veo31 kling3 wan22 wan26

# 3c. generate score report
python -m video_gen_eval.eval report --results-dir ./results/<run_id>
```

## Two evaluation paths

| Track | Where | What reviewers can do |
|---|---|---|
| **VLM evaluation** | `video_gen_eval/eval/` | Run the dual-judge VLM pipeline (Gemini 3 Flash + GPT-4o) end to end with the API keys above. Outputs in `results/`. |
| **Human evaluation** | `video_gen_eval/templates/*.xlsx` | Blank English evaluation sheets (one per subject) used by domain experts. See `video_gen_eval/templates/HUMAN_EVAL_GUIDE.md`. The full evaluation-sheet forms are also mirrored here (view-only): **https://drive.google.com/drive/folders/1kD6-dl8fUiSzsdeCJuE96O5s2oz2qnEC?usp=sharing** |
| **Expert interview (TPACK & TAM)** | `video_gen_eval/templates/TPACK_TAM_interview_questionnaire.pdf` | Blank TPACK & TAM interview questionnaire administered to domain experts. |

Pre-computed evaluation outputs (model score cards, comparative reports) are in `results/`.

## Repository layout

```
EduVideoBench-anonymous/
├── eduvbench-dataset/          # benchmark prompts, rubrics, scoring config
├── video_gen_eval/
│   ├── video_generators/       # generation adapters (Sora 2, Veo 3.1, Kling 3.0, Wan 2.2/2.6)
│   ├── eval/                   # KSA scorers, aggregator, CLI
│   ├── scripts/                # batch runners, helpers
│   ├── templates/              # blank English human-eval sheets + guide
│   ├── main.py                 # generation entry point
│   └── requirements.txt
├── results/                    # per-model evaluation outputs (merged)
├── .env.example                # API key placeholders
└── README.md
```

## Dataset

The JSON dataset under `eduvbench-dataset/` contains the **complete 215-prompt set** reported in the paper, split across the three KSA dimensions:

| File | Dimension | Categories | Prompts |
|---|---|---|---|
| `knowledge_prompts.json` | Knowledge | 61 K-CK + 21 K-PK | 82 |
| `skills_prompts.json` | Skills | 35 S-PF + 42 S-UC + 9 S-VIU | 86 |
| `attitude_prompts.json` | Attitude | 13 A-ES + 7 A-IS + 12 A-NE + 15 A-DD | 47 |
| **Total** | | | **215** |

Every prompt entry carries an embedded `ground_truth` field (`correct_answer` / `key_visual_elements` / `key_steps` for Knowledge and Skills; `misconception` / `correct_concept` / `boundary_type` for Attitude). A flattened `ground_truth_sheet.csv` consolidating these across all 215 prompts is provided for convenience.

**Evaluation.** The primary evaluation is the **expert human evaluation over all 215 prompts** (two PhD-level domain experts per subject, 18 experts total, double-scoring). The aggregated KSA leaderboard and the raw per-item human scores are in **`results/human_eval/`** (`leaderboard_ksa.csv`, `human_eval_215_scores.csv`); the completed source evaluation sheets are also available here (view-only): **https://drive.google.com/drive/folders/1pkOtLANw_G6TSS83EK4c8StWn2klSBMv?usp=sharing**. An auxiliary dual-VLM judging signal is provided under `results/` for cross-checking.

## License

Code and rubrics are released under a permissive license for research use. Generated videos follow the terms of the respective providers.
