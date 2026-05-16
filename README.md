# EduVideoBench-anonymous

Anonymous repository for the EduVideoBench submission. A KSA-grounded benchmark for evaluating video generation models (VGMs) in the education domain across 9 categories and 215 prompts.

## Generated videos

All 1,070 videos used in the evaluation are released on Google Drive:

https://drive.google.com/drive/folders/1caQFcX2bSGN0n4Z_XmSx0gwSJy33voxr?usp=sharing

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
| **Human evaluation** | `video_gen_eval/templates/*.xlsx` | Blank English evaluation sheets (one per subject) used by domain experts. See `video_gen_eval/templates/HUMAN_EVAL_GUIDE.md`. |

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

## Dataset version note

The xlsx human-evaluation templates and the `results/` outputs cover the 215-prompt set reported in the paper. The accompanying JSON dataset under `eduvbench-dataset/` is an earlier 185-prompt release; an updated JSON drop matching the 215-prompt set will accompany the camera-ready version.

## License

Code and rubrics are released under a permissive license for research use. Generated videos follow the terms of the respective providers.
