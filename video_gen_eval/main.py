#!/usr/bin/env python3
"""
Educational Video Generation CLI Tool

Usage:
    python main.py generate --prompt "..." --model veo31
    python main.py batch --input problems.json --model sora2
    python main.py status
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

# Add video_generators to path
sys.path.insert(0, str(Path(__file__).parent))

from video_generators.base import VideoGenerationConfig, VideoGenerationResult
from video_generators.factory import create_generator, get_available_generators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Available model choices
MODEL_CHOICES = ["sora2", "veo31", "kling3", "wan22", "wan26"]


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def create_output_structure(base_dir: str, model: str = "") -> tuple[Path, Path]:
    """
    Create output directory structure.

    Returns:
        tuple: (run_dir, videos_dir)

    Structure:
        outputs/
        └── veo31/
            └── 20260217_143052/
                ├── videos/
                │   └── 001_veo31_title.mp4
                ├── 001_veo31_title.json
                └── run_summary.json
    """
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model:
        run_dir = base_path / model / timestamp
    else:
        run_dir = base_path / timestamp

    videos_dir = run_dir / "videos"

    run_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {run_dir}")
    return run_dir, videos_dir


def generate_video_filename(index: int, model: str, label: Optional[str] = None) -> str:
    """Generate a labeled video filename."""
    if label:
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
        safe_label = safe_label[:50]
        return f"{index:03d}_{model}_{safe_label}.mp4"
    return f"{index:03d}_{model}.mp4"


def save_paired_json(
    run_dir: Path,
    video_path: Path,
    result: VideoGenerationResult,
    index: int,
    model: str,
    prompt: str,
    subject: str = "",
    grade_level: str = "",
    title: str = "",
    prompt_id: str = "",
    extra: Optional[dict] = None,
):
    """
    Save a JSON metadata file paired with the video file.

    Video: run_dir/videos/001_veo31_example.mp4
    JSON:  run_dir/001_veo31_example.json
    """
    json_filename = video_path.with_suffix(".json").name
    json_path = run_dir / json_filename

    metadata = {
        "index": index,
        "video_file": video_path.name,
        "model": model,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "prompt": prompt,
        "title": title,
        "subject": subject,
        "grade_level": grade_level,
        "success": result.success,
        "generation_time_sec": round(result.generation_time, 2) if result.generation_time else None,
        "video_url": result.video_url,
        "duration": result.duration,
        "resolution": result.resolution,
        "error": result.error_message,
        "created_at": datetime.now().isoformat(),
        "metadata": result.metadata,
    }

    if extra:
        metadata.update(extra)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Metadata saved to: {json_path}")
    return json_path


def save_run_summary(run_dir: Path, metadata_list: List[dict]):
    """Save a summary JSON for the entire run."""
    summary_path = run_dir / "run_summary.json"

    summary = {
        "total": len(metadata_list),
        "success": sum(1 for m in metadata_list if m.get("success")),
        "failed": sum(1 for m in metadata_list if not m.get("success")),
        "created_at": datetime.now().isoformat(),
        "results": metadata_list,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Run summary saved to: {summary_path}")
    return summary_path


def print_result(result: VideoGenerationResult, index: int = None):
    """Print generation result in a readable format."""
    prefix = f"[{index:03d}] " if index else ""
    print("\n" + "=" * 60)
    if result.success:
        print(f"{prefix}Video generation successful!")
        print(f"  Model: {result.model_name} v{result.model_version}")
        if result.video_path:
            print(f"  Saved to: {result.video_path}")
        if result.video_url:
            print(f"  URL: {result.video_url}")
        if result.generation_time:
            print(f"  Generation time: {result.generation_time:.2f}s")
        if result.duration:
            print(f"  Video duration: {result.duration}s")
    else:
        print(f"{prefix}Video generation failed!")
        print(f"  Error: {result.error_message}")
    print("=" * 60 + "\n")


async def cmd_generate(args, config: dict):
    """Handle 'generate' command - single video generation."""
    try:
        model_config = config.get("models", {}).get(args.model, {})
        generator = create_generator(args.model, model_config)

        gen_config = VideoGenerationConfig(
            prompt=args.prompt,
            duration=args.duration,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            input_image=args.image,
        )

        base_dir = args.output or config.get("output_dir", "./outputs")
        run_dir, videos_dir = create_output_structure(base_dir, args.model)

        logger.info(f"Generating video with {args.model}...")
        result = await generator.generate(gen_config, videos_dir)

        # Rename video file with label
        if result.success and result.video_path:
            old_path = Path(result.video_path)
            new_filename = generate_video_filename(1, args.model, args.label)
            new_path = videos_dir / new_filename
            shutil.move(old_path, new_path)
            result.video_path = str(new_path)

            save_paired_json(
                run_dir=run_dir,
                video_path=new_path,
                result=result,
                index=1,
                model=args.model,
                prompt=args.prompt,
                title=args.label or "",
            )

        print_result(result, index=1)

        save_run_summary(run_dir, [{
            "index": 1,
            "video_file": Path(result.video_path).name if result.video_path else "",
            "model": args.model,
            "prompt": args.prompt,
            "success": result.success,
            "generation_time_sec": round(result.generation_time, 2) if result.generation_time else None,
            "error": result.error_message,
        }])

        return 0 if result.success else 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


async def cmd_generate_edu(args, config: dict):
    """Handle 'edu' command - educational video from problem."""
    try:
        model_config = config.get("models", {}).get(args.model, {})
        generator = create_generator(args.model, model_config)

        base_dir = args.output or config.get("output_dir", "./outputs")
        run_dir, videos_dir = create_output_structure(base_dir, args.model)

        logger.info(f"Generating educational video with {args.model}...")
        result = await generator.generate_from_problem(
            problem_text=args.problem,
            subject=args.subject,
            grade_level=args.grade,
            output_dir=videos_dir,
            duration=args.duration,
            resolution=args.resolution,
        )

        # Rename video file with label
        if result.success and result.video_path:
            old_path = Path(result.video_path)
            label = args.label or f"{args.subject}_{args.grade.replace(' ', '_')}"
            new_filename = generate_video_filename(1, args.model, label)
            new_path = videos_dir / new_filename
            shutil.move(old_path, new_path)
            result.video_path = str(new_path)

            save_paired_json(
                run_dir=run_dir,
                video_path=new_path,
                result=result,
                index=1,
                model=args.model,
                prompt=args.problem,
                subject=args.subject,
                grade_level=args.grade,
                title=args.label or "",
            )

        print_result(result, index=1)

        save_run_summary(run_dir, [{
            "index": 1,
            "video_file": Path(result.video_path).name if result.video_path else "",
            "model": args.model,
            "prompt": args.problem,
            "subject": args.subject,
            "grade_level": args.grade,
            "success": result.success,
            "generation_time_sec": round(result.generation_time, 2) if result.generation_time else None,
            "error": result.error_message,
        }])

        return 0 if result.success else 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


def find_latest_run_dir(base_dir: str, model: str) -> Optional[Path]:
    """Find the most recent run directory for a model."""
    model_dir = Path(base_dir) / model
    if not model_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def get_completed_ids(run_dir: Path) -> set:
    """Scan paired JSON files in a run directory and return completed prompt IDs/titles.

    Both successful generations and recorded failures (e.g. content_policy_violation)
    are considered "completed" to prevent infinite retries on permanently blocked prompts.
    """
    completed = set()
    for json_file in run_dir.glob("*.json"):
        if json_file.name == "run_summary.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            title = meta.get("title", "")
            if title:
                completed.add(title)
        except (json.JSONDecodeError, OSError):
            continue
    return completed


async def cmd_batch(args, config: dict):
    """Handle 'batch' command - batch video generation."""
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both top-level list and {"prompts": [...]} wrapper
        if isinstance(data, list):
            problems = data
        elif isinstance(data, dict):
            problems = data.get("prompts", [data])
        else:
            problems = [data]

        model_config = config.get("models", {}).get(args.model, {})
        generator = create_generator(args.model, model_config)

        base_dir = args.output or config.get("output_dir", "./outputs")

        # --resume: reuse latest run directory and skip completed prompts
        completed_ids: set = set()
        if args.resume:
            latest = find_latest_run_dir(base_dir, args.model)
            if latest:
                run_dir = latest
                videos_dir = run_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)
                completed_ids = get_completed_ids(run_dir)
                logger.info(
                    f"Resuming from {run_dir} — {len(completed_ids)} already completed"
                )
            else:
                logger.info("No previous run found. Starting fresh.")
                run_dir, videos_dir = create_output_structure(base_dir, args.model)
        else:
            run_dir, videos_dir = create_output_structure(base_dir, args.model)

        summary_list = []
        success_count = 0
        skipped_count = 0

        # Filter out already completed prompts
        pending = []
        for i, problem in enumerate(problems, 1):
            title = problem.get("title", problem.get("id", f"video_{i}"))
            if title in completed_ids:
                skipped_count += 1
                continue
            pending.append((i, problem, title))

        if skipped_count:
            logger.info(f"Skipped {skipped_count} already completed, {len(pending)} remaining")

        concurrency = getattr(args, "concurrency", 1) or 1

        if concurrency <= 1:
            # --- Sequential mode (original) ---
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    pending,
                    desc=f"{args.model}",
                    unit="video",
                    dynamic_ncols=True,
                )
            except ImportError:
                pbar = pending

            for i, problem, title in pbar:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix_str(title[:40], refresh=True)

                if "prompt" in problem or "prompt_text" in problem:
                    prompt = problem.get("prompt") or problem.get("prompt_text", "")
                    gen_config = VideoGenerationConfig(
                        prompt=prompt,
                        duration=problem.get("duration", args.duration),
                        resolution=problem.get("resolution", args.resolution),
                        aspect_ratio=problem.get("aspect_ratio", "16:9"),
                    )
                    result = await generator.generate(gen_config, videos_dir)
                    subject = problem.get("subject", "")
                    grade_level = problem.get("grade_level", "")
                else:
                    prompt = problem.get("problem_text", "")
                    subject = problem.get("subject", "math")
                    grade_level = problem.get("grade_level", "middle school")
                    result = await generator.generate_from_problem(
                        problem_text=prompt,
                        subject=subject,
                        grade_level=grade_level,
                        output_dir=videos_dir,
                        duration=problem.get("duration", args.duration),
                    )

                if result.success and result.video_path:
                    old_path = Path(result.video_path)
                    new_filename = generate_video_filename(i, args.model, title)
                    new_path = videos_dir / new_filename
                    shutil.move(old_path, new_path)
                    result.video_path = str(new_path)
                    success_count += 1

                    save_paired_json(
                        run_dir=run_dir,
                        video_path=new_path,
                        result=result,
                        index=i,
                        model=args.model,
                        prompt=prompt,
                        subject=subject,
                        grade_level=grade_level,
                        title=title,
                        prompt_id=problem.get("id", ""),
                    )
                elif not result.success:
                    # Save failure metadata so block_test scorer can detect refusals
                    fail_filename = f"{i:03d}_{args.model}_{title}.json"
                    save_paired_json(
                        run_dir=run_dir,
                        video_path=run_dir / "videos" / fail_filename,
                        result=result,
                        index=i,
                        model=args.model,
                        prompt=prompt,
                        subject=subject,
                        grade_level=grade_level,
                        title=title,
                        prompt_id=problem.get("id", ""),
                    )

                summary_list.append({
                    "index": i,
                    "video_file": Path(result.video_path).name if result.video_path else "",
                    "model": args.model,
                    "prompt": prompt,
                    "title": title,
                    "subject": subject,
                    "grade_level": grade_level,
                    "success": result.success,
                    "generation_time_sec": round(result.generation_time, 2) if result.generation_time else None,
                    "error": result.error_message,
                })

            if hasattr(pbar, "close"):
                pbar.close()

        else:
            # --- Concurrent mode ---
            sem = asyncio.Semaphore(concurrency)
            lock = asyncio.Lock()

            logger.info(f"Concurrent mode: {concurrency} simultaneous requests")

            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=len(pending),
                    desc=f"{args.model}",
                    unit="video",
                    dynamic_ncols=True,
                )
            except ImportError:
                pbar = None

            async def _generate_one(i, problem, title):
                nonlocal success_count
                async with sem:
                    if pbar:
                        pbar.set_postfix_str(title[:40], refresh=True)

                    if "prompt" in problem or "prompt_text" in problem:
                        prompt = problem.get("prompt") or problem.get("prompt_text", "")
                        gen_config = VideoGenerationConfig(
                            prompt=prompt,
                            duration=problem.get("duration", args.duration),
                            resolution=problem.get("resolution", args.resolution),
                            aspect_ratio=problem.get("aspect_ratio", "16:9"),
                        )
                        result = await generator.generate(gen_config, videos_dir)
                        subject = problem.get("subject", "")
                        grade_level = problem.get("grade_level", "")
                    else:
                        prompt = problem.get("problem_text", "")
                        subject = problem.get("subject", "math")
                        grade_level = problem.get("grade_level", "middle school")
                        result = await generator.generate_from_problem(
                            problem_text=prompt,
                            subject=subject,
                            grade_level=grade_level,
                            output_dir=videos_dir,
                            duration=problem.get("duration", args.duration),
                        )

                # File operations under lock to avoid race conditions
                async with lock:
                    if result.success and result.video_path:
                        old_path = Path(result.video_path)
                        new_filename = generate_video_filename(i, args.model, title)
                        new_path = videos_dir / new_filename
                        shutil.move(old_path, new_path)
                        result.video_path = str(new_path)
                        success_count += 1

                        save_paired_json(
                            run_dir=run_dir,
                            video_path=new_path,
                            result=result,
                            index=i,
                            model=args.model,
                            prompt=prompt,
                            subject=subject,
                            grade_level=grade_level,
                            title=title,
                            prompt_id=problem.get("id", ""),
                        )
                    elif not result.success:
                        # Save failure metadata so block_test scorer can detect refusals
                        fail_filename = f"{i:03d}_{args.model}_{title}.json"
                        save_paired_json(
                            run_dir=run_dir,
                            video_path=run_dir / "videos" / fail_filename,
                            result=result,
                            index=i,
                            model=args.model,
                            prompt=prompt,
                            subject=subject,
                            grade_level=grade_level,
                            title=title,
                            prompt_id=problem.get("id", ""),
                        )

                    summary_list.append({
                        "index": i,
                        "video_file": Path(result.video_path).name if result.video_path else "",
                        "model": args.model,
                        "prompt": prompt,
                        "title": title,
                        "subject": subject,
                        "grade_level": grade_level,
                        "success": result.success,
                        "generation_time_sec": round(result.generation_time, 2) if result.generation_time else None,
                        "error": result.error_message,
                    })

                    if pbar:
                        pbar.update(1)

            tasks = [_generate_one(i, problem, title) for i, problem, title in pending]
            await asyncio.gather(*tasks, return_exceptions=True)

            if pbar:
                pbar.close()

        # Save run summary
        save_run_summary(run_dir, summary_list)

        print(f"\n{'=' * 60}")
        if skipped_count:
            print(f"Batch complete: {success_count} new + {skipped_count} skipped / {len(problems)} total")
        else:
            print(f"Batch complete: {success_count}/{len(problems)} successful")
        print(f"Output directory: {run_dir}")
        print(f"{'=' * 60}\n")

        return 0 if (success_count + skipped_count) == len(problems) else 1

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


def cmd_status(args, config: dict):
    """Handle 'status' command - show available generators."""
    print("\n" + "=" * 60)
    print("Video Generator Status")
    print("=" * 60)

    generators = get_available_generators()

    for name, info in generators.items():
        status = "Available" if info["available"] else "Unavailable"
        print(f"\n{name.upper()}")
        print(f"  Status: {status}")
        print(f"  Class: {info['class']}")
        print(f"  Requires GPU: {'Yes' if info['requires_gpu'] else 'No'}")
        print(f"  Requires API Key: {'Yes' if info['requires_api_key'] else 'No'}")
        if info.get("reason"):
            print(f"  Reason: {info['reason']}")

    print("\n" + "-" * 60)
    print("Environment Variables:")
    fal_key = os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")
    print(f"  FAL_AI_API_KEY: {'Set' if fal_key else 'Not set'}")

    enable_gpu = os.getenv("ENABLE_GPU_MODELS", "false")
    print(f"  ENABLE_GPU_MODELS: {enable_gpu}")

    print("=" * 60 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Educational Video Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single video
  python main.py generate --prompt "Explain the Pythagorean theorem" --model veo31

  # Generate educational video from a math problem
  python main.py edu --problem "Solve: 2x + 5 = 15" --subject math --grade "middle school"

  # Batch generate from JSON file
  python main.py batch --input problems.json --model sora2

  # Check available generators
  python main.py status

Models:
  sora2   - OpenAI Sora 2 (fal.ai, $0.30-0.50/s)
  veo31   - Google Veo 3.1 (fal.ai, $0.20-0.60/s, up to 4K)
  kling3  - Kuaishou Kling 3.0 (fal.ai, $0.168-0.252/s)
  wan22   - Alibaba Wan 2.2 (fal.ai, open-source, up to 720p)
  wan26   - Alibaba Wan 2.6 (fal.ai, $0.10-0.15/s, multi-shot)

Output Structure:
  outputs/
  └── veo31/
      └── 20260217_143052/
          ├── videos/
          │   ├── 001_veo31_title.mp4
          │   └── 002_veo31_title.mp4
          ├── 001_veo31_title.json   # paired metadata
          ├── 002_veo31_title.json
          └── run_summary.json
        """
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to config.yaml file",
        default=None
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a single video")
    gen_parser.add_argument("--prompt", "-p", required=True, help="Video generation prompt")
    gen_parser.add_argument("--model", "-m", default="veo31",
                           choices=MODEL_CHOICES,
                           help="Model to use (default: veo31)")
    gen_parser.add_argument("--duration", "-d", type=int, default=8,
                           help="Video duration in seconds (default: 8)")
    gen_parser.add_argument("--resolution", "-r", default="1080p",
                           choices=["480p", "720p", "1080p", "4k"],
                           help="Video resolution (default: 1080p)")
    gen_parser.add_argument("--aspect-ratio", "-a", default="16:9",
                           help="Aspect ratio (default: 16:9)")
    gen_parser.add_argument("--image", "-i", help="Input image for image-to-video")
    gen_parser.add_argument("--label", "-l", help="Label for the video filename")
    gen_parser.add_argument("--output", "-o", help="Output base directory")

    # Educational video command
    edu_parser = subparsers.add_parser("edu", help="Generate educational video from problem")
    edu_parser.add_argument("--problem", "-p", required=True, help="Problem text")
    edu_parser.add_argument("--subject", "-s", default="math",
                           help="Subject area (default: math)")
    edu_parser.add_argument("--grade", "-g", default="middle school",
                           help="Grade level (default: middle school)")
    edu_parser.add_argument("--model", "-m", default="veo31",
                           choices=MODEL_CHOICES,
                           help="Model to use (default: veo31)")
    edu_parser.add_argument("--duration", "-d", type=int, default=8,
                           help="Video duration in seconds (default: 8)")
    edu_parser.add_argument("--resolution", "-r", default="1080p",
                           help="Video resolution (default: 1080p)")
    edu_parser.add_argument("--label", "-l", help="Label for the video filename")
    edu_parser.add_argument("--output", "-o", help="Output base directory")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch generate videos")
    batch_parser.add_argument("--input", "-i", required=True,
                             help="Input JSON file with problems")
    batch_parser.add_argument("--model", "-m", default="veo31",
                             choices=MODEL_CHOICES,
                             help="Model to use (default: veo31)")
    batch_parser.add_argument("--duration", "-d", type=int, default=8,
                             help="Default video duration (default: 8)")
    batch_parser.add_argument("--resolution", "-r", default="1080p",
                             help="Default resolution (default: 1080p)")
    batch_parser.add_argument("--output", "-o", help="Output base directory")
    batch_parser.add_argument("--resume", action="store_true",
                             help="Resume from the latest run (skip already completed prompts)")
    batch_parser.add_argument("--concurrency", type=int, default=1,
                             help="Number of concurrent video generations (default: 1)")

    # Status command
    subparsers.add_parser("status", help="Show available generators and status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Load config
    config = load_config(args.config)

    # Load .env from project root
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    # Map FAL_AI_API_KEY -> FAL_KEY for fal_client compatibility
    fal_api_key = os.getenv("FAL_AI_API_KEY")
    if fal_api_key and not os.getenv("FAL_KEY"):
        os.environ["FAL_KEY"] = fal_api_key

    # Execute command
    if args.command == "generate":
        return asyncio.run(cmd_generate(args, config))
    elif args.command == "edu":
        return asyncio.run(cmd_generate_edu(args, config))
    elif args.command == "batch":
        return asyncio.run(cmd_batch(args, config))
    elif args.command == "status":
        return cmd_status(args, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
