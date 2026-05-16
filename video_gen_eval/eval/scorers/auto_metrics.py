"""Automatic cognitive-load metric scorer for K-PK-CL items.

Computes six frame-level and audio-level metrics that serve as proxies
for cognitive load, then checks each against grade-level thresholds
from the scoring configuration.

Metrics:
    CL-V1  Visual complexity (Canny edge density + JPEG compression ratio)
    CL-V2  On-screen text density (OCR character count / frame area)
    CL-V3  Scene change frequency (histogram difference between frames)
    CL-A1  Speech rate in WPM (ASR transcription via Whisper)
    CL-X1  Text-narration redundancy (cosine similarity ASR vs OCR)
    CL-X2  Concurrent information elements per 10-second window

Applies to: K-PK-CL subcategories.
"""

import io
import logging
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import BaseScorer
from ..models import ItemEvalResult, Prompt, ScoringMethod, VideoRef
from ..video_utils import extract_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import helpers (heavy dependencies may not be installed)
# ---------------------------------------------------------------------------

def _import_cv2():
    """Import OpenCV at call time."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for auto_metric scoring. "
            "Install it with: pip install opencv-python"
        )


def _import_whisper():
    """Import OpenAI Whisper at call time."""
    try:
        import whisper
        return whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is required for speech rate analysis. "
            "Install it with: pip install openai-whisper"
        )


# ---------------------------------------------------------------------------
# CL-V1: Visual Complexity
# ---------------------------------------------------------------------------

def _compute_visual_complexity(video_path: str, num_frames: int = 16) -> Optional[float]:
    """Compute visual complexity as the average of edge density and compression ratio.

    * Edge density = edge_pixels / total_pixels (Canny detector).
    * Compression ratio = JPEG_size / raw_size.

    Returns:
        Combined metric in [0, 1], or ``None`` if computation fails.
    """
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video for CL-V1: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    edge_densities: List[float] = []
    compression_ratios: List[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Edge density (Canny)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = int((edges > 0).sum())
        if total_pixels > 0:
            edge_densities.append(edge_pixels / total_pixels)

        # JPEG compression ratio
        raw_size = frame.nbytes
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if success and raw_size > 0:
            jpeg_size = len(encoded)
            compression_ratios.append(jpeg_size / raw_size)

    cap.release()

    if not edge_densities:
        return None

    avg_edge = sum(edge_densities) / len(edge_densities)
    avg_comp = (
        sum(compression_ratios) / len(compression_ratios)
        if compression_ratios
        else 0.5
    )
    # Combine: both range roughly [0, 1]; average them
    return (avg_edge + avg_comp) / 2.0


# ---------------------------------------------------------------------------
# CL-V2: Text Density
# ---------------------------------------------------------------------------

def _compute_text_density(video_path: str, num_frames: int = 8) -> Optional[float]:
    """Compute on-screen text density via OCR (chars / pixel area).

    Uses pytesseract if available; returns ``None`` if OCR is not installed.
    """
    try:
        import pytesseract
    except ImportError:
        logger.info("pytesseract not installed; skipping CL-V2 metric")
        return None

    cv2 = _import_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    densities: List[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        frame_area = height * width
        if frame_area == 0:
            continue

        try:
            text = pytesseract.image_to_string(frame)
            char_count = len(text.strip())
            densities.append(char_count / frame_area)
        except Exception as exc:
            logger.debug("OCR failed on frame %d: %s", idx, exc)
            continue

    cap.release()

    if not densities:
        return None

    return sum(densities) / len(densities)


# ---------------------------------------------------------------------------
# CL-V3: Scene Change Frequency
# ---------------------------------------------------------------------------

def _compute_scene_change_rate(
    video_path: str,
    threshold: float = 0.5,
) -> Optional[float]:
    """Compute scene change frequency in changes per minute.

    Uses normalised histogram correlation between consecutive frames;
    a drop below *threshold* counts as a scene change.
    """
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1:
        cap.release()
        return None

    duration_min = total_frames / fps / 60.0
    if duration_min <= 0:
        cap.release()
        return None

    # Sample every N-th frame to keep computation tractable
    sample_interval = max(1, int(fps / 4))  # ~4 samples per second
    prev_hist = None
    scene_changes = 0

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < threshold:
                scene_changes += 1

        prev_hist = hist
        frame_idx += sample_interval
        if frame_idx >= total_frames:
            break

    cap.release()

    return scene_changes / duration_min


# ---------------------------------------------------------------------------
# CL-A1: Speech Rate (WPM)
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> Optional[str]:
    """Extract audio track from video to a temporary WAV file.

    Returns the path to the WAV file, or ``None`` if extraction fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix="eduvbench_audio_")
    wav_path = os.path.join(tmp_dir, "audio.wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                wav_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 or not Path(wav_path).exists():
            logger.debug("ffmpeg audio extraction failed: %s", result.stderr[:200])
            return None
        if Path(wav_path).stat().st_size == 0:
            return None
        return wav_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Audio extraction error: %s", exc)
        return None


def _compute_speech_rate(video_path: str) -> Optional[float]:
    """Compute speech rate in words per minute using Whisper ASR.

    Returns WPM or ``None`` if ASR fails or no speech is detected.
    """
    wav_path = _extract_audio(video_path)
    if wav_path is None:
        return None

    try:
        whisper = _import_whisper()
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, language=None, fp16=False)

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        if not text:
            return None

        word_count = len(text.split())

        # Duration from segments
        if segments:
            duration_sec = segments[-1].get("end", 0) - segments[0].get("start", 0)
        else:
            duration_sec = 0

        if duration_sec <= 0:
            return None

        wpm = word_count / (duration_sec / 60.0)
        return wpm

    except Exception as exc:
        logger.warning("ASR failed for %s: %s", video_path, exc)
        return None
    finally:
        # Clean up temp file
        try:
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink()
                Path(wav_path).parent.rmdir()
        except OSError:
            pass


def _get_asr_text(video_path: str) -> Optional[str]:
    """Get ASR transcript text from the video."""
    wav_path = _extract_audio(video_path)
    if wav_path is None:
        return None
    try:
        whisper = _import_whisper()
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, language=None, fp16=False)
        return result.get("text", "").strip() or None
    except Exception:
        return None
    finally:
        try:
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink()
                Path(wav_path).parent.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CL-X1: Text-Narration Redundancy
# ---------------------------------------------------------------------------

def _cosine_similarity_bow(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using bag-of-words."""
    if not text_a or not text_b:
        return 0.0

    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    all_tokens = set(counter_a.keys()) | set(counter_b.keys())
    if not all_tokens:
        return 0.0

    dot = sum(counter_a.get(t, 0) * counter_b.get(t, 0) for t in all_tokens)
    mag_a = math.sqrt(sum(v ** 2 for v in counter_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _compute_text_narration_redundancy(
    video_path: str,
    num_frames: int = 8,
) -> Optional[float]:
    """Compute cosine similarity between ASR transcript and OCR text.

    Returns similarity in [0, 1], or ``None`` if either source is unavailable.
    """
    try:
        import pytesseract
    except ImportError:
        logger.info("pytesseract not installed; skipping CL-X1 metric")
        return None

    cv2 = _import_cv2()

    # Get ASR text
    asr_text = _get_asr_text(video_path)
    if not asr_text:
        logger.debug("No ASR text for CL-X1; skipping")
        return None

    # Get OCR text from frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    ocr_texts: List[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            text = pytesseract.image_to_string(frame).strip()
            if text:
                ocr_texts.append(text)
        except Exception:
            continue

    cap.release()

    if not ocr_texts:
        return None

    combined_ocr = " ".join(ocr_texts)
    return _cosine_similarity_bow(asr_text, combined_ocr)


# ---------------------------------------------------------------------------
# CL-X2: Concurrent Information Elements
# ---------------------------------------------------------------------------

def _compute_concurrent_elements(
    video_path: str,
    num_frames: int = 8,
) -> Optional[float]:
    """Count concurrent information elements per 10-second window.

    Counts visual objects (contours), text regions (OCR blocks), and
    narration entities (unique words in ASR segments) averaged across
    sampled windows.

    Returns average element count, or ``None`` on failure.
    """
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if total_frames <= 0:
        cap.release()
        return None

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    element_counts: List[int] = []

    # Attempt OCR import
    try:
        import pytesseract
        has_ocr = True
    except ImportError:
        has_ocr = False

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        count = 0

        # Visual objects: count significant contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter by minimum area to avoid noise
        min_area = (frame.shape[0] * frame.shape[1]) * 0.005
        significant_contours = sum(
            1 for c in contours if cv2.contourArea(c) > min_area
        )
        count += significant_contours

        # Text regions via OCR
        if has_ocr:
            try:
                data = pytesseract.image_to_data(
                    frame, output_type=pytesseract.Output.DICT
                )
                text_blocks = sum(
                    1 for conf in data.get("conf", [])
                    if isinstance(conf, (int, float)) and conf > 50
                )
                count += text_blocks
            except Exception:
                pass

        element_counts.append(count)

    cap.release()

    if not element_counts:
        return None

    # Approximate narration entities from ASR (count unique words per segment)
    asr_text = _get_asr_text(video_path)
    narration_entities = 0
    if asr_text:
        # Rough estimate: unique content words
        words = set(w.lower() for w in asr_text.split() if len(w) > 2)
        duration_sec = total_frames / fps
        windows_10s = max(1, duration_sec / 10.0)
        narration_entities = len(words) / windows_10s

    avg_visual = sum(element_counts) / len(element_counts)
    return avg_visual + narration_entities


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------

def _check_in_range(
    value: float,
    grade_level: str,
    metric_config: dict,
    range_key: str = "grade_ranges",
) -> bool:
    """Check whether *value* falls within the grade-level range.

    Returns ``True`` if within range (pass) or if no threshold is defined.
    """
    ranges = metric_config.get(range_key, {})
    grade_range = ranges.get(grade_level)

    if grade_range is None:
        # No threshold for this grade level -- pass by default
        return True

    low = grade_range.get("min", float("-inf"))
    high = grade_range.get("max", float("inf"))

    # Handle density-style thresholds (max_density instead of max)
    if "max_density" in grade_range:
        high = grade_range["max_density"]
        low = 0.0

    return low <= value <= high


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class AutoMetricScorer(BaseScorer):
    """Automated cognitive-load metric scorer for K-PK-CL items.

    Computes up to six metrics, checks each against grade-level thresholds,
    and produces a pass rate (metrics_passed / metrics_computed).  Metrics
    that cannot be computed (missing dependencies, no audio, etc.) are
    excluded from the denominator.

    Args:
        scoring_config: The full scoring configuration dictionary loaded from
            ``scoring_config.json``.  Must contain the
            ``cognitive_load_thresholds`` key.
    """

    def __init__(self, scoring_config: dict) -> None:
        self._thresholds = scoring_config.get("cognitive_load_thresholds", {})

    @property
    def method_name(self) -> str:
        return "auto_metric"

    async def score(
        self,
        prompt: Prompt,
        video_ref: VideoRef,
        vlm_client: Any,
        config: Dict[str, Any],
    ) -> ItemEvalResult:
        """Compute cognitive-load metrics and derive a pass-rate score.

        Args:
            prompt:     Evaluation prompt.
            video_ref:  Reference to the generated video.
            vlm_client: VLM client (unused for auto metrics but included
                for interface consistency).
            config:     Configuration dict (unused; thresholds come from
                the constructor).

        Returns:
            :class:`ItemEvalResult` with ``score`` = pass_rate in [0, 1].
        """
        video_path = video_ref.video_path
        grade_level = prompt.grade_level or "middle"

        metrics_computed = 0
        metrics_passed = 0
        metric_details: Dict[str, Any] = {}
        skipped: List[str] = []
        errors: List[str] = []

        # --- CL-V1: Visual Complexity ---
        try:
            v1 = _compute_visual_complexity(video_path)
            if v1 is not None:
                v1_config = self._thresholds.get("CL-V1", {})
                passed = _check_in_range(v1, grade_level, v1_config)
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-V1"] = {
                    "value": round(v1, 4),
                    "passed": passed,
                    "description": "Visual complexity (edge density + compression ratio)",
                }
            else:
                skipped.append("CL-V1")
        except ImportError as exc:
            skipped.append("CL-V1")
            errors.append(f"CL-V1: {exc}")
        except Exception as exc:
            skipped.append("CL-V1")
            errors.append(f"CL-V1: {exc}")

        # --- CL-V2: Text Density ---
        try:
            v2 = _compute_text_density(video_path)
            if v2 is not None:
                v2_config = self._thresholds.get("CL-V2", {})
                passed = _check_in_range(v2, grade_level, v2_config)
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-V2"] = {
                    "value": round(v2, 6),
                    "passed": passed,
                    "description": "On-screen text density (chars/pixel^2)",
                }
            else:
                skipped.append("CL-V2")
        except ImportError:
            skipped.append("CL-V2")
        except Exception as exc:
            skipped.append("CL-V2")
            errors.append(f"CL-V2: {exc}")

        # --- CL-V3: Scene Change Frequency ---
        try:
            hist_threshold = self._thresholds.get("CL-V3", {}).get(
                "histogram_threshold", 0.5
            )
            v3 = _compute_scene_change_rate(video_path, threshold=hist_threshold)
            if v3 is not None:
                v3_config = self._thresholds.get("CL-V3", {})
                passed = _check_in_range(v3, grade_level, v3_config)
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-V3"] = {
                    "value": round(v3, 2),
                    "passed": passed,
                    "description": "Scene change frequency (changes/min)",
                }
            else:
                skipped.append("CL-V3")
        except ImportError as exc:
            skipped.append("CL-V3")
            errors.append(f"CL-V3: {exc}")
        except Exception as exc:
            skipped.append("CL-V3")
            errors.append(f"CL-V3: {exc}")

        # --- CL-A1: Speech Rate ---
        try:
            a1 = _compute_speech_rate(video_path)
            if a1 is not None:
                a1_config = self._thresholds.get("CL-A1", {})
                passed = _check_in_range(a1, grade_level, a1_config)
                # Also check critical threshold
                critical = a1_config.get("critical_threshold", {})
                max_wpm = critical.get("max_wpm", 275)
                if a1 > max_wpm:
                    passed = False
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-A1"] = {
                    "value": round(a1, 1),
                    "passed": passed,
                    "description": "Speech rate (WPM)",
                }
            else:
                skipped.append("CL-A1")
        except ImportError as exc:
            skipped.append("CL-A1")
            errors.append(f"CL-A1: {exc}")
        except Exception as exc:
            skipped.append("CL-A1")
            errors.append(f"CL-A1: {exc}")

        # --- CL-X1: Text-Narration Redundancy ---
        try:
            x1 = _compute_text_narration_redundancy(video_path)
            if x1 is not None:
                x1_config = self._thresholds.get("CL-X1", {})
                optimal = x1_config.get("optimal_range", {"min": 0.1, "max": 0.25})
                excessive = x1_config.get("excessive_range", {"min": 0.26, "max": 0.5})

                # Pass if within optimal range (not excessive)
                passed = x1 <= optimal.get("max", 0.25)
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-X1"] = {
                    "value": round(x1, 4),
                    "passed": passed,
                    "in_optimal_range": (
                        optimal.get("min", 0.1) <= x1 <= optimal.get("max", 0.25)
                    ),
                    "is_excessive": x1 >= excessive.get("min", 0.26),
                    "description": "Text-narration redundancy (cosine similarity)",
                }
            else:
                skipped.append("CL-X1")
        except ImportError:
            skipped.append("CL-X1")
        except Exception as exc:
            skipped.append("CL-X1")
            errors.append(f"CL-X1: {exc}")

        # --- CL-X2: Concurrent Information Elements ---
        try:
            x2 = _compute_concurrent_elements(video_path)
            if x2 is not None:
                x2_config = self._thresholds.get("CL-X2", {})
                passed = _check_in_range(x2, grade_level, x2_config)
                metrics_computed += 1
                if passed:
                    metrics_passed += 1
                metric_details["CL-X2"] = {
                    "value": round(x2, 1),
                    "passed": passed,
                    "description": "Concurrent information elements (per 10s window)",
                }
            else:
                skipped.append("CL-X2")
        except ImportError as exc:
            skipped.append("CL-X2")
            errors.append(f"CL-X2: {exc}")
        except Exception as exc:
            skipped.append("CL-X2")
            errors.append(f"CL-X2: {exc}")

        # --- Compute pass rate ---
        if metrics_computed > 0:
            pass_rate = metrics_passed / metrics_computed
        else:
            pass_rate = 0.0
            errors.append("No metrics could be computed")

        logger.info(
            "AutoMetric [%s]: %d/%d passed (%.2f), skipped=%s",
            prompt.id,
            metrics_passed,
            metrics_computed,
            pass_rate,
            skipped,
        )

        return ItemEvalResult(
            prompt_id=prompt.id,
            model_id=video_ref.model_id,
            scoring_method=ScoringMethod.AUTO_METRIC,
            score=pass_rate,
            raw_score={
                "metrics_passed": metrics_passed,
                "metrics_computed": metrics_computed,
            },
            details={
                "grade_level": grade_level,
                "metrics": metric_details,
                "skipped_metrics": skipped,
                "errors": errors,
                "pass_rate": round(pass_rate, 4),
            },
        )
