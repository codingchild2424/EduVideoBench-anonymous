"""Video processing utilities for EduVBench evaluation.

Provides frame extraction, metadata retrieval, audio extraction, and
transcription with graceful fallbacks when optional dependencies
(cv2, whisper, ffmpeg) are unavailable.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dependency checks
# ------------------------------------------------------------------


def check_cv_available() -> bool:
    """Check whether OpenCV (cv2) is importable.

    Returns:
        True if cv2 can be imported, False otherwise.
    """
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def check_whisper_available() -> bool:
    """Check whether OpenAI Whisper is importable.

    Returns:
        True if whisper can be imported, False otherwise.
    """
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def check_ffmpeg_available() -> bool:
    """Check whether ffmpeg is available on the system PATH.

    Returns:
        True if ``ffmpeg -version`` exits successfully, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ------------------------------------------------------------------
# Frame extraction
# ------------------------------------------------------------------


def extract_frames(
    video_path: str,
    num_frames: int = 8,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Extract uniformly-sampled frames from a video file.

    Tries OpenCV first for speed and reliability; falls back to ffmpeg
    subprocess if cv2 is unavailable.

    Args:
        video_path: Path to the input video file.
        num_frames: Number of frames to extract (uniformly spaced).
        output_dir: Directory to write JPEG frames into. If None, a
            temporary directory is created automatically.

    Returns:
        Ordered list of absolute paths to the extracted JPEG frame files.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If neither cv2 nor ffmpeg is available, or if
            frame extraction fails with both backends.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="eduvbench_frames_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Try OpenCV first
    if check_cv_available():
        try:
            return _extract_frames_cv2(str(video), num_frames, output_dir)
        except Exception as exc:
            logger.warning("cv2 frame extraction failed, trying ffmpeg: %s", exc)

    # Fallback to ffmpeg
    if check_ffmpeg_available():
        try:
            return _extract_frames_ffmpeg(str(video), num_frames, output_dir)
        except Exception as exc:
            logger.error("ffmpeg frame extraction also failed: %s", exc)
            raise RuntimeError(
                f"Frame extraction failed with both cv2 and ffmpeg: {exc}"
            ) from exc

    raise RuntimeError(
        "Neither cv2 nor ffmpeg is available. "
        "Install opencv-python or ensure ffmpeg is on PATH."
    )


def _extract_frames_cv2(
    video_path: str,
    num_frames: int,
    output_dir: str,
) -> List[str]:
    """Extract frames using OpenCV.

    Samples frames uniformly across the video duration by computing
    evenly-spaced frame indices.

    Args:
        video_path: Path to the video file.
        num_frames: Desired number of frames.
        output_dir: Directory to write output JPEGs.

    Returns:
        List of paths to saved JPEG files.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video reports {total_frames} frames")

        # Clamp num_frames to available frames
        actual_num = min(num_frames, total_frames)
        if actual_num < num_frames:
            logger.info(
                "Video has only %d frames; extracting %d instead of %d",
                total_frames,
                actual_num,
                num_frames,
            )

        # Compute uniformly-spaced frame indices
        if actual_num == 1:
            indices = [total_frames // 2]
        else:
            step = (total_frames - 1) / (actual_num - 1)
            indices = [int(round(i * step)) for i in range(actual_num)]

        saved_paths: List[str] = []
        for idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Could not read frame at index %d", frame_idx)
                continue

            out_path = str(Path(output_dir) / f"frame_{idx:03d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        if not saved_paths:
            raise RuntimeError("No frames could be read from the video")

        logger.info(
            "Extracted %d frames via cv2 from %s", len(saved_paths), video_path
        )
        return saved_paths

    finally:
        cap.release()


def _extract_frames_ffmpeg(
    video_path: str,
    num_frames: int,
    output_dir: str,
) -> List[str]:
    """Extract frames using ffmpeg subprocess as a fallback.

    Uses the ``select`` video filter to pick every Nth frame, then
    limits output to the desired count.

    Args:
        video_path: Path to the video file.
        num_frames: Desired number of frames.
        output_dir: Directory to write output JPEGs.

    Returns:
        List of paths to saved JPEG files.
    """
    # First, probe total frame count to compute step
    probe_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "json",
        video_path,
    ]

    step = 1
    try:
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=60
        )
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            streams = probe_data.get("streams", [])
            if streams:
                total = int(streams[0].get("nb_read_frames", 0))
                if total > num_frames:
                    step = total // num_frames
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.debug("Frame count probe failed, using step=1: %s", exc)

    output_pattern = str(Path(output_dir) / "frame_%03d.jpg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select=not(mod(n\\,{step}))",
        "-frames:v", str(num_frames),
        "-fps_mode", "vfr",
        "-q:v", "2",
        output_pattern,
        "-y",
        "-loglevel", "warning",
    ]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr}")

    # Collect output files in sorted order
    saved_paths = sorted(
        str(p) for p in Path(output_dir).glob("frame_*.jpg")
    )

    if not saved_paths:
        raise RuntimeError("ffmpeg produced no output frames")

    logger.info(
        "Extracted %d frames via ffmpeg from %s", len(saved_paths), video_path
    )
    return saved_paths


# ------------------------------------------------------------------
# Video metadata
# ------------------------------------------------------------------


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Retrieve basic metadata from a video file.

    Tries OpenCV first, then falls back to ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with keys: ``duration_sec``, ``width``, ``height``,
        ``fps``, ``codec``. Values may be None if unavailable.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If neither cv2 nor ffprobe can read the video.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Try OpenCV
    if check_cv_available():
        try:
            return _get_metadata_cv2(str(video))
        except Exception as exc:
            logger.warning("cv2 metadata extraction failed, trying ffprobe: %s", exc)

    # Fallback to ffprobe
    if check_ffmpeg_available():
        try:
            return _get_metadata_ffprobe(str(video))
        except Exception as exc:
            logger.error("ffprobe metadata extraction also failed: %s", exc)
            raise RuntimeError(
                f"Metadata extraction failed with both cv2 and ffprobe: {exc}"
            ) from exc

    raise RuntimeError(
        "Neither cv2 nor ffprobe is available. "
        "Install opencv-python or ensure ffmpeg/ffprobe is on PATH."
    )


def _get_metadata_cv2(video_path: str) -> Dict[str, Any]:
    """Extract metadata using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        Metadata dictionary.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decode FourCC to string
        codec = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip()

        duration_sec = frame_count / fps if fps > 0 else None

        metadata = {
            "duration_sec": round(duration_sec, 2) if duration_sec else None,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "codec": codec if codec else None,
        }

        logger.debug("cv2 metadata for %s: %s", video_path, metadata)
        return metadata

    finally:
        cap.release()


def _get_metadata_ffprobe(video_path: str) -> Dict[str, Any]:
    """Extract metadata using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Metadata dictionary.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (rc={result.returncode}): {result.stderr}")

    data = json.loads(result.stdout)

    # Find the video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    format_info = data.get("format", {})

    # Parse duration
    duration_sec = None
    if "duration" in format_info:
        try:
            duration_sec = round(float(format_info["duration"]), 2)
        except (ValueError, TypeError):
            pass

    # Parse video stream properties
    width = None
    height = None
    fps = None
    codec = None

    if video_stream:
        width = video_stream.get("width")
        height = video_stream.get("height")
        codec = video_stream.get("codec_name")

        # Parse frame rate from r_frame_rate (e.g., "30/1")
        r_frame_rate = video_stream.get("r_frame_rate", "")
        if "/" in r_frame_rate:
            try:
                num, den = r_frame_rate.split("/")
                fps = round(int(num) / int(den), 2) if int(den) != 0 else None
            except (ValueError, ZeroDivisionError):
                pass

    metadata = {
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
    }

    logger.debug("ffprobe metadata for %s: %s", video_path, metadata)
    return metadata


# ------------------------------------------------------------------
# Audio extraction and transcription
# ------------------------------------------------------------------


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Extract audio track from a video file as a 16kHz WAV.

    Uses ffmpeg to demux and convert the audio stream to PCM 16-bit
    mono at 16kHz, which is the expected input format for Whisper.

    Args:
        video_path: Path to the input video file.
        output_path: Path to write the output WAV file. If None, a
            temporary file is created.

    Returns:
        Path to the extracted WAV file, or None if extraction fails.
    """
    video = Path(video_path)
    if not video.exists():
        logger.error("Video not found for audio extraction: %s", video_path)
        return None

    if not check_ffmpeg_available():
        logger.error("ffmpeg is not available; cannot extract audio")
        return None

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="eduvbench_audio_", delete=False
        )
        output_path = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg",
        "-i", str(video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
        "-y",
        "-loglevel", "warning",
    ]

    logger.debug("Extracting audio: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(
                "ffmpeg audio extraction failed (rc=%d): %s",
                result.returncode,
                result.stderr,
            )
            return None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg audio extraction timed out for %s", video_path)
        return None
    except OSError as exc:
        logger.error("ffmpeg execution error: %s", exc)
        return None

    # Verify output exists and is non-empty
    out = Path(output_path)
    if not out.exists() or out.stat().st_size == 0:
        logger.error("Audio extraction produced empty output: %s", output_path)
        return None

    logger.info("Extracted audio to %s (%.1f KB)", output_path, out.stat().st_size / 1024)
    return output_path


def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe an audio file using OpenAI Whisper.

    Loads the ``base`` Whisper model and runs transcription. If the
    whisper package is not installed, returns None with a warning.

    Args:
        audio_path: Path to a WAV audio file (16kHz recommended).

    Returns:
        Transcribed text string, or None if Whisper is unavailable
        or transcription fails.
    """
    if not check_whisper_available():
        logger.warning(
            "Whisper is not installed. Install with: pip install openai-whisper"
        )
        return None

    audio = Path(audio_path)
    if not audio.exists():
        logger.error("Audio file not found: %s", audio_path)
        return None

    try:
        import whisper

        logger.info("Loading Whisper 'base' model...")
        model = whisper.load_model("base")

        logger.info("Transcribing %s...", audio_path)
        result = model.transcribe(str(audio))
        text = result.get("text", "").strip()

        if text:
            logger.info(
                "Transcription complete: %d characters from %s",
                len(text),
                audio_path,
            )
        else:
            logger.warning("Transcription returned empty text for %s", audio_path)

        return text if text else None

    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", audio_path, exc)
        return None
