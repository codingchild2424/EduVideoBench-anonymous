"""
Veo 3.1 Video Generator using fal.ai API.

References:
- fal.ai Veo 3.1: https://fal.ai/models/fal-ai/veo3.1
- fal.ai API: https://docs.fal.ai/
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx

logger = logging.getLogger(__name__)

# Try to import fal_client
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
except ImportError:
    FAL_CLIENT_AVAILABLE = False
    logger.warning("fal-client library not installed. Install with: pip install fal-client")

from ..base import (
    BaseVideoGenerator,
    VideoGenerationConfig,
    VideoGenerationResult,
    APIKeyNotConfiguredError,
)


class Veo31Generator(BaseVideoGenerator):
    """
    Video generator using Google's Veo 3.1 model via fal.ai API.

    Veo 3.1 is Google DeepMind's latest video generation model with:
    - Up to 4K resolution support
    - Native audio generation
    - Improved prompt adherence
    - Duration: 4s, 6s, 8s

    Pricing (fal.ai):
    - 720p/1080p: $0.20/s (no audio), $0.40/s (with audio)
    - 4K: $0.40/s (no audio), $0.60/s (with audio)
    """

    MODEL_NAME = "veo-3.1"
    MODEL_VERSION = "3.1"

    # fal.ai model endpoints
    FAL_ENDPOINT = "fal-ai/veo3.1"
    FAL_FAST_ENDPOINT = "fal-ai/veo3.1/fast"
    FAL_I2V_ENDPOINT = "fal-ai/veo3.1/image-to-video"

    # Valid parameter values
    VALID_DURATIONS = ("4s", "6s", "8s")
    VALID_RESOLUTIONS = ("720p", "1080p", "4k")
    VALID_ASPECT_RATIOS = ("16:9", "9:16")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_key = self.config.get("api_key") or os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")
        self.generate_audio = self.config.get("generate_audio", True)
        self.use_fast = self.config.get("use_fast", False)
        self.timeout = self.config.get("timeout", 600)
        self._initialized = False

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def requires_gpu(self) -> bool:
        return False

    @property
    def requires_api_key(self) -> bool:
        return True

    def validate_config(self) -> bool:
        if not FAL_CLIENT_AVAILABLE:
            raise ImportError(
                "fal-client library not installed. "
                "Install with: pip install fal-client"
            )

        if not self.api_key:
            raise APIKeyNotConfiguredError(
                "fal.ai API key not configured. "
                "Set FAL_AI_API_KEY environment variable or pass api_key in config."
            )

        os.environ["FAL_KEY"] = self.api_key
        self._initialized = True
        return True

    async def generate(
        self,
        config: VideoGenerationConfig,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> VideoGenerationResult:
        self.validate_config()
        start_time = time.time()

        try:
            if config.input_image:
                endpoint = self.FAL_I2V_ENDPOINT
            elif self.use_fast:
                endpoint = self.FAL_FAST_ENDPOINT
            else:
                endpoint = self.FAL_ENDPOINT

            request_params = self._build_request_params(config)

            logger.info(f"Generating video with Veo 3.1 ({endpoint}): {config.prompt[:100]}...")

            result = await self._run_fal_request(endpoint, request_params)

            generation_time = time.time() - start_time

            if result and "video" in result:
                video_url = result["video"].get("url")

                video_path = None
                if video_url and output_dir:
                    video_path = await self._download_video(video_url, output_dir)

                return VideoGenerationResult(
                    success=True,
                    video_path=str(video_path) if video_path else None,
                    video_url=video_url,
                    duration=config.duration,
                    resolution=config.resolution,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    generation_time=generation_time,
                    metadata={
                        "prompt": config.prompt,
                        "aspect_ratio": config.aspect_ratio,
                        "has_audio": self.generate_audio,
                        "fal_endpoint": endpoint,
                    }
                )
            else:
                return VideoGenerationResult(
                    success=False,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    error_message="No video in response",
                    generation_time=generation_time,
                )

        except Exception as e:
            logger.error(f"Veo 3.1 generation failed: {e}")
            return VideoGenerationResult(
                success=False,
                model_name=self.model_name,
                model_version=self.model_version,
                error_message=str(e),
                generation_time=time.time() - start_time,
            )

    def _build_request_params(self, config: VideoGenerationConfig) -> Dict[str, Any]:
        params = {
            "prompt": config.prompt,
        }

        # Aspect ratio (Veo 3.1: "16:9" or "9:16")
        if config.aspect_ratio in self.VALID_ASPECT_RATIOS:
            params["aspect_ratio"] = config.aspect_ratio
        else:
            params["aspect_ratio"] = "16:9"

        # Duration (Veo 3.1 uses string format: "4s", "6s", "8s")
        duration_str = f"{config.duration}s" if config.duration else "8s"
        if duration_str in self.VALID_DURATIONS:
            params["duration"] = duration_str
        else:
            # Snap to nearest valid duration
            if config.duration and config.duration <= 4:
                params["duration"] = "4s"
            elif config.duration and config.duration <= 6:
                params["duration"] = "6s"
            else:
                params["duration"] = "8s"

        # Resolution (Veo 3.1: "720p", "1080p", "4k")
        if config.resolution and config.resolution.lower() in self.VALID_RESOLUTIONS:
            params["resolution"] = config.resolution.lower()
        else:
            params["resolution"] = "720p"

        # Audio
        params["generate_audio"] = self.generate_audio

        # Seed
        if config.seed is not None:
            params["seed"] = config.seed

        return params

    async def _run_fal_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.debug(f"fal.ai: {log['message']}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: fal_client.subscribe(
                endpoint,
                arguments=params,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
        )

        return result

    async def generate_from_problem(
        self,
        problem_text: str,
        subject: str,
        grade_level: str,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> VideoGenerationResult:
        prompt = self._create_educational_prompt(
            problem_text=problem_text,
            subject=subject,
            grade_level=grade_level,
            style=kwargs.get("style", "step-by-step explanation"),
        )

        config = VideoGenerationConfig(
            prompt=prompt,
            duration=kwargs.get("duration", 8),
            resolution=kwargs.get("resolution", "1080p"),
            aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
            subject=subject,
            grade_level=grade_level,
            problem_text=problem_text,
        )

        return await self.generate(config, output_dir)

    async def _download_video(
        self,
        url: str,
        output_dir: Union[str, Path]
    ) -> Optional[Path]:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"veo31_{int(time.time())}.mp4"
            output_path = output_dir / filename

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=120)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

            logger.info(f"Video downloaded to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None
