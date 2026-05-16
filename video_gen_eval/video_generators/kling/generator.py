"""
Kling 3.0 Video Generator using fal.ai API.

References:
- fal.ai Kling 3.0: https://fal.ai/models/fal-ai/kling-video/v3/standard/text-to-video
- fal.ai API: https://docs.fal.ai/

Endpoints:
- Text-to-Video: fal-ai/kling-video/v3/standard/text-to-video
- Image-to-Video: fal-ai/kling-video/v3/standard/image-to-video

Pricing:
- $0.168/s (no audio), $0.252/s (with audio), $0.308/s (voice control)
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


class Kling3Generator(BaseVideoGenerator):
    """
    Video generator using Kuaishou's Kling 3.0 model via fal.ai API.

    Kling 3.0 features:
    - Duration: 3-15 seconds (flexible range)
    - Native audio generation
    - Multi-shot sequences
    - Aspect ratios: 16:9, 9:16, 1:1

    Pricing (fal.ai):
    - $0.168/s (no audio)
    - $0.252/s (with audio)
    """

    MODEL_NAME = "kling-3.0"
    MODEL_VERSION = "3.0"

    # fal.ai model endpoints
    FAL_TEXT_TO_VIDEO = "fal-ai/kling-video/v3/standard/text-to-video"
    FAL_IMAGE_TO_VIDEO = "fal-ai/kling-video/v3/standard/image-to-video"

    # Valid parameter values
    VALID_ASPECT_RATIOS = ("16:9", "9:16", "1:1")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_key = self.config.get("api_key") or os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")
        self.generate_audio = self.config.get("generate_audio", True)
        self.cfg_scale = self.config.get("cfg_scale", 0.5)
        self.negative_prompt = self.config.get(
            "negative_prompt", "blur, distort, and low quality"
        )
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
                endpoint = self.FAL_IMAGE_TO_VIDEO
            else:
                endpoint = self.FAL_TEXT_TO_VIDEO

            request_params = self._build_request_params(config)

            logger.info(f"Generating video with Kling 3.0 ({endpoint}): {config.prompt[:100]}...")

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
                        "cfg_scale": self.cfg_scale,
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
            logger.error(f"Kling 3.0 generation failed: {e}")
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

        # Aspect ratio (Kling 3.0: "16:9", "9:16", "1:1")
        if config.aspect_ratio in self.VALID_ASPECT_RATIOS:
            params["aspect_ratio"] = config.aspect_ratio
        else:
            params["aspect_ratio"] = "16:9"

        # Duration (Kling 3.0: string "3" to "15")
        duration = config.duration if config.duration else 8
        duration = max(3, min(15, duration))
        params["duration"] = str(duration)

        # Negative prompt
        if self.negative_prompt:
            params["negative_prompt"] = self.negative_prompt

        # CFG scale
        params["cfg_scale"] = self.cfg_scale

        # Audio
        params["generate_audio"] = self.generate_audio

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

            filename = f"kling3_{int(time.time())}.mp4"
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
