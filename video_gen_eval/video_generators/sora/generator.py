"""
Sora 2 Video Generator using fal.ai API.

References:
- fal.ai Sora 2: https://fal.ai/models/fal-ai/sora-2
- fal.ai API: https://docs.fal.ai/

Endpoints:
- Standard: fal-ai/sora-2/text-to-video ($0.10/s, 720p only)
- Pro: fal-ai/sora-2/text-to-video/pro ($0.30-0.50/s, up to 1080p)
- Image-to-Video: fal-ai/sora-2/image-to-video/pro
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


class Sora2Generator(BaseVideoGenerator):
    """
    Video generator using OpenAI's Sora 2 model via fal.ai API.

    Supported modes:
    - text-to-video (Standard): 720p, $0.10/s
    - text-to-video (Pro): 720p-1080p, $0.30-0.50/s
    - image-to-video (Pro): from image + text

    Duration: 4, 8, or 12 seconds
    Aspect ratios: 16:9, 9:16
    """

    MODEL_NAME = "sora-2"
    MODEL_VERSION = "2.0"

    # fal.ai model endpoints
    FAL_TEXT_TO_VIDEO = "fal-ai/sora-2/text-to-video"
    FAL_TEXT_TO_VIDEO_PRO = "fal-ai/sora-2/text-to-video/pro"
    FAL_IMAGE_TO_VIDEO_PRO = "fal-ai/sora-2/image-to-video/pro"

    # Valid parameter values
    VALID_DURATIONS = (4, 8, 12)
    VALID_RESOLUTIONS_STD = ("720p",)
    VALID_RESOLUTIONS_PRO = ("720p", "1080p")
    VALID_ASPECT_RATIOS = ("16:9", "9:16")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_key = self.config.get("api_key") or os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")
        self.use_pro = self.config.get("use_pro", True)
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
            # Determine endpoint
            if config.input_image:
                endpoint = self.FAL_IMAGE_TO_VIDEO_PRO
                request_params = await self._build_image_to_video_params(config)
            elif self.use_pro:
                endpoint = self.FAL_TEXT_TO_VIDEO_PRO
                request_params = self._build_text_to_video_params(config, pro=True)
            else:
                endpoint = self.FAL_TEXT_TO_VIDEO
                request_params = self._build_text_to_video_params(config, pro=False)

            logger.info(f"Generating video with Sora 2 ({endpoint}): {config.prompt[:100]}...")

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
                        "fal_endpoint": endpoint,
                        "pro": self.use_pro,
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
            logger.error(f"Sora 2 generation failed: {e}")
            return VideoGenerationResult(
                success=False,
                model_name=self.model_name,
                model_version=self.model_version,
                error_message=str(e),
                generation_time=time.time() - start_time,
            )

    def _build_text_to_video_params(
        self, config: VideoGenerationConfig, pro: bool = False
    ) -> Dict[str, Any]:
        params = {
            "prompt": config.prompt,
        }

        # Aspect ratio
        if config.aspect_ratio in self.VALID_ASPECT_RATIOS:
            params["aspect_ratio"] = config.aspect_ratio
        else:
            params["aspect_ratio"] = "16:9"

        # Duration (Sora 2 uses integer: 4, 8, 12)
        if config.duration in self.VALID_DURATIONS:
            params["duration"] = config.duration
        else:
            # Snap to nearest valid duration
            if config.duration and config.duration <= 4:
                params["duration"] = 4
            elif config.duration and config.duration <= 8:
                params["duration"] = 8
            else:
                params["duration"] = 12

        # Resolution
        if pro:
            valid = self.VALID_RESOLUTIONS_PRO
            default = "1080p"
        else:
            valid = self.VALID_RESOLUTIONS_STD
            default = "720p"

        if config.resolution and config.resolution.lower() in valid:
            params["resolution"] = config.resolution.lower()
        else:
            params["resolution"] = default

        return params

    async def _build_image_to_video_params(
        self, config: VideoGenerationConfig
    ) -> Dict[str, Any]:
        params = self._build_text_to_video_params(config, pro=True)

        image_path = Path(config.input_image)
        if image_path.exists():
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            suffix = image_path.suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/png")

            params["image_url"] = f"data:{mime_type};base64,{image_data}"

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
            duration=kwargs.get("duration", 12),
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

            filename = f"sora2_{int(time.time())}.mp4"
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
