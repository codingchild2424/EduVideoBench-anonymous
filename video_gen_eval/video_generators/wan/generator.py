"""
Wan 2.2 Video Generator using fal.ai API.

References:
- fal.ai Wan 2.2 5B: https://fal.ai/models/fal-ai/wan/v2.2-5b/text-to-video/api
- fal.ai Wan 2.2 A14B: https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/api
- GitHub: https://github.com/Wan-Video/Wan2.2

Endpoints:
- 5B: fal-ai/wan/v2.2-5b/text-to-video
- A14B: fal-ai/wan/v2.2-a14b/text-to-video

Wan 2.2 (2025-07-28): Open-source (Apache 2.0), MoE architecture, up to 720p
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx

logger = logging.getLogger(__name__)

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


class Wan22Generator(BaseVideoGenerator):
    """
    Video generator using Alibaba's Wan 2.2 model via fal.ai API.

    Wan 2.2 (open-source, Apache 2.0):
    - Duration: up to ~5 seconds (17-161 frames)
    - Resolution: up to 720p
    - Aspect ratios: 16:9, 9:16, 1:1
    """

    MODEL_NAME = "wan2.2"
    MODEL_VERSION = "2.2"

    FAL_ENDPOINT_5B = "fal-ai/wan/v2.2-5b/text-to-video"
    FAL_ENDPOINT_A14B = "fal-ai/wan/v2.2-a14b/text-to-video"

    VALID_ASPECT_RATIOS = ("16:9", "9:16", "1:1")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("FAL_AI_API_KEY") or os.getenv("FAL_KEY")
        self.use_a14b = self.config.get("use_a14b", False)
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
            raise ImportError("fal-client not installed. Install with: pip install fal-client")
        if not self.api_key:
            raise APIKeyNotConfiguredError(
                "fal.ai API key not configured. Set FAL_AI_API_KEY environment variable."
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
            endpoint = self.FAL_ENDPOINT_A14B if self.use_a14b else self.FAL_ENDPOINT_5B
            params = self._build_request_params(config)

            logger.info(f"Generating video with Wan 2.2 ({endpoint}): {config.prompt[:100]}...")

            result = await self._run_fal_request(endpoint, params)
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
                        "open_source": True,
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
            logger.error(f"Wan 2.2 generation failed: {e}")
            return VideoGenerationResult(
                success=False,
                model_name=self.model_name,
                model_version=self.model_version,
                error_message=str(e),
                generation_time=time.time() - start_time,
            )

    def _build_request_params(self, config: VideoGenerationConfig) -> Dict[str, Any]:
        params = {"prompt": config.prompt}

        if config.aspect_ratio in self.VALID_ASPECT_RATIOS:
            params["aspect_ratio"] = config.aspect_ratio
        else:
            params["aspect_ratio"] = "16:9"

        # Wan 2.2: resolution as string "580p" or "720p"
        if config.resolution in ("480p", "580p"):
            params["resolution"] = "580p"
        else:
            params["resolution"] = "720p"

        # Wan 2.2: num_frames 17-161 at 24fps. max 161 frames ≈ 6.7s
        fps = 24
        duration = config.duration if config.duration else 5
        num_frames = min(161, max(17, duration * fps))
        params["num_frames"] = num_frames

        if config.seed is not None:
            params["seed"] = config.seed

        return params

    async def _run_fal_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            duration=kwargs.get("duration", 5),
            resolution=kwargs.get("resolution", "720p"),
            aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
            subject=subject,
            grade_level=grade_level,
            problem_text=problem_text,
        )
        return await self.generate(config, output_dir)

    async def _download_video(self, url: str, output_dir: Union[str, Path]) -> Optional[Path]:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"wan22_{int(time.time())}.mp4"
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
