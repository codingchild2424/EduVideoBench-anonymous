"""OpenRouter VLM client for EduVBench evaluation.

Provides async access to vision-language models via OpenRouter,
with caching, rate limiting, retry logic, and support for dual-judge
configurations (native video URL vs. frame-based input).
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from .config import JudgeConfig

logger = logging.getLogger(__name__)


class VLMClient:
    """Async client for querying vision-language models through OpenRouter.

    Supports text-only, multi-frame image, and native video URL queries
    with disk caching, semaphore-based rate limiting, exponential backoff
    retries, and structured JSON output (``response_format``).

    Args:
        model_id: OpenRouter model identifier.
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        judge_id: Short identifier for this judge (e.g. ``gemini-3-pro``).
        input_method: ``video_url`` for native video or ``frame_images``
            for base64-encoded frames.
        reasoning_effort: Reasoning level for thinking models
            (``low`` / ``medium`` / ``high``). None disables.
        temperature: Sampling temperature for generation.
        max_retries: Maximum number of retry attempts on transient errors.
        rate_limit_rpm: Requests per minute limit (controls concurrency).
        cache_dir: Directory for disk-based response caching. None disables.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model_id: str = "google/gemini-3-pro-preview",
        api_key: Optional[str] = None,
        judge_id: str = "",
        input_method: str = "frame_images",
        reasoning_effort: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        rate_limit_rpm: int = 60,
        cache_dir: Optional[str] = None,
        timeout: int = 120,
    ):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required. Pass api_key or set OPENROUTER_API_KEY env var."
            )

        self.model_id = model_id
        self.judge_id = judge_id or model_id
        self.input_method = input_method
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_key,
        )
        self._semaphore = asyncio.Semaphore(max(1, rate_limit_rpm // 60))
        self._cache_dir = Path(cache_dir) if cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("VLM response cache enabled at %s", self._cache_dir)

        logger.info(
            "VLMClient initialized: judge=%s, model=%s, input=%s, "
            "reasoning=%s, temperature=%.2f",
            self.judge_id,
            self.model_id,
            self.input_method,
            self.reasoning_effort,
            self.temperature,
        )

    @classmethod
    def from_judge_config(
        cls,
        config: JudgeConfig,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "VLMClient":
        """Create a VLMClient from a JudgeConfig instance.

        Args:
            config: Judge configuration with model, input method, and
                rate limit settings.
            api_key: Optional API key override.
            cache_dir: Optional cache directory override.

        Returns:
            A fully configured :class:`VLMClient`.
        """
        return cls(
            model_id=config.model_id,
            api_key=api_key,
            judge_id=config.id,
            input_method=config.input_method,
            reasoning_effort=config.reasoning_effort,
            temperature=config.temperature,
            max_retries=config.max_retries,
            rate_limit_rpm=config.rate_limit_rpm,
            cache_dir=cache_dir,
            timeout=config.timeout,
        )

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    async def query_text(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a text-only query to the VLM (used for S-VIU baseline).

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User-level question or task description.
            response_format: Optional structured output schema (json_schema).

        Returns:
            The model's text response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self._call_with_retry(
            messages, response_format=response_format
        )

    async def query_with_frames(
        self,
        system_prompt: str,
        user_prompt: str,
        frame_paths: List[str],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a multi-modal query with video frames to the VLM.

        Each frame is read from disk, base64-encoded, and included as an
        inline image_url content part alongside the text prompt.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User-level question or task description.
            frame_paths: Ordered list of JPEG frame file paths.
            response_format: Optional structured output schema (json_schema).

        Returns:
            The model's text response.

        Raises:
            FileNotFoundError: If any frame path does not exist.
        """
        content_parts: list = []

        # Add each frame as a base64-encoded image
        for frame_path in frame_paths:
            path = Path(frame_path)
            if not path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")

            image_data = path.read_bytes()
            b64_str = base64.b64encode(image_data).decode("utf-8")
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}",
                    },
                }
            )

        # Add the text prompt after images
        content_parts.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        logger.debug(
            "Querying VLM with %d frames, prompt length=%d",
            len(frame_paths),
            len(user_prompt),
        )
        return await self._call_with_retry(
            messages, response_format=response_format
        )

    async def query_with_video_url(
        self,
        system_prompt: str,
        user_prompt: str,
        video_url: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a query with a native video URL to the VLM.

        Used by video-native models (e.g. Gemini) that can process the full
        video directly rather than requiring extracted frames.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User-level question or task description.
            video_url: Publicly accessible URL to the video file.
            response_format: Optional structured output schema (json_schema).

        Returns:
            The model's text response.
        """
        content_parts = [
            {
                "type": "image_url",
                "image_url": {"url": video_url},
            },
            {"type": "text", "text": user_prompt},
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        logger.debug(
            "Querying VLM with video URL, prompt length=%d",
            len(user_prompt),
        )
        return await self._call_with_retry(
            messages, response_format=response_format
        )

    async def query_with_video(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        frame_paths: Optional[List[str]] = None,
        num_frames: int = 8,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Auto-dispatch query based on the judge's input_method.

        Chooses the appropriate input modality based on ``self.input_method``:
        - ``video_url``: Uses native video URL (requires *video_url*).
        - ``frame_images``: Uses extracted frames (requires *frame_paths*
          or *video_path* for on-the-fly extraction).

        Args:
            system_prompt: System-level instruction.
            user_prompt: User-level prompt text.
            video_path: Path to the video file (for frame extraction).
            video_url: Publicly accessible URL to the video.
            frame_paths: Pre-extracted frame file paths.
            num_frames: Frames to extract if *video_path* is used.
            response_format: Optional structured output schema.

        Returns:
            The model's text response.
        """
        if self.input_method == "video_url" and video_url:
            return await self.query_with_video_url(
                system_prompt, user_prompt, video_url,
                response_format=response_format,
            )

        # Fall back to frame-based input
        if frame_paths is None:
            if video_path is None:
                raise ValueError(
                    "Either frame_paths or video_path is required "
                    "for frame_images input method"
                )
            from .video_utils import extract_frames
            frame_paths = extract_frames(video_path, num_frames=num_frames)

        return await self.query_with_frames(
            system_prompt, user_prompt, frame_paths,
            response_format=response_format,
        )

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, messages: list, extra: Optional[dict] = None) -> str:
        """Compute a deterministic cache key from model ID and messages.

        Uses SHA-256 of the JSON-serialized model + messages payload to
        produce a collision-resistant key. Any extra kwargs (like
        response_format) are included in the hash.

        Args:
            messages: The full messages list sent to the API.
            extra: Additional parameters that affect the response.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        payload_dict: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if extra:
            payload_dict["extra"] = extra
        payload = json.dumps(payload_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[str]:
        """Retrieve a cached response from disk.

        Args:
            key: Cache key (SHA-256 hex digest).

        Returns:
            The cached response string, or None if not found.
        """
        if self._cache_dir is None:
            return None

        cache_file = self._cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.debug("Cache hit: %s", key[:16])
            return data.get("response")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache file %s: %s", cache_file, exc)
            return None

    def _set_cached(self, key: str, response: str) -> None:
        """Write a response to the disk cache.

        Args:
            key: Cache key (SHA-256 hex digest).
            response: The model response text to cache.
        """
        if self._cache_dir is None:
            return

        cache_file = self._cache_dir / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "model": self.model_id,
                        "judge_id": self.judge_id,
                        "response": response,
                        "cached_at": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.debug("Cached response: %s", key[:16])
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)

    # ------------------------------------------------------------------
    # Core API call with retry
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        messages: list,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call the OpenRouter API with caching, rate limiting, and retries.

        Flow:
        1. Check disk cache; return immediately on hit.
        2. Acquire semaphore to respect rate limits.
        3. Attempt the API call up to ``max_retries`` times with
           exponential backoff on transient errors.
        4. Cache and return the successful response.

        Args:
            messages: The complete messages payload for the chat completion.
            response_format: Optional structured output format (json_schema).

        Returns:
            The model's text response.

        Raises:
            openai.APIError: If all retry attempts are exhausted.
        """
        # 1. Check cache
        extra_for_cache = {}
        if response_format:
            extra_for_cache["response_format"] = response_format
        key = self._cache_key(messages, extra_for_cache or None)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        # Build extra_body for provider-specific params
        extra_body: Dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}

        # Build API kwargs
        api_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if response_format:
            api_kwargs["response_format"] = response_format
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # 2. Retry loop with semaphore
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                try:
                    logger.debug(
                        "API call attempt %d/%d for %s (%s)",
                        attempt,
                        self.max_retries,
                        self.judge_id,
                        self.model_id,
                    )
                    response = await self._client.chat.completions.create(
                        **api_kwargs
                    )

                    result = response.choices[0].message.content
                    if result is None:
                        result = ""

                    # 3. Cache the successful response
                    self._set_cached(key, result)

                    if response.usage:
                        logger.debug(
                            "Token usage [%s]: prompt=%d, completion=%d, total=%d",
                            self.judge_id,
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                            response.usage.total_tokens,
                        )

                    return result

                except openai.RateLimitError as exc:
                    last_exception = exc
                    wait = 2**attempt
                    logger.warning(
                        "Rate limited [%s] (attempt %d/%d). Retrying in %ds: %s",
                        self.judge_id,
                        attempt,
                        self.max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

                except openai.APITimeoutError as exc:
                    last_exception = exc
                    wait = 2**attempt
                    logger.warning(
                        "Request timed out [%s] (attempt %d/%d). Retrying in %ds: %s",
                        self.judge_id,
                        attempt,
                        self.max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

                except openai.APIConnectionError as exc:
                    last_exception = exc
                    wait = 2**attempt
                    logger.warning(
                        "Connection error [%s] (attempt %d/%d). Retrying in %ds: %s",
                        self.judge_id,
                        attempt,
                        self.max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

                except openai.APIStatusError as exc:
                    # Non-retryable API errors (4xx other than 429)
                    logger.error(
                        "Non-retryable API error [%s] (status %d): %s",
                        self.judge_id,
                        exc.status_code,
                        exc,
                    )
                    raise

        # All retries exhausted
        logger.error(
            "All %d retry attempts failed for %s (%s)",
            self.max_retries,
            self.judge_id,
            self.model_id,
        )
        raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structured output schema helpers
# ---------------------------------------------------------------------------

RUBRIC_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_eval_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Score on a 1-10 scale",
                },
                "reasoning": {
                    "type": "string",
                    "description": "2-5 sentences explaining why this score was given",
                },
                "key_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific visual/audio evidence observed",
                },
                "rubric_alignment": {
                    "type": "object",
                    "properties": {
                        "criteria_met": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "criteria_missed": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["criteria_met", "criteria_missed"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "score",
                "reasoning",
                "key_observations",
                "rubric_alignment",
            ],
            "additionalProperties": False,
        },
    },
}

EXACT_MATCH_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "exact_match_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "extracted_answer": {
                    "type": "string",
                    "description": "The factual answer extracted from the video",
                },
                "is_correct": {
                    "type": "boolean",
                    "description": "Whether the extracted answer matches the expected answer",
                },
                "score": {
                    "type": "number",
                    "description": "1.0 if correct, 0.0 if not",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the extraction and comparison",
                },
            },
            "required": ["extracted_answer", "is_correct", "score", "reasoning"],
            "additionalProperties": False,
        },
    },
}

SAFETY_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "safety_eval_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "contains_harmful_content": {
                    "type": "boolean",
                    "description": "Whether the video contains harmful content",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the safety assessment",
                },
                "key_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific safety-related observations",
                },
            },
            "required": [
                "contains_harmful_content",
                "reasoning",
                "key_observations",
            ],
            "additionalProperties": False,
        },
    },
}
