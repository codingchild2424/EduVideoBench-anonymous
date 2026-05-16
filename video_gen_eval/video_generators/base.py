"""
Base classes and interfaces for video generation models.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoGeneratorType(Enum):
    """Supported video generation model types."""
    SORA_2 = "sora2"
    VEO_31 = "veo31"
    WAN_22 = "wan22"
    WAN_26 = "wan26"
    KLING_3 = "kling3"


class GPUNotAvailableError(Exception):
    """Raised when GPU is required but not available."""
    pass


class APIKeyNotConfiguredError(Exception):
    """Raised when required API key is not configured."""
    pass


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation."""
    prompt: str
    duration: int = 10  # seconds
    resolution: str = "1080p"
    fps: int = 24
    aspect_ratio: str = "16:9"
    style: Optional[str] = None

    # Input options
    input_image: Optional[str] = None  # Path to input image for image-to-video
    reference_images: Optional[List[str]] = None  # For multi-image reference

    # Educational context
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    problem_text: Optional[str] = None

    # Advanced options
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@dataclass
class VideoGenerationResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None

    # Metadata
    model_name: str = ""
    model_version: str = ""
    generation_time: Optional[float] = None  # seconds

    # Error info
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check if GPU is available for local model inference.

    Returns:
        Dictionary with GPU availability info
    """
    gpu_info = {
        "available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "total_memory_gb": 0,
    }

    try:
        import torch

        gpu_info["cuda_available"] = torch.cuda.is_available()

        if gpu_info["cuda_available"]:
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()

            for i in range(gpu_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": device_props.name,
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
                gpu_info["devices"].append(device_info)
                gpu_info["total_memory_gb"] += device_info["total_memory_gb"]

    except ImportError:
        logger.warning("PyTorch not installed. GPU check unavailable.")
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")

    return gpu_info


def is_gpu_available() -> bool:
    """Simple check if GPU is available."""
    return check_gpu_availability()["available"]


def get_required_vram(model_size: str) -> float:
    """
    Get required VRAM for Wan 2.2 model sizes.

    Args:
        model_size: Model size identifier (1.3B, 5B, 14B)

    Returns:
        Required VRAM in GB
    """
    vram_requirements = {
        "1.3B": 8.0,
        "5B": 16.0,
        "14B": 24.0,
    }
    return vram_requirements.get(model_size, 24.0)


class BaseVideoGenerator(ABC):
    """Abstract base class for video generation models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the video generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._initialized = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version."""
        pass

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """Return whether this model requires GPU."""
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Return whether this model requires an API key."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if configuration is valid

        Raises:
            APIKeyNotConfiguredError: If API key is required but not set
            GPUNotAvailableError: If GPU is required but not available
        """
        pass

    @abstractmethod
    async def generate(
        self,
        config: VideoGenerationConfig,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> VideoGenerationResult:
        """
        Generate a video based on the configuration.

        Args:
            config: Video generation configuration
            output_dir: Directory to save the generated video

        Returns:
            VideoGenerationResult with generation results
        """
        pass

    @abstractmethod
    async def generate_from_problem(
        self,
        problem_text: str,
        subject: str,
        grade_level: str,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> VideoGenerationResult:
        """
        Generate an educational video for a specific problem.

        Args:
            problem_text: The problem/question text
            subject: Subject area (math, science, etc.)
            grade_level: Target grade level
            output_dir: Directory to save the generated video
            **kwargs: Additional generation parameters

        Returns:
            VideoGenerationResult with generation results
        """
        pass

    def _create_educational_prompt(
        self,
        problem_text: str,
        subject: str,
        grade_level: str,
        style: str = "step-by-step explanation",
    ) -> str:
        """
        Create an educational video generation prompt.

        Args:
            problem_text: The problem to solve
            subject: Subject area
            grade_level: Target grade level
            style: Explanation style

        Returns:
            Formatted prompt string
        """
        prompt = f"""Create an educational video that explains how to solve the following {subject} problem for {grade_level} students.

Problem: {problem_text}

The video should:
1. Clearly present the problem
2. Show a {style} to solve it
3. Highlight key concepts and steps
4. Display the final answer prominently
5. Use clear visuals and annotations

Style: Educational, clear, engaging for {grade_level} level students."""

        return prompt

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the generator.

        Returns:
            Status dictionary
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "requires_gpu": self.requires_gpu,
            "requires_api_key": self.requires_api_key,
            "initialized": self._initialized,
            "config_valid": self.validate_config() if self._initialized else False,
        }
