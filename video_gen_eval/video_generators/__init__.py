"""
Video Generation Models for Educational AI Evaluation Platform.

Supported Models:
- Sora 2 (OpenAI via fal.ai API)
- Veo 3.1 (Google via fal.ai API)
- Kling 3.0 (Kuaishou via fal.ai API)
- Wan 2.2 (Open Source, requires local GPU)
"""

from .base import BaseVideoGenerator, VideoGenerationResult, GPUNotAvailableError
from .sora.generator import Sora2Generator
from .veo.generator import Veo31Generator
from .kling.generator import Kling3Generator
from .wan.generator import Wan22Generator
from .factory import VideoGeneratorFactory, get_available_generators

__all__ = [
    'BaseVideoGenerator',
    'VideoGenerationResult',
    'GPUNotAvailableError',
    'Sora2Generator',
    'Veo31Generator',
    'Kling3Generator',
    'Wan22Generator',
    'VideoGeneratorFactory',
    'get_available_generators',
]
