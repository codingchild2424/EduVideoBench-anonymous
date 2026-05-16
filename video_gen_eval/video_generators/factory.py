"""
Factory for creating video generator instances.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Type

from .base import (
    BaseVideoGenerator,
    VideoGeneratorType,
    GPUNotAvailableError,
    APIKeyNotConfiguredError,
    check_gpu_availability,
)
from .sora import Sora2Generator
from .veo import Veo31Generator
from .wan import Wan22Generator, Wan26Generator
from .kling import Kling3Generator

logger = logging.getLogger(__name__)


# Registry of available generators
GENERATOR_REGISTRY: Dict[VideoGeneratorType, Type[BaseVideoGenerator]] = {
    VideoGeneratorType.SORA_2: Sora2Generator,
    VideoGeneratorType.VEO_31: Veo31Generator,
    VideoGeneratorType.WAN_22: Wan22Generator,
    VideoGeneratorType.WAN_26: Wan26Generator,
    VideoGeneratorType.KLING_3: Kling3Generator,
}


class VideoGeneratorFactory:
    """Factory for creating and managing video generators."""

    def __init__(self):
        self._instances: Dict[VideoGeneratorType, BaseVideoGenerator] = {}
        self._availability_cache: Optional[Dict[str, Any]] = None

    def create(
        self,
        generator_type: VideoGeneratorType,
        config: Optional[Dict[str, Any]] = None,
        force_new: bool = False,
    ) -> BaseVideoGenerator:
        if generator_type not in GENERATOR_REGISTRY:
            raise ValueError(f"Unsupported generator type: {generator_type}")

        if not force_new and generator_type in self._instances:
            return self._instances[generator_type]

        generator_class = GENERATOR_REGISTRY[generator_type]
        generator = generator_class(config)

        self._instances[generator_type] = generator

        return generator

    def create_by_name(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseVideoGenerator:
        name_map = {
            # Sora 2
            "sora2": VideoGeneratorType.SORA_2,
            "sora-2": VideoGeneratorType.SORA_2,
            "sora": VideoGeneratorType.SORA_2,
            # Veo 3.1
            "veo31": VideoGeneratorType.VEO_31,
            "veo-31": VideoGeneratorType.VEO_31,
            "veo3.1": VideoGeneratorType.VEO_31,
            "veo": VideoGeneratorType.VEO_31,
            "veo3": VideoGeneratorType.VEO_31,
            # Wan 2.2
            "wan22": VideoGeneratorType.WAN_22,
            "wan-22": VideoGeneratorType.WAN_22,
            "wan2.2": VideoGeneratorType.WAN_22,
            # Wan 2.6
            "wan26": VideoGeneratorType.WAN_26,
            "wan-26": VideoGeneratorType.WAN_26,
            "wan2.6": VideoGeneratorType.WAN_26,
            "wan": VideoGeneratorType.WAN_26,
            # Kling 3.0
            "kling3": VideoGeneratorType.KLING_3,
            "kling-3": VideoGeneratorType.KLING_3,
            "kling3.0": VideoGeneratorType.KLING_3,
            "kling": VideoGeneratorType.KLING_3,
        }

        generator_type = name_map.get(name.lower())
        if not generator_type:
            raise ValueError(
                f"Unknown generator name: {name}. "
                f"Available: {list(name_map.keys())}"
            )

        return self.create(generator_type, config)

    def get_available_generators(self) -> Dict[str, Dict[str, Any]]:
        if self._availability_cache:
            return self._availability_cache

        gpu_info = check_gpu_availability()
        enable_gpu = os.getenv("ENABLE_GPU_MODELS", "false").lower() == "true"

        availability = {}

        for gen_type, gen_class in GENERATOR_REGISTRY.items():
            info = {
                "name": gen_type.value,
                "class": gen_class.__name__,
                "requires_gpu": gen_class({}).requires_gpu,
                "requires_api_key": gen_class({}).requires_api_key,
                "available": False,
                "reason": None,
            }

            try:
                temp_instance = gen_class({})

                if temp_instance.requires_gpu:
                    if not enable_gpu:
                        info["reason"] = "GPU models disabled (ENABLE_GPU_MODELS=false)"
                    elif not gpu_info["available"]:
                        info["reason"] = "No GPU available"
                    else:
                        info["available"] = True
                elif temp_instance.requires_api_key:
                    try:
                        temp_instance.validate_config()
                        info["available"] = True
                    except APIKeyNotConfiguredError as e:
                        info["reason"] = str(e)
                else:
                    info["available"] = True

            except Exception as e:
                info["reason"] = str(e)

            availability[gen_type.value] = info

        self._availability_cache = availability
        return availability

    def clear_cache(self):
        self._availability_cache = None

    def unload_all(self):
        for generator in self._instances.values():
            if hasattr(generator, "unload_model"):
                generator.unload_model()

        self._instances.clear()
        logger.info("All generators unloaded")


# Global factory instance
_factory: Optional[VideoGeneratorFactory] = None


def get_factory() -> VideoGeneratorFactory:
    global _factory
    if _factory is None:
        _factory = VideoGeneratorFactory()
    return _factory


def get_available_generators() -> Dict[str, Dict[str, Any]]:
    return get_factory().get_available_generators()


def create_generator(
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseVideoGenerator:
    return get_factory().create_by_name(name, config)
