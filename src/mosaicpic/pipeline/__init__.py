"""Composable image processing pipeline for mosaic conversion.

This module provides a pipeline-based approach to image processing,
enabling flexible composition of steps like pooling, quantization,
and dithering.

Main Components:
    - Pipeline: Compose and execute processing steps
    - Steps: PoolStep, QuantizeStep, DitherStep
    - Configs: PoolConfig, QuantizeConfig, DitherConfig
    - Types: RGBImage, IndexMap
    - Profiles: Built-in profiles (classic, sharp, dithered)

Example:
    >>> from mosaicpic.pipeline import (
    ...     Pipeline, PoolStep, DitherStep,
    ...     PoolConfig, DitherConfig,
    ...     PoolMethod, ColorSpace, DitherAlgorithm, ScanOrder,
    ...     get_profile, list_profiles,
    ... )
    >>>
    >>> # Use built-in profile
    >>> pipeline = get_profile("dithered")
    >>>
    >>> # Or build custom pipeline
    >>> custom_pipeline = Pipeline([
    ...     PoolStep(PoolConfig(output_size=(96, 96), color_space=ColorSpace.LAB)),
    ...     DitherStep(DitherConfig(algorithm=DitherAlgorithm.FLOYD_STEINBERG)),
    ...     PoolStep(PoolConfig()),  # Uses target_size from context
    ... ])
"""

from .config import (
    ColorSpace,
    DitherAlgorithm,
    DitherConfig,
    PoolConfig,
    PoolMethod,
    QuantizeConfig,
    ScanOrder,
)
from .context import PipelineContext
from .pipeline import Pipeline
from .profiles import get_profile, list_profiles, register_profile
from .steps import DitherStep, PoolStep, QuantizeStep, Step
from .types import IndexMap, RGBImage, StepData

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineContext",
    # Steps
    "Step",
    "PoolStep",
    "QuantizeStep",
    "DitherStep",
    # Configs
    "PoolConfig",
    "QuantizeConfig",
    "DitherConfig",
    # Enums
    "PoolMethod",
    "ColorSpace",
    "DitherAlgorithm",
    "ScanOrder",
    # Types
    "RGBImage",
    "IndexMap",
    "StepData",
    # Profiles
    "get_profile",
    "list_profiles",
    "register_profile",
]
