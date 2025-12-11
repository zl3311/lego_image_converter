"""Pipeline step implementations.

This module exports all available pipeline steps:
- PoolStep: Spatial downsampling
- QuantizeStep: Color quantization without error diffusion
- DitherStep: Color quantization with error diffusion
"""

from .base import Step
from .dither import DitherStep
from .pool import PoolStep
from .quantize import QuantizeStep

__all__ = [
    "Step",
    "PoolStep",
    "QuantizeStep",
    "DitherStep",
]
