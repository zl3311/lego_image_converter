"""Configuration classes for pipeline steps.

This module provides enums and dataclasses for configuring pipeline steps:
- PoolMethod: Aggregation method for pooling (mean, median, mode, etc.)
- ColorSpace: Color space for RGB pooling operations
- DitherAlgorithm: Error diffusion algorithm for dithering
- ScanOrder: Pixel scanning order for error diffusion
- PoolConfig, QuantizeConfig, DitherConfig: Step configuration dataclasses
"""

from dataclasses import dataclass
from enum import Enum


class PoolMethod(Enum):
    """Aggregation method for pooling operations.

    Attributes:
        MEAN: Average of values (best for smooth gradients)
        MEDIAN: Median of values (robust to outliers)
        MODE: Most common value (for IndexMap, preserves edges)
        MAX: Maximum value
        MIN: Minimum value
    """

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    MAX = "max"
    MIN = "min"


class ColorSpace(Enum):
    """Color space for RGB pooling operations.

    Attributes:
        RGB: Standard sRGB (fast, simple)
        LAB: CIE Lab (perceptually uniform, better for averaging)
        LINEAR_RGB: Linear RGB (gamma-corrected, physically accurate)
    """

    RGB = "rgb"
    LAB = "lab"
    LINEAR_RGB = "linear_rgb"


class DitherAlgorithm(Enum):
    """Error diffusion algorithm for dithering.

    Attributes:
        FLOYD_STEINBERG: Classic, balanced (7/16 diffusion kernel)
        ATKINSON: Lighter, high-contrast (loses 25% of error intentionally)
        JARVIS_JUDICE_NINKE: Smoother gradients (larger kernel)
        STUCKI: High quality (similar to Jarvis)
        SIERRA: Similar to Jarvis
        SIERRA_LITE: Fast approximation (small kernel)
        BAYER: Ordered dithering (not error diffusion, parallelizable)
    """

    FLOYD_STEINBERG = "floyd_steinberg"
    ATKINSON = "atkinson"
    JARVIS_JUDICE_NINKE = "jarvis"
    STUCKI = "stucki"
    SIERRA = "sierra"
    SIERRA_LITE = "sierra_lite"
    BAYER = "bayer"


class ScanOrder(Enum):
    """Pixel scanning order for error diffusion.

    Attributes:
        RASTER: Left-to-right, top-to-bottom (may cause drift artifacts)
        SERPENTINE: Alternating direction each row (reduces drift)
    """

    RASTER = "raster"
    SERPENTINE = "serpentine"


@dataclass
class PoolConfig:
    """Configuration for PoolStep.

    Attributes:
        output_size: Target dimensions as (width, height).
            If None, uses context.target_size at runtime.
        method: Aggregation method for RGB pooling.
            Ignored for IndexMap (always uses mode).
        color_space: Color space for RGB averaging.
            Only applies when method is MEAN or MEDIAN.

    Example:
        >>> config = PoolConfig(
        ...     output_size=(48, 48),
        ...     method=PoolMethod.MEAN,
        ...     color_space=ColorSpace.LAB
        ... )
    """

    output_size: tuple[int, int] | None = None
    method: PoolMethod = PoolMethod.MEAN
    color_space: ColorSpace = ColorSpace.RGB


@dataclass
class QuantizeConfig:
    """Configuration for QuantizeStep.

    Attributes:
        metric: Color distance metric for matching.
            "delta_e": CIE2000 Delta E (perceptually accurate, default)

    Example:
        >>> config = QuantizeConfig(metric="delta_e")
    """

    metric: str = "delta_e"


@dataclass
class DitherConfig:
    """Configuration for DitherStep.

    Attributes:
        algorithm: Error diffusion algorithm to use.
        order: Pixel scanning order.
        strength: Error diffusion strength (0.0 to 1.0).
            1.0 = full error propagation (default)
            0.5 = half error propagation (lighter dithering)
            0.0 = no error propagation (same as QuantizeStep)
        metric: Color distance metric for matching.

    Example:
        >>> config = DitherConfig(
        ...     algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        ...     order=ScanOrder.SERPENTINE,
        ...     strength=1.0
        ... )
    """

    algorithm: DitherAlgorithm = DitherAlgorithm.FLOYD_STEINBERG
    order: ScanOrder = ScanOrder.SERPENTINE
    strength: float = 1.0
    metric: str = "delta_e"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0.0, 1.0], got {self.strength}")
