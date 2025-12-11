"""Pipeline context for shared state across steps.

This module provides PipelineContext, which holds shared state
that is passed to every step's process() method.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Palette
    from .types import RGBImage


@dataclass
class PipelineContext:
    """Shared context available to all pipeline steps.

    This is passed to every step's process() method and contains
    information that may be needed across multiple steps.

    Attributes:
        palette: Available colors for quantization/dithering steps.
        target_size: Final canvas size as (width, height) in cells.
        original_image: Reference to original input for delta_e calculation.
    """

    palette: "Palette"
    target_size: tuple[int, int]  # (width, height)
    original_image: "RGBImage | None" = None
