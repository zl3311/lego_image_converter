"""Quantization step for color matching to palette.

This module provides QuantizeStep, which maps each pixel to the
nearest palette color without error diffusion.
"""

from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import get_delta_e_matrix

from ..config import QuantizeConfig
from ..types import IndexMap, RGBImage, StepData

if TYPE_CHECKING:
    from ..context import PipelineContext


class QuantizeStep:
    """Quantize RGB image to palette indices.

    Maps each pixel to the nearest palette color using the configured
    color distance metric. No error diffusion — each pixel is processed
    independently.

    Input: RGBImage
    Output: IndexMap

    Attributes:
        config: QuantizeConfig with metric setting.

    Example:
        >>> step = QuantizeStep(QuantizeConfig(metric="delta_e"))
        >>> index_map = step.process(rgb_image, context)
    """

    def __init__(self, config: QuantizeConfig | None = None):
        """Initialize QuantizeStep with configuration.

        Args:
            config: QuantizeConfig with metric setting.
                Uses defaults if None.
        """
        self.config = config or QuantizeConfig()

    @property
    def input_types(self) -> tuple[type, ...]:
        """Types this step can accept as input."""
        return (RGBImage,)

    def output_type_for_input(self, input_type: type) -> type:
        """Determine output type given an input type.

        QuantizeStep always outputs IndexMap from RGBImage input.
        """
        if input_type == RGBImage:
            return IndexMap
        else:
            raise TypeError(f"QuantizeStep does not accept {input_type.__name__}")

    def process(self, input: StepData, context: "PipelineContext") -> IndexMap:
        """Quantize RGB image to palette indices.

        Args:
            input: RGBImage to quantize.
            context: Pipeline context with palette.

        Returns:
            IndexMap with same dimensions as input.
        """
        if not isinstance(input, RGBImage):
            raise TypeError(f"Expected RGBImage, got {type(input).__name__}")

        height, width = input.height, input.width
        palette_rgb = context.palette.to_rgb_array()

        # Flatten image to (N, 3) for batch processing
        pixels = input.data.reshape(-1, 3)

        # Compute distances from each pixel to each palette color
        # get_delta_e_matrix returns shape (n_pixels, n_palette)
        distances = get_delta_e_matrix(pixels, palette_rgb)

        # Find index of closest palette color for each pixel
        best_indices = np.argmin(distances, axis=1)

        # Reshape to (height, width)
        index_data = best_indices.reshape(height, width).astype(np.intp)

        return IndexMap(data=index_data, palette=context.palette)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"QuantizeStep(config={self.config})"
