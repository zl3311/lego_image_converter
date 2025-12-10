"""Dithering step for color quantization with error diffusion.

This module provides DitherStep, which maps pixels to palette colors
while propagating quantization error to neighboring pixels for
visually pleasing results on gradients and photos.

Supported Algorithms:
    - Floyd-Steinberg: Classic, balanced error diffusion
    - Atkinson: Lighter, high-contrast (loses 25% of error)
    - Jarvis-Judice-Ninke: Smoother gradients (larger kernel)
    - Stucki: High quality (similar to Jarvis)
    - Sierra: Similar to Jarvis
    - Sierra Lite: Fast approximation
    - Bayer: Ordered dithering (not error diffusion, parallelizable)
"""

from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import get_delta_e_matrix

from ..config import DitherAlgorithm, DitherConfig, ScanOrder
from ..types import IndexMap, RGBImage, StepData

if TYPE_CHECKING:
    from ..context import PipelineContext


# Error diffusion kernels
# Each kernel is a list of (dy, dx, weight) tuples where:
#   - dy: row offset from current pixel (positive = below)
#   - dx: column offset from current pixel (positive = right)
#   - weight: fraction of error to distribute
#
# For serpentine scanning (right-to-left rows), dx values are negated.

# Floyd-Steinberg (total = 16)
#       [ * ]  7/16
# 3/16  5/16  1/16
_FLOYD_STEINBERG = [
    (0, 1, 7 / 16),
    (1, -1, 3 / 16),
    (1, 0, 5 / 16),
    (1, 1, 1 / 16),
]

# Atkinson (total = 8, only 6/8 diffused - intentionally loses 25%)
#       [ * ]  1/8  1/8
# 1/8   1/8   1/8
#       1/8
_ATKINSON = [
    (0, 1, 1 / 8),
    (0, 2, 1 / 8),
    (1, -1, 1 / 8),
    (1, 0, 1 / 8),
    (1, 1, 1 / 8),
    (2, 0, 1 / 8),
]

# Jarvis-Judice-Ninke (total = 48)
#             [ * ]  7/48  5/48
# 3/48  5/48  7/48  5/48  3/48
# 1/48  3/48  5/48  3/48  1/48
_JARVIS = [
    (0, 1, 7 / 48),
    (0, 2, 5 / 48),
    (1, -2, 3 / 48),
    (1, -1, 5 / 48),
    (1, 0, 7 / 48),
    (1, 1, 5 / 48),
    (1, 2, 3 / 48),
    (2, -2, 1 / 48),
    (2, -1, 3 / 48),
    (2, 0, 5 / 48),
    (2, 1, 3 / 48),
    (2, 2, 1 / 48),
]

# Stucki (total = 42)
#             [ * ]  8/42  4/42
# 2/42  4/42  8/42  4/42  2/42
# 1/42  2/42  4/42  2/42  1/42
_STUCKI = [
    (0, 1, 8 / 42),
    (0, 2, 4 / 42),
    (1, -2, 2 / 42),
    (1, -1, 4 / 42),
    (1, 0, 8 / 42),
    (1, 1, 4 / 42),
    (1, 2, 2 / 42),
    (2, -2, 1 / 42),
    (2, -1, 2 / 42),
    (2, 0, 4 / 42),
    (2, 1, 2 / 42),
    (2, 2, 1 / 42),
]

# Sierra (total = 32)
#             [ * ]  5/32  3/32
# 2/32  4/32  5/32  4/32  2/32
#       2/32  3/32  2/32
_SIERRA = [
    (0, 1, 5 / 32),
    (0, 2, 3 / 32),
    (1, -2, 2 / 32),
    (1, -1, 4 / 32),
    (1, 0, 5 / 32),
    (1, 1, 4 / 32),
    (1, 2, 2 / 32),
    (2, -1, 2 / 32),
    (2, 0, 3 / 32),
    (2, 1, 2 / 32),
]

# Sierra Lite (total = 4)
#       [ * ]  2/4
# 1/4   1/4
_SIERRA_LITE = [
    (0, 1, 2 / 4),
    (1, -1, 1 / 4),
    (1, 0, 1 / 4),
]

# Mapping from algorithm enum to kernel
_KERNELS: dict[DitherAlgorithm, list[tuple[int, int, float]]] = {
    DitherAlgorithm.FLOYD_STEINBERG: _FLOYD_STEINBERG,
    DitherAlgorithm.ATKINSON: _ATKINSON,
    DitherAlgorithm.JARVIS_JUDICE_NINKE: _JARVIS,
    DitherAlgorithm.STUCKI: _STUCKI,
    DitherAlgorithm.SIERRA: _SIERRA,
    DitherAlgorithm.SIERRA_LITE: _SIERRA_LITE,
}

# 4×4 Bayer matrix for ordered dithering
_BAYER_4X4 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ]
    )
    / 16.0
    - 0.5
)


class DitherStep:
    """Quantize RGB image with error diffusion dithering.

    Maps pixels to palette colors while propagating quantization error
    to neighboring pixels. This creates visual blending that better
    represents gradients and subtle color variations.

    Input: RGBImage
    Output: IndexMap

    The algorithm processes pixels in scan order, and for each pixel:
    1. Add accumulated error from previous pixels
    2. Find nearest palette color
    3. Compute error (difference between desired and actual color)
    4. Distribute error to neighboring pixels according to kernel

    Attributes:
        config: DitherConfig with algorithm, order, strength, and metric.

    Example:
        >>> step = DitherStep(DitherConfig(
        ...     algorithm=DitherAlgorithm.FLOYD_STEINBERG,
        ...     order=ScanOrder.SERPENTINE
        ... ))
        >>> index_map = step.process(rgb_image, context)
    """

    def __init__(self, config: DitherConfig | None = None):
        """Initialize DitherStep with configuration.

        Args:
            config: DitherConfig with algorithm, order, strength, and metric.
                Uses defaults if None.
        """
        self.config = config or DitherConfig()

    @property
    def input_types(self) -> tuple[type, ...]:
        """Types this step can accept as input."""
        return (RGBImage,)

    def output_type_for_input(self, input_type: type) -> type:
        """Determine output type given an input type.

        DitherStep always outputs IndexMap from RGBImage input.
        """
        if input_type == RGBImage:
            return IndexMap
        else:
            raise TypeError(f"DitherStep does not accept {input_type.__name__}")

    def process(self, input: StepData, context: "PipelineContext") -> IndexMap:
        """Quantize RGB image with error diffusion.

        Args:
            input: RGBImage to quantize.
            context: Pipeline context with palette.

        Returns:
            IndexMap with same dimensions as input.
        """
        if not isinstance(input, RGBImage):
            raise TypeError(f"Expected RGBImage, got {type(input).__name__}")

        if self.config.algorithm == DitherAlgorithm.BAYER:
            return self._bayer_dither(input, context)
        else:
            return self._error_diffusion_dither(input, context)

    def _error_diffusion_dither(self, input: RGBImage, context: "PipelineContext") -> IndexMap:
        """Apply error diffusion dithering.

        Args:
            input: RGBImage to dither.
            context: Pipeline context with palette.

        Returns:
            IndexMap with dithered result.
        """
        height, width = input.height, input.width
        palette_rgb = context.palette.to_rgb_array()

        # Work with float for error accumulation
        # Use float64 for precision
        working = input.data.astype(np.float64)

        # Output indices
        output = np.zeros((height, width), dtype=np.intp)

        # Get kernel for this algorithm
        kernel = _KERNELS[self.config.algorithm]

        # Error buffer - store errors for the full image
        # Using full image buffer for simplicity; could optimize for memory
        error_buffer = np.zeros((height, width, 3), dtype=np.float64)

        # Process rows
        for y in range(height):
            # Determine scan direction for this row
            if self.config.order == ScanOrder.SERPENTINE and y % 2 == 1:
                # Right to left for odd rows
                x_range = range(width - 1, -1, -1)
                direction = -1
            else:
                # Left to right for even rows (and all rows in raster mode)
                x_range = range(width)
                direction = 1

            for x in x_range:
                # Get target color (original + accumulated error)
                target = working[y, x] + error_buffer[y, x]

                # Clamp to valid range
                target_clamped = np.clip(target, 0, 255)

                # Find nearest palette color
                target_2d = target_clamped.reshape(1, 3).astype(np.uint8)
                distances = get_delta_e_matrix(target_2d, palette_rgb)
                best_idx = int(np.argmin(distances))

                # Store result
                output[y, x] = best_idx

                # Compute error
                actual = palette_rgb[best_idx].astype(np.float64)
                error = target_clamped - actual

                # Apply strength
                error = error * self.config.strength

                # Distribute error to neighbors
                for dy, dx, weight in kernel:
                    # Adjust dx for scan direction
                    actual_dx = dx * direction
                    ny, nx = y + dy, x + actual_dx

                    # Check bounds
                    if 0 <= ny < height and 0 <= nx < width:
                        error_buffer[ny, nx] += error * weight

        return IndexMap(data=output, palette=context.palette)

    def _bayer_dither(self, input: RGBImage, context: "PipelineContext") -> IndexMap:
        """Apply Bayer ordered dithering.

        Bayer dithering uses a threshold matrix to perturb pixel values
        before quantization. Unlike error diffusion, it can be parallelized.

        Args:
            input: RGBImage to dither.
            context: Pipeline context with palette.

        Returns:
            IndexMap with dithered result.
        """
        height, width = input.height, input.width
        palette_rgb = context.palette.to_rgb_array()

        # Work with float
        working = input.data.astype(np.float64)

        # Create threshold matrix tiled to image size
        # Bayer matrix values are in [-0.5, 0.5], scale by color range
        bayer_h, bayer_w = _BAYER_4X4.shape
        threshold = np.tile(_BAYER_4X4, (height // bayer_h + 1, width // bayer_w + 1))[
            :height, :width
        ]

        # Apply threshold (scale by 255 for full color range, modulated by strength)
        threshold_scaled = threshold * 255.0 * self.config.strength

        # Add threshold to each channel
        adjusted = working + threshold_scaled[:, :, np.newaxis]

        # Clamp to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        # Flatten for batch quantization
        pixels = adjusted.reshape(-1, 3)

        # Find nearest palette color for each pixel
        distances = get_delta_e_matrix(pixels, palette_rgb)
        best_indices = np.argmin(distances, axis=1)

        # Reshape to image dimensions
        output = best_indices.reshape(height, width).astype(np.intp)

        return IndexMap(data=output, palette=context.palette)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DitherStep(config={self.config})"
