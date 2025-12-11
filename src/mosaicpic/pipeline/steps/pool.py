"""Spatial pooling step for downsampling images.

This module provides PoolStep, which reduces image resolution by
aggregating blocks of pixels into single values.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from ..color_utils import lab_to_rgb, linear_to_rgb, rgb_to_lab, rgb_to_linear
from ..config import ColorSpace, PoolConfig, PoolMethod
from ..types import IndexMap, RGBImage, StepData

if TYPE_CHECKING:
    from ..context import PipelineContext


class PoolStep:
    """Spatial downsampling step.

    Reduces image/map resolution by aggregating blocks of cells
    into single values.

    Input: RGBImage OR IndexMap
    Output: Same type as input (with reduced dimensions)

    Behavior by input type:
        - RGBImage: Uses configured method (mean, median, etc.) in configured color space
        - IndexMap: Always uses mode (most common index) regardless of config

    Attributes:
        config: PoolConfig with output_size, method, and color_space.

    Example:
        >>> step = PoolStep(PoolConfig(output_size=(48, 48), method=PoolMethod.MEAN))
        >>> small_img = step.process(large_img, context)
    """

    def __init__(self, config: PoolConfig | None = None):
        """Initialize PoolStep with configuration.

        Args:
            config: PoolConfig with output_size, method, and color_space.
                Uses defaults if None.
        """
        self.config = config or PoolConfig()

    @property
    def input_types(self) -> tuple[type, ...]:
        """Types this step can accept as input."""
        return (RGBImage, IndexMap)

    def output_type_for_input(self, input_type: type) -> type:
        """Determine output type given an input type.

        PoolStep preserves input type (RGBImage -> RGBImage, IndexMap -> IndexMap).
        """
        if input_type == RGBImage:
            return RGBImage
        elif input_type == IndexMap:
            return IndexMap
        else:
            raise TypeError(f"PoolStep does not accept {input_type.__name__}")

    def process(self, input: StepData, context: "PipelineContext") -> StepData:
        """Pool input to smaller dimensions.

        For RGBImage: Aggregates pixel blocks using configured method.
        For IndexMap: Takes mode (most common index) of each block.

        Args:
            input: RGBImage or IndexMap to downsample.
            context: Pipeline context (used for target_size if output_size is None).

        Returns:
            Downsampled RGBImage or IndexMap.

        Raises:
            ValueError: If output dimensions are larger than input (upsampling).
            ValueError: If stride is not uniform (width ratio != height ratio).
        """
        output_size = self.config.output_size or context.target_size
        # output_size is (width, height), but arrays are (height, width)
        out_w, out_h = output_size

        if isinstance(input, RGBImage):
            return self._pool_rgb(input, out_h, out_w, context)
        elif isinstance(input, IndexMap):
            return self._pool_index(input, out_h, out_w, context)
        else:
            raise TypeError(f"Unexpected input type: {type(input)}")

    def _validate_dimensions(self, in_h: int, in_w: int, out_h: int, out_w: int) -> tuple[int, int]:
        """Validate dimensions and compute strides.

        Args:
            in_h: Input height.
            in_w: Input width.
            out_h: Output height.
            out_w: Output width.

        Returns:
            Tuple of (stride_h, stride_w).

        Raises:
            ValueError: If upsampling or non-uniform stride.
        """
        if out_h > in_h or out_w > in_w:
            raise ValueError(f"Cannot upsample: input ({in_w}×{in_h}) to output ({out_w}×{out_h})")

        stride_h = in_h // out_h
        stride_w = in_w // out_w

        if stride_h != stride_w:
            raise ValueError(
                f"Non-uniform stride: {stride_h} (height) vs {stride_w} (width). "
                f"Input ({in_w}×{in_h}) with output ({out_w}×{out_h})"
            )

        return stride_h, stride_w

    def _pool_rgb(
        self,
        input: RGBImage,
        out_h: int,
        out_w: int,
        context: "PipelineContext",  # noqa: ARG002
    ) -> RGBImage:
        """Pool RGB image using configured method and color space.

        Args:
            input: RGBImage to pool.
            out_h: Output height.
            out_w: Output width.
            context: Pipeline context.

        Returns:
            Pooled RGBImage.
        """
        in_h, in_w = input.height, input.width
        stride_h, stride_w = self._validate_dimensions(in_h, in_w, out_h, out_w)
        stride = stride_h  # Uniform stride

        data = input.data

        # Convert to working color space if needed
        if self.config.color_space == ColorSpace.LAB:
            working_data = rgb_to_lab(data)
        elif self.config.color_space == ColorSpace.LINEAR_RGB:
            working_data = rgb_to_linear(data)
        else:  # RGB
            working_data = data.astype(np.float64)

        # Initialize output
        output_data = np.zeros((out_h, out_w, 3), dtype=np.float64)

        # Process each output cell
        for cy in range(out_h):
            for cx in range(out_w):
                # Get block boundaries
                y_start = cy * stride
                y_end = min((cy + 1) * stride, in_h)
                x_start = cx * stride
                x_end = min((cx + 1) * stride, in_w)

                # Extract block
                block = working_data[y_start:y_end, x_start:x_end].reshape(-1, 3)

                # Aggregate based on method
                if self.config.method == PoolMethod.MEAN:
                    output_data[cy, cx] = np.mean(block, axis=0)
                elif self.config.method == PoolMethod.MEDIAN:
                    output_data[cy, cx] = np.median(block, axis=0)
                elif self.config.method == PoolMethod.MAX:
                    output_data[cy, cx] = np.max(block, axis=0)
                elif self.config.method == PoolMethod.MIN:
                    output_data[cy, cx] = np.min(block, axis=0)
                elif self.config.method == PoolMethod.MODE:
                    # For RGB mode, find most common color (exact match)
                    # Convert to tuple for counting
                    colors, inverse, counts = np.unique(
                        block.astype(np.int64), axis=0, return_inverse=True, return_counts=True
                    )
                    mode_idx = np.argmax(counts)
                    output_data[cy, cx] = colors[mode_idx]

        # Convert back to RGB if needed
        if self.config.color_space == ColorSpace.LAB:
            result_data = lab_to_rgb(output_data)
        elif self.config.color_space == ColorSpace.LINEAR_RGB:
            result_data = linear_to_rgb(output_data)
        else:  # RGB
            result_data = np.clip(output_data, 0, 255).astype(np.uint8)

        return RGBImage(data=result_data)

    def _pool_index(
        self,
        input: IndexMap,
        out_h: int,
        out_w: int,
        context: "PipelineContext",  # noqa: ARG002
    ) -> IndexMap:
        """Pool IndexMap using mode (most common index).

        Args:
            input: IndexMap to pool.
            out_h: Output height.
            out_w: Output width.
            context: Pipeline context.

        Returns:
            Pooled IndexMap with same palette reference.
        """
        in_h, in_w = input.height, input.width
        stride_h, stride_w = self._validate_dimensions(in_h, in_w, out_h, out_w)
        stride = stride_h  # Uniform stride

        data = input.data

        # Initialize output
        output_data = np.zeros((out_h, out_w), dtype=np.intp)

        # Process each output cell
        for cy in range(out_h):
            for cx in range(out_w):
                # Get block boundaries
                y_start = cy * stride
                y_end = min((cy + 1) * stride, in_h)
                x_start = cx * stride
                x_end = min((cx + 1) * stride, in_w)

                # Extract block and find mode
                block = data[y_start:y_end, x_start:x_end].flatten()
                mode_result = stats.mode(block, keepdims=False)
                output_data[cy, cx] = mode_result.mode

        return IndexMap(data=output_data, palette=input.palette)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PoolStep(config={self.config})"
