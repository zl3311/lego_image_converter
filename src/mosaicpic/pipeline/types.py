"""Data types for the pipeline system.

This module defines the typed data containers that flow through pipeline steps:
- RGBImage: Full RGB color data at any resolution
- IndexMap: Quantized palette indices at any resolution
- StepData: Union type for step I/O

These types enforce type-safe step composition at pipeline construction time.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models import Palette


@dataclass
class RGBImage:
    """RGB image data at any resolution.

    Attributes:
        data: Numpy array of shape (height, width, 3) with dtype uint8.
            Values are in range [0, 255] for each RGB channel.

    Properties:
        height: Image height in pixels.
        width: Image width in pixels.
        shape: Tuple of (height, width).

    Example:
        >>> img = RGBImage(data=np.zeros((100, 100, 3), dtype=np.uint8))
        >>> img.height
        100
        >>> img.shape
        (100, 100)
    """

    data: "NDArray[np.uint8]"  # Shape: (H, W, 3)

    def __post_init__(self) -> None:
        """Validate data shape and dtype."""
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {self.data.ndim}D")
        if self.data.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {self.data.shape[2]}")
        if self.data.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {self.data.dtype}")

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Image dimensions as (height, width)."""
        return (self.height, self.width)


@dataclass
class IndexMap:
    """Palette index map (quantized colors).

    Each cell contains an index into the associated Palette's color list.
    This is the output of quantization/dithering steps.

    Attributes:
        data: Numpy array of shape (height, width) with integer dtype.
            Each value is an index into palette.colors.
        palette: The Palette these indices reference.

    Properties:
        height: Map height in cells.
        width: Map width in cells.
        shape: Tuple of (height, width).

    Methods:
        to_rgb(): Convert back to RGBImage using palette colors.

    Example:
        >>> index_map = IndexMap(data=np.zeros((48, 48), dtype=np.intp), palette=my_palette)
        >>> rgb = index_map.to_rgb()  # Convert to viewable image
    """

    data: "NDArray[np.intp]"  # Shape: (H, W)
    palette: "Palette"

    def __post_init__(self) -> None:
        """Validate data shape and dtype."""
        if self.data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {self.data.ndim}D")
        if not np.issubdtype(self.data.dtype, np.integer):
            raise ValueError(f"Expected integer dtype, got {self.data.dtype}")

    @property
    def height(self) -> int:
        """Map height in cells."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Map width in cells."""
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Map dimensions as (height, width)."""
        return (self.height, self.width)

    def to_rgb(self) -> RGBImage:
        """Convert index map to RGB image using palette colors.

        Returns:
            RGBImage with the same dimensions, where each cell's
            index is replaced with the corresponding palette color.
        """
        palette_rgb = self.palette.to_rgb_array()
        rgb_data = palette_rgb[self.data]
        return RGBImage(data=rgb_data.astype(np.uint8))


# Union type for step I/O
StepData = RGBImage | IndexMap
