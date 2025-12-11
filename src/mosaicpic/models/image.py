"""Image model for representing input images to be converted.

This module provides the Image class which wraps a numpy array representation
of an image and provides convenient access methods.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .cell import Cell
from .color import Color


class Image:
    """Represents an input image as a grid of Cells.

    The Image class wraps a numpy array and provides cell-based access
    for downstream processing (downsizing, color matching).

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        cells: List of Cell objects representing each pixel.

    Note:
        Coordinate system: (0, 0) is top-left corner.
        x increases rightward (columns), y increases downward (rows).
    """

    def __init__(self, input_image: "NDArray[np.uint8]"):
        """Initialize an Image from a numpy array.

        Args:
            input_image (NDArray[np.uint8]): A 3D numpy array of shape
                (height, width, 3) with dtype uint8, representing RGB
                pixel values.

        Raises:
            ValueError: If the input array has invalid shape or dtype.
        """
        # Validate input array
        if not isinstance(input_image, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray, got {type(input_image).__name__}.")

        if input_image.ndim != 3:
            raise ValueError(
                f"Expected 3D array (height, width, channels), "
                f"got {input_image.ndim}D array with shape {input_image.shape}."
            )

        if input_image.shape[2] != 3:
            raise ValueError(
                f"Expected 3 color channels (RGB), got {input_image.shape[2]} channels."
            )

        self._array = input_image
        self.height, self.width = input_image.shape[:2]

        # Build cells list: iterate row by row (y), then column by column (x)
        self.cells = [
            Cell(Color(tuple(input_image[y, x, :])), x, y)
            for y in range(self.height)
            for x in range(self.width)
        ]

    @classmethod
    def from_file(cls, filepath: str) -> "Image":
        """Load an Image from a local file.

        Args:
            filepath (str): Path to the image file. Supports common formats
                (JPEG, PNG, BMP, etc.) via PIL.

        Returns:
            Image: A new Image instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read as an image.
        """
        from pathlib import Path

        from PIL import Image as PILImage

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")

        try:
            pil_image = PILImage.open(filepath).convert("RGB")
            array = np.array(pil_image, dtype=np.uint8)
            return cls(array)
        except Exception as e:
            raise ValueError(f"Failed to load image from {filepath}: {e}") from e

    @classmethod
    def from_url(cls, url: str) -> "Image":
        """Load an Image from a URL.

        Args:
            url (str): URL pointing to an image file.

        Returns:
            Image: A new Image instance.

        Raises:
            ValueError: If the URL cannot be fetched or parsed as an image.
        """
        import requests
        from PIL import Image as PILImage

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            pil_image = PILImage.open(response.raw).convert("RGB")  # type: ignore[arg-type]
            array = np.array(pil_image, dtype=np.uint8)
            return cls(array)
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch image from URL: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse image from URL: {e}") from e

    def to_array(self) -> "NDArray[np.uint8]":
        """Return the underlying numpy array.

        Returns:
            NDArray[np.uint8]: The RGB image as a numpy array of shape
                (height, width, 3).
        """
        return self._array.copy()  # type: ignore[no-any-return]

    def get_cell(self, x: int, y: int) -> Cell:
        """Get the Cell at a specific coordinate.

        Args:
            x (int): X-coordinate (column), 0-indexed.
            y (int): Y-coordinate (row), 0-indexed.

        Returns:
            Cell: The Cell at position (x, y).

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"Coordinates ({x}, {y}) out of bounds for "
                f"image of size ({self.width}, {self.height})."
            )
        return self.cells[y * self.width + x]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """Return string representation of the Image."""
        return f"Image(width={self.width}, height={self.height})"
