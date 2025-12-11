"""Canvas model representing the output tile mosaic grid.

A Canvas is the target grid where each cell represents a single tile/block.
It can be constructed empty or from processed image data.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .cell import Cell
from .color import Color


class Canvas:
    """Represents a tile mosaic canvas as a 2D grid of Cells.

    The Canvas is the output of the conversion process, where each Cell
    represents a single tile with its matched color.

    Attributes:
        width: Canvas width in tiles/cells.
        height: Canvas height in tiles/cells.
        cells: 2D list of Cells, indexed as cells[y][x].
    """

    def __init__(self, width: int, height: int):
        """Initialize an empty Canvas with the given dimensions.

        Args:
            width (int): Number of columns (studs horizontally).
            height (int): Number of rows (studs vertically).

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Canvas dimensions must be positive. Got width={width}, height={height}."
            )

        self.width = width
        self.height = height
        # Initialize with None colors (to be filled during conversion)
        self.cells: list[list[Cell]] = [
            [Cell(None, x, y) for x in range(width)]  # type: ignore[arg-type]
            for y in range(height)
        ]

    @classmethod
    def from_set(cls, set_id: str) -> "Canvas":
        """Create an empty Canvas with dimensions from a palette.

        Args:
            set_id (str): Palette identifier (e.g., "world_map_128x80").

        Returns:
            Canvas: A new empty Canvas with the palette's canvas dimensions.

        Raises:
            ValueError: If the set_id is not found in the data files.

        Example:
            >>> canvas = Canvas.from_set("world_map_128x80")
            >>> print(canvas)  # Canvas(width=128, height=80)
        """
        from ..data.loader import get_set_dimensions

        width, height = get_set_dimensions(set_id)
        return cls(width, height)

    @classmethod
    def from_cells(cls, cells: list[list[Cell]]) -> "Canvas":
        """Create a Canvas from a pre-built 2D list of Cells.

        Args:
            cells (list[list[Cell]]): 2D list of Cells, indexed as cells[y][x].
                All rows must have the same length.

        Returns:
            Canvas: A new Canvas instance with the provided cells.

        Raises:
            ValueError: If cells is empty or rows have inconsistent lengths.
        """
        if not cells or not cells[0]:
            raise ValueError("Cannot create Canvas from empty cell list.")

        height = len(cells)
        width = len(cells[0])

        # Validate all rows have same width
        for i, row in enumerate(cells):
            if len(row) != width:
                raise ValueError(
                    f"Inconsistent row lengths: row 0 has {width} cells, "
                    f"but row {i} has {len(row)} cells."
                )

        canvas = cls(width, height)
        canvas.cells = cells
        return canvas

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
                f"canvas of size ({self.width}, {self.height})."
            )
        return self.cells[y][x]

    def set_cell(self, x: int, y: int, color: Color) -> None:
        """Set the color of a Cell at a specific coordinate.

        Args:
            x (int): X-coordinate (column), 0-indexed.
            y (int): Y-coordinate (row), 0-indexed.
            color (Color): The Color to assign to the cell.

        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"Coordinates ({x}, {y}) out of bounds for "
                f"canvas of size ({self.width}, {self.height})."
            )
        self.cells[y][x] = Cell(color, x, y)

    def to_array(self) -> "NDArray[np.uint8]":
        """Convert the Canvas to a numpy RGB array.

        Returns:
            NDArray[np.uint8]: A numpy array of shape (height, width, 3) with
                dtype uint8, representing the canvas as an RGB image.

        Raises:
            ValueError: If any cell has no color assigned.
        """
        array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                if cell.color is None:
                    raise ValueError(
                        f"Cell at ({x}, {y}) has no color assigned. "
                        f"Canvas conversion may be incomplete."
                    )
                array[y, x, :] = cell.color.rgb

        return array

    def __repr__(self) -> str:
        """Return string representation of the Canvas."""
        return f"Canvas(width={self.width}, height={self.height})"
