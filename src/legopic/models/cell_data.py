"""Cell data model for grid export.

CellData is a lightweight data transfer object used when exporting
canvas data for building guide rendering. It contains all information
needed to render a single cell in the visual guide.

Attributes:
    x: X-coordinate (column), 0-indexed.
    y: Y-coordinate (row), 0-indexed.
    color: The Color of this cell.
    delta_e: Perceptual distance to original image color.
    pinned: True if this cell was manually pinned by user.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .color import Color


@dataclass
class CellData:
    """Cell data for grid export.

    A lightweight snapshot of a cell's state for building guide rendering.
    Unlike Cell, this is a pure data object with no methods.

    Attributes:
        x: X-coordinate (column), 0-indexed.
        y: Y-coordinate (row), 0-indexed.
        color: The Color of this cell.
        delta_e: Perceptual distance to original image color.
            Lower values indicate better color match.
        pinned: True if this cell was manually pinned by user.

    Example:
        >>> grid = session.get_grid_data()
        >>> for row in grid:
        ...     for cell in row:
        ...         print(f"({cell.x},{cell.y}): {cell.color.name}")
    """

    x: int
    y: int
    color: "Color"
    delta_e: float
    pinned: bool
