"""Cell model representing a single pixel/block in an image or canvas.

A Cell represents the smallest unit in both source images and target canvases.
It holds color information, optional position coordinates, and conversion metadata.
"""

from .color import Color


class Cell:
    """A single pixel/block unit with color and optional position.

    Cells are used to represent:
    - Individual pixels in a source Image
    - Individual tiles/blocks in a target Canvas

    Attributes:
        color: The Color of this cell.
        x: X-coordinate (column index), 0-indexed. None if not tied to a grid.
        y: Y-coordinate (row index), 0-indexed. None if not tied to a grid.
        pinned: If True, this cell is locked during re-conversion.
            Pinned cells preserve their color when the session reconverts.
        delta_e: Perceptual color distance (Delta E CIE2000) between this
            cell's color and the original image color at this position.
            Lower values indicate a better match. 0.0 if not computed.
    """

    def __init__(
        self,
        color: Color,
        x: int | None = None,
        y: int | None = None,
        pinned: bool = False,
        delta_e: float = 0.0,
    ):
        """Initialize a Cell with color and optional coordinates.

        Args:
            color (Color): The Color of this cell.
            x (int | None): Optional x-coordinate (column). 0-indexed from left.
            y (int | None): Optional y-coordinate (row). 0-indexed from top.
            pinned (bool): Whether this cell is pinned (locked during
                re-conversion).
            delta_e (float): Color distance to original image. Default 0.0.
        """
        self.color = color
        self.x = x
        self.y = y
        self.pinned = pinned
        self.delta_e = delta_e

    def __repr__(self) -> str:
        """Return string representation of the Cell."""
        parts = [f"color={self.color}", f"x={self.x}", f"y={self.y}"]
        if self.pinned:
            parts.append("pinned=True")
        if self.delta_e > 0:
            parts.append(f"delta_e={self.delta_e:.2f}")
        return f"Cell({', '.join(parts)})"
