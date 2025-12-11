"""Bill of Materials entry model for building guide export.

A BOMEntry represents a single color entry in the bill of materials,
tracking how many tiles of that color are needed and whether the color
is available in the selected palette.

Attributes:
    color: The Color used.
    count_needed: Number of tiles needed for this color.
    in_palette: True if color is in palette, False if custom.
    elements: List of Element variants available for this color.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .color import Color
    from .element import Element


@dataclass
class BOMEntry:
    """Bill of Materials entry for a single color.

    Used when exporting building guide data. Each entry represents
    one color used in the canvas with its required count.

    Attributes:
        color: The Color used.
        count_needed: Number of tiles needed for this color.
        in_palette: True if color is in the palette, False if user
            pinned an out-of-palette color (requires manual sourcing).
        elements: List of Element variants available for this color.
            Empty if color is out-of-palette.

    Example:
        >>> bom = session.get_bill_of_materials()
        >>> for entry in bom:
        ...     status = "✓" if entry.in_palette else "⚠ custom"
        ...     print(f"{entry.color.name}: {entry.count_needed} {status}")
    """

    color: "Color"
    count_needed: int
    in_palette: bool
    elements: list["Element"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize elements to empty list if None."""
        if self.elements is None:
            self.elements = []
