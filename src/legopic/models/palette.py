"""Palette model for managing available Lego colors.

A Palette represents the set of colors available for matching,
typically derived from a specific Lego set's included pieces.

The Palette stores a mapping from each unique Color to its available
Element variants, supporting both color matching (needs unique colors)
and inventory tracking (needs element details).
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .color import Color
from .element import Element


class Palette:
    """A collection of available colors for matching.

    The Palette holds the colors that can be used when converting an image
    to a Lego mosaic. Typically populated from a Lego set's element list.

    The internal storage is a dict mapping Color -> list[Element], where:
    - Each Color key is unique by RGB
    - Each Element list contains all variants available for that color

    Attributes:
        _color_elements: Internal dict mapping Color to list of Elements.

    Properties:
        colors: List of unique Color objects (for color matching).
        elements: Flat list of all Element objects (for inventory queries).
    """

    def __init__(self, colors: list[Color] | dict[Color, list[Element]]):
        """Initialize a Palette with colors.

        Args:
            colors (list[Color] | dict[Color, list[Element]]): Either:

                - A list of Color objects (Elements will be empty lists)
                - A dict mapping Color -> list[Element] (full element info)

        Raises:
            ValueError: If colors is empty.
        """
        if isinstance(colors, dict):
            if not colors:
                raise ValueError("Palette must contain at least one color.")
            self._color_elements: dict[Color, list[Element]] = colors
        else:
            # List of Colors - convert to dict with empty element lists
            if not colors:
                raise ValueError("Palette must contain at least one color.")
            self._color_elements = {color: [] for color in colors}

    @classmethod
    def from_set(cls, set_id: int | None = None, standard_only: bool = True) -> "Palette":
        """Create a Palette from a Lego set ID or all available colors.

        Args:
            set_id (int | None): The Lego set identifier (e.g., 31197). If None,
                loads all colors from the color database.
            standard_only (bool): If True and set_id is None, only include
                standard (opaque) colors, excluding transparent/metallic/glow
                variants. Ignored when set_id is specified. Default True.

        Returns:
            Palette: A Palette containing the colors available.

        Raises:
            ValueError: If the set_id is not found in the data files.

        Example:
            >>> # Load colors from a specific set
            >>> palette = Palette.from_set(31197)  # Andy Warhol set
            >>> # Load all standard colors
            >>> palette = Palette.from_set()  # All standard colors
            >>> # Load all colors including transparent/metallic
            >>> palette = Palette.from_set(standard_only=False)
        """
        from ..data.loader import get_all_colors, get_colors_for_set

        if set_id is not None:
            # Load from specific set - ignore standard_only
            color_elements = get_colors_for_set(set_id)
        else:
            # Load all colors, optionally filtered by standard_only
            color_elements = get_all_colors(standard_only=standard_only)

        return cls(color_elements)

    @property
    def colors(self) -> list[Color]:
        """Get the list of unique colors in the palette.

        Returns:
            list[Color]: List of Color objects, one per unique RGB value.
        """
        return list(self._color_elements.keys())

    @property
    def elements(self) -> list[Element]:
        """Get all elements across all colors.

        Returns:
            list[Element]: Flat list of all Element objects in the palette.
        """
        all_elements = []
        for elem_list in self._color_elements.values():
            all_elements.extend(elem_list)
        return all_elements

    def get_elements_for_color(self, color: Color) -> list[Element]:
        """Get the elements (variants) available for a specific color.

        Args:
            color (Color): The Color to look up.

        Returns:
            list[Element]: List of Element objects for that color, or empty
                list if the color is not in the palette.
        """
        return self._color_elements.get(color, [])

    def to_rgb_array(self) -> "NDArray[np.uint8]":
        """Convert palette colors to a numpy array.

        Returns:
            NDArray[np.uint8]: A numpy array of shape (n_colors, 3) with
                dtype uint8, where each row is an RGB color.
        """
        return np.array([color.rgb for color in self.colors], dtype=np.uint8)

    def to_rgb_list(self) -> list[tuple[int, int, int]]:
        """Get palette colors as a list of RGB tuples.

        Returns:
            list[tuple[int, int, int]]: List of (r, g, b) tuples for each
                color in the palette.
        """
        return [color.rgb for color in self.colors]

    def __len__(self) -> int:
        """Return the number of unique colors in the palette."""
        return len(self._color_elements)

    def __iter__(self) -> "Iterator[Color]":
        """Iterate over unique colors in the palette."""
        return iter(self._color_elements.keys())

    def __contains__(self, color: Color) -> bool:
        """Check if a color is in the palette."""
        return color in self._color_elements

    def __repr__(self) -> str:
        """Return string representation of the Palette."""
        n_elements = sum(len(e) for e in self._color_elements.values())
        if n_elements > 0:
            return f"Palette({len(self)} colors, {n_elements} elements)"
        return f"Palette({len(self)} colors)"
