"""Data loader for LEGO color and set information.

This module provides functions to load and parse the CSV data files containing
LEGO color definitions, set information, and element mappings.

Data Files:
    - colors.csv: Maps element_id → color name, RGB, variant_id, is_standard.
    - sets.csv: Maps set_id → set name, canvas dimensions.
    - elements.csv: Maps set_id + element_id → piece count.

Example:
    >>> from legopic.data.loader import get_colors_for_set, get_all_colors
    >>> palette_data = get_colors_for_set(31197)  # Andy Warhol set
    >>> all_standard = get_all_colors(standard_only=True)

Note:
    Raw CSV data is cached at module level after first load. The high-level
    functions build on the cached data to avoid repeated file I/O.
"""

import csv
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.color import Color
    from ..models.element import Element

# Type aliases for raw data structures
ColorRawData = dict[str, Any]
SetRawData = dict[str, Any]
ElementRawData = dict[str, Any]


def _get_data_path(filename: str) -> Path:
    """Get the absolute path to a data file.

    Args:
        filename (str): Name of the file in the data directory
            (e.g., "colors.csv").

    Returns:
        Path: Absolute Path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    data_dir = Path(__file__).parent
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return file_path


@functools.cache
def _load_colors_raw() -> dict[int, ColorRawData]:
    """Load colors.csv and return raw color data.

    Returns:
        Dict mapping element_id to color info:
        {
            element_id: {
                "name": str,
                "design_id": int,
                "variant_id": int,
                "r": int,
                "g": int,
                "b": int,
                "is_standard": bool
            }
        }

    Note:
        Results are cached after first call.
    """
    path = _get_data_path("colors.csv")
    colors = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            element_id = int(row["element_id"])
            colors[element_id] = {
                "name": row["name"],
                "design_id": int(row["design_id"]),
                "variant_id": int(row["variant_id"]),
                "r": int(row["r"]),
                "g": int(row["g"]),
                "b": int(row["b"]),
                "is_standard": row["is_standard"].lower() == "true",
            }

    return colors


@functools.cache
def _load_sets_raw() -> dict[int, SetRawData]:
    """Load sets.csv and return raw set data.

    Returns:
        Dict mapping set_id to set info:
        {
            set_id: {
                "name": str,
                "canvas_width": int,
                "canvas_height": int
            }
        }

    Note:
        Results are cached after first call.
    """
    path = _get_data_path("sets.csv")
    sets = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            set_id = int(row["set_id"])
            sets[set_id] = {
                "name": row["name"],
                "canvas_width": int(row["canvas_width"]),
                "canvas_height": int(row["canvas_height"]),
            }

    return sets


@functools.cache
def _load_elements_raw() -> dict[int, list[ElementRawData]]:
    """Load elements.csv and return raw element data grouped by set.

    Returns:
        Dict mapping set_id to list of element info:
        {
            set_id: [
                {
                    "element_id": int,
                    "design_id": int,
                    "count": int
                },
                ...
            ]
        }

    Note:
        Results are cached after first call.
    """
    path = _get_data_path("elements.csv")
    elements: dict[int, list[ElementRawData]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            set_id = int(row["set_id"])
            if set_id not in elements:
                elements[set_id] = []
            elements[set_id].append(
                {
                    "element_id": int(row["element_id"]),
                    "design_id": int(row["design_id"]),
                    "count": int(row["count"]),
                }
            )

    return elements


def get_colors_for_set(set_id: int) -> dict["Color", list["Element"]]:
    """Get all colors available in a specific LEGO set.

    Joins elements.csv with colors.csv to build a mapping from each unique
    Color to its available Element variants within the set.

    Args:
        set_id (int): LEGO set identifier (e.g., 31197).

    Returns:
        dict[Color, list[Element]]: Dict mapping Color objects to list of
            Element objects. Each Color key is unique by RGB, and its value
            contains the Element(s) available in this set for that color.

    Raises:
        ValueError: If the set_id is not found in the data files.

    Example:
        >>> colors = get_colors_for_set(31197)
        >>> for color, elements in colors.items():
        ...     print(f"{color.name}: {len(elements)} variant(s)")
    """
    # Import here to avoid circular imports
    from ..models.color import Color
    from ..models.element import Element

    elements_raw = _load_elements_raw()
    colors_raw = _load_colors_raw()

    if set_id not in elements_raw:
        available = sorted(elements_raw.keys())
        raise ValueError(f"Set ID {set_id} not found. Available sets: {available}")

    # Build Color -> [Element] mapping
    result: dict[Color, list[Element]] = {}

    for elem_data in elements_raw[set_id]:
        element_id = elem_data["element_id"]

        if element_id not in colors_raw:
            # Skip elements not in colors.csv (shouldn't happen with clean data)
            continue

        color_data = colors_raw[element_id]

        # Create Color object (will be deduplicated by __eq__ based on RGB)
        color = Color(
            rgb=(color_data["r"], color_data["g"], color_data["b"]), name=color_data["name"]
        )

        # Create Element object
        element = Element(
            element_id=element_id,
            design_id=elem_data["design_id"],
            variant_id=color_data["variant_id"],
            count=elem_data["count"],
        )

        # Add to result, grouping by Color
        if color not in result:
            result[color] = []
        result[color].append(element)

    return result


def get_all_colors(standard_only: bool = True) -> dict["Color", list["Element"]]:
    """Get all available colors from the color database.

    Returns all colors from colors.csv, optionally filtered to standard
    (opaque) colors only. Colors are deduplicated by RGB, with all variants
    of each color grouped together.

    Args:
        standard_only (bool): If True, only include colors where is_standard
            is true (opaque colors, excluding transparent/metallic/glow).
            Default True.

    Returns:
        dict[Color, list[Element]]: Dict mapping Color objects to list of
            Element objects. Each Color key is unique by RGB, and its value
            contains all Element variants available for that color.

    Example:
        >>> standard_colors = get_all_colors(standard_only=True)
        >>> all_colors = get_all_colors(standard_only=False)
    """
    # Import here to avoid circular imports
    from ..models.color import Color
    from ..models.element import Element

    colors_raw = _load_colors_raw()

    # Build Color -> [Element] mapping
    result: dict[Color, list[Element]] = {}

    # Track which color names we've seen for standard colors
    # When deduplicating, prefer standard color names
    color_names_by_rgb: dict[tuple[int, int, int], str] = {}

    # First pass: collect standard color names for each RGB
    if not standard_only:
        for _element_id, color_data in colors_raw.items():
            if color_data["is_standard"]:
                rgb = (color_data["r"], color_data["g"], color_data["b"])
                if rgb not in color_names_by_rgb:
                    color_names_by_rgb[rgb] = color_data["name"]

    # Second pass: build the result
    for element_id, color_data in colors_raw.items():
        # Filter by standard_only if requested
        if standard_only and not color_data["is_standard"]:
            continue

        rgb = (color_data["r"], color_data["g"], color_data["b"])

        # Use standard color name if available, otherwise use this color's name
        name = color_names_by_rgb.get(rgb, color_data["name"])

        # Create Color object
        color = Color(rgb=rgb, name=name)

        # Create Element object (count is None since we're not loading set inventory)
        element = Element(
            element_id=element_id,
            design_id=color_data["design_id"],
            variant_id=color_data["variant_id"],
            count=None,
        )

        # Add to result, grouping by Color
        if color not in result:
            result[color] = []
        result[color].append(element)

    return result


def get_set_dimensions(set_id: int) -> tuple[int, int]:
    """Get the canvas dimensions for a LEGO set.

    Args:
        set_id (int): LEGO set identifier (e.g., 31203).

    Returns:
        tuple[int, int]: Tuple of (canvas_width, canvas_height) in studs.

    Raises:
        ValueError: If the set_id is not found.

    Example:
        >>> width, height = get_set_dimensions(31203)  # World Map
        >>> print(f"Canvas size: {width}x{height}")  # 128x80
    """
    sets_raw = _load_sets_raw()

    if set_id not in sets_raw:
        available = sorted(sets_raw.keys())
        raise ValueError(f"Set ID {set_id} not found. Available sets: {available}")

    set_data = sets_raw[set_id]
    return (set_data["canvas_width"], set_data["canvas_height"])


def get_set_info(set_id: int) -> SetRawData:
    """Get full information about a LEGO set.

    Args:
        set_id (int): LEGO set identifier.

    Returns:
        SetRawData: Dict with set information containing keys "name",
            "canvas_width", and "canvas_height".

    Raises:
        ValueError: If the set_id is not found.
    """
    sets_raw = _load_sets_raw()

    if set_id not in sets_raw:
        available = sorted(sets_raw.keys())
        raise ValueError(f"Set ID {set_id} not found. Available sets: {available}")

    return sets_raw[set_id].copy()


def list_available_sets() -> list[tuple[int, str]]:
    """List all available LEGO sets.

    Returns:
        list[tuple[int, str]]: List of (set_id, set_name) tuples, sorted
            by set_id.

    Example:
        >>> for set_id, name in list_available_sets():
        ...     print(f"{set_id}: {name}")
    """
    sets_raw = _load_sets_raw()
    return sorted((sid, data["name"]) for sid, data in sets_raw.items())
