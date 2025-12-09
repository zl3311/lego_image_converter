"""LEGO data files and loader utilities.

This subpackage contains CSV data files for LEGO colors, sets, and elements,
along with functions to load and query this data.

Data Files:
    - colors.csv: Maps element_id → color name, RGB, variant_id, is_standard,
      bl_id (BrickLink), rb_id (Rebrickable).
    - sets.csv: Maps set_id → set name, canvas dimensions.
    - elements.csv: Maps set_id + element_id → piece count.

Functions:
    get_colors_for_set: Get Color → Element mapping for a specific set.
    get_all_colors: Get all colors (optionally filtered to standard).
    get_set_dimensions: Get (width, height) for a set's canvas.
    get_set_info: Get full info dict for a set.
    list_available_sets: List all available (set_id, name) pairs.
    get_color_external_ids: Get BrickLink/Rebrickable IDs for a color name.
"""

from .loader import (
    get_all_colors,
    get_color_external_ids,
    get_colors_for_set,
    get_set_dimensions,
    get_set_info,
    list_available_sets,
)

__all__ = [
    "get_colors_for_set",
    "get_all_colors",
    "get_set_dimensions",
    "get_set_info",
    "list_available_sets",
    "get_color_external_ids",
]
