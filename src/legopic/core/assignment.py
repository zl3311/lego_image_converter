"""Assignment algorithms for inventory-limited color matching.

This module provides algorithms for assigning palette colors to canvas cells
while respecting inventory constraints. When inventory is limited, cells
compete for available colors, and the algorithm must decide which cells
get their preferred colors vs. fallback options.

Algorithms:
    priority_greedy: Fast heuristic based on cell "desperation" (difference
        between best and 2nd best match). Most desperate cells assigned first.
    optimal: Placeholder for min-cost max-flow optimal assignment.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ..models import Color, Palette
from .match_color import match_color


@dataclass
class AssignmentResult:
    """Result of a color assignment algorithm.

    Attributes:
        assignments: 2D array of palette indices, shape (height, width).
            assignments[y][x] is the index into palette.colors for cell (x, y).
        inventory_used: Dict mapping Color to count used.
        fallback_count: Number of cells that couldn't get their best color.
    """

    assignments: "NDArray[np.intp]"
    inventory_used: dict[Color, int]
    fallback_count: int


def _get_inventory_from_palette(palette: Palette) -> dict[int, int]:
    """Extract inventory counts from a palette.

    For each color index, sums the counts across all element variants.

    Args:
        palette (Palette): The palette with element counts.

    Returns:
        dict[int, int]: Dict mapping palette color index to available count.
            Returns -1 (representing unlimited) if no counts are set.
    """
    inventory = {}
    colors = palette.colors

    for idx, color in enumerate(colors):
        elements = palette.get_elements_for_color(color)
        if not elements:
            # No elements = unlimited (custom palette without inventory)
            inventory[idx] = -1  # -1 means unlimited
        else:
            # Sum counts across all variants
            total = sum(e.count for e in elements if e.count is not None)
            if total == 0 and any(e.count is None for e in elements):
                # Some elements have no count set = unlimited
                inventory[idx] = -1
            else:
                inventory[idx] = total

    return inventory


def priority_greedy(
    target_colors: "NDArray[np.uint8]",
    palette: Palette,
    pinned_cells: dict[tuple[int, int], Color] | None = None,
    canvas_width: int = 0,
    canvas_height: int = 0,
) -> AssignmentResult:
    """Assign colors using priority greedy algorithm.

    This algorithm assigns cells in order of "desperation" - how much each
    cell needs its best color relative to alternatives. Cells with large
    gaps between their best and 2nd-best match are assigned first.

    Algorithm:
        1. Compute distances from each target to all palette colors
        2. Compute desperation = distance[2nd best] - distance[best]
        3. Reserve inventory for pinned cells
        4. Sort unpinned cells by desperation (descending)
        5. For each cell, assign best available color (that has inventory)

    Args:
        target_colors: Array of shape (height, width, 3) with target RGB
            colors for each cell (typically mean of original image block).
        palette: Available colors with optional inventory counts.
        pinned_cells: Dict of (x, y) -> Color for cells that are fixed.
            These consume inventory but aren't reassigned.
        canvas_width: Width of the canvas (for coordinate mapping).
        canvas_height: Height of the canvas (for coordinate mapping).

    Returns:
        AssignmentResult with assignments and statistics.

    Example:
        >>> result = priority_greedy(target_colors, palette)
        >>> for y in range(height):
        ...     for x in range(width):
        ...         color_idx = result.assignments[y, x]
        ...         color = palette.colors[color_idx]
    """
    height, width = target_colors.shape[:2]
    canvas_width = canvas_width or width
    canvas_height = canvas_height or height

    # Flatten target colors for batch processing
    target_flat = target_colors.reshape(-1, 3)
    n_cells = target_flat.shape[0]

    palette_rgb = palette.to_rgb_array()
    n_palette = len(palette_rgb)
    palette_colors = palette.colors

    # Get inventory (count per palette color index)
    inventory = _get_inventory_from_palette(palette)
    remaining = inventory.copy()

    # Compute distances from all targets to all palette colors
    distances, rankings = match_color(target_flat, palette_rgb)
    # distances: shape (n_cells, n_palette)
    # rankings: shape (n_cells, n_palette), sorted indices

    # Initialize assignments to -1 (unassigned)
    assignments = np.full((height, width), -1, dtype=np.intp)

    # Track which cells are pinned
    pinned_set = set(pinned_cells.keys()) if pinned_cells else set()

    # Step 1: Reserve inventory for pinned cells
    if pinned_cells:
        for (x, y), color in pinned_cells.items():
            # Find palette index for this color
            palette_idx = None
            for idx, pc in enumerate(palette_colors):
                if pc.rgb == color.rgb:
                    palette_idx = idx
                    break

            if palette_idx is not None:
                # Assign and consume inventory
                assignments[y, x] = palette_idx
                if remaining[palette_idx] > 0:  # -1 means unlimited
                    remaining[palette_idx] -= 1
            else:
                # Pinned to out-of-palette color - assign -2 to mark special
                # We'll handle this separately
                assignments[y, x] = -2

    # Step 2: Compute desperation for unpinned cells
    # Desperation = distance to 2nd best - distance to best
    # Higher desperation = more critical to get preferred color

    # Get indices of unpinned cells
    unpinned_list: list[int] = []
    for idx in range(n_cells):
        y, x = divmod(idx, width)
        if (x, y) not in pinned_set:
            unpinned_list.append(idx)

    if not unpinned_list:
        # All cells are pinned
        return AssignmentResult(
            assignments=assignments,
            inventory_used=_count_usage(assignments, palette_colors),
            fallback_count=0,
        )

    unpinned_indices = np.array(unpinned_list)

    # For each unpinned cell, compute desperation
    # best_dist = distances[idx, rankings[idx, 0]]
    # second_dist = distances[idx, rankings[idx, 1]]
    # desperation = second_dist - best_dist

    if n_palette == 1:
        # Only one color - no desperation to compute, just assign
        desperations = np.zeros(len(unpinned_indices))
        sorted_indices = unpinned_indices
    else:
        best_indices = rankings[unpinned_indices, 0]
        second_indices = rankings[unpinned_indices, 1]

        best_distances = distances[unpinned_indices, best_indices]
        second_distances = distances[unpinned_indices, second_indices]

        desperations = second_distances - best_distances

        # Sort by desperation (descending - most desperate first)
        sort_order = np.argsort(-desperations)
        sorted_indices = unpinned_indices[sort_order]

    # Step 3: Assign colors in desperation order
    fallback_count = 0

    for flat_idx in sorted_indices:
        y, x = divmod(int(flat_idx), width)

        # Find best available color (with remaining inventory)
        cell_rankings = rankings[flat_idx]

        assigned = False
        for rank_pos in range(n_palette):
            palette_idx = cell_rankings[rank_pos]

            # Check inventory
            if remaining[palette_idx] == -1 or remaining[palette_idx] > 0:
                # Assign this color
                assignments[y, x] = palette_idx
                if remaining[palette_idx] > 0:
                    remaining[palette_idx] -= 1

                # Track if this was a fallback (not best choice)
                if rank_pos > 0:
                    fallback_count += 1

                assigned = True
                break

        if not assigned:
            # No inventory for any color - shouldn't happen with proper data
            # Fall back to best color anyway (inventory exhausted)
            assignments[y, x] = cell_rankings[0]
            fallback_count += 1

    return AssignmentResult(
        assignments=assignments,
        inventory_used=_count_usage(assignments, palette_colors),
        fallback_count=fallback_count,
    )


def optimal(
    target_colors: "NDArray[np.uint8]",
    palette: Palette,
    pinned_cells: dict[tuple[int, int], Color] | None = None,
    canvas_width: int = 0,
    canvas_height: int = 0,
) -> AssignmentResult:
    """Assign colors using optimal min-cost max-flow algorithm.

    This algorithm finds the globally optimal assignment that minimizes
    total color distance while respecting inventory constraints.

    NOT YET IMPLEMENTED - falls back to priority_greedy.

    Args:
        target_colors: Array of shape (height, width, 3) with target RGB.
        palette: Available colors with optional inventory counts.
        pinned_cells: Dict of (x, y) -> Color for cells that are fixed.
        canvas_width: Width of the canvas.
        canvas_height: Height of the canvas.

    Returns:
        AssignmentResult with optimal assignments.

    Raises:
        NotImplementedError: Always (placeholder for future implementation).
    """
    # TODO: Implement min-cost max-flow optimal assignment
    # For now, fall back to priority_greedy with a warning
    import warnings

    warnings.warn(
        "Optimal assignment algorithm not yet implemented. Falling back to priority_greedy.",
        UserWarning,
        stacklevel=2,
    )
    return priority_greedy(target_colors, palette, pinned_cells, canvas_width, canvas_height)


def _count_usage(assignments: "NDArray[np.intp]", palette_colors: list[Color]) -> dict[Color, int]:
    """Count how many of each color was used.

    Args:
        assignments (NDArray[np.intp]): 2D array of palette indices.
        palette_colors (list[Color]): List of Color objects in palette.

    Returns:
        dict[Color, int]: Dict mapping Color to usage count.
    """
    from collections import Counter

    flat = assignments.flatten()
    counts = Counter(flat)

    result = {}
    for idx, count in counts.items():
        if idx >= 0 and idx < len(palette_colors):
            result[palette_colors[idx]] = count

    return result
