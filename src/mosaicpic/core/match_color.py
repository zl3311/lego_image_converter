"""Color matching utilities using perceptual color distance.

This module provides functions for matching colors from an image to the
nearest colors in a palette using the Delta E (CIE2000) perceptual
color difference metric.
"""

from typing import TYPE_CHECKING

import numpy as np
from basic_colormath import get_delta_e_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


def match_color(
    target_colors: "NDArray[np.uint8]", palette_colors: "NDArray[np.uint8]"
) -> tuple["NDArray[np.float64]", "NDArray[np.intp]"]:
    """Match target colors to the nearest palette colors.

    Uses the Delta E (CIE2000) metric for perceptually accurate color matching.
    This accounts for non-linearities in human color perception.

    Automatically deduplicates colors when beneficial (>2x reduction), which
    can speed up matching by 20-30x for real photographs with many repeated
    colors (e.g., JPEG compression artifacts).

    Args:
        target_colors: Array of shape (n_targets, 3) containing RGB values
            to be matched.
        palette_colors: Array of shape (n_palette, 3) containing available
            RGB colors to match against.

    Returns:
        A tuple of (distances, rankings):
            - distances: Array of shape (n_targets, n_palette) with Delta E
              distances from each target to each palette color.
            - rankings: Array of shape (n_targets, n_palette) with palette
              indices sorted by distance (best match first).

    Raises:
        ValueError: If input arrays have incorrect shapes.

    Example:
        >>> target = np.array([[255, 0, 0], [0, 255, 0]])  # Red, Green
        >>> palette = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]])
        >>> dists, ranks = match_color(target, palette)
        >>> ranks[0, 0]  # Best match for red -> index 0 (red)
        0
        >>> ranks[1, 0]  # Best match for green -> index 2 (green)
        2
    """
    # Validate target_colors shape
    if target_colors.ndim != 2:
        raise ValueError(
            f"target_colors must be 2D array (n_targets, 3). "
            f"Got {target_colors.ndim}D array with shape {target_colors.shape}."
        )
    if target_colors.shape[1] != 3:
        raise ValueError(
            f"target_colors must have 3 columns (RGB). Got {target_colors.shape[1]} columns."
        )

    # Validate palette_colors shape
    if palette_colors.ndim != 2:
        raise ValueError(
            f"palette_colors must be 2D array (n_palette, 3). "
            f"Got {palette_colors.ndim}D array with shape {palette_colors.shape}."
        )
    if palette_colors.shape[1] != 3:
        raise ValueError(
            f"palette_colors must have 3 columns (RGB). Got {palette_colors.shape[1]} columns."
        )

    n_targets = target_colors.shape[0]

    # Deduplicate colors to reduce Delta E calculations (20-30x speedup for photos)
    unique_colors, inverse_indices = np.unique(target_colors, axis=0, return_inverse=True)
    n_unique = unique_colors.shape[0]
    use_dedup = n_unique < n_targets // 2

    if use_dedup:
        unique_distances = get_delta_e_matrix(unique_colors, palette_colors)
        unique_rankings = np.argsort(unique_distances, axis=1)
        distances = unique_distances[inverse_indices]
        rankings = unique_rankings[inverse_indices]
    else:
        distances = get_delta_e_matrix(target_colors, palette_colors)
        rankings = np.argsort(distances, axis=1)

    return distances, rankings
