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

    Args:
        target_colors (NDArray[np.uint8]): Array of shape (n_targets, 3)
            containing RGB values to be matched.
        palette_colors (NDArray[np.uint8]): Array of shape (n_palette, 3)
            containing available RGB colors to match against.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.intp]]: A tuple of (distances,
            rankings):

            - distances: Array of shape (n_targets, n_palette) containing
              Delta E distances from each target to each palette color.
            - rankings: Array of shape (n_targets, n_palette) containing
              palette indices sorted by distance (best match first).
              rankings[i, 0] is the index of the best match for target i.

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

    # Compute perceptual color distances
    distances = get_delta_e_matrix(target_colors, palette_colors)

    # Rank palette colors by distance (ascending) for each target
    rankings = np.argsort(distances, axis=1)

    return distances, rankings
