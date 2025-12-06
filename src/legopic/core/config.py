"""Configuration classes for image-to-Lego conversion.

This module provides configuration dataclasses that control the conversion
process. ConvertConfig contains "soft" parameters that can be changed
between re-conversions without restarting a session.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ConvertConfig:
    """Configuration for image-to-Lego conversion.

    These are "soft" parameters that can be changed between re-conversions
    without restarting the session. Hard parameters (image, palette,
    canvas_size) are set when creating a ConversionSession.

    Attributes:
        method: Downsampling method for color matching.
            - 'mean_then_match': Average block colors first, then match to palette.
              Best for smooth color transitions.
            - 'match_then_mean': Match each pixel to palette, then average results.
              Can produce intermediate colors.
            - 'match_then_mode': Match each pixel to palette, take most common.
              Best for preserving sharp edges and distinct colors. (default)
        limit_inventory: If True, respect palette element counts during
            assignment. Colors that run out fall back to next best match.
            If False, use unlimited tiles of any palette color.
        algorithm: Assignment algorithm used when limit_inventory=True.
            - 'priority_greedy': Fast heuristic that assigns cells in order
              of how much they "need" their best color (gap between 1st and
              2nd choice). Good balance of speed and quality.
            - 'optimal': Slower algorithm that finds globally optimal
              assignment minimizing total color distance. (not yet implemented)

    Example:
        >>> config = ConvertConfig(
        ...     method='match_then_mode',
        ...     limit_inventory=True,
        ...     algorithm='priority_greedy'
        ... )
        >>> session.convert(config)
    """

    method: Literal["mean_then_match", "match_then_mean", "match_then_mode"] = "match_then_mode"
    limit_inventory: bool = False
    algorithm: Literal["priority_greedy", "optimal"] = "priority_greedy"
