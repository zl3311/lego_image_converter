"""Color model for representing RGB colors.

This module provides the Color class for representing colors with RGB values
and optional names. It supports conversion from various color formats (hex, HSV).
"""

import colorsys
import re

import numpy as np


class Color:
    """Represents an RGB color with an optional name.

    Attributes:
        rgb: Tuple of (red, green, blue) values, each in range [0, 255].
        name: Optional human-readable name for the color (e.g., "Bright Red").
    """

    def __init__(self, rgb: tuple[int, int, int], name: str | None = None):
        """Initialize a Color with RGB values.

        Args:
            rgb (tuple[int, int, int]): Tuple of (red, green, blue) values,
                each must be in [0, 255]. Accepts Python ints or numpy
                integer types.
            name (str | None): Optional name for the color. For tile colors,
                this is typically provided from the elements data file.

        Raises:
            ValueError: If any RGB channel is outside the valid range [0, 255].
        """
        validated_rgb = []
        for channel, channel_name in zip(rgb, ("red", "green", "blue"), strict=True):
            # Accept both Python int and numpy integer types
            if isinstance(channel, (int, np.integer)):
                val = int(channel)
                if val < 0 or val > 255:
                    raise ValueError(f"RGB {channel_name} value must be in [0, 255]. Got {val}.")
                validated_rgb.append(val)
            else:
                raise ValueError(
                    f"RGB {channel_name} value must be an integer in [0, 255]. "
                    f"Got {channel!r} (type: {type(channel).__name__})."
                )
        self.rgb: tuple[int, int, int] = (validated_rgb[0], validated_rgb[1], validated_rgb[2])
        self.name = name

    @classmethod
    def from_hex(cls, hex_code: str, name: str | None = None) -> "Color":
        """Create a Color from a hexadecimal color code.

        Args:
            hex_code (str): Hex color string, with or without leading '#'.
                Supports both 3-character (e.g., "FFF") and 6-character
                (e.g., "FFFFFF") formats.
            name (str | None): Optional name for the color.

        Returns:
            Color: A new Color instance.

        Raises:
            ValueError: If the hex code is invalid.

        Example:
            >>> Color.from_hex("FF0000")
            >>> Color.from_hex("#00FF00", name="Green")
            >>> Color.from_hex("00F")  # Short form for #0000FF
        """
        # Remove leading '#' if present
        hex_code = hex_code.lstrip("#")

        # Validate format
        if not re.match(r"^[0-9A-Fa-f]{3}$|^[0-9A-Fa-f]{6}$", hex_code):
            raise ValueError(
                f"Invalid hex color code: '{hex_code}'. "
                f"Expected 3 or 6 hexadecimal characters (with optional '#' prefix)."
            )

        # Expand short form (e.g., "F0A" -> "FF00AA")
        if len(hex_code) == 3:
            hex_code = "".join(c * 2 for c in hex_code)

        r, g, b = (int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
        return cls((r, g, b), name)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, name: str | None = None) -> "Color":
        """Create a Color from HSV (Hue, Saturation, Value) values.

        Args:
            h (float): Hue in range [0, 360) degrees.
            s (float): Saturation in range [0, 1].
            v (float): Value (brightness) in range [0, 1].
            name (str | None): Optional name for the color.

        Returns:
            Color: A new Color instance.

        Raises:
            ValueError: If any HSV value is outside its valid range.

        Example:
            >>> Color.from_hsv(0, 1, 1)      # Pure red
            >>> Color.from_hsv(120, 1, 1)    # Pure green
            >>> Color.from_hsv(240, 1, 0.5)  # Dark blue
        """
        if not (0 <= h < 360):
            raise ValueError(f"Hue must be in [0, 360). Got {h}.")
        if not (0 <= s <= 1):
            raise ValueError(f"Saturation must be in [0, 1]. Got {s}.")
        if not (0 <= v <= 1):
            raise ValueError(f"Value must be in [0, 1]. Got {v}.")

        # colorsys expects h in [0, 1], so normalize
        r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        return cls(rgb, name)

    def __hash__(self) -> int:
        """Return hash based on RGB values for use in sets/dicts."""
        return hash(self.rgb)

    def __eq__(self, other: object) -> bool:
        """Check equality based on RGB values."""
        if not isinstance(other, Color):
            return NotImplemented
        return self.rgb == other.rgb

    def __repr__(self) -> str:
        """Return string representation of the Color."""
        if self.name:
            return f"Color(rgb={self.rgb}, name={self.name!r})"
        return f"Color(rgb={self.rgb})"
