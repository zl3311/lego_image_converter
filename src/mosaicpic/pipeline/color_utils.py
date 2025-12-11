"""Color space conversion utilities.

This module provides functions for converting between color spaces:
- RGB to/from CIE Lab (perceptually uniform)
- RGB to/from Linear RGB (gamma-corrected)

These conversions are used by PoolStep for perceptually-accurate color averaging.

Implementation Notes:
    - Uses D65 illuminant (standard daylight)
    - sRGB gamma curve with linear segment for dark colors
    - Lab uses standard CIELAB formulas
"""

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# D65 illuminant reference white point
_D65_WHITE = np.array([95.047, 100.0, 108.883])


def rgb_to_linear(rgb: "NDArray[np.uint8]") -> "NDArray[np.float64]":
    """Convert sRGB to linear RGB.

    Applies inverse sRGB gamma curve to convert from perceptual (gamma-encoded)
    sRGB values to linear light values.

    Args:
        rgb: Array of shape (..., 3) with uint8 RGB values [0, 255].

    Returns:
        Array of shape (..., 3) with float64 linear RGB values [0, 1].
    """
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # sRGB inverse gamma: linear segment for dark colors, power curve for brighter
    # Threshold is approximately 0.04045
    linear = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
    )

    return linear


def linear_to_rgb(linear: "NDArray[np.float64]") -> "NDArray[np.uint8]":
    """Convert linear RGB to sRGB.

    Applies sRGB gamma curve to convert from linear light values
    to perceptual (gamma-encoded) sRGB values.

    Args:
        linear: Array of shape (..., 3) with float64 linear RGB values [0, 1].

    Returns:
        Array of shape (..., 3) with uint8 RGB values [0, 255].
    """
    # Clip to valid range
    linear = np.clip(linear, 0.0, 1.0)

    # sRGB gamma: linear segment for dark colors, power curve for brighter
    # Threshold is approximately 0.0031308
    rgb_norm = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1.0 / 2.4)) - 0.055,
    )

    # Scale to [0, 255] and convert to uint8
    return np.round(rgb_norm * 255.0).astype(np.uint8)


def rgb_to_xyz(rgb: "NDArray[np.uint8]") -> "NDArray[np.float64]":
    """Convert sRGB to CIE XYZ.

    Args:
        rgb: Array of shape (..., 3) with uint8 RGB values [0, 255].

    Returns:
        Array of shape (..., 3) with float64 XYZ values.
    """
    # Convert to linear RGB first
    linear = rgb_to_linear(rgb)

    # sRGB to XYZ matrix (D65 illuminant)
    # Source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )

    # Apply matrix transformation
    # Reshape for matrix multiplication: (..., 3) @ (3, 3).T -> (..., 3)
    xyz = linear @ matrix.T

    # Scale to reference white
    return cast("NDArray[np.float64]", xyz * 100.0)


def xyz_to_rgb(xyz: "NDArray[np.float64]") -> "NDArray[np.uint8]":
    """Convert CIE XYZ to sRGB.

    Args:
        xyz: Array of shape (..., 3) with float64 XYZ values.

    Returns:
        Array of shape (..., 3) with uint8 RGB values [0, 255].
    """
    # Scale from reference white
    xyz_norm = xyz / 100.0

    # XYZ to sRGB matrix (D65 illuminant)
    matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )

    # Apply matrix transformation
    linear = xyz_norm @ matrix.T

    # Convert linear RGB to sRGB
    return linear_to_rgb(linear)


def rgb_to_lab(rgb: "NDArray[np.uint8]") -> "NDArray[np.float64]":
    """Convert sRGB to CIE Lab.

    Args:
        rgb: Array of shape (..., 3) with uint8 RGB values [0, 255].

    Returns:
        Array of shape (..., 3) with float64 Lab values.
            L: [0, 100]
            a: approximately [-128, 127]
            b: approximately [-128, 127]
    """
    # Convert to XYZ first
    xyz = rgb_to_xyz(rgb)

    # Normalize by D65 white point
    xyz_norm = xyz / _D65_WHITE

    # Apply Lab transformation function
    # f(t) = t^(1/3) if t > (6/29)^3, else (1/3)*(29/6)^2*t + 4/29
    delta = 6.0 / 29.0
    delta_cube = delta**3

    f_xyz = np.where(
        xyz_norm > delta_cube,
        xyz_norm ** (1.0 / 3.0),
        (xyz_norm / (3.0 * delta**2)) + (4.0 / 29.0),
    )

    # Calculate Lab
    L = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: "NDArray[np.float64]") -> "NDArray[np.uint8]":
    """Convert CIE Lab to sRGB.

    Args:
        lab: Array of shape (..., 3) with float64 Lab values.

    Returns:
        Array of shape (..., 3) with uint8 RGB values [0, 255].
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Calculate f values
    f_y = (L + 16.0) / 116.0
    f_x = a / 500.0 + f_y
    f_z = f_y - b / 200.0

    # Apply inverse Lab transformation
    # t = f^3 if f > 6/29, else 3*delta^2*(f - 4/29)
    delta = 6.0 / 29.0

    def inverse_f(f: "NDArray[np.float64]") -> "NDArray[np.float64]":
        return np.where(f > delta, f**3, 3.0 * delta**2 * (f - 4.0 / 29.0))

    xyz_norm = np.stack([inverse_f(f_x), inverse_f(f_y), inverse_f(f_z)], axis=-1)

    # Denormalize by D65 white point
    xyz = xyz_norm * _D65_WHITE

    # Convert to RGB
    return xyz_to_rgb(xyz)
