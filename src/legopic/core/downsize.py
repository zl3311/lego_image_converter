"""Image downsizing algorithms for Lego mosaic conversion.

This module provides functions to downsize an image to canvas dimensions
while matching colors to a palette.

Methods:
    mean_then_match: Average pixel colors first, then match to palette.
    match_then_mean: Match each pixel to palette first, then average.
    match_then_mode: Match each pixel to palette first, then take mode.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ..models import Canvas, Image, Palette
from .match_color import match_color


def _validate_dimensions(image: Image, canvas_width: int, canvas_height: int) -> int:
    """Validate image and canvas dimensions are compatible.

    For uniform stride downsampling, we require that both dimensions
    produce the same stride value (floor of ratio). The last row/column
    of canvas cells may have more pixels than others (incomplete stride
    is allowed).

    Args:
        image (Image): Source image.
        canvas_width (int): Target canvas width.
        canvas_height (int): Target canvas height.

    Returns:
        int: The computed uniform stride value.

    Raises:
        ValueError: If dimensions are incompatible for uniform stride.
    """
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            f"Canvas dimensions must be positive. Got width={canvas_width}, height={canvas_height}."
        )

    if image.width < canvas_width or image.height < canvas_height:
        raise ValueError(
            f"Image ({image.width}×{image.height}) must be at least as large as "
            f"canvas ({canvas_width}×{canvas_height})."
        )

    # Use floor() so last cells get remaining pixels (incomplete stride OK)
    stride_w = image.width // canvas_width
    stride_h = image.height // canvas_height

    if stride_w != stride_h:
        raise ValueError(
            f"Incompatible dimensions for uniform stride downsampling. "
            f"Image ({image.width}×{image.height}) with canvas ({canvas_width}×{canvas_height}) "
            f"requires stride {stride_w} for width but stride {stride_h} for height. "
            f"Resize your image so (image_width // canvas_width) == (image_height // canvas_height)."
        )

    return int(stride_w)


def _get_block_pixels(
    image_array: "NDArray[np.uint8]", canvas_x: int, canvas_y: int, stride: int
) -> "NDArray[np.uint8]":
    """Extract pixels from a block region of the image.

    Each canvas cell maps to a block of pixels. Most blocks have exactly
    stride × stride pixels, but the last row/column of cells extend to
    the image edge and may contain more pixels.

    Args:
        image_array (NDArray[np.uint8]): Source image array (height, width, 3).
        canvas_x (int): X-coordinate of the canvas cell.
        canvas_y (int): Y-coordinate of the canvas cell.
        stride (int): Base block size (pixels per canvas cell). Last cells may
            have more pixels due to floor division of image dimensions.

    Returns:
        NDArray[np.uint8]: Array of shape (n_pixels, 3) containing RGB values.
    """
    y_start = canvas_y * stride
    y_end = min((canvas_y + 1) * stride, image_array.shape[0])
    x_start = canvas_x * stride
    x_end = min((canvas_x + 1) * stride, image_array.shape[1])

    block = image_array[y_start:y_end, x_start:x_end, :]
    return block.reshape(-1, 3)


def downsize(
    image: Image,
    palette: Palette,
    canvas_width: int,
    canvas_height: int,
    method: str = "match_then_mode",
) -> Canvas:
    """Downsize an image to a canvas, matching colors to a palette.

    This function reduces image resolution to canvas dimensions while
    mapping all colors to the available palette colors.

    Args:
        image (Image): Source image to downsize.
        palette (Palette): Available colors for matching.
        canvas_width (int): Target canvas width in cells.
        canvas_height (int): Target canvas height in cells.
        method (str): Downsampling method. One of:

            - ``'mean_then_match'``: Average block colors, then match to palette.
              Best for smooth color transitions.
            - ``'match_then_mean'``: Match each pixel to palette, average the
              results. Can produce colors not in palette (intermediate values).
            - ``'match_then_mode'``: Match each pixel to palette, take most
              common. Best for preserving sharp edges and distinct colors.

    Returns:
        Canvas: A Canvas with dimensions (canvas_width, canvas_height) where
            each cell has a color from the palette.

    Raises:
        ValueError: If dimensions are incompatible or method is invalid.

    Example:
        >>> image = Image.from_file("photo.jpg")
        >>> palette = Palette([Color((255, 0, 0)), Color((0, 0, 255))])
        >>> canvas = downsize(image, palette, 48, 48)
    """
    valid_methods = ["mean_then_match", "match_then_mean", "match_then_mode"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}.")

    stride = _validate_dimensions(image, canvas_width, canvas_height)
    image_array = image.to_array()
    palette_rgb = palette.to_rgb_array()

    # Initialize canvas
    canvas = Canvas(canvas_width, canvas_height)

    if method == "mean_then_match":
        # Step 1: Compute mean color for each block
        mean_colors = np.zeros((canvas_height * canvas_width, 3), dtype=np.uint8)

        for cy in range(canvas_height):
            for cx in range(canvas_width):
                block_pixels = _get_block_pixels(image_array, cx, cy, stride)
                mean_colors[cy * canvas_width + cx] = np.mean(block_pixels, axis=0).astype(np.uint8)

        # Step 2: Match mean colors to palette
        _, rankings = match_color(mean_colors, palette_rgb)

        # Step 3: Assign best match to each canvas cell
        for cy in range(canvas_height):
            for cx in range(canvas_width):
                idx = cy * canvas_width + cx
                best_palette_idx = rankings[idx, 0]
                canvas.set_cell(cx, cy, palette.colors[best_palette_idx])

    elif method == "match_then_mean":
        # For each block: match all pixels to palette, then average RGB values
        for cy in range(canvas_height):
            for cx in range(canvas_width):
                block_pixels = _get_block_pixels(image_array, cx, cy, stride)

                # Match each pixel to palette
                _, rankings = match_color(block_pixels, palette_rgb)
                matched_colors = palette_rgb[rankings[:, 0]]

                # Average the matched colors
                mean_matched = np.mean(matched_colors, axis=0).astype(np.uint8)

                # Find closest palette color to this mean
                mean_2d = mean_matched.reshape(1, 3)
                _, final_ranking = match_color(mean_2d, palette_rgb)
                best_idx = final_ranking[0, 0]
                canvas.set_cell(cx, cy, palette.colors[best_idx])

    elif method == "match_then_mode":
        # Optimized approach: batch all pixels into a single match_color call.
        # This reduces O(n_cells) function calls to O(1), significantly faster.

        # Match all image pixels to palette at once (single call for all ~1M pixels)
        all_pixels = image_array.reshape(-1, 3)
        _, rankings = match_color(all_pixels, palette_rgb)
        best_per_pixel = rankings[:, 0]

        # Reshape matched indices back to image dimensions
        matched_image = best_per_pixel.reshape(image_array.shape[0], image_array.shape[1])

        # Compute mode (most common palette index) for each canvas cell's block
        n_palette = len(palette_rgb)
        for cy in range(canvas_height):
            for cx in range(canvas_width):
                y_start = cy * stride
                y_end = min((cy + 1) * stride, matched_image.shape[0])
                x_start = cx * stride
                x_end = min((cx + 1) * stride, matched_image.shape[1])

                block_indices = matched_image[y_start:y_end, x_start:x_end].flatten()
                counts = np.bincount(block_indices, minlength=n_palette)
                mode_idx = counts.argmax()
                canvas.set_cell(cx, cy, palette.colors[mode_idx])

    return canvas
