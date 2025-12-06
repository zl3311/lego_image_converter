"""Shared pytest fixtures for legopic tests.

This module provides reusable fixtures for creating test data including
sample images, palettes, and colors across all test modules.

Fixtures:
    sample_colors: Basic RGB primaries plus black and white.
    sample_palette: Palette containing basic RGB colors.
    tiny_red_image: A 10x10 solid red test image.
    tiny_gradient_image: A 10x10 gradient test image.
    compatible_image: A 100x100 random image compatible with 10x10 canvas.
    lego_art_palette: Palette simulating LEGO Art set colors.
"""

import numpy as np
import pytest

from legopic import Color, Image, Palette


@pytest.fixture
def sample_colors() -> list[Color]:
    """Return a basic set of test colors (RGB primaries + black/white)."""
    return [
        Color((255, 0, 0), name="Red"),
        Color((0, 255, 0), name="Green"),
        Color((0, 0, 255), name="Blue"),
        Color((255, 255, 255), name="White"),
        Color((0, 0, 0), name="Black"),
    ]


@pytest.fixture
def sample_palette(sample_colors: list[Color]) -> Palette:
    """Return a palette containing basic RGB colors."""
    return Palette(sample_colors)


@pytest.fixture
def tiny_red_image() -> Image:
    """Return a 10x10 solid red image."""
    array = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
    return Image(array)


@pytest.fixture
def tiny_gradient_image() -> Image:
    """Return a 10x10 image with a color gradient."""
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    for y in range(10):
        for x in range(10):
            array[y, x, 0] = x * 25  # Red increases with x
            array[y, x, 1] = y * 25  # Green increases with y
            array[y, x, 2] = 128  # Blue constant
    return Image(array)


@pytest.fixture
def compatible_image() -> Image:
    """Return a 100x100 image compatible with 10x10 canvas (stride=10)."""
    array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return Image(array)


@pytest.fixture
def lego_art_palette() -> Palette:
    """Return a palette simulating LEGO Art set colors."""
    return Palette(
        [
            Color((0, 0, 0), name="Black"),
            Color((255, 255, 255), name="White"),
            Color((180, 0, 0), name="Dark Red"),
            Color((255, 85, 0), name="Orange"),
            Color((255, 205, 0), name="Bright Yellow"),
            Color((0, 100, 0), name="Dark Green"),
            Color((0, 150, 200), name="Medium Azure"),
            Color((0, 50, 150), name="Dark Blue"),
            Color((100, 50, 150), name="Medium Lilac"),
            Color((150, 100, 50), name="Medium Nougat"),
        ]
    )
