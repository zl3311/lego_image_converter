"""Unit tests for pipeline data types."""

import numpy as np
import pytest

from legopic import Color, Palette
from legopic.pipeline import IndexMap, RGBImage


class TestRGBImage:
    """Tests for RGBImage data type."""

    def test_valid_creation(self):
        """RGBImage accepts valid 3D uint8 array."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        img = RGBImage(data=data)

        assert img.height == 100
        assert img.width == 100
        assert img.shape == (100, 100)

    def test_rejects_wrong_ndim(self):
        """RGBImage rejects non-3D arrays."""
        data = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected 3D array"):
            RGBImage(data=data)

    def test_rejects_wrong_channels(self):
        """RGBImage rejects arrays without 3 channels."""
        data = np.zeros((100, 100, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected 3 channels"):
            RGBImage(data=data)

    def test_rejects_wrong_dtype(self):
        """RGBImage rejects non-uint8 arrays."""
        data = np.zeros((100, 100, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected uint8 dtype"):
            RGBImage(data=data)


class TestIndexMap:
    """Tests for IndexMap data type."""

    @pytest.fixture
    def simple_palette(self):
        """A simple 3-color palette."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
            ]
        )

    def test_valid_creation(self, simple_palette):
        """IndexMap accepts valid 2D integer array."""
        data = np.zeros((10, 10), dtype=np.intp)
        index_map = IndexMap(data=data, palette=simple_palette)

        assert index_map.height == 10
        assert index_map.width == 10
        assert index_map.shape == (10, 10)

    def test_rejects_wrong_ndim(self, simple_palette):
        """IndexMap rejects non-2D arrays."""
        data = np.zeros((10, 10, 3), dtype=np.intp)

        with pytest.raises(ValueError, match="Expected 2D array"):
            IndexMap(data=data, palette=simple_palette)

    def test_rejects_non_integer_dtype(self, simple_palette):
        """IndexMap rejects non-integer arrays."""
        data = np.zeros((10, 10), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected integer dtype"):
            IndexMap(data=data, palette=simple_palette)

    def test_to_rgb_conversion(self, simple_palette):
        """IndexMap.to_rgb converts indices to RGB colors."""
        # Create index map with alternating indices
        data = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.intp)
        index_map = IndexMap(data=data, palette=simple_palette)

        rgb_image = index_map.to_rgb()

        assert isinstance(rgb_image, RGBImage)
        assert rgb_image.shape == (2, 3)

        # Verify colors
        assert tuple(rgb_image.data[0, 0]) == (255, 0, 0)  # Red
        assert tuple(rgb_image.data[0, 1]) == (0, 255, 0)  # Green
        assert tuple(rgb_image.data[0, 2]) == (0, 0, 255)  # Blue
