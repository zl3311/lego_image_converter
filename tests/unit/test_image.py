"""Unit tests for the Image class."""

import numpy as np
import pytest

from mosaicpic import Image


class TestImageInit:
    """Tests for Image.__init__."""

    def test_valid_array(self):
        """Image accepts valid 3D RGB array."""
        array = np.zeros((100, 200, 3), dtype=np.uint8)
        image = Image(array)
        assert image.height == 100
        assert image.width == 200

    def test_cells_created(self):
        """Image creates cells for each pixel."""
        array = np.zeros((10, 20, 3), dtype=np.uint8)
        image = Image(array)
        assert len(image.cells) == 10 * 20

    def test_cell_coordinates(self):
        """Cells have correct coordinates."""
        array = np.zeros((5, 5, 3), dtype=np.uint8)
        image = Image(array)

        # Check first cell (0, 0)
        assert image.cells[0].x == 0
        assert image.cells[0].y == 0

        # Check cell at (2, 1) - index = y * width + x = 1 * 5 + 2 = 7
        cell = image.get_cell(2, 1)
        assert cell.x == 2
        assert cell.y == 1

    def test_cell_colors(self):
        """Cells have correct colors from array."""
        array = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [128, 128, 128]],
            ],
            dtype=np.uint8,
        )
        image = Image(array)

        assert image.get_cell(0, 0).color.rgb == (255, 0, 0)
        assert image.get_cell(1, 0).color.rgb == (0, 255, 0)
        assert image.get_cell(0, 1).color.rgb == (0, 0, 255)
        assert image.get_cell(1, 1).color.rgb == (128, 128, 128)

    def test_non_array_raises(self):
        """Image rejects non-numpy input."""
        with pytest.raises(ValueError, match="numpy.ndarray"):
            Image([[1, 2, 3]])  # type: ignore

    def test_wrong_dimensions_raises(self):
        """Image rejects arrays with wrong number of dimensions."""
        with pytest.raises(ValueError, match="3D array"):
            Image(np.zeros((100, 200), dtype=np.uint8))

        with pytest.raises(ValueError, match="3D array"):
            Image(np.zeros((100,), dtype=np.uint8))

    def test_wrong_channels_raises(self):
        """Image rejects arrays with wrong number of channels."""
        with pytest.raises(ValueError, match="3 color channels"):
            Image(np.zeros((100, 200, 4), dtype=np.uint8))  # RGBA

        with pytest.raises(ValueError, match="3 color channels"):
            Image(np.zeros((100, 200, 1), dtype=np.uint8))  # Grayscale


class TestImageGetCell:
    """Tests for Image.get_cell."""

    def test_valid_coordinates(self):
        """get_cell returns correct cell for valid coordinates."""
        array = np.zeros((10, 20, 3), dtype=np.uint8)
        array[5, 10, :] = [100, 150, 200]
        image = Image(array)

        cell = image.get_cell(10, 5)
        assert cell.x == 10
        assert cell.y == 5
        assert cell.color.rgb == (100, 150, 200)

    def test_boundary_coordinates(self):
        """get_cell works at image boundaries."""
        array = np.zeros((10, 20, 3), dtype=np.uint8)
        image = Image(array)

        # Corners
        assert image.get_cell(0, 0) is not None
        assert image.get_cell(19, 0) is not None
        assert image.get_cell(0, 9) is not None
        assert image.get_cell(19, 9) is not None

    def test_out_of_bounds_raises(self):
        """get_cell raises IndexError for invalid coordinates."""
        array = np.zeros((10, 20, 3), dtype=np.uint8)
        image = Image(array)

        with pytest.raises(IndexError, match="out of bounds"):
            image.get_cell(-1, 0)

        with pytest.raises(IndexError, match="out of bounds"):
            image.get_cell(0, -1)

        with pytest.raises(IndexError, match="out of bounds"):
            image.get_cell(20, 0)

        with pytest.raises(IndexError, match="out of bounds"):
            image.get_cell(0, 10)


class TestImageToArray:
    """Tests for Image.to_array."""

    def test_returns_copy(self):
        """to_array returns a copy, not the original."""
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        image = Image(original)

        result = image.to_array()
        result[0, 0, 0] = 255

        # Original should be unchanged
        assert image.to_array()[0, 0, 0] == 0

    def test_preserves_values(self):
        """to_array preserves all pixel values."""
        original = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        image = Image(original)

        np.testing.assert_array_equal(image.to_array(), original)
