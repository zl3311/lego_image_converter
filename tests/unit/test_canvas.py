"""Unit tests for the Canvas class.

Tests cover initialization, factory methods, cell access, and conversion
with comprehensive coverage of valid and invalid inputs.
"""

import numpy as np
import pytest

from mosaicpic import Canvas, Color
from mosaicpic.models import Cell


class TestCanvasInit:
    """Tests for Canvas.__init__."""

    def test_valid_dimensions(self):
        """Canvas accepts valid positive dimensions."""
        canvas = Canvas(10, 20)

        assert canvas.width == 10
        assert canvas.height == 20

    def test_cells_created(self):
        """Canvas creates cells for all positions."""
        canvas = Canvas(5, 3)

        assert len(canvas.cells) == 3  # Height
        assert len(canvas.cells[0]) == 5  # Width

    def test_cells_have_correct_coordinates(self):
        """Cells have correct x, y coordinates."""
        canvas = Canvas(3, 4)

        for y in range(4):
            for x in range(3):
                cell = canvas.cells[y][x]
                assert cell.x == x
                assert cell.y == y

    def test_cells_initialized_with_none_color(self):
        """Cells are initialized with None color."""
        canvas = Canvas(2, 2)

        for row in canvas.cells:
            for cell in row:
                assert cell.color is None

    def test_zero_width_raises(self):
        """Canvas rejects zero width."""
        with pytest.raises(ValueError, match="must be positive"):
            Canvas(0, 10)

    def test_zero_height_raises(self):
        """Canvas rejects zero height."""
        with pytest.raises(ValueError, match="must be positive"):
            Canvas(10, 0)

    def test_negative_width_raises(self):
        """Canvas rejects negative width."""
        with pytest.raises(ValueError, match="must be positive"):
            Canvas(-5, 10)

    def test_negative_height_raises(self):
        """Canvas rejects negative height."""
        with pytest.raises(ValueError, match="must be positive"):
            Canvas(10, -5)

    def test_both_negative_raises(self):
        """Canvas rejects both dimensions negative."""
        with pytest.raises(ValueError, match="must be positive"):
            Canvas(-5, -10)

    def test_minimum_dimensions(self):
        """Canvas accepts minimum 1x1 dimensions."""
        canvas = Canvas(1, 1)

        assert canvas.width == 1
        assert canvas.height == 1
        assert len(canvas.cells) == 1
        assert len(canvas.cells[0]) == 1

    def test_large_dimensions(self):
        """Canvas accepts large dimensions."""
        canvas = Canvas(100, 200)

        assert canvas.width == 100
        assert canvas.height == 200


class TestCanvasFromSet:
    """Tests for Canvas.from_set factory method."""

    def test_from_set_valid_id(self):
        """from_set creates canvas with correct dimensions for known palette."""
        canvas = Canvas.from_set("marilyn_48x48")

        assert canvas.width == 48
        assert canvas.height == 48

    def test_from_set_world_map(self):
        """from_set creates correct dimensions for World Map palette."""
        canvas = Canvas.from_set("world_map_128x80")

        assert canvas.width == 128
        assert canvas.height == 80

    def test_from_set_invalid_id_raises(self):
        """from_set raises ValueError for unknown set ID."""
        with pytest.raises(ValueError, match="not found"):
            Canvas.from_set("nonexistent_palette")


class TestCanvasFromCells:
    """Tests for Canvas.from_cells factory method."""

    def test_from_cells_valid(self):
        """from_cells creates canvas from 2D cell list."""
        red = Color((255, 0, 0), name="Red")
        cells = [
            [Cell(red, 0, 0), Cell(red, 1, 0)],
            [Cell(red, 0, 1), Cell(red, 1, 1)],
        ]
        canvas = Canvas.from_cells(cells)

        assert canvas.width == 2
        assert canvas.height == 2
        assert canvas.cells is cells

    def test_from_cells_empty_raises(self):
        """from_cells rejects empty list."""
        with pytest.raises(ValueError, match="empty"):
            Canvas.from_cells([])

    def test_from_cells_empty_row_raises(self):
        """from_cells rejects list with empty first row."""
        with pytest.raises(ValueError, match="empty"):
            Canvas.from_cells([[]])

    def test_from_cells_inconsistent_row_lengths_raises(self):
        """from_cells rejects rows with inconsistent lengths."""
        red = Color((255, 0, 0), name="Red")
        cells = [
            [Cell(red, 0, 0), Cell(red, 1, 0)],  # 2 cells
            [Cell(red, 0, 1)],  # 1 cell
        ]

        with pytest.raises(ValueError, match="Inconsistent row lengths"):
            Canvas.from_cells(cells)

    def test_from_cells_single_cell(self):
        """from_cells works with single cell."""
        red = Color((255, 0, 0), name="Red")
        cells = [[Cell(red, 0, 0)]]
        canvas = Canvas.from_cells(cells)

        assert canvas.width == 1
        assert canvas.height == 1


class TestCanvasGetCell:
    """Tests for Canvas.get_cell."""

    def test_get_cell_valid_coordinates(self):
        """get_cell returns correct cell for valid coordinates."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")
        canvas.set_cell(2, 3, red)

        cell = canvas.get_cell(2, 3)

        assert cell.x == 2
        assert cell.y == 3
        assert cell.color.rgb == (255, 0, 0)

    def test_get_cell_corners(self):
        """get_cell works at all corners."""
        canvas = Canvas(10, 20)

        # Top-left
        assert canvas.get_cell(0, 0) is not None
        # Top-right
        assert canvas.get_cell(9, 0) is not None
        # Bottom-left
        assert canvas.get_cell(0, 19) is not None
        # Bottom-right
        assert canvas.get_cell(9, 19) is not None

    def test_get_cell_negative_x_raises(self):
        """get_cell raises IndexError for negative x."""
        canvas = Canvas(5, 5)

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.get_cell(-1, 0)

    def test_get_cell_negative_y_raises(self):
        """get_cell raises IndexError for negative y."""
        canvas = Canvas(5, 5)

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.get_cell(0, -1)

    def test_get_cell_x_too_large_raises(self):
        """get_cell raises IndexError for x >= width."""
        canvas = Canvas(5, 5)

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.get_cell(5, 0)

    def test_get_cell_y_too_large_raises(self):
        """get_cell raises IndexError for y >= height."""
        canvas = Canvas(5, 5)

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.get_cell(0, 5)


class TestCanvasSetCell:
    """Tests for Canvas.set_cell."""

    def test_set_cell_valid(self):
        """set_cell updates cell color at valid coordinates."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        canvas.set_cell(2, 3, red)

        assert canvas.get_cell(2, 3).color.rgb == (255, 0, 0)

    def test_set_cell_overwrites(self):
        """set_cell overwrites existing color."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")

        canvas.set_cell(2, 3, red)
        canvas.set_cell(2, 3, blue)

        assert canvas.get_cell(2, 3).color.rgb == (0, 0, 255)

    def test_set_cell_preserves_coordinates(self):
        """set_cell preserves cell coordinates."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        canvas.set_cell(2, 3, red)
        cell = canvas.get_cell(2, 3)

        assert cell.x == 2
        assert cell.y == 3

    def test_set_cell_negative_x_raises(self):
        """set_cell raises IndexError for negative x."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.set_cell(-1, 0, red)

    def test_set_cell_negative_y_raises(self):
        """set_cell raises IndexError for negative y."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.set_cell(0, -1, red)

    def test_set_cell_x_too_large_raises(self):
        """set_cell raises IndexError for x >= width."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.set_cell(5, 0, red)

    def test_set_cell_y_too_large_raises(self):
        """set_cell raises IndexError for y >= height."""
        canvas = Canvas(5, 5)
        red = Color((255, 0, 0), name="Red")

        with pytest.raises(IndexError, match="out of bounds"):
            canvas.set_cell(0, 5, red)


class TestCanvasToArray:
    """Tests for Canvas.to_array."""

    def test_to_array_shape(self):
        """to_array returns array with correct shape."""
        canvas = Canvas(10, 20)
        red = Color((255, 0, 0), name="Red")

        # Fill all cells
        for y in range(20):
            for x in range(10):
                canvas.set_cell(x, y, red)

        array = canvas.to_array()

        assert array.shape == (20, 10, 3)  # (height, width, channels)
        assert array.dtype == np.uint8

    def test_to_array_values(self):
        """to_array contains correct RGB values."""
        canvas = Canvas(2, 2)
        red = Color((255, 0, 0), name="Red")
        green = Color((0, 255, 0), name="Green")
        blue = Color((0, 0, 255), name="Blue")
        white = Color((255, 255, 255), name="White")

        canvas.set_cell(0, 0, red)
        canvas.set_cell(1, 0, green)
        canvas.set_cell(0, 1, blue)
        canvas.set_cell(1, 1, white)

        array = canvas.to_array()

        assert tuple(array[0, 0]) == (255, 0, 0)
        assert tuple(array[0, 1]) == (0, 255, 0)
        assert tuple(array[1, 0]) == (0, 0, 255)
        assert tuple(array[1, 1]) == (255, 255, 255)

    def test_to_array_unassigned_cell_raises(self):
        """to_array raises ValueError if any cell has no color."""
        canvas = Canvas(2, 2)
        red = Color((255, 0, 0), name="Red")

        # Only set one cell
        canvas.set_cell(0, 0, red)

        with pytest.raises(ValueError, match="no color assigned"):
            canvas.to_array()


class TestCanvasRepr:
    """Tests for Canvas.__repr__."""

    def test_repr(self):
        """__repr__ shows width and height."""
        canvas = Canvas(48, 48)

        repr_str = repr(canvas)

        assert "Canvas" in repr_str
        assert "width=48" in repr_str
        assert "height=48" in repr_str
