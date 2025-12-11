"""Unit tests for the Cell class.

Tests cover initialization, attributes, and string representation
with comprehensive coverage of valid inputs and edge cases.
"""

from mosaicpic import Color
from mosaicpic.models import Cell


class TestCellInit:
    """Tests for Cell.__init__."""

    def test_init_with_color_only(self):
        """Cell accepts color without coordinates."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red)

        assert cell.color == red
        assert cell.x is None
        assert cell.y is None
        assert cell.pinned is False
        assert cell.delta_e == 0.0

    def test_init_with_coordinates(self):
        """Cell accepts color with coordinates."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red, x=5, y=10)

        assert cell.color == red
        assert cell.x == 5
        assert cell.y == 10

    def test_init_with_all_parameters(self):
        """Cell accepts all parameters."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red, x=5, y=10, pinned=True, delta_e=15.5)

        assert cell.color == red
        assert cell.x == 5
        assert cell.y == 10
        assert cell.pinned is True
        assert cell.delta_e == 15.5

    def test_init_with_zero_coordinates(self):
        """Cell accepts zero coordinates."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red, x=0, y=0)

        assert cell.x == 0
        assert cell.y == 0

    def test_init_with_large_coordinates(self):
        """Cell accepts large coordinate values."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red, x=1000, y=2000)

        assert cell.x == 1000
        assert cell.y == 2000

    def test_init_pinned_default_false(self):
        """Cell pinned defaults to False."""
        cell = Cell(Color((0, 0, 0)))

        assert cell.pinned is False

    def test_init_delta_e_default_zero(self):
        """Cell delta_e defaults to 0.0."""
        cell = Cell(Color((0, 0, 0)))

        assert cell.delta_e == 0.0


class TestCellAttributes:
    """Tests for Cell attribute modification."""

    def test_color_is_mutable(self):
        """Cell color can be changed."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")
        cell = Cell(red)

        cell.color = blue

        assert cell.color == blue

    def test_pinned_is_mutable(self):
        """Cell pinned status can be changed."""
        cell = Cell(Color((0, 0, 0)))

        cell.pinned = True
        assert cell.pinned is True

        cell.pinned = False
        assert cell.pinned is False

    def test_delta_e_is_mutable(self):
        """Cell delta_e can be changed."""
        cell = Cell(Color((0, 0, 0)))

        cell.delta_e = 25.7
        assert cell.delta_e == 25.7

    def test_coordinates_are_mutable(self):
        """Cell coordinates can be changed."""
        cell = Cell(Color((0, 0, 0)), x=0, y=0)

        cell.x = 10
        cell.y = 20

        assert cell.x == 10
        assert cell.y == 20


class TestCellWithNoneColor:
    """Tests for Cell with None color (uninitialized state)."""

    def test_init_with_none_color(self):
        """Cell can be created with None color."""
        cell = Cell(None, x=0, y=0)  # type: ignore[arg-type]

        assert cell.color is None
        assert cell.x == 0
        assert cell.y == 0


class TestCellRepr:
    """Tests for Cell.__repr__."""

    def test_repr_basic(self):
        """__repr__ shows color and coordinates."""
        red = Color((255, 0, 0), name="Red")
        cell = Cell(red, x=5, y=10)

        repr_str = repr(cell)

        assert "Cell" in repr_str
        assert "color=" in repr_str
        assert "x=5" in repr_str
        assert "y=10" in repr_str

    def test_repr_includes_pinned_when_true(self):
        """__repr__ includes pinned=True when pinned."""
        cell = Cell(Color((0, 0, 0)), x=0, y=0, pinned=True)

        repr_str = repr(cell)

        assert "pinned=True" in repr_str

    def test_repr_excludes_pinned_when_false(self):
        """__repr__ excludes pinned when False."""
        cell = Cell(Color((0, 0, 0)), x=0, y=0, pinned=False)

        repr_str = repr(cell)

        assert "pinned" not in repr_str

    def test_repr_includes_delta_e_when_positive(self):
        """__repr__ includes delta_e when > 0."""
        cell = Cell(Color((0, 0, 0)), x=0, y=0, delta_e=15.5)

        repr_str = repr(cell)

        assert "delta_e=" in repr_str
        assert "15.50" in repr_str

    def test_repr_excludes_delta_e_when_zero(self):
        """__repr__ excludes delta_e when 0."""
        cell = Cell(Color((0, 0, 0)), x=0, y=0, delta_e=0.0)

        repr_str = repr(cell)

        assert "delta_e" not in repr_str

    def test_repr_with_all_attributes(self):
        """__repr__ shows all attributes when set."""
        cell = Cell(Color((255, 0, 0), name="Red"), x=5, y=10, pinned=True, delta_e=25.7)

        repr_str = repr(cell)

        assert "Cell" in repr_str
        assert "x=5" in repr_str
        assert "y=10" in repr_str
        assert "pinned=True" in repr_str
        assert "delta_e=" in repr_str


class TestCellEdgeCases:
    """Tests for Cell edge cases."""

    def test_delta_e_negative(self):
        """Cell accepts negative delta_e (though unusual)."""
        cell = Cell(Color((0, 0, 0)), delta_e=-1.0)

        assert cell.delta_e == -1.0

    def test_delta_e_very_large(self):
        """Cell accepts very large delta_e values."""
        cell = Cell(Color((0, 0, 0)), delta_e=1000.0)

        assert cell.delta_e == 1000.0

    def test_delta_e_very_small(self):
        """Cell accepts very small positive delta_e values."""
        cell = Cell(Color((0, 0, 0)), delta_e=0.001)

        assert cell.delta_e == 0.001
