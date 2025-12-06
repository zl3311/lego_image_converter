"""Unit tests for export data models: BOMEntry and CellData.

Tests cover initialization, attributes, and post_init behavior
with comprehensive coverage of valid inputs.
"""

from legopic import Color
from legopic.models import BOMEntry, CellData, Element


class TestBOMEntryInit:
    """Tests for BOMEntry initialization."""

    def test_init_required_params(self):
        """BOMEntry accepts required parameters."""
        red = Color((255, 0, 0), name="Red")
        entry = BOMEntry(color=red, count_needed=50, in_palette=True)

        assert entry.color == red
        assert entry.count_needed == 50
        assert entry.in_palette is True
        assert entry.elements == []  # Default empty list

    def test_init_with_elements(self):
        """BOMEntry accepts elements list."""
        red = Color((255, 0, 0), name="Red")
        elem = Element(element_id=1234567, design_id=98138, variant_id=1, count=100)
        entry = BOMEntry(color=red, count_needed=50, in_palette=True, elements=[elem])

        assert len(entry.elements) == 1
        assert entry.elements[0] == elem

    def test_init_multiple_elements(self):
        """BOMEntry accepts multiple element variants."""
        red = Color((255, 0, 0), name="Red")
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1, count=50)
        elem2 = Element(element_id=7654321, design_id=98138, variant_id=2, count=50)
        entry = BOMEntry(color=red, count_needed=100, in_palette=True, elements=[elem1, elem2])

        assert len(entry.elements) == 2

    def test_init_out_of_palette(self):
        """BOMEntry supports out-of-palette colors."""
        custom = Color((123, 45, 67), name="Custom")
        entry = BOMEntry(color=custom, count_needed=10, in_palette=False, elements=[])

        assert entry.in_palette is False
        assert entry.elements == []

    def test_init_none_elements_becomes_empty_list(self):
        """BOMEntry converts None elements to empty list via post_init."""
        red = Color((255, 0, 0), name="Red")
        entry = BOMEntry(color=red, count_needed=50, in_palette=True, elements=None)

        assert entry.elements == []


class TestBOMEntryAttributes:
    """Tests for BOMEntry attribute values."""

    def test_count_needed_zero(self):
        """BOMEntry accepts count_needed of zero."""
        red = Color((255, 0, 0), name="Red")
        entry = BOMEntry(color=red, count_needed=0, in_palette=True)

        assert entry.count_needed == 0

    def test_count_needed_large(self):
        """BOMEntry accepts large count_needed values."""
        red = Color((255, 0, 0), name="Red")
        entry = BOMEntry(color=red, count_needed=10000, in_palette=True)

        assert entry.count_needed == 10000


class TestBOMEntryDataclass:
    """Tests for BOMEntry dataclass behavior."""

    def test_is_dataclass(self):
        """BOMEntry is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(BOMEntry)

    def test_equality(self):
        """BOMEntry instances with same values are equal."""
        red = Color((255, 0, 0), name="Red")
        entry1 = BOMEntry(color=red, count_needed=50, in_palette=True)
        entry2 = BOMEntry(color=red, count_needed=50, in_palette=True)

        assert entry1 == entry2

    def test_inequality_different_count(self):
        """BOMEntry instances with different count are not equal."""
        red = Color((255, 0, 0), name="Red")
        entry1 = BOMEntry(color=red, count_needed=50, in_palette=True)
        entry2 = BOMEntry(color=red, count_needed=60, in_palette=True)

        assert entry1 != entry2


class TestCellDataInit:
    """Tests for CellData initialization."""

    def test_init_all_params(self):
        """CellData accepts all parameters."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)

        assert cell_data.x == 5
        assert cell_data.y == 10
        assert cell_data.color == red
        assert cell_data.delta_e == 15.5
        assert cell_data.pinned is True

    def test_init_unpinned(self):
        """CellData works with pinned=False."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=0, y=0, color=red, delta_e=0.0, pinned=False)

        assert cell_data.pinned is False

    def test_init_zero_delta_e(self):
        """CellData accepts delta_e of zero."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=0, y=0, color=red, delta_e=0.0, pinned=False)

        assert cell_data.delta_e == 0.0

    def test_init_zero_coordinates(self):
        """CellData accepts zero coordinates."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=0, y=0, color=red, delta_e=0.0, pinned=False)

        assert cell_data.x == 0
        assert cell_data.y == 0


class TestCellDataAttributes:
    """Tests for CellData attribute values."""

    def test_large_coordinates(self):
        """CellData accepts large coordinate values."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=1000, y=2000, color=red, delta_e=0.0, pinned=False)

        assert cell_data.x == 1000
        assert cell_data.y == 2000

    def test_large_delta_e(self):
        """CellData accepts large delta_e values."""
        red = Color((255, 0, 0), name="Red")
        cell_data = CellData(x=0, y=0, color=red, delta_e=100.0, pinned=False)

        assert cell_data.delta_e == 100.0


class TestCellDataDataclass:
    """Tests for CellData dataclass behavior."""

    def test_is_dataclass(self):
        """CellData is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(CellData)

    def test_equality(self):
        """CellData instances with same values are equal."""
        red = Color((255, 0, 0), name="Red")
        cell1 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)
        cell2 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)

        assert cell1 == cell2

    def test_inequality_different_position(self):
        """CellData instances with different positions are not equal."""
        red = Color((255, 0, 0), name="Red")
        cell1 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)
        cell2 = CellData(x=6, y=10, color=red, delta_e=15.5, pinned=True)

        assert cell1 != cell2

    def test_inequality_different_color(self):
        """CellData instances with different colors are not equal."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")
        cell1 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)
        cell2 = CellData(x=5, y=10, color=blue, delta_e=15.5, pinned=True)

        assert cell1 != cell2

    def test_inequality_different_pinned(self):
        """CellData instances with different pinned status are not equal."""
        red = Color((255, 0, 0), name="Red")
        cell1 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=True)
        cell2 = CellData(x=5, y=10, color=red, delta_e=15.5, pinned=False)

        assert cell1 != cell2


class TestExportModelsUseCases:
    """Tests for realistic use cases of export models."""

    def test_bom_entry_for_building_guide(self):
        """BOMEntry works for typical building guide output."""
        # Simulate output from session.get_bill_of_materials()
        white = Color((255, 255, 255), name="White")
        black = Color((33, 33, 33), name="Black")

        white_elem = Element(element_id=6284572, design_id=98138, variant_id=2, count=500)
        black_elem = Element(element_id=6284070, design_id=98138, variant_id=2, count=300)

        bom = [
            BOMEntry(color=white, count_needed=450, in_palette=True, elements=[white_elem]),
            BOMEntry(color=black, count_needed=250, in_palette=True, elements=[black_elem]),
        ]

        # Verify total tiles
        total = sum(entry.count_needed for entry in bom)
        assert total == 700

        # Verify all in palette
        assert all(entry.in_palette for entry in bom)

    def test_cell_data_for_grid_rendering(self):
        """CellData works for typical grid rendering output."""
        # Simulate output from session.get_grid_data()
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")

        grid = [
            [
                CellData(x=0, y=0, color=red, delta_e=5.2, pinned=False),
                CellData(x=1, y=0, color=blue, delta_e=3.1, pinned=True),
            ],
            [
                CellData(x=0, y=1, color=blue, delta_e=4.5, pinned=False),
                CellData(x=1, y=1, color=red, delta_e=2.8, pinned=False),
            ],
        ]

        # Verify grid structure
        assert len(grid) == 2
        assert len(grid[0]) == 2

        # Verify coordinates
        assert grid[0][0].x == 0 and grid[0][0].y == 0
        assert grid[1][1].x == 1 and grid[1][1].y == 1

        # Find pinned cells
        pinned_cells = [(cell.x, cell.y) for row in grid for cell in row if cell.pinned]
        assert pinned_cells == [(1, 0)]
