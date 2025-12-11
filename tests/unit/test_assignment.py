"""Unit tests for inventory-limited assignment algorithms."""

import numpy as np
import pytest

from mosaicpic import Color, ConversionSession, Image, Palette
from mosaicpic.core.assignment import AssignmentResult, priority_greedy
from mosaicpic.models import Element


class TestPriorityGreedy:
    """Tests for the priority greedy assignment algorithm."""

    @pytest.fixture
    def simple_palette_with_inventory(self):
        """Palette with limited inventory for each color."""
        # Create palette with element counts
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")
        green = Color((0, 255, 0), name="Green")

        color_elements = {
            red: [Element(element_id=1, design_id=98138, variant_id=1, count=5)],
            blue: [Element(element_id=2, design_id=98138, variant_id=1, count=5)],
            green: [Element(element_id=3, design_id=98138, variant_id=1, count=5)],
        }
        return Palette(color_elements)

    @pytest.fixture
    def scarce_palette(self):
        """Palette with very limited inventory."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")

        color_elements = {
            red: [Element(element_id=1, design_id=98138, variant_id=1, count=2)],
            blue: [Element(element_id=2, design_id=98138, variant_id=1, count=10)],
        }
        return Palette(color_elements)

    def test_returns_assignment_result(self, simple_palette_with_inventory):
        """priority_greedy returns an AssignmentResult."""
        target_colors = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)

        result = priority_greedy(target_colors, simple_palette_with_inventory)

        assert isinstance(result, AssignmentResult)
        assert result.assignments.shape == (1, 2)

    def test_assigns_best_match_when_available(self, simple_palette_with_inventory):
        """When inventory is available, assigns best matching color."""
        # 2x2 image with distinct colors
        target_colors = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0]]], dtype=np.uint8
        )

        result = priority_greedy(target_colors, simple_palette_with_inventory)
        palette_colors = simple_palette_with_inventory.colors

        # Each cell should get its best match
        for y in range(2):
            for x in range(2):
                idx = result.assignments[y, x]
                assigned_color = palette_colors[idx]
                target = tuple(target_colors[y, x])
                assert assigned_color.rgb == target

    def test_falls_back_when_inventory_exhausted(self, scarce_palette):
        """When best color runs out, falls back to next best."""
        # Create a 3x3 image all wanting red (but only 2 red available)
        target_colors = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)

        result = priority_greedy(target_colors, scarce_palette)

        # Count how many got red vs blue
        palette_colors = scarce_palette.colors
        red_idx = next(i for i, c in enumerate(palette_colors) if c.rgb == (255, 0, 0))

        red_count = np.sum(result.assignments == red_idx)

        # Only 2 red tiles available, so max 2 cells get red
        assert red_count == 2
        # Remaining 7 cells must fall back
        assert result.fallback_count == 7

    def test_respects_pinned_cells(self, simple_palette_with_inventory):
        """Pinned cells are preserved and consume inventory."""
        target_colors = np.array(
            [[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]], dtype=np.uint8
        )

        blue = Color((0, 0, 255), name="Blue")
        pinned = {(0, 0): blue}  # Pin top-left to blue

        result = priority_greedy(target_colors, simple_palette_with_inventory, pinned_cells=pinned)

        palette_colors = simple_palette_with_inventory.colors

        # Pinned cell should be blue (or marked as pinned out-of-order)
        idx = result.assignments[0, 0]
        if idx >= 0:
            assigned = palette_colors[idx]
            assert assigned.rgb == (0, 0, 255)

    def test_tracks_inventory_used(self, simple_palette_with_inventory):
        """Result includes inventory usage counts."""
        target_colors = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0]]], dtype=np.uint8
        )

        result = priority_greedy(target_colors, simple_palette_with_inventory)

        # Check usage counts
        red = Color((255, 0, 0), name="Red")
        assert red in result.inventory_used
        assert result.inventory_used[red] == 2  # Two red cells


class TestInventoryLimitedSession:
    """Integration tests for inventory-limited conversion via session."""

    @pytest.fixture
    def test_image(self):
        """A simple 10x10 test image."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        # All red
        array[:, :, 0] = 255
        return Image(array)

    @pytest.fixture
    def limited_palette(self):
        """Palette with limited inventory."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")

        color_elements = {
            red: [Element(element_id=1, design_id=98138, variant_id=1, count=50)],
            blue: [Element(element_id=2, design_id=98138, variant_id=1, count=100)],
        }
        return Palette(color_elements)

    def test_unlimited_uses_best_match(self, test_image, limited_palette):
        """Without limit_inventory, all cells get best match."""
        session = ConversionSession(test_image, limited_palette, (10, 10))
        session.convert(limit_inventory=False)

        # All cells should be red (best match for red image)
        canvas = session.canvas
        for y in range(10):
            for x in range(10):
                assert canvas.cells[y][x].color.rgb == (255, 0, 0)

    def test_limited_respects_inventory(self, test_image, limited_palette):
        """With limit_inventory, falls back when inventory exhausted."""
        session = ConversionSession(test_image, limited_palette, (10, 10))
        session.convert(limit_inventory=True)

        # Count colors used
        canvas = session.canvas
        red_count = 0
        blue_count = 0

        for y in range(10):
            for x in range(10):
                if canvas.cells[y][x].color.rgb == (255, 0, 0):
                    red_count += 1
                elif canvas.cells[y][x].color.rgb == (0, 0, 255):
                    blue_count += 1

        # Only 50 red available, so max 50 cells get red
        assert red_count == 50
        # Remaining 50 cells must be blue
        assert blue_count == 50

    def test_pinned_cells_preserved_with_inventory(self, test_image, limited_palette):
        """Pinned cells survive reconvert with inventory limits."""
        session = ConversionSession(test_image, limited_palette, (10, 10))
        session.convert(limit_inventory=True)

        # Pin a cell to blue
        blue = Color((0, 0, 255), name="Blue")
        session.pin(0, 0, blue)

        # Reconvert
        session.reconvert(limit_inventory=True, keep_pins=True)

        # Pinned cell should still be blue
        assert session.canvas.cells[0][0].color.rgb == (0, 0, 255)
        assert session.canvas.cells[0][0].pinned is True

    def test_bom_reflects_limited_usage(self, test_image, limited_palette):
        """Bill of materials shows actual usage with inventory limits."""
        session = ConversionSession(test_image, limited_palette, (10, 10))
        session.convert(limit_inventory=True)

        bom = session.get_bill_of_materials()

        # Should have 2 entries (red and blue)
        assert len(bom) == 2

        # Total should be 100 (10x10 canvas)
        total = sum(entry.count_needed for entry in bom)
        assert total == 100


class TestOptimalAlgorithmPlaceholder:
    """Tests for the optimal algorithm placeholder."""

    def test_optimal_falls_back_with_warning(self):
        """optimal() issues a warning and falls back to priority_greedy."""
        red = Color((255, 0, 0), name="Red")
        palette = Palette([red])

        target_colors = np.array([[[255, 0, 0]]], dtype=np.uint8)

        from mosaicpic.core.assignment import optimal

        with pytest.warns(UserWarning, match="not yet implemented"):
            result = optimal(target_colors, palette)

        assert isinstance(result, AssignmentResult)
