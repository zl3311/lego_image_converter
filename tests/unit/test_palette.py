"""Unit tests for the Palette class.

Tests cover initialization, factory methods, properties, and iteration
with comprehensive coverage of valid and invalid inputs.
"""

import numpy as np
import pytest

from legopic import Color, Palette
from legopic.models import Element


class TestPaletteInit:
    """Tests for Palette.__init__."""

    def test_init_with_color_list(self):
        """Palette accepts a list of Color objects."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((0, 255, 0), name="Green"),
        ]
        palette = Palette(colors)

        assert len(palette) == 2
        assert palette.colors[0].rgb == (255, 0, 0)
        assert palette.colors[1].rgb == (0, 255, 0)

    def test_init_with_color_element_dict(self):
        """Palette accepts a dict mapping Color to list of Elements."""
        red = Color((255, 0, 0), name="Red")
        green = Color((0, 255, 0), name="Green")

        color_elements = {
            red: [Element(element_id=1, design_id=98138, variant_id=1, count=10)],
            green: [Element(element_id=2, design_id=98138, variant_id=1, count=20)],
        }
        palette = Palette(color_elements)

        assert len(palette) == 2
        assert len(palette.elements) == 2

    def test_init_with_multiple_variants_per_color(self):
        """Palette handles multiple element variants for the same color."""
        white = Color((255, 255, 255), name="White")

        color_elements = {
            white: [
                Element(element_id=1, design_id=98138, variant_id=1, count=5),
                Element(element_id=2, design_id=98138, variant_id=2, count=10),
                Element(element_id=3, design_id=98138, variant_id=3, count=15),
            ],
        }
        palette = Palette(color_elements)

        assert len(palette) == 1
        assert len(palette.elements) == 3
        elements = palette.get_elements_for_color(white)
        assert len(elements) == 3
        assert sum(e.count for e in elements) == 30

    def test_init_empty_list_raises(self):
        """Palette rejects empty color list."""
        with pytest.raises(ValueError, match="at least one color"):
            Palette([])

    def test_init_empty_dict_raises(self):
        """Palette rejects empty color dict."""
        with pytest.raises(ValueError, match="at least one color"):
            Palette({})

    def test_init_single_color(self):
        """Palette accepts a single color."""
        colors = [Color((0, 0, 0), name="Black")]
        palette = Palette(colors)

        assert len(palette) == 1

    def test_init_deduplicates_colors_by_rgb(self):
        """Palette deduplicates colors with same RGB (list input)."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((255, 0, 0), name="Crimson"),  # Same RGB
            Color((0, 0, 255), name="Blue"),
        ]
        palette = Palette(colors)

        # Dict uses Color as key, which hashes by RGB
        # So duplicates overwrite
        assert len(palette) == 2


class TestPaletteFromSet:
    """Tests for Palette.from_set factory method."""

    def test_from_set_with_valid_id(self):
        """from_set loads palette from known LEGO set."""
        palette = Palette.from_set(31197)  # Andy Warhol

        assert len(palette) > 0
        assert len(palette.elements) > 0

    def test_from_set_elements_have_counts(self):
        """Elements loaded from set have inventory counts."""
        palette = Palette.from_set(31197)

        for element in palette.elements:
            assert element.count is not None
            assert element.count > 0

    def test_from_set_invalid_id_raises(self):
        """from_set raises ValueError for unknown set ID."""
        with pytest.raises(ValueError, match="not found"):
            Palette.from_set(99999)

    def test_from_set_no_id_loads_all_colors(self):
        """from_set with no ID loads all standard colors."""
        palette = Palette.from_set()

        # Should have many colors (41+ standard colors)
        assert len(palette) >= 30

    def test_from_set_standard_only_true(self):
        """from_set with standard_only=True excludes non-standard colors."""
        standard_palette = Palette.from_set(standard_only=True)
        all_palette = Palette.from_set(standard_only=False)

        # All colors includes transparent/metallic, so should be larger
        assert len(all_palette) >= len(standard_palette)

    def test_from_set_standard_only_false(self):
        """from_set with standard_only=False includes all colors."""
        palette = Palette.from_set(standard_only=False)

        # Should have more colors than standard-only
        assert len(palette) >= 30


class TestPaletteProperties:
    """Tests for Palette properties."""

    def test_colors_property(self):
        """colors property returns list of unique Color objects."""
        red = Color((255, 0, 0), name="Red")
        green = Color((0, 255, 0), name="Green")
        blue = Color((0, 0, 255), name="Blue")

        palette = Palette([red, green, blue])
        colors = palette.colors

        assert isinstance(colors, list)
        assert len(colors) == 3
        assert all(isinstance(c, Color) for c in colors)

    def test_elements_property_with_elements(self):
        """elements property returns flat list of all Elements."""
        red = Color((255, 0, 0), name="Red")
        color_elements = {
            red: [
                Element(element_id=1, design_id=98138, variant_id=1, count=5),
                Element(element_id=2, design_id=98138, variant_id=2, count=10),
            ],
        }
        palette = Palette(color_elements)

        elements = palette.elements
        assert len(elements) == 2
        assert all(isinstance(e, Element) for e in elements)

    def test_elements_property_without_elements(self):
        """elements property returns empty list for list-initialized palette."""
        palette = Palette([Color((255, 0, 0), name="Red")])

        elements = palette.elements
        assert elements == []


class TestPaletteGetElementsForColor:
    """Tests for Palette.get_elements_for_color."""

    def test_get_elements_for_existing_color(self):
        """Returns element list for color in palette."""
        red = Color((255, 0, 0), name="Red")
        elem = Element(element_id=1, design_id=98138, variant_id=1, count=5)

        palette = Palette({red: [elem]})
        result = palette.get_elements_for_color(red)

        assert result == [elem]

    def test_get_elements_for_nonexistent_color(self):
        """Returns empty list for color not in palette."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")

        palette = Palette([red])
        result = palette.get_elements_for_color(blue)

        assert result == []

    def test_get_elements_matches_by_rgb(self):
        """get_elements_for_color matches by RGB, not object identity."""
        red1 = Color((255, 0, 0), name="Red")
        elem = Element(element_id=1, design_id=98138, variant_id=1, count=5)

        palette = Palette({red1: [elem]})

        # Create new Color with same RGB but different name
        red2 = Color((255, 0, 0), name="Crimson")
        result = palette.get_elements_for_color(red2)

        assert result == [elem]


class TestPaletteConversions:
    """Tests for Palette conversion methods."""

    def test_to_rgb_array(self):
        """to_rgb_array returns numpy array of RGB values."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((0, 255, 0), name="Green"),
            Color((0, 0, 255), name="Blue"),
        ]
        palette = Palette(colors)

        array = palette.to_rgb_array()

        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 3)
        assert array.dtype == np.uint8
        assert tuple(array[0]) == (255, 0, 0)
        assert tuple(array[1]) == (0, 255, 0)
        assert tuple(array[2]) == (0, 0, 255)

    def test_to_rgb_list(self):
        """to_rgb_list returns list of RGB tuples."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((0, 255, 0), name="Green"),
        ]
        palette = Palette(colors)

        rgb_list = palette.to_rgb_list()

        assert rgb_list == [(255, 0, 0), (0, 255, 0)]


class TestPaletteDunderMethods:
    """Tests for Palette __len__, __iter__, __contains__, __repr__."""

    def test_len(self):
        """__len__ returns number of unique colors."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((0, 255, 0), name="Green"),
            Color((0, 0, 255), name="Blue"),
        ]
        palette = Palette(colors)

        assert len(palette) == 3

    def test_iter(self):
        """__iter__ yields colors in palette."""
        colors = [
            Color((255, 0, 0), name="Red"),
            Color((0, 255, 0), name="Green"),
        ]
        palette = Palette(colors)

        iterated = list(palette)

        assert len(iterated) == 2
        assert all(isinstance(c, Color) for c in iterated)

    def test_contains_true(self):
        """__contains__ returns True for color in palette."""
        red = Color((255, 0, 0), name="Red")
        palette = Palette([red])

        assert red in palette

    def test_contains_false(self):
        """__contains__ returns False for color not in palette."""
        red = Color((255, 0, 0), name="Red")
        blue = Color((0, 0, 255), name="Blue")
        palette = Palette([red])

        assert blue not in palette

    def test_contains_matches_by_rgb(self):
        """__contains__ matches by RGB, not object identity."""
        red1 = Color((255, 0, 0), name="Red")
        red2 = Color((255, 0, 0), name="Crimson")
        palette = Palette([red1])

        assert red2 in palette

    def test_repr_without_elements(self):
        """__repr__ shows color count for list-initialized palette."""
        palette = Palette([Color((255, 0, 0)), Color((0, 255, 0))])

        repr_str = repr(palette)

        assert "Palette" in repr_str
        assert "2 colors" in repr_str

    def test_repr_with_elements(self):
        """__repr__ shows color and element count for dict-initialized palette."""
        red = Color((255, 0, 0), name="Red")
        palette = Palette(
            {
                red: [
                    Element(element_id=1, design_id=98138, variant_id=1, count=5),
                    Element(element_id=2, design_id=98138, variant_id=2, count=10),
                ]
            }
        )

        repr_str = repr(palette)

        assert "Palette" in repr_str
        assert "1 colors" in repr_str
        assert "2 elements" in repr_str
