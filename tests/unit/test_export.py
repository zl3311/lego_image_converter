"""Unit tests for the export module.

Tests cover external ID lookup, BrickLink XML export, and Rebrickable CSV export
with comprehensive coverage of valid inputs and error cases.
"""

import pytest

from mosaicpic import BOMEntry, Color, export_bricklink_xml, export_rebrickable_csv
from mosaicpic.data.loader import get_all_colors, get_color_external_ids


class TestGetColorExternalIds:
    """Tests for get_color_external_ids function."""

    def test_valid_color_returns_dict(self):
        """Returns dict with bl_id, rb_id, and design_id for valid color."""
        ids = get_color_external_ids("Black")

        assert "bl_id" in ids
        assert "rb_id" in ids
        assert "design_id" in ids

    def test_black_returns_correct_ids(self):
        """Black returns correct BrickLink and Rebrickable IDs."""
        ids = get_color_external_ids("Black")

        assert ids["bl_id"] == 11
        assert ids["rb_id"] == 0
        assert ids["design_id"] == 98138

    def test_white_returns_correct_ids(self):
        """White returns correct BrickLink and Rebrickable IDs."""
        ids = get_color_external_ids("White")

        assert ids["bl_id"] == 1
        assert ids["rb_id"] == 15
        assert ids["design_id"] == 98138

    def test_coral_returns_correct_ids(self):
        """Coral returns correct BrickLink and Rebrickable IDs."""
        ids = get_color_external_ids("Coral")

        assert ids["bl_id"] == 220
        assert ids["rb_id"] == 1056

    def test_trans_clear_returns_correct_ids(self):
        """Trans-Clear returns correct BrickLink and Rebrickable IDs."""
        ids = get_color_external_ids("Trans-Clear")

        assert ids["bl_id"] == 12
        assert ids["rb_id"] == 80

    def test_invalid_color_raises(self):
        """Invalid color name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_color_external_ids("InvalidColorName")

    def test_returns_copy(self):
        """Returns a copy of the data, not a reference."""
        ids1 = get_color_external_ids("Black")
        ids2 = get_color_external_ids("Black")
        ids1["bl_id"] = 9999

        assert ids2["bl_id"] == 11


class TestExportBricklinkXml:
    """Tests for export_bricklink_xml function."""

    def test_empty_bom_returns_empty_inventory(self):
        """Empty BOM returns inventory with no items."""
        xml = export_bricklink_xml([])

        assert xml == "<INVENTORY>\n</INVENTORY>"

    def test_single_item_xml_structure(self):
        """Single BOM entry produces correct XML structure."""
        black = Color((33, 33, 33), name="Black")
        bom = [BOMEntry(color=black, count_needed=100, in_palette=True)]

        xml = export_bricklink_xml(bom)

        assert "<INVENTORY>" in xml
        assert "</INVENTORY>" in xml
        assert "<ITEM>" in xml
        assert "</ITEM>" in xml
        assert "<ITEMTYPE>P</ITEMTYPE>" in xml
        assert "<ITEMID>98138</ITEMID>" in xml
        assert "<COLOR>11</COLOR>" in xml
        assert "<MINQTY>100</MINQTY>" in xml

    def test_multiple_items(self):
        """Multiple BOM entries produce multiple items."""
        black = Color((33, 33, 33), name="Black")
        white = Color((255, 255, 255), name="White")
        bom = [
            BOMEntry(color=black, count_needed=655, in_palette=True),
            BOMEntry(color=white, count_needed=598, in_palette=True),
        ]

        xml = export_bricklink_xml(bom)

        assert xml.count("<ITEM>") == 2
        assert xml.count("</ITEM>") == 2
        assert "<COLOR>11</COLOR>" in xml
        assert "<COLOR>1</COLOR>" in xml
        assert "<MINQTY>655</MINQTY>" in xml
        assert "<MINQTY>598</MINQTY>" in xml

    def test_transparent_color(self):
        """Transparent colors export correctly."""
        trans_clear = Color((238, 238, 238), name="Trans-Clear")
        bom = [BOMEntry(color=trans_clear, count_needed=50, in_palette=True)]

        xml = export_bricklink_xml(bom)

        assert "<COLOR>12</COLOR>" in xml

    def test_color_without_name_raises(self):
        """Color without name raises ValueError."""
        unnamed = Color((100, 100, 100))
        bom = [BOMEntry(color=unnamed, count_needed=10, in_palette=True)]

        with pytest.raises(ValueError, match="no name"):
            export_bricklink_xml(bom)


class TestExportRebrickableCsv:
    """Tests for export_rebrickable_csv function."""

    def test_empty_bom_returns_header_only(self):
        """Empty BOM returns CSV with header only."""
        csv = export_rebrickable_csv([])

        assert csv == "Part,Color,Quantity"

    def test_single_item_csv_structure(self):
        """Single BOM entry produces correct CSV row."""
        black = Color((33, 33, 33), name="Black")
        bom = [BOMEntry(color=black, count_needed=100, in_palette=True)]

        csv = export_rebrickable_csv(bom)
        lines = csv.split("\n")

        assert len(lines) == 2
        assert lines[0] == "Part,Color,Quantity"
        assert lines[1] == "98138,0,100"

    def test_multiple_items(self):
        """Multiple BOM entries produce multiple rows."""
        black = Color((33, 33, 33), name="Black")
        white = Color((255, 255, 255), name="White")
        bom = [
            BOMEntry(color=black, count_needed=655, in_palette=True),
            BOMEntry(color=white, count_needed=598, in_palette=True),
        ]

        csv = export_rebrickable_csv(bom)
        lines = csv.split("\n")

        assert len(lines) == 3
        assert lines[0] == "Part,Color,Quantity"
        assert lines[1] == "98138,0,655"
        assert lines[2] == "98138,15,598"

    def test_transparent_color(self):
        """Transparent colors export correctly."""
        trans_clear = Color((238, 238, 238), name="Trans-Clear")
        bom = [BOMEntry(color=trans_clear, count_needed=50, in_palette=True)]

        csv = export_rebrickable_csv(bom)
        lines = csv.split("\n")

        assert lines[1] == "98138,80,50"

    def test_color_without_name_raises(self):
        """Color without name raises ValueError."""
        unnamed = Color((100, 100, 100))
        bom = [BOMEntry(color=unnamed, count_needed=10, in_palette=True)]

        with pytest.raises(ValueError, match="no name"):
            export_rebrickable_csv(bom)


class TestAllColorsHaveExternalIds:
    """Tests to verify all colors in the database have external IDs."""

    def test_all_standard_colors_have_ids(self):
        """All standard colors have BrickLink and Rebrickable IDs."""
        colors = get_all_colors(standard_only=True)

        for color in colors:
            ids = get_color_external_ids(color.name)
            assert ids["bl_id"] is not None, f"{color.name} missing BrickLink ID"
            assert ids["rb_id"] is not None, f"{color.name} missing Rebrickable ID"

    def test_all_colors_have_ids(self):
        """All colors including special types have BrickLink and Rebrickable IDs."""
        colors = get_all_colors(standard_only=False)

        for color in colors:
            ids = get_color_external_ids(color.name)
            assert ids["bl_id"] is not None, f"{color.name} missing BrickLink ID"
            assert ids["rb_id"] is not None, f"{color.name} missing Rebrickable ID"
