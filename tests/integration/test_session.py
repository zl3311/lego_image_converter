"""Integration tests for the full conversion pipeline."""

import numpy as np
import pytest

from legopic import Canvas, Color, ConversionSession, Image, Palette


class TestConversionSession:
    """End-to-end tests for the ConversionSession API."""

    @pytest.fixture
    def test_palette(self):
        """A test palette with basic colors."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
                Color((255, 255, 0), name="Yellow"),
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
            ]
        )

    @pytest.fixture
    def test_image(self):
        """A 100x100 test image with colored quadrants."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Top-left: Red
        array[:50, :50, :] = [255, 0, 0]
        # Top-right: Green
        array[:50, 50:, :] = [0, 255, 0]
        # Bottom-left: Blue
        array[50:, :50, :] = [0, 0, 255]
        # Bottom-right: Yellow
        array[50:, 50:, :] = [255, 255, 0]
        return Image(array)

    def test_session_convert_with_image_object(self, test_image, test_palette):
        """ConversionSession.convert works with Image object."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        canvas = session.convert()

        assert isinstance(canvas, Canvas)
        assert canvas.width == 10
        assert canvas.height == 10

    def test_session_preserves_quadrants(self, test_image, test_palette):
        """Session convert preserves color regions from source image."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()
        canvas = session.canvas

        # Check corners of each quadrant
        # Top-left should be red
        assert canvas.get_cell(0, 0).color.rgb == (255, 0, 0)
        # Top-right should be green
        assert canvas.get_cell(9, 0).color.rgb == (0, 255, 0)
        # Bottom-left should be blue
        assert canvas.get_cell(0, 9).color.rgb == (0, 0, 255)
        # Bottom-right should be yellow
        assert canvas.get_cell(9, 9).color.rgb == (255, 255, 0)

    def test_canvas_to_array(self, test_image, test_palette):
        """Canvas.to_array produces valid RGB array."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()
        array = session.canvas.to_array()

        assert array.shape == (10, 10, 3)
        assert array.dtype == np.uint8

    def test_all_profiles_produce_valid_output(self, test_image, test_palette):
        """All pipeline profiles produce valid Canvas objects."""
        for profile in ["classic", "sharp", "dithered"]:
            session = ConversionSession(test_image, test_palette, (10, 10))
            canvas = session.convert(profile)

            assert isinstance(canvas, Canvas)
            assert canvas.width == 10
            assert canvas.height == 10

            # All cells should have colors from the palette
            palette_rgbs = {c.rgb for c in test_palette.colors}
            for y in range(canvas.height):
                for x in range(canvas.width):
                    assert canvas.get_cell(x, y).color.rgb in palette_rgbs

    def test_session_similarity_score(self, test_image, test_palette):
        """Session provides aggregate similarity score."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        # Similarity score should be a non-negative float
        assert isinstance(session.similarity_score, float)
        assert session.similarity_score >= 0

    def test_session_pin_and_reconvert(self, test_image, test_palette):
        """Pinned cells survive reconversion."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        # Pin a cell and change its color
        white = Color((255, 255, 255), name="White")
        session.pin(0, 0, white)

        # Reconvert with different profile
        session.reconvert("classic", keep_pins=True)

        # Pinned cell should still be white
        assert session.canvas.get_cell(0, 0).color.rgb == (255, 255, 255)
        assert session.canvas.get_cell(0, 0).pinned is True

    def test_session_swap_color(self, test_image, test_palette):
        """swap_color replaces all instances of a color."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        # Get current color at (0, 0) - should be red
        old_color = session.canvas.get_cell(0, 0).color
        new_color = Color((255, 255, 255), name="White")

        # Swap all instances
        count = session.swap_color(old_color, new_color)

        # Should have swapped some cells
        assert count > 0
        # The cell should now be white
        assert session.canvas.get_cell(0, 0).color.rgb == (255, 255, 255)

    def test_session_bill_of_materials(self, test_image, test_palette):
        """get_bill_of_materials returns BOM entries."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        bom = session.get_bill_of_materials()

        assert len(bom) > 0
        total_count = sum(entry.count_needed for entry in bom)
        assert total_count == 10 * 10  # All cells accounted for

    def test_session_grid_data(self, test_image, test_palette):
        """get_grid_data returns 2D cell data."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        grid = session.get_grid_data()

        assert len(grid) == 10  # Height
        assert len(grid[0]) == 10  # Width
        assert grid[0][0].x == 0
        assert grid[0][0].y == 0

    def test_session_similarity_map(self, test_image, test_palette):
        """get_similarity_map returns per-cell delta E values."""
        session = ConversionSession(test_image, test_palette, (10, 10))
        session.convert()

        sim_map = session.get_similarity_map()

        assert len(sim_map) == 10  # Height
        assert len(sim_map[0]) == 10  # Width
        # All values should be non-negative floats
        for row in sim_map:
            for val in row:
                assert isinstance(val, float)
                assert val >= 0

    def test_session_with_set_palette(self, test_image):
        """Session works with Palette.from_set()."""
        # 31197 is Andy Warhol's Marilyn Monroe set
        palette = Palette.from_set(31197)
        session = ConversionSession(test_image, palette, (10, 10))
        canvas = session.convert()

        assert isinstance(canvas, Canvas)
        assert canvas.width == 10
        assert canvas.height == 10

    def test_session_export_bricklink_xml(self, test_image):
        """export_bricklink_xml returns valid XML string."""
        palette = Palette.from_set(31197)
        session = ConversionSession(test_image, palette, (10, 10))
        session.convert()

        xml = session.export_bricklink_xml()

        assert isinstance(xml, str)
        assert "<INVENTORY>" in xml
        assert "</INVENTORY>" in xml
        assert "<ITEM>" in xml
        assert "<ITEMTYPE>P</ITEMTYPE>" in xml
        assert "<ITEMID>98138</ITEMID>" in xml
        assert "<COLOR>" in xml
        assert "<MINQTY>" in xml

    def test_session_export_rebrickable_csv(self, test_image):
        """export_rebrickable_csv returns valid CSV string."""
        palette = Palette.from_set(31197)
        session = ConversionSession(test_image, palette, (10, 10))
        session.convert()

        csv = session.export_rebrickable_csv()

        assert isinstance(csv, str)
        lines = csv.split("\n")
        assert lines[0] == "Part,Color,Quantity"
        assert len(lines) > 1

        # Verify CSV row format (Part,Color,Quantity)
        for line in lines[1:]:
            parts = line.split(",")
            assert len(parts) == 3
            assert parts[0] == "98138"  # Design ID
            assert parts[1].isdigit()  # Rebrickable color ID
            assert parts[2].isdigit()  # Quantity

    def test_session_export_before_convert_raises(self, test_image):
        """Export methods raise RuntimeError if convert() not called."""
        palette = Palette.from_set(31197)
        session = ConversionSession(test_image, palette, (10, 10))

        with pytest.raises(RuntimeError, match="No conversion yet"):
            session.export_bricklink_xml()

        with pytest.raises(RuntimeError, match="No conversion yet"):
            session.export_rebrickable_csv()


class TestPipelineProfiles:
    """Tests for pipeline profile usage."""

    @pytest.fixture
    def test_palette(self):
        """A test palette with basic colors."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
            ]
        )

    @pytest.fixture
    def test_image(self):
        """A 100x100 test image."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[:, :, 0] = 255  # All red
        return Image(array)

    def test_invalid_profile_raises(self, test_image, test_palette):
        """Invalid profile name raises ValueError."""
        session = ConversionSession(test_image, test_palette, (10, 10))

        with pytest.raises(ValueError, match="Unknown profile"):
            session.convert("nonexistent_profile")

    def test_custom_pipeline(self, test_image, test_palette):
        """Custom pipeline can be passed to convert()."""
        from legopic.pipeline import (
            DitherConfig,
            DitherStep,
            Pipeline,
            PoolConfig,
            PoolStep,
        )

        custom_pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                DitherStep(DitherConfig()),
            ]
        )

        session = ConversionSession(test_image, test_palette, (10, 10))
        canvas = session.convert(custom_pipeline)

        assert isinstance(canvas, Canvas)
        assert canvas.width == 10
        assert canvas.height == 10
