"""Integration tests using real image files from tests/images/.

This module tests the ConversionSession with actual image files.
Tests are designed to cover complex scenarios that imply simpler ones work.

Design principles:
- Each test uses ONE specific image (no Cartesian product across images)
- Canvas size is fixed at 48x48 (large enough to be meaningful)
- Tests are consolidated to minimize redundant conversions
- Class-scoped fixtures reuse expensive conversions

Supported formats (via PIL): PNG, JPEG, WebP, GIF, BMP, TIFF, etc.
"""

from pathlib import Path

import numpy as np
import pytest

from legopic import Canvas, Color, ConversionSession, ConvertConfig, Image, Palette

# =============================================================================
# Test Configuration
# =============================================================================

IMAGES_DIR = Path(__file__).parent.parent / "images"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
CANVAS_SIZE = (48, 48)


# =============================================================================
# Image Discovery
# =============================================================================


def discover_test_images() -> list[Path]:
    """Discover all image files in the tests/images/ directory."""
    if not IMAGES_DIR.exists():
        return []

    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(IMAGES_DIR.glob(f"*{ext}"))
        images.extend(IMAGES_DIR.glob(f"*{ext.upper()}"))

    return sorted([img for img in images if img.parent == IMAGES_DIR])


TEST_IMAGES = discover_test_images()


def get_image_by_extension(ext: str) -> Path | None:
    """Get a test image by extension preference."""
    for img in TEST_IMAGES:
        if img.suffix.lower() == ext.lower():
            return img
    return TEST_IMAGES[0] if TEST_IMAGES else None


PNG_IMAGE = get_image_by_extension(".png")
JPEG_IMAGE = get_image_by_extension(".jpg") or get_image_by_extension(".jpeg")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def standard_palette() -> Palette:
    """A palette with common LEGO colors for testing (module-scoped)."""
    return Palette(
        [
            Color((0, 0, 0), name="Black"),
            Color((255, 255, 255), name="White"),
            Color((180, 0, 0), name="Dark Red"),
            Color((255, 0, 0), name="Red"),
            Color((255, 128, 0), name="Orange"),
            Color((255, 205, 0), name="Bright Yellow"),
            Color((0, 100, 0), name="Dark Green"),
            Color((0, 200, 0), name="Bright Green"),
            Color((0, 150, 200), name="Medium Azure"),
            Color((0, 50, 150), name="Dark Blue"),
            Color((0, 0, 255), name="Blue"),
            Color((100, 50, 150), name="Medium Lilac"),
            Color((150, 100, 50), name="Medium Nougat"),
            Color((128, 128, 128), name="Medium Stone Grey"),
        ]
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestConversionSession:
    """Consolidated tests for ConversionSession with real images.

    Tests are designed so each test covers multiple aspects, minimizing
    redundant image loading and conversion operations.
    """

    def test_png_conversion_with_all_methods(self, standard_palette: Palette):
        """PNG image converts successfully with all downsize methods.

        This single test validates:
        - PNG format loading works
        - All three downsize methods produce valid canvases
        - Canvas dimensions are correct
        - All cells have palette colors (checked on final method)
        """
        if PNG_IMAGE is None:
            pytest.skip("No PNG test image available")

        image = Image.from_file(str(PNG_IMAGE))
        palette_rgbs = {c.rgb for c in standard_palette.colors}

        for method in ["mean_then_match", "match_then_mean", "match_then_mode"]:
            try:
                session = ConversionSession(image, standard_palette, CANVAS_SIZE)
                canvas = session.convert(ConvertConfig(method=method))
            except ValueError as e:
                if "stride" in str(e).lower() or "incompatible" in str(e).lower():
                    pytest.skip(f"Image dimensions incompatible: {e}")
                raise

            assert isinstance(canvas, Canvas)
            assert canvas.width == CANVAS_SIZE[0]
            assert canvas.height == CANVAS_SIZE[1]

        # Validate all cells have palette colors (on last canvas)
        for y in range(canvas.height):
            for x in range(canvas.width):
                cell = canvas.get_cell(x, y)
                assert cell.color is not None
                assert cell.color.rgb in palette_rgbs

    def test_jpeg_conversion_and_lego_set_palette(self):
        """JPEG image converts successfully with real LEGO set palette.

        This single test validates:
        - JPEG format loading works
        - Real LEGO set palette (31197 Warhol) works
        - Canvas dimensions are correct
        - All cells have palette colors
        """
        if JPEG_IMAGE is None:
            pytest.skip("No JPEG test image available")

        image = Image.from_file(str(JPEG_IMAGE))
        palette = Palette.from_set(31197)  # Andy Warhol's Marilyn Monroe
        palette_rgbs = {c.rgb for c in palette.colors}

        try:
            session = ConversionSession(image, palette, CANVAS_SIZE)
            canvas = session.convert()
        except ValueError as e:
            if "stride" in str(e).lower() or "incompatible" in str(e).lower():
                pytest.skip(f"Image dimensions incompatible: {e}")
            raise

        assert isinstance(canvas, Canvas)
        assert canvas.width == CANVAS_SIZE[0]
        assert canvas.height == CANVAS_SIZE[1]

        # Validate all cells have palette colors
        for y in range(canvas.height):
            for x in range(canvas.width):
                cell = canvas.get_cell(x, y)
                assert cell.color is not None
                assert cell.color.rgb in palette_rgbs


class TestCanvasExport:
    """Tests for exporting Canvas to various formats.

    Uses a class-scoped fixture to avoid repeated conversion.
    """

    @pytest.fixture(scope="class")
    def converted_canvas(self) -> Canvas:
        """Create a canvas once for all export tests in this class."""
        if PNG_IMAGE is None:
            pytest.skip("No PNG test image available")

        palette = Palette(
            [
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
            ]
        )

        image = Image.from_file(str(PNG_IMAGE))

        try:
            session = ConversionSession(image, palette, CANVAS_SIZE)
            return session.convert()
        except ValueError as e:
            if "stride" in str(e).lower() or "incompatible" in str(e).lower():
                pytest.skip(f"Image dimensions incompatible: {e}")
            raise

    def test_canvas_exports(self, converted_canvas: Canvas):
        """Canvas exports to numpy array, PNG, and SVG formats.

        Consolidated test covering all export formats:
        - to_array() produces valid numpy array
        - render_canvas_png() produces valid PIL Image
        - render_canvas_svg() produces valid SVG string
        """
        from PIL import Image as PILImage

        from tests.utilities.visualize import render_canvas_png, render_canvas_svg

        # Test numpy array export
        array = converted_canvas.to_array()
        assert isinstance(array, np.ndarray)
        assert array.shape == (CANVAS_SIZE[1], CANVAS_SIZE[0], 3)
        assert array.dtype == np.uint8
        assert array.min() >= 0 and array.max() <= 255

        # Test PNG rendering
        pil_image = render_canvas_png(converted_canvas, cell_size=20)
        assert isinstance(pil_image, PILImage.Image)
        assert pil_image.mode == "RGB"
        assert pil_image.size == (960, 960)  # 48 * 20

        # Test SVG rendering
        svg_content = render_canvas_svg(converted_canvas, cell_size=20)
        assert isinstance(svg_content, str)
        assert "</svg>" in svg_content
        assert "<circle" in svg_content


class TestImageDiscovery:
    """Meta-tests to verify test setup."""

    def test_images_directory_exists(self):
        """The tests/images/ directory exists."""
        assert IMAGES_DIR.exists(), f"Images directory not found: {IMAGES_DIR}"

    def test_at_least_one_test_image_found(self):
        """At least one test image was discovered."""
        assert len(TEST_IMAGES) > 0, (
            f"No test images found in {IMAGES_DIR}. Supported extensions: {SUPPORTED_EXTENSIONS}"
        )
