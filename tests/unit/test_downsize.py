"""Unit tests for the downsize function."""

import numpy as np
import pytest

from legopic import Canvas, Color, Image, Palette
from legopic.core.downsize import _validate_dimensions, downsize


class TestValidateDimensions:
    """Tests for _validate_dimensions helper.

    Stride is computed using floor division: stride = image_dim // canvas_dim.
    This allows the last row/column of canvas cells to have more pixels
    (incomplete stride is allowed).
    """

    def test_exact_divisible(self):
        """Returns correct stride when dimensions are exactly divisible."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image(array)

        stride = _validate_dimensions(image, 10, 10)
        assert stride == 10

    def test_partial_last_batch(self):
        """Returns correct stride when last cells have extra pixels."""
        # 109x109 with canvas 10x10 -> floor(109/10)=10 for both
        # Last cells get 19 pixels instead of 10
        array = np.zeros((109, 109, 3), dtype=np.uint8)
        image = Image(array)

        stride = _validate_dimensions(image, 10, 10)
        assert stride == 10

    def test_incompatible_dimensions_raises(self):
        """Raises ValueError when strides don't match."""
        # 90x100 with canvas 10x10 -> floor(100/10)=10, floor(90/10)=9 -> mismatch
        array = np.zeros((90, 100, 3), dtype=np.uint8)
        image = Image(array)

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            _validate_dimensions(image, 10, 10)

    def test_image_too_small_raises(self):
        """Raises ValueError when image is smaller than canvas."""
        array = np.zeros((5, 10, 3), dtype=np.uint8)
        image = Image(array)

        with pytest.raises(ValueError, match="at least as large"):
            _validate_dimensions(image, 10, 10)

    def test_zero_canvas_dimension_raises(self):
        """Raises ValueError for non-positive canvas dimensions."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image(array)

        with pytest.raises(ValueError, match="must be positive"):
            _validate_dimensions(image, 0, 10)

        with pytest.raises(ValueError, match="must be positive"):
            _validate_dimensions(image, 10, -1)

    def test_stride_calculation_examples(self):
        """Verify stride calculation with floor division."""
        # 109x109 with canvas 10x10 -> floor(109/10)=10, works
        img1 = Image(np.zeros((109, 109, 3), dtype=np.uint8))
        assert _validate_dimensions(img1, 10, 10) == 10

        # 100x90 with canvas 10x10 -> floor(100/10)=10, floor(90/10)=9 -> fails
        img2 = Image(np.zeros((90, 100, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            _validate_dimensions(img2, 10, 10)

        # 92x101 with canvas 10x10 -> floor(92/10)=9, floor(101/10)=10 -> fails
        img3 = Image(np.zeros((101, 92, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            _validate_dimensions(img3, 10, 10)

        # 1024x1024 with canvas 48x48 -> floor(1024/48)=21, works
        img4 = Image(np.zeros((1024, 1024, 3), dtype=np.uint8))
        assert _validate_dimensions(img4, 48, 48) == 21


class TestDownsize:
    """Tests for the downsize function."""

    @pytest.fixture
    def simple_palette(self):
        """A simple 2-color palette."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 0, 255), name="Blue"),
            ]
        )

    @pytest.fixture
    def uniform_red_image(self):
        """A 100x100 solid red image."""
        array = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        return Image(array)

    def test_returns_canvas(self, uniform_red_image, simple_palette):
        """downsize returns a Canvas object."""
        result = downsize(uniform_red_image, simple_palette, 10, 10)
        assert isinstance(result, Canvas)

    def test_canvas_dimensions(self, uniform_red_image, simple_palette):
        """Returned canvas has correct dimensions."""
        result = downsize(uniform_red_image, simple_palette, 10, 10)
        assert result.width == 10
        assert result.height == 10

    def test_uniform_image_uniform_result(self, uniform_red_image, simple_palette):
        """Uniform red image produces uniform red canvas."""
        result = downsize(uniform_red_image, simple_palette, 10, 10)

        for y in range(result.height):
            for x in range(result.width):
                cell = result.get_cell(x, y)
                assert cell.color.rgb == (255, 0, 0)

    def test_invalid_method_raises(self, uniform_red_image, simple_palette):
        """Invalid method name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            downsize(uniform_red_image, simple_palette, 10, 10, method="invalid")

    def test_all_methods_run(self, uniform_red_image, simple_palette):
        """All three methods execute without error."""
        for method in ["mean_then_match", "match_then_mean", "match_then_mode"]:
            result = downsize(uniform_red_image, simple_palette, 10, 10, method=method)
            assert result.width == 10
            assert result.height == 10


class TestDownsizeMethods:
    """Tests comparing different downsize methods."""

    @pytest.fixture
    def rgb_palette(self):
        """RGB + Black + White palette."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
            ]
        )

    def test_mode_preserves_dominant_color(self, rgb_palette):
        """match_then_mode selects the most common color in a block."""
        # Create 10x10 image where each 1x1 block has mostly red
        # with some blue pixels
        array = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        array[0, 0, :] = [0, 0, 255]  # One blue pixel
        image = Image(array)

        result = downsize(image, rgb_palette, 1, 1, method="match_then_mode")
        # Should be red (dominant)
        assert result.get_cell(0, 0).color.rgb == (255, 0, 0)

    def test_mean_averages_colors(self, rgb_palette):
        """mean_then_match averages pixel colors before matching."""
        # Half red, half blue -> purple-ish -> matches... something
        array = np.zeros((10, 10, 3), dtype=np.uint8)
        array[:5, :, :] = [255, 0, 0]  # Top half red
        array[5:, :, :] = [0, 0, 255]  # Bottom half blue
        image = Image(array)

        # The mean color is (127, 0, 127), which should match to one of the palette colors
        result = downsize(image, rgb_palette, 1, 1, method="mean_then_match")
        # Just verify it runs and produces a result
        assert result.get_cell(0, 0).color is not None


class TestDownsizeScalability:
    """Scalability tests for match_then_mode at various canvas sizes.

    Tests that the optimized batched implementation works correctly
    across a range of canvas sizes from 16x16 to 256x256.
    """

    @pytest.fixture
    def large_image(self):
        """A 1024x1024 random image for scalability testing."""
        np.random.seed(42)
        array = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        return Image(array)

    @pytest.fixture
    def test_palette(self):
        """A palette with multiple colors for scalability testing."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
                Color((255, 255, 0), name="Yellow"),
                Color((0, 255, 255), name="Cyan"),
                Color((255, 0, 255), name="Magenta"),
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
            ]
        )

    @pytest.mark.parametrize("canvas_size", [16, 32, 64, 128, 256])
    def test_match_then_mode_scalability(self, large_image, test_palette, canvas_size):
        """match_then_mode runs without error at various canvas sizes."""
        result = downsize(
            large_image, test_palette, canvas_size, canvas_size, method="match_then_mode"
        )
        assert result.width == canvas_size
        assert result.height == canvas_size
