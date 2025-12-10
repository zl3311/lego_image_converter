"""Unit tests for color space conversion utilities.

This module tests the color_utils functions for converting between:
- RGB and Linear RGB (gamma correction)
- RGB and CIE Lab (perceptually uniform color space)
- RGB and XYZ (intermediate color space)

Tests verify:
- Round-trip conversions preserve colors (within floating-point tolerance)
- Known reference values for specific colors
- Edge cases (black, white, saturated primaries)
- Array shape preservation across conversions
"""

import numpy as np

from legopic.pipeline.color_utils import (
    lab_to_rgb,
    linear_to_rgb,
    rgb_to_lab,
    rgb_to_linear,
    rgb_to_xyz,
    xyz_to_rgb,
)


class TestRGBToLinear:
    """Tests for RGB to Linear RGB conversion."""

    def test_black_stays_black(self):
        """Pure black (0,0,0) remains black after conversion."""
        rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)
        linear = rgb_to_linear(rgb)

        np.testing.assert_allclose(linear, 0.0, atol=1e-10)

    def test_white_stays_white(self):
        """Pure white (255,255,255) remains ~1.0 after conversion."""
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        linear = rgb_to_linear(rgb)

        np.testing.assert_allclose(linear, 1.0, atol=1e-10)

    def test_mid_gray_is_darker_linear(self):
        """Mid-gray (128,128,128) should be ~0.2 in linear space due to gamma."""
        rgb = np.array([[[128, 128, 128]]], dtype=np.uint8)
        linear = rgb_to_linear(rgb)

        # sRGB gamma makes perceptual mid-gray darker in linear space
        # Value should be approximately 0.2 (not 0.5)
        assert linear[0, 0, 0] < 0.25
        assert linear[0, 0, 0] > 0.15

    def test_output_shape_preserved(self):
        """Output array has same shape as input."""
        rgb = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
        linear = rgb_to_linear(rgb)

        assert linear.shape == rgb.shape

    def test_output_dtype_is_float64(self):
        """Output dtype is float64."""
        rgb = np.array([[[100, 150, 200]]], dtype=np.uint8)
        linear = rgb_to_linear(rgb)

        assert linear.dtype == np.float64


class TestLinearToRGB:
    """Tests for Linear RGB to RGB conversion."""

    def test_black_stays_black(self):
        """Linear black (0,0,0) remains black after conversion."""
        linear = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
        rgb = linear_to_rgb(linear)

        np.testing.assert_array_equal(rgb, 0)

    def test_white_stays_white(self):
        """Linear white (1,1,1) becomes (255,255,255)."""
        linear = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)
        rgb = linear_to_rgb(linear)

        np.testing.assert_array_equal(rgb, 255)

    def test_clipping_above_one(self):
        """Values above 1.0 are clipped to 255."""
        linear = np.array([[[1.5, 2.0, 10.0]]], dtype=np.float64)
        rgb = linear_to_rgb(linear)

        np.testing.assert_array_equal(rgb, 255)

    def test_clipping_below_zero(self):
        """Negative values are clipped to 0."""
        linear = np.array([[[-0.5, -1.0, -10.0]]], dtype=np.float64)
        rgb = linear_to_rgb(linear)

        np.testing.assert_array_equal(rgb, 0)

    def test_output_dtype_is_uint8(self):
        """Output dtype is uint8."""
        linear = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        rgb = linear_to_rgb(linear)

        assert rgb.dtype == np.uint8


class TestRGBLinearRoundTrip:
    """Tests for RGB <-> Linear RGB round-trip conversions."""

    def test_round_trip_black(self):
        """Black survives round-trip conversion."""
        original = np.array([[[0, 0, 0]]], dtype=np.uint8)
        result = linear_to_rgb(rgb_to_linear(original))

        np.testing.assert_array_equal(result, original)

    def test_round_trip_white(self):
        """White survives round-trip conversion."""
        original = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = linear_to_rgb(rgb_to_linear(original))

        np.testing.assert_array_equal(result, original)

    def test_round_trip_primaries(self):
        """Primary colors survive round-trip conversion."""
        original = np.array(
            [
                [[255, 0, 0]],  # Red
                [[0, 255, 0]],  # Green
                [[0, 0, 255]],  # Blue
            ],
            dtype=np.uint8,
        )
        result = linear_to_rgb(rgb_to_linear(original))

        np.testing.assert_array_equal(result, original)

    def test_round_trip_random_values(self):
        """Random RGB values survive round-trip within ±1 due to rounding."""
        np.random.seed(42)
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = linear_to_rgb(rgb_to_linear(original))

        # Allow ±1 difference due to floating-point rounding
        np.testing.assert_allclose(result, original, atol=1)


class TestRGBToLab:
    """Tests for RGB to CIE Lab conversion."""

    def test_black_lab_values(self):
        """Black should have L=0, a=0, b=0."""
        rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        np.testing.assert_allclose(lab[0, 0, 0], 0.0, atol=0.1)  # L
        np.testing.assert_allclose(lab[0, 0, 1], 0.0, atol=0.1)  # a
        np.testing.assert_allclose(lab[0, 0, 2], 0.0, atol=0.1)  # b

    def test_white_lab_values(self):
        """White should have L=100, a=0, b=0."""
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        np.testing.assert_allclose(lab[0, 0, 0], 100.0, atol=0.5)  # L
        np.testing.assert_allclose(lab[0, 0, 1], 0.0, atol=0.5)  # a
        np.testing.assert_allclose(lab[0, 0, 2], 0.0, atol=0.5)  # b

    def test_red_has_positive_a(self):
        """Red should have positive a value (red-green axis)."""
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        assert lab[0, 0, 1] > 0  # a is positive for red

    def test_green_has_negative_a(self):
        """Green should have negative a value."""
        rgb = np.array([[[0, 255, 0]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        assert lab[0, 0, 1] < 0  # a is negative for green

    def test_blue_has_negative_b(self):
        """Blue should have negative b value (blue-yellow axis)."""
        rgb = np.array([[[0, 0, 255]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        assert lab[0, 0, 2] < 0  # b is negative for blue

    def test_yellow_has_positive_b(self):
        """Yellow should have positive b value."""
        rgb = np.array([[[255, 255, 0]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        assert lab[0, 0, 2] > 0  # b is positive for yellow

    def test_output_shape_preserved(self):
        """Output shape matches input."""
        rgb = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
        lab = rgb_to_lab(rgb)

        assert lab.shape == rgb.shape


class TestLabToRGB:
    """Tests for CIE Lab to RGB conversion."""

    def test_black_conversion(self):
        """Lab (0, 0, 0) converts to black RGB."""
        lab = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
        rgb = lab_to_rgb(lab)

        np.testing.assert_array_equal(rgb, 0)

    def test_white_conversion(self):
        """Lab (100, 0, 0) converts to white RGB."""
        lab = np.array([[[100.0, 0.0, 0.0]]], dtype=np.float64)
        rgb = lab_to_rgb(lab)

        np.testing.assert_allclose(rgb, 255, atol=1)


class TestRGBLabRoundTrip:
    """Tests for RGB <-> Lab round-trip conversions."""

    def test_round_trip_black(self):
        """Black survives round-trip through Lab."""
        original = np.array([[[0, 0, 0]]], dtype=np.uint8)
        result = lab_to_rgb(rgb_to_lab(original))

        np.testing.assert_array_equal(result, original)

    def test_round_trip_white(self):
        """White survives round-trip through Lab."""
        original = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = lab_to_rgb(rgb_to_lab(original))

        np.testing.assert_allclose(result, original, atol=1)

    def test_round_trip_primaries(self):
        """Primary colors survive round-trip within tolerance."""
        original = np.array(
            [
                [[255, 0, 0]],  # Red
                [[0, 255, 0]],  # Green
                [[0, 0, 255]],  # Blue
            ],
            dtype=np.uint8,
        )
        result = lab_to_rgb(rgb_to_lab(original))

        # Allow ±2 difference due to color space conversion precision
        np.testing.assert_allclose(result, original, atol=2)

    def test_round_trip_random_values(self):
        """Random RGB values survive round-trip within tolerance."""
        np.random.seed(42)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = lab_to_rgb(rgb_to_lab(original))

        # Allow ±2 difference due to color space conversion precision
        np.testing.assert_allclose(result, original, atol=2)


class TestRGBToXYZ:
    """Tests for RGB to XYZ conversion."""

    def test_black_xyz_values(self):
        """Black should have X=0, Y=0, Z=0."""
        rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)
        xyz = rgb_to_xyz(rgb)

        np.testing.assert_allclose(xyz, 0.0, atol=0.1)

    def test_white_xyz_values(self):
        """White should have specific XYZ values (D65 reference)."""
        rgb = np.array([[[255, 255, 255]]], dtype=np.uint8)
        xyz = rgb_to_xyz(rgb)

        # D65 white point reference
        np.testing.assert_allclose(xyz[0, 0, 0], 95.047, atol=1)  # X
        np.testing.assert_allclose(xyz[0, 0, 1], 100.0, atol=1)  # Y
        np.testing.assert_allclose(xyz[0, 0, 2], 108.883, atol=1)  # Z


class TestXYZToRGB:
    """Tests for XYZ to RGB conversion."""

    def test_black_conversion(self):
        """XYZ (0, 0, 0) converts to black RGB."""
        xyz = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
        rgb = xyz_to_rgb(xyz)

        np.testing.assert_array_equal(rgb, 0)

    def test_white_conversion(self):
        """D65 white point XYZ converts to white RGB."""
        xyz = np.array([[[95.047, 100.0, 108.883]]], dtype=np.float64)
        rgb = xyz_to_rgb(xyz)

        np.testing.assert_allclose(rgb, 255, atol=1)


class TestRGBXYZRoundTrip:
    """Tests for RGB <-> XYZ round-trip conversions."""

    def test_round_trip_black(self):
        """Black survives round-trip through XYZ."""
        original = np.array([[[0, 0, 0]]], dtype=np.uint8)
        result = xyz_to_rgb(rgb_to_xyz(original))

        np.testing.assert_array_equal(result, original)

    def test_round_trip_white(self):
        """White survives round-trip through XYZ."""
        original = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = xyz_to_rgb(rgb_to_xyz(original))

        np.testing.assert_allclose(result, original, atol=1)

    def test_round_trip_random_values(self):
        """Random RGB values survive round-trip within tolerance."""
        np.random.seed(42)
        original = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = xyz_to_rgb(rgb_to_xyz(original))

        # Allow ±1 difference due to floating-point rounding
        np.testing.assert_allclose(result, original, atol=1)
