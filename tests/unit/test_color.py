"""Unit tests for the Color class."""

import pytest

from legopic import Color


class TestColorInit:
    """Tests for Color.__init__."""

    def test_valid_rgb(self):
        """Color accepts valid RGB tuple."""
        color = Color((128, 64, 255))
        assert color.rgb == (128, 64, 255)
        assert color.name is None

    def test_valid_rgb_with_name(self):
        """Color accepts RGB with optional name."""
        color = Color((255, 0, 0), name="Bright Red")
        assert color.rgb == (255, 0, 0)
        assert color.name == "Bright Red"

    def test_boundary_values(self):
        """Color accepts boundary values 0 and 255."""
        color = Color((0, 0, 0))
        assert color.rgb == (0, 0, 0)

        color = Color((255, 255, 255))
        assert color.rgb == (255, 255, 255)

    def test_negative_value_raises(self):
        """Color rejects negative RGB values."""
        with pytest.raises(ValueError, match="red"):
            Color((-1, 0, 0))
        with pytest.raises(ValueError, match="green"):
            Color((0, -1, 0))
        with pytest.raises(ValueError, match="blue"):
            Color((0, 0, -1))

    def test_value_over_255_raises(self):
        """Color rejects RGB values > 255."""
        with pytest.raises(ValueError, match="red"):
            Color((256, 0, 0))
        with pytest.raises(ValueError, match="green"):
            Color((0, 300, 0))
        with pytest.raises(ValueError, match="blue"):
            Color((0, 0, 1000))

    def test_float_value_raises(self):
        """Color rejects non-integer RGB values."""
        with pytest.raises(ValueError):
            Color((128.5, 64, 255))  # type: ignore


class TestColorFromHex:
    """Tests for Color.from_hex."""

    def test_six_digit_hex(self):
        """from_hex parses 6-digit hex codes."""
        color = Color.from_hex("FF0000")
        assert color.rgb == (255, 0, 0)

        color = Color.from_hex("00FF00")
        assert color.rgb == (0, 255, 0)

        color = Color.from_hex("0000FF")
        assert color.rgb == (0, 0, 255)

    def test_hex_with_hash(self):
        """from_hex handles leading # character."""
        color = Color.from_hex("#FF00FF")
        assert color.rgb == (255, 0, 255)

    def test_three_digit_hex(self):
        """from_hex expands 3-digit shorthand."""
        color = Color.from_hex("F00")
        assert color.rgb == (255, 0, 0)

        color = Color.from_hex("#ABC")
        assert color.rgb == (170, 187, 204)

    def test_case_insensitive(self):
        """from_hex is case-insensitive."""
        assert Color.from_hex("ff0000").rgb == Color.from_hex("FF0000").rgb
        assert Color.from_hex("AbCdEf").rgb == (171, 205, 239)

    def test_with_name(self):
        """from_hex accepts optional name."""
        color = Color.from_hex("FF0000", name="Red")
        assert color.name == "Red"

    def test_invalid_hex_raises(self):
        """from_hex rejects invalid hex codes."""
        with pytest.raises(ValueError, match="Invalid hex"):
            Color.from_hex("GG0000")
        with pytest.raises(ValueError, match="Invalid hex"):
            Color.from_hex("FF00")  # 4 digits
        with pytest.raises(ValueError, match="Invalid hex"):
            Color.from_hex("FF00000")  # 7 digits


class TestColorFromHsv:
    """Tests for Color.from_hsv."""

    def test_pure_red(self):
        """from_hsv creates pure red at H=0."""
        color = Color.from_hsv(0, 1, 1)
        assert color.rgb == (255, 0, 0)

    def test_pure_green(self):
        """from_hsv creates pure green at H=120."""
        color = Color.from_hsv(120, 1, 1)
        assert color.rgb == (0, 255, 0)

    def test_pure_blue(self):
        """from_hsv creates pure blue at H=240."""
        color = Color.from_hsv(240, 1, 1)
        assert color.rgb == (0, 0, 255)

    def test_white(self):
        """from_hsv creates white at S=0, V=1."""
        color = Color.from_hsv(0, 0, 1)
        assert color.rgb == (255, 255, 255)

    def test_black(self):
        """from_hsv creates black at V=0."""
        color = Color.from_hsv(0, 1, 0)
        assert color.rgb == (0, 0, 0)

    def test_invalid_hue_raises(self):
        """from_hsv rejects hue outside [0, 360)."""
        with pytest.raises(ValueError, match="Hue"):
            Color.from_hsv(-1, 1, 1)
        with pytest.raises(ValueError, match="Hue"):
            Color.from_hsv(360, 1, 1)

    def test_invalid_saturation_raises(self):
        """from_hsv rejects saturation outside [0, 1]."""
        with pytest.raises(ValueError, match="Saturation"):
            Color.from_hsv(0, -0.1, 1)
        with pytest.raises(ValueError, match="Saturation"):
            Color.from_hsv(0, 1.1, 1)

    def test_invalid_value_raises(self):
        """from_hsv rejects value outside [0, 1]."""
        with pytest.raises(ValueError, match="Value"):
            Color.from_hsv(0, 1, -0.1)
        with pytest.raises(ValueError, match="Value"):
            Color.from_hsv(0, 1, 1.1)


class TestColorHashEquality:
    """Tests for Color.__hash__ and __eq__."""

    def test_equal_colors(self):
        """Colors with same RGB are equal."""
        c1 = Color((100, 150, 200))
        c2 = Color((100, 150, 200))
        assert c1 == c2

    def test_unequal_colors(self):
        """Colors with different RGB are not equal."""
        c1 = Color((100, 150, 200))
        c2 = Color((100, 150, 201))
        assert c1 != c2

    def test_name_ignored_for_equality(self):
        """Color equality ignores name."""
        c1 = Color((255, 0, 0), name="Red")
        c2 = Color((255, 0, 0), name="Crimson")
        assert c1 == c2

    def test_hashable(self):
        """Colors can be used in sets/dicts."""
        colors = {Color((255, 0, 0)), Color((0, 255, 0)), Color((255, 0, 0))}
        assert len(colors) == 2  # Duplicate red removed

    def test_same_hash_for_equal(self):
        """Equal colors have same hash."""
        c1 = Color((100, 150, 200))
        c2 = Color((100, 150, 200), name="Different Name")
        assert hash(c1) == hash(c2)
