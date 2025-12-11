"""Unit tests for pipeline configuration classes.

This module tests configuration dataclasses and enums:
- PoolConfig: Spatial pooling configuration
- QuantizeConfig: Color quantization configuration
- DitherConfig: Dithering configuration (with validation)
- Enum values for PoolMethod, ColorSpace, DitherAlgorithm, ScanOrder
"""

import pytest

from mosaicpic.pipeline import (
    ColorSpace,
    DitherAlgorithm,
    DitherConfig,
    PoolConfig,
    PoolMethod,
    QuantizeConfig,
    ScanOrder,
)


class TestPoolMethod:
    """Tests for PoolMethod enum."""

    def test_all_methods_defined(self):
        """All expected pool methods are defined."""
        assert hasattr(PoolMethod, "MEAN")
        assert hasattr(PoolMethod, "MEDIAN")
        assert hasattr(PoolMethod, "MODE")
        assert hasattr(PoolMethod, "MAX")
        assert hasattr(PoolMethod, "MIN")

    def test_method_values(self):
        """Pool methods have expected string values."""
        assert PoolMethod.MEAN.value == "mean"
        assert PoolMethod.MEDIAN.value == "median"
        assert PoolMethod.MODE.value == "mode"
        assert PoolMethod.MAX.value == "max"
        assert PoolMethod.MIN.value == "min"

    def test_method_count(self):
        """There are exactly 5 pool methods."""
        assert len(PoolMethod) == 5


class TestColorSpace:
    """Tests for ColorSpace enum."""

    def test_all_colorspaces_defined(self):
        """All expected color spaces are defined."""
        assert hasattr(ColorSpace, "RGB")
        assert hasattr(ColorSpace, "LAB")
        assert hasattr(ColorSpace, "LINEAR_RGB")

    def test_colorspace_values(self):
        """Color spaces have expected string values."""
        assert ColorSpace.RGB.value == "rgb"
        assert ColorSpace.LAB.value == "lab"
        assert ColorSpace.LINEAR_RGB.value == "linear_rgb"

    def test_colorspace_count(self):
        """There are exactly 3 color spaces."""
        assert len(ColorSpace) == 3


class TestDitherAlgorithm:
    """Tests for DitherAlgorithm enum."""

    def test_all_algorithms_defined(self):
        """All expected dithering algorithms are defined."""
        assert hasattr(DitherAlgorithm, "FLOYD_STEINBERG")
        assert hasattr(DitherAlgorithm, "ATKINSON")
        assert hasattr(DitherAlgorithm, "JARVIS_JUDICE_NINKE")
        assert hasattr(DitherAlgorithm, "STUCKI")
        assert hasattr(DitherAlgorithm, "SIERRA")
        assert hasattr(DitherAlgorithm, "SIERRA_LITE")
        assert hasattr(DitherAlgorithm, "BAYER")

    def test_algorithm_values(self):
        """Dithering algorithms have expected string values."""
        assert DitherAlgorithm.FLOYD_STEINBERG.value == "floyd_steinberg"
        assert DitherAlgorithm.ATKINSON.value == "atkinson"
        assert DitherAlgorithm.JARVIS_JUDICE_NINKE.value == "jarvis"
        assert DitherAlgorithm.STUCKI.value == "stucki"
        assert DitherAlgorithm.SIERRA.value == "sierra"
        assert DitherAlgorithm.SIERRA_LITE.value == "sierra_lite"
        assert DitherAlgorithm.BAYER.value == "bayer"

    def test_algorithm_count(self):
        """There are exactly 7 dithering algorithms."""
        assert len(DitherAlgorithm) == 7


class TestScanOrder:
    """Tests for ScanOrder enum."""

    def test_all_orders_defined(self):
        """All expected scan orders are defined."""
        assert hasattr(ScanOrder, "RASTER")
        assert hasattr(ScanOrder, "SERPENTINE")

    def test_order_values(self):
        """Scan orders have expected string values."""
        assert ScanOrder.RASTER.value == "raster"
        assert ScanOrder.SERPENTINE.value == "serpentine"

    def test_order_count(self):
        """There are exactly 2 scan orders."""
        assert len(ScanOrder) == 2


class TestPoolConfig:
    """Tests for PoolConfig dataclass."""

    def test_default_values(self):
        """PoolConfig has correct default values."""
        config = PoolConfig()

        assert config.output_size is None
        assert config.method == PoolMethod.MEAN
        assert config.color_space == ColorSpace.RGB

    def test_custom_output_size(self):
        """PoolConfig accepts custom output_size."""
        config = PoolConfig(output_size=(48, 48))

        assert config.output_size == (48, 48)

    def test_custom_method(self):
        """PoolConfig accepts custom method."""
        config = PoolConfig(method=PoolMethod.MEDIAN)

        assert config.method == PoolMethod.MEDIAN

    def test_custom_colorspace(self):
        """PoolConfig accepts custom color_space."""
        config = PoolConfig(color_space=ColorSpace.LAB)

        assert config.color_space == ColorSpace.LAB

    def test_all_parameters(self):
        """PoolConfig accepts all parameters together."""
        config = PoolConfig(
            output_size=(96, 96),
            method=PoolMethod.MODE,
            color_space=ColorSpace.LINEAR_RGB,
        )

        assert config.output_size == (96, 96)
        assert config.method == PoolMethod.MODE
        assert config.color_space == ColorSpace.LINEAR_RGB


class TestQuantizeConfig:
    """Tests for QuantizeConfig dataclass."""

    def test_default_values(self):
        """QuantizeConfig has correct default values."""
        config = QuantizeConfig()

        assert config.metric == "delta_e"

    def test_custom_metric(self):
        """QuantizeConfig accepts custom metric."""
        config = QuantizeConfig(metric="custom_metric")

        assert config.metric == "custom_metric"


class TestDitherConfig:
    """Tests for DitherConfig dataclass."""

    def test_default_values(self):
        """DitherConfig has correct default values."""
        config = DitherConfig()

        assert config.algorithm == DitherAlgorithm.FLOYD_STEINBERG
        assert config.order == ScanOrder.SERPENTINE
        assert config.strength == 1.0
        assert config.metric == "delta_e"

    def test_custom_algorithm(self):
        """DitherConfig accepts custom algorithm."""
        config = DitherConfig(algorithm=DitherAlgorithm.ATKINSON)

        assert config.algorithm == DitherAlgorithm.ATKINSON

    def test_custom_order(self):
        """DitherConfig accepts custom order."""
        config = DitherConfig(order=ScanOrder.RASTER)

        assert config.order == ScanOrder.RASTER

    def test_custom_strength(self):
        """DitherConfig accepts custom strength."""
        config = DitherConfig(strength=0.5)

        assert config.strength == 0.5

    def test_custom_metric(self):
        """DitherConfig accepts custom metric."""
        config = DitherConfig(metric="custom_metric")

        assert config.metric == "custom_metric"

    def test_all_parameters(self):
        """DitherConfig accepts all parameters together."""
        config = DitherConfig(
            algorithm=DitherAlgorithm.JARVIS_JUDICE_NINKE,
            order=ScanOrder.RASTER,
            strength=0.75,
            metric="euclidean",
        )

        assert config.algorithm == DitherAlgorithm.JARVIS_JUDICE_NINKE
        assert config.order == ScanOrder.RASTER
        assert config.strength == 0.75
        assert config.metric == "euclidean"

    def test_strength_validation_too_high(self):
        """DitherConfig rejects strength > 1.0."""
        with pytest.raises(ValueError, match="strength must be"):
            DitherConfig(strength=1.1)

    def test_strength_validation_too_low(self):
        """DitherConfig rejects strength < 0.0."""
        with pytest.raises(ValueError, match="strength must be"):
            DitherConfig(strength=-0.1)

    def test_strength_boundary_zero(self):
        """DitherConfig accepts strength = 0.0."""
        config = DitherConfig(strength=0.0)

        assert config.strength == 0.0

    def test_strength_boundary_one(self):
        """DitherConfig accepts strength = 1.0."""
        config = DitherConfig(strength=1.0)

        assert config.strength == 1.0

    @pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_strength_values(self, strength):
        """DitherConfig accepts valid strength values."""
        config = DitherConfig(strength=strength)

        assert config.strength == strength

    @pytest.mark.parametrize("strength", [-1.0, -0.001, 1.001, 2.0, 100.0])
    def test_invalid_strength_values(self, strength):
        """DitherConfig rejects invalid strength values."""
        with pytest.raises(ValueError, match="strength must be"):
            DitherConfig(strength=strength)
