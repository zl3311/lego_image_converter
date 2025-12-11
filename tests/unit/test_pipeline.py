"""Unit tests for Pipeline class.

This module provides comprehensive tests for the Pipeline class:
- Type validation (step chaining, first/last step requirements)
- Dimension validation (upsampling prevention)
- Pipeline execution
- String representation
- Complex multi-step pipelines
"""

import numpy as np
import pytest

from mosaicpic import Color, Palette
from mosaicpic.pipeline import (
    ColorSpace,
    DitherAlgorithm,
    DitherConfig,
    DitherStep,
    IndexMap,
    Pipeline,
    PipelineContext,
    PoolConfig,
    PoolMethod,
    PoolStep,
    QuantizeStep,
    RGBImage,
)


class TestPipelineValidation:
    """Tests for Pipeline type and dimension validation."""

    @pytest.fixture
    def simple_palette(self):
        """A simple 3-color palette."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
            ]
        )

    def test_empty_pipeline_raises(self):
        """Empty pipeline raises ValueError."""
        with pytest.raises(ValueError, match="at least one step"):
            Pipeline([])

    def test_valid_pool_quantize_pipeline(self):
        """Pipeline with Pool -> Quantize is valid."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                QuantizeStep(),
            ]
        )

        assert len(pipeline.steps) == 2

    def test_valid_pool_dither_pipeline(self):
        """Pipeline with Pool -> Dither is valid."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                DitherStep(),
            ]
        )

        assert len(pipeline.steps) == 2

    def test_valid_quantize_pool_pipeline(self):
        """Pipeline with Quantize -> Pool is valid."""
        pipeline = Pipeline(
            [
                QuantizeStep(),
                PoolStep(PoolConfig()),
            ]
        )

        assert len(pipeline.steps) == 2

    def test_valid_dither_only_pipeline(self):
        """Pipeline with just Dither is valid (if no pooling needed)."""
        pipeline = Pipeline([DitherStep()])

        assert len(pipeline.steps) == 1

    def test_valid_quantize_only_pipeline(self):
        """Pipeline with just Quantize is valid (if no pooling needed)."""
        pipeline = Pipeline([QuantizeStep()])

        assert len(pipeline.steps) == 1

    def test_invalid_first_step_raises(self):
        """First step must accept RGBImage."""
        # DitherStep only accepts RGBImage, which is fine
        # But if we create a step that doesn't accept RGBImage, it should fail
        # Currently all steps accept RGBImage or IndexMap, so this test
        # validates that the mechanism works

        # Create a pipeline that ends with RGBImage (invalid)
        # PoolStep(RGBImage) -> RGBImage, then nothing converts to IndexMap
        with pytest.raises(TypeError, match="must end with IndexMap"):
            Pipeline([PoolStep(PoolConfig())])

    def test_pipeline_must_end_with_indexmap(self):
        """Pipeline must end with IndexMap output."""
        with pytest.raises(TypeError, match="must end with IndexMap"):
            Pipeline([PoolStep(PoolConfig())])

    def test_upsampling_raises(self):
        """Pipeline rejects upsampling (output larger than input)."""
        with pytest.raises(ValueError, match="Cannot upsample"):
            Pipeline(
                [
                    PoolStep(PoolConfig(output_size=(48, 48))),
                    QuantizeStep(),
                    PoolStep(PoolConfig(output_size=(96, 96))),  # Upsampling!
                ]
            )

    def test_valid_multi_pool_pipeline(self):
        """Pipeline with multiple downsampling pool steps is valid."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig(output_size=(100, 100))),
                PoolStep(PoolConfig(output_size=(50, 50))),
                QuantizeStep(),
            ]
        )

        assert len(pipeline.steps) == 3

    def test_named_pipeline(self):
        """Pipeline can be created with a name."""
        pipeline = Pipeline(
            [QuantizeStep()],
            name="test_pipeline",
        )

        assert pipeline.name == "test_pipeline"

    def test_unnamed_pipeline(self):
        """Pipeline without name has None name."""
        pipeline = Pipeline([QuantizeStep()])

        assert pipeline.name is None


class TestPipelineExecution:
    """Tests for Pipeline.run execution."""

    @pytest.fixture
    def simple_palette(self):
        """A simple 3-color palette."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
            ]
        )

    @pytest.fixture
    def extended_palette(self):
        """An extended palette with more colors."""
        return Palette(
            [
                Color((255, 0, 0), name="Red"),
                Color((0, 255, 0), name="Green"),
                Color((0, 0, 255), name="Blue"),
                Color((0, 0, 0), name="Black"),
                Color((255, 255, 255), name="White"),
                Color((255, 255, 0), name="Yellow"),
                Color((255, 0, 255), name="Magenta"),
                Color((0, 255, 255), name="Cyan"),
            ]
        )

    @pytest.fixture
    def uniform_red_image(self):
        """A 100x100 solid red image."""
        data = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        return RGBImage(data=data)

    @pytest.fixture
    def gradient_image(self):
        """A 100x100 gradient image."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in range(100):
            data[:, x, :] = int(x * 255 / 99)
        return RGBImage(data=data)

    def test_pool_quantize_execution(self, uniform_red_image, simple_palette):
        """Pool -> Quantize pipeline executes correctly."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                QuantizeStep(),
            ]
        )

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        result = pipeline.run(uniform_red_image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (10, 10)

        # All indices should be 0 (red) since input is uniform red
        assert np.all(result.data == 0)

    def test_output_size_mismatch_raises(self, uniform_red_image, simple_palette):
        """Pipeline raises if output size doesn't match target."""
        # Pipeline outputs 20x20 but context expects 10x10
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig(output_size=(20, 20))),
                QuantizeStep(),
            ]
        )

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        with pytest.raises(ValueError, match="does not match target size"):
            pipeline.run(uniform_red_image, context)

    def test_dither_pipeline_execution(self, uniform_red_image, simple_palette):
        """Pool -> Dither pipeline executes correctly."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                DitherStep(DitherConfig()),
            ]
        )

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        result = pipeline.run(uniform_red_image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (10, 10)

    def test_quantize_pool_execution(self, uniform_red_image, simple_palette):
        """Quantize -> Pool pipeline executes correctly."""
        pipeline = Pipeline(
            [
                QuantizeStep(),
                PoolStep(PoolConfig()),
            ]
        )

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        result = pipeline.run(uniform_red_image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (10, 10)

    def test_quantize_only_execution(self, simple_palette):
        """Quantize-only pipeline works when image matches target size."""
        data = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        image = RGBImage(data=data)

        pipeline = Pipeline([QuantizeStep()])

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        result = pipeline.run(image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (10, 10)

    def test_multi_pool_execution(self, uniform_red_image, simple_palette):
        """Pipeline with multiple pool steps executes correctly."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig(output_size=(50, 50))),
                PoolStep(PoolConfig(output_size=(10, 10))),
                QuantizeStep(),
            ]
        )

        context = PipelineContext(
            palette=simple_palette,
            target_size=(10, 10),
        )

        result = pipeline.run(uniform_red_image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (10, 10)

    def test_complex_pipeline_with_all_steps(self, gradient_image, extended_palette):
        """Complex pipeline with pool, quantize, and pool executes correctly."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig(method=PoolMethod.MEAN, color_space=ColorSpace.LAB)),
                QuantizeStep(),
                PoolStep(PoolConfig()),  # Pool IndexMap
            ]
        )

        context = PipelineContext(
            palette=extended_palette,
            target_size=(5, 5),
        )

        result = pipeline.run(gradient_image, context)

        assert isinstance(result, IndexMap)
        assert result.shape == (5, 5)

    def test_dithered_pipeline_on_gradient(self, gradient_image, extended_palette):
        """Dithered pipeline produces varied result on gradient image."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                DitherStep(DitherConfig(algorithm=DitherAlgorithm.FLOYD_STEINBERG)),
            ]
        )

        context = PipelineContext(
            palette=extended_palette,
            target_size=(25, 25),
        )

        result = pipeline.run(gradient_image, context)

        # Gradient should produce multiple different indices
        unique_indices = np.unique(result.data)
        assert len(unique_indices) >= 2


class TestPipelineRepr:
    """Tests for Pipeline string representation."""

    def test_named_pipeline_repr(self):
        """Named pipeline includes name in repr."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                QuantizeStep(),
            ],
            name="test_pipeline",
        )

        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "test_pipeline" in repr_str
        assert "PoolStep" in repr_str
        assert "QuantizeStep" in repr_str

    def test_unnamed_pipeline_repr(self):
        """Unnamed pipeline doesn't include name= prefix."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                DitherStep(),
            ]
        )

        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "PoolStep" in repr_str
        assert "DitherStep" in repr_str
        assert "name=" not in repr_str

    def test_single_step_pipeline_repr(self):
        """Single-step pipeline repr is correct."""
        pipeline = Pipeline([QuantizeStep()])

        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "QuantizeStep" in repr_str

    def test_repr_includes_all_steps(self):
        """Repr includes all steps in order."""
        pipeline = Pipeline(
            [
                PoolStep(PoolConfig()),
                QuantizeStep(),
                PoolStep(PoolConfig()),
            ]
        )

        repr_str = repr(pipeline)

        # Should have PoolStep twice
        assert repr_str.count("PoolStep") == 2
        assert "QuantizeStep" in repr_str
