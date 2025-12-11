"""Pipeline class for composing and executing processing steps.

This module provides the Pipeline class that validates and executes
an ordered sequence of processing steps.
"""

from typing import TYPE_CHECKING, Any

from .steps.base import Step
from .types import IndexMap, RGBImage

if TYPE_CHECKING:
    from .context import PipelineContext


class Pipeline:
    """An ordered sequence of processing steps.

    The pipeline validates at construction time that:
    1. Step types chain correctly (each step accepts previous step's output)
    2. First step accepts RGBImage
    3. Last step produces IndexMap
    4. Dimensions are valid (when explicitly specified)

    Attributes:
        steps: Ordered list of steps to execute.
        name: Optional name for this pipeline configuration.

    Example:
        >>> pipeline = Pipeline([
        ...     PoolStep(PoolConfig(output_size=(48, 48))),
        ...     DitherStep(DitherConfig(algorithm=DitherAlgorithm.FLOYD_STEINBERG)),
        ... ], name="dithered")
        >>> result = pipeline.run(image, context)
    """

    def __init__(self, steps: list[Step], name: str | None = None):
        """Initialize pipeline with validation.

        Args:
            steps: List of Step objects to execute in order.
            name: Optional name for this pipeline.

        Raises:
            ValueError: If steps list is empty.
            TypeError: If step types don't chain correctly.
            ValueError: If dimensions are incompatible.
        """
        if not steps:
            raise ValueError("Pipeline must have at least one step.")

        self.steps = steps
        self.name = name
        self._validate_types()
        self._validate_dimensions()

    def _validate_types(self) -> None:
        """Validate that step input/output types chain correctly.

        Raises:
            TypeError: If first step doesn't accept RGBImage.
            TypeError: If adjacent steps have incompatible types.
            TypeError: If last step doesn't produce IndexMap.
        """
        # First step must accept RGBImage
        if RGBImage not in self.steps[0].input_types:
            raise TypeError(
                f"First step must accept RGBImage. "
                f"{type(self.steps[0]).__name__} accepts: "
                f"{[t.__name__ for t in self.steps[0].input_types]}"
            )

        # Track current type through the chain
        current_type: type = RGBImage

        for i, step in enumerate(self.steps):
            # Check if current step accepts the current type
            if current_type not in step.input_types:
                raise TypeError(
                    f"Step {i} ({type(step).__name__}) cannot accept "
                    f"{current_type.__name__}. Accepts: "
                    f"{[t.__name__ for t in step.input_types]}"
                )

            # Update current type to this step's output
            current_type = step.output_type_for_input(current_type)

        # Last step must produce IndexMap
        if current_type != IndexMap:
            raise TypeError(
                f"Pipeline must end with IndexMap. Last step produces: {current_type.__name__}"
            )

    def _validate_dimensions(self) -> None:
        """Validate dimension compatibility between steps.

        This validates that:
        - Pooling steps downsample (output <= input)
        - No upsampling between explicit sizes

        Note: Full validation requires input image size, which happens at run().
        This method only validates steps with explicit output_size.

        Raises:
            ValueError: If explicit dimensions are incompatible.
        """
        # Import here to avoid circular import
        from .steps.pool import PoolStep

        # Collect explicit sizes from pool steps
        explicit_sizes: list[tuple[int, tuple[int, int]]] = []

        for i, step in enumerate(self.steps):
            if isinstance(step, PoolStep) and step.config.output_size is not None:
                explicit_sizes.append((i, step.config.output_size))

        # Validate that each explicit size is <= the previous one
        for j in range(1, len(explicit_sizes)):
            prev_idx, prev_size = explicit_sizes[j - 1]
            curr_idx, curr_size = explicit_sizes[j]

            if curr_size[0] > prev_size[0] or curr_size[1] > prev_size[1]:
                raise ValueError(
                    f"Cannot upsample: step {prev_idx} outputs {prev_size}, "
                    f"but step {curr_idx} outputs {curr_size}"
                )

    def run(self, image: RGBImage, context: "PipelineContext") -> IndexMap:
        """Execute all steps in order.

        Args:
            image: Source image to process.
            context: Shared pipeline context.

        Returns:
            Final IndexMap ready for canvas conversion.

        Raises:
            ValueError: If final output size doesn't match context.target_size.
        """
        current: Any = image

        for step in self.steps:
            current = step.process(current, context)

        # Validate final size matches target
        result: IndexMap = current
        expected_size = context.target_size  # (width, height)
        actual_size = (result.width, result.height)

        if actual_size != expected_size:
            raise ValueError(
                f"Pipeline output size {actual_size} does not match target size {expected_size}"
            )

        return result

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        name_str = f"name={self.name!r}, " if self.name else ""
        steps_str = ", ".join(type(s).__name__ for s in self.steps)
        return f"Pipeline({name_str}steps=[{steps_str}])"
