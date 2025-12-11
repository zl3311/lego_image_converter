"""Base protocol for pipeline steps.

This module defines the Step protocol that all pipeline steps must implement.
The protocol enables type-safe step composition validation at pipeline construction.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..types import StepData


@runtime_checkable
class Step(Protocol):
    """Protocol for pipeline steps.

    Each step transforms data from one type to another.
    Steps declare their accepted input types and produced output types.

    Attributes:
        input_types: Tuple of types this step can accept.

    Methods:
        output_type_for_input: Determine output type given input type.
        process: Transform input data.

    Example:
        >>> class MyStep:
        ...     @property
        ...     def input_types(self) -> tuple[type, ...]:
        ...         return (RGBImage,)
        ...
        ...     def output_type_for_input(self, input_type: type) -> type:
        ...         return IndexMap
        ...
        ...     def process(self, input: StepData, context: PipelineContext) -> StepData:
        ...         # Transform input
        ...         return result
    """

    @property
    def input_types(self) -> tuple[type, ...]:
        """Types this step can accept as input."""
        ...

    def output_type_for_input(self, input_type: type) -> type:
        """Determine output type given an input type.

        Args:
            input_type: The type of data that will be passed to process().

        Returns:
            The type that process() will return.

        Raises:
            TypeError: If input_type is not in input_types.
        """
        ...

    def process(self, input: "StepData", context: "PipelineContext") -> "StepData":
        """Process input and return output.

        Args:
            input: The input data (must be one of input_types).
            context: Shared context with palette, target size, etc.

        Returns:
            Processed output (type determined by output_type_for_input).
        """
        ...
