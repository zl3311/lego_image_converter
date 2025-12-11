"""Built-in pipeline profiles.

This module provides:
- Profile registry for named pipeline configurations
- Built-in profiles: "classic", "sharp", "dithered"

Profiles are registered at module load time and can be retrieved by name.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline

# Global profile registry
_PROFILES: dict[str, "Pipeline"] = {}


def register_profile(name: str, pipeline: "Pipeline") -> None:
    """Register a named profile.

    Args:
        name: Profile name (e.g., "classic", "dithered").
        pipeline: Pipeline configuration.
    """
    pipeline.name = name
    _PROFILES[name] = pipeline


def get_profile(name: str) -> "Pipeline":
    """Get a profile by name.

    Args:
        name: Profile name.

    Returns:
        The registered Pipeline.

    Raises:
        ValueError: If profile name is not found.
    """
    if name not in _PROFILES:
        available = list(_PROFILES.keys())
        raise ValueError(f"Unknown profile: '{name}'. Available: {available}")
    return _PROFILES[name]


def list_profiles() -> list[str]:
    """List all registered profile names.

    Returns:
        List of available profile names.
    """
    return list(_PROFILES.keys())


def _register_builtin_profiles() -> None:
    """Register built-in profiles.

    Called at module import time to set up default profiles.
    """
    from .config import DitherAlgorithm, DitherConfig, PoolConfig, PoolMethod
    from .pipeline import Pipeline
    from .steps import DitherStep, PoolStep, QuantizeStep

    # Classic: Simple mean pooling + nearest-neighbor quantization
    # Equivalent to old method='mean_then_match'
    register_profile(
        "classic",
        Pipeline(
            [
                PoolStep(PoolConfig(method=PoolMethod.MEAN)),
                QuantizeStep(),
            ]
        ),
    )

    # Sharp: Quantize first, then mode-pool
    # Equivalent to old method='match_then_mode'
    # Good for preserving sharp edges and distinct colors
    register_profile(
        "sharp",
        Pipeline(
            [
                QuantizeStep(),
                PoolStep(PoolConfig(method=PoolMethod.MODE)),
            ]
        ),
    )

    # Dithered: Mean pooling + Floyd-Steinberg dithering
    # NEW! Good for photo-realistic images with gradients
    register_profile(
        "dithered",
        Pipeline(
            [
                PoolStep(PoolConfig(method=PoolMethod.MEAN)),
                DitherStep(DitherConfig(algorithm=DitherAlgorithm.FLOYD_STEINBERG)),
            ]
        ),
    )


# Register built-in profiles when module is imported
_register_builtin_profiles()
