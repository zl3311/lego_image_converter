"""Unit tests for pipeline profile registry.

This module tests the profile registry system in profiles.py:
- Profile registration and retrieval
- Built-in profile existence and structure
- Error handling for invalid profiles
- Custom profile registration
"""

import pytest

from legopic.pipeline import (
    DitherStep,
    Pipeline,
    PoolStep,
    QuantizeStep,
    get_profile,
    list_profiles,
    register_profile,
)


class TestBuiltinProfiles:
    """Tests for built-in pipeline profiles."""

    def test_classic_profile_exists(self):
        """'classic' profile is registered and retrievable."""
        profile = get_profile("classic")

        assert isinstance(profile, Pipeline)
        assert profile.name == "classic"

    def test_sharp_profile_exists(self):
        """'sharp' profile is registered and retrievable."""
        profile = get_profile("sharp")

        assert isinstance(profile, Pipeline)
        assert profile.name == "sharp"

    def test_dithered_profile_exists(self):
        """'dithered' profile is registered and retrievable."""
        profile = get_profile("dithered")

        assert isinstance(profile, Pipeline)
        assert profile.name == "dithered"

    def test_list_profiles_contains_builtins(self):
        """list_profiles() returns all built-in profile names."""
        profiles = list_profiles()

        assert "classic" in profiles
        assert "sharp" in profiles
        assert "dithered" in profiles

    def test_list_profiles_returns_list(self):
        """list_profiles() returns a list type."""
        profiles = list_profiles()

        assert isinstance(profiles, list)
        assert len(profiles) >= 3  # At least the 3 built-ins


class TestProfileStructure:
    """Tests for the structure of built-in profiles."""

    def test_classic_profile_structure(self):
        """Classic profile has Pool -> Quantize steps."""
        profile = get_profile("classic")

        assert len(profile.steps) == 2
        assert isinstance(profile.steps[0], PoolStep)
        assert isinstance(profile.steps[1], QuantizeStep)

    def test_sharp_profile_structure(self):
        """Sharp profile has Quantize -> Pool steps."""
        profile = get_profile("sharp")

        assert len(profile.steps) == 2
        assert isinstance(profile.steps[0], QuantizeStep)
        assert isinstance(profile.steps[1], PoolStep)

    def test_dithered_profile_structure(self):
        """Dithered profile has Pool -> Dither steps."""
        profile = get_profile("dithered")

        assert len(profile.steps) == 2
        assert isinstance(profile.steps[0], PoolStep)
        assert isinstance(profile.steps[1], DitherStep)


class TestProfileRetrieval:
    """Tests for profile retrieval and error handling."""

    def test_get_profile_invalid_raises(self):
        """get_profile raises ValueError for unknown profile names."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent_profile")

    def test_get_profile_error_lists_available(self):
        """Error message includes list of available profiles."""
        with pytest.raises(ValueError, match="Available:") as exc_info:
            get_profile("invalid")

        error_msg = str(exc_info.value)
        assert "classic" in error_msg
        assert "sharp" in error_msg
        assert "dithered" in error_msg

    def test_get_profile_case_sensitive(self):
        """Profile names are case-sensitive."""
        with pytest.raises(ValueError):
            get_profile("CLASSIC")

        with pytest.raises(ValueError):
            get_profile("Classic")


class TestProfileRegistration:
    """Tests for custom profile registration."""

    def test_register_custom_profile(self):
        """Custom profiles can be registered and retrieved."""
        custom_pipeline = Pipeline(
            [
                PoolStep(),
                QuantizeStep(),
            ],
            name="test_custom",
        )

        # Register the custom profile
        register_profile("test_custom_registration", custom_pipeline)

        # Retrieve it
        retrieved = get_profile("test_custom_registration")

        assert retrieved is custom_pipeline
        assert retrieved.name == "test_custom_registration"  # Name updated by registration

    def test_register_profile_updates_name(self):
        """Registration updates the pipeline's name to match registration name."""
        custom_pipeline = Pipeline(
            [
                PoolStep(),
                QuantizeStep(),
            ],
            name="original_name",
        )

        register_profile("new_name", custom_pipeline)

        assert custom_pipeline.name == "new_name"

    def test_register_profile_appears_in_list(self):
        """Registered profiles appear in list_profiles()."""
        custom_pipeline = Pipeline(
            [
                PoolStep(),
                QuantizeStep(),
            ],
        )

        profile_name = "test_list_appearance"
        register_profile(profile_name, custom_pipeline)

        profiles = list_profiles()
        assert profile_name in profiles

    def test_register_can_override_builtin(self):
        """Custom profile can override built-in (not recommended but possible)."""
        # Get original classic
        original = get_profile("classic")
        original_step_count = len(original.steps)

        # Create a different pipeline
        override = Pipeline(
            [
                QuantizeStep(),  # Different first step
                PoolStep(),
            ],
            name="classic",
        )

        # Register override
        register_profile("classic", override)

        # Retrieve - should get override
        retrieved = get_profile("classic")

        # Should have same step types in different order
        assert isinstance(retrieved.steps[0], QuantizeStep)

        # Restore original for other tests (cleanup)
        register_profile("classic", original)
        assert len(get_profile("classic").steps) == original_step_count


class TestProfileRepr:
    """Tests for profile Pipeline string representation."""

    def test_named_profile_repr(self):
        """Named profile includes name in repr."""
        profile = get_profile("classic")
        repr_str = repr(profile)

        assert "Pipeline" in repr_str
        assert "classic" in repr_str
        assert "PoolStep" in repr_str
        assert "QuantizeStep" in repr_str

    def test_unnamed_pipeline_repr(self):
        """Unnamed pipeline repr doesn't include name."""
        pipeline = Pipeline(
            [
                PoolStep(),
                QuantizeStep(),
            ]
        )
        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "PoolStep" in repr_str
        assert "QuantizeStep" in repr_str
        # No name= prefix when name is None
        assert "name=" not in repr_str
