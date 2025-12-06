"""Unit tests for the match_color function.

Tests cover input validation, distance computation, and ranking
with comprehensive coverage of various color matching scenarios.
"""

import numpy as np
import pytest

from legopic.core.match_color import match_color


class TestMatchColorValidation:
    """Tests for match_color input validation."""

    def test_valid_inputs(self):
        """match_color accepts valid 2D arrays with 3 columns."""
        targets = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        distances, rankings = match_color(targets, palette)

        assert distances is not None
        assert rankings is not None

    def test_target_1d_array_raises(self):
        """match_color rejects 1D target array."""
        targets = np.array([255, 0, 0], dtype=np.uint8)
        palette = np.array([[255, 0, 0]], dtype=np.uint8)

        with pytest.raises(ValueError, match="target_colors must be 2D"):
            match_color(targets, palette)

    def test_target_3d_array_raises(self):
        """match_color rejects 3D target array."""
        targets = np.array([[[255, 0, 0]]], dtype=np.uint8)
        palette = np.array([[255, 0, 0]], dtype=np.uint8)

        with pytest.raises(ValueError, match="target_colors must be 2D"):
            match_color(targets, palette)

    def test_target_wrong_columns_raises(self):
        """match_color rejects target array without 3 columns."""
        targets = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        palette = np.array([[255, 0, 0]], dtype=np.uint8)

        with pytest.raises(ValueError, match="target_colors must have 3 columns"):
            match_color(targets, palette)

    def test_palette_1d_array_raises(self):
        """match_color rejects 1D palette array."""
        targets = np.array([[255, 0, 0]], dtype=np.uint8)
        palette = np.array([255, 0, 0], dtype=np.uint8)

        with pytest.raises(ValueError, match="palette_colors must be 2D"):
            match_color(targets, palette)

    def test_palette_wrong_columns_raises(self):
        """match_color rejects palette array without 3 columns."""
        targets = np.array([[255, 0, 0]], dtype=np.uint8)
        palette = np.array([[255, 0, 0, 255]], dtype=np.uint8)

        with pytest.raises(ValueError, match="palette_colors must have 3 columns"):
            match_color(targets, palette)


class TestMatchColorOutputShape:
    """Tests for match_color output shapes."""

    def test_distances_shape(self):
        """Distances array has shape (n_targets, n_palette)."""
        targets = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        distances, _ = match_color(targets, palette)

        assert distances.shape == (3, 2)  # 3 targets, 2 palette colors

    def test_rankings_shape(self):
        """Rankings array has shape (n_targets, n_palette)."""
        targets = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        _, rankings = match_color(targets, palette)

        assert rankings.shape == (3, 2)  # 3 targets, 2 palette colors

    def test_single_target_single_palette(self):
        """Works with single target and single palette color."""
        targets = np.array([[255, 0, 0]], dtype=np.uint8)
        palette = np.array([[255, 0, 0]], dtype=np.uint8)

        distances, rankings = match_color(targets, palette)

        assert distances.shape == (1, 1)
        assert rankings.shape == (1, 1)

    def test_many_targets_many_palette(self):
        """Works with many targets and many palette colors."""
        targets = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        palette = np.random.randint(0, 256, (20, 3), dtype=np.uint8)

        distances, rankings = match_color(targets, palette)

        assert distances.shape == (100, 20)
        assert rankings.shape == (100, 20)


class TestMatchColorExactMatches:
    """Tests for exact color matching."""

    def test_exact_match_distance_zero(self):
        """Exact matching colors have distance (near) zero."""
        red = np.array([[255, 0, 0]], dtype=np.uint8)
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        distances, _ = match_color(red, palette)

        # First palette color is exact match
        assert distances[0, 0] < 0.1  # Should be essentially zero

    def test_exact_match_ranked_first(self):
        """Exact matching color is ranked first."""
        red = np.array([[255, 0, 0]], dtype=np.uint8)
        palette = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        # Red is at index 1 in palette

        _, rankings = match_color(red, palette)

        assert rankings[0, 0] == 1  # Index 1 (red) should be first rank


class TestMatchColorRankings:
    """Tests for ranking correctness."""

    def test_rankings_are_sorted_indices(self):
        """Rankings contain palette indices sorted by distance."""
        target = np.array([[255, 0, 0]], dtype=np.uint8)  # Red
        palette = np.array(
            [
                [0, 0, 255],  # Blue (far from red)
                [255, 0, 0],  # Red (exact match)
                [200, 50, 50],  # Reddish (close to red)
            ],
            dtype=np.uint8,
        )

        distances, rankings = match_color(target, palette)

        # First rank should be index 1 (exact red)
        assert rankings[0, 0] == 1

        # Verify distances are sorted
        ranked_distances = distances[0, rankings[0]]
        assert np.all(ranked_distances[:-1] <= ranked_distances[1:])

    def test_all_palette_indices_present(self):
        """Rankings contain all palette indices exactly once per target."""
        targets = np.array([[128, 128, 128]], dtype=np.uint8)
        palette = np.array(
            [
                [0, 0, 0],
                [255, 255, 255],
                [128, 128, 128],
                [255, 0, 0],
            ],
            dtype=np.uint8,
        )

        _, rankings = match_color(targets, palette)

        # Should contain all indices 0, 1, 2, 3
        assert set(rankings[0]) == {0, 1, 2, 3}


class TestMatchColorPerceptualDistance:
    """Tests for perceptual color distance properties."""

    def test_similar_colors_small_distance(self):
        """Perceptually similar colors have small distance."""
        target = np.array([[100, 100, 100]], dtype=np.uint8)  # Gray
        palette = np.array(
            [
                [105, 100, 100],  # Slightly different
                [255, 0, 0],  # Very different (red)
            ],
            dtype=np.uint8,
        )

        distances, _ = match_color(target, palette)

        # First palette color should be much closer
        assert distances[0, 0] < distances[0, 1]

    def test_black_white_large_distance(self):
        """Black and white have large perceptual distance."""
        black = np.array([[0, 0, 0]], dtype=np.uint8)
        palette = np.array([[255, 255, 255]], dtype=np.uint8)

        distances, _ = match_color(black, palette)

        # Delta E between black and white should be substantial
        assert distances[0, 0] > 50

    def test_primary_colors_distinct(self):
        """Primary colors are perceptually distinct."""
        targets = np.array(
            [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
            ],
            dtype=np.uint8,
        )

        palette = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )

        _, rankings = match_color(targets, palette)

        # Each primary should match itself first
        assert rankings[0, 0] == 0  # Red -> Red
        assert rankings[1, 0] == 1  # Green -> Green
        assert rankings[2, 0] == 2  # Blue -> Blue


class TestMatchColorEdgeCases:
    """Tests for edge cases."""

    def test_all_same_targets(self):
        """Works when all targets are identical."""
        targets = np.array(
            [
                [128, 128, 128],
                [128, 128, 128],
                [128, 128, 128],
            ],
            dtype=np.uint8,
        )
        palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

        distances, rankings = match_color(targets, palette)

        # All rows should be identical
        assert np.all(distances[0] == distances[1])
        assert np.all(distances[1] == distances[2])
        assert np.all(rankings[0] == rankings[1])
        assert np.all(rankings[1] == rankings[2])

    def test_all_same_palette(self):
        """Works when all palette colors are identical."""
        targets = np.array([[128, 128, 128]], dtype=np.uint8)
        palette = np.array(
            [
                [100, 100, 100],
                [100, 100, 100],
            ],
            dtype=np.uint8,
        )

        distances, _ = match_color(targets, palette)

        # Both palette colors should have same distance
        assert abs(distances[0, 0] - distances[0, 1]) < 0.1

    def test_boundary_rgb_values(self):
        """Works with boundary RGB values (0 and 255)."""
        targets = np.array(
            [
                [0, 0, 0],
                [255, 255, 255],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        )
        palette = np.array(
            [
                [0, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

        distances, rankings = match_color(targets, palette)

        # Black should match black
        assert rankings[0, 0] == 0
        # White should match white
        assert rankings[1, 0] == 1


class TestMatchColorDistanceProperties:
    """Tests for distance matrix properties."""

    def test_distances_non_negative(self):
        """All distances are non-negative."""
        targets = np.random.randint(0, 256, (10, 3), dtype=np.uint8)
        palette = np.random.randint(0, 256, (5, 3), dtype=np.uint8)

        distances, _ = match_color(targets, palette)

        assert np.all(distances >= 0)

    def test_self_distance_zero(self):
        """Distance to same color is (near) zero."""
        colors = np.array(
            [
                [100, 150, 200],
                [50, 100, 150],
            ],
            dtype=np.uint8,
        )

        # Match colors against themselves
        distances, _ = match_color(colors, colors)

        # Diagonal should be zero
        assert distances[0, 0] < 0.1
        assert distances[1, 1] < 0.1
