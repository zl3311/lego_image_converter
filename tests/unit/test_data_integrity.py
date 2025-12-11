"""Data integrity tests for tile color and palette CSV files.

This module provides comprehensive validation of the CSV data files to ensure
data quality and catch issues when contributors add new colors or palettes via PRs.

Validates:
    - colors.csv: Primary key uniqueness, color consistency, RGB validity
    - sets.csv: Primary key uniqueness, name uniqueness, dimension constraints
    - elements.csv: Composite key uniqueness, referential integrity

These tests run as part of CI to prevent data corruption from being merged.
"""

import csv
from collections import defaultdict
from pathlib import Path

import pytest

# Path to data directory
DATA_DIR = Path(__file__).parent.parent.parent / "src" / "mosaicpic" / "data"

# Constraints
MAX_CANVAS_DIMENSION = 1024  # Maximum allowed canvas width/height in studs
RGB_MIN = 0
RGB_MAX = 255


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def colors_data() -> list[dict[str, str]]:
    """Load colors.csv as a list of row dicts."""
    path = DATA_DIR / "colors.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def sets_data() -> list[dict[str, str]]:
    """Load sets.csv as a list of row dicts."""
    path = DATA_DIR / "sets.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def elements_data() -> list[dict[str, str]]:
    """Load elements.csv as a list of row dicts."""
    path = DATA_DIR / "elements.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# colors.csv Tests
# =============================================================================


class TestColorsCSV:
    """Tests for colors.csv data integrity."""

    def test_colors_file_exists(self) -> None:
        """Verify colors.csv exists in the data directory."""
        path = DATA_DIR / "colors.csv"
        assert path.exists(), f"colors.csv not found at {path}"

    def test_colors_has_required_columns(self, colors_data: list[dict[str, str]]) -> None:
        """Verify colors.csv has all required columns."""
        required_columns = {
            "design_id",
            "element_id",
            "name",
            "variant_id",
            "r",
            "g",
            "b",
            "is_standard",
        }
        if colors_data:
            actual_columns = set(colors_data[0].keys())
            missing = required_columns - actual_columns
            assert not missing, f"colors.csv missing required columns: {missing}"

    def test_no_duplicate_element_ids(self, colors_data: list[dict[str, str]]) -> None:
        """Verify element_id is unique (primary key constraint).

        Each element_id should appear exactly once in colors.csv since it
        uniquely identifies a specific tile element.
        """
        element_ids: dict[str, list[int]] = defaultdict(list)
        for row_num, row in enumerate(colors_data, start=2):  # CSV row 2 is first data row
            element_id = row["element_id"]
            element_ids[element_id].append(row_num)

        duplicates = {eid: rows for eid, rows in element_ids.items() if len(rows) > 1}
        assert not duplicates, "Duplicate element_id values found in colors.csv:\n" + "\n".join(
            f"  element_id={eid}: rows {rows}" for eid, rows in duplicates.items()
        )

    def test_no_exact_duplicate_colors(self, colors_data: list[dict[str, str]]) -> None:
        """Verify no exact duplicate colors within the same design_id.

        A duplicate is defined as same (design_id, name, r, g, b) tuple.
        Different variant_ids for the same color are allowed (different molds),
        but truly identical entries indicate a data entry error.
        """
        seen: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
        for row_num, row in enumerate(colors_data, start=2):
            key = (row["design_id"], row["name"], row["r"], row["g"], row["b"])
            seen[key].append(row_num)

        # Filter to find entries with duplicate variant_ids (true duplicates)
        # Group by (design_id, name, r, g, b, variant_id)
        full_duplicates: dict[tuple, list[int]] = defaultdict(list)
        for row_num, row in enumerate(colors_data, start=2):
            key = (
                row["design_id"],
                row["name"],
                row["r"],
                row["g"],
                row["b"],
                row["variant_id"],
            )
            full_duplicates[key].append(row_num)

        duplicates = {k: rows for k, rows in full_duplicates.items() if len(rows) > 1}
        assert not duplicates, "Exact duplicate color entries found in colors.csv:\n" + "\n".join(
            f"  {k}: rows {rows}" for k, rows in duplicates.items()
        )

    def test_consistent_rgb_for_color_names(self, colors_data: list[dict[str, str]]) -> None:
        """Verify colors with the same name have consistent RGB values.

        For a given color name (e.g., "Red"), all entries should have the same
        RGB values. This catches typos or data entry errors where the same
        logical color is entered with different RGB values.

        Note: Some colors may legitimately have the same name but different RGB
        in different contexts (e.g., design_id variations). This test groups
        by name only and flags any inconsistencies for review.
        """
        rgb_by_name: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        rows_by_name_rgb: dict[str, dict[tuple[str, str, str], list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for row_num, row in enumerate(colors_data, start=2):
            name = row["name"]
            rgb = (row["r"], row["g"], row["b"])
            rgb_by_name[name].add(rgb)
            rows_by_name_rgb[name][rgb].append(row_num)

        inconsistent = {
            name: dict(rows_by_name_rgb[name])
            for name, rgbs in rgb_by_name.items()
            if len(rgbs) > 1
        }

        assert not inconsistent, (
            "Inconsistent RGB values for color names in colors.csv:\n"
            + "\n".join(f"  '{name}': {dict(rgb_rows)}" for name, rgb_rows in inconsistent.items())
        )

    def test_rgb_values_in_valid_range(self, colors_data: list[dict[str, str]]) -> None:
        """Verify all RGB values are integers in range [0, 255]."""
        errors: list[str] = []
        for row_num, row in enumerate(colors_data, start=2):
            for channel in ("r", "g", "b"):
                try:
                    value = int(row[channel])
                    if not (RGB_MIN <= value <= RGB_MAX):
                        errors.append(f"Row {row_num}: {channel}={value} out of range [0, 255]")
                except ValueError:
                    errors.append(
                        f"Row {row_num}: {channel}='{row[channel]}' is not a valid integer"
                    )

        assert not errors, "Invalid RGB values in colors.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )

    def test_is_standard_valid_boolean(self, colors_data: list[dict[str, str]]) -> None:
        """Verify is_standard column contains valid boolean strings."""
        valid_values = {"true", "false"}
        errors: list[str] = []
        for row_num, row in enumerate(colors_data, start=2):
            if row["is_standard"].lower() not in valid_values:
                errors.append(
                    f"Row {row_num}: is_standard='{row['is_standard']}' "
                    f"(expected 'true' or 'false')"
                )

        assert not errors, "Invalid is_standard values in colors.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )

    def test_variant_id_positive_integer(self, colors_data: list[dict[str, str]]) -> None:
        """Verify variant_id is a positive integer."""
        errors: list[str] = []
        for row_num, row in enumerate(colors_data, start=2):
            try:
                value = int(row["variant_id"])
                if value < 1:
                    errors.append(f"Row {row_num}: variant_id={value} must be >= 1")
            except ValueError:
                errors.append(
                    f"Row {row_num}: variant_id='{row['variant_id']}' is not a valid integer"
                )

        assert not errors, "Invalid variant_id values in colors.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )


# =============================================================================
# sets.csv Tests
# =============================================================================


class TestSetsCSV:
    """Tests for sets.csv data integrity."""

    def test_sets_file_exists(self) -> None:
        """Verify sets.csv exists in the data directory."""
        path = DATA_DIR / "sets.csv"
        assert path.exists(), f"sets.csv not found at {path}"

    def test_sets_has_required_columns(self, sets_data: list[dict[str, str]]) -> None:
        """Verify sets.csv has all required columns."""
        required_columns = {"set_id", "name", "canvas_width", "canvas_height"}
        if sets_data:
            actual_columns = set(sets_data[0].keys())
            missing = required_columns - actual_columns
            assert not missing, f"sets.csv missing required columns: {missing}"

    def test_no_duplicate_set_ids(self, sets_data: list[dict[str, str]]) -> None:
        """Verify set_id is unique (primary key constraint)."""
        set_ids: dict[str, list[int]] = defaultdict(list)
        for row_num, row in enumerate(sets_data, start=2):
            set_id = row["set_id"]
            set_ids[set_id].append(row_num)

        duplicates = {sid: rows for sid, rows in set_ids.items() if len(rows) > 1}
        assert not duplicates, "Duplicate set_id values found in sets.csv:\n" + "\n".join(
            f"  set_id={sid}: rows {rows}" for sid, rows in duplicates.items()
        )

    def test_no_duplicate_set_names(self, sets_data: list[dict[str, str]]) -> None:
        """Verify set names are unique.

        While technically allowed, duplicate names likely indicate a data entry
        error since each palette has a unique name.
        """
        names: dict[str, list[int]] = defaultdict(list)
        for row_num, row in enumerate(sets_data, start=2):
            name = row["name"]
            names[name].append(row_num)

        duplicates = {name: rows for name, rows in names.items() if len(rows) > 1}
        assert not duplicates, "Duplicate set names found in sets.csv:\n" + "\n".join(
            f"  name='{name}': rows {rows}" for name, rows in duplicates.items()
        )

    def test_canvas_dimensions_positive(self, sets_data: list[dict[str, str]]) -> None:
        """Verify canvas dimensions are positive integers."""
        errors: list[str] = []
        for row_num, row in enumerate(sets_data, start=2):
            for dim in ("canvas_width", "canvas_height"):
                try:
                    value = int(row[dim])
                    if value <= 0:
                        errors.append(f"Row {row_num}: {dim}={value} must be > 0")
                except ValueError:
                    errors.append(f"Row {row_num}: {dim}='{row[dim]}' is not a valid integer")

        assert not errors, "Invalid canvas dimensions in sets.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )

    def test_canvas_dimensions_within_limit(self, sets_data: list[dict[str, str]]) -> None:
        """Verify canvas dimensions do not exceed maximum allowed size.

        Canvas dimensions are limited to 1024x1024 studs to prevent
        unreasonably large mosaics that could cause performance issues.
        """
        errors: list[str] = []
        for row_num, row in enumerate(sets_data, start=2):
            width = int(row["canvas_width"])
            height = int(row["canvas_height"])
            if width > MAX_CANVAS_DIMENSION:
                errors.append(
                    f"Row {row_num}: canvas_width={width} exceeds max {MAX_CANVAS_DIMENSION}"
                )
            if height > MAX_CANVAS_DIMENSION:
                errors.append(
                    f"Row {row_num}: canvas_height={height} exceeds max {MAX_CANVAS_DIMENSION}"
                )

        assert not errors, (
            f"Canvas dimensions exceed maximum ({MAX_CANVAS_DIMENSION}) in sets.csv:\n"
            + "\n".join(f"  {e}" for e in errors)
        )


# =============================================================================
# elements.csv Tests
# =============================================================================


class TestElementsCSV:
    """Tests for elements.csv data integrity."""

    def test_elements_file_exists(self) -> None:
        """Verify elements.csv exists in the data directory."""
        path = DATA_DIR / "elements.csv"
        assert path.exists(), f"elements.csv not found at {path}"

    def test_elements_has_required_columns(self, elements_data: list[dict[str, str]]) -> None:
        """Verify elements.csv has all required columns."""
        required_columns = {"set_id", "design_id", "element_id", "count"}
        if elements_data:
            actual_columns = set(elements_data[0].keys())
            missing = required_columns - actual_columns
            assert not missing, f"elements.csv missing required columns: {missing}"

    def test_no_duplicate_set_element_pairs(self, elements_data: list[dict[str, str]]) -> None:
        """Verify (set_id, element_id) pairs are unique.

        Each element should appear at most once per set.
        """
        pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row_num, row in enumerate(elements_data, start=2):
            key = (row["set_id"], row["element_id"])
            pairs[key].append(row_num)

        duplicates = {pair: rows for pair, rows in pairs.items() if len(rows) > 1}
        assert not duplicates, (
            "Duplicate (set_id, element_id) pairs found in elements.csv:\n"
            + "\n".join(
                f"  (set_id={pair[0]}, element_id={pair[1]}): rows {rows}"
                for pair, rows in duplicates.items()
            )
        )

    def test_element_ids_exist_in_colors(
        self,
        elements_data: list[dict[str, str]],
        colors_data: list[dict[str, str]],
    ) -> None:
        """Verify all element_ids in elements.csv exist in colors.csv.

        This is a referential integrity check to ensure the elements
        reference valid colors.
        """
        valid_element_ids = {row["element_id"] for row in colors_data}

        errors: list[str] = []
        for row_num, row in enumerate(elements_data, start=2):
            element_id = row["element_id"]
            if element_id not in valid_element_ids:
                errors.append(
                    f"Row {row_num}: element_id={element_id} not found in colors.csv "
                    f"(set_id={row['set_id']})"
                )

        assert not errors, "Invalid element_id references in elements.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )

    def test_set_ids_exist_in_sets(
        self,
        elements_data: list[dict[str, str]],
        sets_data: list[dict[str, str]],
    ) -> None:
        """Verify all set_ids in elements.csv exist in sets.csv.

        This is a referential integrity check to ensure the elements
        reference valid sets.
        """
        valid_set_ids = {row["set_id"] for row in sets_data}

        errors: list[str] = []
        for row_num, row in enumerate(elements_data, start=2):
            set_id = row["set_id"]
            if set_id not in valid_set_ids:
                errors.append(f"Row {row_num}: set_id={set_id} not found in sets.csv")

        assert not errors, "Invalid set_id references in elements.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )

    def test_count_positive_integer(self, elements_data: list[dict[str, str]]) -> None:
        """Verify count values are positive integers."""
        errors: list[str] = []
        for row_num, row in enumerate(elements_data, start=2):
            try:
                value = int(row["count"])
                if value <= 0:
                    errors.append(
                        f"Row {row_num}: count={value} must be > 0 "
                        f"(set_id={row['set_id']}, element_id={row['element_id']})"
                    )
            except ValueError:
                errors.append(f"Row {row_num}: count='{row['count']}' is not a valid integer")

        assert not errors, "Invalid count values in elements.csv:\n" + "\n".join(
            f"  {e}" for e in errors
        )


# =============================================================================
# Cross-file Consistency Tests
# =============================================================================


class TestCrossFileConsistency:
    """Tests for consistency across multiple data files."""

    def test_all_sets_have_elements(
        self,
        sets_data: list[dict[str, str]],
        elements_data: list[dict[str, str]],
    ) -> None:
        """Verify every set defined in sets.csv has at least one element.

        A set without elements is likely a data entry error.
        """
        defined_sets = {row["set_id"] for row in sets_data}
        sets_with_elements = {row["set_id"] for row in elements_data}

        empty_sets = defined_sets - sets_with_elements
        assert not empty_sets, (
            "Sets defined in sets.csv but have no elements in elements.csv:\n"
            + "\n".join(f"  set_id={sid}" for sid in sorted(empty_sets))
        )

    def test_design_ids_consistent(
        self,
        colors_data: list[dict[str, str]],
        elements_data: list[dict[str, str]],
    ) -> None:
        """Verify design_id is consistent between colors.csv and elements.csv.

        For each element_id in elements.csv, the design_id should match
        what's specified in colors.csv.
        """
        color_design_ids = {row["element_id"]: row["design_id"] for row in colors_data}

        errors: list[str] = []
        for row_num, row in enumerate(elements_data, start=2):
            element_id = row["element_id"]
            if element_id in color_design_ids:
                expected_design_id = color_design_ids[element_id]
                actual_design_id = row["design_id"]
                if expected_design_id != actual_design_id:
                    errors.append(
                        f"Row {row_num}: element_id={element_id} has design_id={actual_design_id} "
                        f"but colors.csv has design_id={expected_design_id}"
                    )

        assert not errors, (
            "Inconsistent design_id values between colors.csv and elements.csv:\n"
            + "\n".join(f"  {e}" for e in errors)
        )
