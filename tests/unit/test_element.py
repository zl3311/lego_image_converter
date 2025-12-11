"""Unit tests for the Element class.

Tests cover initialization, validation, hash/equality, and string representation
with comprehensive coverage of valid and invalid inputs.
"""

import pytest

from mosaicpic.models import Element


class TestElementInit:
    """Tests for Element.__init__."""

    def test_init_required_params(self):
        """Element accepts required parameters."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1)

        assert elem.element_id == 1234567
        assert elem.design_id == 98138
        assert elem.variant_id == 1
        assert elem.count is None

    def test_init_with_count(self):
        """Element accepts optional count."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1, count=50)

        assert elem.count == 50

    def test_init_count_zero(self):
        """Element accepts count of zero."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1, count=0)

        assert elem.count == 0

    def test_init_large_values(self):
        """Element accepts large ID values."""
        elem = Element(element_id=99999999, design_id=99999999, variant_id=100, count=10000)

        assert elem.element_id == 99999999
        assert elem.design_id == 99999999
        assert elem.variant_id == 100
        assert elem.count == 10000


class TestElementValidation:
    """Tests for Element validation."""

    def test_element_id_zero_raises(self):
        """Element rejects element_id of zero."""
        with pytest.raises(ValueError, match="element_id must be positive"):
            Element(element_id=0, design_id=98138, variant_id=1)

    def test_element_id_negative_raises(self):
        """Element rejects negative element_id."""
        with pytest.raises(ValueError, match="element_id must be positive"):
            Element(element_id=-1, design_id=98138, variant_id=1)

    def test_design_id_zero_raises(self):
        """Element rejects design_id of zero."""
        with pytest.raises(ValueError, match="design_id must be positive"):
            Element(element_id=1234567, design_id=0, variant_id=1)

    def test_design_id_negative_raises(self):
        """Element rejects negative design_id."""
        with pytest.raises(ValueError, match="design_id must be positive"):
            Element(element_id=1234567, design_id=-1, variant_id=1)

    def test_variant_id_zero_raises(self):
        """Element rejects variant_id of zero."""
        with pytest.raises(ValueError, match="variant_id must be positive"):
            Element(element_id=1234567, design_id=98138, variant_id=0)

    def test_variant_id_negative_raises(self):
        """Element rejects negative variant_id."""
        with pytest.raises(ValueError, match="variant_id must be positive"):
            Element(element_id=1234567, design_id=98138, variant_id=-1)

    def test_count_negative_raises(self):
        """Element rejects negative count."""
        with pytest.raises(ValueError, match="count must be non-negative"):
            Element(element_id=1234567, design_id=98138, variant_id=1, count=-1)


class TestElementHashEquality:
    """Tests for Element.__hash__ and __eq__."""

    def test_equal_elements(self):
        """Elements with same element_id are equal."""
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1)
        elem2 = Element(element_id=1234567, design_id=98138, variant_id=1)

        assert elem1 == elem2

    def test_unequal_elements(self):
        """Elements with different element_id are not equal."""
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1)
        elem2 = Element(element_id=7654321, design_id=98138, variant_id=1)

        assert elem1 != elem2

    def test_equality_based_on_element_id_only(self):
        """Equality only considers element_id, not other attributes."""
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1, count=10)
        elem2 = Element(element_id=1234567, design_id=99999, variant_id=2, count=20)

        # Same element_id means equal (despite other differences)
        assert elem1 == elem2

    def test_equality_with_non_element(self):
        """Element is not equal to non-Element objects."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1)

        assert elem != 1234567
        assert elem != "1234567"
        assert elem is not None

    def test_hashable(self):
        """Elements can be used in sets/dicts."""
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1)
        elem2 = Element(element_id=7654321, design_id=98138, variant_id=1)
        elem3 = Element(element_id=1234567, design_id=99999, variant_id=2)  # Same as elem1

        elements = {elem1, elem2, elem3}

        # elem1 and elem3 have same element_id, so only 2 unique
        assert len(elements) == 2

    def test_same_hash_for_equal(self):
        """Equal elements have same hash."""
        elem1 = Element(element_id=1234567, design_id=98138, variant_id=1)
        elem2 = Element(element_id=1234567, design_id=99999, variant_id=2)

        assert hash(elem1) == hash(elem2)


class TestElementRepr:
    """Tests for Element.__repr__."""

    def test_repr_without_count(self):
        """__repr__ shows all attributes without count."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1)

        repr_str = repr(elem)

        assert "Element" in repr_str
        assert "element_id=1234567" in repr_str
        assert "design_id=98138" in repr_str
        assert "variant_id=1" in repr_str
        assert "count" not in repr_str

    def test_repr_with_count(self):
        """__repr__ shows count when provided."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1, count=50)

        repr_str = repr(elem)

        assert "Element" in repr_str
        assert "element_id=1234567" in repr_str
        assert "design_id=98138" in repr_str
        assert "variant_id=1" in repr_str
        assert "count=50" in repr_str

    def test_repr_with_count_zero(self):
        """__repr__ shows count when zero."""
        elem = Element(element_id=1234567, design_id=98138, variant_id=1, count=0)

        repr_str = repr(elem)

        assert "count=0" in repr_str


class TestElementUseCases:
    """Tests for Element real-world use cases."""

    def test_typical_tile_element(self):
        """Element works with typical tile element data."""
        # Real tile element: White 1x1 Round Tile
        elem = Element(element_id=6284572, design_id=98138, variant_id=2, count=100)

        assert elem.element_id == 6284572
        assert elem.design_id == 98138  # 1x1 Round Tile
        assert elem.variant_id == 2
        assert elem.count == 100

    def test_multiple_variants_same_design(self):
        """Elements can have same design but different element IDs."""
        # White variants (different production runs)
        white_v1 = Element(element_id=4646844, design_id=98138, variant_id=1)
        white_v2 = Element(element_id=6284572, design_id=98138, variant_id=2)
        white_v3 = Element(element_id=6564903, design_id=98138, variant_id=3)

        # All different element_ids
        elements = {white_v1, white_v2, white_v3}
        assert len(elements) == 3

        # But same design_id
        assert white_v1.design_id == white_v2.design_id == white_v3.design_id
