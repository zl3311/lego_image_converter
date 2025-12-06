"""Element model representing a specific LEGO piece variant.

An Element is a specific LEGO piece identified by its element_id. Multiple
elements can represent the same color (variants), allowing for substitution
when inventory runs low.

Data Model Relationships:
    - Each Element has a unique element_id
    - Multiple Elements can share the same color name and RGB (variants)
    - Elements belong to sets via the elements.csv mapping
    - The design_id identifies the brick shape (98138 = 1x1 round tile)
"""


class Element:
    """Represents a specific LEGO element (a color variant with inventory).

    Elements are the physical LEGO pieces that can be purchased and used.
    Multiple elements may have the same color (RGB) but different element_ids,
    representing different production runs or variants.

    Attributes:
        element_id: Unique LEGO element identifier (e.g., 6284572).
        design_id: LEGO design/mold identifier (e.g., 98138 for 1x1 round tile).
        variant_id: Variant number for colors with multiple elements (1, 2, 3...).
        count: Number of pieces available in inventory. None if not tracking.

    Example:
        >>> # White has multiple variants with the same RGB
        >>> white_v1 = Element(element_id=4646844, design_id=98138, variant_id=1)
        >>> white_v2 = Element(element_id=6284572, design_id=98138, variant_id=2)
        >>> # Both represent the same color but are different physical elements
    """

    def __init__(self, element_id: int, design_id: int, variant_id: int, count: int | None = None):
        """Initialize an Element with its identifiers.

        Args:
            element_id (int): Unique LEGO element identifier.
            design_id (int): LEGO design/mold identifier.
            variant_id (int): Variant number (1-indexed).
            count (int | None): Optional inventory count. None means not
                tracking inventory.

        Raises:
            ValueError: If element_id, design_id, or variant_id is not positive.
        """
        if element_id <= 0:
            raise ValueError(f"element_id must be positive. Got {element_id}.")
        if design_id <= 0:
            raise ValueError(f"design_id must be positive. Got {design_id}.")
        if variant_id <= 0:
            raise ValueError(f"variant_id must be positive. Got {variant_id}.")
        if count is not None and count < 0:
            raise ValueError(f"count must be non-negative. Got {count}.")

        self.element_id = element_id
        self.design_id = design_id
        self.variant_id = variant_id
        self.count = count

    def __hash__(self) -> int:
        """Return hash based on element_id for use in sets/dicts."""
        return hash(self.element_id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on element_id."""
        if not isinstance(other, Element):
            return NotImplemented
        return self.element_id == other.element_id

    def __repr__(self) -> str:
        """Return string representation of the Element."""
        count_str = f", count={self.count}" if self.count is not None else ""
        return (
            f"Element(element_id={self.element_id}, "
            f"design_id={self.design_id}, "
            f"variant_id={self.variant_id}{count_str})"
        )
