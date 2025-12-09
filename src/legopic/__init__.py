"""legopic: Convert images to Lego mosaics.

A Python package for converting images to Lego mosaic patterns by
matching colors to available Lego brick colors.

Example usage:
    >>> from legopic import ConversionSession, ConvertConfig, Palette, load_image
    >>>
    >>> # Setup (hard params)
    >>> image = load_image("photo.jpg")
    >>> palette = Palette.from_set(31197)  # Andy Warhol set
    >>> session = ConversionSession(image, palette, (48, 48))
    >>>
    >>> # Convert (soft params)
    >>> config = ConvertConfig(method='match_then_mode', limit_inventory=True)
    >>> session.convert(config)
    >>> print(f"Similarity: {session.similarity_score:.2f}")
    >>>
    >>> # Adjust colors
    >>> session.pin(3, 5, some_blue_color)  # Pin and optionally change
    >>> session.swap_color(old_red, new_orange)  # Bulk swap
    >>>
    >>> # Re-convert with different method, keep pins
    >>> session.reconvert(ConvertConfig(method='mean_then_match'), keep_pins=True)
    >>>
    >>> # Export for building guide
    >>> bom = session.get_bill_of_materials()
    >>> grid = session.get_grid_data()
    >>> similarity_map = session.get_similarity_map()
    >>>
    >>> # Export for external platforms
    >>> xml = session.export_bricklink_xml()  # BrickLink wanted list
    >>> csv = session.export_rebrickable_csv()  # Rebrickable parts list
"""

__version__ = "0.4.3"

from .core import (
    ConversionSession,
    ConvertConfig,
    downsize,
    export_bricklink_xml,
    export_rebrickable_csv,
    load_image,
    match_color,
)
from .models import BOMEntry, Canvas, Cell, CellData, Color, Element, Image, Palette

__all__ = [
    "__version__",
    # Main API
    "ConversionSession",
    "ConvertConfig",
    # Models
    "Color",
    "Cell",
    "Element",
    "Image",
    "Canvas",
    "Palette",
    "BOMEntry",
    "CellData",
    # Internal functions (exposed for advanced use)
    "load_image",
    "downsize",
    "match_color",
    # Export functions
    "export_bricklink_xml",
    "export_rebrickable_csv",
]
