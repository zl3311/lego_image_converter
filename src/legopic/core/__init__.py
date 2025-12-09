"""Core algorithms for the legopic package.

This module exports the main API and internal functions for image-to-Lego
conversion.

Public API:
    ConversionSession: Main API for image-to-Lego conversion workflow.
    ConvertConfig: Configuration for conversion parameters.

Internal Functions (exposed for advanced use):
    load_image: Load images from files or URLs.
    downsize: Reduce image to canvas dimensions.
    match_color: Match colors to nearest palette colors.

Export Functions:
    export_bricklink_xml: Export BOM to BrickLink XML format.
    export_rebrickable_csv: Export BOM to Rebrickable CSV format.
"""

from .config import ConvertConfig
from .downsize import downsize
from .export import export_bricklink_xml, export_rebrickable_csv
from .loader import load_image
from .match_color import match_color
from .session import ConversionSession

__all__ = [
    "ConversionSession",
    "ConvertConfig",
    "load_image",
    "downsize",
    "match_color",
    "export_bricklink_xml",
    "export_rebrickable_csv",
]
