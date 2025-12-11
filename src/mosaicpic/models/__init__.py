"""Data models for the mosaicpic package.

This module exports all model classes used to represent images, canvases,
colors, cells, elements, palettes, and export data structures.

Classes:
    Color: RGB color with optional name.
    Cell: Single pixel/block unit with color and position.
    Element: Tile piece variant with inventory.
    Image: Input image as grid of cells.
    Canvas: Output mosaic grid.
    Palette: Collection of available colors for matching.
    BOMEntry: Bill of materials entry for building guide.
    CellData: Cell data for grid export.
"""

from .bom_entry import BOMEntry
from .canvas import Canvas
from .cell import Cell
from .cell_data import CellData
from .color import Color
from .element import Element
from .image import Image
from .palette import Palette

__all__ = [
    "Color",
    "Cell",
    "Element",
    "Image",
    "Canvas",
    "Palette",
    "BOMEntry",
    "CellData",
]
