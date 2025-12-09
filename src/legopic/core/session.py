"""Conversion session for managing the full image-to-Lego workflow.

The ConversionSession is the main API for the legopic package. It owns
the relationship between an image, palette, and canvas, handling conversion,
re-conversion with preserved pins, color adjustments, and data exports.

Typical Workflow:
    1. Create session with hard params (image, palette, canvas_size).
    2. Call convert() with soft params (method, limit_inventory, etc.).
    3. Adjust: pin cells, swap colors.
    4. Optionally reconvert() with different soft params.
    5. Export: get_bill_of_materials(), get_grid_data(), etc.
"""

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ..models import BOMEntry, Canvas, CellData, Color, Image, Palette
from .config import ConvertConfig
from .downsize import _get_block_pixels, _validate_dimensions


class ConversionSession:
    """Manages the full image-to-Lego conversion workflow.

    A session owns the relationship between an image, palette, and canvas.
    It handles conversion, re-conversion with preserved pins, and exports.

    The session separates "hard" parameters (set at init, immutable) from
    "soft" parameters (in ConvertConfig, can change between reconversions).

    Hard parameters (immutable after init):
        - image: Source image
        - palette: Available colors/elements
        - canvas_size: Output dimensions in studs

    Soft parameters (in ConvertConfig):
        - method: Downsampling algorithm
        - limit_inventory: Whether to respect element counts
        - algorithm: Assignment algorithm for inventory-limited mode

    Attributes:
        image: Source image (read-only).
        palette: Available colors and elements (read-only).
        canvas_size: Target (width, height) in studs (read-only).
        canvas: Current conversion result (updated by convert/reconvert).
        config: Current conversion configuration.
        similarity_score: Aggregate Delta E across all cells.

    Example:
        >>> from legopic import ConversionSession, ConvertConfig, Palette, load_image
        >>>
        >>> # Setup (hard params)
        >>> image = load_image("photo.jpg")
        >>> palette = Palette.from_set(31197)
        >>> session = ConversionSession(image, palette, (48, 48))
        >>>
        >>> # Convert (soft params)
        >>> config = ConvertConfig(method='match_then_mode', limit_inventory=True)
        >>> session.convert(config)
        >>> print(f"Similarity: {session.similarity_score:.2f}")
        >>>
        >>> # Adjust
        >>> session.pin(3, 5, some_blue_color)
        >>> session.swap_color(old_red, new_orange)
        >>>
        >>> # Re-convert with different method, keep pins
        >>> session.reconvert(ConvertConfig(method='mean_then_match'), keep_pins=True)
        >>>
        >>> # Export for building guide
        >>> bom = session.get_bill_of_materials()
        >>> grid = session.get_grid_data()
    """

    def __init__(self, image: Image, palette: Palette, canvas_size: tuple[int, int]):
        """Initialize a conversion session.

        Args:
            image (Image): Source image to convert.
            palette (Palette): Available colors and elements.
            canvas_size (tuple[int, int]): Target (width, height) in studs.

        Raises:
            ValueError: If canvas_size dimensions are invalid or incompatible
                with image dimensions for uniform stride downsampling.
        """
        self._image = image
        self._palette = palette
        self._canvas_size = canvas_size

        self._canvas: Canvas | None = None
        self._config: ConvertConfig | None = None
        self._pinned_index: set[tuple[int, int]] = set()

        # Validate dimensions and pre-compute target colors
        # (downsampled original for delta_e calculation)
        self._target_colors = self._compute_target_colors()

    @property
    def image(self) -> Image:
        """Source image (read-only)."""
        return self._image

    @property
    def palette(self) -> Palette:
        """Available colors and elements (read-only)."""
        return self._palette

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Target canvas dimensions (width, height) in studs (read-only)."""
        return self._canvas_size

    @property
    def canvas(self) -> Canvas:
        """Current canvas result.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        if self._canvas is None:
            raise RuntimeError("No conversion yet. Call convert() first.")
        return self._canvas

    @property
    def config(self) -> ConvertConfig | None:
        """Current conversion configuration, or None if not yet converted."""
        return self._config

    @property
    def similarity_score(self) -> float:
        """Aggregate similarity score (average Delta E across all cells).

        Lower values indicate better overall color matching.
        Typical range is 0-100, with <10 being excellent, <20 good.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        canvas = self.canvas  # Raises if not converted
        total = sum(
            canvas.cells[y][x].delta_e for y in range(canvas.height) for x in range(canvas.width)
        )
        return total / (canvas.width * canvas.height)

    def _compute_target_colors(self) -> "NDArray[np.uint8]":
        """Compute the target color for each canvas cell.

        The target color is the mean RGB of all pixels in the block
        corresponding to each canvas cell. Used for delta_e calculation.

        Returns:
            NDArray[np.uint8]: Array of shape (height, width, 3) with mean
                colors per cell.
        """
        width, height = self._canvas_size
        stride = _validate_dimensions(self._image, width, height)
        image_array = self._image.to_array()

        target_colors = np.zeros((height, width, 3), dtype=np.uint8)
        for cy in range(height):
            for cx in range(width):
                block_pixels = _get_block_pixels(image_array, cx, cy, stride)
                target_colors[cy, cx] = np.mean(block_pixels, axis=0).astype(np.uint8)

        return target_colors

    def convert(self, config: ConvertConfig | None = None) -> Canvas:
        """Run initial conversion.

        This clears any existing pins and runs a fresh conversion.

        Args:
            config (ConvertConfig | None): Conversion configuration. Uses
                defaults if None.

        Returns:
            Canvas: The converted Canvas.
        """
        self._config = config or ConvertConfig()
        self._pinned_index.clear()

        self._canvas = self._run_conversion(pinned_cells=None)
        self._compute_all_delta_e()

        return self._canvas

    def reconvert(self, config: ConvertConfig | None = None, keep_pins: bool = True) -> Canvas:
        """Re-run conversion, optionally preserving pinned cells.

        Pinned cells are treated as fixed constraints: their colors are
        reserved from inventory before running the assignment algorithm.

        Args:
            config (ConvertConfig | None): New configuration. If None, uses
                current config.
            keep_pins (bool): If True, pinned cells preserve their colors and
                are treated as constraints during assignment.

        Returns:
            Canvas: The newly converted Canvas.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        if self._canvas is None:
            raise RuntimeError("No initial conversion. Call convert() first.")

        if config is not None:
            self._config = config

        pinned_cells = None
        if keep_pins and self._pinned_index:
            pinned_cells = {(x, y): self._canvas.cells[y][x].color for x, y in self._pinned_index}

        self._canvas = self._run_conversion(pinned_cells=pinned_cells)

        # Restore pinned flags
        if keep_pins:
            for x, y in self._pinned_index:
                self._canvas.cells[y][x].pinned = True
        else:
            self._pinned_index.clear()

        self._compute_all_delta_e()
        return self._canvas

    def _run_conversion(self, pinned_cells: dict[tuple[int, int], Color] | None) -> Canvas:
        """Run the conversion algorithm.

        Routes to either unlimited conversion (standard downsize) or
        inventory-limited conversion (assignment algorithms).

        Args:
            pinned_cells: Dict of (x, y) -> Color for cells to preserve
                as fixed constraints, or None.

        Returns:
            Canvas: Converted Canvas.
        """
        assert self._config is not None
        if self._config.limit_inventory:
            return self._run_inventory_limited_conversion(pinned_cells)
        return self._run_unlimited_conversion(pinned_cells)

    def _run_unlimited_conversion(
        self, pinned_cells: dict[tuple[int, int], Color] | None
    ) -> Canvas:
        """Run conversion without inventory limits.

        Uses the standard downsize algorithm that picks the best matching
        color for each cell regardless of inventory.

        Args:
            pinned_cells: Dict of (x, y) -> Color for cells to preserve,
                or None.

        Returns:
            Canvas: Converted Canvas.
        """
        from .downsize import downsize

        assert self._config is not None
        width, height = self._canvas_size
        canvas = downsize(self._image, self._palette, width, height, method=self._config.method)

        # Apply pinned cells (overwrite algorithm's choices)
        if pinned_cells:
            for (x, y), color in pinned_cells.items():
                canvas.cells[y][x].color = color
                canvas.cells[y][x].pinned = True

        return canvas

    def _run_inventory_limited_conversion(
        self, pinned_cells: dict[tuple[int, int], Color] | None
    ) -> Canvas:
        """Run conversion with inventory limits.

        Uses the configured assignment algorithm to assign colors while
        respecting palette element counts. Cells that can't get their
        preferred color fall back to the next best available.

        Args:
            pinned_cells: Dict of (x, y) -> Color for cells to preserve,
                or None.

        Returns:
            Canvas: Converted Canvas.
        """
        from .assignment import optimal, priority_greedy

        assert self._config is not None
        width, height = self._canvas_size
        assign_func = optimal if self._config.algorithm == "optimal" else priority_greedy
        result = assign_func(
            target_colors=self._target_colors,
            palette=self._palette,
            pinned_cells=pinned_cells,
            canvas_width=width,
            canvas_height=height,
        )

        # Build canvas from assignments
        canvas = Canvas(width, height)
        palette_colors = self._palette.colors

        for y in range(height):
            for x in range(width):
                palette_idx = result.assignments[y, x]

                if palette_idx == -2:
                    # Out-of-palette pinned cell
                    assert pinned_cells is not None  # Required for palette_idx == -2
                    color = pinned_cells[(x, y)]
                    canvas.cells[y][x].color = color
                    canvas.cells[y][x].pinned = True
                elif palette_idx >= 0:
                    color = palette_colors[palette_idx]
                    canvas.cells[y][x].color = color

                    # Mark pinned if applicable
                    if pinned_cells and (x, y) in pinned_cells:
                        canvas.cells[y][x].pinned = True

        return canvas

    def _compute_all_delta_e(self) -> None:
        """Compute Delta E for all cells against original image colors.

        Uses pairwise comparison (each cell vs its corresponding target)
        via get_deltas_e, which is O(n) rather than O(n²).
        """
        from basic_colormath import get_deltas_e

        assert self._canvas is not None  # Set by convert() before calling this method
        canvas = self._canvas

        # Build array of canvas colors (assigned palette colors)
        canvas_colors = np.array(
            [
                canvas.cells[y][x].color.rgb
                for y in range(canvas.height)
                for x in range(canvas.width)
            ],
            dtype=np.uint8,
        )

        target_colors = self._target_colors.reshape(-1, 3)

        # Compute delta E pairwise: each canvas cell vs its corresponding target.
        # get_deltas_e(A, B) returns array of distances: A[i] vs B[i] for all i.
        delta_e_values = get_deltas_e(canvas_colors, target_colors)

        # Assign to cells
        idx = 0
        for y in range(canvas.height):
            for x in range(canvas.width):
                canvas.cells[y][x].delta_e = float(delta_e_values[idx])
                idx += 1

    def _compute_delta_e_for_cell(self, x: int, y: int) -> float:
        """Compute Delta E for a single cell.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.

        Returns:
            float: Delta E value for the cell.
        """
        from basic_colormath import get_delta_e_matrix

        assert self._canvas is not None  # Method only called after convert()
        cell_rgb = np.array([self._canvas.cells[y][x].color.rgb], dtype=np.uint8)
        target_rgb = self._target_colors[y, x].reshape(1, 3)
        # get_delta_e_matrix returns shape (1, 1) for single comparison
        return float(get_delta_e_matrix(cell_rgb, target_rgb)[0, 0])

    def pin(self, x: int, y: int, new_color: Color | None = None) -> None:
        """Pin a cell, optionally changing its color.

        Pinned cells are preserved during reconvert(keep_pins=True).

        Args:
            x (int): X-coordinate (column).
            y (int): Y-coordinate (row).
            new_color (Color | None): If provided, change the cell to this
                color. Can be any Color, including out-of-palette colors.

        Raises:
            RuntimeError: If convert() has not been called yet.
            IndexError: If coordinates are out of bounds.
        """
        canvas = self.canvas  # Raises if not converted

        if not (0 <= x < canvas.width and 0 <= y < canvas.height):
            raise IndexError(
                f"Coordinates ({x}, {y}) out of bounds for "
                f"canvas of size ({canvas.width}, {canvas.height})."
            )

        cell = canvas.cells[y][x]
        cell.pinned = True
        self._pinned_index.add((x, y))

        if new_color is not None:
            cell.color = new_color
            cell.delta_e = self._compute_delta_e_for_cell(x, y)

    def unpin(self, x: int, y: int) -> None:
        """Unpin a cell.

        Args:
            x (int): X-coordinate (column).
            y (int): Y-coordinate (row).

        Raises:
            RuntimeError: If convert() has not been called yet.
            IndexError: If coordinates are out of bounds.
        """
        canvas = self.canvas

        if not (0 <= x < canvas.width and 0 <= y < canvas.height):
            raise IndexError(
                f"Coordinates ({x}, {y}) out of bounds for "
                f"canvas of size ({canvas.width}, {canvas.height})."
            )

        canvas.cells[y][x].pinned = False
        self._pinned_index.discard((x, y))

    def swap_color(self, old_color: Color, new_color: Color, pin: bool = True) -> int:
        """Bulk swap all cells of one color to another.

        This is useful for replacing a color throughout the canvas,
        e.g., changing all dark red tiles to bright red.

        Args:
            old_color (Color): Color to replace (matched by RGB).
            new_color (Color): Replacement color.
            pin (bool): If True, pin all affected cells so they survive
                reconvert.

        Returns:
            int: Number of cells changed.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        canvas = self.canvas
        count = 0

        for y in range(canvas.height):
            for x in range(canvas.width):
                cell = canvas.cells[y][x]
                if cell.color == old_color:
                    cell.color = new_color
                    cell.delta_e = self._compute_delta_e_for_cell(x, y)
                    if pin:
                        cell.pinned = True
                        self._pinned_index.add((x, y))
                    count += 1

        return count

    def get_pinned_cells(self) -> list[tuple[int, int]]:
        """Get coordinates of all pinned cells.

        Returns:
            list[tuple[int, int]]: List of (x, y) tuples for pinned cells.
        """
        return list(self._pinned_index)

    def get_bill_of_materials(self) -> list[BOMEntry]:
        """Generate bill of materials for building guide.

        Returns a list of colors used and how many tiles are needed,
        sorted by count (most needed first).

        Returns:
            list[BOMEntry]: List of BOMEntry, one per unique color used.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        canvas = self.canvas
        color_counts: Counter[Color] = Counter()

        for y in range(canvas.height):
            for x in range(canvas.width):
                color_counts[canvas.cells[y][x].color] += 1

        result = []
        for color, count in color_counts.items():
            in_palette = color in self._palette
            elements = self._palette.get_elements_for_color(color) if in_palette else []
            result.append(
                BOMEntry(color=color, count_needed=count, in_palette=in_palette, elements=elements)
            )

        return sorted(result, key=lambda e: e.count_needed, reverse=True)

    def get_grid_data(self) -> list[list[CellData]]:
        """Get grid data for visual building guide.

        Returns a 2D structure with all cell information needed
        for rendering a visual guide.

        Returns:
            list[list[CellData]]: 2D list of CellData, indexed as [y][x].

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        canvas = self.canvas
        return [
            [
                CellData(
                    x=x,
                    y=y,
                    color=canvas.cells[y][x].color,
                    delta_e=canvas.cells[y][x].delta_e,
                    pinned=canvas.cells[y][x].pinned,
                )
                for x in range(canvas.width)
            ]
            for y in range(canvas.height)
        ]

    def get_similarity_map(self) -> list[list[float]]:
        """Get per-cell similarity scores (Delta E).

        Useful for identifying problem areas where color matching
        is poor.

        Returns:
            list[list[float]]: 2D list of Delta E values, indexed as [y][x].
                Lower values indicate better match to original image.

        Raises:
            RuntimeError: If convert() has not been called yet.
        """
        canvas = self.canvas
        return [
            [canvas.cells[y][x].delta_e for x in range(canvas.width)] for y in range(canvas.height)
        ]

    def export_bricklink_xml(self) -> str:
        """Export canvas BOM to BrickLink XML wanted list format.

        Returns a string containing BrickLink-compatible XML that can be
        uploaded directly to BrickLink.com as a wanted list.

        Returns:
            str: XML string for BrickLink wanted list upload.

        Raises:
            RuntimeError: If convert() has not been called yet.

        Example:
            >>> xml = session.export_bricklink_xml()
            >>> with open("wanted_list.xml", "w") as f:
            ...     f.write(xml)
        """
        from .export import export_bricklink_xml

        bom = self.get_bill_of_materials()
        return export_bricklink_xml(bom)

    def export_rebrickable_csv(self) -> str:
        """Export canvas BOM to Rebrickable CSV format.

        Returns a string containing Rebrickable-compatible CSV that can be
        imported to Rebrickable.com as a parts list.

        Returns:
            str: CSV string for Rebrickable parts list import.

        Raises:
            RuntimeError: If convert() has not been called yet.

        Example:
            >>> csv = session.export_rebrickable_csv()
            >>> with open("parts_list.csv", "w") as f:
            ...     f.write(csv)
        """
        from .export import export_rebrickable_csv

        bom = self.get_bill_of_materials()
        return export_rebrickable_csv(bom)

    def __repr__(self) -> str:
        """Return string representation of the session."""
        status = "converted" if self._canvas else "not converted"
        pins = len(self._pinned_index)
        return (
            f"ConversionSession(canvas_size={self._canvas_size}, "
            f"palette={len(self._palette)} colors, status={status}, pins={pins})"
        )
