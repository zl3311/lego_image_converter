"""Visualization utilities for testing and debugging.

This module provides plotting and rendering functions for visualizing images
and canvases during test development and debugging. Not part of the public API.

Rendering Functions:
    render_canvas_png: Render canvas as PNG with round tile style.
    render_canvas_svg: Render canvas as SVG with round tile style.
    save_canvas_png: Render and save a Canvas as PNG.
    save_canvas_svg: Render and save a Canvas as SVG.

Matplotlib Plotting Functions (for interactive use):
    plot_image: Plot an Image using matplotlib.
    plot_canvas: Plot a Canvas using matplotlib.
    plot_comparison: Plot original and result side by side.
"""

from typing import TYPE_CHECKING, Optional

from PIL import Image as PILImage
from PIL import ImageDraw

if TYPE_CHECKING:
    from mosaicpic import Canvas, Color, Image


# =============================================================================
# Tile-Style Rendering Functions (PNG and SVG)
# =============================================================================

# Default proportions (visible gap for round tiles)
DEFAULT_TILE_RATIO = 0.85  # Tile diameter as ratio of cell size
DEFAULT_OUTLINE_RATIO = 0.025  # Outline width as ratio of tile diameter


def render_canvas_png(
    canvas: "Canvas",
    cell_size: int = 20,
    background_color: Optional["Color"] = None,
    tile_ratio: float = DEFAULT_TILE_RATIO,
    show_outline: bool = True,
    outline_color: Optional["Color"] = None,
    outline_ratio: float = DEFAULT_OUTLINE_RATIO,
) -> PILImage.Image:
    """Render a Canvas as a PNG image with round tile style.

    Creates a mosaic representation with circular tiles on a
    baseplate-style background, with optional stud outlines.

    Args:
        canvas (Canvas): The Canvas to render.
        cell_size (int): Size of each cell in pixels. The tile diameter will
            be cell_size * tile_ratio. Default 20px.
        background_color (Color | None): Color for the baseplate background.
            If None, defaults to black RGB(33, 33, 33).
        tile_ratio (float): Tile diameter as ratio of cell size. Default 0.85
            (85% tile, 15% gap - visible gap for round tiles).
        show_outline (bool): Whether to draw stud outlines on tiles.
            Default True.
        outline_color (Color | None): Color for the stud outline. If None,
            uses a slightly darker/lighter shade of the tile color.
        outline_ratio (float): Outline stroke width as ratio of tile diameter.
            Default 0.025 (2.5% of tile size).

    Returns:
        PILImage.Image: PIL Image object that can be saved or displayed.

    Example:
        >>> canvas = convert("photo.jpg", Palette.from_set("marilyn_48x48"), (48, 48))
        >>> img = render_canvas_png(canvas, cell_size=20)
        >>> img.save("output.png")
    """
    from mosaicpic import Color

    # Default background: black
    if background_color is None:
        background_color = Color((33, 33, 33), name="Black")

    # Calculate dimensions
    img_width = canvas.width * cell_size
    img_height = canvas.height * cell_size
    tile_diameter = int(cell_size * tile_ratio)
    tile_radius = tile_diameter / 2
    outline_width = max(1, int(tile_diameter * outline_ratio))

    # Create image with background
    img = PILImage.new("RGB", (img_width, img_height), background_color.rgb)
    draw = ImageDraw.Draw(img)

    # Draw tiles
    for y in range(canvas.height):
        for x in range(canvas.width):
            cell = canvas.get_cell(x, y)
            if cell.color is None:
                continue

            # Calculate circle center and bounding box
            cx = x * cell_size + cell_size / 2
            cy = y * cell_size + cell_size / 2

            # Bounding box for ellipse
            x0 = cx - tile_radius
            y0 = cy - tile_radius
            x1 = cx + tile_radius
            y1 = cy + tile_radius

            # Draw filled circle
            draw.ellipse([x0, y0, x1, y1], fill=cell.color.rgb)

            # Draw outline if enabled
            if show_outline:
                if outline_color is not None:
                    stroke_color = outline_color.rgb
                else:
                    # Generate subtle darker outline
                    stroke_color = tuple(max(0, c - 30) for c in cell.color.rgb)
                draw.ellipse([x0, y0, x1, y1], outline=stroke_color, width=outline_width)

    return img


def render_canvas_svg(
    canvas: "Canvas",
    cell_size: float = 20.0,
    background_color: Optional["Color"] = None,
    tile_ratio: float = DEFAULT_TILE_RATIO,
    show_outline: bool = True,
    outline_color: Optional["Color"] = None,
    outline_ratio: float = DEFAULT_OUTLINE_RATIO,
) -> str:
    """Render a Canvas as an SVG string with round tile style.

    Creates a vector SVG representation with circular tiles on a baseplate
    background. SVG output is scalable and ideal for high-quality prints.

    Args:
        canvas (Canvas): The Canvas to render.
        cell_size (float): Size of each cell in SVG units. Default 20.
        background_color (Color | None): Color for the baseplate background.
            If None, defaults to black RGB(33, 33, 33).
        tile_ratio (float): Tile diameter as ratio of cell size. Default 0.85.
        show_outline (bool): Whether to draw stud outlines on tiles.
            Default True.
        outline_color (Color | None): Color for the stud outline. If None,
            uses a slightly darker shade of the tile color.
        outline_ratio (float): Outline stroke width as ratio of tile diameter.
            Default 0.025.

    Returns:
        str: SVG content as a string. Can be saved to a .svg file.

    Example:
        >>> canvas = convert("photo.jpg", Palette.from_set("marilyn_48x48"), (48, 48))
        >>> svg_content = render_canvas_svg(canvas, cell_size=20)
        >>> with open("output.svg", "w") as f:
        ...     f.write(svg_content)
    """
    import svgwrite

    from mosaicpic import Color

    # Default background: black
    if background_color is None:
        background_color = Color((33, 33, 33), name="Black")

    # Calculate dimensions
    img_width = canvas.width * cell_size
    img_height = canvas.height * cell_size
    tile_radius = (cell_size * tile_ratio) / 2
    outline_width = tile_radius * 2 * outline_ratio

    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(f"{img_width}px", f"{img_height}px"))
    dwg.viewbox(0, 0, img_width, img_height)

    # Background rectangle
    bg_color = _rgb_to_hex(background_color.rgb)
    dwg.add(dwg.rect(insert=(0, 0), size=(img_width, img_height), fill=bg_color))

    # Draw tiles
    for y in range(canvas.height):
        for x in range(canvas.width):
            cell = canvas.get_cell(x, y)
            if cell.color is None:
                continue

            # Calculate circle center
            cx = x * cell_size + cell_size / 2
            cy = y * cell_size + cell_size / 2

            # Tile color
            fill_color = _rgb_to_hex(cell.color.rgb)

            # Outline settings
            if show_outline:
                if outline_color is not None:
                    stroke_color = _rgb_to_hex(outline_color.rgb)
                else:
                    # Subtle darker outline
                    stroke_color = _rgb_to_hex(tuple(max(0, c - 30) for c in cell.color.rgb))
                stroke_width = outline_width
            else:
                stroke_color = "none"
                stroke_width = 0

            # Add circle
            dwg.add(
                dwg.circle(
                    center=(cx, cy),
                    r=tile_radius,
                    fill=fill_color,
                    stroke=stroke_color,
                    stroke_width=stroke_width,
                )
            )

    return dwg.tostring()


def save_canvas_png(canvas: "Canvas", path: str, **kwargs) -> None:
    """Render and save a Canvas as PNG.

    Convenience function that calls render_canvas_png and saves the result.

    Args:
        canvas (Canvas): The Canvas to render.
        path (str): Output file path.
        **kwargs: Additional arguments passed to render_canvas_png.
    """
    img = render_canvas_png(canvas, **kwargs)
    img.save(path)


def save_canvas_svg(canvas: "Canvas", path: str, **kwargs) -> None:
    """Render and save a Canvas as SVG.

    Convenience function that calls render_canvas_svg and saves the result.

    Args:
        canvas (Canvas): The Canvas to render.
        path (str): Output file path.
        **kwargs: Additional arguments passed to render_canvas_svg.
    """
    svg_content = render_canvas_svg(canvas, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_content)


def _rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color string.

    Args:
        rgb (tuple): RGB tuple of integers (r, g, b).

    Returns:
        str: Hex color string (e.g., "#ff0000").
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# =============================================================================
# Matplotlib Plotting Functions (for interactive use)
# =============================================================================


def plot_image(
    image: "Image",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 100,
) -> None:
    """Plot an Image for visual inspection.

    Args:
        image (Image): The Image to plot.
        title (str | None): Optional title for the plot.
        save_path (str | None): If provided, save the figure to this path.
        show (bool): Whether to display the plot (set False for headless).
        figsize (tuple[float, float]): Figure size in inches (width, height).
        dpi (int): Resolution for saved figures.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image.to_array())
    ax.axis("off")

    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_canvas(
    canvas: "Canvas",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 100,
    show_grid: bool = True,
    stud_style: bool = True,
) -> None:
    """Plot a Canvas for visual inspection.

    Args:
        canvas (Canvas): The Canvas to plot.
        title (str | None): Optional title for the plot.
        save_path (str | None): If provided, save the figure to this path.
        show (bool): Whether to display the plot (set False for headless).
        figsize (tuple[float, float]): Figure size in inches (width, height).
        dpi (int): Resolution for saved figures.
        show_grid (bool): Whether to draw grid lines between cells.
        stud_style (bool): If True, draw circular studs; if False, draw squares.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    w, h = canvas.width, canvas.height

    if stud_style:
        # Draw as round studs (circles)
        ax.set_facecolor("black")

        if show_grid:
            ax.vlines(x=list(range(w + 1)), ymin=0, ymax=h, colors="gray", linewidths=0.5)
            ax.hlines(y=list(range(h + 1)), xmin=0, xmax=w, colors="gray", linewidths=0.5)

        for y in range(h):
            for x in range(w):
                cell = canvas.get_cell(x, y)
                if cell.color is not None:
                    color = tuple(c / 255 for c in cell.color.rgb)
                    circle = plt.Circle(
                        (x + 0.5, h - y - 0.5),  # Flip y for display
                        0.35,
                        color=color,
                    )
                    ax.add_patch(circle)

        ax.set_xlim([0, w])
        ax.set_ylim([0, h])
        ax.set_aspect("equal")
    else:
        # Draw as simple image
        ax.imshow(canvas.to_array())

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison(
    original: "Image",
    result: "Canvas",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (16, 8),
    dpi: int = 100,
) -> None:
    """Plot original image and converted canvas side by side.

    Args:
        original (Image): The original Image.
        result (Canvas): The converted Canvas.
        title (str | None): Optional title for the entire figure.
        save_path (str | None): If provided, save the figure to this path.
        show (bool): Whether to display the plot.
        figsize (tuple[float, float]): Figure size in inches (width, height).
        dpi (int): Resolution for saved figures.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original image
    axes[0].imshow(original.to_array())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Converted canvas
    axes[1].imshow(result.to_array())
    axes[1].set_title(f"Canvas ({result.width}×{result.height})")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
