"""Image loading utilities.

This module provides convenient functions for loading images from
various sources (file paths, URLs).
"""

from ..models import Image


def load_image(source: str) -> Image:
    """Load an image from a file path or URL.

    Automatically detects whether the source is a URL or file path
    and uses the appropriate loading method.

    Args:
        source (str): Either a local file path or a URL starting with
            ``'http://'`` or ``'https://'``.

    Returns:
        Image: An Image instance containing the loaded image data.

    Raises:
        FileNotFoundError: If a file path is given but file doesn't exist.
        ValueError: If the source cannot be loaded as an image.

    Example:
        >>> image = load_image("photo.jpg")
        >>> image = load_image("https://example.com/image.png")
    """
    if source.startswith("http://") or source.startswith("https://"):
        return Image.from_url(source)
    else:
        return Image.from_file(source)
