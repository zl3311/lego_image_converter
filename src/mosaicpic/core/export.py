"""Export functions for BrickLink and Rebrickable formats.

This module provides functions to export a Bill of Materials (BOM) to formats
compatible with external tile marketplaces and inventory management systems.

Supported Formats:
    - BrickLink XML: For uploading wanted lists to BrickLink.com
    - Rebrickable CSV: For importing parts lists to Rebrickable.com

Example:
    >>> from mosaicpic import ConversionSession, Palette, load_image
    >>> from mosaicpic.core.export import export_bricklink_xml, export_rebrickable_csv
    >>>
    >>> # After conversion...
    >>> bom = session.get_bill_of_materials()
    >>> xml_content = export_bricklink_xml(bom)
    >>> csv_content = export_rebrickable_csv(bom)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import BOMEntry

from ..data.loader import get_color_external_ids


def export_bricklink_xml(bom: "list[BOMEntry]") -> str:
    """Export BOM to BrickLink XML wanted list format.

    BrickLink XML format uses design_id (ITEMID) and BrickLink color ID (COLOR)
    to identify parts. This is the standard format for uploading wanted lists
    to BrickLink.com.

    Args:
        bom (list[BOMEntry]): Bill of materials from ConversionSession.

    Returns:
        str: XML string compatible with BrickLink wanted list upload.

    Raises:
        ValueError: If any color in the BOM has no name assigned.

    Example:
        >>> bom = session.get_bill_of_materials()
        >>> xml = export_bricklink_xml(bom)
        >>> print(xml)
        <INVENTORY>
          <ITEM>
            <ITEMTYPE>P</ITEMTYPE>
            <ITEMID>98138</ITEMID>
            <COLOR>11</COLOR>
            <MINQTY>655</MINQTY>
          </ITEM>
          ...
        </INVENTORY>

    Note:
        - ITEMTYPE is always "P" (Part) for mosaic tiles.
        - ITEMID is the design_id (98138 for 1x1 round tiles).
        - COLOR is the BrickLink-specific color ID.
        - MINQTY is the quantity needed.
    """
    lines = ["<INVENTORY>"]

    for entry in bom:
        color_name = entry.color.name
        if color_name is None:
            raise ValueError(f"Color {entry.color.rgb} has no name. Cannot export to BrickLink.")

        ids = get_color_external_ids(color_name)

        lines.append("  <ITEM>")
        lines.append("    <ITEMTYPE>P</ITEMTYPE>")
        lines.append(f"    <ITEMID>{ids['design_id']}</ITEMID>")
        lines.append(f"    <COLOR>{ids['bl_id']}</COLOR>")
        lines.append(f"    <MINQTY>{entry.count_needed}</MINQTY>")
        lines.append("  </ITEM>")

    lines.append("</INVENTORY>")

    return "\n".join(lines)


def export_rebrickable_csv(bom: "list[BOMEntry]") -> str:
    """Export BOM to Rebrickable CSV format.

    Rebrickable CSV uses Part (design_id), Color (Rebrickable color ID), and
    Quantity columns. This format is compatible with Rebrickable.com's parts
    list import feature.

    Args:
        bom (list[BOMEntry]): Bill of materials from ConversionSession.

    Returns:
        str: CSV string compatible with Rebrickable parts list import.

    Raises:
        ValueError: If any color in the BOM has no name assigned.

    Example:
        >>> bom = session.get_bill_of_materials()
        >>> csv = export_rebrickable_csv(bom)
        >>> print(csv)
        Part,Color,Quantity
        98138,0,655
        98138,15,598
        ...

    Note:
        - Part is the design_id (98138 for 1x1 round tiles).
        - Color is the Rebrickable-specific color ID.
        - Quantity is the count needed.
    """
    lines = ["Part,Color,Quantity"]

    for entry in bom:
        color_name = entry.color.name
        if color_name is None:
            raise ValueError(f"Color {entry.color.rgb} has no name. Cannot export to Rebrickable.")

        ids = get_color_external_ids(color_name)
        lines.append(f"{ids['design_id']},{ids['rb_id']},{entry.count_needed}")

    return "\n".join(lines)
