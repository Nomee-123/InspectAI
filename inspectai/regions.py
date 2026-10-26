"""Region helpers: mask cleanup and connected-component localization."""

from __future__ import annotations

import cv2
import numpy as np


def adaptive_minimum_area(valid_pixels: int) -> int:
    """Scale the smallest reportable region with the number of valid pixels.

    The rule is deliberately conservative: it grows slowly so that a small,
    high-contrast change on a large image is still reportable, while
    single-pixel compression speckle is not.
    """
    if valid_pixels <= 0:
        return 16
    return max(16, int(round(valid_pixels * 0.00002)))


def cleanup_mask(mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Apply a light open/close pass and return the cleaned mask and rule text."""
    if mask.dtype != np.uint8:
        raise ValueError("Difference mask must use uint8 binary values of 0 or 255.")
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cleaned, "3x3 open, then 3x3 close"


def detect_changed_regions(
    difference_mask: np.ndarray,
    minimum_area: int = 25,
) -> dict[str, object]:
    """Find 8-connected changed pixel regions in a binary uint8 mask.

    Regions smaller than ``minimum_area`` pixels are excluded from ``regions``
    and counted in ``filtered_region_count``.
    """
    if not isinstance(difference_mask, np.ndarray) or difference_mask.ndim != 2:
        raise ValueError("Difference mask must be a two-dimensional NumPy array.")
    if difference_mask.dtype != np.uint8:
        raise ValueError("Difference mask must use uint8 binary values of 0 or 255.")
    if (
        isinstance(minimum_area, (bool, np.bool_))
        or not isinstance(minimum_area, (int, np.integer))
        or minimum_area < 1
    ):
        raise ValueError("Minimum region area must be a positive whole number of pixels.")
    if difference_mask.size and np.any((difference_mask != 0) & (difference_mask != 255)):
        raise ValueError("Difference mask must use uint8 binary values of 0 or 255.")

    minimum_area = int(minimum_area)
    if difference_mask.size == 0:
        detected_count = 0
        regions: list[dict[str, int]] = []
    else:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            difference_mask, connectivity=8
        )
        detected_count = component_count - 1
        regions = []
        for component_index in range(1, component_count):
            x, y, width, height, area = (
                int(value) for value in stats[component_index]
            )
            if area < minimum_area:
                continue
            regions.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area_pixels": area,
                }
            )

    regions.sort(key=lambda region: (-region["area_pixels"], region["y"], region["x"]))
    return {
        "regions": regions,
        "region_count": len(regions),
        "detected_region_count": detected_count,
        "filtered_region_count": detected_count - len(regions),
        "minimum_area_pixels": minimum_area,
    }
