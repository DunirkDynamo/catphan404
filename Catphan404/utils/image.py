# -----------------------------
# File: catphan404/utils/image.py
# -----------------------------
"""Image utilities for CTP515 low-contrast module.

Contains helpers for blob detection and ROI masking.
"""
from typing import Tuple
import numpy as np
from skimage import feature


def circular_roi_mask(shape: Tuple[int, int], center: Tuple[float, float], radius: float) -> np.ndarray:
    """Return a boolean mask with a circle filled in.

    Args:
        shape (Tuple[int, int]): Image shape as (rows, cols).
        center (Tuple[float, float]): (x, y) center coordinates.
        radius (float): Radius in pixels.

    Returns:
        np.ndarray: Boolean mask.
    """
    ny, nx = shape
    Y, X = np.ogrid[:ny, :nx]
    cx, cy = center
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return dist <= radius


