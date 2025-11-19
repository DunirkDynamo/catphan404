# -----------------------------
# File: catphan404/slice_thickness.py
# -----------------------------
"""Slice thickness measurement module for Catphan 404.

This implementation uses a simple technique: it finds a bright ramp/profile (often
visible in Catphan slice-thickness module), extracts a profile perpendicular to
that feature, and estimates FWHM as a proxy for slice thickness.
"""
from typing import Dict, Tuple
import numpy as np
from scipy import ndimage
from .utils.image import find_edge_region


def slice_thickness_analysis(image: np.ndarray) -> Dict:
    """Estimate slice thickness using FWHM on a bright ramp/profile.

    Args:
        image (np.ndarray): Input image.

    Returns:
        Dict: Estimated FWHM in pixels and (if spacing known) in mm.
    """
    y0, y1, x0, x1 = find_edge_region(image, width=60)
    roi = image[y0:y1, x0:x1]
    # project along columns to get a profile across the ramp
    profile = np.mean(roi, axis=1)
    # normalize
    p = (profile - profile.min()) / (profile.max() - profile.min() + 1e-12)
    # find fwhm
    half_max = 0.5
    indices = np.where(p >= half_max)[0]
    if indices.size == 0:
        fwhm = float('nan')
    else:
        fwhm = float(indices.max() - indices.min())
    return {'fwhm_pixels': fwhm}