# -----------------------------
# File: catphan404/high_contrast.py
# -----------------------------
"""High-contrast / line-pair / MTF analysis for Catphan 404."""
from typing import Tuple, Dict
import numpy as np
from .utils.image import find_edge_region
from .utils.math import compute_mtf_from_edge


def high_contrast_analysis(image: np.ndarray) -> Dict:
    """Perform MTF estimation from a prominent edge in the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        Dict: MTF data and estimated f50 (frequency at 50% MTF).
    """
    y0, y1, x0, x1 = find_edge_region(image, width=40)
    roi = image[y0:y1, x0:x1]
    freqs, mtf = compute_mtf_from_edge(roi)
    # estimate f50
    try:
        f50 = np.interp(0.5, mtf[::-1], freqs[::-1])
    except Exception:
        f50 = float('nan')
    return {'freqs': freqs.tolist(), 'mtf': mtf.tolist(), 'f50': float(f50)}

