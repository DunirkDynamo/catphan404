# -----------------------------
# File: catphan404/geometry.py
# -----------------------------
"""Geometric accuracy module for Catphan 404.

Detects fiducial blobs and measures pairwise distances against expected values.
"""
from typing import Dict, Tuple
import numpy as np
from .utils.image import detect_blobs


def geometry_analysis(image: np.ndarray, expected_distance_mm: float = 50.0, spacing_mm: float = 1.0) -> Dict:
    """Detect fiducial markers and compute distance errors.

    Args:
        image (np.ndarray): Input image.
        expected_distance_mm (float): Known reference distance between markers in mm.
        spacing_mm (float): Pixel spacing in mm.

    Returns:
        Dict: Detected marker positions and measured distances (mm).
    """
    blobs = detect_blobs(image, min_sigma=2, max_sigma=8, threshold=np.std(image) * 0.4)
    centers = [(int(b[1]), int(b[0])) for b in blobs]
    results = {'n_markers': len(centers), 'markers': centers, 'pairs': []}
    # compute pairwise distances
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]
            dist_px = np.hypot(x2 - x1, y2 - y1)
            dist_mm = dist_px * spacing_mm
            pct_err = (dist_mm - expected_distance_mm) / expected_distance_mm * 100.0
            results['pairs'].append({'i': i, 'j': j, 'dist_mm': float(dist_mm), 'pct_error': float(pct_err)})
    return results
