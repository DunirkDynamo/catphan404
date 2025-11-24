# -----------------------------
# File: catphan404/geometry.py
# -----------------------------
"""Geometric accuracy module for Catphan 404.

Detects fiducial blobs and measures pairwise distances against expected values.
"""

from typing import Dict, Tuple
import numpy as np
from .utils.image import detect_blobs


class GeometryAnalyzer:
    """
    Class-based geometric accuracy analysis for Catphan 404.

    Attributes:
        image (np.ndarray): Input image slice.
        expected_distance_mm (float): Known reference distance between markers in mm.
        spacing_mm (float): Pixel spacing in mm.
    """

    def __init__(self, image: np.ndarray, expected_distance_mm: float = 50.0, spacing_mm: float = 1.0):
        self.image = image
        self.expected_distance_mm = float(expected_distance_mm)
        self.spacing_mm = float(spacing_mm)
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")
        if self.expected_distance_mm <= 0:
            raise ValueError("expected_distance_mm must be positive")
        if self.spacing_mm <= 0:
            raise ValueError("spacing_mm must be positive")

    def analyze(self) -> Dict:
        """Detect fiducial markers and compute distance errors.

        Returns:
            Dict: Detected marker positions and measured distances (mm and % error).
        """
        # Detect blobs
        blobs = detect_blobs(self.image, min_sigma=2, max_sigma=8, threshold=np.std(self.image) * 0.4)
        centers = [(int(b[1]), int(b[0])) for b in blobs]

        results = {'n_markers': len(centers), 'markers': centers, 'pairs': []}

        # Compute pairwise distances
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                x1, y1 = centers[i]
                x2, y2 = centers[j]
                dist_px = np.hypot(x2 - x1, y2 - y1)
                dist_mm = dist_px * self.spacing_mm
                pct_err = (dist_mm - self.expected_distance_mm) / self.expected_distance_mm * 100.0
                results['pairs'].append({
                    'i': i,
                    'j': j,
                    'dist_mm': float(dist_mm),
                    'pct_error': float(pct_err)
                })

        return results
