# -----------------------------
# File: catphan404/high_contrast.py
# -----------------------------
"""High-contrast / line-pair / MTF analysis for Catphan 404."""

from typing import Dict
import numpy as np
from .utils.image import find_edge_region
from .utils.math import compute_mtf_from_edge


class HighContrastAnalyzer:
    """
    Class-based high-contrast / MTF analysis for Catphan 404.

    Attributes:
        image (np.ndarray): Input image slice.
    """

    def __init__(self, image: np.ndarray):
        self.image = image
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")

    def analyze(self) -> Dict:
        """Perform MTF estimation from a prominent edge in the image.

        Returns:
            Dict: MTF data and estimated f50 (frequency at 50% MTF).
        """
        # Detect edge ROI
        y0, y1, x0, x1 = find_edge_region(self.image, width=40)
        roi = self.image[y0:y1, x0:x1]

        # Compute MTF
        freqs, mtf = compute_mtf_from_edge(roi)

        # Estimate f50 (frequency at 50% MTF)
        try:
            f50 = float(np.interp(0.5, mtf[::-1], freqs[::-1]))
        except Exception:
            f50 = float('nan')

        return {
            'freqs': freqs.tolist(),
            'mtf': mtf.tolist(),
            'f50': f50
        }
