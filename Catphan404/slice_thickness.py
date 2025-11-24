# -----------------------------
# File: catphan404/slice_thickness.py
# -----------------------------
"""Slice thickness measurement module for Catphan 404.

This implementation uses a simple technique: it finds a bright ramp/profile,
extracts a profile perpendicular to that feature, and estimates FWHM as a proxy
for slice thickness.
"""

from typing import Dict
import numpy as np
from .utils.image import find_edge_region


class SliceThicknessAnalyzer:
    """
    Class-based slice thickness analysis for Catphan 404.

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
        """Estimate slice thickness using FWHM on a bright ramp/profile.

        Returns:
            Dict: Estimated FWHM in pixels.
        """
        # Detect edge region
        y0, y1, x0, x1 = find_edge_region(self.image, width=60)
        roi = self.image[y0:y1, x0:x1]

        # Project along columns to get a profile
        profile = np.mean(roi, axis=1)

        # Normalize profile
        p = (profile - profile.min()) / (profile.max() - profile.min() + 1e-12)

        # Find indices above half maximum
        half_max = 0.5
        indices = np.where(p >= half_max)[0]

        fwhm = float(indices.max() - indices.min()) if indices.size > 0 else float('nan')

        return {'fwhm_pixels': fwhm}
