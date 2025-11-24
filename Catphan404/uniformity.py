# -----------------------------
# File: catphan404/uniformity.py
# -----------------------------
"""Uniformity and CT-number module for Catphan 404."""

from typing import Dict, Tuple
import numpy as np
from .utils.image import circular_roi_mask


class UniformityAnalyzer:
    """
    Class-based uniformity analysis for Catphan 404.

    Attributes:
        image (np.ndarray): Input image slice.
        center (Tuple[float, float]): (x, y) center of phantom.
        radius (float): Radius for peripheral ROIs.
    """

    def __init__(self, image: np.ndarray, center: Tuple[float, float], radius: float):
        self.image = image
        self.center = center
        self.radius = radius
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")
        if not (isinstance(self.center, tuple) and len(self.center) == 2):
            raise TypeError("center must be a tuple of (x, y)")
        if not (isinstance(self.radius, (int, float)) and self.radius > 0):
            raise ValueError("radius must be a positive number")

    def analyze(self) -> Dict:
        """Compute uniformity ROIs and statistics.

        Returns:
            Dict: Results including per-ROI mean/std and uniformity metrics.
        """
        # Generate circular ROI masks
        masks = {
            'center': circular_roi_mask(self.image.shape, self.center, self.radius * 0.5),
            'right': circular_roi_mask(self.image.shape, (self.center[0] + self.radius * 0.7, self.center[1]), self.radius * 0.3),
            'left': circular_roi_mask(self.image.shape, (self.center[0] - self.radius * 0.7, self.center[1]), self.radius * 0.3),
            'top': circular_roi_mask(self.image.shape, (self.center[0], self.center[1] - self.radius * 0.7), self.radius * 0.3),
            'bottom': circular_roi_mask(self.image.shape, (self.center[0], self.center[1] + self.radius * 0.7), self.radius * 0.3),
        }

        stats = {}
        for k, m in masks.items():
            vals = self.image[m]
            stats[k] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'count': int(vals.size)
            }

        # Compute deviations relative to center ROI
        center_mean = stats['center']['mean']
        deviations = {k: abs(v['mean'] - center_mean) for k, v in stats.items()}

        return {
            'rois': stats,
            'deviations': deviations,
            'max_deviation': max(deviations.values())
        }
