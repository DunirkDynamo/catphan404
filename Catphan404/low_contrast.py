# -----------------------------
# File: catphan404/low_contrast.py
# -----------------------------
"""Low-contrast detectability analysis for Catphan 404."""

from typing import Dict, Tuple
import numpy as np
from .utils.image import detect_blobs, circular_roi_mask


class LowContrastAnalyzer:
    """
    Class-based low-contrast detectability analysis for Catphan 404.

    Attributes:
        image (np.ndarray): Input image slice.
        center (Tuple[float, float]): Phantom center (x, y).
    """

    def __init__(self, image: np.ndarray, center: Tuple[float, float]):
        self.image = image
        self.center = center
        self._validate_inputs()

    def _validate_inputs(self):
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")
        if not (isinstance(self.center, tuple) and len(self.center) == 2):
            raise TypeError("center must be a tuple of (x, y)")

    def analyze(self) -> Dict:
        """Detect low-contrast inserts and compute CNR for each.

        Returns:
            Dict: Detected insert statistics and CNR values.
        """
        ny, nx = self.image.shape
        cy, cx = int(self.center[1]), int(self.center[0])

        # Define search region around phantom center
        search = self.image[
            max(0, cy - ny // 3):min(ny, cy + ny // 3),
            max(0, cx - nx // 3):min(nx, cx + nx // 3)
        ]

        # Detect low-contrast blobs
        blobs = detect_blobs(search, min_sigma=1.5, max_sigma=6, threshold=np.std(search) * 0.5)

        results = {}
        for i, b in enumerate(blobs):
            y, x, r = b
            # Translate to full image coordinates
            y_full = int(y) + max(0, cy - ny // 3)
            x_full = int(x) + max(0, cx - nx // 3)

            # ROI for blob signal
            mask = circular_roi_mask(self.image.shape, (x_full, y_full), r)
            vals = self.image[mask]

            # Background annulus
            mask_bg = circular_roi_mask(self.image.shape, (x_full, y_full), r * 2) & ~mask
            bg_vals = self.image[mask_bg]

            if vals.size < 10 or bg_vals.size < 10:
                continue

            mean_signal = float(np.mean(vals))
            mean_bg = float(np.mean(bg_vals))
            std_bg = float(np.std(bg_vals))
            cnr = abs(mean_signal - mean_bg) / (std_bg + 1e-8)

            results[f'blob_{i+1}'] = {
                'x': int(x_full),
                'y': int(y_full),
                'r': float(r),
                'mean': mean_signal,
                'bg_mean': mean_bg,
                'bg_std': std_bg,
                'cnr': float(cnr)
            }

        return {'n_detected': len(results), 'blobs': results}

