# -----------------------------
# File: catphan404/ctp515_analyzer.py
# -----------------------------
"""Low-contrast detectability analysis for Catphan 404."""

from typing import Dict, Tuple
import numpy as np
import math
from .utils.image import detect_blobs, circular_roi_mask


class AnalyzerCTP515:
    """
    Class-based low-contrast detectability analysis for Catphan 401.

    This analyzer detects small, low-contrast inserts (blobs) in the phantom
    and computes Contrast-to-Noise Ratio (CNR) for each, quantifying how
    detectable they are against background noise.

    "Sets" of inserts, defined by their nominal contrast, include multiple 
    inserts of differing diameters. That is, for each nominal contrast value, 
    multiple ROIs exist as inserts of different diameters.

    Attributes:
        image (np.ndarray)                  : 2D CT image
        center (tuple[float, float])        : (x, y) center of phantom in pixels
        pixel_spacing float)                : Pixel spacing in mm
    """

    def __init__(self, image: np.ndarray, center: Tuple[float, float], pixel_spacing: float):
        """
        Initialize the analyzer with the CT image and phantom center.

        Args:
            image (np.ndarray): 2D array of HU values from the CT slice.
            center (Tuple[float, float]): (x, y) coordinates of phantom center.
        """
        self.image  = image
        self.center = center
        # Validate inputs to ensure correct types and dimensions
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validate the input parameters to prevent runtime errors.

        Raises:
            TypeError: If image is not a numpy array or center is not a tuple.
            ValueError: If image is not 2D.
        """
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.ndim != 2:
            raise ValueError("image must be a 2D array")
        if not (isinstance(self.center, tuple) and len(self.center) == 2):
            raise TypeError("center must be a tuple of (x, y)")

    def analyze(self) -> Dict:




    roi_angles = [
            -87.4 + 180,
            -69.1 + 180,
            -52.7 + 180,
            -38.5 + 180,
            -25.1 + 180,
            -12.9 + 180,
        ]
        roi_dist_mm = 50
        roi_radius_mm = [6, 3.5, 3, 2.5, 2, 1.5]
        roi_settings = {
            "15": {
                "angle": roi_angles[0],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[0],
            },
            "9": {
                "angle": roi_angles[1],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[1],
            },
            "8": {
                "angle": roi_angles[2],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[2],
            },
            "7": {
                "angle": roi_angles[3],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[3],
            },
            "6": {
                "angle": roi_angles[4],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[4],
            },
            "5": {
                "angle": roi_angles[5],
                "distance": roi_dist_mm,
                "radius": roi_radius_mm[5],
            },
        }


        """
        Perform low-contrast detectability analysis.

        This method:
        1. Defines a search region around the phantom center.
        2. Detects potential low-contrast blobs using blob detection.
        3. For each blob, computes signal/background stats and CNR.
        4. Returns a summary of detected blobs and their metrics.

        Returns:
            Dict: Contains 'n_detected' (int) and 'blobs' (dict of blob stats).
                  Each blob entry has position, size, means, std, and CNR.
        """
        # Get image dimensions and center coordinates
        ny, nx = self.image.shape
        cy, cx = int(self.center[1]), int(self.center[0])

        # Define a central search region to focus blob detection
        # This avoids edge artifacts and focuses on the phantom's core
        # where low-contrast inserts are typically placed
        search = self.image[
            max(0, cy - ny // 3):min(ny, cy + ny // 3),
            max(0, cx - nx // 3):min(nx, cx + nx // 3)
        ]

        # Detect blobs using Laplacian of Gaussian (LoG) filtering
        # Parameters tuned for typical low-contrast inserts:
        # - min_sigma/max_sigma: Size range for blobs (pixels)
        # - threshold: Adaptive based on local noise level
        blobs = detect_blobs(search, min_sigma=1.5, max_sigma=6, threshold=np.std(search) * 0.5)

        # Initialize results dictionary
        results = {}
        
        # Process each detected blob
        for i, b in enumerate(blobs):
            # Unpack blob properties: y, x (in search coords), radius
            y, x, r = b
            
            # Translate coordinates from search region to full image
            y_full = int(y) + max(0, cy - ny // 3)
            x_full = int(x) + max(0, cx - nx // 3)

            # Create circular ROI mask for the blob's signal area
            mask = circular_roi_mask(self.image.shape, (x_full, y_full), r)
            vals = self.image[mask]  # HU values in the blob

            # Create annulus mask for background (ring around blob)
            # Radius 2x blob size, excluding the signal area
            mask_bg = circular_roi_mask(self.image.shape, (x_full, y_full), r * 2) & ~mask
            bg_vals = self.image[mask_bg]  # HU values in background

            # Skip if insufficient data (avoids division by zero or noise)
            if vals.size < 10 or bg_vals.size < 10:
                continue

            # Compute statistics
            mean_signal = float(np.mean(vals))      # Average HU in blob
            mean_bg = float(np.mean(bg_vals))        # Average HU in background
            std_bg = float(np.std(bg_vals))          # Noise (std dev) in background
            
            # Contrast-to-Noise Ratio: measures detectability
            # Higher CNR means the blob is more visible
            cnr = abs(mean_signal - mean_bg) / (std_bg + 1e-8)  # Add epsilon to avoid div by zero

            # Calculate angle of blob center relative to phantom center
            # atan2(y_delta, x_delta) gives angle in radians from positive x-axis
            angle_rad = math.atan2(y_full - cy, x_full - cx)
            angle_deg = math.degrees(angle_rad)  # Convert to degrees for readability

            # Store results for this blob
            x_delta = int(x_full - cx)               # x offset from phantom center
            y_delta = int(y_full - cy)               # y offset from phantom center
            r_delta = (x_delta**2 + y_delta**2)**0.5 # radial offset from phantom center
            results[f'blob_{i+1}'] = {
                'x'       : int(x_full),          # Blob center x-coordinate
                'y'       : int(y_full),          # Blob center y-coordinate
                'r'       : float(r),             # Blob radius (pixels)
                'x_delta' : x_delta,              # x offset from phantom center
                'y_delta' : y_delta,              # y offset from phantom center
                'r_delta' : r_delta,              # radial offset from phantom center
                'angle'   : float(angle_deg),     # Angle in degrees from phantom center
                'mean'    : mean_signal,          # Mean HU in blob
                'bg_mean' : mean_bg,              # Mean HU in background
                'bg_std'  : std_bg,               # Std dev HU in background
                'cnr'     : float(cnr)            # Contrast-to-Noise Ratio
            }

        # Return summary: number detected and detailed blob data
        return {'n_detected': len(results), 'blobs': results}
