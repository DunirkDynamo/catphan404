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

    # Attributes common to ALL instances of the class:
    roi_angles = [
            -87.4 + 180,
            -69.1 + 180,
            -52.7 + 180,
            -38.5 + 180,
            -25.1 + 180,
            -12.9 + 180,
        ]
    roi_dist_mm   = 50
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


    def __init__(self, image: np.ndarray, center: Tuple[float, float], pixel_spacing: float, angle_offset:float = 0.0):
        """
        Initialize the analyzer with the CT image and phantom center.

        Args:
            image (np.ndarray): 2D array of HU values from the CT slice.
            center (Tuple[float, float]): (x, y) coordinates of phantom center.
        """
        self.image         = image
        self.center        = center
        self.angle_offset  = angle_offset
        self.pixel_spacing = pixel_spacing
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
        if not isinstance(self.pixel_spacing, (float, int)):
            raise TypeError("pixel_spacing must be a float or int")
        if not isinstance(self.angle_offset, (float, int)):
            raise TypeError("angle_offset must be a float or int")

    def analyze(self) -> Dict:
        """
        Perform low-contrast detectability analysis.

        This method:
        1. Defines ROI locations based on predefined angles and distances.
        2. For each ROI, creates a circular mask and computes mean/std.
        3. Computes a common background ROI for noise reference.
        4. Calculates CNR for each ROI against the background.
        5. Returns a summary of detected ROIs and their metrics.

        Returns:
            Dict: Contains 'n_detected' (int) and 'blobs' (dict of blob stats).
                  Each blob entry has position, size, means, std, and CNR.
        """
        # Get image dimensions and center coordinates
        ny, nx = self.image.shape
        cy, cx = int(self.center[1]), int(self.center[0])

        # Initialize results dictionary
        results = {}
        
        # Compute background statistics from a common background ROI
        # Background ROI: 38mm from center, 12mm diameter (6mm radius)
        bg_dist_mm   = 38.0
        bg_radius_mm = 6.0
        bg_angle_deg = self.roi_angles[0] + self.angle_offset  # Use first angle for background
        bg_angle_rad = math.radians(bg_angle_deg)
        
        # Convert background ROI location to pixels
        bg_dist_px   = bg_dist_mm / self.pixel_spacing
        bg_radius_px = bg_radius_mm / self.pixel_spacing
        bg_x         = cx + bg_dist_px * math.cos(bg_angle_rad)
        bg_y         = cy + bg_dist_px * math.sin(bg_angle_rad)
        
        # Create background mask and extract values
        mask_bg = circular_roi_mask(self.image.shape, (bg_x, bg_y), bg_radius_px)
        bg_vals = self.image[mask_bg]
        
        # Compute background statistics
        if bg_vals.size < 10:
            raise ValueError("Insufficient background pixels for analysis")
        mean_bg = float(np.mean(bg_vals))
        std_bg  = float(np.std(bg_vals))
        
        # Process each ROI
        for i, (roi_name, roi_specs) in enumerate(self.roi_settings.items()):
            # Extract ROI specifications
            angle_deg   = roi_specs["angle"] + self.angle_offset
            distance_mm = roi_specs["distance"]
            radius_mm   = roi_specs["radius"]
            
            # Convert to pixels
            distance_px = distance_mm / self.pixel_spacing
            radius_px = radius_mm / self.pixel_spacing
            
            # Calculate ROI center position
            angle_rad = math.radians(angle_deg)
            x_full    = cx + distance_px * math.cos(angle_rad)
            y_full    = cy + distance_px * math.sin(angle_rad)
            
            # Create circular ROI mask
            mask = circular_roi_mask(self.image.shape, (x_full, y_full), radius_px)
            vals = self.image[mask]
            
            # Skip if insufficient data
            if vals.size < 10:
                continue
            
            # Compute ROI statistics
            mean_signal = float(np.mean(vals))
            std_signal  = float(np.std(vals))
            

            # Contrast: relative difference between signal and background magnitudes:
            contrast = abs(mean_signal - mean_bg) / (abs(mean_bg) + 1e-8)

            # Contrast-to-Noise Ratio: measures detectability
            # Higher CNR means the ROI is more visible
            cnr = abs(mean_signal - mean_bg) / (std_bg + 1e-8)
            
            # Store results for this ROI
            x_delta = int(x_full - cx)
            y_delta = int(y_full - cy)
            r_delta = (x_delta**2 + y_delta**2)**0.5
            
            results[f'roi_{roi_name}mm'] = {
                'x'       : int(x_full),
                'y'       : int(y_full),
                'r'       : float(radius_px),
                'x_delta' : x_delta,
                'y_delta' : y_delta,
                'r_delta' : r_delta,
                'angle'   : float(angle_deg),
                'mean'    : mean_signal,
                'std'     : std_signal,
                'bg_mean' : mean_bg,
                'bg_std'  : std_bg,
                'cnr'     : float(cnr),
                'contrast': float(contrast),
            }

        # Store and return summary: number detected and detailed ROI data
        self.results = {'n_detected': len(results), 'blobs': results}
        return self.results
