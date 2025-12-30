from typing import Optional, Tuple, Dict, Any
import numpy as np
import json
from .uniformity import UniformityAnalyzer
from .high_contrast import HighContrastAnalyzer
from .ctp401_analyzer import AnalyzerCTP401
from .ctp515_analyzer import AnalyzerCTP515
from pathlib import Path


class Catphan404Analyzer:
    """
    Central coordinator for Catphan 404 phantom analysis.

    Orchestrates analysis of individual phantom modules (uniformity, high-contrast,
    linearity, low-contrast) on single-slice CT images. Each module is run via
    a dedicated run_* method that initializes the appropriate analyzer, executes
    analysis, and stores results.

    Attributes:
        image (np.ndarray): Input 2D CT image.
        spacing (Optional[Tuple[float, float]]): Pixel spacing (x, y) in mm.
        results (dict): Dictionary storing JSON-compatible results from each module.
    """

    def __init__(self, image: np.ndarray, spacing: Optional[Tuple[float, float]] = None):
        """Initialize analyzer."""
        self.image                   = np.array(image, dtype=float)
        self.spacing                 = (float(spacing[0]), float(spacing[1])) if spacing else None
        self.results: Dict[str, Any] = {}

        # Store actual analyzer objects for plotting:
        self._uniformity_analyzer    = None
        self._high_contrast_analyzer = None
        self._ctp401_analyzer        = None
        self._ctp515_analyzer        = None

    # ------------------ Existing uniformity / CT-number ------------------
    def run_uniformity(self):
        """
        Run uniformity analysis (CTP486 module).

        Estimates phantom center, creates UniformityAnalyzer instance,
        analyzes five ROIs, and stores results. Also stores the detected
        center for use by other modules.

        Populates:
            self.results['uniformity']: ROI statistics and uniformity metric.
            self.results['center']: Detected (x, y) center coordinates.
        """
        # Estimate phantom center from the image
        cy, cx = self._estimate_center(self.image)

        # Initialize the uniformity analyzer with image and center
        analyzer = UniformityAnalyzer(self.image, (cx, cy), self.spacing[0])

        # Run analysis and store results
        self.results['uniformity'] = analyzer.analyze()

        # Also store center for reference
        self.results['center'] = (float(cx), float(cy))

        # Store the analyzer:
        self._uniformity_analyzer = analyzer


    # ------------------ High Contrast Module (Line pairs) ----------------
    def run_high_contrast(self):
        """
        Run high-contrast line pair analysis (CTP528 module).

        Analyzes spatial resolution by measuring MTF from line pair patterns.
        Uses center from uniformity analysis if available, otherwise estimates it.

        Populates:
            self.results['high_contrast']: MTF curve data and resolution metrics.
        """

        # Use center from uniformity analysis if available
        center = self.results.get('center', None)
        if center is None:
            cy, cx = self._estimate_center(self.image)
            center = (cy, cx)

        spacing = self.spacing[0] if self.spacing else 1.0

        analyzer = HighContrastAnalyzer(
            image=self.image,
            center=center,      # Important
            pixel_spacing=spacing
        )


        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['high_contrast'] = res

        # Store the analyzer:
        self._high_contrast_analyzer = analyzer

    # --------------  Linearity Module (HU material inserts) --------------
    def run_ctp401(self, t_offset: float = 0):
        """
        Run linearity/scaling analysis (CTP401 module).

        Analyzes material insert ROIs to measure HU values for different
        materials (LDPE, Air, Teflon, Acrylic) and derives a calibration scale.

        Args:
            t_offset (float): Rotational offset in degrees for ROI positioning.

        Populates:
            self.results['ctp401']: Material ROI statistics and calibration data.
        """


        # Use center from uniformity analysis if available
        center = self.results.get('center', None)
        if center is None:
            cy, cx = self._estimate_center(self.image)
            center = (cy, cx)

        spacing = self.spacing[0] if self.spacing else 1.0

        analyzer = AnalyzerCTP401(
            image=self.image,
            center=center,      # Important
            pixel_spacing=spacing
        )


        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['ctp401'] = res

        # Store the analyzer:
        self._ctp401_analyzer = analyzer




    # ------------------ As yet undeveloped modules ---------------- 

    def run_ctp515(self, crop_x=150, crop_y=150):
        """
        Run low-contrast detectability analysis (CTP515 module).

        Detects low-contrast inserts of varying diameters and computes CNR
        and contrast values to assess detectability. Uses geometric center
        of potentially cropped image.

        Args:
            crop_x (int): Number of pixels to crop from left and right edges.
            crop_y (int): Number of pixels to crop from top and bottom edges.

        Populates:
            self.results['ctp515']: Low-contrast ROI statistics, CNR, and contrast values.
        """
        """
        # Crop the image if requested
        if crop_x > 0 or crop_y > 0:
            h, w = self.image.shape
            cropped_image = self.image[crop_y:h-crop_y, crop_x:w-crop_x]
            # Adjust center for cropped image
            cy, cx = cropped_image.shape[0] // 2, cropped_image.shape[1] // 2
        else:
            cropped_image = self.image
            cy, cx = self.image.shape[0] // 2, self.image.shape[1] // 2
        
        center = (cx, cy)  # Store as (x, y) = (col, row)

        spacing  = self.spacing[0] if self.spacing else 1.0
        analyzer = AnalyzerCTP515(cropped_image, center, spacing, angle_offset=-7.5)
        
        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['ctp515'] = res

        # Store the analyzer:
        self._ctp515_analyzer = analyzer

    def run_slice_thickness(self):
        """Run slice thickness (FWHM) analysis."""
        analyzer = SliceThicknessAnalyzer(self.image)
        self.results['slice_thickness'] = analyzer.analyze()


    # ------------------ Helper functions ------------------
    def _estimate_center(self, img: np.ndarray) -> Tuple[int, int]:
        """
        Estimate phantom center using intensity-weighted center of mass.

        Applies Gaussian smoothing, thresholds at median, and computes the
        center of mass of the resulting binary mask.

        Args:
            img (np.ndarray): Input 2D image.

        Returns:
            Tuple[int, int]: (row, col) center coordinates in pixels.
        """
        from scipy import ndimage
        sm = ndimage.gaussian_filter(img, sigma=3)
        thresh = np.percentile(sm, 50)
        bw = sm > thresh
        com = ndimage.center_of_mass(bw.astype(float))
        if np.isnan(com[0]):
            return img.shape[0] // 2, img.shape[1] // 2
        return int(com[0]), int(com[1])
    

    def save_results_json(self, path):
        """
        Save all collected analysis results to a JSON file.

        Args:
            path (str | Path): Where to save the JSON output.

        Raises:
            ValueError: If no analysis results exist yet.
            OSError: If writing the file fails.
        """
        if not self.results:
            raise ValueError(
                "No results available. Run at least one analysis module before saving."
            )

        out_path = Path(path)

        # Create parent directory if needed
        if out_path.parent != Path('.'):
            out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
        except OSError as e:
            raise OSError(f"Failed to write JSON to {out_path}: {e}")

        return out_path
