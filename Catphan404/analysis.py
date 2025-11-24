from typing import Optional, Tuple, Dict, Any
import numpy as np
from .uniformity import UniformityAnalyzer
from .geometry import GeometryAnalyzer
from .high_contrast import HighContrastAnalyzer
from .low_contrast import LowContrastAnalyzer
from .slice_thickness import SliceThicknessAnalyzer


class Catphan404Analyzer:
    """
    Central analyzer for Catphan 404 phantom.

    Can run individual analysis modules or all of them using `run_all()`.

    Attributes:
        image (np.ndarray): Input image.
        spacing (Optional[Tuple[float, float]]): Pixel spacing.
        results (dict): Dictionary storing results of each module.
    """

    def __init__(self, image: np.ndarray, spacing: Optional[Tuple[float, float]] = None):
        """Initialize analyzer."""
        self.image = np.array(image, dtype=float)
        self.spacing = (float(spacing[0]), float(spacing[1])) if spacing else None
        self.results: Dict[str, Any] = {}

    # ------------------ Existing uniformity / CT-number ------------------
    def run_uniformity(self):
        """Run the uniformity and CT-number insert analysis."""
        cy, cx = self._estimate_center(self.image)
        radius = min(self.image.shape) * 0.15
        analyzer = UniformityAnalyzer(self.image, (cx, cy), radius)
        self.results['uniformity'] = analyzer.analyze()
        self.results['center'] = (float(cx), float(cy))

    # ------------------ Other module run methods ------------------
    def run_high_contrast(self):
        """Run high-contrast / edge / MTF analysis."""
        analyzer = HighContrastAnalyzer(self.image)
        self.results['high_contrast'] = analyzer.analyze()

    def run_low_contrast(self):
        """Run low-contrast detectability analysis."""
        cy, cx = self._estimate_center(self.image)
        analyzer = LowContrastAnalyzer(self.image, (cx, cy))
        self.results['low_contrast'] = analyzer.analyze()

    def run_slice_thickness(self):
        """Run slice thickness (FWHM) analysis."""
        analyzer = SliceThicknessAnalyzer(self.image)
        self.results['slice_thickness'] = analyzer.analyze()

    def run_geometry(self):
        """Run geometric accuracy / fiducial analysis."""
        analyzer = GeometryAnalyzer(self.image, expected_distance_mm=50.0, spacing_mm=1.0 if not self.spacing else self.spacing[0])
        self.results['geometry'] = analyzer.analyze()

    # ------------------ Run all modules ------------------
    def run_all(self):
        """
        Run all available analysis modules in sequence and aggregate results.
        """
        self.run_uniformity()
        self.run_high_contrast()
        self.run_low_contrast()
        self.run_slice_thickness()
        self.run_geometry()
        return self.results

    # ------------------ Helper functions ------------------
    def _estimate_center(self, img: np.ndarray) -> Tuple[int, int]:
        """Estimate phantom center using intensity-weighted center of mass."""
        from scipy import ndimage
        sm = ndimage.gaussian_filter(img, sigma=3)
        thresh = np.percentile(sm, 50)
        bw = sm > thresh
        com = ndimage.center_of_mass(bw.astype(float))
        if np.isnan(com[0]):
            return img.shape[0] // 2, img.shape[1] // 2
        return int(com[0]), int(com[1])
