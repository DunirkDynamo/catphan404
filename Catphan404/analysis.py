from typing import Optional, Tuple, Dict, Any
import numpy as np
import json
from .uniformity import UniformityAnalyzer
from .geometry import GeometryAnalyzer
from .high_contrast import HighContrastAnalyzer
from .ctp401_analyzer import AnalyzerCTP401
from .ctp515_analyzer import AnalyzerCTP515
from .slice_thickness import SliceThicknessAnalyzer
from pathlib import Path


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
        Run the uniformity analysis using the UniformityAnalyzer.
        Populates self.results['uniformity'] with the computed statistics.
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
        Run high-contrast (CTP528) analysis.
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
        Run CTP401 material / scaling analysis.
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

    def run_ctp515(self):
        """Run CTP515 low-contrast detectability analysis."""
        # Use center from uniformity analysis if available
        center = self.results.get('center', None)
        if center is None:
            cy, cx = self._estimate_center(self.image)
            center = (cy, cx)

        spacing = self.spacing[0] if self.spacing else 1.0
        analyzer = AnalyzerCTP515(self.image, center, spacing)
        
        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['ctp515'] = res

        # Store the analyzer:
        self._ctp515_analyzer = analyzer

    def run_slice_thickness(self):
        """Run slice thickness (FWHM) analysis."""
        analyzer = SliceThicknessAnalyzer(self.image)
        self.results['slice_thickness'] = analyzer.analyze()


    # ------------------ Run all modules ------------------
    def run_all(self):
        """
        Run all available analysis modules in sequence and aggregate results.
        """
        self.run_uniformity()
        self.run_high_contrast()
        self.run_ctp515_analyzer()
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
