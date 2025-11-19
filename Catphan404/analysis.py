# -----------------------------
# File: catphan404/analysis.py
# -----------------------------
# analysis.py

from typing import Optional, Tuple, Dict
import numpy as np
from scipy import ndimage
from . import io  # Keep io import for image loading convenience

class Catphan404Analyzer:
    """
    Analyzer for a single slice containing the Catphan 404 phantom.

    This class computes standard uniformity and CT-number metrics,
    and can be extended to include other analysis modules.

    Example:
        ```python
        from catphan404 import Catphan404Analyzer, io
        img, meta = io.load_image('catphan_slice.dcm')
        analyzer = Catphan404Analyzer(img, spacing=meta.get('Spacing'))
        analyzer.run_all()
        print(analyzer.results)
        ```

    Attributes:
        image (np.ndarray): Image array.
        spacing (Optional[Tuple[float, float]]): Pixel spacing (mm).
        results (dict): Dictionary of computed results.
    """

    def __init__(self, image: np.ndarray, spacing: Optional[Tuple[float, float]] = None):
        """Initialize the analyzer.

        Args:
            image (np.ndarray): Input image array.
            spacing (Optional[Tuple[float, float]]): Pixel spacing in mm (row, col).
        """
        self.image = np.array(image, dtype=float)
        self.spacing: Optional[Tuple[float, float]] = None
        if spacing is not None:
            try:
                self.spacing = (float(spacing[0]), float(spacing[1]))
            except Exception:
                pass
        self.results: Dict = {}

    def analyze(self) -> Dict:
        """
        Perform basic Catphan 404 analysis (uniformity + CT-number inserts).

        Returns:
            dict: Computed metrics for uniformity and CT-number inserts.
        """
        return self._run_uniformity() | self._run_ct_number()

    def run_all(self) -> Dict:
        """
        Run the full suite of Catphan 404 analyses.

        Returns:
            dict: Results dictionary including all analysis modules.
        """
        self.results.clear()
        # Currently we only have uniformity + CT-number; can extend later
        self.results.update(self._run_uniformity())
        self.results.update(self._run_ct_number())
        return self.results

    # -------------------------
    # Internal helper routines
    # -------------------------

    def _run_uniformity(self) -> Dict:
        """Compute uniformity ROIs and deviations."""
        img = self.image
        cy, cx = self._estimate_center(img)
        radius = min(img.shape) * 0.15
        rois = self._make_circular_rois((cx, cy), radius, img.shape)
        roi_stats = {}
        for name, mask in rois.items():
            vals = img[mask]
            roi_stats[name] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'count': int(vals.size)
            }

        center_mean = roi_stats['center']['mean']
        deviations = {k: abs(v['mean'] - center_mean) for k, v in roi_stats.items()}

        return {
            'center': (float(cx), float(cy)),
            'uniformity_rois': roi_stats,
            'uniformity_deviations': deviations,
            'max_uniformity_deviation': float(max(deviations.values()))
        }

    def _run_ct_number(self) -> Dict:
        """Compute CT-number insert ROIs."""
        img = self.image
        cy, cx = self._estimate_center(img)
        radius = min(img.shape) * 0.15 * 2.2  # larger ring for inserts
        inserts = self._estimate_ct_number_rois((cx, cy), radius, img.shape)
        insert_stats = {}
        for i, mask in enumerate(inserts):
            vals = img[mask]
            insert_stats[f'insert_{i+1}'] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'count': int(vals.size)
            }
        return {'ct_number_rois': insert_stats}

    # -------------------------
    # Internal geometry helpers
    # -------------------------

    def _estimate_center(self, img: np.ndarray) -> Tuple[int, int]:
        """Estimate phantom center using intensity-weighted COM."""
        sm = ndimage.gaussian_filter(img, sigma=3)
        thresh = np.percentile(sm, 50)
        bw = sm > thresh
        com = ndimage.center_of_mass(bw.astype(float))
        if np.isnan(com[0]):
            return img.shape[0] // 2, img.shape[1] // 2
        return int(com[0]), int(com[1])

    def _make_circular_rois(self, center_xy: Tuple[float, float], radius: float, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Generate circular ROIs for uniformity analysis."""
        cx, cy = center_xy
        ny, nx = shape
        Y, X = np.ogrid[:ny, :nx]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        masks = {
            'center': dist <= (radius * 0.5),
            'right': (dist <= radius) & (X > cx + radius * 0.3),
            'left': (dist <= radius) & (X < cx - radius * 0.3),
            'top': (dist <= radius) & (Y < cy - radius * 0.3),
            'bottom': (dist <= radius) & (Y > cy + radius * 0.3),
        }
        return masks

    def _estimate_ct_number_rois(self, center_xy: Tuple[float, float], ring_radius: float, shape: Tuple[int, int]) -> list[np.ndarray]:
        """Generate approximate CT-number insert ROIs."""
        cx, cy = center_xy
        ny, nx = shape
        n_inserts = 8
        roi_radius = int(min(nx, ny) * 0.03)
        masks = []
        angles = np.linspace(0, 2 * np.pi, n_inserts, endpoint=False)
        Y, X = np.ogrid[:ny, :nx]
        for ang in angles:
            rx = cx + ring_radius * np.cos(ang)
            ry = cy + ring_radius * np.sin(ang)
            dist = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2)
            masks.append(dist <= roi_radius)
        return masks
