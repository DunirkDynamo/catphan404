# -----------------------------
# File: catphan404/uniformity.py
# -----------------------------
import numpy as np

class UniformityAnalyzer:
    """
    Analyzer for CT scanner uniformity using the CTP486 module.

    This class evaluates uniformity by measuring mean and standard deviation
    in five fixed ROIs (center, north, south, east, west) relative to the
    phantom center. The class also computes a uniformity metric as the
    percent difference between the maximum and minimum mean ROI values.

    Attributes:
        image (np.ndarray): 2D CT image of the uniformity module.
        center (tuple[float, float]): (x, y) coordinates of the phantom center in pixels.
        roi_size (float): Width/height of each square ROI in pixels (internal constant).
        roi_offset (float): Distance from center to offset peripheral ROIs (internal constant).
    """

    def __init__(self, image: np.ndarray, center: tuple[float, float], spacing: np.float64):
        """
        Initialize the UniformityAnalyzer.

        Args:
            image (np.ndarray): 2D CT image of the uniformity module.
            center (tuple[float, float]): (x, y) coordinates of the phantom center in pixels.
        """
        self.image = image.astype(float)
        self.center = center
        # ROI size and offsets in pixels (internal defaults)
        self.roi_size   = 15
        self.roi_offset = 50

   


        # Compute regions of interest based on initialization input.
        cy, cx          = self.center
        self.roi_size   = self.roi_size/spacing
        self.roi_offset = self.roi_offset/spacing

        # Compute ROI bounds
        half_size = int(self.roi_size // 2)
        offset    = int(self.roi_offset)
        # Center ROI
        self.mc = self.image[int(cx)-half_size:int(cx)+half_size,
                        int(cy)-half_size:int(cy)+half_size]
        # North ROI
        self.mn = self.image[int(cx)-half_size:int(cx)+half_size,
                        int(cy)+offset-half_size:int(cy)+offset+half_size]
        # South ROI
        self.ms = self.image[int(cx)-half_size:int(cx)+half_size,
                        int(cy)-offset-half_size:int(cy)-offset+half_size]
        # East ROI
        self.me = self.image[int(cx)+offset-half_size:int(cx)+offset+half_size,
                        int(cy)-half_size:int(cy)+half_size]
        # West ROI
        self.mw = self.image[int(cx)-offset-half_size:int(cx)-offset+half_size,
                        int(cy)-half_size:int(cy)+half_size]

    def analyze(self) -> dict:
        """
        Perform the uniformity analysis.

        Returns:
            dict: JSON-compatible dictionary containing mean, standard deviation
                  for each ROI and the overall uniformity metric.
        """



        # Compute means and standard deviations
        rois = {"centre": self.mc, "north": self.mn, "south": self.ms, "east": self.me, "west": self.mw}
        results = {}
        means = []

        for name, roi in rois.items():
            mean_val = float(np.mean(roi))
            std_val = float(np.std(roi))
            results[name] = {"mean": mean_val, "std": std_val}
            means.append(mean_val)

        # Overall uniformity metric
        uniformity = (max(means) - min(means)) / max(means) * 100
        results["uniformity"] = uniformity

        # Print diagnostics
        print("ROI statistics:")
        for name, stats in results.items():
            if name != "uniformity":
                print(f"{name.title()}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
        print(f"Uniformity: {uniformity:.2f} %\n")

        return results

    def to_dict(self) -> dict:
        """
        Return results in a JSON-compatible format.

        Returns:
            dict: Dictionary with ROI statistics and uniformity metric.
        """
        return self.analyze()
