import numpy as np

class AnalyzerCTP401:
    """
    Analyzer for CTP401.

    Performs ROI analysis on various material inserts of differing nominal HU values.

    Attributes:
        image (np.ndarray)                  : 2D CT image (already slice-averaged)
        center (tuple[float, float])        : (x, y) center of phantom in pixels
        pixel_spacing (float)               : Pixel spacing in mm
    """

    def __init__(self, image: np.ndarray, center: tuple[float, float], pixel_spacing: float):
        self.image         = image.astype(float)
        self.center        = center
        self.pixel_spacing = pixel_spacing
        self.results       = {}
        self.scale         = None
        self.roi_traces    = None

    def create_circular_mask(self, h, w, center=None, radius=None):
        """Create a boolean circular mask."""
        if center is None:
            center = (int(w/2), int(h/2))
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        return dist_from_center <= radius

    def analyze(self, t_offset: float = 0):
        """
        Perform ROI analysis on the stored image.

        Args:
            t_offset (float): rotational offset for ROIs in degrees

        Returns:
            dict: JSON-compatible results containing mean, std, LCV, scale
        """
        image  = self.image
        h, w   = image.shape[:2]
        c0     = self.center
        space  = self.pixel_spacing

        # ROI parameters
        r      = 3.5 / space      # ROI radius in pixels
        ring_r = 58.5 / space     # radius to ring center

        # Only use ROIs 1,4,6,9
        roi_angles = {
            'ROI0_LDPE'      : 0,
            'ROI90_Air'      : 90,
            'ROI180_Teflon'  : 180,
            'ROI270_Acrylic' : -90
        }
        masks = {}
        results = {}

        for name, angle in roi_angles.items():
            cx            = ring_r * np.cos(np.radians(angle + t_offset)) + c0[0]
            cy            = ring_r * np.sin(np.radians(angle + t_offset)) + c0[1]
            mask          = self.create_circular_mask(h, w, center=(cx, cy), radius=r)
            masks[name]   = mask
            roi_mean      = float(np.mean(image[mask]))
            roi_std       = float(np.std(image[mask]))
            results[name] = {'mean': roi_mean, 'std': roi_std}

        # Compute Low Contrast Visibility (LCV) using Delrin (ROI1) and LDPE (ROI6)
        lcv = 3.25 * (results['ROI90_Air']['std'] + results['ROI0_LDPE']['std']) / \
              (results['ROI90_Air']['mean'] - results['ROI0_LDPE']['mean'])

        # Scaling factors along x and y using ROI1 vs ROI6 (x) and ROI4 vs ROI9 (y)
        px = image[int(round(c0[0])), :].astype(float)
        py = image[:, int(round(c0[1]))].astype(float)
        profile_length = 26

        idx_x1 = int(round(ring_r * np.cos(np.radians(0 + t_offset)) + c0[0]))
        idx_x2 = int(round(ring_r * np.cos(np.radians(180 + t_offset)) + c0[0]))
        idx_y1 = int(round(ring_r * np.sin(np.radians(90 + t_offset)) + c0[1]))
        idx_y2 = int(round(ring_r * np.sin(np.radians(-90 + t_offset)) + c0[1]))

        px1 = px[idx_x1-profile_length:idx_x1+profile_length]
        px2 = px[idx_x2-profile_length:idx_x2+profile_length]
        py1 = py[idx_y1-profile_length:idx_y1+profile_length]
        py2 = py[idx_y2-profile_length:idx_y2+profile_length]

        dpx1, dpx2 = np.diff(px1), np.diff(px2)
        dpy1, dpy2 = np.diff(py1), np.diff(py2)

        minx1, maxx1 = np.argmin(dpx1), np.argmax(dpx1)
        minx2, maxx2 = np.argmin(dpx2), np.argmax(dpx2)
        miny1, maxy1 = np.argmin(dpy1), np.argmax(dpy1)
        miny2, maxy2 = np.argmin(dpy2), np.argmax(dpy2)

        scaleX1 = (abs(idx_x2 - idx_x1) * space - abs(minx1 - minx2) * space) / 10
        scaleX2 = (abs(idx_x2 - idx_x1) * space - abs(maxx1 - maxx2) * space) / 10
        scaleY1 = (abs(idx_y1 - idx_y2) * space - abs(miny1 - miny2) * space) / 10
        scaleY2 = (abs(idx_y1 - idx_y2) * space - abs(maxy1 - maxy2) * space) / 10

        scaleX = float(np.mean([scaleX1, scaleX2]))
        scaleY = float(np.mean([scaleY1, scaleY2]))
        self.scale = {'scaleX_cm': scaleX, 'scaleY_cm': scaleY}

        # Store results
        self.results = {
            'ROIs'        : results,
            'LCV_percent' : float(lcv),
            'Scale'       : self.scale
        }

        return self.results
