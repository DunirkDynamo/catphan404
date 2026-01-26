import numpy as np
from scipy import ndimage
from scipy.interpolate import interpn
from scipy.signal import find_peaks

class AnalyzerCTP401:
    """
    Analyzer for CTP401 (linearity module with material inserts).

    Performs ROI analysis on four material inserts (LDPE, Air, Teflon, Acrylic)
    at fixed angular positions. Computes mean HU, standard deviation, and
    low-contrast visibility (LCV) for each insert. Also derives a calibration
    scale relating nominal material density to measured HU.

    Attributes:
        image (np.ndarray): 2D CT image of the linearity module.
        center (tuple[float, float]): (x, y) center of phantom in pixels.
        pixel_spacing (float): Pixel spacing in mm.
        results (dict): Analysis results populated by analyze().
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

    def detect_rotation(self, initial_angle_deg: float = 0) -> tuple[float, tuple, tuple]:
        """
        Detect phantom rotation using air ROI positions.
        
        Similar to FindCTP404Rotation in the original code - finds the top and bottom
        air ROIs and calculates rotation based on their offset from vertical alignment.
        
        Args:
            initial_angle_deg (float): Initial rotation guess in degrees (default 0)
        
        Returns:
            tuple: (rotation_angle, top_air_center, bottom_air_center)
                - rotation_angle: rotation offset in degrees
                - top_air_center: (x, y) center of top air ROI
                - bottom_air_center: (x, y) center of bottom air ROI
        """
        image = self.image
        h, w = image.shape
        c = self.center
        space = self.pixel_spacing
        
        # Known radius to air ROI centers
        ring_r = 58.5 / space
        
        # Initial positions of top and bottom air ROIs (assuming 90° and -90°)
        ct = (ring_r * np.cos(np.radians(90 + initial_angle_deg)) + c[0],
              ring_r * np.sin(np.radians(90 + initial_angle_deg)) + c[1])
        cb = (ring_r * np.cos(np.radians(-90 + initial_angle_deg)) + c[0],
              ring_r * np.sin(np.radians(-90 + initial_angle_deg)) + c[1])
        
        # Setup for interpolation
        x = np.linspace(0, h - 1, h)
        y = np.linspace(0, w - 1, w)
        
        # Profile parameters
        profile_length = 25  # pixels to sample on each side
        granularity = 3      # sampling density
        
        def find_air_roi_center(roi_pos, profile_len, gran):
            """Find precise center of an air ROI by interpolating profiles."""
            # Horizontal and vertical coordinates for sampling
            x_horiz = np.linspace(roi_pos[0] - profile_len, roi_pos[0] + profile_len, profile_len * gran)
            x_vert = np.linspace(roi_pos[1] - profile_len, roi_pos[1] + profile_len, profile_len * gran)
            
            # Initialize profiles
            prof_h = np.zeros(len(x_horiz))
            prof_v = np.zeros(len(x_vert))
            
            # Interpolate profiles
            for i in range(len(x_horiz)):
                # Horizontal profile at fixed y
                prof_h[i] = interpn((x, y), image, [roi_pos[1], x_horiz[i]], 
                                   bounds_error=False, fill_value=0)
                # Vertical profile at fixed x
                prof_v[i] = interpn((x, y), image, [x_vert[i], roi_pos[0]], 
                                   bounds_error=False, fill_value=0)
            
            # Take derivatives to find edges
            dh = np.diff(prof_h)
            dv = np.diff(prof_v)
            
            # Find peaks in absolute derivative (edges of air ROI)
            try:
                peaks_h, _ = find_peaks(np.abs(dh), height=100)
                peaks_v, _ = find_peaks(np.abs(dv), height=100)
                
                # Calculate center offset from initial position
                if len(peaks_h) >= 2 and len(peaks_v) >= 2:
                    offset_len = len(x_horiz) / 2
                    mid_h = np.mean(peaks_h) - offset_len
                    mid_v = np.mean(peaks_v) - offset_len
                    return (roi_pos[0] + mid_h, roi_pos[1] + mid_v)
                else:
                    return roi_pos  # Fallback to initial position
            except:
                return roi_pos  # Fallback to initial position
        
        # Iteratively refine air ROI centers (similar to original code)
        iterations = 3
        for i in range(iterations):
            ct = find_air_roi_center(ct, profile_length, granularity)
            cb = find_air_roi_center(cb, profile_length, granularity)
        
        # Calculate rotation angle from offset between top and bottom ROIs
        tx = ct[0] - cb[0]
        ty = ct[1] - cb[1]
        rotation_angle = -np.arctan(tx / ty) * 180 / np.pi
        
        return rotation_angle, ct, cb
