import numpy as np
from scipy.interpolate import interpn
from scipy.signal import find_peaks

class HighContrastAnalyzer:
    """
    Purpose: Analyze Catphan's CTP528 module.

    Order of operations:
      - Computes line-pair center coordinates from lp radius + angles
      - For each adjacent center pair, samples a profile between the two centers
        using `scipy.interpolate.interpn`
      - Derives derivative of profile, finds derivative peaks, identifies
        local maxima/minima and computes modulation (MTF) per pair
      - Normalizes the per-pair MTFs to produce an aggregated normalized MTF
      - Interpolates to find MTF10/30/50/80 in lp/mm

    Constructor arguments mirror what `analysis_CTP528` needs, but simplified:
      image: 2D numpy array (the single-slice image already selected/averaged)
      pixel_spacing: mm/pixel (scalar; uses same value for x and y)
      center: (x_pixel, y_pixel) tuple matching `outer_c` in original script
      t_offset_deg: rotation offset in degrees (same role as `t_offset`)
      lp_r_mm: radius in mm used in original script (default 48)
      samples_per_segment: number of sample points between adjacent centers (default 50)
    """

    def __init__(
        self,
        image              : np.ndarray,
        pixel_spacing      : float,
        center             : tuple,
        t_offset_deg       : float = 0.0,
        lp_r_mm            : float = 48.0,
        samples_per_segment: int   = 50,
    ):
        self.image         = image.astype(float)
        self.pixel_spacing = float(pixel_spacing)
        # center expected as (x_pixel, y_pixel) to match processDICOM convention
        self.center_x            = float(center[0])
        self.center_y            = float(center[1])
        self.t_offset_deg        = float(t_offset_deg)
        self.lp_r_mm             = float(lp_r_mm)
        self.samples_per_segment = int(samples_per_segment)

        # default theta list (degrees) per original script (i.e. Devin's) plus t_offset
        self.theta_deg = np.array([10, 40, 62, 85, 103, 121, 140, 157, 173, 186]) + self.t_offset_deg
        # internal storage
        self.lpx       = None
        self.lpy       = None
        self.npeaks    = [[1, 2], [2, 3], [3, 4], [4, 4], [5, 4], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5]]

        # outputs to be filled by analyze()
        self.per_pair_mtf   = []    # scalar MTF per line pair (length 9 in original loop)
        self.profiles       = []    # f arrays per pair
        self.peaks_max      = []    # derivative peaks maxima arrays per pair
        self.peaks_min      = []    # derivative peaks minima arrays per pair
        self.peaks_combined = []    # combined peaks positions per pair
        self.lp_x           = []    # tmpx per pair (lpx values passed back in original)
        self.lp_y           = []    # tmpy per pair
        self.nMTF           = None  # aggregated normalized MTF (same style as script)
        self.lp_axis        = None  # lp axis (lp/mm) as used in script
        self.mtf_points     = {}    # MTF80/50/30/10 results

    # -------------------------
    # Geometry: compute centers
    # -------------------------
    def _compute_centers(self):
        """Compute the pixel coordinates of the line-pair centers (lpx, lpy)."""
        r_pixels   = self.lp_r_mm / self.pixel_spacing
        thetas_rad = np.deg2rad(self.theta_deg)
        self.lpx   = r_pixels * np.cos(thetas_rad) + self.center_x
        self.lpy   = r_pixels * np.sin(thetas_rad) + self.center_y

    # -------------------------
    # Core get_MTF translator
    # -------------------------
    def _get_MTF_for_pair(self, x_coords, y_coords, npeaks_expected):
        """
        Implements the logic of the inner get_MTF in analysis_CTP528.

        x_coords, y_coords: tuples (x0, x1), (y0, y1) in pixel coordinates for segment endpoints
        npeaks_expected: [pair_number, expected_num_peaks] (same format as original)
        Returns: (MTF_scalar, profile_f1, peaks_max1, peaks_min1, peaks1, lpx_pair, lpy_pair)
        """
        # build sample points along the segment (50 samples in original code)
        x1 = np.linspace(x_coords[0], x_coords[1], self.samples_per_segment)
        y1 = np.linspace(y_coords[0], y_coords[1], self.samples_per_segment)
        f1 = np.zeros(len(x1))

        # prepare grid coords like original script
        ny, nx = self.image.shape
        x      = np.linspace(0, (nx - 1), nx)  # careful: original on some parts uses x/y reverse; here we use (cols) as x
        y      = np.linspace(0, (ny - 1), ny)

        # interpolate profile across the segment (matching interpn((x,y), img, [y,x]) ordering)
        for i in range(len(x1)):
            # interpn expects points (y, x)
            f1[i] = interpn((y, x), self.image, [ [y1[i], x1[i]] ], method='linear', bounds_error=False, fill_value=0.0)[0]

        # derivative
        df1 = np.diff(f1)

        # find derivative peaks (height threshold starts at 50)
        h             = 50
        peaks_max1, _ = find_peaks(df1, height=h)
        peaks_min1, _ = find_peaks(-df1, height=h)

        # reduce threshold until expected count or until h <= 10 (same logic)
        while (len(peaks_max1) < npeaks_expected[1]) or (len(peaks_min1) < npeaks_expected[1]):
            if h <= 10:
                # cannot resolve line pair — return MTF = 0 as original script
                return 0.0, f1, peaks_max1, peaks_min1, np.array([], dtype=int), x_coords, y_coords
            h -= 1
            peaks_max1, _ = find_peaks(df1, height=h)
            peaks_min1, _ = find_peaks(-df1, height=h)

        # combine and sort peaks
        peaks1 = np.hstack((peaks_max1, peaks_min1))
        peaks1 = np.array(sorted(peaks1))

        # sample local maxima/minima intensities between consecutive derivative peaks
        idxmax = []
        idxmin = []
        Imax   = []
        Imin   = []
        offset = 1
        for k in range(len(peaks1) - 1):
            if k % 2 == 0:
                tmp_idx = np.array(f1[peaks1[k] - offset:peaks1[k + 1] + offset]).argmax()
                idx_at = tmp_idx - offset + peaks1[k]
                idxmax.append(idx_at)
                Imax.append(f1[idx_at])
            else:
                tmp_idx = np.array(f1[peaks1[k] - offset:peaks1[k + 1] + offset]).argmin()
                idx_at = tmp_idx - offset + peaks1[k]
                idxmin.append(idx_at)
                Imin.append(f1[idx_at])

        # compute MTF (modulation) for this pair
        if (len(Imax) == 0) or (len(Imin) == 0):
            MTF_value = 0.0
        else:
            MTF_value = (np.mean(Imax) - np.mean(Imin)) / (np.mean(Imax) + np.mean(Imin))

        return float(MTF_value), f1, peaks_max1, peaks_min1, peaks1, x_coords, y_coords

    # -------------------------
    # High-level analysis
    # -------------------------
    def analyze(self):
        """
        Execute the full CTP528-style analysis and populate the instance attributes:
          - per_pair_mtf, profiles, peaks arrays, lp_x/lp_y
          - nMTF (aggregated normalized MTF), lp_axis (lp/mm)
          - mtf_points (MTF80/50/30/10)
        """
        # compute centers
        self._compute_centers()

        # iterate adjacent pairs (original loop does range(len(theta)-1))
        n_pairs = len(self.lpx) - 1
        per_pair_mtf = []
        profiles = []
        pmax_list = []
        pmin_list = []
        pcomb_list = []
        lp_x_list = []
        lp_y_list = []

        # prepare original x,y axes consistent with original: x = np.linspace(0,(sz[0]-1),sz[0]) and y = np.linspace(0,(sz[1]-1),sz[1])
        ny, nx = self.image.shape
        # In original script `x` and `y` naming is a bit swapped; we kept consistent interpolation ordering in get_MTF_for_pair

        for i in range(n_pairs):
            npeaks = self.npeaks[i]
            # pair endpoints
            x_coords = (self.lpx[i], self.lpx[i + 1])
            y_coords = (self.lpy[i], self.lpy[i + 1])

            mtf_val, f1, pmax, pmin, pcomb, tmpx, tmpy = self._get_MTF_for_pair(x_coords, y_coords, npeaks)
            # store
            per_pair_mtf.append(mtf_val)
            profiles.append(f1)
            pmax_list.append(pmax)
            pmin_list.append(pmin)
            pcomb_list.append(pcomb)
            lp_x_list.append(tmpx)
            lp_y_list.append(tmpy)

        # save collected results
        self.per_pair_mtf   = per_pair_mtf
        self.profiles       = profiles
        self.peaks_max      = pmax_list
        self.peaks_min      = pmin_list
        self.peaks_combined = pcomb_list
        self.lp_x           = lp_x_list
        self.lp_y           = lp_y_list

        # Normalize MTF (original: nMTF = MTF/max(np.array(MTF)))
        mtf_array = np.array(self.per_pair_mtf, dtype=float)
        max_val   = mtf_array.max() if mtf_array.size > 0 else 0.0
        if max_val > 0:
            self.nMTF = mtf_array / max_val
        else:
            self.nMTF = mtf_array.copy()

        # spatial axis same as original: lp = np.linspace(1,len(MTF),len(MTF))/10
        if len(self.per_pair_mtf) > 0:
            self.lp_axis = np.linspace(1, len(self.per_pair_mtf), len(self.per_pair_mtf)) / 10.0
        else:
            self.lp_axis = np.array([])

        # interpolate for requested MTF sample points (0.8, 0.5, 0.3, 0.1) using reversed arrays
        if self.nMTF.size > 0 and self.lp_axis.size > 0:
            targets = (0.8, 0.5, 0.3, 0.1)
            # handle monotonicity & edge cases by using np.interp on reversed arrays like original
            try:
                fMTF = np.interp(targets, self.nMTF[::-1], self.lp_axis[::-1])
                self.mtf_points = {
                    "MTF80": float(fMTF[0]),
                    "MTF50": float(fMTF[1]),
                    "MTF30": float(fMTF[2]),
                    "MTF10": float(fMTF[3]),
                }
            except Exception:
                # fallback to NaNs if interpolation fails
                self.mtf_points = {"MTF80": np.nan, "MTF50": np.nan, "MTF30": np.nan, "MTF10": np.nan}
        else:
            self.mtf_points = {"MTF80": np.nan, "MTF50": np.nan, "MTF30": np.nan, "MTF10": np.nan}

        return self.to_dict()


    # -------------------------
    # JSON output helpers
    # -------------------------
    def to_dict(self):
        """Return JSON-compatible results (includes full nMTF & lp_axis)."""
        return {
            "pixel_spacing_mm" : self.pixel_spacing,
            "image_shape"      : self.image.shape,
            "centers_x"        : self.lpx.tolist() if self.lpx is not None else None,
            "centers_y"        : self.lpy.tolist() if self.lpy is not None else None,
            "per_pair_mtf"     : [float(x) for x in self.per_pair_mtf],
            "MTF10_lp_per_mm"  : self.mtf_points.get("MTF10"),
            "MTF30_lp_per_mm"  : self.mtf_points.get("MTF30"),
            "MTF50_lp_per_mm"  : self.mtf_points.get("MTF50"),
            "MTF80_lp_per_mm"  : self.mtf_points.get("MTF80"),
            "lp_mm"            : self.lp_axis.tolist() if self.lp_axis is not None else None,
            "nmtf"             : self.nMTF.tolist() if self.nMTF is not None else None,
        }


