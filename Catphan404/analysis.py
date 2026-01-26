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
    linearity, low-contrast) on DICOM series. Each module is run via
    a dedicated run_* method that selects the appropriate slice, initializes
    the analyzer, executes analysis, and stores results.

    Attributes:
        dicom_series (list): List of DICOM slice dictionaries from load_dicom_series().
        image (np.ndarray): Currently selected 2D CT image (for backwards compatibility).
        spacing (Optional[Tuple[float, float]]): Pixel spacing (x, y) in mm.
        results (dict): Dictionary storing JSON-compatible results from each module.
    """

    def __init__(self, dicom_series=None, image: np.ndarray = None, spacing: Optional[Tuple[float, float]] = None, use_slice_averaging: bool = False):
        """Initialize analyzer with DICOM series or single image.
        
        Args:
            dicom_series (list, optional): List of DICOM dictionaries from load_dicom_series().
            image (np.ndarray, optional): Single 2D image (for backwards compatibility).
            spacing (Optional[Tuple[float, float]], optional): Pixel spacing in mm.
            use_slice_averaging (bool, optional): Enable 3-slice averaging. Default False.
        
        Note:
            Either dicom_series or image must be provided.
            Slice averaging requires dicom_series to be provided.
        """
        if dicom_series is None and image is None:
            raise ValueError("Either dicom_series or image must be provided")
        
        # Store the DICOM series
        self.dicom_series = dicom_series
        
        # Determine if we should use slice averaging
        # Only enabled if explicitly requested AND DICOM series is available
        self.use_slice_averaging = use_slice_averaging and (dicom_series is not None)
        
        # For backwards compatibility, support single image input
        if image is not None:
            self.image = np.array(image, dtype=float)
            self.spacing = (float(spacing[0]), float(spacing[1])) if spacing else None
        else:
            # Will be set when a specific slice is selected
            self.image = None
            self.spacing = None
        
        # Slice indices for each analysis module (hardcoded defaults)
        # Only used when use_slice_averaging is True
        self.uniformity_slice_index = 6
        self.high_contrast_slice_index = 2
        self.ctp401_slice_index = 0
        self.ctp515_slice_index = 4
        
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
        # Get image from DICOM series if available
        if self.dicom_series is not None:
            if self.use_slice_averaging:
                self.image, self.spacing = self._average_slices(self.uniformity_slice_index)
            else:
                # Select single slice without averaging
                slice_data = self.dicom_series[self.uniformity_slice_index]
                self.image = slice_data['image']
                spacing_raw = slice_data['metadata'].get('Spacing')
                self.spacing = (float(spacing_raw[0]), float(spacing_raw[1])) if spacing_raw else None
        
        # Estimate phantom center from the image
        cy, cx = self._estimate_center(self.image)

        # Get pixel spacing (default to 1.0 if not available)
        spacing = self.spacing[0] if self.spacing else 1.0

        # Initialize the uniformity analyzer with image and center
        analyzer = UniformityAnalyzer(self.image, (cx, cy), spacing)

        # Run analysis and store results
        self.results['uniformity'] = analyzer.analyze()

        # Also store center for reference
        self.results['center'] = (float(cx), float(cy))

        # Store the analyzer:
        self._uniformity_analyzer = analyzer


    # ------------------ High Contrast Module (Line pairs) ----------------
    def run_high_contrast(self, t_offset: float = None):
        """
        Run high-contrast line pair analysis (CTP528 module).

        Analyzes spatial resolution by measuring MTF from line pair patterns.
        Uses center from uniformity analysis if available, otherwise estimates it.
        Will use rotation angle from CTP401 if available.

        Args:
            t_offset (float, optional): Manual rotation offset in degrees.
                                       If None, uses rotation_angle from results.

        Populates:
            self.results['high_contrast']: MTF curve data and resolution metrics.
        """
        # Get image from DICOM series if available
        if self.dicom_series is not None:
            if self.use_slice_averaging:
                self.image, self.spacing = self._average_slices(self.high_contrast_slice_index)
            else:
                # Select single slice without averaging
                slice_data = self.dicom_series[self.high_contrast_slice_index]
                self.image = slice_data['image']
                spacing_raw = slice_data['metadata'].get('Spacing')
                self.spacing = (float(spacing_raw[0]), float(spacing_raw[1])) if spacing_raw else None

        # Use center from uniformity analysis if available
        center = self.results.get('center', None)
        if center is None:
            cy, cx = self._estimate_center(self.image)
            center = (cx, cy)  # Store as (x, y) to match uniformity format

        # Use detected rotation angle if available and t_offset not provided
        if t_offset is None and 'rotation_angle' in self.results:
            t_offset = self.results['rotation_angle']
            print(f"📐 Using detected rotation angle: {t_offset:.2f}° for high contrast")
        elif t_offset is None:
            t_offset = 0

        spacing = self.spacing[0] if self.spacing else 1.0

        analyzer = HighContrastAnalyzer(
            image=self.image,
            center=center,
            pixel_spacing=spacing,
            t_offset_deg=t_offset
        )


        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['high_contrast'] = res

        # Store the analyzer:
        self._high_contrast_analyzer = analyzer

    # --------------  Linearity Module (HU material inserts) --------------
    def run_ctp401(self, t_offset: float = None, detect_rotation: bool = True):
        """
        Run linearity/scaling analysis (CTP401 module).

        Analyzes material insert ROIs to measure HU values for different
        materials (LDPE, Air, Teflon, Acrylic) and derives a calibration scale.
        Can automatically detect phantom rotation using air ROI positions.

        Args:
            t_offset (float, optional): Rotational offset in degrees for ROI positioning.
                                       If None and detect_rotation=True, will auto-detect.
            detect_rotation (bool): Whether to automatically detect rotation (default True).

        Populates:
            self.results['ctp401']: Material ROI statistics and calibration data.
            self.results['rotation_angle']: Detected rotation angle (if detect_rotation=True).
        """
        # Get image from DICOM series if available
        if self.dicom_series is not None:
            if self.use_slice_averaging:
                self.image, self.spacing = self._average_slices(self.ctp401_slice_index)
            else:
                # Select single slice without averaging
                slice_data = self.dicom_series[self.ctp401_slice_index]
                self.image = slice_data['image']
                spacing_raw = slice_data['metadata'].get('Spacing')
                self.spacing = (float(spacing_raw[0]), float(spacing_raw[1])) if spacing_raw else None

        # Use center from uniformity analysis if available
        center = self.results.get('center', None)
        if center is None:
            cy, cx = self._estimate_center(self.image)
            center = (cx, cy)  # Store as (x, y) to match uniformity format

        spacing = self.spacing[0] if self.spacing else 1.0

        analyzer = AnalyzerCTP401(
            image=self.image,
            center=center,
            pixel_spacing=spacing
        )

        # Detect rotation if requested and t_offset not provided
        if t_offset is None and detect_rotation:
            rotation_angle, top_air, bottom_air = analyzer.detect_rotation()
            t_offset = rotation_angle
            self.results['rotation_angle'] = float(rotation_angle)
            print(f"🔄 Detected rotation: {rotation_angle:.2f}°")
        elif t_offset is not None:
            self.results['rotation_angle'] = float(t_offset)

        # Run analysis with detected or provided rotation offset
        res = analyzer.analyze(t_offset=t_offset if t_offset is not None else 0)
        self.results['ctp401'] = res

        # Store the analyzer:
        self._ctp401_analyzer = analyzer
    # ------------------ As yet undeveloped modules ---------------- 

    def run_ctp515(self, crop_x=0, crop_y=0, angle_offset: float = None):
        """
        Run low-contrast detectability analysis (CTP515 module).

        Detects low-contrast inserts of varying diameters and computes CNR
        and contrast values to assess detectability. Uses geometric center
        of potentially cropped image. Will use rotation angle from CTP401 if available.

        Args:
            crop_x (int): Number of pixels to crop from left and right edges.
            crop_y (int): Number of pixels to crop from top and bottom edges.
            angle_offset (float, optional): Manual angle offset in degrees.
                                          If None, uses rotation_angle from results.

        Populates:
            self.results['ctp515']: Low-contrast ROI statistics, CNR, and contrast values.
        """
        # Get image from DICOM series if available
        if self.dicom_series is not None:
            if self.use_slice_averaging:
                self.image, self.spacing = self._average_slices(self.ctp515_slice_index)
            else:
                # Select single slice without averaging
                slice_data = self.dicom_series[self.ctp515_slice_index]
                self.image = slice_data['image']
                spacing_raw = slice_data['metadata'].get('Spacing')
                self.spacing = (float(spacing_raw[0]), float(spacing_raw[1])) if spacing_raw else None
        
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

        # Use detected rotation angle if available and angle_offset not provided
        if angle_offset is None:
            angle_offset = self.results.get('rotation_angle', 0.0)
            if 'rotation_angle' in self.results:
                print(f"📐 Using detected rotation angle: {angle_offset:.2f}° for CTP515")
        else:
            angle_offset = angle_offset

        spacing  = self.spacing[0] if self.spacing else 1.0
        analyzer = AnalyzerCTP515(cropped_image, center, spacing, angle_offset=angle_offset)
        
        # Store the results of the analysis:
        res = analyzer.analyze()
        self.results['ctp515'] = res

        # Store the analyzer:
        self._ctp515_analyzer = analyzer


    # ------------------ Helper functions ------------------
    def _average_slices(self, slice_index: int) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        Average a slice with its two neighboring slices.
        
        Averages the specified slice with the slices immediately before and after it
        (e.g., if slice_index=50, averages slices 49, 50, 51). Handles edge cases
        by only averaging available slices.
        
        Args:
            slice_index (int): Index of the center slice to average.
        
        Returns:
            Tuple[np.ndarray, Optional[Tuple[float, float]]]: 
                - Averaged image array
                - Pixel spacing from the center slice metadata
        
        Raises:
            ValueError: If dicom_series is not available or slice_index is out of range.
        """
        if self.dicom_series is None:
            raise ValueError("Cannot average slices: no DICOM series loaded")
        
        if slice_index < 0 or slice_index >= len(self.dicom_series):
            raise ValueError(f"Slice index {slice_index} out of range (0-{len(self.dicom_series)-1})")
        
        # Determine which slices to average
        start_idx = max(0, slice_index - 1)
        end_idx = min(len(self.dicom_series) - 1, slice_index + 1)
        
        # Collect images to average
        images_to_average = []
        for idx in range(start_idx, end_idx + 1):
            images_to_average.append(self.dicom_series[idx]['image'])
        
        # Average the images
        averaged_image = np.mean(images_to_average, axis=0)
        
        # Get spacing from the center slice
        center_metadata = self.dicom_series[slice_index]['metadata']
        spacing_raw = center_metadata.get('Spacing')
        spacing = (float(spacing_raw[0]), float(spacing_raw[1])) if spacing_raw else None
        
        return averaged_image, spacing
    
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
        thresh = np.percentile(sm, 75)
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
