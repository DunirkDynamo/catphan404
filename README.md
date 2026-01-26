# -----------------------------
# File: README.md
# -----------------------------
# Catphan 404 Analysis

A modular Python package for analyzing Catphan 404 CT phantom DICOM series.
Supports automatic slice selection, 3-slice averaging, automatic rotation detection, and comprehensive QA analysis.

## Features

- **Automatic Rotation Detection**: Detects phantom rotation using air ROI positions in CTP401 module, automatically applies correction to all subsequent modules
- **3-Slice Averaging**: Improves image quality by averaging target slice with neighbors, reducing noise
- **Timestamp-Based Slice Ordering**: Automatically sorts DICOM slices chronologically for correct sequence
- **Robust DICOM Loading**: Recursively searches directories and reads files regardless of extension using `force=True`
- **Multi-Slice Series Support**: Load entire DICOM series with automatic per-module slice selection
- **Modular Architecture**: Run individual QA modules or complete analysis workflows
- **CLI + Programmatic API**: Use via command-line or Python scripts

## Quick Start

**CLI (Recommended):**
```bash
# Open folder selection dialog - saves plots to current directory
catphan404 -m full_analysis --plot

# Display plots interactively
catphan404 -m full_analysis --plot --show-plot

# Or specify folder path and output directory
catphan404 path/to/dicom_folder -m uniformity high_contrast --plot --save-plot results/
```

**Programmatic Usage:**
```python
from catphan404.io import load_dicom_series
from catphan404.analysis import Catphan404Analyzer

# Load DICOM series
series = load_dicom_series('path/to/dicom_folder')

# Create analyzer (automatically handles slice selection)
ana = Catphan404Analyzer(dicom_series=series)

# Run modules - rotation automatically detected in ctp401 and applied to others
ana.run_uniformity()
ana.run_ctp401()  # Detects rotation, stores in ana.results['rotation_angle']
ana.run_high_contrast()  # Automatically uses detected rotation
ana.run_ctp515()  # Automatically uses detected rotation

# Manual rotation override (if needed)
ana.run_ctp401(t_offset=2.5)  # Manually set 2.5° rotation
ana.run_high_contrast(t_offset=2.5)
ana.run_ctp515(angle_offset=2.5)

# Disable automatic rotation detection
ana.run_ctp401(detect_rotation=False, t_offset=0.0)

# Save results
ana.save_results_json('results.json')
```

**Legacy Single-Image Mode:**
```python
from catphan404.io import load_image
from catphan404.analysis import Catphan404Analyzer

img, meta = load_image('slice.dcm')
ana = Catphan404Analyzer(image=img, spacing=meta.get('Spacing'))
ana.run_uniformity()
```

**Using Individual Analyzers Directly:**

You can use any analyzer module independently without `Catphan404Analyzer`:

```python
from catphan404.uniformity import UniformityAnalyzer
from catphan404.io import load_image
import numpy as np

# Load image
img, meta = load_image('test_scans/uniformity.dcm')

# Estimate phantom center
threshold = np.percentile(img, 75)
mask = img > threshold
coords = np.argwhere(mask)
cy, cx = coords.mean(axis=0)

# Get pixel spacing
spacing = meta.get('Spacing', [1.0, 1.0])
pixel_spacing = float(spacing[0])

# Use analyzer directly
analyzer = UniformityAnalyzer(
    image=img,
    center=(cx, cy),
    pixel_spacing=pixel_spacing
)

results = analyzer.analyze()
print(results)
```

All analyzer modules follow the same pattern:
- `UniformityAnalyzer(image, center, pixel_spacing)`
- `HighContrastAnalyzer(image, center, pixel_spacing)`
- `AnalyzerCTP401(image, center, pixel_spacing)`
- `AnalyzerCTP515(image, center, pixel_spacing)`

## Requirements
- numpy
- scipy
- pydicom (for DICOM files)
- imageio (for TIFF/JPG/PNG formats)
- scikit-image (for image processing)
- matplotlib (for plotting)

## Documentation
You can generate HTML docs using Sphinx:
```bash
sphinx-quickstart
make html
```

