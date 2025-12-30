# Catphan404 CLI Usage Guide

This guide provides examples and explanations for using the Catphan404 command-line interface.

## Installation

Ensure the package and its dependencies are installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

```bash
python -m Catphan404.cli <image_path> -m <module> [options]
```

Or if installed as a package:

```bash
catphan404 <image_path> -m <module> [options]
```

## Available Modules

- `uniformity`      - Uniformity analysis (CTP486 module)
- `high_contrast`   - High-contrast resolution / line pairs (CTP528 module)
- `ctp401`          - HU linearity / material analysis (CTP401 module)
- `ctp515`          - Low-contrast detectability (CTP515 module)
- `slice_thickness` - Slice thickness (FWHM) analysis
- `geometry`        - Geometric accuracy analysis

**Note:** Each image should be analyzed by only one module corresponding to the specific phantom slice.

## Common Options

- `-m, --modules` - **Required.** Specify which module(s) to run
- `--out, -o`     - JSON output file path (default: `<module>.json`)
- `--plot`        - Generate diagnostic plots
- `--save-plot`   - Directory or file path to save plots as PNG
- `--no-save`     - Skip saving JSON results (useful for quick visual checks)

## Example Commands

### Uniformity Analysis

Analyze uniformity with plotting and save results:

```bash
python -m Catphan404.cli Catphan404/test_scans/uniformity.dcm -m uniformity --plot --save-plot uniformity_results.png
```

Output:
- JSON file: `uniformity.json`
- Plot saved: `uniformity_results.png`

### High Contrast (Line Pairs)

```bash
python -m Catphan404.cli Catphan404/test_scans/linepairs.dcm -m high_contrast --plot --save-plot high_contrast.png
```

### CTP401 Linearity

```bash
python -m Catphan404.cli Catphan404/test_scans/linearity.dcm -m ctp401 --plot --save-plot ctp401_results.png --out ctp401_analysis.json
```

### CTP515 Low-Contrast Detectability

```bash
python -m Catphan404.cli Catphan404/test_scans/low_contrast.647055 -m ctp515 --plot --save-plot ctp515_results.png
```

Output includes:
- Image with color-coded ROI overlays
- CNR and Contrast vs. ROI Diameter plots
- Statistics table with ROI measurements

### Quick Visual Check (No Save)

View plots without saving JSON results:

```bash
python -m Catphan404.cli path/to/image.dcm -m uniformity --plot --no-save
```

### Save Plots to Directory

Save plots to a specific directory:

```bash
python -m Catphan404.cli image.dcm -m ctp515 --plot --save-plot output_directory/
```

Plot will be saved as: `output_directory/ctp515.png`

### Custom Output File

Specify custom JSON output filename:

```bash
python -m Catphan404.cli image.dcm -m uniformity --out my_results.json
```

## Supported Image Formats

- **DICOM** (`.dcm`) - Primary format, requires `pydicom`
- **PNG, TIFF, JPG** - Supported via `imageio`
- **Raw binary** - Any format loadable by `imageio`

**Note:** Non-DICOM formats may not contain pixel spacing metadata, which will default to 1.0 mm/pixel.

## Programmatic Usage

For scripting or batch processing:

```python
from Catphan404.io import load_image
from Catphan404.analysis import Catphan404Analyzer

# Load image
img, meta = load_image('path/to/image.dcm')

# Create analyzer
analyzer = Catphan404Analyzer(img, spacing=meta.get('Spacing'))

# Run specific module
analyzer.run_ctp515()

# Save results
analyzer.save_results_json('output.json')

# Generate plot
from Catphan404.plots.plotters import CTP515Plotter
plotter = CTP515Plotter(analyzer._ctp515_analyzer)
fig = plotter.plot()
fig.savefig('plot.png')
```

## Workflow Example

Complete analysis workflow for multiple modules:

```bash
# 1. Uniformity slice
python -m Catphan404.cli scans/uniformity.dcm -m uniformity --plot --save-plot results/uniformity.png

# 2. High contrast slice
python -m Catphan404.cli scans/linepairs.dcm -m high_contrast --plot --save-plot results/high_contrast.png

# 3. Linearity slice
python -m Catphan404.cli scans/linearity.dcm -m ctp401 --plot --save-plot results/ctp401.png

# 4. Low contrast slice
python -m Catphan404.cli scans/low_contrast.dcm -m ctp515 --plot --save-plot results/ctp515.png
```

## Output Files

### JSON Results

Each module produces JSON output with module-specific metrics:

**uniformity.json:**
```json
{
  "uniformity": {
    "centre": {"mean": 5.2, "std": 3.1},
    "north": {"mean": 5.5, "std": 3.0},
    ...
    "uniformity": 1.23
  },
  "center": [256.0, 256.0]
}
```

**ctp515.json:**
```json
{
  "ctp515": {
    "n_detected": 6,
    "blobs": {
      "roi_15mm": {
        "x": 250, "y": 180,
        "mean": -35.2, "std": 4.1,
        "cnr": 8.5, "contrast": -2.3
      },
      ...
    }
  }
}
```

### Plot Files

PNG images showing:
- **Uniformity:** Image with ROIs, histograms, error bars, boxplots, profiles
- **High Contrast:** Image with segments, MTF curve, stacked line profiles
- **CTP401:** Image with ROIs, per-ROI histograms, heatmaps
- **CTP515:** Image with color-coded ROIs, CNR/Contrast plots, statistics table

## Troubleshooting

### Module Not Found Error

If you see `ModuleNotFoundError`, ensure you're running from the repository root:

```bash
cd c:\GitHub\catphan404
python -m Catphan404.cli ...
```

### Missing Dependencies

Install required packages:

```bash
pip install numpy scipy matplotlib pydicom imageio scikit-image
```

### DICOM Loading Issues

If DICOM files fail to load, ensure `pydicom` is installed:

```bash
pip install pydicom
```

### Wrong Module for Image

Each phantom slice should use its corresponding module:
- Uniformity slice → `uniformity`
- Line pair slice → `high_contrast`
- Material inserts → `ctp401`
- Low-contrast discs → `ctp515`

Running the wrong module will produce incorrect or meaningless results.

## Tips

1. **Always use `--plot`** when first analyzing a new dataset to visually verify results
2. **Check JSON output** for quantitative metrics and QA records
3. **Use `--no-save`** for quick visual checks during setup
4. **Organize output directories** by scan date or phantom serial number
5. **Keep original DICOM files** as they contain critical metadata (pixel spacing, etc.)

## Getting Help

View CLI help:

```bash
python -m Catphan404.cli --help
```

For more information, see:
- `README.md` - Package overview
- `.github/copilot-instructions.md` - Developer guidance
- Source code in `Catphan404/` directory
