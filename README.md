# -----------------------------
# File: README.md
# -----------------------------
# Catphan 404 Analysis (Minimal)

A small, PEP 8–compliant Python package for analyzing Catphan 404 CT phantom
slices. Ready for Sphinx documentation generation.

## Example
```python
from catphan404 import load_image, Catphan404Analyzer
img, meta = load_image('catphan_slice.dcm')
ana = Catphan404Analyzer(img, spacing=meta.get('Spacing'))
res = ana.run_all()
print(res)
```

## Requirements
- numpy
- scipy
- pydicom (optional, for DICOM)
- imageio (optional, for TIFF/JPG/PNG)
- scikit-image
- matplotlib

## Documentation
You can generate HTML docs using Sphinx:
```bash
sphinx-quickstart
make html
```

