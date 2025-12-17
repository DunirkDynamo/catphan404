<!-- Copilot / agent instructions for the Catphan404 repository -->
# Catphan404 — Agent Instructions

This file contains concise, actionable notes to help an AI coding agent work productively in this repository.

Overview
- Purpose: a small, modular Python toolkit to analyze Catphan 404 CT phantom single-slice images.
- Key entry points: `Catphan404.Catphan404Analyzer` (central orchestrator), CLI `catphan404` (`catphan404.cli:main`), and plotting helpers under `Catphan404/plots/plotters.py`.

Important files & directories
- `Catphan404/analysis.py` — `Catphan404Analyzer` exposes `run_<module>()` methods (e.g. `run_uniformity`, `run_high_contrast`, `run_ctp401`) and `run_all()`.
- `Catphan404/io.py` — `load_image(path)` handles DICOM (requires `pydicom`) and other images (requires `imageio`). It returns `(image_array, metadata_dict)`.
- `Catphan404/*_analyzer.py` — individual analyzer classes (e.g. `UniformityAnalyzer`, `HighContrastAnalyzer`, `AnalyzerCTP401`) implement `analyze()` and usually populate `self.results`.
- `Catphan404/plots/plotters.py` — plotter classes accept analyzer objects and create `matplotlib` `Figure` objects (return `fig`).
- `test_scans/` — contains sample DICOMs (`linearity.dcm`, `linepairs.dcm`, `uniformity.dcm`) useful for manual testing.
- `pyproject.toml` / `requirements.txt` — lists runtime dependencies (`numpy`, `scipy`, `pydicom`, `imageio`, `matplotlib`, `scikit-image`).

Architecture & conventions (what to know)
- Single-slice focused: the package assumes a single CT slice per run (images are 2D arrays).
- Central analyzer pattern: `Catphan404Analyzer` is the coordinator. It:
  - stores raw image in `self.image` and `self.spacing` (pixel spacing)
  - provides `run_<module>()` methods which call the corresponding analyzer classes
  - stores results in `self.results` (a flat dict) and keeps analyzer objects for plotting on underscored attributes: `_uniformity_analyzer`, `_high_contrast_analyzer`, `_ctp401_analyzer`.
- Analyzer contract:
  - Typical method: `.analyze()` -> returns a JSON-serializable `dict` and sets `self.results`.
  - Many analyzers set derived attributes used by plotters (e.g. `.image`, `.center`, `.pixel_spacing`, `.lp_axis`, `.nMTF`). Check the analyzer implementation before using in plotters.
- Plotter contract:
  - Plotter classes in `Catphan404/plots/plotters.py` expect a completed analyzer instance (some call `.analyze()` again inside constructors).
  - They return a `matplotlib.Figure` and do not save files unless code outside calls `fig.savefig()`.

Common pitfalls & repo-specific gotchas
- Inconsistent coordinate ordering: some helpers return `(row, col)` while others treat `center` as `(x, y)` or `(cx, cy)`. Before manipulating coordinates, inspect the specific analyzer's code.
- I/O optional imports: `io.py` imports `pydicom` and `imageio` inside try/except; missing packages raise `ImportError` when a DICOM or other image is requested. When writing tests or running the CLI, ensure the environment has the required packages.
- Plotters sometimes call `analyzer.analyze()` inside their constructor — be careful to avoid double-analysis or unwanted side-effects.
- Plotter expectations: some plotters expect analyzer attributes or methods that might be missing (e.g. `HighContrastPlotter` expects `to_dict()`, `lpx`, `lp_x`, `lp_y`, `lp_axis`, `nMTF`). Confirm the analyzer exposes those.

Developer workflows / commands
- Quick manual run (requires dependencies installed):
  - From repo root: `python -m Catphan404.cli path/to/slice.dcm -m all --plot --save-plot outdir`
  - If package is installed (entry point): `catphan404 path/to/slice.dcm -m ctp401 --out ctp401.json`
- Programmatic use (example):
  ```python
  from catphan404.io import load_image
  from catphan404.analysis import Catphan404Analyzer

  img, meta = load_image('test_scans/uniformity.dcm')
  ana = Catphan404Analyzer(img, spacing=meta.get('Spacing'))
  ana.run_uniformity()
  ana.save_results_json('uniformity.json')
  ```
- Use `test_scans/` images for quick visual/manual checks.

Code review hints for agents
- When adding/modifying analyzers, keep the `.analyze()` return value JSON-serializable and place final, user-facing values into `self.results`.
- Maintain `Catphan404Analyzer`'s pattern of exposing per-module analyzers on underscored attributes if plotters rely on them.
- Avoid changing the public API signatures (`Catphan404Analyzer.__init__`, `run_<module>`, `load_image`) without updating the CLI mapping in `cli.py`.

When to run tests / manual checks
- There are no unit tests in the repository; rely on `test_scans/` for manual verification.
- Validate plotting visually — plotters return figures; use `fig.savefig()` in CI/dev scripts if adding automated visual checks.

Where to look next (for new agents)
- `Catphan404/*_analyzer.py` for algorithm details and I/O expectations.
- `Catphan404/plots/plotters.py` for visualization contracts and attributes required from analyzers.
- `Catphan404/cli.py` for how modules are discovered, how plots are saved, and the JSON output flow.

If anything above is unclear or you want more detail (e.g., automatic tests, stricter API contracts, or examples for a specific analyzer), say which area to expand and I will iterate.
