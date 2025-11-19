# -----------------------------
# File: catphan404/__init__.py
# -----------------------------
"""Minimal Catphan 404 analysis package.

This package provides core functionality to analyze Catphan 404 CT phantom
slices. It includes tools for image loading and basic quantitative analysis
of uniformity and CT number inserts. Designed for simplicity and full
PEP 8/257 compliance, compatible with Sphinx autodoc.
"""

from .analysis import Catphan404Analyzer
from .io import load_image

__all__ = ["Catphan404Analyzer", "load_image"]