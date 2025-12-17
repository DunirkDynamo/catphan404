# -----------------------------
# File: catphan404/io.py
# -----------------------------
"""Simple image I/O utilities.

Supports single-slice DICOM images via ``pydicom`` and common image formats
(TIFF, PNG, JPG) via ``imageio``.
"""
from typing import Tuple, Optional
import numpy as np

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import imageio
except Exception:
    imageio = None


def _read_dicom(path: str) -> Tuple[np.ndarray, dict]:
    """Read a DICOM image file.

    Args:
        path (str): Path to the DICOM file.

    Returns:
        Tuple[np.ndarray, dict]: The pixel array and extracted metadata.

    Raises:
        ImportError: If ``pydicom`` is not installed.
    """
    if pydicom is None:
        raise ImportError("pydicom required to read DICOM files. Install with 'pip install pydicom'.")
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(float)
    #if hasattr(ds, "RescaleIntercept") and hasattr(ds, "RescaleSlope"):
     #   arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    meta = {
        "Spacing": getattr(ds, 'PixelSpacing', None),
        "SliceThickness": getattr(ds, 'SliceThickness', None),
        "Modality": getattr(ds, 'Modality', None)
    }
    return arr, meta


def _read_imageio(path: str) -> Tuple[np.ndarray, dict]:
    """Read a non-DICOM image using ``imageio``.

    Args:
        path (str): Path to the image file.

    Returns:
        Tuple[np.ndarray, dict]: The image array and an empty metadata dictionary.

    Raises:
        ImportError: If ``imageio`` is not installed.
    """
    if imageio is None:
        raise ImportError("imageio required to read non-DICOM images. Install with 'pip install imageio'.")
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    arr = arr.astype(float)
    return arr, {}


def load_image(path: str) -> Tuple[np.ndarray, dict]:
    """Load an image file (DICOM or standard image format).

    Args:
        path (str): File path to load.

    Returns:
        Tuple[np.ndarray, dict]: The image and metadata dictionary.
    """
    lower = path.lower()
    if lower.endswith('.dcm') or lower.endswith('.dicom'):
        return _read_dicom(path)
    

    ds = pydicom.dcmread(path, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    arr = ds.pixel_array.astype(float)
    #if hasattr(ds, "RescaleIntercept") and hasattr(ds, "RescaleSlope"):
     #   arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    meta = {
        "Spacing": getattr(ds, 'PixelSpacing', None),
        "SliceThickness": getattr(ds, 'SliceThickness', None),
        "Modality": getattr(ds, 'Modality', None)
    }
    return arr, meta
    #return _read_imageio(path)