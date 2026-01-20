# -----------------------------
# File: catphan404/io.py
# -----------------------------
"""Simple image I/O utilities.

Supports single-slice DICOM images via ``pydicom`` and common image formats
(TIFF, PNG, JPG) via ``imageio``.
"""
from typing import Tuple, Optional, List, Dict
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

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
    \"\"\"
    Load an image file (DICOM or standard image format).

    Automatically detects format based on file extension. DICOM files are
    read with pydicom (with metadata extraction), while other formats use
    imageio. Handles both explicit .dcm/.dicom extensions and DICOM files
    without standard extensions.

    Args:
        path (str): File path to load.

    Returns:
        Tuple[np.ndarray, dict]: Image array and metadata dictionary.
                                Metadata includes 'Spacing', 'SliceThickness',
                                and 'Modality' for DICOM files, empty dict otherwise.

    Raises:
        ImportError: If required package (pydicom or imageio) is not installed.
    \"\"\"
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


def select_dicom_folder() -> Optional[str]:
    """Open a folder selection dialog to choose a DICOM directory.
    
    Returns:
        Optional[str]: Path to the selected folder, or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    folder_path = filedialog.askdirectory(
        title="Select DICOM Folder",
        mustexist=True
    )
    
    root.destroy()
    return folder_path if folder_path else None


def load_dicom_series(folder_path: str) -> List[Dict[str, any]]:
    """Load all DICOM files from a folder.
    
    Args:
        folder_path (str): Path to folder containing DICOM files.
    
    Returns:
        List[Dict]: List of dictionaries, each containing:
            - 'image': np.ndarray pixel array
            - 'metadata': dict with Spacing, SliceThickness, Modality
            - 'path': str path to the DICOM file
            - 'instance_number': int slice/instance number (if available)
    
    Raises:
        ImportError: If pydicom is not installed.
        ValueError: If no valid DICOM files found in folder.
    """
    if pydicom is None:
        raise ImportError("pydicom required to read DICOM files. Install with 'pip install pydicom'.")
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find all potential DICOM files (common extensions and no extension)
    dicom_files = []
    for ext in ['*.dcm', '*.dicom', '*.DCM', '*.DICOM']:
        dicom_files.extend(folder.glob(ext))
    
    # Also try files without extension (common for DICOM)
    for file in folder.iterdir():
        if file.is_file() and file.suffix == '':
            dicom_files.append(file)
    
    # Remove duplicates
    dicom_files = list(set(dicom_files))
    
    # Load each DICOM file
    series = []
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(str(file_path))
            arr = ds.pixel_array.astype(float)
            
            # Extract metadata
            meta = {
                "Spacing": getattr(ds, 'PixelSpacing', None),
                "SliceThickness": getattr(ds, 'SliceThickness', None),
                "Modality": getattr(ds, 'Modality', None),
                "InstanceNumber": getattr(ds, 'InstanceNumber', None),
                "SliceLocation": getattr(ds, 'SliceLocation', None)
            }
            
            series.append({
                'image': arr,
                'metadata': meta,
                'path': str(file_path),
                'instance_number': meta.get('InstanceNumber', 0)
            })
        except Exception as e:
            # Skip files that can't be read as DICOM
            print(f"Warning: Could not read {file_path} as DICOM: {e}")
            continue
    
    if not series:
        raise ValueError(f"No valid DICOM files found in {folder_path}")
    
    # Sort by instance number
    series.sort(key=lambda x: x['instance_number'] or 0)
    
    return series