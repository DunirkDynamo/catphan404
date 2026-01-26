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
    """
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
    """Load all DICOM files from a folder (recursively searches subdirectories).
    
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
    
    # Recursively find all files, skip obvious non-DICOM files
    # DICOM files can have any extension or none
    import os
    dicom_files = []
    for root, _, file_list in os.walk(folder_path):
        for filename in file_list:
            # Skip known non-DICOM files
            if any(x in filename.lower() for x in ['dir', '.txt', '.json', '.md']):
                continue
            dicom_files.append(Path(root, filename))
    
    print(f"Found {len(dicom_files)} potential files to check")
    
    # Load each DICOM file
    series = []
    failed_count = 0
    for file_path in dicom_files:
        try:
            # Use force=True for better compatibility with non-standard DICOM
            ds = pydicom.dcmread(str(file_path), force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            arr = ds.pixel_array.astype(float)
            
            # Extract acquisition time for proper sorting
            acquisition_time = getattr(ds, 'AcquisitionTime', None)
            acquisition_datetime = getattr(ds, 'AcquisitionDateTime', None)
            series_time = getattr(ds, 'SeriesTime', None)
            content_time = getattr(ds, 'ContentTime', None)
            
            # Use the first available timestamp (in priority order)
            timestamp = acquisition_datetime or acquisition_time or series_time or content_time or '000000.000000'
            
            # Extract metadata
            meta = {
                "Spacing": getattr(ds, 'PixelSpacing', None),
                "SliceThickness": getattr(ds, 'SliceThickness', None),
                "Modality": getattr(ds, 'Modality', None),
                "InstanceNumber": getattr(ds, 'InstanceNumber', None),
                "SliceLocation": getattr(ds, 'SliceLocation', None),
                "AcquisitionTime": acquisition_time,
                "AcquisitionDateTime": acquisition_datetime
            }
            
            series.append({
                'image': arr,
                'metadata': meta,
                'path': str(file_path),
                'instance_number': meta.get('InstanceNumber', 0),
                'timestamp': timestamp
            })
        except Exception as e:
            # Skip files that can't be read as DICOM
            failed_count += 1
            # Only print first few errors to avoid spam
            if failed_count <= 3:
                print(f"⚠️  Could not read {file_path.name} as DICOM: {e}")
            continue
    
    if failed_count > 3:
        print(f"⚠️  ({failed_count - 3} more files failed to load)")
    
    if not series:
        raise ValueError(f"No valid DICOM files found in {folder_path}. Checked {len(dicom_files)} files, all failed.")
    
    # Sort by timestamp (reverse chronological order)
    series.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Print slice information for debugging
    print(f"\n📊 Slice order (by acquisition time):")
    for idx, slice_data in enumerate(series):
        timestamp = slice_data['timestamp']
        path_name = Path(slice_data['path']).name
        print(f"  [{idx}] {path_name} - Time: {timestamp}")
    
    return series