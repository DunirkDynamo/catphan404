# -----------------------------
# File: catphan404/low_contrast.py
# -----------------------------
"""Low-contrast detectability analysis for Catphan 404."""
from typing import Dict, Tuple
import numpy as np
from .utils.image import detect_blobs, circular_roi_mask


def low_contrast_analysis(image: np.ndarray, center: Tuple[float, float]) -> Dict:
    """Detect low-contrast inserts and compute CNR for each.

    Args:
        image (np.ndarray): Input image.
        center (Tuple[float, float]): Phantom center (x, y).

    Returns:
        Dict: Detected insert statistics and CNR values.
    """
    # search local region around center
    ny, nx = image.shape
    cy, cx = int(center[1]), int(center[0])
    search = image[max(0, cy - ny // 3):min(ny, cy + ny // 3), max(0, cx - nx // 3):min(nx, cx + nx // 3)]
    blobs = detect_blobs(search, min_sigma=1.5, max_sigma=6, threshold=np.std(search) * 0.5)
    results = {}
    for i, b in enumerate(blobs):
        y, x, r = b
        # translate to full image coordinates
        y_full = int(y) + max(0, cy - ny // 3)
        x_full = int(x) + max(0, cx - nx // 3)
        mask = circular_roi_mask(image.shape, (x_full, y_full), r)
        vals = image[mask]
        # background annulus
        mask_bg = circular_roi_mask(image.shape, (x_full, y_full), r * 2) & ~mask
        bg_vals = image[mask_bg]
        if vals.size < 10 or bg_vals.size < 10:
            continue
        mean_signal = float(np.mean(vals))
        mean_bg = float(np.mean(bg_vals))
        std_bg = float(np.std(bg_vals))
        cnr = abs(mean_signal - mean_bg) / (std_bg + 1e-8)
        results[f'blob_{i+1}'] = {'x': int(x_full), 'y': int(y_full), 'r': float(r), 'mean': mean_signal, 'bg_mean': mean_bg, 'bg_std': std_bg, 'cnr': float(cnr)}
    return {'n_detected': len(results), 'blobs': results}