# -----------------------------
# File: catphan404/uniformity.py
# -----------------------------
"""Uniformity and CT-number module for Catphan 404."""
from typing import Dict, Tuple
import numpy as np
from .utils.image import circular_roi_mask


def uniformity_analysis(image: np.ndarray, center: Tuple[float, float], radius: float) -> Dict:
    """Compute uniformity ROIs and statistics.

    Args:
        image (np.ndarray): Input image.
        center (Tuple[float, float]): (x, y) center of phantom.
        radius (float): Radius for peripheral ROIs.

    Returns:
        Dict: Results including per-ROI mean/std and uniformity metrics.
    """
    masks = {
        'center': circular_roi_mask(image.shape, center, radius * 0.5),
        'right': circular_roi_mask(image.shape, (center[0] + radius * 0.7, center[1]), radius * 0.3),
        'left': circular_roi_mask(image.shape, (center[0] - radius * 0.7, center[1]), radius * 0.3),
        'top': circular_roi_mask(image.shape, (center[0], center[1] - radius * 0.7), radius * 0.3),
        'bottom': circular_roi_mask(image.shape, (center[0], center[1] + radius * 0.7), radius * 0.3),
    }
    stats = {}
    for k, m in masks.items():
        vals = image[m]
        stats[k] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals)), 'count': int(vals.size)}
    center_mean = stats['center']['mean']
    deviations = {k: abs(v['mean'] - center_mean) for k, v in stats.items()}
    return {'rois': stats, 'deviations': deviations, 'max_deviation': max(deviations.values())}