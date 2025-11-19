# -----------------------------
# File: catphan404/utils/image.py
# -----------------------------
"""Image utilities used by multiple Catphan modules.

This file contains helpers for ROI extraction, filtering, edge detection and
basic blob detection. It intentionally uses only numpy/scipy/skimage to keep
dependencies standard for image analysis.
"""
from typing import Tuple
import numpy as np
from scipy import ndimage
from skimage import feature, filters, morphology


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to an image.

    Args:
        image (np.ndarray): Input image.
        sigma (float): Gaussian sigma.

    Returns:
        np.ndarray: Blurred image.
    """
    return ndimage.gaussian_filter(image, sigma=sigma)


def find_edge_region(image: np.ndarray, width: int = 50) -> Tuple[int, int, int, int]:
    """Find a prominent edge region suitable for ESF/MTF calculation.

    The function computes a gradient magnitude and finds the largest connected
    edge cluster, then returns a bounding box padded by ``width`` pixels.

    Args:
        image (np.ndarray): Input image.
        width (int): Padding width for the returned bounding box.

    Returns:
        Tuple[int, int, int, int]: (y0, y1, x0, x1) bounding box.
    """
    grad = filters.sobel(image)
    edges = grad > np.percentile(grad, 85)
    edges = morphology.remove_small_objects(edges, min_size=64)
    labels, n = ndimage.label(edges)
    if n == 0:
        # fallback to central strip
        ny, nx = image.shape
        y0 = ny // 2 - width
        y1 = ny // 2 + width
        x0 = nx // 4
        x1 = 3 * nx // 4
        return max(0, y0), min(ny, y1), max(0, x0), min(nx, x1)
    # pick largest label
    counts = np.bincount(labels.flat)
    largest = np.argmax(counts[1:]) + 1
    ys, xs = np.where(labels == largest)
    y0 = ys.min() - width
    y1 = ys.max() + width
    x0 = xs.min() - width
    x1 = xs.max() + width
    ny, nx = image.shape
    return max(0, y0), min(ny, y1), max(0, x0), min(nx, x1)


def circular_roi_mask(shape: Tuple[int, int], center: Tuple[float, float], radius: float) -> np.ndarray:
    """Return a boolean mask with a circle filled in.

    Args:
        shape (Tuple[int, int]): Image shape as (rows, cols).
        center (Tuple[float, float]): (x, y) center coordinates.
        radius (float): Radius in pixels.

    Returns:
        np.ndarray: Boolean mask.
    """
    ny, nx = shape
    Y, X = np.ogrid[:ny, :nx]
    cx, cy = center
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return dist <= radius


def detect_blobs(image: np.ndarray, min_sigma: float = 2, max_sigma: float = 10, threshold: float = 0.02):
    """Detect blobs using Laplacian of Gaussian detector (skimage).

    Args:
        image (np.ndarray): Input image (float).
        min_sigma (float): Minimum sigma for blobs.
        max_sigma (float): Maximum sigma.
        threshold (float): Absolute threshold for detection.

    Returns:
        np.ndarray: Array of detected blobs [[y, x, r], ...].
    """
    blobs = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
    # convert radius
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs
