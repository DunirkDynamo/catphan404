# -----------------------------
# File: catphan404/plots/plotters.py
# -----------------------------
"""Matplotlib-based plotting helpers for Catphan modules.

Each function returns a matplotlib.Figure object. The analyzer will save these
figures when requested by ``plot_all()``.
"""
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


def _ensure_fig():
    fig = plt.figure(figsize=(6, 6))
    return fig


def plot_uniformity(image: np.ndarray, center_info: Tuple[float, float, float], results: Dict):
    """Plot uniformity ROIs and a heatmap.

    Args:
        image: 2D image
        center_info: (cx, cy, radius)
        results: output of uniformity_analysis

    Returns:
        matplotlib.figure.Figure
    """
    cx, cy, radius = center_info
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(image, cmap='gray')
    fig.colorbar(im, ax=ax)
    ax.set_title('Uniformity')

    # draw center ROI
    circ = plt.Circle((cx, cy), radius * 0.5, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circ)
    # peripheral
    offs = [(radius * 0.7, 0), (-radius * 0.7, 0), (0, -radius * 0.7), (0, radius * 0.7)]
    for dx, dy in offs:
        c = plt.Circle((cx + dx, cy + dy), radius * 0.3, edgecolor='blue', facecolor='none', linewidth=1.5)
        ax.add_patch(c)

    # annotate means
    rois = results.get('rois', {})
    if rois:
        ax.text(0.01, 0.01, f"Center: {rois.get('center', {}).get('mean', 'n/a'):.2f}", transform=ax.transAxes, color='yellow')
    return fig


def plot_high_contrast(image: np.ndarray, results: Dict):
    """Plot MTF curve and edge ROI crop.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    y0, y1, x0, x1 = 0, image.shape[0], 0, image.shape[1]
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('High-contrast region (full image)')

    freqs = np.array(results.get('freqs', []))
    mtf = np.array(results.get('mtf', []))
    if freqs.size and mtf.size:
        axes[1].plot(freqs, mtf)
        axes[1].set_xlabel('spatial freq (px^-1)')
        axes[1].set_ylabel('MTF')
        axes[1].axhline(0.5, color='gray', linestyle='--')
        axes[1].set_title(f"MTF (f50={results.get('f50', float('nan')):.3f})")
    return fig


def plot_low_contrast(image: np.ndarray, results: Dict):
    """Plot detected low-contrast blobs over image and CNR bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Low-contrast detections')
    blobs = results.get('blobs', {})
    cnrs = []
    labels = []
    for k, v in blobs.items():
        x = v['x']
        y = v['y']
        r = v['r']
        circ = plt.Circle((x, y), r, edgecolor='lime', facecolor='none')
        axes[0].add_patch(circ)
        labels.append(k)
        cnrs.append(v['cnr'])
    axes[1].bar(labels, cnrs)
    axes[1].set_title('CNR per detection')
    axes[1].set_ylabel('CNR')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    return fig


def plot_slice_thickness(image: np.ndarray, results: Dict):
    """Plot slice thickness profile and indicate FWHM.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    y0, y1, x0, x1 = 0, image.shape[0], 0, image.shape[1]
    roi = image[y0:y1, x0:x1]
    profile = np.mean(roi, axis=1)
    x = np.arange(profile.size)
    ax.plot(x, profile)
    fwhm = results.get('fwhm_pixels', float('nan'))
    ax.set_title(f'Slice thickness profile (FWHM ≈ {fwhm:.1f} px)')
    return fig


def plot_geometry(image: np.ndarray, results: Dict):
    """Plot detected fiducials and pairwise distances.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.set_title('Geometry fiducials')
    markers = results.get('markers', [])
    for i, (x, y) in enumerate(markers):
        circ = plt.Circle((x, y), 8, edgecolor='cyan', facecolor='none')
        ax.add_patch(circ)
        ax.text(x + 5, y + 5, str(i), color='yellow')
    return fig