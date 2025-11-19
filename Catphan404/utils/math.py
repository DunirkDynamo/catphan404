# -----------------------------
# File: catphan404/utils/math.py
# -----------------------------
"""Mathematical helpers: ESF -> LSF -> MTF, simple fits and metrics."""
from typing import Tuple
import numpy as np
from scipy import ndimage, fftpack
from scipy.interpolate import UnivariateSpline


def compute_esf(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Edge Spread Function (ESF) from a 1D profile.

    Args:
        profile (np.ndarray): Intensity profile across an edge.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (positions, esf_values) where positions are indices.
    """
    # smooth profile
    x = np.arange(profile.size)
    s = UnivariateSpline(x, profile, s=profile.size * 0.1)
    esf = s(x)
    return x, esf


def esf_to_lsf(esf_x: np.ndarray, esf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Differentiate ESF to obtain Line Spread Function (LSF).

    Args:
        esf_x (np.ndarray): Positions for ESF.
        esf (np.ndarray): ESF values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (freqs, mtf_values) after FFT of normalized LSF.
    """
    # derivative
    lsf = np.gradient(esf, esf_x)
    # window and normalize
    lsf = lsf - lsf.mean()
    lsf = lsf / np.trapz(np.abs(lsf), esf_x)
    # FFT -> MTF
    n = lsf.size
    fft = fftpack.fft(lsf)
    freqs = fftpack.fftfreq(n, d=1.0)
    mtf = np.abs(fft)
    # only positive freqs
    pos = freqs >= 0
    return freqs[pos], mtf[pos] / mtf[pos].max()


def compute_mtf_from_edge(profile_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MTF from a 2D edge ROI by averaging perpendicular profiles.

    Args:
        profile_2d (np.ndarray): 2D ROI containing an edge approximately vertical.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (spatial_freqs, mtf_values)
    """
    # collapse rows to make 1D profile across edge (average columns)
    proj = np.mean(profile_2d, axis=0)
    x, esf = compute_esf(proj)
    freqs, mtf = esf_to_lsf(x, esf)
    # spatial freq conversion: pixels^-1; user can convert to lp/mm if spacing known
    return freqs, mtf
