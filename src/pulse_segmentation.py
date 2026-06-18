import numpy as np
from scipy.signal import find_peaks

from .config import (
    N_PULSES_FALLBACK, PEAK_PROMINENCE_FRAC,
    PEAK_MIN_DISTANCE_FRAC, PULSE_HALF_WIDTH_FRAC,
)


def _interp_nans(y: np.ndarray) -> np.ndarray:
    p = np.asarray(y, dtype=np.float64)
    if not np.any(np.isnan(p)):
        return p
    n = len(p)
    mask = ~np.isnan(p)
    if mask.sum() < 2:
        return np.zeros_like(p)
    idx = np.arange(n)
    return np.interp(idx, idx[mask], p[mask])


def _smooth(p: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return p
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(p, k, mode="same")


def find_pulse_centers(y: np.ndarray) -> np.ndarray:
    if len(y) < 10:
        return np.array([], dtype=int)

    span = np.nanmax(y) - np.nanmin(y)
    if span < 1e-6:
        return np.array([], dtype=int)

    prominence = PEAK_PROMINENCE_FRAC * span
    distance = max(5, int(PEAK_MIN_DISTANCE_FRAC * len(y)))

    peaks_max, _ = find_peaks(y, prominence=prominence, distance=distance)
    peaks_min, _ = find_peaks(-y, prominence=prominence, distance=distance)

    if len(peaks_max) >= len(peaks_min):
        return peaks_max
    return peaks_min


def split_by_peaks(x: np.ndarray, y: np.ndarray,
                   peaks: np.ndarray) -> list[tuple[int, int]]:
    n = len(y)
    half = max(15, int(PULSE_HALF_WIDTH_FRAC * n))
    bounds = []
    for p in peaks:
        a = max(0, int(p) - half)
        b = min(n, int(p) + half)
        if b - a >= 30:
            bounds.append((a, b))
    return bounds


def find_rise_onsets(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 30:
        return np.array([], dtype=int)

    p = _interp_nans(y)
    span = float(np.nanmax(p) - np.nanmin(p))
    if span < 1e-6:
        return np.array([], dtype=int)

    win = max(5, n // 100)
    p_smooth = _smooth(p, win)

    deriv = np.diff(p_smooth)
    if deriv.size == 0:
        return np.array([], dtype=int)

    max_abs_deriv = float(np.max(np.abs(deriv)))
    if max_abs_deriv < 1e-6:
        return np.array([], dtype=int)

    deriv_threshold = 0.15 * max_abs_deriv
    min_dist = max(20, int(PEAK_MIN_DISTANCE_FRAC * n))

    deriv_peaks_pos, _ = find_peaks(deriv, prominence=deriv_threshold,
                                    distance=min_dist)
    deriv_peaks_neg, _ = find_peaks(-deriv, prominence=deriv_threshold,
                                    distance=min_dist)

    if len(deriv_peaks_pos) >= len(deriv_peaks_neg):
        deriv_peaks = deriv_peaks_pos
    else:
        deriv_peaks = deriv_peaks_neg

    if len(deriv_peaks) < 1:
        return np.array([], dtype=int)

    if len(deriv_peaks) >= 2:
        avg_period = int(np.mean(np.diff(deriv_peaks)))
        backstep = max(5, avg_period // 7)
    else:
        backstep = max(5, n // 60)

    onsets = np.array([max(0, int(d) - backstep) for d in deriv_peaks],
                      dtype=int)
    return onsets


def split_by_full_cycle(y: np.ndarray,
                        onsets: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    for i in range(len(onsets) - 1):
        a = int(onsets[i])
        b = int(onsets[i + 1])
        if b - a >= 30:
            bounds.append((a, b))
    return bounds


def split_evenly(n: int, n_parts: int = N_PULSES_FALLBACK) -> list[tuple[int, int]]:
    if n_parts <= 0 or n <= 0:
        return []
    edges = np.linspace(0, n, n_parts + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]


def find_pulse_boundaries(y: np.ndarray,
                          method: str = "full_cycle"
                          ) -> list[tuple[int, int]]:
    n = len(y)

    if method == "full_cycle":
        onsets = find_rise_onsets(y)
        if len(onsets) >= 2:
            bounds = split_by_full_cycle(y, onsets)
            if len(bounds) >= 1:
                return bounds

    if method in ("peak_centered", "full_cycle"):
        peaks = find_pulse_centers(y)
        if len(peaks) >= 2:
            return split_by_peaks(np.arange(n), y, peaks)

    return split_evenly(n, N_PULSES_FALLBACK)


def crop_pulse_image(image: np.ndarray, start: int, end: int) -> np.ndarray:
    if image.ndim == 2:
        return image[:, start:end].copy()
    return image[:, start:end, :].copy()


def crop_pulse_signal(y: np.ndarray, start: int, end: int) -> np.ndarray:
    return y[start:end].copy()
