from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


TIME_FEATURE_NAMES = [
    "amplitude", "min_val", "max_val", "mean_val", "median_val", "std_val",
    "area_under_curve", "rise_time", "fall_time", "max_slope", "min_slope",
    "energy", "num_local_max", "skewness", "kurtosis_val",
    "amp_to_noise", "duration", "argmax_rel", "argmin_rel",
]

FFT_FEATURE_NAMES = [
    "fft_centroid",
    "fft_bandwidth",
    "fft_rolloff",
    "fft_ratio_low",
    "fft_ratio_mid",
    "fft_ratio_high",
    "fft_peak_freq",
    "fft_peak_mag",
    "fft_flatness",
]

SHAPE_FEATURE_NAMES = [
    "rise_fall_ratio",
    "fwhm",
]

WAVELET_FEATURE_NAMES = [
    "wl_energy_a3",
    "wl_energy_d3",
    "wl_energy_d2",
    "wl_energy_d1",
    "wl_entropy",
]

FEATURE_NAMES = (TIME_FEATURE_NAMES + FFT_FEATURE_NAMES
                 + SHAPE_FEATURE_NAMES + WAVELET_FEATURE_NAMES)


def _safe_signal(pulse: np.ndarray) -> np.ndarray:
    p = np.asarray(pulse, dtype=np.float64)
    if not np.any(np.isnan(p)):
        return p
    idx = np.arange(len(p))
    mask = ~np.isnan(p)
    if mask.sum() < 2:
        return np.zeros_like(p)
    p = np.interp(idx, idx[mask], p[mask])
    return p


def _rise_fall_times(p: np.ndarray) -> tuple[float, float]:
    if len(p) < 4:
        return 0.0, 0.0

    a_min = np.min(p)
    a_max = np.max(p)
    span = a_max - a_min
    if span < 1e-9:
        return 0.0, 0.0

    low = a_min + 0.1 * span
    high = a_min + 0.9 * span
    peak_idx = int(np.argmax(p))

    left = p[:peak_idx + 1]
    if len(left) < 2:
        rise = 0.0
    else:
        i_low_candidates = np.where(left <= low)[0]
        i_high_candidates = np.where(left >= high)[0]
        if len(i_low_candidates) and len(i_high_candidates):
            i_low = i_low_candidates[-1]
            i_high = i_high_candidates[0]
            rise = float(max(0, i_high - i_low))
        else:
            rise = 0.0

    right = p[peak_idx:]
    if len(right) < 2:
        fall = 0.0
    else:
        i_high_candidates = np.where(right >= high)[0]
        i_low_candidates = np.where(right <= low)[0]
        if len(i_high_candidates) and len(i_low_candidates):
            i_high = i_high_candidates[-1]
            i_low = i_low_candidates[0]
            fall = float(max(0, i_low - i_high))
        else:
            fall = 0.0

    return rise, fall


def _fft_features(p: np.ndarray) -> dict:
    n = len(p)
    if n < 4:
        return {k: 0.0 for k in FFT_FEATURE_NAMES}

    p_centered = p - float(np.mean(p))
    Y = np.abs(rfft(p_centered))
    freqs = rfftfreq(n, d=1.0)

    total = float(np.sum(Y))
    if total <= 1e-12 or len(Y) < 2:
        return {k: 0.0 for k in FFT_FEATURE_NAMES}

    centroid = float(np.sum(freqs * Y) / total)
    bw_var = float(np.sum(((freqs - centroid) ** 2) * Y) / total)
    bandwidth = float(np.sqrt(max(0.0, bw_var)))

    cumulative = np.cumsum(Y) / total
    rolloff_idx = int(np.argmax(cumulative >= 0.85))
    rolloff = float(freqs[rolloff_idx])

    Y_pow = Y ** 2
    third = max(1, len(Y) // 3)
    e_low = float(np.sum(Y_pow[:third]))
    e_mid = float(np.sum(Y_pow[third:2 * third]))
    e_high = float(np.sum(Y_pow[2 * third:]))
    e_total = e_low + e_mid + e_high
    if e_total > 1e-12:
        ratio_low = e_low / e_total
        ratio_mid = e_mid / e_total
        ratio_high = e_high / e_total
    else:
        ratio_low = ratio_mid = ratio_high = 0.0

    peak_idx = int(np.argmax(Y[1:])) + 1 if len(Y) > 1 else 0
    peak_freq = float(freqs[peak_idx])
    peak_mag = float(Y[peak_idx])

    Y_pos = Y[Y > 1e-12]
    if len(Y_pos) > 1:
        log_mean = float(np.mean(np.log(Y_pos)))
        flatness = float(np.exp(log_mean) / (np.mean(Y_pos) + 1e-12))
    else:
        flatness = 0.0

    return {
        "fft_centroid": centroid,
        "fft_bandwidth": bandwidth,
        "fft_rolloff": rolloff,
        "fft_ratio_low": ratio_low,
        "fft_ratio_mid": ratio_mid,
        "fft_ratio_high": ratio_high,
        "fft_peak_freq": peak_freq,
        "fft_peak_mag": peak_mag,
        "fft_flatness": flatness,
    }


def _fwhm(p: np.ndarray) -> float:
    if len(p) < 4:
        return 0.0
    a_min = float(np.min(p))
    a_max = float(np.max(p))
    span = a_max - a_min
    if span < 1e-9:
        return 0.0
    half = a_min + 0.5 * span
    above = np.where(p >= half)[0]
    if len(above) < 2:
        return 0.0
    return float(above[-1] - above[0] + 1) / len(p)


def _wavelet_features(p: np.ndarray) -> dict:
    if not _HAS_PYWT or len(p) < 8:
        return {k: 0.0 for k in WAVELET_FEATURE_NAMES}
    p_centered = p - float(np.mean(p))
    try:
        coeffs = pywt.wavedec(p_centered, "db4", level=3)
    except Exception:
        return {k: 0.0 for k in WAVELET_FEATURE_NAMES}

    energies = np.array([float(np.sum(np.square(c))) for c in coeffs])
    total = float(np.sum(energies))
    if total <= 1e-12 or len(energies) < 4:
        return {k: 0.0 for k in WAVELET_FEATURE_NAMES}

    fr = energies / total
    entropy = float(-np.sum(fr * np.log(fr + 1e-12)))
    return {
        "wl_energy_a3": float(fr[0]),
        "wl_energy_d3": float(fr[1]),
        "wl_energy_d2": float(fr[2]),
        "wl_energy_d1": float(fr[3]),
        "wl_entropy": entropy,
    }


def extract_features(pulse: np.ndarray) -> dict:
    p = _safe_signal(pulse)
    n = len(p)
    if n < 2:
        return {k: 0.0 for k in FEATURE_NAMES}

    a_min = float(np.min(p))
    a_max = float(np.max(p))
    amp = a_max - a_min
    mean_v = float(np.mean(p))
    median_v = float(np.median(p))
    std_v = float(np.std(p))

    area = float(np.trapezoid(p)) if hasattr(np, "trapezoid") else float(np.trapz(p))
    energy = float(np.sum(p * p))

    diff = np.diff(p)
    max_slope = float(np.max(diff)) if diff.size else 0.0
    min_slope = float(np.min(diff)) if diff.size else 0.0

    if amp > 1e-9:
        peaks, _ = find_peaks(p, prominence=0.1 * amp)
        n_peaks = int(len(peaks))
    else:
        n_peaks = 0

    try:
        sk = float(skew(p))
    except Exception:
        sk = 0.0
    try:
        kt = float(kurtosis(p))
    except Exception:
        kt = 0.0

    rise, fall = _rise_fall_times(p)

    noise = float(np.std(diff)) if diff.size else 1e-9
    amp_to_noise = amp / (noise + 1e-9)

    duration = float(n)

    arg_max = int(np.argmax(p))
    arg_min = int(np.argmin(p))
    argmax_rel = arg_max / max(1, n - 1)
    argmin_rel = arg_min / max(1, n - 1)

    feats = {
        "amplitude": amp,
        "min_val": a_min,
        "max_val": a_max,
        "mean_val": mean_v,
        "median_val": median_v,
        "std_val": std_v,
        "area_under_curve": area,
        "rise_time": rise,
        "fall_time": fall,
        "max_slope": max_slope,
        "min_slope": min_slope,
        "energy": energy,
        "num_local_max": n_peaks,
        "skewness": sk,
        "kurtosis_val": kt,
        "amp_to_noise": amp_to_noise,
        "duration": duration,
        "argmax_rel": argmax_rel,
        "argmin_rel": argmin_rel,
        "rise_fall_ratio": rise / (rise + fall + 1e-9),
        "fwhm": _fwhm(p),
    }
    feats.update(_fft_features(p))
    feats.update(_wavelet_features(p))
    return feats
