import numpy as np

from .config import (
    QC_MIN_AMPLITUDE, QC_MIN_COVERAGE, QC_MIN_WIDTH,
    QC_MAX_WIDTH, QC_MAX_NOISE,
)


def check_amplitude(pulse: np.ndarray) -> tuple[bool, float]:
    if len(pulse) == 0:
        return False, 0.0
    amp = float(np.nanmax(pulse) - np.nanmin(pulse))
    return (amp >= QC_MIN_AMPLITUDE), amp


def check_width(pulse: np.ndarray) -> tuple[bool, int]:
    w = len(pulse)
    return (QC_MIN_WIDTH <= w <= QC_MAX_WIDTH), w


def check_coverage(coverage_value: float) -> bool:
    return coverage_value >= QC_MIN_COVERAGE


def check_missing(pulse: np.ndarray) -> tuple[bool, float]:
    if len(pulse) == 0:
        return False, 0.0
    frac = float(np.sum(~np.isnan(pulse))) / len(pulse)
    return (frac >= QC_MIN_COVERAGE), frac


def check_noise(pulse: np.ndarray) -> tuple[bool, float]:
    if len(pulse) < 5:
        return False, np.inf
    sig = pulse[~np.isnan(pulse)]
    if len(sig) < 5 or np.std(sig) < 1e-6:
        return False, np.inf
    ratio = float(np.std(np.diff(sig)) / (np.std(sig) + 1e-9))
    return (ratio <= QC_MAX_NOISE), ratio


def is_valid_pulse(pulse: np.ndarray, coverage_value: float = 1.0) -> tuple[bool, dict]:
    info = {}
    ok_amp, amp = check_amplitude(pulse)
    info["amplitude"] = amp
    info["ok_amplitude"] = ok_amp

    ok_w, w = check_width(pulse)
    info["width"] = w
    info["ok_width"] = ok_w

    ok_cov = check_coverage(coverage_value)
    info["coverage"] = coverage_value
    info["ok_coverage"] = ok_cov

    ok_miss, frac = check_missing(pulse)
    info["non_nan_fraction"] = frac
    info["ok_missing"] = ok_miss

    ok_noise, noise_ratio = check_noise(pulse)
    info["noise_ratio"] = noise_ratio
    info["ok_noise"] = ok_noise

    valid = ok_amp and ok_w and ok_cov and ok_miss and ok_noise
    info["valid"] = valid
    return valid, info
