from __future__ import annotations

import numpy as np


def _add_gaussian_noise(p: np.ndarray, rng: np.random.Generator,
                        sigma_frac: float = 0.03) -> np.ndarray:
    s = float(np.nanstd(p))
    if s < 1e-9:
        return p.copy()
    sigma = sigma_frac * s
    return p + rng.normal(0.0, sigma, size=p.shape)


def _scale_amplitude(p: np.ndarray, rng: np.random.Generator,
                     low: float = 0.92, high: float = 1.08) -> np.ndarray:
    mean_v = float(np.nanmean(p))
    scale = float(rng.uniform(low, high))
    return (p - mean_v) * scale + mean_v


def _time_shift(p: np.ndarray, rng: np.random.Generator,
                max_frac: float = 0.04) -> np.ndarray:
    n = len(p)
    if n < 4:
        return p.copy()
    max_shift = max(1, int(max_frac * n))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return p.copy()
    return np.roll(p, shift)


def _time_warp(p: np.ndarray, rng: np.random.Generator,
               max_stretch: float = 0.06) -> np.ndarray:
    n = len(p)
    if n < 8:
        return p.copy()
    factor = 1.0 + float(rng.uniform(-max_stretch, max_stretch))
    new_n = max(4, int(n * factor))
    src_x = np.linspace(0, 1, n)
    dst_x = np.linspace(0, 1, new_n)
    stretched = np.interp(dst_x, src_x, p)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, new_n), stretched)


def augment_pulse(pulse: np.ndarray, n: int,
                  rng: np.random.Generator | None = None) -> list[np.ndarray]:
    if n <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    p = np.asarray(pulse, dtype=np.float64)
    out = []
    for _ in range(n):
        a = p.copy()
        a = _add_gaussian_noise(a, rng, sigma_frac=float(rng.uniform(0.02, 0.05)))
        a = _scale_amplitude(a, rng,
                             low=float(rng.uniform(0.90, 0.96)),
                             high=float(rng.uniform(1.04, 1.10)))
        a = _time_shift(a, rng, max_frac=0.04)
        if rng.random() < 0.5:
            a = _time_warp(a, rng, max_stretch=0.05)
        out.append(a)
    return out
