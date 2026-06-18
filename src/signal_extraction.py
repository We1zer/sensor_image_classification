import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def mask_to_signal(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    x = np.arange(w)
    y = np.full(w, np.nan, dtype=np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return x, y

    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]

    unique_cols, idx_starts = np.unique(xs_s, return_index=True)
    idx_ends = np.append(idx_starts[1:], len(xs_s))

    medians = np.empty(len(unique_cols), dtype=np.float32)
    for i, (a, b) in enumerate(zip(idx_starts, idx_ends)):
        medians[i] = np.median(ys_s[a:b])

    y[unique_cols] = (h - 1) - medians
    return x, y


def coverage(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sum(~np.isnan(y))) / len(y)


def interpolate_missing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return y.copy()
    f = interp1d(x[mask], y[mask], kind="linear",
                 bounds_error=False, fill_value=(y[mask][0], y[mask][-1]))
    return f(x).astype(np.float32)


def smooth_signal(y: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    n = len(y)
    if n < 5:
        return y.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        w = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < polyorder + 2:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=polyorder)


def normalize_signal(y: np.ndarray) -> np.ndarray:
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    if y_max - y_min < 1e-8:
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)


def mask_to_clean_signal(mask: np.ndarray,
                         do_smooth: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    x, y_raw = mask_to_signal(mask)
    cov = coverage(y_raw)
    y = interpolate_missing(x, y_raw)
    if do_smooth:
        y = smooth_signal(y)
    return x, y, cov
