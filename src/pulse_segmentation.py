import numpy as np
from scipy.signal import find_peaks

from .config import (
    N_PULSES_FALLBACK, PEAK_PROMINENCE_FRAC,
    PEAK_MIN_DISTANCE_FRAC, PULSE_HALF_WIDTH_FRAC,
    HYST_LEVEL_PERCENTILES, HYST_LOW_FRAC, HYST_HIGH_FRAC,
    HYST_MIN_DWELL_FRAC, SEGMENT_MIN_WIDTH,
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
    return np.convolve(np.pad(p, win // 2, mode="edge"), k, mode="valid")


def _smooth_window(n: int) -> int:
    return max(5, n // 100)


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


def _hysteresis_states(p: np.ndarray, t_lo: float, t_hi: float,
                       min_dwell: int) -> np.ndarray:
    n = len(p)
    s = np.full(n, -1, dtype=np.int8)
    state = -1
    for i, v in enumerate(p):
        if v >= t_hi:
            state = 1
        elif v <= t_lo:
            state = 0
        s[i] = state
    known = np.flatnonzero(s >= 0)
    if len(known) == 0:
        return np.zeros(n, dtype=np.int8)
    s[:known[0]] = s[known[0]]

    while True:
        edges = np.flatnonzero(np.diff(s)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [n]))
        lens = ends - starts
        if len(lens) == 1:
            return s
        short = [(lens[i], i) for i in range(len(lens)) if lens[i] < min_dwell]
        if not short:
            return s
        _, i = min(short)
        fill = s[starts[i] - 1] if i > 0 else s[ends[i]]
        s[starts[i]:ends[i]] = fill


def find_rise_onsets(y: np.ndarray, return_info: bool = False):
    n = len(y)
    info = {"detector": "hysteresis"}
    empty = np.array([], dtype=int)
    if n < 30:
        info["reason"] = f"сигнал закороткий ({n} відліків < 30)"
        return (empty, info) if return_info else empty

    p = _smooth(_interp_nans(y), _smooth_window(n))
    lo, hi = np.percentile(p, HYST_LEVEL_PERCENTILES)
    span = float(hi - lo)
    if span < 1e-6:
        info["reason"] = "сигнал плаский — немає розмаху між рівнями"
        return (empty, info) if return_info else empty

    t_lo = lo + HYST_LOW_FRAC * span
    t_hi = lo + HYST_HIGH_FRAC * span
    min_dwell = max(5, int(HYST_MIN_DWELL_FRAC * n))
    s = _hysteresis_states(p, t_lo, t_hi, min_dwell)

    d = np.diff(s.astype(np.int16))
    ups = np.flatnonzero(d == 1) + 1
    downs = np.flatnonzero(d == -1) + 1
    sign = 1 if len(ups) >= len(downs) else -1
    fronts = ups if sign == 1 else downs

    def foot(i: int) -> int:
        while i > 0 and sign * (p[i] - p[i - 1]) > 0:
            i -= 1
        return i

    onsets = [foot(int(f)) for f in fronts]

    info["edge_front"] = None
    if len(onsets) >= 2:
        k0 = max(0, n - min_dwell)
        top = k0 + int(np.argmax(sign * p[k0:]))
        j = foot(top)
        rise = sign * (p[top] - p[j])
        period = float(np.median(np.diff(onsets)))
        gap = j - onsets[-1]
        if (rise >= (HYST_HIGH_FRAC - HYST_LOW_FRAC) * span
                and 0.75 * period <= gap <= 1.25 * period):
            onsets.append(j)
            info["edge_front"] = int(j)
    onsets = np.array(onsets, dtype=int)

    info.update({
        "smoothed": p, "level_lo": float(lo), "level_hi": float(hi),
        "t_lo": float(t_lo), "t_hi": float(t_hi), "min_dwell": int(min_dwell),
        "direction": "up" if sign == 1 else "down",
        "fronts": fronts.astype(int), "onsets": onsets,
    })
    if len(onsets) == 0:
        info["reason"] = "жодного переходу між рівнями"
    return (onsets, info) if return_info else onsets


def split_by_full_cycle(y: np.ndarray,
                        onsets: np.ndarray) -> list[tuple[int, int]]:
    bounds = []
    for i in range(len(onsets) - 1):
        a = int(onsets[i])
        b = int(onsets[i + 1])
        if b - a >= SEGMENT_MIN_WIDTH:
            bounds.append((a, b))
    return bounds


def split_evenly(n: int, n_parts: int = N_PULSES_FALLBACK) -> list[tuple[int, int]]:
    if n_parts <= 0 or n <= 0:
        return []
    edges = np.linspace(0, n, n_parts + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]


def _discarded_full_cycle(n: int, onsets: np.ndarray) -> list[dict]:
    out = []
    if len(onsets) == 0:
        return out
    if onsets[0] > 0:
        out.append({"start": 0, "end": int(onsets[0]),
                    "reason": "край кадру до першого фронту — неповний цикл"})
    for a, b in zip(onsets[:-1], onsets[1:]):
        if b - a < SEGMENT_MIN_WIDTH:
            out.append({"start": int(a), "end": int(b),
                        "reason": f"відрізок між фронтами {b - a} px "
                                  f"< {SEGMENT_MIN_WIDTH} px"})
    if onsets[-1] < n:
        out.append({"start": int(onsets[-1]), "end": n,
                    "reason": "хвіст після останнього фронту — немає "
                              "наступного початку наростання, цикл неповний"})
    return out


def find_pulse_boundaries(y: np.ndarray,
                          method: str = "full_cycle",
                          return_info: bool = False):
    n = len(y)
    info: dict = {"requested": method, "notes": [], "discarded": []}

    def done(bounds, used):
        info["method_used"] = used
        return (bounds, info) if return_info else bounds

    if method == "full_cycle":
        onsets, oinfo = find_rise_onsets(y, return_info=True)
        info["onset_info"] = oinfo
        if len(onsets) >= 2:
            bounds = split_by_full_cycle(y, onsets)
            if len(bounds) >= 1:
                info["discarded"] = _discarded_full_cycle(n, onsets)
                return done(bounds, "full_cycle")
            info["notes"].append(
                "full_cycle: усі відрізки між фронтами коротші за "
                f"{SEGMENT_MIN_WIDTH} px")
        else:
            info["notes"].append(
                f"full_cycle: знайдено фронтів {len(onsets)} (< 2), повного "
                "циклу немає" + (f" — {oinfo['reason']}"
                                 if oinfo.get("reason") else ""))

    if method in ("peak_centered", "full_cycle"):
        peaks = find_pulse_centers(y)
        info["peaks"] = peaks
        if len(peaks) >= 2:
            return done(split_by_peaks(np.arange(n), y, peaks), "peak_centered")
        info["notes"].append(f"peak_centered: знайдено піків {len(peaks)} (< 2)")

    info["notes"].append(
        f"резервний поділ на {N_PULSES_FALLBACK} рівні частини — межі "
        "не прив'язані до фронтів")
    return done(split_evenly(n, N_PULSES_FALLBACK), "split_evenly")


def crop_pulse_image(image: np.ndarray, start: int, end: int) -> np.ndarray:
    if image.ndim == 2:
        return image[:, start:end].copy()
    return image[:, start:end, :].copy()


def crop_pulse_signal(y: np.ndarray, start: int, end: int) -> np.ndarray:
    return y[start:end].copy()
