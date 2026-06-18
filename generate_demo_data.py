import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as cfg


W, H = 1024, 600
TOP_PANEL = 35
BOTTOM_PANEL = 60
RIGHT_PANEL = 200


def _draw_oscilloscope_frame(img: np.ndarray) -> None:
    img[:TOP_PANEL, :] = (40, 40, 40)
    img[H - BOTTOM_PANEL:, :] = (40, 40, 40)
    img[:, W - RIGHT_PANEL:] = (50, 50, 50)
    grid_color = (60, 60, 60)
    inner_x1, inner_y1 = 0, TOP_PANEL
    inner_x2, inner_y2 = W - RIGHT_PANEL, H - BOTTOM_PANEL
    for x in range(inner_x1, inner_x2, 80):
        cv2.line(img, (x, inner_y1), (x, inner_y2), grid_color, 1)
    for y in range(inner_y1, inner_y2, 60):
        cv2.line(img, (inner_x1, y), (inner_x2, y), grid_color, 1)


def _draw_signal(img: np.ndarray, signal: np.ndarray) -> None:
    inner_x1, inner_y1 = 0, TOP_PANEL
    inner_x2, inner_y2 = W - RIGHT_PANEL, H - BOTTOM_PANEL
    width = inner_x2 - inner_x1
    height = inner_y2 - inner_y1

    s_min, s_max = signal.min(), signal.max()
    span = max(1e-9, s_max - s_min)
    norm = (signal - s_min) / span

    pts = []
    for i, v in enumerate(norm):
        x = inner_x1 + int(i * (width - 1) / (len(norm) - 1))
        y = inner_y2 - int(v * (height - 1))
        pts.append((x, y))

    yellow = (0, 255, 255)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], yellow, 2)


def _make_pulse(t: np.ndarray, t0: float, rise: float, fall: float,
                amp: float) -> np.ndarray:
    out = np.zeros_like(t)
    rising = (t >= t0) & (t < t0 + rise)
    out[rising] = amp * (1 - np.exp(-(t[rising] - t0) / max(0.05, rise / 4)))
    falling = t >= t0 + rise
    if np.any(falling):
        peak = out[rising][-1] if np.any(rising) else amp
        out[falling] = peak * np.exp(-(t[falling] - t0 - rise) / max(0.05, fall / 4))
    return out


def _signal_for_class(cls: str, file_number: int, rng: np.random.Generator) -> np.ndarray:
    n_points = 950
    t = np.linspace(0, 10, n_points)
    base = 0.05

    conc_factor = 1.0 + 0.6 * (file_number / 100.0) if cls != "RH" else 1.0

    if cls == "acetone":
        amp = 0.55 * conc_factor
        rise, fall = 0.3, 1.6
    elif cls == "ammonia":
        amp = 0.75 * conc_factor
        rise, fall = 0.7, 0.9
    elif cls == "ethanol":
        amp = 0.45 * conc_factor
        rise, fall = 0.5, 2.5
    else:
        amp = 0.30
        rise, fall = 0.8, 1.5

    centers = [1.5, 4.5, 7.5]
    sig = np.full_like(t, base)
    for c in centers:
        sig += _make_pulse(t, c + rng.normal(0, 0.05), rise, fall, amp)

    sig += rng.normal(0, 0.012, size=sig.shape)
    return sig


def generate(n_per_class: int = 30, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for folder, cls in cfg.CLASS_FOLDERS.items():
        out_dir = os.path.join(cfg.DATASET_DIR, folder)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(1, n_per_class + 1):
            fn = i
            sig = _signal_for_class(cls, fn, rng)

            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[:] = (15, 15, 15)
            _draw_oscilloscope_frame(img)
            _draw_signal(img, sig)

            out_path = os.path.join(out_dir, f"{i:03d}.png")
            success, buf = cv2.imencode(".png", img)
            if success:
                buf.tofile(out_path)
        print(f"[OK] {folder}: створено {n_per_class} зображень")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30,
                   help="кількість зображень на клас (default 30)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.n, args.seed)
