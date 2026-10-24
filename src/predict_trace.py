from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from . import fusion as fz
from .image_preprocessing import read_image, preprocess_pipeline
from .signal_extraction import mask_to_clean_signal, segment_coverage
from .pulse_segmentation import (
    find_pulse_boundaries, crop_pulse_image, crop_pulse_signal,
)
from .pulse_quality import is_valid_pulse
from .feature_extraction import extract_features

_GREEN = (40, 170, 40)
_RED = (40, 40, 220)
_GREY = (150, 150, 150)
_STATUS_COLOR = {"accepted": _GREEN, "rejected": _RED, "discarded": _GREY}


def _img_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _qc_reasons(qc: dict) -> list[str]:
    out = []
    if not qc.get("ok_amplitude", True):
        out.append(f"розмах {qc['amplitude']:.1f} px < {cfg.QC_MIN_AMPLITUDE} px "
                   "— майже плаский шматок, імпульсу немає")
    if not qc.get("ok_width", True):
        out.append(f"ширина {qc['width']} px поза межами "
                   f"[{cfg.QC_MIN_WIDTH}, {cfg.QC_MAX_WIDTH}] px")
    if not qc.get("ok_coverage", True):
        out.append(f"маска знайшла лінію лише в {qc['coverage']:.0%} стовпців "
                   f"(< {cfg.QC_MIN_COVERAGE:.0%}) — решту домальовано "
                   "інтерполяцією")
    if not qc.get("ok_missing", True):
        out.append(f"заповнених значень {qc['non_nan_fraction']:.0%} "
                   f"< {cfg.QC_MIN_COVERAGE:.0%}")
    if not qc.get("ok_noise", True):
        r = qc.get("noise_ratio", np.inf)
        out.append("не вдалося оцінити шум — шматок закороткий або плаский"
                   if not np.isfinite(r) else
                   f"шумовість std(Δ)/std = {r:.2f} > {cfg.QC_MAX_NOISE} "
                   "— рваний сигнал")
    return out


def _draw_crop_box(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y1, y2 = max(0, cfg.CROP_TOP), max(1, h - cfg.CROP_BOTTOM)
    x1, x2 = max(0, cfg.CROP_LEFT), max(1, w - cfg.CROP_RIGHT)
    vis = img.copy()
    shade = vis.copy()
    shade[:] = (0, 0, 0)
    keep = np.zeros(vis.shape[:2], bool)
    keep[y1:y2, x1:x2] = True
    vis[~keep] = cv2.addWeighted(vis, 0.35, shade, 0.65, 0)[~keep]
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    return vis


def _draw_mask_overlay(cropped: np.ndarray, mask: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
    vis = (cropped * 0.45).astype(np.uint8)
    vis[mask > 0] = (255, 0, 255)
    h = vis.shape[0]
    pts = [(int(x), int(round(h - 1 - v))) for x, v in enumerate(y)
           if np.isfinite(v)]
    for p, q in zip(pts[:-1], pts[1:]):
        cv2.line(vis, p, q, (0, 255, 0), 1)
    return vis


def _draw_segments(cropped: np.ndarray, segments: list[dict]) -> np.ndarray:
    vis = cropped.copy()
    h = vis.shape[0]
    for s in segments:
        color = _STATUS_COLOR[s["status"]]
        overlay = vis.copy()
        cv2.rectangle(overlay, (s["start"], 0), (s["end"] - 1, h - 1), color, -1)
        alpha = 0.10 if s["status"] == "accepted" else 0.30
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        cv2.rectangle(vis, (s["start"], 1), (s["end"] - 1, h - 2), color, 2)
        tag = s["label"] if s["status"] != "discarded" else "edge"
        cv2.putText(vis, tag, (s["start"] + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def _plot_signal(y_raw: np.ndarray, y: np.ndarray, seg_info: dict,
                 segments: list[dict]) -> str:
    n = len(y)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)

    ax = axes[0]
    ok = np.isfinite(y_raw)
    ax.plot(x[ok], y_raw[ok], ".", ms=1.5, color="#888", label="сирий (маска)")
    if (~ok).any():
        ax.plot(x[~ok], np.full((~ok).sum(), np.nanmin(y)), "|", color="#d62728",
                ms=6, label="немає сигналу (NaN)")
    ax.plot(x, y, color="#1f77b4", lw=1.2, label="після інтерполяції і згладжування")
    ax.set_ylabel("висота, px")
    ax.set_title("Крок 3. Маска → одновимірний сигнал", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    oi = seg_info.get("onset_info") or {}
    sm = oi.get("smoothed")
    ax.plot(x, sm if sm is not None else y, color="#1f77b4", lw=1.2,
            label="згладжений сигнал")
    if "t_lo" in oi:
        ax.axhline(oi["t_lo"], color="#555", ls="--", lw=0.9,
                   label="поріг «низ» (1/3 розмаху)")
        ax.axhline(oi["t_hi"], color="#555", ls=":", lw=0.9,
                   label="поріг «верх» (2/3 розмаху)")
    for s in segments:
        c = np.array(_STATUS_COLOR[s["status"]][::-1]) / 255.0
        ax.axvspan(s["start"], s["end"], color=c, alpha=0.18)
        ax.text((s["start"] + s["end"]) / 2, ax.get_ylim()[1], s["label"],
                ha="center", va="top", fontsize=8)
    for o in oi.get("onsets", []):
        ax.axvline(o, color="#9467bd", lw=1)
    for f in oi.get("fronts", []):
        ax.plot(f, (sm if sm is not None else y)[f], "v", color="#9467bd", ms=5)
    ax.set_xlabel("стовпець обрізаного знімка, px")
    ax.set_ylabel("висота, px")
    ax.set_title("Крок 4. Фронти (▼ перетин порогу, | початок наростання) і межі",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_b64(fig)


def _model_input_vis(arr: np.ndarray, channels: str = "") -> np.ndarray:
    names = channels.split("+") if channels else []
    tiles = []
    for i in range(arr.shape[2]):
        t = (np.clip(arr[:, :, i], 0, 1) * 255).astype(np.uint8)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        if i < len(names):
            cv2.putText(t, names[i], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 200, 255), 1)
        tiles.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                        value=(90, 90, 90)))
    return np.hstack(tiles)


def build_trace(image_path: str, invert: bool = False,
                segmentation: str = "full_cycle",
                image_spec=None) -> dict:
    steps: list[dict] = []
    img = read_image(image_path)
    h0, w0 = img.shape[:2]

    prep = preprocess_pipeline(img, invert=invert)
    cropped, mask = prep["cropped"], prep["mask"]
    steps.append({
        "title": "1. Обрізання інтерфейсу осцилографа",
        "text": (f"Знімок {w0}×{h0} px. Відрізаємо {cfg.CROP_TOP} px зверху, "
                 f"{cfg.CROP_BOTTOM} px знизу, {cfg.CROP_RIGHT} px справа "
                 f"(панель налаштувань) → лишається поле графіка "
                 f"{cropped.shape[1]}×{cropped.shape[0]} px (червона рамка). "
                 "Межі фіксовані — розраховані на скріншоти 800×480."),
        "images": [_img_b64(_draw_crop_box(img))],
    })
    if (w0, h0) != (800, 480):
        steps[-1]["warn"] = (f"Розмір {w0}×{h0} відрізняється від 800×480, на "
                             "яких навчались моделі: рамка обрізання може "
                             "захопити зайве або відрізати частину графіка.")

    x, y, cov, y_raw = mask_to_clean_signal(mask, return_raw=True)
    how = ("темні пікселі після інверсії (поріг "
           f"{cfg.INVERTED_THRESHOLD})" if invert else
           f"жовті пікселі в HSV: H {cfg.YELLOW_HSV_LOW[0]}–{cfg.YELLOW_HSV_HIGH[0]}, "
           f"S ≥ {cfg.YELLOW_HSV_LOW[1]}, V ≥ {cfg.YELLOW_HSV_LOW[2]}")
    steps.append({
        "title": "2. Маска кривої",
        "text": (f"Лишаємо {how}; морфологія open+close 3×3 прибирає "
                 f"поодинокі точки й зашиває дрібні розриви. Пікселів маски: "
                 f"{int((mask > 0).sum())}. Пурпурове — маска, зелене — "
                 "лінія сигналу, яку з неї отримано (медіана по кожному "
                 "стовпцю)."),
        "images": [_img_b64(mask), _img_b64(_draw_mask_overlay(cropped, mask, y))],
    })

    result: dict = {"steps": steps, "segments": [], "stopped": None,
                    "summary": {}}
    if cov < 0.05:
        result["stopped"] = (f"Маска знайшла сигнал лише в {cov:.1%} стовпців "
                             "(< 5%) — знімок відкинуто цілком, сегментація не "
                             "виконувалась. Перевірте колір кривої або "
                             "галочку «Інвертувати».")
        return result

    bounds, seg_info = find_pulse_boundaries(y, method=segmentation,
                                             return_info=True)

    segments: list[dict] = []
    for k, (a, b) in enumerate(bounds):
        pulse = crop_pulse_signal(y, a, b)
        local_cov = segment_coverage(y_raw, a, b)
        valid, qc = is_valid_pulse(pulse, coverage_value=local_cov)
        seg = {"k": k, "start": a, "end": b, "width": b - a,
               "label": f"p{k}", "status": "accepted" if valid else "rejected",
               "reasons": [] if valid else _qc_reasons(qc),
               "amplitude": qc["amplitude"], "coverage": local_cov,
               "noise": qc["noise_ratio"],
               "thumb": _img_b64(crop_pulse_image(cropped, a, b))}
        if valid:
            feats = extract_features(pulse)
            seg["amp_to_noise"] = feats.get("amp_to_noise", 1.0)
            seg["weight"] = fz.pulse_quality_weight(local_cov, seg["amp_to_noise"])
            if image_spec is not None:
                from .image_prep import prepare_array
                arr = prepare_array(crop_pulse_image(cropped, a, b), image_spec)
                if arr is None:
                    seg["status"] = "rejected"
                    seg["reasons"] = ["не вдалося підготувати вхід картинкової моделі"]
                else:
                    seg["model_input"] = _img_b64(_model_input_vis(
                        arr, getattr(image_spec, "channels", "")))
        segments.append(seg)
    for d in seg_info.get("discarded", []):
        segments.append({"k": None, "start": d["start"], "end": d["end"],
                         "width": d["end"] - d["start"], "label": "край",
                         "status": "discarded", "reasons": [d["reason"]],
                         "thumb": _img_b64(crop_pulse_image(cropped, d["start"],
                                                            d["end"]))
                         if d["end"] - d["start"] > 1 else ""})
    segments.sort(key=lambda s: s["start"])

    oi = seg_info.get("onset_info") or {}
    used = seg_info.get("method_used")
    if used == "full_cycle":
        seg_text = (
            f"Знайдено фронтів наростання: {len(oi.get('onsets', []))}. "
            "Фронт — момент, коли згладжений сигнал переходить з нижньої третини "
            "розмаху у верхню (рівні — 10-й і 90-й перцентилі, "
            f"{oi.get('level_lo', 0):.0f}…{oi.get('level_hi', 0):.0f} px) і "
            f"тримається там щонайменше {oi.get('min_dwell', 0)} px; коротші "
            "стрибки — викиди. Від фронту відступаємо назад до підніжжя підйому. "
            "Імпульс = від одного підніжжя до наступного (повний цикл), тому з N "
            f"фронтів виходить N−1 імпульс: {len(bounds)}. Шматки до першого і "
            "після останнього фронту — неповні цикли, вони сірі.")
    else:
        seg_text = ("Основний метод не спрацював, використано резервний: "
                    f"{used}. " + " ".join(seg_info.get("notes", [])))
    steps.append({
        "title": "3–4. Сигнал і пошук імпульсів",
        "text": seg_text,
        "warn": (" ".join(seg_info["notes"]) if seg_info.get("notes")
                 and used == "full_cycle" else None),
        "images": [_plot_signal(y_raw, y, seg_info, segments)],
    })
    steps.append({
        "title": "5. Межі на знімку",
        "text": ("Зелене — імпульс пройшов контроль якості і пішов у модель; "
                 "червоне — відкинутий контролем якості; сіре — не імпульс "
                 "(неповний цикл на краю кадру). Причини — у таблиці нижче."),
        "images": [_img_b64(_draw_segments(cropped, segments))],
    })

    result["segments"] = segments
    result["segmentation_used"] = used
    result["summary"] = {
        "found": len(bounds),
        "accepted": sum(s["status"] == "accepted" for s in segments),
        "rejected": sum(s["status"] == "rejected" for s in segments),
        "discarded": sum(s["status"] == "discarded" for s in segments),
    }
    result["qc_rules"] = (
        f"розмах ≥ {cfg.QC_MIN_AMPLITUDE} px; ширина {cfg.QC_MIN_WIDTH}–"
        f"{cfg.QC_MAX_WIDTH} px; покриття маскою ≥ {cfg.QC_MIN_COVERAGE:.0%}; "
        f"шумовість std(Δ)/std ≤ {cfg.QC_MAX_NOISE}")
    result["image_model"] = image_spec is not None
    if image_spec is not None:
        result["image_spec"] = getattr(image_spec, "name", str(image_spec))
    return result
