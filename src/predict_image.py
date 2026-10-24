from __future__ import annotations

import json
import os

import numpy as np

from . import config as cfg
from .image_preprocessing import read_image, preprocess_pipeline
from .signal_extraction import mask_to_clean_signal, segment_coverage
from .pulse_segmentation import (
    find_pulse_boundaries, crop_pulse_image, crop_pulse_signal,
)
from .pulse_quality import is_valid_pulse
from .feature_extraction import extract_features
from .image_prep import prepare_array
from . import image_model_store as store
from . import fusion as fz


DEFAULT_METHOD = "MobileNetV2 frozen + LogReg"


def _segmentation_method() -> str:
    try:
        p = os.path.join(cfg.FEATURES_DIR, "pipeline_meta.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("segmentation", "full_cycle")
    except Exception:
        pass
    return "full_cycle"


def _empty(image_path: str, method: str, n_total: int, warning: str) -> dict:
    return {
        "image": image_path,
        "model": method,
        "branch": "image",
        "n_pulses_total": n_total,
        "n_pulses_valid": 0,
        "pulse_predictions": [],
        "final_class": None,
        "final_confidence": 0.0,
        "average_probabilities": {},
        "warning": warning,
    }


def predict_image_model(image_path: str,
                        method: str = DEFAULT_METHOD,
                        invert: bool = False,
                        fusion_strategy: str = "weighted",
                        verbose: bool = True) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    bundle = store.load_any(method)
    spec = store.spec_from_bundle(bundle)
    class_names = list(bundle["class_names"])

    img = read_image(image_path)
    prep = preprocess_pipeline(img, invert=invert)
    cropped = prep["cropped"]
    x, y, cov, y_raw = mask_to_clean_signal(prep["mask"], return_raw=True)

    if cov < 0.05:
        return _empty(image_path, method, 0,
                      "Не вдалось виділити сигнал на зображенні.")

    bounds = find_pulse_boundaries(y, method=_segmentation_method())

    arrays, weights_acc, indices = [], [], []
    for k, (a, b) in enumerate(bounds):
        pulse = crop_pulse_signal(y, a, b)
        local_cov = segment_coverage(y_raw, a, b)
        valid, _qc = is_valid_pulse(pulse, coverage_value=local_cov)
        if not valid:
            continue

        cut = crop_pulse_image(cropped, a, b)
        arr = prepare_array(cut, spec)
        if arr is None:
            continue

        feats = extract_features(pulse)
        arrays.append(arr)
        weights_acc.append(fz.pulse_quality_weight(
            local_cov, feats.get("amp_to_noise", 1.0)))
        indices.append(k)

    if not arrays:
        return _empty(image_path, method, len(bounds),
                      "Класифікацію не виконано: не вдалося виділити "
                      "достатньо якісних імпульсів.")

    X = np.stack(arrays)
    probs = store.predict_proba(bundle, X)

    pulse_predictions = []
    for row, k in zip(probs, indices):
        idx = int(np.argmax(row))
        pulse_predictions.append({
            "pulse_index": k,
            "class": class_names[idx],
            "confidence": float(row[idx]),
            "probabilities": {class_names[i]: float(row[i])
                              for i in range(len(class_names))},
        })

    fused = fz.fuse_proba(probs, strategy=fusion_strategy,
                          weights=np.asarray(weights_acc, dtype=float))
    final_idx = int(np.argmax(fused))
    result = {
        "image": image_path,
        "model": method,
        "branch": "image",
        "spec": spec.name,
        "n_pulses_total": len(bounds),
        "n_pulses_valid": len(pulse_predictions),
        "pulse_predictions": pulse_predictions,
        "final_class": class_names[final_idx],
        "final_confidence": float(fused[final_idx]),
        "fusion_strategy": fusion_strategy,
        "average_probabilities": {class_names[i]: float(fused[i])
                                  for i in range(len(class_names))},
        "warning": None if float(fused[final_idx]) >= 0.55 else
                   "Низька впевненість моделі. Рекомендовано перевірити "
                   "якість зображення.",
    }

    if verbose:
        print(f"Файл: {os.path.basename(image_path)}  [{method}]")
        print(f"  знайдено імпульсів: {result['n_pulses_total']}, "
              f"якісних: {result['n_pulses_valid']}")
        for p in pulse_predictions:
            print(f"  імпульс {p['pulse_index']}: {p['class']:8s}  "
                  f"confidence={p['confidence']:.2f}")
        print(f"  → Фінальний клас: {result['final_class']}  "
              f"({result['final_confidence'] * 100:.1f}%)")
    return result


def predict_all_models(image_path: str,
                       invert: bool = False,
                       fusion_strategy: str = "weighted",
                       verbose: bool = False) -> list[dict]:
    out = []
    for info in store.list_saved():
        try:
            out.append(predict_image_model(
                image_path, method=info["method"], invert=invert,
                fusion_strategy=fusion_strategy, verbose=verbose))
        except Exception as e:
            out.append({"model": info["method"], "error": str(e),
                        "final_class": None, "final_confidence": 0.0})
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Використання: python -m src.predict_image <PNG> [метод]")
        sys.exit(0)
    m = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_METHOD
    predict_image_model(sys.argv[1], method=m)
