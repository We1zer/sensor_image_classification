import os
import json
import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .image_preprocessing import read_image, preprocess_pipeline
from .signal_extraction import mask_to_clean_signal
from .pulse_segmentation import find_pulse_boundaries, crop_pulse_signal
from .pulse_quality import is_valid_pulse
from .feature_extraction import extract_features
from . import fusion as fz


def _load_model_bundle():
    model = joblib.load(os.path.join(cfg.MODELS_DIR, "best_ml_model.pkl"))
    scaler = joblib.load(os.path.join(cfg.MODELS_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(cfg.MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(cfg.MODELS_DIR, "best_model_meta.json"),
              "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, le, meta


def predict_image(image_path: str,
                  invert: bool = False,
                  file_number: int = 0,
                  fusion_strategy: str = "weighted",
                  verbose: bool = True) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    model, scaler, le, meta = _load_model_bundle()
    feature_cols = meta["feature_cols"]
    use_file_number = meta.get("use_file_number", False)
    class_names = list(le.classes_)

    img = read_image(image_path)
    prep = preprocess_pipeline(img, invert=invert)
    x, y, cov = mask_to_clean_signal(prep["mask"])

    if cov < 0.05:
        return {
            "image": image_path,
            "n_pulses_total": 0,
            "n_pulses_valid": 0,
            "pulse_predictions": [],
            "final_class": None,
            "final_confidence": 0.0,
            "average_probabilities": {},
            "warning": "Не вдалось виділити сигнал на зображенні.",
        }

    seg_method = "full_cycle"
    try:
        pipe_meta_path = os.path.join(cfg.FEATURES_DIR, "pipeline_meta.json")
        if os.path.exists(pipe_meta_path):
            with open(pipe_meta_path, "r", encoding="utf-8") as _f:
                seg_method = json.load(_f).get("segmentation", "full_cycle")
    except Exception:
        pass
    bounds = find_pulse_boundaries(y, method=seg_method)
    pulse_predictions = []
    probs_acc = []
    weights_acc = []

    for k, (a, b) in enumerate(bounds):
        pulse = crop_pulse_signal(y, a, b)
        local_cov = float(np.sum(~np.isnan(pulse))) / max(1, len(pulse))
        valid, qc = is_valid_pulse(pulse, coverage_value=local_cov)
        if not valid:
            continue

        feats = extract_features(pulse)
        if use_file_number:
            feats["file_number"] = file_number

        feat_vec = np.array([[feats[c] for c in feature_cols]],
                            dtype=np.float64)
        feat_vec_s = scaler.transform(feat_vec)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat_vec_s)[0]
        else:
            pred_idx = int(model.predict(feat_vec_s)[0])
            probs = np.zeros(len(class_names))
            probs[pred_idx] = 1.0

        if hasattr(model, "classes_"):
            ordered = np.zeros(len(class_names))
            for i, cls_idx in enumerate(model.classes_):
                ordered[int(cls_idx)] = probs[i]
            probs = ordered

        idx = int(np.argmax(probs))
        pulse_predictions.append({
            "pulse_index": k,
            "class": class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {class_names[i]: float(probs[i])
                              for i in range(len(class_names))},
        })
        probs_acc.append(probs)
        weights_acc.append(fz.pulse_quality_weight(
            local_cov, feats.get("amp_to_noise", 1.0)))

    if not pulse_predictions:
        return {
            "image": image_path,
            "n_pulses_total": len(bounds),
            "n_pulses_valid": 0,
            "pulse_predictions": [],
            "final_class": None,
            "final_confidence": 0.0,
            "average_probabilities": {},
            "warning": ("Класифікацію не виконано: не вдалося виділити "
                        "достатньо якісних імпульсів."),
        }

    probs_stack = np.stack(probs_acc)
    weights_arr = np.asarray(weights_acc, dtype=float)
    fused = fz.fuse_proba(probs_stack, strategy=fusion_strategy,
                          weights=weights_arr)
    final_idx = int(np.argmax(fused))
    final_class = class_names[final_idx]
    final_conf = float(fused[final_idx])

    result = {
        "image": image_path,
        "n_pulses_total": len(bounds),
        "n_pulses_valid": len(pulse_predictions),
        "pulse_predictions": pulse_predictions,
        "final_class": final_class,
        "final_confidence": final_conf,
        "fusion_strategy": fusion_strategy,
        "average_probabilities": {class_names[i]: float(fused[i])
                                  for i in range(len(class_names))},
        "warning": None if final_conf >= 0.55 else
                   "Низька впевненість моделі. Рекомендовано перевірити якість зображення.",
    }

    if verbose:
        print(f"Файл: {os.path.basename(image_path)}")
        print(f"  знайдено імпульсів: {result['n_pulses_total']}, "
              f"якісних: {result['n_pulses_valid']}")
        for p in pulse_predictions:
            print(f"  імпульс {p['pulse_index']}: {p['class']:8s}  "
                  f"confidence={p['confidence']:.2f}")
        print(f"  → Фінальний клас: {final_class}  ({final_conf*100:.1f}%)")
        if result["warning"]:
            print(f"  ! {result['warning']}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Використання: python -m src.predict <шлях_до_PNG>")
        sys.exit(0)
    predict_image(sys.argv[1])
