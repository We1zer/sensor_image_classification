from __future__ import annotations

import os
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as cfg


ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

UPLOADS_DIR = Path(cfg.BASE_DIR) / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def project_status() -> dict[str, Any]:
    dataset_counts = _dataset_counts()
    total_images = sum(dataset_counts.values())

    has_features = os.path.exists(cfg.FEATURES_CSV)
    n_pulses = 0
    feature_class_counts: dict[str, int] = {}
    if has_features:
        try:
            df = pd.read_csv(cfg.FEATURES_CSV)
            n_pulses = len(df)
            feature_class_counts = (
                df["class"].value_counts().to_dict() if "class" in df.columns else {}
            )
        except Exception:
            pass

    model_meta_path = os.path.join(cfg.MODELS_DIR, "best_model_meta.json")
    has_model = os.path.exists(model_meta_path)
    model_meta: dict[str, Any] = {}
    if has_model:
        try:
            with open(model_meta_path, "r", encoding="utf-8") as f:
                model_meta = json.load(f)
        except Exception:
            model_meta = {}

    n_results = 0
    if os.path.isdir(cfg.RESULTS_DIR):
        n_results = len(glob.glob(os.path.join(cfg.RESULTS_DIR, "*.png")))

    return {
        "dataset": {
            "counts": dataset_counts,
            "total": total_images,
            "ready": total_images > 0,
        },
        "features": {
            "ready": has_features and n_pulses > 0,
            "path": cfg.FEATURES_CSV,
            "n_pulses": n_pulses,
            "class_counts": feature_class_counts,
        },
        "model": {
            "ready": has_model,
            "meta": model_meta,
        },
        "results": {
            "n_plots": n_results,
            "dir": cfg.RESULTS_DIR,
        },
    }


def _dataset_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for folder, cls in cfg.CLASS_FOLDERS.items():
        full = os.path.join(cfg.DATASET_DIR, folder)
        if not os.path.isdir(full):
            counts[cls] = 0
            continue
        found = set(glob.glob(os.path.join(full, "*.png"))) | \
                set(glob.glob(os.path.join(full, "*.PNG")))
        counts[cls] = len(found)
    return counts


def list_result_plots() -> list[dict[str, str]]:
    descriptions = {
        "class_distribution.png":
            "Кількість імпульсів за класами після фільтрації якості.",
        "feature_distributions.png":
            "Boxplot розподілу ключових ознак за класами.",
        "feature_correlation_matrix.png":
            "Матриця кореляції числових ознак.",
        "signal_extraction_example.png":
            "Приклад: оригінал → обрізане → маска → числовий сигнал.",
        "examples_of_detected_pulses.png":
            "Приклад сегментації сигналу на окремі імпульси.",
        "concentration_analysis_amplitude.png":
            "Амплітуда імпульсу vs file_number (≈ концентрація).",
        "concentration_analysis_area.png":
            "Площа під кривою vs file_number (≈ концентрація).",
        "model_comparison.png":
            "Порівняння accuracy / precision / recall / f1 6 моделей.",
    }
    out: list[dict[str, str]] = []
    if not os.path.isdir(cfg.RESULTS_DIR):
        return out
    for path in sorted(glob.glob(os.path.join(cfg.RESULTS_DIR, "*.png"))):
        name = os.path.basename(path)
        desc = descriptions.get(name)
        if desc is None:
            if name.startswith("confusion_matrix"):
                desc = "Матриця помилок для найкращої моделі."
            elif name.startswith("concentration_analysis"):
                desc = "Залежність ознаки від file_number (≈ концентрація)."
            else:
                desc = "Графік дослідження."
        out.append({"name": name, "desc": desc})
    return out


def list_result_files() -> list[dict[str, str]]:
    out = []
    for name in ("model_comparison.csv", "classification_report.txt",
                 "original_vs_inverted.csv"):
        p = os.path.join(cfg.RESULTS_DIR, name)
        if os.path.exists(p):
            out.append({"name": name, "path": p})
    return out


def is_allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXT


def first_image_for_class(cls: str) -> str | None:
    for folder, c in cfg.CLASS_FOLDERS.items():
        if c != cls:
            continue
        full = os.path.join(cfg.DATASET_DIR, folder)
        if not os.path.isdir(full):
            return None
        found = set(glob.glob(os.path.join(full, "*.png"))) | \
                set(glob.glob(os.path.join(full, "*.PNG")))
        files = sorted(found)
        return files[0] if files else None
    return None


def format_proba(p: float) -> str:
    return f"{p * 100:.1f}%"
