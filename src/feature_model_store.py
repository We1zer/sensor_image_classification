from __future__ import annotations

import os
import re
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

from . import config as cfg
from .feature_extraction import FEATURE_NAMES
from .train_ml_models import _build_models


def _slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def model_path(method: str) -> str:
    return os.path.join(cfg.MODELS_DIR, f"feat_{_slug(method)}.pkl")


def available_methods(use_xgboost: bool = True) -> list[str]:
    return list(_build_models(use_xgboost=use_xgboost).keys())


def _load_dataset(features_csv: str = cfg.FEATURES_CSV):
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Немає {cfg.to_rel(features_csv)}. "
            "Спочатку: python -m src.dataset_builder")
    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError("Файл ознак порожній.")

    feature_cols = list(FEATURE_NAMES)
    X = df[feature_cols].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["class"].values)
    n_aug = int(df["is_augmented"].astype(bool).sum()) \
        if "is_augmented" in df.columns else 0
    return X, y, feature_cols, list(le.classes_), n_aug


def save_all(only: list[str] | None = None,
             use_xgboost: bool = True,
             verbose: bool = True) -> list[str]:
    cfg.set_global_seeds()
    X, y, feature_cols, class_names, n_aug = _load_dataset()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    names = only or available_methods(use_xgboost)
    saved = []
    for name in names:
        model = _build_models(use_xgboost=use_xgboost).get(name)
        if model is None:
            if verbose:
                print(f"  [--] {name}: недоступна, пропускаю")
            continue

        t0 = time.time()
        model.fit(Xs, y)
        bundle = {
            "method": name,
            "branch": "features",
            "estimator": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "class_names": class_names,
            "use_file_number": False,
            "n_train_rows": int(len(y)),
            "n_augmented": n_aug,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_sec": round(time.time() - t0, 1),
        }
        joblib.dump(bundle, model_path(name))
        saved.append(name)
        if verbose:
            print(f"  [OK] features + {name:20s} -> "
                  f"{cfg.to_rel(model_path(name))} "
                  f"({bundle['train_sec']} с)")
    return saved


def load(method: str) -> dict:
    p = model_path(method)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Немає збереженої моделі «{method}» ({cfg.to_rel(p)}). "
            "Спочатку: python -m src.feature_model_store")
    return joblib.load(p)


def list_saved(use_xgboost: bool = True) -> list[dict]:
    out = []
    for name in available_methods(use_xgboost):
        p = model_path(name)
        if not os.path.exists(p):
            continue
        try:
            b = joblib.load(p)
        except Exception:
            continue
        out.append({
            "method": name,
            "path": cfg.to_rel(p),
            "trained_at": b.get("trained_at"),
            "n_train_rows": b.get("n_train_rows"),
            "n_features": len(b.get("feature_cols") or []),
        })
    return out


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or None
    print("Навчання моделей гілки ознак на повному наборі...")
    save_all(only=only)
