import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

from . import config as cfg


def evaluate_saved_model(features_csv: str = cfg.FEATURES_CSV,
                         verbose: bool = True) -> dict:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)

    model_path = os.path.join(cfg.MODELS_DIR, "best_ml_model.pkl")
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.pkl")
    le_path = os.path.join(cfg.MODELS_DIR, "label_encoder.pkl")
    meta_path = os.path.join(cfg.MODELS_DIR, "best_model_meta.json")

    for p in [model_path, scaler_path, le_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Не знайдено {p}. Спочатку запустіть train_ml_models.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = pd.read_csv(features_csv)
    X = df[feature_cols].values.astype(np.float64)
    y_raw = df["class"].values
    y = le.transform(y_raw)
    groups = df["source_image"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                            random_state=cfg.RANDOM_STATE)
    _, test_idx = next(gss.split(X, y, groups=groups))

    if "is_augmented" in df.columns:
        is_aug_arr = df["is_augmented"].astype(bool).values
        test_idx = test_idx[~is_aug_arr[test_idx]]

    X_test, y_test = X[test_idx], y[test_idx]
    X_test_s = scaler.transform(X_test)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                   zero_division=0)

    if verbose:
        print(f"Модель: {meta['best_name']}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-macro:  {f1:.4f}")
        print()
        print(report)

    return {
        "best_name": meta["best_name"],
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion": cm,
        "report": report,
        "class_names": list(le.classes_),
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    evaluate_saved_model()
