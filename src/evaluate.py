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


def cross_validate_with_fusion(features_csv: str = cfg.FEATURES_CSV,
                               model_name: str = "RandomForest",
                               n_splits: int = 5,
                               use_xgboost: bool = True,
                               save_plots: bool = True,
                               verbose: bool = True) -> dict:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import GroupKFold

    from .train_ml_models import _build_models
    from .feature_extraction import FEATURE_NAMES
    from . import fusion as fz

    cfg.set_global_seeds()

    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)
    df = pd.read_csv(features_csv).reset_index(drop=True)
    if df.empty:
        raise ValueError("Файл ознак порожній.")

    feature_cols = list(FEATURE_NAMES)
    X = df[feature_cols].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["class"].values)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    groups = df["source_image"].values
    image_key = df["image_path"].values

    is_aug = (df["is_augmented"].astype(bool).values
              if "is_augmented" in df.columns else np.zeros(len(df), bool))

    weights_all = np.array([
        fz.pulse_quality_weight(c, s)
        for c, s in zip(df["local_coverage"].values, df["amp_to_noise"].values)
    ], dtype=float)

    models_avail = _build_models(use_xgboost=use_xgboost)
    if model_name not in models_avail:
        raise ValueError(f"Модель {model_name} недоступна. "
                         f"Є: {list(models_avail)}")

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    strategies = list(fz.STRATEGIES.keys())

    oof_proba = np.full((len(df), n_classes), np.nan, dtype=float)

    fold_pulse_f1: list[float] = []
    fold_img_f1: dict[str, list[float]] = {s: [] for s in strategies}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        te = te[~is_aug[te]]
        if len(te) == 0:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        model = _build_models(use_xgboost=use_xgboost)[model_name]
        model.fit(Xtr, y[tr])
        proba = model.predict_proba(Xte)
        ordered = np.zeros((len(te), n_classes))
        for j, cls_idx in enumerate(model.classes_):
            ordered[:, int(cls_idx)] = proba[:, j]
        oof_proba[te] = ordered

        y_pulse = np.argmax(ordered, axis=1)
        fold_pulse_f1.append(
            f1_score(y[te], y_pulse, average="macro", zero_division=0))

        df_te = df.iloc[te]
        for strat in strategies:
            true_img, pred_img = [], []
            for img, sub in df_te.groupby("image_path"):
                idx = sub.index.values
                p = oof_proba[idx]
                w = weights_all[idx]
                pred_img.append(fz.fuse(p, strategy=strat, weights=w))
                true_img.append(y[idx[0]])
            fold_img_f1[strat].append(
                f1_score(true_img, pred_img, average="macro", zero_division=0))

    evaluated = ~np.isnan(oof_proba).any(axis=1) & (~is_aug)
    y_eval = y[evaluated]
    pulse_pred = np.argmax(oof_proba[evaluated], axis=1)
    pulse_f1_oof = f1_score(y_eval, pulse_pred, average="macro", zero_division=0)
    pulse_acc_oof = accuracy_score(y_eval, pulse_pred)
    cm_pulse = confusion_matrix(y_eval, pulse_pred)

    img_results = {}
    df_eval = df[evaluated]
    for strat in strategies:
        true_img, pred_img = [], []
        for img, sub in df_eval.groupby("image_path"):
            idx = sub.index.values
            pred_img.append(fz.fuse(oof_proba[idx], strategy=strat,
                                    weights=weights_all[idx]))
            true_img.append(y[idx[0]])
        img_results[strat] = {
            "f1_oof": f1_score(true_img, pred_img, average="macro",
                               zero_division=0),
            "acc_oof": accuracy_score(true_img, pred_img),
            "cm": confusion_matrix(true_img, pred_img),
            "f1_fold_mean": float(np.mean(fold_img_f1[strat])),
            "f1_fold_std": float(np.std(fold_img_f1[strat])),
        }

    best_strat = max(img_results, key=lambda s: img_results[s]["f1_oof"])

    rows = [{
        "level": "pulse",
        "strategy": "-",
        "f1_macro_oof": round(pulse_f1_oof, 4),
        "accuracy_oof": round(pulse_acc_oof, 4),
        "f1_macro_fold_mean": round(float(np.mean(fold_pulse_f1)), 4),
        "f1_macro_fold_std": round(float(np.std(fold_pulse_f1)), 4),
    }]
    for strat in strategies:
        r = img_results[strat]
        rows.append({
            "level": "image",
            "strategy": strat,
            "f1_macro_oof": round(r["f1_oof"], 4),
            "accuracy_oof": round(r["acc_oof"], 4),
            "f1_macro_fold_mean": round(r["f1_fold_mean"], 4),
            "f1_macro_fold_std": round(r["f1_fold_std"], 4),
        })
    results_df = pd.DataFrame(rows)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(cfg.RESULTS_DIR, "fusion_comparison.csv")
    results_df.to_csv(out_csv, index=False)

    plot_paths = {}
    if save_plots:
        from .visualization import plot_confusion
        plot_paths["pulse"] = plot_confusion(
            cm_pulse, class_names,
            title=f"Pulse-level ({model_name})",
            out_name=f"confusion_pulse_{model_name}.png")
        plot_paths["image"] = plot_confusion(
            img_results[best_strat]["cm"], class_names,
            title=f"Image-level ({model_name}, {best_strat})",
            out_name=f"confusion_image_{model_name}_{best_strat}.png")

    if verbose:
        print(f"  Модель: {model_name} | GroupKFold={n_splits} | "
              f"знімків={df_eval['image_path'].nunique()} | "
              f"імпульсів(оцінено)={int(evaluated.sum())}")
        print()
        print(f"  pulse-level : f1_macro {pulse_f1_oof:.4f} "
              f"(fold {np.mean(fold_pulse_f1):.4f} ± {np.std(fold_pulse_f1):.4f})")
        print("  image-level (по стратегіях злиття):")
        for strat in strategies:
            r = img_results[strat]
            mark = "  <- best" if strat == best_strat else ""
            print(f"    {strat:11s}: f1_macro {r['f1_oof']:.4f} "
                  f"(fold {r['f1_fold_mean']:.4f} ± {r['f1_fold_std']:.4f})"
                  f"  acc {r['acc_oof']:.4f}{mark}")
        print()
        print(f"  Δ (image[{best_strat}] − pulse) = "
              f"{img_results[best_strat]['f1_oof'] - pulse_f1_oof:+.4f}")
        print(f"[OK] Збережено: {out_csv}")

    return {
        "results": results_df,
        "best_strategy": best_strat,
        "pulse_f1_oof": pulse_f1_oof,
        "image_results": img_results,
        "cm_pulse": cm_pulse,
        "class_names": class_names,
        "model_name": model_name,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    evaluate_saved_model()
