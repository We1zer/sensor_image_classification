import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

from . import config as cfg
from .feature_extraction import FEATURE_NAMES, extract_features
from .signal_augmentation import augment_pulse, _add_gaussian_noise


def _load_originals_with_signals(features_csv: str = cfg.FEATURES_CSV
                                 ) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    if "is_augmented" in df.columns:
        df = df[~df["is_augmented"].astype(bool)]
    df = df[df["pulse_signal_path"].fillna("").astype(str).str.len() > 0]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Немає оригінальних рядків зі збереженими сигналами. "
            "Перебудуйте датасет: build_dataset(save_pulse_signals=True).")
    return df


def _features_from_signal(sig: np.ndarray) -> list[float]:
    feats = extract_features(sig)
    return [feats[c] for c in FEATURE_NAMES]


def augment_factor_sweep(factors=(0, 1, 2, 3),
                         track_models=("RandomForest", "XGBoost"),
                         n_splits: int = 5,
                         verbose: bool = True) -> pd.DataFrame:
    from .dataset_builder import build_dataset
    from .train_ml_models import cross_validate_models

    rows = []
    try:
        for f in factors:
            if verbose:
                print(f"  [sweep] augment_factor={f}: перебудова датасету ...")
            df = build_dataset(augment_factor=int(f),
                               save_pulse_signals=True, verbose=False)
            n_aug = int(df["is_augmented"].sum()) if "is_augmented" in df else 0
            res = cross_validate_models(n_splits=n_splits, use_xgboost=True,
                                        verbose=False)
            rdf = res["results"].set_index("model")
            for m in track_models:
                if m in rdf.index:
                    rows.append({
                        "augment_factor": int(f),
                        "model": m,
                        "n_augmented_rows": n_aug,
                        "f1_macro_mean": round(float(rdf.loc[m, "f1_macro_mean"]), 4),
                        "f1_macro_std": round(float(rdf.loc[m, "f1_macro_std"]), 4),
                        "accuracy_mean": round(float(rdf.loc[m, "accuracy_mean"]), 4),
                    })
            if verbose:
                for m in track_models:
                    if m in rdf.index:
                        print(f"      {m:14s}: f1 {rdf.loc[m,'f1_macro_mean']:.4f} "
                              f"± {rdf.loc[m,'f1_macro_std']:.4f}")
    finally:
        if verbose:
            print("  [sweep] відновлення канонічного датасету (factor=0) ...")
        build_dataset(augment_factor=0, save_pulse_signals=True, verbose=False)

    out = pd.DataFrame(rows)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.RESULTS_DIR, "augment_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in track_models:
        sub = out[out["model"] == m].sort_values("augment_factor")
        if sub.empty:
            continue
        ax.errorbar(sub["augment_factor"], sub["f1_macro_mean"],
                    yerr=sub["f1_macro_std"], marker="o", capsize=4, label=m)
    ax.set_xlabel("augment_factor (копій на імпульс)")
    ax.set_ylabel("f1_macro (GroupKFold, mean ± std)")
    ax.set_title("Вплив аугментації на якість (чисті дані)")
    ax.set_xticks(list(factors))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "augment_sweep.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/augment_sweep.csv (+ .png)")
    return out


def robustness_curve(noise_levels=(0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
                     aug_factor: int = 3,
                     model_name: str = "RandomForest",
                     test_size: float = 0.25,
                     seed: int = cfg.RANDOM_STATE,
                     verbose: bool = True) -> pd.DataFrame:
    from .train_ml_models import _build_models

    cfg.set_global_seeds(seed)
    df = _load_originals_with_signals()

    le = LabelEncoder()
    y = le.fit_transform(df["class"].values)
    groups = df["source_image"].values
    class_names = list(le.classes_)

    signals = [np.load(cfg.to_abs(p)).astype(np.float64)
               for p in df["pulse_signal_path"].values]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(signals, y, groups=groups))

    X_base = df[FEATURE_NAMES].values.astype(np.float64)
    Xtr_noaug = X_base[train_idx]
    ytr_noaug = y[train_idx]

    rng = np.random.default_rng(seed)
    Xtr_aug_rows = list(Xtr_noaug)
    ytr_aug_rows = list(ytr_noaug)
    for i in train_idx:
        for aug in augment_pulse(signals[i], aug_factor, rng=rng):
            Xtr_aug_rows.append(_features_from_signal(aug))
            ytr_aug_rows.append(y[i])
    Xtr_aug = np.asarray(Xtr_aug_rows, dtype=np.float64)
    ytr_aug = np.asarray(ytr_aug_rows)

    if verbose:
        print(f"  Модель: {model_name} | train(no-aug)={len(ytr_noaug)} | "
              f"train(aug×{aug_factor})={len(ytr_aug)} | test={len(test_idx)}")

    trained = {}
    for tag, (Xtr, ytr) in {
        "no-aug": (Xtr_noaug, ytr_noaug),
        f"aug×{aug_factor}": (Xtr_aug, ytr_aug),
    }.items():
        scaler = StandardScaler().fit(Xtr)
        model = _build_models(use_xgboost=True)[model_name]
        model.fit(scaler.transform(Xtr), ytr)
        trained[tag] = (scaler, model)

    y_test = y[test_idx]
    test_signals = [signals[i] for i in test_idx]

    rng_noise = np.random.default_rng(seed + 1)
    rows = []
    for sigma in noise_levels:
        Xte = np.asarray([
            _features_from_signal(
                _add_gaussian_noise(s, rng_noise, sigma_frac=float(sigma)))
            for s in test_signals
        ], dtype=np.float64)
        rec = {"noise_sigma_frac": float(sigma)}
        for tag, (scaler, model) in trained.items():
            y_pred = model.predict(scaler.transform(Xte))
            rec[f"f1_{tag}"] = round(
                f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
            rec[f"acc_{tag}"] = round(accuracy_score(y_test, y_pred), 4)
        rows.append(rec)
        if verbose:
            parts = "  ".join(f"{t}: {rec[f'f1_{t}']:.4f}" for t in trained)
            print(f"    σ={sigma:.2f}  {parts}")

    out = pd.DataFrame(rows)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.RESULTS_DIR, "robustness_curve.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in trained:
        ax.plot(out["noise_sigma_frac"], out[f"f1_{tag}"],
                marker="o", label=tag)
    ax.set_xlabel("рівень шуму σ (частка std сигналу)")
    ax.set_ylabel("f1_macro на тесті")
    ax.set_title(f"Стійкість до спотворень сигналу ({model_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "robustness_curve.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/robustness_curve.csv (+ .png)")
    return out


if __name__ == "__main__":
    print("== A. Свіп augment_factor ==")
    augment_factor_sweep()
    print("\n== B. Крива робастності ==")
    robustness_curve()
