from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import config as cfg


ALL_MODELS_CSV = os.path.join(cfg.RESULTS_DIR, "all_models_comparison.csv")

COLUMNS = [
    "branch", "method", "classifier", "input",
    "pulse_f1", "image_f1_majority", "image_f1_mean_proba",
    "image_f1_weighted", "image_f1_best", "best_strategy", "image_f1_std",
    "n_pulses", "n_images", "train_sec", "source",
    "params_key", "params_full",
]


_KEY_PARAMS = (
    "C", "gamma", "kernel", "n_neighbors", "weights", "max_iter",
    "n_estimators", "max_depth", "learning_rate", "min_samples_split",
    "criterion", "subsample",
)


_STRATEGY_RANK = {"weighted": 2, "mean_proba": 1, "majority": 0}


def _pick_best(scores: dict) -> str:
    known = {s: v for s, v in scores.items() if v is not None}
    if not known:
        return "weighted"
    return max(known, key=lambda s: (known[s], _STRATEGY_RANK.get(s, -1)))


def describe_estimator(est) -> tuple[str, str]:
    from sklearn.pipeline import Pipeline

    pre = ""
    if isinstance(est, Pipeline):
        pre = " + ".join(type(s).__name__ for _, s in est.steps[:-1])
        est = est.steps[-1][1]

    params = est.get_params(deep=False)
    short = ", ".join(f"{k}={params[k]}" for k in _KEY_PARAMS if k in params)
    name = f"{type(est).__name__}({short})"
    if pre:
        name = f"{pre} -> {name}"
    full = json.dumps(params, default=str, ensure_ascii=False, sort_keys=True)
    return name, full


def features_rows(n_splits: int = 5,
                  models: list[str] | None = None,
                  verbose: bool = True) -> list[dict]:
    from .train_ml_models import _build_models
    from .evaluate import cross_validate_with_fusion

    if not os.path.exists(cfg.FEATURES_CSV):
        raise FileNotFoundError(
            f"Немає {cfg.to_rel(cfg.FEATURES_CSV)}. "
            "Спочатку: python -m src.dataset_builder")

    df = pd.read_csv(cfg.FEATURES_CSV)
    is_aug = (df["is_augmented"].astype(bool) if "is_augmented" in df.columns
              else pd.Series(False, index=df.index))
    n_pulses = int((~is_aug).sum())
    n_images = int(df.loc[~is_aug, "image_path"].nunique())

    built = _build_models(use_xgboost=True)
    names = models or list(built)

    rows = []
    for name in names:
        if name not in built:
            continue
        t0 = time.time()
        if verbose:
            print(f"  -> features + {name}")
        res = cross_validate_with_fusion(
            model_name=name, n_splits=n_splits,
            save_plots=False, save_csv=False, verbose=False)

        img = res["image_results"]
        best = _pick_best({s: d["f1_oof"] for s, d in img.items()})
        key, full = describe_estimator(built[name])
        rows.append({
            "branch": "features",
            "method": f"features + {name}",
            "classifier": name,
            "input": "35 числових ознак",
            "pulse_f1": round(float(res["pulse_f1_oof"]), 4),
            "image_f1_majority": round(float(img["majority"]["f1_oof"]), 4),
            "image_f1_mean_proba": round(float(img["mean_proba"]["f1_oof"]), 4),
            "image_f1_weighted": round(float(img["weighted"]["f1_oof"]), 4),
            "image_f1_best": round(float(img[best]["f1_oof"]), 4),
            "best_strategy": best,
            "image_f1_std": round(float(img[best]["f1_fold_std"]), 4),
            "n_pulses": n_pulses,
            "n_images": n_images,
            "train_sec": round(time.time() - t0, 1),
            "source": "computed",
            "params_key": key,
            "params_full": full,
        })
        if verbose:
            r = rows[-1]
            print(f"     pulse {r['pulse_f1']:.4f} | image "
                  f"{r['image_f1_best']:.4f} ({best}) | {r['train_sec']}s")
    return rows


def _image_row_from_evaluate(row: dict, classifier: str, spec_name: str,
                             n_feat: int | None, params: tuple[str, str],
                             prefix: str) -> dict:
    strategies = ("majority", "mean_proba", "weighted")
    vals = {s: row.get(f"image_f1_{s}") for s in strategies}
    if all(v is None for v in vals.values()):
        vals[row.get("fusion", "weighted")] = row.get("image_f1_oof")
    known = {s: v for s, v in vals.items() if v is not None}
    best = _pick_best(vals)
    key, full = params
    return {
        "branch": "image",
        "method": row["method"],
        "classifier": classifier,
        "input": f"{spec_name}" + (f" ({prefix} {n_feat})" if n_feat else ""),
        "pulse_f1": row.get("pulse_f1_oof"),
        "image_f1_majority": vals.get("majority"),
        "image_f1_mean_proba": vals.get("mean_proba"),
        "image_f1_weighted": vals.get("weighted"),
        "image_f1_best": known.get(best),
        "best_strategy": best,
        "image_f1_std": row.get(f"image_f1_{best}_std",
                                row.get("image_f1_fold_std")),
        "n_pulses": row.get("n_pulses"),
        "n_images": row.get("n_images"),
        "train_sec": row.get("train_sec"),
        "source": "computed",
        "params_key": key,
        "params_full": full,
    }


def image_rows(n_splits: int = 5,
               backbone: str = "MobileNetV2",
               do_hog: bool = True,
               do_transfer: bool = True,
               run_cnn: bool = False,
               use_cached_cnn: bool = True,
               verbose: bool = True) -> list[dict]:
    from .image_prep import SPEC_MAIN, build_image_array
    from . import image_methods as im

    data = build_image_array(spec=SPEC_MAIN, verbose=verbose)
    weights, w_source = im.load_quality_weights(data.pulse_id)
    if verbose:
        print(f"  ваги злиття: {w_source}")

    rows: list[dict] = []

    if do_hog:
        hog_clfs = im._hog_classifiers()
        n_feat = None
        for r in im.run_hog(data, weights=weights, n_splits=n_splits,
                            all_strategies=True, verbose=verbose):
            head = r["method"].split("+")[-1].strip()
            rows.append(_image_row_from_evaluate(
                r, head, data.spec.name, n_feat,
                describe_estimator(hog_clfs[head]), "HOG"))

    if do_transfer:
        heads = im.transfer_heads()
        for r in im.run_transfer(data, backbone=backbone, weights=weights,
                                 n_splits=n_splits, all_strategies=True,
                                 verbose=verbose):
            head = r["method"].split("+")[-1].strip()
            rows.append(_image_row_from_evaluate(
                r, head, data.spec.name, None,
                describe_estimator(heads[head]), "emb"))

    if run_cnn:
        for r in im.run_cnn2d(data, weights=weights, n_splits=n_splits,
                              all_strategies=True, verbose=verbose):
            rows.append(_image_row_from_evaluate(
                r, "CNN2D", data.spec.name, None,
                (f"Conv2D(32,64,128,128)+GroupNorm, head={im.CNN_DEFAULT_HEAD}, "
                 "adam, epochs<=60, EarlyStopping(patience=15)", "{}"), ""))
    elif use_cached_cnn:
        cached = _cached_cnn_row()
        if cached:
            rows.append(cached)

    from . import image_model_store as store
    for name, members in store.ENSEMBLES.items():
        oof_names = [store.OOF_NAME.get(m, m) for m in members]
        r = im.ensemble_row(name, oof_names, data.spec.name)
        if r is None:
            if verbose:
                print(f"  [--] {name}: немає OOF усіх учасників на "
                      f"{data.spec.name} — пропускаю")
            continue
        row = _image_row_from_evaluate(
            r, "ensemble", data.spec.name, None,
            ("mean(" + ", ".join(members) + "), рівні ваги", "{}"), "")
        row["source"] = "OOF-ансамбль"
        rows.append(row)

    return rows


def _cnn_head() -> str:
    from .image_methods import CNN_DEFAULT_HEAD
    return CNN_DEFAULT_HEAD


def _cached_cnn_row() -> dict | None:
    from .image_methods import IMAGE_RESULTS_CSV

    if not os.path.exists(IMAGE_RESULTS_CSV):
        return None
    try:
        df = pd.read_csv(IMAGE_RESULTS_CSV)
    except Exception:
        return None
    from .image_prep import SPEC_MAIN
    hit = df[(df["method"] == "2D-CNN (fixed)") & (df["spec"] == SPEC_MAIN.name)]
    if hit.empty:
        return None
    r = hit.iloc[0].to_dict()
    row = _image_row_from_evaluate(
        r, "CNN2D", r.get("spec", ""), None,
        ("Conv2D(32,64,128,128) + GroupNormalization(groups=-1), "
         f"head={_cnn_head()}, adam, epochs<=60, batch=32, "
         "EarlyStopping(val_loss, patience=15), ReduceLROnPlateau(patience=6)",
         "{}"), "")
    row["source"] = "cached (image_branch_results.csv)"
    return row


def compare_all(n_splits: int = 5,
                run_cnn: bool = False,
                use_cached_cnn: bool = True,
                backbone: str = "MobileNetV2",
                do_features: bool = True,
                do_image: bool = True,
                save: bool = True,
                progress=None,
                verbose: bool = True) -> pd.DataFrame:
    def _tick(pct: int, msg: str) -> None:
        if verbose:
            print(f"\n=== [{pct}%] {msg} ===")
        if progress:
            progress(pct, msg)

    t0 = time.time()
    rows: list[dict] = []

    if do_features:
        _tick(10, "Гілка ознак: усі моделі, pulse + 3 стратегії злиття")
        rows += features_rows(n_splits=n_splits, verbose=verbose)

    if do_image:
        _tick(55, "Гілка картинок: HOG + transfer"
                  + (" + 2D-CNN" if run_cnn else " (2D-CNN з кешу)"))
        rows += image_rows(n_splits=n_splits, backbone=backbone,
                           run_cnn=run_cnn, use_cached_cnn=use_cached_cnn,
                           verbose=verbose)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].sort_values("image_f1_best", ascending=False)
    df = df.reset_index(drop=True)

    if save:
        _tick(95, "Збереження")
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        df.to_csv(ALL_MODELS_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        show = ["branch", "method", "pulse_f1", "image_f1_best",
                "best_strategy", "train_sec"]
        print()
        print(df[show].to_string(index=False))
        print(f"\n[OK] {cfg.to_rel(ALL_MODELS_CSV)}")
        print(f"[OK] порівняння завершено за {time.time() - t0:.0f} с")

    _tick(100, "Готово")
    return df


def load_comparison() -> pd.DataFrame | None:
    if not os.path.exists(ALL_MODELS_CSV):
        return None
    try:
        return pd.read_csv(ALL_MODELS_CSV)
    except Exception:
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Єдина таблиця порівняння всіх моделей обох гілок.")
    ap.add_argument("--with-cnn", action="store_true",
                    help="перерахувати 2D-CNN замість читання з кешу (~25 хв)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--branch", choices=["features", "image", "both"],
                    default="both")
    a = ap.parse_args()
    compare_all(n_splits=a.splits, run_cnn=a.with_cnn,
                do_features=a.branch in ("features", "both"),
                do_image=a.branch in ("image", "both"))
