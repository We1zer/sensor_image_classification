from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import config as cfg


FEATURES_RESULTS_CSV = os.path.join(cfg.RESULTS_DIR,
                                    "features_branch_results.csv")


def run_feature_branch(n_splits: int = 5,
                       do_cv: bool = True,
                       do_fusion: bool = True,
                       do_dimreduction: bool = True,
                       model_for_fusion: str = "RandomForest",
                       progress=None,
                       verbose: bool = True) -> dict:
    from .train_ml_models import cross_validate_models, cross_validate_dimreduction
    from .evaluate import cross_validate_with_fusion

    def _tick(pct: int, msg: str) -> None:
        if verbose:
            print(f"\n=== [{pct}%] {msg} ===")
        if progress:
            progress(pct, msg)

    if not os.path.exists(cfg.FEATURES_CSV):
        raise FileNotFoundError(
            f"Немає {cfg.to_rel(cfg.FEATURES_CSV)}. "
            "Спочатку: python -m src.dataset_builder")

    t0 = time.time()
    rows: list[dict] = []
    cv_df = dimred_df = None
    fusion_res = None

    if do_cv:
        _tick(10, "GroupKFold(5) по базових моделях (без file_number)")
        cv = cross_validate_models(n_splits=n_splits, use_file_number=False,
                                   verbose=verbose)
        cv_df = cv["results"]
        for _, r in cv_df.iterrows():
            rows.append({
                "method": f"features + {r['model']}",
                "level": "pulse",
                "f1_macro": round(float(r["f1_macro_mean"]), 4),
                "f1_macro_std": round(float(r["f1_macro_std"]), 4),
                "accuracy": round(float(r["accuracy_mean"]), 4),
                "detail": f"GroupKFold({int(r['n_folds'])})",
            })

    if do_fusion:
        _tick(55, f"Decision-level fusion ({model_for_fusion})")
        fusion_res = cross_validate_with_fusion(
            model_name=model_for_fusion, n_splits=n_splits,
            save_plots=True, verbose=verbose)
        rows.append({
            "method": f"features + {model_for_fusion}",
            "level": "pulse",
            "f1_macro": round(float(fusion_res["pulse_f1_oof"]), 4),
            "f1_macro_std": np.nan,
            "accuracy": np.nan,
            "detail": "OOF, pulse-level",
        })
        best = fusion_res.get("best_strategy")
        for strat, d in (fusion_res.get("image_results") or {}).items():
            rows.append({
                "method": f"features + {model_for_fusion}",
                "level": f"image ({strat})",
                "f1_macro": round(float(d["f1_oof"]), 4),
                "f1_macro_std": round(float(d["f1_fold_std"]), 4),
                "accuracy": round(float(d["acc_oof"]), 4),
                "detail": "OOF, image-level"
                          + (" <- best" if strat == best else ""),
            })

    if do_dimreduction:
        _tick(80, "Зниження розмірності (PCA / LDA)")
        dimred_df = cross_validate_dimreduction(n_splits=n_splits,
                                                verbose=verbose)
        for _, r in dimred_df.iterrows():
            rows.append({
                "method": f"features [{r['reducer']}] + {r['classifier']}",
                "level": "pulse",
                "f1_macro": round(float(r["f1_macro_mean"]), 4),
                "f1_macro_std": round(float(r["f1_macro_std"]), 4),
                "accuracy": np.nan,
                "detail": "dim-reduction",
            })

    _tick(95, "Збереження результатів")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        df.to_csv(FEATURES_RESULTS_CSV, index=False)

    if verbose and not df.empty:
        print()
        print(df.to_string(index=False))
        print(f"\n[OK] {cfg.to_rel(FEATURES_RESULTS_CSV)}")
        print(f"[OK] features-гілка завершена за {time.time() - t0:.0f} с")

    _tick(100, "Готово")
    return {
        "branch": "features",
        "results": df,
        "cv": cv_df,
        "fusion": fusion_res,
        "dimreduction": dimred_df,
        "elapsed_sec": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    run_feature_branch()
