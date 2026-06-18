import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

from . import config as cfg
from .feature_extraction import FEATURE_NAMES


def feature_importance_report(features_csv: str = cfg.FEATURES_CSV,
                              model_name: str = "RandomForest",
                              top_k: int = 20,
                              verbose: bool = True) -> pd.DataFrame:
    from .train_ml_models import _build_models

    cfg.set_global_seeds()
    df = pd.read_csv(features_csv)
    feature_cols = list(FEATURE_NAMES)
    X = df[feature_cols].values.astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(df["class"].values)
    groups = df["source_image"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                            random_state=cfg.RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))

    scaler = StandardScaler().fit(X[tr])
    model = _build_models(use_xgboost=True)[model_name]
    model.fit(scaler.transform(X[tr]), y[tr])

    if hasattr(model, "feature_importances_"):
        builtin = model.feature_importances_
    else:
        builtin = np.full(len(feature_cols), np.nan)

    perm = permutation_importance(
        model, scaler.transform(X[te]), y[te],
        scoring="f1_macro", n_repeats=20, random_state=cfg.RANDOM_STATE, n_jobs=-1)

    out = pd.DataFrame({
        "feature": feature_cols,
        "builtin_importance": np.round(builtin, 5),
        "perm_importance_mean": np.round(perm.importances_mean, 5),
        "perm_importance_std": np.round(perm.importances_std, 5),
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.RESULTS_DIR, "feature_importances.csv"),
               index=False)

    top = out.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["perm_importance_mean"],
            xerr=top["perm_importance_std"], color="#4c72b0")
    ax.set_xlabel("Перестановочна важливість (падіння f1_macro)")
    ax.set_title(f"Топ-{top_k} ознак ({model_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.RESULTS_DIR, "feature_importances.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Інтерпретованість ({model_name}), перестановочна важливість (TEST):")
        for _, r in out.head(top_k).iterrows():
            print(f"    {r['feature']:18s} perm={r['perm_importance_mean']:+.4f} "
                  f"± {r['perm_importance_std']:.4f}  builtin={r['builtin_importance']:.4f}")
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/feature_importances.csv (+ .png)")
    return out


if __name__ == "__main__":
    feature_importance_report()
