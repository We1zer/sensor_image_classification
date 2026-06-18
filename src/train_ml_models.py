import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

from . import config as cfg
from .feature_extraction import FEATURE_NAMES


PARAM_GRIDS: dict[str, dict] = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 15],
        "min_samples_split": [2, 5],
    },
    "SVM": {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", 0.01, 0.1],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
    },
    "DecisionTree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
}


def _build_models(random_state: int = cfg.RANDOM_STATE,
                  use_xgboost: bool = True) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale",
                   probability=True, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state),
    }
    if use_xgboost and HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
        )
    return models


def _evaluate(model, X_test_s: np.ndarray,
              y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test_s)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0),
    }, y_pred


def _select_X_y(df: pd.DataFrame, use_file_number: bool = False
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_cols = list(FEATURE_NAMES)
    if use_file_number:
        feature_cols = feature_cols + ["file_number"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["class"].values
    groups = df["source_image"].values
    return X, y, groups, feature_cols


def _grid_combinations(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def _tune_with_gridsearch(name: str, base_model, X_train_s: np.ndarray,
                          y_train: np.ndarray, groups_train: np.ndarray,
                          verbose: bool = False) -> tuple:
    grid = PARAM_GRIDS.get(name)
    if not grid:
        return None, None

    n_groups = len(np.unique(groups_train))
    n_splits = max(2, min(3, n_groups))
    cv = GroupKFold(n_splits=n_splits)

    if verbose:
        print(f"  → GridSearchCV для {name}: "
              f"{_grid_combinations(grid)} комбінацій, cv={n_splits}")

    gs = GridSearchCV(
        base_model, grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, refit=True,
    )
    gs.fit(X_train_s, y_train, groups=groups_train)
    return gs.best_estimator_, gs.best_params_


def _build_top_estimator_list(top_names: list[str], use_xgboost: bool):
    fresh = _build_models(use_xgboost=use_xgboost)
    return [(n, fresh[n]) for n in top_names if n in fresh]


def train_and_compare(features_csv: str = cfg.FEATURES_CSV,
                      use_file_number: bool = False,
                      use_xgboost: bool | None = None,
                      use_ensembles: bool = True,
                      tune_top_k: int = 0,
                      verbose: bool = True) -> dict:
    cfg.set_global_seeds()

    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Не знайдено {features_csv}. Спочатку запустіть dataset_builder.")

    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError("Файл ознак порожній.")

    if use_xgboost is None:
        use_xgboost = HAS_XGBOOST

    X, y_raw, groups, feature_cols = _select_X_y(df, use_file_number)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                            random_state=cfg.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    n_test_full = len(test_idx)
    n_test_augmented = 0
    if "is_augmented" in df.columns:
        is_aug_arr = df["is_augmented"].astype(bool).values
        n_test_augmented = int(is_aug_arr[test_idx].sum())
        test_idx = test_idx[~is_aug_arr[test_idx]]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    if verbose:
        n_train_aug = int(df.iloc[train_idx]["is_augmented"].sum()) \
            if "is_augmented" in df.columns else 0
        print(f"  Train: {len(train_idx)} (з них аугментованих: {n_train_aug})")
        print(f"  Test:  {len(test_idx)} (оригіналів). "
              f"Виключено з test {n_test_augmented} аугментованих "
              f"із початкових {n_test_full}.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows: list[dict] = []
    fitted: dict = {}

    base_models = _build_models(use_xgboost=use_xgboost)
    for name, model in base_models.items():
        if verbose:
            print(f"  → Тренуємо {name} ...")
        t0 = time.time()
        model.fit(X_train_s, y_train)
        m, _ = _evaluate(model, X_test_s, y_test)
        rows.append({
            "model": name, "kind": "base",
            "train_sec": round(time.time() - t0, 3),
            "best_params": "",
            **m,
        })
        fitted[name] = model

    if tune_top_k and tune_top_k > 0:
        df_tmp = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
        top_to_tune = list(df_tmp.head(tune_top_k)["model"])
        if verbose:
            print(f"  [GridSearchCV] tune top {tune_top_k}: {top_to_tune}")

        fresh_for_tuning = _build_models(use_xgboost=use_xgboost)
        for name in top_to_tune:
            if name not in fresh_for_tuning:
                continue
            t0 = time.time()
            try:
                best_est, best_params = _tune_with_gridsearch(
                    name, fresh_for_tuning[name],
                    X_train_s, y_train, groups_train, verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  [!] GridSearchCV {name}: {e}")
                continue
            if best_est is None:
                continue
            m, _ = _evaluate(best_est, X_test_s, y_test)
            tuned_name = f"{name}_tuned"
            rows.append({
                "model": tuned_name, "kind": "tuned",
                "train_sec": round(time.time() - t0, 3),
                "best_params": json.dumps(best_params, ensure_ascii=False),
                **m,
            })
            fitted[tuned_name] = best_est
            if verbose:
                print(f"     best params: {best_params}")

    if use_ensembles:
        df_base = pd.DataFrame([r for r in rows if r["kind"] == "base"]) \
            .sort_values("f1_macro", ascending=False)
        top3_names = list(df_base.head(3)["model"])
        all_base_names = list(df_base["model"])

        if verbose:
            print(f"  → Voting (soft) з top-3: {top3_names}")
        try:
            voting_estimators = _build_top_estimator_list(top3_names, use_xgboost)
            voting = VotingClassifier(
                estimators=voting_estimators, voting="soft", n_jobs=-1)
            t0 = time.time()
            voting.fit(X_train_s, y_train)
            m, _ = _evaluate(voting, X_test_s, y_test)
            rows.append({
                "model": "Voting (soft, top-3)", "kind": "ensemble",
                "train_sec": round(time.time() - t0, 3),
                "best_params": json.dumps({"members": top3_names}),
                **m,
            })
            fitted["Voting (soft, top-3)"] = voting
        except Exception as e:
            if verbose:
                print(f"  [!] Voting не зібрався: {e}")

        if verbose:
            print(f"  → Stacking (passthrough, cv=5) з усіх base: {all_base_names}")
        try:
            stack_estimators = _build_top_estimator_list(
                all_base_names, use_xgboost)

            n_groups_train = len(np.unique(groups_train))
            stack_cv = min(5, max(2, n_groups_train // 2))

            stacking = StackingClassifier(
                estimators=stack_estimators,
                final_estimator=LogisticRegression(
                    max_iter=5000, C=1.0,
                    random_state=cfg.RANDOM_STATE),
                cv=stack_cv,
                n_jobs=1,
                passthrough=True,
                stack_method="predict_proba",
            )
            t0 = time.time()
            stacking.fit(X_train_s, y_train)
            m, _ = _evaluate(stacking, X_test_s, y_test)
            rows.append({
                "model": f"Stacking (all base, passthrough, cv={stack_cv})",
                "kind": "ensemble",
                "train_sec": round(time.time() - t0, 3),
                "best_params": json.dumps({
                    "members": all_base_names,
                    "final": "LogisticRegression",
                    "passthrough": True,
                    "cv": stack_cv,
                }),
                **m,
            })
            fitted[f"Stacking (all base, passthrough, cv={stack_cv})"] = stacking
        except Exception as e:
            if verbose:
                import traceback as _tb
                print(f"  [!] Stacking не зібрався: {e}")
                _tb.print_exc()

    results_df = pd.DataFrame(rows).sort_values(
        "f1_macro", ascending=False).reset_index(drop=True)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]

    y_pred_best = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(
        y_test, y_pred_best, target_names=le.classes_, zero_division=0)

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(cfg.MODELS_DIR, "best_ml_model.pkl"))
    joblib.dump(scaler,     os.path.join(cfg.MODELS_DIR, "scaler.pkl"))
    joblib.dump(le,         os.path.join(cfg.MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(cfg.MODELS_DIR, "best_model_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "best_name": best_name,
            "feature_cols": feature_cols,
            "use_file_number": use_file_number,
            "use_xgboost": bool(use_xgboost and HAS_XGBOOST),
            "use_ensembles": bool(use_ensembles),
            "tuned_top_k": int(tune_top_k or 0),
        }, f, ensure_ascii=False, indent=2)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(
        os.path.join(cfg.RESULTS_DIR, "model_comparison.csv"), index=False)
    with open(os.path.join(cfg.RESULTS_DIR, "classification_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"Найкраща модель: {best_name}\n\n")
        f.write(report)

    if verbose:
        cols_to_show = ["model", "kind", "accuracy",
                        "precision_macro", "recall_macro", "f1_macro"]
        print()
        print(results_df[cols_to_show].to_string(index=False))
        print()
        print(f"[OK] Найкраща модель: {best_name}")
        print(f"[OK] Збережено у: {cfg.MODELS_DIR}/")
        print()
        print(report)

    return {
        "results": results_df,
        "best_name": best_name,
        "best_model": best_model,
        "report": report,
        "confusion": cm,
        "label_encoder": le,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "y_pred": y_pred_best,
        "class_names": list(le.classes_),
    }


if __name__ == "__main__":
    train_and_compare()
