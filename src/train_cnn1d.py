import os
import json
import numpy as np
import pandas as pd

from . import config as cfg
from .feature_extraction import _safe_signal
from . import fusion as fz


def _require_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow не встановлено. Встановіть: pip install tensorflow"
        ) from e


def _load_signal(path: str, length: int) -> np.ndarray | None:
    ap = cfg.to_abs(path)
    if not ap or not os.path.exists(ap):
        return None
    sig = np.load(ap).astype(np.float64)
    sig = _safe_signal(sig)
    if len(sig) != length:
        src_x = np.linspace(0.0, 1.0, len(sig))
        dst_x = np.linspace(0.0, 1.0, length)
        sig = np.interp(dst_x, src_x, sig)
    sig = (sig - float(np.mean(sig))) / cfg.CNN_SIG_SCALE
    return sig.astype(np.float32)


def build_signal_dataset(features_csv: str = cfg.FEATURES_CSV,
                         length: int = cfg.CNN_SIG_LEN,
                         originals_only: bool = True):
    df = pd.read_csv(features_csv)
    if originals_only and "is_augmented" in df.columns:
        df = df[~df["is_augmented"].astype(bool)]
    df = df[df["pulse_signal_path"].fillna("").astype(str).str.len() > 0]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Немає сигналів. Перебудуйте: build_dataset(save_pulse_signals=True).")

    class_names = sorted(df["class"].unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    Xs, ys, groups, image_key, weights = [], [], [], [], []
    for _, row in df.iterrows():
        sig = _load_signal(row["pulse_signal_path"], length)
        if sig is None:
            continue
        Xs.append(sig)
        ys.append(cls_to_idx[row["class"]])
        groups.append(row["source_image"])
        image_key.append(row["image_path"])
        weights.append(fz.pulse_quality_weight(
            row.get("local_coverage", 1.0), row.get("amp_to_noise", 1.0)))

    X = np.expand_dims(np.stack(Xs), -1)
    return (X, np.asarray(ys, dtype=np.int64), np.asarray(groups),
            np.asarray(image_key), np.asarray(weights, dtype=float), class_names)


def build_cnn1d(input_len: int, n_classes: int):
    tf = _require_tf()
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(16, 7, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling1D(2),
        layers.Conv1D(32, 5, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="embedding"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cross_validate_cnn1d(n_splits: int = 5,
                         epochs: int = 100,
                         batch_size: int = 32,
                         fusion_strategy: str = "weighted",
                         save_plots: bool = True,
                         verbose: bool = True) -> dict:
    tf = _require_tf()
    from tensorflow.keras import callbacks
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

    cfg.set_global_seeds()

    X, y, groups, image_key, weights, class_names = build_signal_dataset()
    n_classes = len(class_names)
    L = X.shape[1]
    if verbose:
        print(f"  1D-CNN dataset: X={X.shape}, класів={n_classes} {class_names}")

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof_proba = np.full((len(X), n_classes), np.nan, dtype=float)
    fold_pulse_f1, fold_img_f1 = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                random_state=cfg.RANDOM_STATE)
        tr_in, val_in = next(gss.split(X[tr], y[tr], groups=groups[tr]))
        tr2, val2 = tr[tr_in], tr[val_in]

        model = build_cnn1d(L, n_classes)
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8,
                                        factor=0.5, min_lr=1e-5),
        ]
        model.fit(X[tr2], y[tr2],
                  validation_data=(X[val2], y[val2]),
                  epochs=epochs, batch_size=batch_size,
                  callbacks=cbs, verbose=0)

        proba = model.predict(X[te], verbose=0)
        oof_proba[te] = proba

        y_pulse = np.argmax(proba, axis=1)
        fold_pulse_f1.append(
            f1_score(y[te], y_pulse, average="macro", zero_division=0))

        true_img, pred_img = [], []
        df_te_keys = pd.Series(image_key[te])
        for img in df_te_keys.unique():
            mask = image_key[te] == img
            idx_local = np.where(mask)[0]
            p = proba[idx_local]
            w = weights[te][idx_local]
            pred_img.append(fz.fuse(p, strategy=fusion_strategy, weights=w))
            true_img.append(y[te][idx_local][0])
        fold_img_f1.append(
            f1_score(true_img, pred_img, average="macro", zero_division=0))

        if verbose:
            print(f"    фолд {fold}/{n_splits}: pulse f1 {fold_pulse_f1[-1]:.4f}, "
                  f"image f1 {fold_img_f1[-1]:.4f}")
        tf.keras.backend.clear_session()

    evaluated = ~np.isnan(oof_proba).any(axis=1)
    y_eval = y[evaluated]
    pulse_pred = np.argmax(oof_proba[evaluated], axis=1)
    pulse_f1_oof = f1_score(y_eval, pulse_pred, average="macro", zero_division=0)
    pulse_acc_oof = accuracy_score(y_eval, pulse_pred)

    true_img, pred_img = [], []
    keys_eval = image_key[evaluated]
    proba_eval = oof_proba[evaluated]
    w_eval = weights[evaluated]
    y_ev = y[evaluated]
    for img in pd.unique(keys_eval):
        mask = keys_eval == img
        true_img.append(y_ev[mask][0])
        pred_img.append(fz.fuse(proba_eval[mask], strategy=fusion_strategy,
                                weights=w_eval[mask]))
    img_f1_oof = f1_score(true_img, pred_img, average="macro", zero_division=0)
    img_acc_oof = accuracy_score(true_img, pred_img)
    cm_img = confusion_matrix(true_img, pred_img)

    results_df = pd.DataFrame([
        {"level": "pulse", "strategy": "-",
         "f1_macro_oof": round(pulse_f1_oof, 4),
         "accuracy_oof": round(pulse_acc_oof, 4),
         "f1_macro_fold_mean": round(float(np.mean(fold_pulse_f1)), 4),
         "f1_macro_fold_std": round(float(np.std(fold_pulse_f1)), 4)},
        {"level": "image", "strategy": fusion_strategy,
         "f1_macro_oof": round(img_f1_oof, 4),
         "accuracy_oof": round(img_acc_oof, 4),
         "f1_macro_fold_mean": round(float(np.mean(fold_img_f1)), 4),
         "f1_macro_fold_std": round(float(np.std(fold_img_f1)), 4)},
    ])

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(cfg.RESULTS_DIR, "cnn1d_cv.csv"), index=False)
    with open(os.path.join(cfg.RESULTS_DIR, "cnn1d_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"class_names": class_names, "sig_len": L,
                   "n_splits": n_splits,
                   "image_f1_oof": float(img_f1_oof)}, f,
                  ensure_ascii=False, indent=2)

    if save_plots:
        from .visualization import plot_confusion
        plot_confusion(cm_img, class_names,
                       title=f"Image-level (1D-CNN, {fusion_strategy})",
                       out_name=f"confusion_image_CNN1D_{fusion_strategy}.png")

    if verbose:
        print()
        print(f"  1D-CNN | GroupKFold={n_splits}")
        print(f"    pulse-level : f1 {pulse_f1_oof:.4f} "
              f"(fold {np.mean(fold_pulse_f1):.4f} ± {np.std(fold_pulse_f1):.4f})")
        print(f"    image-level : f1 {img_f1_oof:.4f} "
              f"(fold {np.mean(fold_img_f1):.4f} ± {np.std(fold_img_f1):.4f})  "
              f"acc {img_acc_oof:.4f}")
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/cnn1d_cv.csv")

    return {
        "results": results_df,
        "pulse_f1_oof": pulse_f1_oof,
        "image_f1_oof": img_f1_oof,
        "cm_image": cm_img,
        "class_names": class_names,
    }


if __name__ == "__main__":
    cross_validate_cnn1d()
