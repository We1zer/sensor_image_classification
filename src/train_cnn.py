import os
import json
import numpy as np
import pandas as pd
import cv2

from . import config as cfg


def _load_pulse_image(path: str, size: tuple[int, int]) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def build_cnn_dataset(features_csv: str = cfg.FEATURES_CSV,
                      img_size: tuple[int, int] = cfg.CNN_IMG_SIZE
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not os.path.exists(features_csv):
        raise FileNotFoundError(features_csv)

    df = pd.read_csv(features_csv)
    if df.empty:
        raise ValueError("Файл ознак порожній.")

    if "pulse_image_path" not in df.columns:
        raise KeyError("У датасеті немає колонки pulse_image_path.")

    Xs, ys, groups = [], [], []
    class_names = sorted(df["class"].unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for _, row in df.iterrows():
        path = row["pulse_image_path"]
        if not isinstance(path, str) or path == "":
            continue
        img = _load_pulse_image(cfg.to_abs(path), img_size)
        if img is None:
            continue
        Xs.append(img)
        ys.append(cls_to_idx[row["class"]])
        groups.append(row["source_image"])

    if not Xs:
        raise ValueError("Не вдалось завантажити жодного зображення імпульсу.")

    X = np.expand_dims(np.stack(Xs), -1)
    y = np.array(ys, dtype=np.int64)
    groups = np.array(groups)
    return X, y, groups, class_names


def train_cnn(epochs: int = 15, batch_size: int = 32, verbose: bool = True) -> dict:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError(
            "TensorFlow не встановлено. Встановіть: pip install tensorflow")

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    cfg.set_global_seeds()

    X, y, groups, class_names = build_cnn_dataset()
    if verbose:
        print(f"  CNN dataset: X={X.shape}, classes={class_names}")

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                            random_state=cfg.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_classes = len(class_names)
    h, w = cfg.CNN_IMG_SIZE

    model = models.Sequential([
        layers.Input(shape=(h, w, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=2 if verbose else 0,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0)

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    model.save(os.path.join(cfg.MODELS_DIR, "cnn_model.h5"))
    with open(os.path.join(cfg.MODELS_DIR, "cnn_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "img_size": list(cfg.CNN_IMG_SIZE),
            "test_accuracy": float(acc),
        }, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n[OK] CNN test accuracy: {acc:.4f}")
        print(report)

    return {
        "model": model,
        "accuracy": acc,
        "history": history.history,
        "class_names": class_names,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion": cm,
        "report": report,
    }


def cross_validate_cnn2d(n_splits: int = 5,
                         epochs: int = 60,
                         batch_size: int = 32,
                         fusion_strategy: str = "weighted",
                         save_plots: bool = True,
                         verbose: bool = True) -> dict:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
    except ImportError:
        raise ImportError("TensorFlow не встановлено: pip install tensorflow")

    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from . import fusion as fz

    cfg.set_global_seeds()

    df = pd.read_csv(cfg.FEATURES_CSV)
    if "is_augmented" in df.columns:
        df = df[~df["is_augmented"].astype(bool)]
    df = df[df["pulse_image_path"].fillna("").astype(str).str.len() > 0]
    df = df.reset_index(drop=True)

    class_names = sorted(df["class"].unique().tolist())
    c2i = {c: i for i, c in enumerate(class_names)}
    h, w = cfg.CNN_IMG_SIZE

    Xs, ys, groups, image_key, weights, keep = [], [], [], [], [], []
    for i, row in df.iterrows():
        img = _load_pulse_image(cfg.to_abs(row["pulse_image_path"]), cfg.CNN_IMG_SIZE)
        if img is None:
            continue
        Xs.append(img)
        ys.append(c2i[row["class"]])
        groups.append(row["source_image"])
        image_key.append(row["image_path"])
        weights.append(fz.pulse_quality_weight(
            row.get("local_coverage", 1.0), row.get("amp_to_noise", 1.0)))
    X = np.expand_dims(np.stack(Xs), -1)
    y = np.array(ys, dtype=np.int64)
    groups = np.array(groups)
    image_key = np.array(image_key)
    weights = np.array(weights, dtype=float)
    n_classes = len(class_names)
    if verbose:
        print(f"  2D-CNN dataset: X={X.shape}, класів={n_classes}")

    def _build():
        m = models.Sequential([
            layers.Input(shape=(h, w, 1)),
            layers.Conv2D(16, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ])
        m.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full((len(X), n_classes), np.nan)
    fold_pulse_f1, fold_img_f1 = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        gss = GroupShuffleSplit(1, test_size=0.15, random_state=cfg.RANDOM_STATE)
        tr_in, val_in = next(gss.split(X[tr], y[tr], groups=groups[tr]))
        tr2, val2 = tr[tr_in], tr[val_in]
        model = _build()
        model.fit(X[tr2], y[tr2], validation_data=(X[val2], y[val2]),
                  epochs=epochs, batch_size=batch_size, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor="val_loss",
                             patience=12, restore_best_weights=True),
                             callbacks.ReduceLROnPlateau(monitor="val_loss",
                             patience=5, factor=0.5, min_lr=1e-5)])
        proba = model.predict(X[te], verbose=0)
        oof[te] = proba
        fold_pulse_f1.append(f1_score(y[te], np.argmax(proba, 1),
                                      average="macro", zero_division=0))
        ti, pi = [], []
        for img in pd.unique(image_key[te]):
            m = image_key[te] == img
            ti.append(y[te][m][0])
            pi.append(fz.fuse(proba[m], strategy=fusion_strategy,
                              weights=weights[te][m]))
        fold_img_f1.append(f1_score(ti, pi, average="macro", zero_division=0))
        if verbose:
            print(f"    фолд {fold}/{n_splits}: pulse {fold_pulse_f1[-1]:.4f}, "
                  f"image {fold_img_f1[-1]:.4f}")
        tf.keras.backend.clear_session()

    ev = ~np.isnan(oof).any(axis=1)
    pulse_f1 = f1_score(y[ev], np.argmax(oof[ev], 1), average="macro", zero_division=0)
    ti, pi = [], []
    for img in pd.unique(image_key[ev]):
        m = (image_key == img) & ev
        ti.append(y[m][0])
        pi.append(fz.fuse(oof[m], strategy=fusion_strategy, weights=weights[m]))
    img_f1 = f1_score(ti, pi, average="macro", zero_division=0)
    img_acc = accuracy_score(ti, pi)
    cm = confusion_matrix(ti, pi)

    results_df = pd.DataFrame([
        {"level": "pulse", "f1_macro_oof": round(pulse_f1, 4),
         "f1_macro_fold_mean": round(float(np.mean(fold_pulse_f1)), 4),
         "f1_macro_fold_std": round(float(np.std(fold_pulse_f1)), 4)},
        {"level": "image", "f1_macro_oof": round(img_f1, 4),
         "accuracy_oof": round(img_acc, 4),
         "f1_macro_fold_mean": round(float(np.mean(fold_img_f1)), 4),
         "f1_macro_fold_std": round(float(np.std(fold_img_f1)), 4)},
    ])
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(cfg.RESULTS_DIR, "cnn2d_cv.csv"), index=False)
    if save_plots:
        from .visualization import plot_confusion
        plot_confusion(cm, class_names, title=f"Image-level (2D-CNN, {fusion_strategy})",
                       out_name="confusion_image_CNN2D.png")
    if verbose:
        print(f"  2D-CNN | pulse f1 {pulse_f1:.4f} | image f1 {img_f1:.4f} "
              f"(±{np.std(fold_img_f1):.4f}) acc {img_acc:.4f}")
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/cnn2d_cv.csv")
    return {"results": results_df, "pulse_f1_oof": pulse_f1,
            "image_f1_oof": img_f1, "class_names": class_names}


if __name__ == "__main__":
    train_cnn()
