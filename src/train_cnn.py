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


if __name__ == "__main__":
    train_cnn()
