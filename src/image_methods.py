from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import fusion as fz
from .image_prep import (
    PrepSpec, ImageData, SPEC_MAIN, SPEC_BASELINE,
    build_image_array, scan_pulse_images, default_specs,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None


IMAGE_RESULTS_CSV = os.path.join(cfg.RESULTS_DIR, "image_branch_results.csv")
IMAGE_ABLATION_CSV = os.path.join(cfg.RESULTS_DIR, "image_input_ablation.csv")
OOF_DIR = os.path.join(cfg.RESULTS_DIR, "oof")
FEATURES_OOF_SPEC = "features35"


def _slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def oof_path(method: str, spec_name: str) -> str:
    return os.path.join(OOF_DIR, f"{_slug(method)}__{spec_name}.npz")


def save_oof(method: str, data: "ImageData", oof: np.ndarray,
             weights: np.ndarray, fold_test_idx: list[np.ndarray]) -> str:
    fold = np.full(len(data.y), -1, dtype=int)
    for k, te in enumerate(fold_test_idx):
        fold[te] = k
    return save_oof_arrays(method, data.spec.name, oof, data.y, weights, fold,
                           data.image_key, data.pulse_id, data.class_names)


def save_oof_arrays(method: str, spec_name: str, oof, y, weights, fold,
                    image_key, pulse_id, class_names) -> str:
    os.makedirs(OOF_DIR, exist_ok=True)
    p = oof_path(method, spec_name)
    np.savez_compressed(p, oof=oof, y=y, weights=weights, fold=fold,
                        image_key=np.asarray(image_key, dtype=str),
                        pulse_id=np.asarray(pulse_id, dtype=str),
                        class_names=np.asarray(class_names, dtype=str))
    return p


def _folds_from_keys(y: np.ndarray, image_key: np.ndarray,
                     n_splits: int = 5) -> np.ndarray:
    groups = np.array([k.split("/", 1)[1] + ".png" for k in image_key])
    fold = np.full(len(y), -1, dtype=int)
    gkf = GroupKFold(n_splits=max(2, min(n_splits, len(np.unique(groups)))))
    for k, (_, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups=groups)):
        fold[te] = k
    return fold


def ensemble_row(name: str, members: list, spec_name: str) -> dict | None:
    zs = []
    for m in members:
        method, spec = m if isinstance(m, tuple) else (m, spec_name)
        p = oof_path(method, spec)
        if not os.path.exists(p):
            return None
        z = dict(np.load(p))
        if "fold" not in z:
            z["fold"] = _folds_from_keys(z["y"], z["image_key"])
        zs.append(z)
    ref = zs[0]
    for z in zs[1:]:
        if (list(z["pulse_id"]) != list(ref["pulse_id"])
                or list(z.get("class_names", [])) != list(ref.get("class_names", []))
                or not np.array_equal(z["fold"], ref["fold"])):
            return None
    members = [m[0] if isinstance(m, tuple) else m for m in members]
    oof = np.mean([z["oof"] for z in zs], axis=0)
    y, w, key, fold = ref["y"], ref["weights"], ref["image_key"], ref["fold"]
    row = {
        "method": name, "spec": spec_name,
        "n_pulses": int(len(y)), "n_images": int(len(np.unique(key))),
        "pulse_f1_oof": round(float(f1_score(y, oof.argmax(1), average="macro",
                                             zero_division=0)), 4),
        "pulse_acc_oof": round(float(accuracy_score(y, oof.argmax(1))), 4),
        "fusion": "weighted", "members": " + ".join(members),
    }
    for strat in fz.STRATEGIES:
        ti, pi = _fuse_by_image(y, oof, key, w, strat)
        row[f"image_f1_{strat}"] = round(float(
            f1_score(ti, pi, average="macro", zero_division=0)), 4)
        per_fold = []
        for k in np.unique(fold):
            m_ = fold == k
            tf_, pf_ = _fuse_by_image(y[m_], oof[m_], key[m_], w[m_], strat)
            per_fold.append(f1_score(tf_, pf_, average="macro", zero_division=0))
        row[f"image_f1_{strat}_std"] = round(float(np.std(per_fold)), 4)
    row["image_f1_oof"] = row["image_f1_weighted"]
    row["image_f1_fold_std"] = row["image_f1_weighted_std"]
    return row


def load_quality_weights(pulse_id: np.ndarray) -> tuple[np.ndarray, str]:
    uniform = np.ones(len(pulse_id), dtype=float)
    if not os.path.exists(cfg.FEATURES_CSV):
        return uniform, "uniform (pulse_features.csv відсутній)"
    try:
        df = pd.read_csv(cfg.FEATURES_CSV)
        if not {"pulse_id", "local_coverage", "amp_to_noise"} <= set(df.columns):
            return uniform, "uniform (немає потрібних колонок)"
        df = df.drop_duplicates("pulse_id").set_index("pulse_id")
        w = []
        n_missing = 0
        for pid in pulse_id:
            if pid in df.index:
                r = df.loc[pid]
                w.append(fz.pulse_quality_weight(
                    r["local_coverage"], r["amp_to_noise"]))
            else:
                n_missing += 1
                w.append(1.0)
        note = "pulse_features.csv"
        if n_missing:
            note += f" ({n_missing} без ваги -> 1.0)"
        return np.asarray(w, dtype=float), note
    except Exception:
        return uniform, "uniform (помилка читання)"


def evaluate_oof(data: ImageData,
                 fit_predict,
                 method: str,
                 weights: np.ndarray | None = None,
                 n_splits: int = 5,
                 fusion_strategy: str = "weighted",
                 save_confusion: bool = False,
                 all_strategies: bool = False,
                 verbose: bool = True) -> dict:
    X, y = data.X, data.y
    groups, image_key = data.groups, data.image_key
    n_classes = data.n_classes

    if weights is None:
        weights = np.ones(len(y), dtype=float)

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.full((len(y), n_classes), np.nan)
    fold_pulse, fold_image = [], []
    fold_test_idx: list[np.ndarray] = []
    t0 = time.time()

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        proba = fit_predict(X[tr], y[tr], X[te])
        oof[te] = proba
        fold_test_idx.append(te)

        fold_pulse.append(f1_score(y[te], np.argmax(proba, axis=1),
                                   average="macro", zero_division=0))
        true_i, pred_i = _fuse_by_image(
            y[te], proba, image_key[te], weights[te], fusion_strategy)
        fold_image.append(f1_score(true_i, pred_i, average="macro",
                                   zero_division=0))
        if verbose:
            print(f"    фолд {fold}/{n_splits}: pulse {fold_pulse[-1]:.4f}, "
                  f"image {fold_image[-1]:.4f}")

    ev = ~np.isnan(oof).any(axis=1)
    pulse_f1 = f1_score(y[ev], np.argmax(oof[ev], axis=1),
                        average="macro", zero_division=0)
    pulse_acc = accuracy_score(y[ev], np.argmax(oof[ev], axis=1))

    true_i, pred_i = _fuse_by_image(
        y[ev], oof[ev], image_key[ev], weights[ev], fusion_strategy)
    image_f1 = f1_score(true_i, pred_i, average="macro", zero_division=0)
    image_acc = accuracy_score(true_i, pred_i)

    if save_confusion:
        from .visualization import plot_confusion
        cm = confusion_matrix(true_i, pred_i)
        safe = method.replace(" ", "_").replace("/", "-")
        plot_confusion(cm, data.class_names,
                       title=f"Image-level ({method}, {fusion_strategy})",
                       out_name=f"confusion_image_{safe}.png")

    row = {
        "method": method,
        "spec": data.spec.name,
        "n_pulses": int(ev.sum()),
        "n_images": int(len(true_i)),
        "pulse_f1_oof": round(float(pulse_f1), 4),
        "pulse_acc_oof": round(float(pulse_acc), 4),
        "image_f1_oof": round(float(image_f1), 4),
        "image_acc_oof": round(float(image_acc), 4),
        "image_f1_fold_mean": round(float(np.mean(fold_image)), 4),
        "image_f1_fold_std": round(float(np.std(fold_image)), 4),
        "fusion": fusion_strategy,
        "train_sec": round(time.time() - t0, 1),
    }

    save_oof(method, data, oof, weights, fold_test_idx)

    if all_strategies:
        for strat in fz.STRATEGIES:
            ti_s, pi_s = _fuse_by_image(
                y[ev], oof[ev], image_key[ev], weights[ev], strat)
            row[f"image_f1_{strat}"] = round(float(
                f1_score(ti_s, pi_s, average="macro", zero_division=0)), 4)
            per_fold = []
            for te in fold_test_idx:
                tf_, pf_ = _fuse_by_image(
                    y[te], oof[te], image_key[te], weights[te], strat)
                per_fold.append(f1_score(tf_, pf_, average="macro",
                                         zero_division=0))
            row[f"image_f1_{strat}_std"] = round(float(np.std(per_fold)), 4)

    if verbose:
        print(f"  {method:34s} | pulse {pulse_f1:.4f} | image {image_f1:.4f} "
              f"(±{np.std(fold_image):.4f}) | {row['train_sec']}s")
    return row


def _fuse_by_image(y_true: np.ndarray, proba: np.ndarray,
                   image_key: np.ndarray, weights: np.ndarray,
                   strategy: str) -> tuple[list, list]:
    true_i, pred_i = [], []
    for key in pd.unique(image_key):
        m = image_key == key
        true_i.append(y_true[m][0])
        pred_i.append(fz.fuse(proba[m], strategy=strategy, weights=weights[m]))
    return true_i, pred_i


def hog_transform(X: np.ndarray,
                  orientations: int = 9,
                  pixels_per_cell: tuple[int, int] = (16, 16),
                  cells_per_block: tuple[int, int] = (2, 2),
                  verbose: bool = True) -> np.ndarray:
    from skimage.feature import hog

    n, h, w, c = X.shape
    feats = []
    for i in range(n):
        per_channel = []
        for ch in range(c):
            per_channel.append(hog(
                X[i, :, :, ch],
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                feature_vector=True,
            ))
        feats.append(np.concatenate(per_channel))
    out = np.asarray(feats, dtype=np.float32)
    if verbose:
        print(f"  HOG: {X.shape} -> {out.shape}")
    return out


def _hog_classifiers(random_state: int = cfg.RANDOM_STATE) -> dict:
    models = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=random_state)),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0,
                                       random_state=random_state)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=random_state, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1,
        )
    return models


def run_hog(data: ImageData | None = None,
            spec: PrepSpec = SPEC_MAIN,
            weights: np.ndarray | None = None,
            n_splits: int = 5,
            all_strategies: bool = False,
            verbose: bool = True) -> list[dict]:
    cfg.set_global_seeds()
    if data is None:
        data = build_image_array(spec=spec, verbose=verbose)
    if weights is None:
        weights, _ = load_quality_weights(data.pulse_id)

    H = hog_transform(data.X, verbose=verbose)
    hog_data = ImageData(
        X=H, y=data.y, groups=data.groups, image_key=data.image_key,
        pulse_id=data.pulse_id, class_names=data.class_names, spec=data.spec)

    rows = []
    from sklearn.base import clone
    for name, model in _hog_classifiers().items():
        if verbose:
            print(f"  -> HOG + {name}")

        def fit_predict(Xtr, ytr, Xte, _m=model):
            m = clone(_m)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)

        rows.append(evaluate_oof(
            hog_data, fit_predict, method=f"HOG + {name}",
            weights=weights, n_splits=n_splits,
            all_strategies=all_strategies, verbose=verbose))
    return rows


def _to_rgb3(X: np.ndarray) -> np.ndarray:
    c = X.shape[-1]
    if c == 3:
        out = X
    elif c == 1:
        out = np.repeat(X, 3, axis=-1)
    elif c == 2:
        out = np.stack([X[..., 0], X[..., 1], X[..., 0]], axis=-1)
    else:
        raise ValueError(f"Непідтримувана кількість каналів: {c}")
    return out * 255.0


def extract_backbone_embeddings(X: np.ndarray,
                                backbone: str = "MobileNetV2",
                                batch_size: int = 32,
                                verbose: bool = True) -> np.ndarray:
    from tensorflow.keras import applications

    backbones = {
        "MobileNetV2": (applications.MobileNetV2,
                        applications.mobilenet_v2.preprocess_input),
        "EfficientNetB0": (applications.EfficientNetB0,
                           applications.efficientnet.preprocess_input),
        "ResNet50": (applications.ResNet50,
                     applications.resnet50.preprocess_input),
    }
    if backbone not in backbones:
        raise ValueError(f"Невідомий backbone: {backbone}. "
                         f"Доступні: {list(backbones)}")
    ctor, preprocess = backbones[backbone]

    rgb = _to_rgb3(X)
    h, w = rgb.shape[1], rgb.shape[2]
    model = ctor(include_top=False, weights="imagenet",
                 input_shape=(h, w, 3), pooling="avg")
    model.trainable = False

    emb = model.predict(preprocess(rgb.astype(np.float32)),
                        batch_size=batch_size,
                        verbose=1 if verbose else 0)
    if verbose:
        print(f"  {backbone} (frozen): {X.shape} -> {emb.shape}")
    return emb


def transfer_heads(random_state: int = cfg.RANDOM_STATE) -> dict:
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=random_state)),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000,
                                       random_state=random_state)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1),
    }


def run_transfer(data: ImageData | None = None,
                 spec: PrepSpec = SPEC_MAIN,
                 backbone: str = "MobileNetV2",
                 weights: np.ndarray | None = None,
                 n_splits: int = 5,
                 all_strategies: bool = False,
                 verbose: bool = True) -> list[dict]:
    cfg.set_global_seeds()
    if data is None:
        data = build_image_array(spec=spec, verbose=verbose)
    if weights is None:
        weights, _ = load_quality_weights(data.pulse_id)

    emb = extract_backbone_embeddings(data.X, backbone=backbone,
                                      verbose=verbose)
    emb_data = ImageData(
        X=emb, y=data.y, groups=data.groups, image_key=data.image_key,
        pulse_id=data.pulse_id, class_names=data.class_names, spec=data.spec)

    heads = transfer_heads()

    rows = []
    from sklearn.base import clone
    for name, head in heads.items():
        if verbose:
            print(f"  -> {backbone} frozen + {name}")

        def fit_predict(Xtr, ytr, Xte, _m=head):
            m = clone(_m)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)

        rows.append(evaluate_oof(
            emb_data, fit_predict, method=f"{backbone} frozen + {name}",
            weights=weights, n_splits=n_splits,
            all_strategies=all_strategies, verbose=verbose))
    return rows


def _add_coords_layer():
    import keras
    from keras import ops

    @keras.saving.register_keras_serializable(package="sensor")
    class AddCoords(keras.layers.Layer):
        def call(self, x):
            shape = ops.shape(x)
            b, h, w = shape[0], shape[1], shape[2]
            yy = ops.reshape(ops.linspace(-1.0, 1.0, h), (1, h, 1, 1))
            xx = ops.reshape(ops.linspace(-1.0, 1.0, w), (1, 1, w, 1))
            yy = ops.broadcast_to(yy, (b, h, w, 1))
            xx = ops.broadcast_to(xx, (b, h, w, 1))
            return ops.concatenate([x, ops.cast(yy, x.dtype),
                                    ops.cast(xx, x.dtype)], axis=-1)

        def compute_output_shape(self, s):
            return (*s[:-1], s[-1] + 2)

    return AddCoords()


CNN_HEADS = ("gap", "flatten", "coords")
CNN_DEFAULT_HEAD = "flatten"


def build_cnn2d(input_shape: tuple[int, int, int], n_classes: int,
                augment: bool = True, norm: str = "groupnorm",
                head: str = CNN_DEFAULT_HEAD):
    from tensorflow.keras import layers, models

    if head not in CNN_HEADS:
        raise ValueError(f"Невідомий head: {head}. Є: {CNN_HEADS}")
    inp = layers.Input(shape=input_shape)
    x = inp
    if augment:
        x = layers.RandomTranslation(height_factor=0.06, width_factor=0.06,
                                     fill_mode="constant", fill_value=0.0)(x)
        x = layers.RandomZoom(height_factor=0.10, width_factor=0.10,
                              fill_mode="constant", fill_value=0.0)(x)
        x = layers.RandomContrast(0.15)(x)
    if head == "coords":
        x = _add_coords_layer()(x)

    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same",
                          use_bias=(norm == "none"))(x)
        if norm == "groupnorm":
            x = layers.GroupNormalization(groups=-1)(x)
        elif norm == "batchnorm":
            x = layers.BatchNormalization()(x)
        elif norm != "none":
            raise ValueError(f"Невідома нормалізація: {norm}")
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)

    x = (layers.Flatten()(x) if head == "flatten"
         else layers.GlobalAveragePooling2D()(x))
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="embedding")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_cnn2d(data: ImageData | None = None,
              spec: PrepSpec = SPEC_MAIN,
              weights: np.ndarray | None = None,
              epochs: int = 60,
              batch_size: int = 32,
              augment: bool = True,
              norm: str = "groupnorm",
              n_splits: int = 5,
              all_strategies: bool = False,
              head: str = CNN_DEFAULT_HEAD,
              verbose: bool = True) -> list[dict]:
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from sklearn.model_selection import GroupShuffleSplit

    cfg.set_global_seeds()
    if data is None:
        data = build_image_array(spec=spec, verbose=verbose)
    if weights is None:
        weights, _ = load_quality_weights(data.pulse_id)

    groups = data.groups
    input_shape = data.X.shape[1:]
    n_classes = data.n_classes

    gkf = GroupKFold(n_splits=max(2, min(n_splits, len(np.unique(groups)))))
    oof = np.full((len(data.y), n_classes), np.nan)
    fold_pulse, fold_image = [], []
    fold_test_idx: list[np.ndarray] = []
    t0 = time.time()

    for fold, (tr, te) in enumerate(gkf.split(data.X, data.y, groups=groups),
                                    start=1):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                random_state=cfg.RANDOM_STATE)
        tr_in, val_in = next(gss.split(data.X[tr], data.y[tr], groups=groups[tr]))
        tr2, val2 = tr[tr_in], tr[val_in]

        model = build_cnn2d(input_shape, n_classes, augment=augment, norm=norm,
                            head=head)
        model.fit(
            data.X[tr2], data.y[tr2],
            validation_data=(data.X[val2], data.y[val2]),
            epochs=epochs, batch_size=batch_size, verbose=0,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6,
                                            factor=0.5, min_lr=1e-5),
            ],
        )
        proba = model.predict(data.X[te], verbose=0)
        oof[te] = proba
        fold_test_idx.append(te)

        fold_pulse.append(f1_score(data.y[te], np.argmax(proba, axis=1),
                                   average="macro", zero_division=0))
        ti, pi = _fuse_by_image(data.y[te], proba, data.image_key[te],
                                weights[te], "weighted")
        fold_image.append(f1_score(ti, pi, average="macro", zero_division=0))
        if verbose:
            print(f"    фолд {fold}: pulse {fold_pulse[-1]:.4f}, "
                  f"image {fold_image[-1]:.4f}")
        tf.keras.backend.clear_session()

    ev = ~np.isnan(oof).any(axis=1)
    pulse_f1 = f1_score(data.y[ev], np.argmax(oof[ev], axis=1),
                        average="macro", zero_division=0)
    pulse_acc = accuracy_score(data.y[ev], np.argmax(oof[ev], axis=1))
    ti, pi = _fuse_by_image(data.y[ev], oof[ev], data.image_key[ev],
                            weights[ev], "weighted")
    image_f1 = f1_score(ti, pi, average="macro", zero_division=0)
    image_acc = accuracy_score(ti, pi)

    name = "2D-CNN (fixed)" + ("" if augment else ", no-aug") + \
           ("" if norm == "groupnorm" else f", {norm}") + \
           ("" if head == CNN_DEFAULT_HEAD else f", {head}")

    from .visualization import plot_confusion
    plot_confusion(confusion_matrix(ti, pi), data.class_names,
                   title=f"Image-level ({name}, weighted)",
                   out_name=f"confusion_image_{_slug(name)}.png")
    row = {
        "method": name,
        "spec": data.spec.name,
        "n_pulses": int(ev.sum()),
        "n_images": int(len(ti)),
        "pulse_f1_oof": round(float(pulse_f1), 4),
        "pulse_acc_oof": round(float(pulse_acc), 4),
        "image_f1_oof": round(float(image_f1), 4),
        "image_acc_oof": round(float(image_acc), 4),
        "image_f1_fold_mean": round(float(np.mean(fold_image)), 4),
        "image_f1_fold_std": round(float(np.std(fold_image)), 4),
        "fusion": "weighted",
        "train_sec": round(time.time() - t0, 1),
    }

    if all_strategies:
        for strat in fz.STRATEGIES:
            ti_s, pi_s = _fuse_by_image(data.y[ev], oof[ev], data.image_key[ev],
                                        weights[ev], strat)
            row[f"image_f1_{strat}"] = round(float(
                f1_score(ti_s, pi_s, average="macro", zero_division=0)), 4)
            per_fold = []
            for te in fold_test_idx:
                tf_, pf_ = _fuse_by_image(data.y[te], oof[te],
                                          data.image_key[te], weights[te], strat)
                per_fold.append(f1_score(tf_, pf_, average="macro",
                                         zero_division=0))
            row[f"image_f1_{strat}_std"] = round(float(np.std(per_fold)), 4)

    save_oof(name, data, oof, weights, fold_test_idx)

    if verbose:
        print(f"  {name:34s} | pulse {pulse_f1:.4f} | image {image_f1:.4f} "
              f"(±{np.std(fold_image):.4f}) | {row['train_sec']}s")
    return [row]


CNN_ARCH_CSV = os.path.join(cfg.RESULTS_DIR, "cnn_architecture.csv")


def run_cnn_heads(heads: tuple[str, ...] = CNN_HEADS,
                  spec: PrepSpec = SPEC_MAIN,
                  n_splits: int = 5,
                  verbose: bool = True) -> pd.DataFrame:
    data = build_image_array(spec=spec, verbose=verbose)
    weights, _ = load_quality_weights(data.pulse_id)
    rows = []
    for h in heads:
        if verbose:
            print(f"\n=== 2D-CNN head={h} ===")
        rows += run_cnn2d(data, weights=weights, n_splits=n_splits,
                          all_strategies=True, head=h, verbose=verbose)
        pd.DataFrame(rows).to_csv(CNN_ARCH_CSV, index=False)
    return pd.DataFrame(rows)


def run_input_ablation(specs: list[PrepSpec] | None = None,
                       n_splits: int = 5,
                       verbose: bool = True) -> pd.DataFrame:
    cfg.set_global_seeds()
    if specs is None:
        specs = default_specs()

    index = scan_pulse_images()
    rows = []
    for spec in specs:
        if verbose:
            print(f"[*] {spec.name}")
        data = build_image_array(index, spec, verbose=verbose)
        weights, _ = load_quality_weights(data.pulse_id)
        H = hog_transform(data.X, verbose=False)
        hog_data = ImageData(
            X=H, y=data.y, groups=data.groups, image_key=data.image_key,
            pulse_id=data.pulse_id, class_names=data.class_names, spec=spec)

        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=cfg.RANDOM_STATE)),
        ])

        from sklearn.base import clone

        def fit_predict(Xtr, ytr, Xte, _m=probe):
            m = clone(_m)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)

        row = evaluate_oof(hog_data, fit_predict,
                           method="HOG + SVM (пробник)",
                           weights=weights, n_splits=n_splits,
                           verbose=False)
        row["resize_mode"] = spec.resize_mode
        row["channels"] = spec.channels
        row["size"] = f"{spec.size[0]}x{spec.size[1]}"
        row["n_features_hog"] = int(H.shape[1])
        rows.append(row)
        if verbose:
            print(f"    image f1 {row['image_f1_oof']:.4f}  "
                  f"(pulse {row['pulse_f1_oof']:.4f})")

    df = pd.DataFrame(rows).sort_values(
        "image_f1_oof", ascending=False).reset_index(drop=True)

    base = df[df["spec"] == SPEC_BASELINE.name]
    if not base.empty:
        b = float(base.iloc[0]["image_f1_oof"])
        df["delta_vs_baseline"] = (df["image_f1_oof"] - b).round(4)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    df.to_csv(IMAGE_ABLATION_CSV, index=False)
    if verbose:
        print()
        print(df[["spec", "image_f1_oof", "pulse_f1_oof",
                  "delta_vs_baseline"]].to_string(index=False))
        print(f"\n[OK] Збережено: {cfg.to_rel(IMAGE_ABLATION_CSV)}")
    return df


def save_image_results(rows: list[dict], csv_path: str = IMAGE_RESULTS_CSV,
                       merge: bool = True) -> pd.DataFrame:
    df_new = pd.DataFrame(rows)
    if merge and os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            keys = set(zip(df_new["method"], df_new["spec"]))
            keep = [not (m, s) in keys for m, s in
                    zip(df_old["method"], df_old["spec"])]
            df_new = pd.concat([df_old[keep], df_new], ignore_index=True)
        except Exception:
            pass

    df_new = df_new.sort_values("image_f1_oof",
                                ascending=False).reset_index(drop=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    df_new.to_csv(csv_path, index=False)
    return df_new
