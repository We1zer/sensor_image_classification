from __future__ import annotations

import inspect
import os
import re
import time

import joblib
import numpy as np

from . import config as cfg
from .image_prep import SPEC_MAIN, PrepSpec, ImageData, build_image_array
from . import image_methods as im


HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
}

_hog_defaults = {
    k: v.default
    for k, v in inspect.signature(im.hog_transform).parameters.items()
    if k in HOG_PARAMS
}
if _hog_defaults != HOG_PARAMS:
    raise RuntimeError(
        "HOG_PARAMS розійшлися з дефолтами im.hog_transform: "
        f"{HOG_PARAMS} проти {_hog_defaults}")


def _slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def model_path(method: str) -> str:
    return os.path.join(cfg.MODELS_DIR, f"image_{_slug(method)}.pkl")


def keras_path(method: str) -> str:
    return os.path.join(cfg.MODELS_DIR, f"image_{_slug(method)}.keras")


CNN_METHOD = "2D-CNN"

ENSEMBLES: dict[str, tuple[str, ...]] = {
    "Ансамбль: HOG + XGBoost + 2D-CNN": ("HOG + XGBoost", CNN_METHOD),
}

OOF_NAME = {CNN_METHOD: "2D-CNN (fixed)"}


def is_ensemble(method: str) -> bool:
    return method in ENSEMBLES


def available_methods(backbone: str = "MobileNetV2",
                      include_cnn: bool = True) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for head in im._hog_classifiers():
        out[f"HOG + {head}"] = {"kind": "hog", "head": head}
    for head in im.transfer_heads():
        out[f"{backbone} frozen + {head}"] = {
            "kind": "transfer", "head": head, "backbone": backbone}
    if include_cnn:
        out[CNN_METHOD] = {"kind": "cnn2d", "head": "CNN"}
    return out


def transform(X: np.ndarray, kind: str, backbone: str = "MobileNetV2",
              verbose: bool = False) -> np.ndarray:
    if kind == "hog":
        return im.hog_transform(X, verbose=verbose, **HOG_PARAMS)
    if kind == "transfer":
        return im.extract_backbone_embeddings(X, backbone=backbone,
                                              verbose=verbose)
    if kind == "cnn2d":
        return X
    raise ValueError(f"Невідомий тип методу: {kind}")


def fit_cnn2d(data: ImageData,
              epochs: int = 60,
              batch_size: int = 32,
              augment: bool = True,
              norm: str = "groupnorm",
              val_size: float = 0.15,
              verbose: bool = True):
    from tensorflow.keras import callbacks
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size,
                            random_state=cfg.RANDOM_STATE)
    tr, val = next(gss.split(data.X, data.y, groups=data.groups))

    model = im.build_cnn2d(data.X.shape[1:], data.n_classes,
                           augment=augment, norm=norm)
    hist = model.fit(
        data.X[tr], data.y[tr],
        validation_data=(data.X[val], data.y[val]),
        epochs=epochs, batch_size=batch_size,
        verbose=1 if verbose else 0,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6,
                                        factor=0.5, min_lr=1e-5),
        ],
    )
    best = int(np.argmin(hist.history["val_loss"]))
    return model, {
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": best + 1,
        "val_accuracy": round(float(hist.history["val_accuracy"][best]), 4),
        "val_loss": round(float(hist.history["val_loss"][best]), 4),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_val_groups": int(len(np.unique(data.groups[val]))),
    }


def _fit_and_bundle(name: str, info: dict, data: ImageData,
                    F: np.ndarray, backbone: str,
                    cnn_epochs: int = 60,
                    verbose: bool = True) -> dict:
    kind = info["kind"]
    t0 = time.time()
    extra: dict = {}

    if kind == "cnn2d":
        model, extra = fit_cnn2d(data, epochs=cnn_epochs, verbose=verbose)
        model.save(keras_path(name))
        est = None
        n_features = int(np.prod(data.X.shape[1:]))
    else:
        est = (im._hog_classifiers()[info["head"]] if kind == "hog"
               else im.transfer_heads()[info["head"]])
        est.fit(F, data.y)
        n_features = int(F.shape[1])

    bundle = {
        "method": name,
        "kind": kind,
        "head": info["head"],
        "backbone": backbone if kind == "transfer" else None,
        "spec": {
            "resize_mode": data.spec.resize_mode,
            "channels": data.spec.channels,
            "size": tuple(data.spec.size),
            "denoise": data.spec.denoise,
            "pad_value": data.spec.pad_value,
            "clean_overlay": data.spec.clean_overlay,
        },
        "hog_params": HOG_PARAMS if kind == "hog" else None,
        "class_names": list(data.class_names),
        "n_train_pulses": int(len(data.y)),
        "n_features": n_features,
        "estimator": est,
        "keras_path": cfg.to_rel(keras_path(name)) if kind == "cnn2d" else None,
        "train_info": extra,
        "cnn_head": im.CNN_DEFAULT_HEAD if kind == "cnn2d" else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_sec": round(time.time() - t0, 1),
    }
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, model_path(name))
    return bundle


def fit_and_save(method: str,
                 data: ImageData | None = None,
                 spec: PrepSpec = SPEC_MAIN,
                 backbone: str = "MobileNetV2",
                 cnn_epochs: int = 60,
                 verbose: bool = True) -> dict:
    methods = available_methods(backbone)
    if method not in methods:
        raise ValueError(f"Невідомий метод: {method}. Є: {list(methods)}")
    info = methods[method]

    cfg.set_global_seeds()
    if data is None:
        data = build_image_array(spec=spec, verbose=verbose)

    F = transform(data.X, info["kind"], backbone=backbone, verbose=verbose)
    bundle = _fit_and_bundle(method, info, data, F, backbone,
                             cnn_epochs=cnn_epochs, verbose=verbose)
    if verbose:
        print(f"[OK] {method}: навчено на {len(data.y)} імпульсах "
              f"({bundle['train_sec']} с) -> "
              f"{cfg.to_rel(model_path(method))}")
    return bundle


def save_all(spec: PrepSpec = SPEC_MAIN,
             backbone: str = "MobileNetV2",
             only: list[str] | None = None,
             include_cnn: bool = True,
             cnn_epochs: int = 60,
             verbose: bool = True) -> list[str]:
    cfg.set_global_seeds()
    data = build_image_array(spec=spec, verbose=verbose)
    methods = available_methods(backbone, include_cnn=include_cnn)
    names = only or list(methods)

    cache: dict[str, np.ndarray] = {}
    saved = []
    for name in names:
        info = methods[name]
        kind = info["kind"]
        if kind not in cache:
            cache[kind] = transform(data.X, kind, backbone=backbone,
                                    verbose=verbose)

        bundle = _fit_and_bundle(name, info, data, cache[kind], backbone,
                                 cnn_epochs=cnn_epochs, verbose=verbose)
        saved.append(name)
        if verbose:
            extra = bundle.get("train_info") or {}
            note = (f", епох {extra['epochs_run']}, val_acc "
                    f"{extra['val_accuracy']}" if extra else "")
            print(f"  [OK] {name:36s} -> "
                  f"{cfg.to_rel(model_path(name))} "
                  f"({bundle['train_sec']} с{note})")
    return saved


def load(method: str) -> dict:
    p = model_path(method)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Немає збереженої моделі «{method}» ({cfg.to_rel(p)}). "
            "Спочатку: python -m src.image_model_store")
    return joblib.load(p)


def load_any(method: str) -> dict:
    if not is_ensemble(method):
        return load(method)
    members = [load(m) for m in ENSEMBLES[method]]
    spec0, cls0 = members[0]["spec"], list(members[0]["class_names"])
    for b in members[1:]:
        if b["spec"] != spec0 or list(b["class_names"]) != cls0:
            raise ValueError(
                f"Учасники ансамблю «{method}» мають різний вхід або порядок "
                "класів — перенавчіть їх: python -m src.image_model_store")
    return {"method": method, "kind": "ensemble", "members": members,
            "spec": spec0, "class_names": cls0}


def list_saved(backbone: str = "MobileNetV2") -> list[dict]:
    out = []
    for name in available_methods(backbone):
        p = model_path(name)
        if os.path.exists(p):
            try:
                b = joblib.load(p)
                if b.get("kind") == "cnn2d" and not os.path.exists(
                        cfg.to_abs(b.get("keras_path") or "")):
                    continue
                out.append({
                    "method": name,
                    "path": cfg.to_rel(p),
                    "trained_at": b.get("trained_at"),
                    "n_train_pulses": b.get("n_train_pulses"),
                    "n_features": b.get("n_features"),
                    "spec": b.get("spec", {}),
                })
            except Exception:
                continue
    saved = {d["method"]: d for d in out}
    for name, members in ENSEMBLES.items():
        if all(m in saved for m in members):
            out.append({
                "method": name,
                "members": list(members),
                "trained_at": max(saved[m]["trained_at"] or "" for m in members),
                "n_train_pulses": saved[members[0]]["n_train_pulses"],
                "n_features": None,
                "spec": saved[members[0]]["spec"],
            })
    return out


def spec_from_bundle(bundle: dict) -> PrepSpec:
    s = bundle["spec"]
    return PrepSpec(resize_mode=s["resize_mode"], channels=s["channels"],
                    size=tuple(s["size"]), denoise=s.get("denoise", True),
                    pad_value=s.get("pad_value", cfg.IMG_PAD_VALUE),
                    clean_overlay=s.get("clean_overlay", False))


_KERAS_CACHE: dict[str, object] = {}


def load_keras(bundle: dict):
    p = cfg.to_abs(bundle["keras_path"])
    if p not in _KERAS_CACHE:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Немає файла мережі {cfg.to_rel(p)}. "
                "Перенавчіть: python -m src.image_model_store \"2D-CNN\"")
        from tensorflow import keras
        im._add_coords_layer()
        _KERAS_CACHE[p] = keras.models.load_model(p)
    return _KERAS_CACHE[p]


def predict_proba(bundle: dict, X: np.ndarray,
                  verbose: bool = False) -> np.ndarray:
    if bundle["kind"] == "ensemble":
        return np.mean([predict_proba(b, X, verbose=verbose)
                        for b in bundle["members"]], axis=0)
    if bundle["kind"] == "cnn2d":
        model = load_keras(bundle)
        return np.asarray(model.predict(X, verbose=1 if verbose else 0),
                          dtype=float)

    F = transform(X, bundle["kind"], backbone=bundle.get("backbone")
                  or "MobileNetV2", verbose=verbose)
    est = bundle["estimator"]
    proba = est.predict_proba(F)
    n = len(bundle["class_names"])
    ordered = np.zeros((len(F), n), dtype=float)
    for j, cls_idx in enumerate(est.classes_):
        ordered[:, int(cls_idx)] = proba[:, j]
    return ordered


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or None
    print("Навчання картинкових моделей на повному наборі...")
    save_all(only=only)
