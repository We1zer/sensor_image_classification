import os
import json
import numpy as np
import pandas as pd

from . import config as cfg
from .feature_extraction import FEATURE_NAMES
from . import fusion as fz
from .train_cnn1d import _load_signal, build_cnn1d, _require_tf


def _load_aligned(features_csv: str = cfg.FEATURES_CSV,
                  length: int = cfg.CNN_SIG_LEN):
    df = pd.read_csv(features_csv)
    if "is_augmented" in df.columns:
        df = df[~df["is_augmented"].astype(bool)]
    df = df[df["pulse_signal_path"].fillna("").astype(str).str.len() > 0]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Немає сигналів. build_dataset(save_pulse_signals=True).")

    class_names = sorted(df["class"].unique().tolist())
    c2i = {c: i for i, c in enumerate(class_names)}

    sigs, keep = [], []
    for i, p in enumerate(df["pulse_signal_path"].values):
        s = _load_signal(p, length)
        if s is not None:
            sigs.append(s)
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)

    X_sig = np.expand_dims(np.stack(sigs), -1)
    X_feat = df[FEATURE_NAMES].values.astype(np.float64)
    y = np.array([c2i[c] for c in df["class"]], dtype=np.int64)
    groups = df["source_image"].values
    image_key = df["image_path"].values
    weights = np.array([fz.pulse_quality_weight(c, s)
                        for c, s in zip(df["local_coverage"],
                                        df["amp_to_noise"])], dtype=float)
    return X_sig, X_feat, y, groups, image_key, weights, class_names


def _img_f1(image_key, y, proba, weights, strategy):
    from sklearn.metrics import f1_score
    true_img, pred_img = [], []
    for img in pd.unique(image_key):
        m = image_key == img
        true_img.append(y[m][0])
        pred_img.append(fz.fuse(proba[m], strategy=strategy, weights=weights[m]))
    return f1_score(true_img, pred_img, average="macro", zero_division=0), \
        (true_img, pred_img)


def cross_validate_hybrid(n_splits: int = 5,
                          epochs: int = 100,
                          batch_size: int = 32,
                          model_name: str = "RandomForest",
                          fusion_strategy: str = "weighted",
                          save_plots: bool = True,
                          verbose: bool = True) -> dict:
    tf = _require_tf()
    from tensorflow.keras import callbacks, Model
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from .train_ml_models import _build_models

    cfg.set_global_seeds()
    X_sig, X_feat, y, groups, image_key, weights, class_names = _load_aligned()
    n_classes = len(class_names)
    L = X_sig.shape[1]
    if verbose:
        print(f"  hybrid dataset: sig={X_sig.shape}, feat={X_feat.shape}, "
              f"класів={n_classes}")

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof_feat = np.full((len(y), n_classes), np.nan)
    oof_hyb = np.full((len(y), n_classes), np.nan)

    def _align_proba(model, proba):
        ordered = np.zeros((proba.shape[0], n_classes))
        for j, c in enumerate(model.classes_):
            ordered[:, int(c)] = proba[:, j]
        return ordered

    for fold, (tr, te) in enumerate(gkf.split(X_sig, y, groups=groups), start=1):
        gss = GroupShuffleSplit(1, test_size=0.15, random_state=cfg.RANDOM_STATE)
        tr_in, val_in = next(gss.split(X_sig[tr], y[tr], groups=groups[tr]))
        tr2, val2 = tr[tr_in], tr[val_in]
        cnn = build_cnn1d(L, n_classes)
        cnn.fit(X_sig[tr2], y[tr2], validation_data=(X_sig[val2], y[val2]),
                epochs=epochs, batch_size=batch_size, verbose=0,
                callbacks=[
                    callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                            restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8,
                                                factor=0.5, min_lr=1e-5)])

        emb_model = Model(inputs=cnn.input,
                          outputs=cnn.get_layer("embedding").output)
        emb_tr = emb_model.predict(X_sig[tr], verbose=0)
        emb_te = emb_model.predict(X_sig[te], verbose=0)

        sc_f = StandardScaler().fit(X_feat[tr])
        rf_f = _build_models(use_xgboost=True)[model_name]
        rf_f.fit(sc_f.transform(X_feat[tr]), y[tr])
        oof_feat[te] = _align_proba(rf_f, rf_f.predict_proba(sc_f.transform(X_feat[te])))

        Htr = np.hstack([X_feat[tr], emb_tr])
        Hte = np.hstack([X_feat[te], emb_te])
        sc_h = StandardScaler().fit(Htr)
        rf_h = _build_models(use_xgboost=True)[model_name]
        rf_h.fit(sc_h.transform(Htr), y[tr])
        oof_hyb[te] = _align_proba(rf_h, rf_h.predict_proba(sc_h.transform(Hte)))

        if verbose:
            f_f = f1_score(y[te], np.argmax(oof_feat[te], 1),
                           average="macro", zero_division=0)
            f_h = f1_score(y[te], np.argmax(oof_hyb[te], 1),
                           average="macro", zero_division=0)
            print(f"    фолд {fold}/{n_splits}: pulse f1  feat={f_f:.4f}  "
                  f"hybrid={f_h:.4f}")
        tf.keras.backend.clear_session()

    def _summ(oof):
        pp = np.argmax(oof, 1)
        pulse_f1 = f1_score(y, pp, average="macro", zero_division=0)
        pulse_acc = accuracy_score(y, pp)
        img_f1, (ti, pi) = _img_f1(image_key, y, oof, weights, fusion_strategy)
        img_acc = accuracy_score(ti, pi)
        return pulse_f1, pulse_acc, img_f1, img_acc, confusion_matrix(ti, pi)

    pf, pa, if_, ia, _ = _summ(oof_feat)
    hpf, hpa, hif, hia, hcm = _summ(oof_hyb)

    results_df = pd.DataFrame([
        {"approach": "features-only", "level": "pulse",
         "f1_macro_oof": round(pf, 4), "accuracy_oof": round(pa, 4)},
        {"approach": "features-only", "level": "image",
         "f1_macro_oof": round(if_, 4), "accuracy_oof": round(ia, 4)},
        {"approach": "hybrid", "level": "pulse",
         "f1_macro_oof": round(hpf, 4), "accuracy_oof": round(hpa, 4)},
        {"approach": "hybrid", "level": "image",
         "f1_macro_oof": round(hif, 4), "accuracy_oof": round(hia, 4)},
    ])
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(cfg.RESULTS_DIR, "hybrid_cv.csv"), index=False)
    with open(os.path.join(cfg.RESULTS_DIR, "hybrid_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"classifier": model_name, "emb_dim": 64,
                   "n_features": len(FEATURE_NAMES),
                   "image_f1_features": float(if_),
                   "image_f1_hybrid": float(hif)}, f,
                  ensure_ascii=False, indent=2)

    if save_plots:
        from .visualization import plot_confusion
        plot_confusion(hcm, class_names,
                       title=f"Image-level (Hybrid {model_name}+CNN-emb)",
                       out_name=f"confusion_image_HYBRID_{model_name}.png")

    if verbose:
        print()
        print(f"  Гібрид ({model_name} + 1D-CNN ембединг), GroupKFold={n_splits}")
        print(f"    features-only : pulse {pf:.4f} | image {if_:.4f}")
        print(f"    hybrid        : pulse {hpf:.4f} | image {hif:.4f}")
        print(f"    Δ image (hybrid − features) = {hif - if_:+.4f}")
        print(f"[OK] Збережено: {cfg.RESULTS_DIR}/hybrid_cv.csv")

    return {
        "results": results_df,
        "image_f1_features": if_,
        "image_f1_hybrid": hif,
        "cm_hybrid": hcm,
        "class_names": class_names,
    }


if __name__ == "__main__":
    cross_validate_hybrid()
