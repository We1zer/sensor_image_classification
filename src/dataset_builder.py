import os
import cv2
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from . import config as cfg
from .data_loader import create_image_index
from .image_preprocessing import read_image, preprocess_pipeline
from .signal_extraction import mask_to_clean_signal
from .pulse_segmentation import (
    find_pulse_boundaries, crop_pulse_image, crop_pulse_signal,
)
from .pulse_quality import is_valid_pulse
from .feature_extraction import extract_features, FEATURE_NAMES
from .signal_augmentation import augment_pulse


def _save_pulse_image(pulse_img: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    success, buf = cv2.imencode(".png", pulse_img)
    if success:
        buf.tofile(out_path)


def build_dataset(invert: bool = False,
                  save_pulse_images: bool = True,
                  save_pulse_signals: bool = False,
                  augment_factor: int = 0,
                  augment_seed: int = 42,
                  segmentation: str = "full_cycle",
                  verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(augment_seed) if augment_factor > 0 else None
    df_index = create_image_index()
    if df_index.empty:
        if verbose:
            print("[!] Папки dataset порожні. Додайте PNG-файли в dataset/PNG-1*/.")
        return pd.DataFrame()

    rows = []
    rejected_count = 0

    iterator = df_index.iterrows()
    if verbose:
        iterator = tqdm(list(iterator), desc="Обробка зображень")

    for _, row in iterator:
        path = row["image_path"]
        cls = row["class"]
        file_number = int(row["file_number"])
        source = row["source_image"]
        folder = row["folder"]

        try:
            img = read_image(path)
        except Exception as e:
            if verbose:
                print(f"[!] {source}: {e}")
            continue

        prep = preprocess_pipeline(img, invert=invert)
        cropped = prep["cropped"]
        inverted = prep["inverted"]
        mask = prep["mask"]

        x, y, cov = mask_to_clean_signal(mask, do_smooth=True)

        if cov < 0.05:
            rejected_count += 1
            continue

        bounds = find_pulse_boundaries(y, method=segmentation)

        for k, (a, b) in enumerate(bounds):
            pulse = crop_pulse_signal(y, a, b)
            local_cov = float(np.sum(~np.isnan(pulse))) / max(1, len(pulse))

            valid, qc = is_valid_pulse(pulse, coverage_value=local_cov)

            if not valid:
                rejected_count += 1
                continue

            feats = extract_features(pulse)

            pulse_id = f"{cls}__{os.path.splitext(source)[0]}__p{k}"
            pulse_img_path = ""
            if save_pulse_images:
                cut_img = crop_pulse_image(cropped, a, b)
                out_dir = os.path.join(cfg.PULSE_IMAGES_DIR, folder)
                pulse_img_path = os.path.join(out_dir, f"{pulse_id}.png")
                _save_pulse_image(cut_img, pulse_img_path)

            pulse_sig_path = ""
            if save_pulse_signals:
                out_dir = os.path.join(cfg.PULSE_SIGNALS_DIR, folder)
                os.makedirs(out_dir, exist_ok=True)
                pulse_sig_path = os.path.join(out_dir, f"{pulse_id}.npy")
                np.save(pulse_sig_path, pulse)

            row_dict = {
                "pulse_id": pulse_id,
                "source_image": source,
                "image_path": cfg.to_rel(path),
                "class": cls,
                "file_number": file_number,
                "pulse_index": k,
                "start_x": int(a),
                "end_x": int(b),
                "coverage": float(cov),
                "local_coverage": float(local_cov),
                "pulse_image_path": cfg.to_rel(pulse_img_path),
                "pulse_signal_path": cfg.to_rel(pulse_sig_path),
                "is_augmented": False,
                **feats,
            }
            rows.append(row_dict)

            if augment_factor > 0 and rng is not None:
                aug_pulses = augment_pulse(pulse, augment_factor, rng=rng)
                for ai, aug in enumerate(aug_pulses):
                    aug_feats = extract_features(aug)
                    rows.append({
                        "pulse_id": f"{pulse_id}_aug{ai}",
                        "source_image": source,
                        "image_path": cfg.to_rel(path),
                        "class": cls,
                        "file_number": file_number,
                        "pulse_index": k,
                        "start_x": int(a),
                        "end_x": int(b),
                        "coverage": float(cov),
                        "local_coverage": float(local_cov),
                        "pulse_image_path": "",
                        "pulse_signal_path": "",
                        "is_augmented": True,
                        **aug_feats,
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        os.makedirs(cfg.FEATURES_DIR, exist_ok=True)
        df.to_csv(cfg.FEATURES_CSV, index=False)
        try:
            with open(os.path.join(cfg.FEATURES_DIR, "pipeline_meta.json"),
                      "w", encoding="utf-8") as _f:
                import json as _json
                _json.dump({
                    "segmentation": segmentation,
                    "invert": bool(invert),
                    "augment_factor": int(augment_factor),
                }, _f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if verbose:
            n_orig = int((~df["is_augmented"]).sum()) if "is_augmented" in df.columns else len(df)
            n_aug = int(df["is_augmented"].sum()) if "is_augmented" in df.columns else 0
            print(f"[OK] Збережено: {cfg.FEATURES_CSV}")
            print(f"     Всього імпульсів: {len(df)}  "
                  f"(оригінальних: {n_orig}, аугментованих: {n_aug})")
            print(f"     Відкинуто:       {rejected_count}")
            print("     За класами:")
            for cls, n in df["class"].value_counts().items():
                print(f"        {cls:10s} : {n}")
    else:
        if verbose:
            print("[!] Не отримано жодного імпульсу. Перевірте параметри обрізання та HSV.")

    return df


if __name__ == "__main__":
    build_dataset()
