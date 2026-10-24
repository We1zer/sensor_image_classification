from __future__ import annotations

import os
import time

import pandas as pd

from . import config as cfg
from .image_prep import SPEC_MAIN, build_image_array, scan_pulse_images
from . import image_methods as im


def run_image_branch(spec=SPEC_MAIN,
                     do_ablation: bool = True,
                     do_hog: bool = True,
                     do_transfer: bool = True,
                     do_cnn: bool = True,
                     backbone: str = "MobileNetV2",
                     cnn_epochs: int = 60,
                     n_splits: int = 5,
                     progress=None,
                     verbose: bool = True) -> dict:
    def _tick(pct: int, msg: str) -> None:
        if verbose:
            print(f"\n=== [{pct}%] {msg} ===")
        if progress:
            progress(pct, msg)

    t0 = time.time()
    index = scan_pulse_images()
    if index.empty:
        raise ValueError(
            "Немає вирізаних імпульсів у processed/pulse_images/. "
            "Спочатку: python -m src.dataset_builder")

    rows: list[dict] = []
    ablation = None

    if do_ablation:
        _tick(5, "Ablation представлення входу (Група 0)")
        ablation = im.run_input_ablation(n_splits=n_splits, verbose=verbose)

    _tick(35, f"Підготовка входу {spec.name}")
    data = build_image_array(index, spec, verbose=verbose)
    weights, w_source = im.load_quality_weights(data.pulse_id)
    if verbose:
        print(f"  ваги злиття: {w_source}")

    if do_hog:
        _tick(45, "Метод 2: HOG + класичні класифікатори")
        rows += im.run_hog(data, weights=weights, n_splits=n_splits,
                           verbose=verbose)

    if do_transfer:
        _tick(65, f"Метод 3: {backbone} (frozen) + класичні голови")
        rows += im.run_transfer(data, backbone=backbone, weights=weights,
                                n_splits=n_splits, verbose=verbose)

    if do_cnn:
        _tick(80, "Метод 4: виправлений 2D-CNN")
        rows += im.run_cnn2d(data, weights=weights, epochs=cnn_epochs,
                             n_splits=n_splits, all_strategies=True,
                             verbose=verbose)

    _tick(95, "Збереження результатів")
    df = im.save_image_results(rows) if rows else pd.DataFrame()

    if verbose and not df.empty:
        cols = ["method", "spec", "pulse_f1_oof", "image_f1_oof",
                "image_f1_fold_std", "train_sec"]
        print()
        print(df[cols].to_string(index=False))
        print(f"\n[OK] {cfg.to_rel(im.IMAGE_RESULTS_CSV)}")
        print(f"[OK] image-гілка завершена за {time.time() - t0:.0f} с")

    _tick(100, "Готово")
    return {
        "branch": "image",
        "spec": spec.name,
        "weights_source": w_source,
        "results": df,
        "ablation": ablation,
        "elapsed_sec": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    run_image_branch()
