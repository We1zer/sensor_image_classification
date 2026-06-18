from __future__ import annotations

import os
from flask import Blueprint, render_template, send_file, abort, redirect, url_for, flash

from src import config as cfg
from src.data_loader import create_image_index

from .helpers import first_image_for_class


bp_dataset = Blueprint("dataset", __name__)


@bp_dataset.route("/")
def view():
    df = create_image_index(save_csv=False)
    counts: dict[str, int] = {c: 0 for c in cfg.CLASS_NAMES}
    if not df.empty and "class" in df.columns:
        for cls, n in df["class"].value_counts().items():
            counts[cls] = int(n)

    rows = []
    for cls in cfg.CLASS_NAMES:
        rows.append({
            "class": cls,
            "count": counts.get(cls, 0),
            "has_preview": first_image_for_class(cls) is not None,
        })

    return render_template(
        "dataset.html",
        rows=rows,
        total=int(sum(counts.values())),
        dataset_dir=cfg.DATASET_DIR,
        class_folders=cfg.CLASS_FOLDERS,
    )


@bp_dataset.route("/reindex", methods=["POST"])
def reindex():
    df = create_image_index(save_csv=True)
    flash(f"Індекс оновлено. Всього зображень: {len(df)}.", "success")
    return redirect(url_for("dataset.view"))


@bp_dataset.route("/preview/<cls>")
def preview(cls: str):
    if cls not in cfg.CLASS_NAMES:
        abort(404)
    path = first_image_for_class(cls)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")
