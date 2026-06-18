from __future__ import annotations

import os
import io
import base64
import time
from pathlib import Path
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, send_from_directory, abort, current_app,
)
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.predict import predict_image

from .helpers import (
    UPLOADS_DIR, MAX_UPLOAD_BYTES, is_allowed_image, format_proba,
)


bp_predict = Blueprint("predict", __name__)


def _model_ready() -> bool:
    return all(os.path.exists(os.path.join(cfg.MODELS_DIR, f))
               for f in ("best_ml_model.pkl", "scaler.pkl",
                         "label_encoder.pkl", "best_model_meta.json"))


@bp_predict.route("/", methods=["GET"])
def view():
    return render_template("predict.html",
                           model_ready=_model_ready(),
                           max_mb=MAX_UPLOAD_BYTES // (1024 * 1024))


@bp_predict.route("/upload", methods=["POST"])
def upload():
    if not _model_ready():
        flash("Модель ще не навчена. Виконайте крок «Навчання» спочатку.", "warning")
        return redirect(url_for("predict.view"))

    f = request.files.get("image")
    if f is None or not f.filename:
        flash("Файл не вибрано.", "danger")
        return redirect(url_for("predict.view"))

    if not is_allowed_image(f.filename):
        flash("Дозволено лише PNG / JPG.", "danger")
        return redirect(url_for("predict.view"))

    safe = secure_filename(f.filename)
    if not safe:
        safe = "upload.png"
    stem = Path(safe).stem
    suffix = Path(safe).suffix or ".png"
    unique = f"{stem}_{int(time.time())}{suffix}"
    dest = UPLOADS_DIR / unique
    f.save(dest)

    if dest.stat().st_size > MAX_UPLOAD_BYTES:
        try:
            dest.unlink()
        except Exception:
            pass
        flash(f"Файл перевищує {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ.", "danger")
        return redirect(url_for("predict.view"))

    invert = request.form.get("invert") == "on"
    file_number_raw = request.form.get("file_number", "").strip()
    try:
        file_number = int(file_number_raw) if file_number_raw else 0
    except ValueError:
        file_number = 0

    try:
        result = predict_image(str(dest), invert=invert,
                               file_number=file_number, verbose=False)
    except Exception as e:
        flash(f"Помилка прогнозу: {e}", "danger")
        return redirect(url_for("predict.view"))

    chart_b64 = _render_avg_probs_chart(result.get("average_probabilities") or {})

    return render_template(
        "predict_result.html",
        image_url=url_for("predict.uploaded_file", name=unique),
        result=result,
        format_proba=format_proba,
        chart_b64=chart_b64,
    )


@bp_predict.route("/uploads/<path:name>")
def uploaded_file(name: str):
    safe = secure_filename(name)
    if not safe:
        abort(404)
    full = UPLOADS_DIR / safe
    if not full.exists():
        abort(404)
    return send_from_directory(str(UPLOADS_DIR), safe)


def _render_avg_probs_chart(avg: dict[str, float]) -> str | None:
    if not avg:
        return None
    classes = list(avg.keys())
    values = [float(avg[c]) for c in classes]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(classes, values, color="#3a6ea5")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ймовірність")
    ax.set_title("Середні ймовірності за класами")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v * 100:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
