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

FEATURES_CHOICE = "features"


def _model_ready() -> bool:
    return all(os.path.exists(os.path.join(cfg.MODELS_DIR, f))
               for f in ("best_ml_model.pkl", "scaler.pkl",
                         "label_encoder.pkl", "best_model_meta.json"))


def _features_model_name() -> str:
    try:
        import json
        with open(os.path.join(cfg.MODELS_DIR, "best_model_meta.json"),
                  "r", encoding="utf-8") as f:
            return json.load(f).get("best_name", "best_ml_model")
    except Exception:
        return "best_ml_model"


def _image_models() -> list[dict]:
    try:
        from src.image_model_store import list_saved
        return list_saved()
    except Exception:
        return []


def _feature_models() -> list[dict]:
    try:
        from src.feature_model_store import list_saved
        return list_saved()
    except Exception:
        return []


def _leaderboard() -> dict[str, dict]:
    try:
        import pandas as pd
        from src.compare_all import ALL_MODELS_CSV
        if not os.path.exists(ALL_MODELS_CSV):
            return {}
        df = pd.read_csv(ALL_MODELS_CSV)
        from src.image_model_store import CNN_METHOD
        out = {}
        for _, r in df.iterrows():
            out[str(r["method"])] = {
                "image_f1": round(float(r["image_f1_best"]), 4),
                "best_strategy": r.get("best_strategy"),
            }
        cnn = out.get(f"{CNN_METHOD} (fixed)")
        if cnn and CNN_METHOD not in out:
            out[CNN_METHOD] = cnn
        return out
    except Exception:
        return {}


def _default_choice(feature_models: list[dict],
                    board: dict[str, dict]) -> str:
    scored = [(board[f"features + {m['method']}"]["image_f1"], m["method"])
              for m in feature_models
              if f"features + {m['method']}" in board]
    if not scored:
        return FEATURES_CHOICE
    return f"features:{max(scored)[1]}"


def _features_split_f1() -> float | None:
    try:
        import pandas as pd
        p = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
        df = pd.read_csv(p)
        hit = df[df["model"] == _features_model_name()]
        return None if hit.empty else round(float(hit.iloc[0]["f1_macro"]), 4)
    except Exception:
        return None


@bp_predict.route("/", methods=["GET"])
def view():
    feature_models = _feature_models()
    board = _leaderboard()
    default_choice = _default_choice(feature_models, board)
    return render_template("predict.html",
                           model_ready=_model_ready(),
                           features_model=_features_model_name(),
                           features_split_f1=_features_split_f1(),
                           feature_models=feature_models,
                           image_models=_image_models(),
                           leaderboard=board,
                           default_choice=default_choice,
                           selected_model=request.args.get("model", default_choice),
                           invert_checked=request.args.get("invert") == "on",
                           compare_checked=request.args.get("compare_models") == "on",
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

    choice = request.form.get("model", FEATURES_CHOICE)
    compare = request.form.get("compare_models") == "on"

    try:
        if choice.startswith("image:"):
            from src.predict_image import predict_image_model
            result = predict_image_model(str(dest), method=choice[6:],
                                         invert=invert, verbose=False)
        elif choice.startswith("features:"):
            result = predict_image(str(dest), invert=invert,
                                   file_number=file_number,
                                   model_key=choice[9:], verbose=False)
        else:
            result = predict_image(str(dest), invert=invert,
                                   file_number=file_number, verbose=False)
        result.setdefault("model", _features_model_name())
        result.setdefault("branch", "features")
    except Exception as e:
        flash(f"Помилка прогнозу: {e}", "danger")
        return redirect(url_for("predict.view"))

    comparison = (_compare_models(str(dest), invert=invert,
                                  file_number=file_number) if compare else None)

    trace = _build_trace(str(dest), invert, choice, result)

    chart_b64 = _render_avg_probs_chart(result.get("average_probabilities") or {})

    again_args = {"model": choice}
    if invert:
        again_args["invert"] = "on"
    if compare:
        again_args["compare_models"] = "on"

    return render_template(
        "predict_result.html",
        image_url=url_for("predict.uploaded_file", name=unique),
        result=result,
        comparison=comparison,
        format_proba=format_proba,
        chart_b64=chart_b64,
        again_url=url_for("predict.view", **again_args),
        trace=trace,
    )


def _build_trace(path: str, invert: bool, choice: str,
                 result: dict) -> dict | None:
    try:
        from src.predict_trace import build_trace
        from src.predict_image import _segmentation_method
        spec = None
        if choice.startswith("image:"):
            from src import image_model_store as store
            spec = store.spec_from_bundle(store.load_any(choice[6:]))
        trace = build_trace(path, invert=invert,
                            segmentation=_segmentation_method(),
                            image_spec=spec)
        preds = {p["pulse_index"]: p
                 for p in result.get("pulse_predictions") or []}
        for s in trace.get("segments", []):
            if s.get("k") in preds:
                s["pred_class"] = preds[s["k"]].get("class")
                s["pred_conf"] = preds[s["k"]].get("confidence")
        return trace
    except Exception as e:
        return {"error": str(e)}


def _compare_models(path: str, invert: bool = False,
                    file_number: int = 0) -> list[dict]:
    out: list[dict] = []
    board = _leaderboard()

    def _row(model_name: str, branch: str, key: str, r=None, err=None):
        row = {"model": model_name, "branch": branch,
               "image_f1": (board.get(key) or {}).get("image_f1")}
        if err is not None:
            row["error"] = err
        else:
            row.update(final_class=r.get("final_class"),
                       confidence=r.get("final_confidence"),
                       n_valid=r.get("n_pulses_valid"))
        return row

    default_name = _features_model_name()
    try:
        out.append(_row(default_name, "features", f"features + {default_name}",
                        r=predict_image(path, invert=invert,
                                        file_number=file_number,
                                        verbose=False)))
    except Exception as e:
        out.append(_row(default_name, "features", "", err=str(e)))

    for info in _feature_models():
        try:
            r = predict_image(path, invert=invert, file_number=file_number,
                              model_key=info["method"], verbose=False)
            out.append(_row(info["method"], "features",
                            f"features + {info['method']}", r=r))
        except Exception as e:
            out.append(_row(info["method"], "features", "", err=str(e)))

    try:
        from src.predict_image import predict_image_model
        for info in _image_models():
            try:
                r = predict_image_model(path, method=info["method"],
                                        invert=invert, verbose=False)
                out.append(_row(info["method"], "image", info["method"], r=r))
            except Exception as e:
                out.append(_row(info["method"], "image", "", err=str(e)))
    except Exception:
        pass

    out.sort(key=lambda d: (d.get("image_f1") is None, -(d.get("image_f1") or 0)))
    return out


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
