from __future__ import annotations

import os
import glob
import math

import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    jsonify, send_file, abort,
)

from src import config as cfg
from .jobs import start_job, get_job, public_status


bp_experiments = Blueprint("experiments", __name__)


CURRENT: dict[str, str] = {}

BRANCHES = ("image", "features")

EXTRA_JOBS = ("compare", "save_models")


def _safe(v):
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)
    if isinstance(v, (int, bool)) or v is None:
        return v
    return str(v)


def _read_table(path: str, cols: list[str] | None = None) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return {
        "columns": list(df.columns),
        "rows": [[_safe(v) for v in row] for row in df.values.tolist()],
        "path": cfg.to_rel(path),
    }


def _tables() -> dict:
    from src.image_methods import IMAGE_RESULTS_CSV, IMAGE_ABLATION_CSV
    from src.branch_features import FEATURES_RESULTS_CSV
    from src.compare_all import ALL_MODELS_CSV

    return {
        "all_models": _read_table(ALL_MODELS_CSV, [
            "branch", "method", "input", "pulse_f1",
            "image_f1_majority", "image_f1_mean_proba", "image_f1_weighted",
            "image_f1_best", "best_strategy", "image_f1_std",
            "train_sec", "source", "params_key"]),
        "ablation": _read_table(IMAGE_ABLATION_CSV, [
            "spec", "resize_mode", "channels", "size",
            "pulse_f1_oof", "image_f1_oof", "delta_vs_baseline"]),
        "image": _read_table(IMAGE_RESULTS_CSV, [
            "method", "spec", "pulse_f1_oof", "image_f1_oof",
            "image_f1_fold_std", "train_sec"]),
        "features": _read_table(FEATURES_RESULTS_CSV, [
            "method", "level", "f1_macro", "f1_macro_std", "accuracy"]),
        "comparison": _read_table(
            os.path.join(cfg.RESULTS_DIR, "branch_comparison.csv")),
    }


def _montages() -> list[dict]:
    out = []
    root = cfg.IMAGE_PREP_DEBUG_DIR
    if not os.path.isdir(root):
        return out
    for p in sorted(glob.glob(os.path.join(root, "*", "montage.png"))):
        out.append({"spec": os.path.basename(os.path.dirname(p))})
    return out


def _dataset_ready() -> dict:
    n_pulse_png = 0
    if os.path.isdir(cfg.PULSE_IMAGES_DIR):
        n_pulse_png = sum(
            len(glob.glob(os.path.join(cfg.PULSE_IMAGES_DIR, d, "*.png")))
            for d in cfg.CLASS_FOLDERS)
    return {
        "image": n_pulse_png > 0,
        "n_pulse_images": n_pulse_png,
        "features": os.path.exists(cfg.FEATURES_CSV),
    }


@bp_experiments.route("/", methods=["GET"])
def view():
    jobs = {b: (get_job(CURRENT[b]) if b in CURRENT else None)
            for b in BRANCHES + EXTRA_JOBS}
    any_running = any(j and j["status"] == "running" for j in jobs.values())
    saved_models = []
    try:
        from src.image_model_store import list_saved
        saved_models = list_saved()
    except Exception:
        pass
    return render_template(
        "experiments.html",
        jobs=jobs,
        any_running=any_running,
        job_ids={b: CURRENT.get(b) for b in BRANCHES + EXTRA_JOBS},
        tables=_tables(),
        montages=_montages(),
        ready=_dataset_ready(),
        saved_models=saved_models,
        main_spec="letterbox_gray-mask_192x96",
    )


@bp_experiments.route("/run", methods=["POST"])
def run():
    branch = request.form.get("branch", "both")
    targets = BRANCHES if branch == "both" else (branch,)

    opts = {
        "do_ablation": request.form.get("do_ablation") == "on",
        "do_hog": request.form.get("do_hog") == "on",
        "do_transfer": request.form.get("do_transfer") == "on",
        "do_cnn": request.form.get("do_cnn") == "on",
        "backbone": request.form.get("backbone", "MobileNetV2"),
        "cnn_epochs": int(request.form.get("cnn_epochs") or 60),
    }
    n_splits = int(request.form.get("splits") or 5)

    for b in targets:
        job = get_job(CURRENT.get(b, ""))
        if job and job["status"] == "running":
            continue

        if b == "image":
            from src.branch_image import run_image_branch
            CURRENT[b] = start_job(
                run_image_branch, n_splits=n_splits, verbose=False,
                label="image-гілка", inject_progress=True, **opts)
        else:
            from src.branch_features import run_feature_branch
            CURRENT[b] = start_job(
                run_feature_branch, n_splits=n_splits, verbose=False,
                label="features-гілка", inject_progress=True)

    return redirect(url_for("experiments.view"))


@bp_experiments.route("/compare", methods=["POST"])
def compare():
    from src.compare_all import compare_all

    job = get_job(CURRENT.get("compare", ""))
    if not (job and job["status"] == "running"):
        CURRENT["compare"] = start_job(
            compare_all,
            n_splits=int(request.form.get("splits") or 5),
            run_cnn=request.form.get("with_cnn") == "on",
            verbose=False,
            label="порівняння всіх моделей", inject_progress=True)
    return redirect(url_for("experiments.view"))


@bp_experiments.route("/save-image-models", methods=["POST"])
def save_image_models():
    from src.image_model_store import save_all

    job = get_job(CURRENT.get("save_models", ""))
    if not (job and job["status"] == "running"):
        CURRENT["save_models"] = start_job(
            save_all, verbose=False, label="збереження картинкових моделей")
    return redirect(url_for("experiments.view"))


@bp_experiments.route("/api/status")
def api_status():
    return jsonify({
        b: (public_status(CURRENT[b]) if b in CURRENT
            else {"status": "idle", "progress": 0, "message": ""})
        for b in BRANCHES + EXTRA_JOBS
    })


@bp_experiments.route("/api/tables")
def api_tables():
    return jsonify(_tables())


@bp_experiments.route("/montage/<spec>")
def serve_montage(spec: str):
    safe = os.path.basename(spec)
    p = os.path.join(cfg.IMAGE_PREP_DEBUG_DIR, safe, "montage.png")
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


@bp_experiments.route("/results/<name>")
def serve_result(name: str):
    safe = os.path.basename(name)
    p = os.path.join(cfg.RESULTS_DIR, safe)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)
