from __future__ import annotations

import os
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    send_file, abort, jsonify, flash,
)

import glob
import pandas as pd

from src import config as cfg
from src.dataset_builder import build_dataset
from src.train_ml_models import train_and_compare
from src.evaluate import evaluate_saved_model
from src.data_loader import create_image_index
from src.visualization import generate_all_plots

from .jobs import start_job, get_job


def _find_first_image() -> str | None:
    paths = []
    for folder in cfg.CLASS_FOLDERS:
        paths.extend(sorted(glob.glob(
            os.path.join(cfg.DATASET_DIR, folder, "*.png"))))
    return paths[0] if paths else None


def _full_pipeline() -> dict:
    create_image_index()
    df = build_dataset(invert=False, save_pulse_images=True, verbose=False)
    info = {"n_pulses": int(len(df)), "trained": False, "plots": []}
    if df.empty:
        info["error"] = "Не отримано жодного імпульсу."
        return info
    res = train_and_compare(verbose=False)
    info["trained"] = True
    info["best_name"] = res["best_name"]
    saved = generate_all_plots(df, results_dict=res,
                               example_image=_find_first_image())
    info["plots"] = [os.path.basename(p) for p in saved]
    return info


def _generate_plots_only() -> dict:
    if not os.path.exists(cfg.FEATURES_CSV):
        return {"error": f"Немає {cfg.FEATURES_CSV}. Запустіть pipeline."}
    df = pd.read_csv(cfg.FEATURES_CSV)
    res = None
    try:
        res = evaluate_saved_model(verbose=False)
    except Exception:
        res = None
    cmp_csv = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
    results_dict = None
    if res and os.path.exists(cmp_csv):
        results_dict = {
            "results": pd.read_csv(cmp_csv),
            "best_name": res["best_name"],
            "confusion": res["confusion"],
            "class_names": res["class_names"],
        }
    saved = generate_all_plots(df, results_dict=results_dict,
                               example_image=_find_first_image())
    return {"plots": [os.path.basename(p) for p in saved]}


def _experiment_invert() -> dict:
    df_o = build_dataset(invert=False, save_pulse_images=False, verbose=False)
    if df_o.empty:
        return {"error": "Не вдалось зібрати original-датасет."}
    df_o.to_csv(os.path.join(cfg.FEATURES_DIR, "pulse_features_original.csv"),
                index=False)
    res_o = train_and_compare(features_csv=os.path.join(
        cfg.FEATURES_DIR, "pulse_features_original.csv"), verbose=False)

    df_i = build_dataset(invert=True, save_pulse_images=False, verbose=False)
    if df_i.empty:
        return {"error": "Не вдалось зібрати inverted-датасет."}
    df_i.to_csv(os.path.join(cfg.FEATURES_DIR, "pulse_features_inverted.csv"),
                index=False)
    res_i = train_and_compare(features_csv=os.path.join(
        cfg.FEATURES_DIR, "pulse_features_inverted.csv"), verbose=False)

    out_path = os.path.join(cfg.RESULTS_DIR, "original_vs_inverted.csv")
    pd.DataFrame([
        {"variant": "original",
         "best_model": res_o["best_name"],
         "f1_macro": float(res_o["results"].iloc[0]["f1_macro"]),
         "accuracy": float(res_o["results"].iloc[0]["accuracy"])},
        {"variant": "inverted",
         "best_model": res_i["best_name"],
         "f1_macro": float(res_i["results"].iloc[0]["f1_macro"]),
         "accuracy": float(res_i["results"].iloc[0]["accuracy"])},
    ]).to_csv(out_path, index=False)
    return {
        "original": {"best": res_o["best_name"],
                     "f1_macro": float(res_o["results"].iloc[0]["f1_macro"]),
                     "accuracy": float(res_o["results"].iloc[0]["accuracy"])},
        "inverted": {"best": res_i["best_name"],
                     "f1_macro": float(res_i["results"].iloc[0]["f1_macro"]),
                     "accuracy": float(res_i["results"].iloc[0]["accuracy"])},
        "csv": "original_vs_inverted.csv",
    }


bp_pipeline = Blueprint("pipeline", __name__)


@bp_pipeline.route("/", methods=["GET"])
def view():
    job_id = request.args.get("job")
    job = get_job(job_id) if job_id else None

    summary = None
    extra = None
    if job and job.get("status") == "done" and job.get("result") is not None:
        result = job["result"]
        if isinstance(result, dict):
            extra = result
        else:
            try:
                n_pulses = int(len(result))
                class_counts = (
                    result["class"].value_counts().to_dict()
                    if hasattr(result, "columns") and "class" in result.columns else {}
                )
            except Exception:
                n_pulses, class_counts = 0, {}
            summary = {
                "n_pulses": n_pulses,
                "class_counts": class_counts,
                "features_csv": cfg.FEATURES_CSV,
                "features_csv_exists": os.path.exists(cfg.FEATURES_CSV),
            }

    return render_template(
        "pipeline.html",
        job_id=job_id,
        job=job,
        summary=summary,
        extra=extra,
        features_csv_exists=os.path.exists(cfg.FEATURES_CSV),
    )


@bp_pipeline.route("/run", methods=["POST"])
def run():
    invert = request.form.get("invert") == "on"
    save_pulse_images = request.form.get("save_pulse_images") == "on"
    try:
        augment_factor = max(0, min(10, int(request.form.get("augment_factor", "0"))))
    except ValueError:
        augment_factor = 0
    segmentation = request.form.get("segmentation", "full_cycle")
    if segmentation not in ("full_cycle", "peak_centered"):
        segmentation = "full_cycle"

    job_id = start_job(
        build_dataset,
        invert=invert,
        save_pulse_images=save_pulse_images,
        save_pulse_signals=False,
        augment_factor=augment_factor,
        segmentation=segmentation,
        verbose=False,
        label="build_dataset",
    )
    return redirect(url_for("pipeline.view", job=job_id))


@bp_pipeline.route("/download/features")
def download_features():
    if not os.path.exists(cfg.FEATURES_CSV):
        abort(404)
    return send_file(cfg.FEATURES_CSV, as_attachment=True,
                     download_name="pulse_features.csv")


@bp_pipeline.route("/api/summary/<job_id>")
def api_summary(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"ready": False})
    if job["status"] != "done" or job["result"] is None:
        return jsonify({"ready": False, "status": job["status"]})

    df = job["result"]
    try:
        n_pulses = int(len(df))
        class_counts = (
            df["class"].value_counts().to_dict()
            if hasattr(df, "columns") and "class" in df.columns else {}
        )
    except Exception:
        n_pulses, class_counts = 0, {}

    return jsonify({
        "ready": True,
        "n_pulses": n_pulses,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "features_csv_exists": os.path.exists(cfg.FEATURES_CSV),
    })


@bp_pipeline.route("/full", methods=["POST"])
def run_full():
    job_id = start_job(_full_pipeline, label="full_pipeline")
    flash("Запущено повний pipeline. Може зайняти кілька хвилин.", "info")
    return redirect(url_for("pipeline.view", job=job_id))


@bp_pipeline.route("/plots", methods=["POST"])
def run_plots():
    job_id = start_job(_generate_plots_only, label="generate_plots")
    flash("Запущено генерацію графіків.", "info")
    return redirect(url_for("results.view"))


@bp_pipeline.route("/experiment", methods=["POST"])
def run_experiment():
    job_id = start_job(_experiment_invert, label="experiment_invert")
    flash("Запущено експеримент original vs inverted.", "info")
    return redirect(url_for("pipeline.view", job=job_id))
