from __future__ import annotations

import os
import math
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    jsonify, send_file, abort,
)

from src import config as cfg
from src.train_ml_models import train_and_compare, HAS_XGBOOST

from .jobs import start_job, get_job


bp_training = Blueprint("training", __name__)


def _safe_value(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (int, bool)):
        return v
    if v is None:
        return None
    return str(v)


def _read_classification_report() -> str | None:
    p = os.path.join(cfg.RESULTS_DIR, "classification_report.txt")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _serialize_results(result: dict) -> dict:
    res_df = result.get("results")
    rows = res_df.to_dict(orient="records") if res_df is not None else []
    rows_clean = [{k: _safe_value(v) for k, v in r.items()} for r in rows]
    return {
        "best_name": result.get("best_name"),
        "rows": rows_clean,
        "report": result.get("report"),
        "class_names": list(result.get("class_names") or []),
    }


def _summary_from_disk() -> dict | None:
    import json
    import pandas as pd

    meta_path = os.path.join(cfg.MODELS_DIR, "best_model_meta.json")
    cmp_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
    if not (os.path.exists(meta_path) and os.path.exists(cmp_path)):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        df = pd.read_csv(cmp_path)
        rows = [
            {k: _safe_value(v) for k, v in r.items()}
            for r in df.to_dict(orient="records")
        ]
        return {
            "best_name": meta.get("best_name"),
            "rows": rows,
            "report": _read_classification_report(),
            "class_names": [],
        }
    except Exception:
        return None


@bp_training.route("/", methods=["GET"])
def view():
    job_id = request.args.get("job")
    job = get_job(job_id) if job_id else None

    summary = None
    if job and job.get("status") == "done" and job.get("result") is not None:
        summary = _serialize_results(job["result"])
    elif job is None:
        summary = _summary_from_disk()

    cm_name = None
    if summary and summary.get("best_name"):
        candidate = f"confusion_matrix_{summary['best_name']}.png"
        if os.path.exists(os.path.join(cfg.RESULTS_DIR, candidate)):
            cm_name = candidate
    if cm_name is None:
        import glob as _glob
        c = sorted(_glob.glob(os.path.join(cfg.RESULTS_DIR, "confusion_matrix_*.png")))
        if c:
            cm_name = os.path.basename(c[0])

    has_model_comparison = os.path.exists(
        os.path.join(cfg.RESULTS_DIR, "model_comparison.png"))

    return render_template(
        "training.html",
        job_id=job_id,
        job=job,
        summary=summary,
        cm_name=cm_name,
        has_model_comparison=has_model_comparison,
        report_text=_read_classification_report() if summary else None,
        features_csv_exists=os.path.exists(cfg.FEATURES_CSV),
        has_xgboost=HAS_XGBOOST,
    )


@bp_training.route("/run", methods=["POST"])
def run():
    use_fn = request.form.get("use_file_number") == "on"
    use_xgb = request.form.get("use_xgboost") == "on"
    use_ens = request.form.get("use_ensembles") == "on"
    tune_top_k = 2 if request.form.get("tune_top2") == "on" else 0

    job_id = start_job(
        train_and_compare,
        use_file_number=use_fn,
        use_xgboost=use_xgb,
        use_ensembles=use_ens,
        tune_top_k=tune_top_k,
        verbose=False,
        label="train_and_compare",
    )
    return redirect(url_for("training.view", job=job_id))


@bp_training.route("/api/summary/<job_id>")
def api_summary(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"ready": False})
    if job["status"] != "done" or job["result"] is None:
        return jsonify({"ready": False, "status": job["status"]})
    return jsonify({"ready": True, **_serialize_results(job["result"])})


@bp_training.route("/results/<name>")
def serve_result(name: str):
    safe = os.path.basename(name)
    p = os.path.join(cfg.RESULTS_DIR, safe)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)
