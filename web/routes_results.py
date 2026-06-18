from __future__ import annotations

import os
from flask import Blueprint, render_template, send_file, abort, jsonify

from src import config as cfg

from .helpers import list_result_plots, list_result_files


bp_results = Blueprint("results", __name__)


@bp_results.route("/")
def view():
    plots = list_result_plots()
    files = list_result_files()
    return render_template("results.html", plots=plots, files=files)


@bp_results.route("/img/<path:name>")
def image(name: str):
    safe = os.path.basename(name)
    p = os.path.join(cfg.RESULTS_DIR, safe)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


@bp_results.route("/file/<path:name>")
def file(name: str):
    safe = os.path.basename(name)
    p = os.path.join(cfg.RESULTS_DIR, safe)
    if not os.path.exists(p):
        abort(404)
    return send_file(p, as_attachment=True, download_name=safe)


@bp_results.route("/api/list")
def api_list():
    return jsonify({
        "plots": list_result_plots(),
        "files": [{"name": f["name"]} for f in list_result_files()],
    })
