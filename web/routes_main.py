from __future__ import annotations

from flask import Blueprint, render_template, jsonify

from .helpers import project_status
from .jobs import public_status


bp_main = Blueprint("main", __name__)


@bp_main.route("/")
def index():
    status = project_status()
    return render_template("index.html", status=status)


@bp_main.route("/api/health")
def api_health():
    return jsonify({"ok": True})


@bp_main.route("/api/job/<job_id>")
def api_job(job_id: str):
    return jsonify(public_status(job_id))
