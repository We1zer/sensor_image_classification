from __future__ import annotations

import uuid
import threading
import traceback
from typing import Any, Callable


JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _make_id() -> str:
    return uuid.uuid4().hex[:8]


def start_job(target: Callable, *args, label: str = "", **kwargs) -> str:
    job_id = _make_id()
    with _LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "label": label or getattr(target, "__name__", "job"),
            "status": "running",
            "progress": 0,
            "message": "Запущено...",
            "result": None,
            "error": None,
        }

    def runner() -> None:
        try:
            res = target(*args, **kwargs)
            with _LOCK:
                JOBS[job_id].update(
                    status="done",
                    progress=100,
                    message="Готово.",
                    result=res,
                )
        except Exception as e:
            with _LOCK:
                JOBS[job_id].update(
                    status="error",
                    progress=100,
                    error=str(e),
                    message=traceback.format_exc(),
                )

    threading.Thread(target=runner, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def public_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        return {"status": "unknown", "progress": 0, "message": "Невідомий job_id"}
    out = {
        "id": job["id"],
        "label": job["label"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job["error"],
        "has_result": job["result"] is not None,
    }
    return out
