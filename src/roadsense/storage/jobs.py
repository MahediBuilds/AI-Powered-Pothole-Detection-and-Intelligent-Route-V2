"""Processing jobs: one row per run of the detection pipeline on a session.

Transaction rule for this module: **no function commits**. The caller owns the
transaction (``conn.commit()``), so a job update can be made atomic with other
writes. The pipeline commits its own small progress updates; the job runner
commits after creating a job.

Rows are returned as plain dicts. Timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

STATUSES = ("queued", "running", "done", "failed")
STALE_JOB_ERROR = "interrupted: the server stopped before this job finished"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _one(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
    row = cursor.fetchone()
    if row is None:
        return None
    return {d[0]: v for d, v in zip(cursor.description, tuple(row))}


def create_job(conn: sqlite3.Connection, session_id: str, *, job_id: str | None = None) -> str:
    """Insert a ``queued`` job for the session and return its id. Does not commit."""
    job_id = job_id or uuid.uuid4().hex
    now = _now()
    conn.execute(
        "INSERT INTO jobs (id, session_id, status, created_at, updated_at) "
        "VALUES (?, ?, 'queued', ?, ?)",
        (job_id, session_id, now, now),
    )
    return job_id


def _require_one(cur: sqlite3.Cursor, job_id: str) -> None:
    if cur.rowcount != 1:
        raise KeyError(f"no such job: {job_id}")


def mark_running(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    stage: str | None = None,
    progress: float | None = 0.0,
    pipeline_version: str | None = None,
    model_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Set status ``running`` (clearing any old error) and the run's provenance. Does not commit.

    ``progress`` is the starting progress (0.0 by default; None when it is not known yet).
    """
    cur = conn.execute(
        "UPDATE jobs SET status='running', stage=?, progress=?, error=NULL, "
        "pipeline_version=COALESCE(?, pipeline_version), model_sha256=COALESCE(?, model_sha256), "
        "params_json=COALESCE(?, params_json), updated_at=? WHERE id=?",
        (
            stage,
            progress,
            pipeline_version,
            model_sha256,
            None if params is None else json.dumps(params, sort_keys=True),
            _now(),
            job_id,
        ),
    )
    _require_one(cur, job_id)


def set_progress(
    conn: sqlite3.Connection, job_id: str, stage: str | None, progress: float | None
) -> None:
    """Update stage and progress (clamped to 0..1) and ``updated_at``. Does not commit."""
    if progress is not None:
        progress = min(1.0, max(0.0, float(progress)))
    cur = conn.execute(
        "UPDATE jobs SET stage=?, progress=?, updated_at=? WHERE id=?",
        (stage, progress, _now(), job_id),
    )
    _require_one(cur, job_id)


def mark_done(conn: sqlite3.Connection, job_id: str, summary: dict[str, Any]) -> None:
    """Set status ``done`` with ``summary_json`` and make this the session's active job.

    Sets ``sessions.active_job_id`` for the job's session. Does not commit.
    """
    cur = conn.execute(
        "UPDATE jobs SET status='done', stage='done', progress=1.0, error=NULL, "
        "summary_json=?, updated_at=? WHERE id=?",
        (json.dumps(summary, sort_keys=True), _now(), job_id),
    )
    _require_one(cur, job_id)
    conn.execute(
        "UPDATE sessions SET active_job_id=? WHERE id=(SELECT session_id FROM jobs WHERE id=?)",
        (job_id, job_id),
    )


def mark_failed(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    """Set status ``failed`` with the error text. Does not commit."""
    cur = conn.execute(
        "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
        (error, _now(), job_id),
    )
    _require_one(cur, job_id)


def update_summary(conn: sqlite3.Connection, job_id: str, summary: dict[str, Any]) -> None:
    """Replace ``summary_json`` (used to add late warnings). Does not commit."""
    cur = conn.execute(
        "UPDATE jobs SET summary_json=?, updated_at=? WHERE id=?",
        (json.dumps(summary, sort_keys=True), _now(), job_id),
    )
    _require_one(cur, job_id)


def fail_stale_running_jobs(conn: sqlite3.Connection) -> int:
    """Fail every ``running`` or ``queued`` job; call once at server start-up.

    Only call this while holding the database's ``ProcessLock``
    (``roadsense.storage.proclock``). The job queue lives in memory, so with the
    lock held no other process can be running or queueing jobs, and those rows
    were left behind by a process that has ended. Without the lock they may
    belong to a live server. Returns how many were failed. Does not commit.
    """
    cur = conn.execute(
        "UPDATE jobs SET status='failed', error=?, updated_at=? "
        "WHERE status IN ('queued', 'running')",
        (STALE_JOB_ERROR, _now()),
    )
    return cur.rowcount


def get_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    """The job row as a dict, or None."""
    return _one(conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)))


def latest_job_for_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """The most recently created job of the session, or None."""
    return _one(
        conn.execute(
            "SELECT * FROM jobs WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (session_id,),
        )
    )
