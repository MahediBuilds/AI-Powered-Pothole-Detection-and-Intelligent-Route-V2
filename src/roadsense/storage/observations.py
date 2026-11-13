"""Observations: one row per tracked pothole per processing job.

Transaction rule for this module: **no function commits**; the caller owns the
transaction. Rows are returned as plain dicts. ``edge_*`` and ``pothole_id`` are
left NULL on insert; the map build fills them.
"""

from __future__ import annotations

import sqlite3
from typing import Any

_REQUIRED = (
    "track_id", "n_frames", "confidence", "t_video_s", "frame_index", "x1", "y1", "x2", "y2",
)  # fmt: skip
_OPTIONAL = (
    "first_t_video_s", "frame_w", "frame_h", "image_path", "cam_lat", "cam_lon", "cam_accuracy_m",
)  # fmt: skip
_INSERT_COLUMNS = ("session_id", "job_id", *_REQUIRED, *_OPTIONAL)


def insert_observations(
    conn: sqlite3.Connection, session_id: str, job_id: str, rows: list[dict[str, Any]]
) -> int:
    """Insert observation rows for a job and return how many. Does not commit.

    Each row needs ``track_id, n_frames, confidence, t_video_s, frame_index,
    x1, y1, x2, y2``; the optional keys (``first_t_video_s, frame_w, frame_h,
    image_path, cam_lat, cam_lon, cam_accuracy_m``) default to NULL. A row with
    a missing required key raises ``KeyError`` before anything is inserted.
    """
    params = []
    for row in rows:
        missing = [k for k in _REQUIRED if k not in row]
        if missing:
            raise KeyError(f"observation row is missing {missing}")
        params.append(
            (session_id, job_id, *(row[k] for k in _REQUIRED), *(row.get(k) for k in _OPTIONAL))
        )
    conn.executemany(
        f"INSERT INTO observations ({', '.join(_INSERT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(_INSERT_COLUMNS))})",
        params,
    )
    return len(params)


def _dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, tuple(r))) for r in cursor.fetchall()]


def list_observations(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """Observations of the session's ACTIVE job, ordered by time then track id.

    Empty when the session has no active job.
    """
    return _dicts(
        conn.execute(
            "SELECT o.* FROM observations o JOIN sessions s "
            "ON s.id = o.session_id AND s.active_job_id = o.job_id "
            "WHERE o.session_id = ? ORDER BY o.t_video_s, o.track_id",
            (session_id,),
        )
    )


def list_observations_for_job(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    """All observations of one job, ordered by track id."""
    return _dicts(
        conn.execute("SELECT * FROM observations WHERE job_id = ? ORDER BY track_id", (job_id,))
    )


def count_observations(conn: sqlite3.Connection, session_id: str) -> int:
    """Number of observations of the session's active job (0 without one)."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM observations o JOIN sessions s "
            "ON s.id = o.session_id AND s.active_job_id = o.job_id WHERE o.session_id = ?",
            (session_id,),
        ).fetchone()[0]
    )


def count_observations_for_job(conn: sqlite3.Connection, job_id: str) -> int:
    """Number of observations stored for one job."""
    return int(
        conn.execute("SELECT COUNT(*) FROM observations WHERE job_id = ?", (job_id,)).fetchone()[0]
    )


def count_positioned_observations(conn: sqlite3.Connection, session_id: str) -> int:
    """Observations of the active job that have a camera position."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM observations o JOIN sessions s "
            "ON s.id = o.session_id AND s.active_job_id = o.job_id "
            "WHERE o.session_id = ? AND o.cam_lat IS NOT NULL AND o.cam_lon IS NOT NULL",
            (session_id,),
        ).fetchone()[0]
    )
