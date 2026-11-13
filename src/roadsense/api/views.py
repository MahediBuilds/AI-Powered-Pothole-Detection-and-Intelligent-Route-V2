"""Turning database rows into the JSON objects of ``docs/api.md``."""

import json
import sqlite3
from typing import Any
from urllib.parse import quote

from roadsense.storage import jobs as jobs_db


def one(cur: sqlite3.Cursor) -> dict[str, Any] | None:
    row = cur.fetchone()
    return None if row is None else {d[0]: v for d, v in zip(cur.description, tuple(row))}


def all_rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cur.description]
    return [dict(zip(names, tuple(r))) for r in cur.fetchall()]


def _outcome(job: dict[str, Any]) -> str | None:
    """``complete`` or ``partial`` for a ``done`` job (old summaries have no outcome), else None."""
    if job["status"] != "done":
        return None
    try:
        summary = json.loads(job.get("summary_json") or "{}")
    except ValueError:
        summary = {}
    outcome = summary.get("outcome") if isinstance(summary, dict) else None
    return outcome if isinstance(outcome, str) else "complete"


def job_view(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "outcome": _outcome(job),
        "updated_at": job["updated_at"],
    }


def session_view(conn: sqlite3.Connection, s: dict[str, Any]) -> dict[str, Any]:
    """The session object: the row, GPS counts, its latest job and observation counts."""
    sid = s["id"]
    n_fixes, n_used = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(used), 0) FROM gps_fixes WHERE session_id = ?", (sid,)
    ).fetchone()
    n_obs, n_pos = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(o.cam_lat IS NOT NULL AND o.cam_lon IS NOT NULL), 0) "
        "FROM observations o JOIN sessions s ON s.id = o.session_id AND s.active_job_id = o.job_id "
        "WHERE o.session_id = ?",
        (sid,),
    ).fetchone()
    return {
        "id": sid,
        "name": s["name"],
        "role": s["role"],
        "gps_source": s["gps_source"],
        "include_in_map": bool(s["include_in_map"]),
        "created_at": s["created_at"],
        "started_utc": s["started_utc"],
        "duration_s": s["duration_s"],
        "device": s["device"],
        "gps": {
            "n_fixes": int(n_fixes),
            "n_used": int(n_used),
            "median_accuracy_m": s["median_accuracy_m"],
        },
        "job": job_view(jobs_db.latest_job_for_session(conn, sid)),
        "n_observations": int(n_obs),
        "n_with_position": int(n_pos),
    }


def session_report(conn: sqlite3.Connection, s: dict[str, Any]) -> dict[str, Any] | None:
    """The detection report of the session's latest finished (active) job, or None."""
    if not s["active_job_id"]:
        return None
    job = jobs_db.get_job(conn, s["active_job_id"])
    if job is None or not job["summary_json"]:
        return None
    return json.loads(job["summary_json"])


def observation_view(o: dict[str, Any]) -> dict[str, Any]:
    """One tracked pothole as in ``/detections``."""
    has_pos = o["cam_lat"] is not None and o["cam_lon"] is not None
    image = o["image_path"]
    return {
        "id": o["id"],
        "track_id": o["track_id"],
        "n_frames": o["n_frames"],
        "confidence": o["confidence"],
        "first_t_video_s": o["first_t_video_s"],
        "t_video_s": o["t_video_s"],
        "frame_index": o["frame_index"],
        "box": [o["x1"], o["y1"], o["x2"], o["y2"]],
        "frame_w": o["frame_w"],
        "frame_h": o["frame_h"],
        "position": (
            {"lat": o["cam_lat"], "lon": o["cam_lon"], "accuracy_m": o["cam_accuracy_m"]}
            if has_pos
            else None
        ),
        "pothole_id": o["pothole_id"],
        "image_url": f"/api/artifacts/{quote(image)}" if image else None,
    }


def pothole_view(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": p["id"],
        "lat": p["lat"],
        "lon": p["lon"],
        "confidence": p["confidence"],
        "n_observations": p["n_observations"],
        "n_sessions": p["n_sessions"],
        "gps_source": p["gps_source"],
        "on_road": p["edge_u"] is not None,
        "first_seen_utc": p["first_seen_utc"],
        "last_seen_utc": p["last_seen_utc"],
    }
