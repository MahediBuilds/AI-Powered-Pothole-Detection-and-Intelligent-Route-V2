"""Persist ingested sessions and their GPS fixes."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from roadsense.ingest.gps import GpsFix
from roadsense.ingest.session import Session

_FIX_COLUMNS = "seq, t_video_s, lat, lon, accuracy_m, speed_mps, bearing_deg, utc_ms, used"


def save_session(
    conn: sqlite3.Connection,
    session: Session,
    *,
    role: str = "survey",
    include_in_map: bool = True,
    gps_source: str | None = None,
    session_id: str | None = None,
) -> str:
    """Insert the session row and all its GPS fixes in one transaction.

    ``gps_source`` is ``recorded``, ``synthetic`` or ``none``. When omitted it
    is ``recorded`` if the session has a GPS log and ``none`` otherwise; a
    generated trace must be declared ``synthetic`` by the caller.

    Fixes rejected by the accuracy filter are stored with ``used = 0``. On any
    failure nothing is left behind (the whole insert is one savepoint, so this
    also works inside a caller's transaction). Returns the session id: a new one,
    or ``session_id`` when the caller chose it (for example because the session's
    files already live in a folder named after it).
    """
    session_id = session_id or uuid.uuid4().hex
    median = None
    if session.gps is not None:
        median = session.gps.summary.median_accuracy_m
    if median is None:
        median = session.sidecar_median_accuracy_m
    fixes = session.gps.fixes if session.gps is not None else ()
    if gps_source is None:
        gps_source = "recorded" if session.gps is not None else "none"
    if session.gps is None and gps_source != "none":
        raise ValueError(f"gps_source is {gps_source!r} but the session has no GPS log")

    conn.execute("SAVEPOINT save_session")
    try:
        conn.execute(
            "INSERT INTO sessions (id, name, video_path, gps_path, started_utc, duration_s, "
            "device, median_accuracy_m, role, include_in_map, gps_source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                session.name,
                str(session.video_path),
                None if session.gps_path is None else str(session.gps_path),
                None if session.started_utc is None else session.started_utc.isoformat(),
                session.video.duration_s,
                session.device,
                median,
                role,
                int(include_in_map),
                gps_source,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.executemany(
            f"INSERT INTO gps_fixes (session_id, {_FIX_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    f.seq,
                    f.t_video_s,
                    f.lat,
                    f.lon,
                    f.accuracy_m,
                    f.speed_mps,
                    f.bearing_deg,
                    f.utc_ms,
                    int(f.used),
                )
                for f in fixes
            ],
        )
    except BaseException:
        conn.execute("ROLLBACK TO save_session")
        conn.execute("RELEASE save_session")
        raise
    conn.execute("RELEASE save_session")
    return session_id


def load_gps_fixes(conn: sqlite3.Connection, session_id: str) -> list[GpsFix]:
    """The session's GPS fixes ordered by ``seq``."""
    rows = conn.execute(
        f"SELECT {_FIX_COLUMNS} FROM gps_fixes WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    # Positional unpacking keeps this independent of the connection's row_factory.
    return [
        GpsFix(seq, t, lat, lon, acc, speed, bearing, utc_ms, bool(used))
        for seq, t, lat, lon, acc, speed, bearing, utc_ms, used in rows
    ]


def delete_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Delete the session row; its GPS fixes, jobs, observations and coverage cascade.

    Returns False when there was no such session. Does not commit and does not
    rebuild the map or remove files.
    """
    return conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0
