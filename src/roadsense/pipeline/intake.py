"""Register a recording as a session: copy its files, validate them, save it.

Shared by the HTTP API, the command line and the demo script, so all three
store sessions the same way. The files are copied into
``<uploads.dir>/<session_id>/`` under the recorder naming convention::

    <stem>.<ext>   <stem>_gps.csv   <stem>.json

where ``<stem>`` is a sanitised version of the original video name. If the video
or the GPS file is invalid, the copied files are removed again and the ingest
error is raised unchanged.

Like the storage helpers, ``create_session_from_files`` does not commit; the
caller owns the transaction. (The copied files are removed if the insert fails,
but not if the caller later rolls back.)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from roadsense.config import AppConfig
from roadsense.ingest import IngestError
from roadsense.ingest.session import load_session
from roadsense.storage.sessions import save_session

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
ROLES = ("survey", "tuning", "test", "demo")
GPS_SOURCES = ("recorded", "synthetic")
MAX_STEM_LENGTH = 80

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


class UnsupportedFileError(IngestError):
    """The video file has an extension that is not supported."""


def safe_stem(name: str) -> str:
    """A file-name stem of letters, digits, ``-`` and ``_`` (never empty, length-capped).

    Any extension in ``name`` is dropped first; runs of other characters become
    one underscore.
    """
    stem = Path(name.replace("\\", "/")).stem
    stem = _UNSAFE.sub("_", stem).strip("_")
    return stem[:MAX_STEM_LENGTH].strip("_") or "session"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return path.resolve() != parent.resolve()


def _sidecar_says_synthetic(sidecar: str | Path) -> bool:
    """True if the recorder JSON is an object with ``"synthetic": true`` (unreadable: False)."""
    try:
        data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # the ingest code already warns about a bad sidecar
        return False
    return isinstance(data, dict) and data.get("synthetic") is True


def _place(src: Path, dest: Path, move: bool) -> None:
    """Copy ``src`` to ``dest``, or move it (atomic rename, ``shutil.move`` across volumes)."""
    if not move:
        shutil.copyfile(src, dest)
        return
    try:
        os.replace(src, dest)
    except OSError:
        shutil.move(str(src), str(dest))


def create_session_from_files(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    *,
    video: str | Path,
    gps: str | Path | None = None,
    sidecar: str | Path | None = None,
    name: str | None = None,
    role: str = "survey",
    gps_source: str | None = None,
    include_in_map: bool = True,
    move: bool = False,
) -> str:
    """Copy the files into the uploads folder, load and validate them, save the session.

    ``name`` supplies the session name and file stem (default: the video's own
    name); it is sanitised either way. ``gps_source`` is ``none`` without a GPS
    file; with one it is ``recorded`` unless the caller says ``synthetic``, or
    the sidecar JSON is an object with ``"synthetic": true`` (written by the demo
    script), which forces ``synthetic`` whatever the caller passed; an unreadable
    or non-object sidecar is ignored here. With ``move`` the files are moved into
    the session folder (``os.replace``, falling back to ``shutil.move``) instead
    of copied; use it only for temporary files that belong to the caller's own
    staging area. On failure the session folder, with any moved files, is removed.
    Raises ``UnsupportedFileError`` for a video extension outside
    ``VIDEO_EXTENSIONS``, ``ValueError`` for an unknown role or ``gps_source``,
    and the ingest errors of ``load_session`` (the copies are removed first).
    Returns the session id; the caller commits.
    """
    video = Path(video)
    ext = video.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise UnsupportedFileError(
            f"{video.name}: unsupported video type {ext or '(no extension)'!r}; "
            f"use one of {', '.join(VIDEO_EXTENSIONS)}"
        )
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if gps_source is not None and gps_source not in GPS_SOURCES + ("none",):
        raise ValueError(f"gps_source must be one of {GPS_SOURCES}, got {gps_source!r}")
    if gps is None:
        gps_source = "none"
    elif gps_source in (None, "none"):
        gps_source = "recorded"
    if gps is not None and sidecar is not None and _sidecar_says_synthetic(sidecar):
        gps_source = "synthetic"

    stem = safe_stem(name or video.name)
    session_id = uuid.uuid4().hex
    folder = Path(cfg.uploads.dir) / session_id
    folder.mkdir(parents=True, exist_ok=False)
    try:
        video_copy = folder / f"{stem}{ext}"
        _place(Path(video), video_copy, move)
        gps_copy = sidecar_copy = None
        if gps is not None:
            gps_copy = folder / f"{stem}_gps.csv"
            _place(Path(gps), gps_copy, move)
        if sidecar is not None:
            sidecar_copy = folder / f"{stem}.json"
            _place(Path(sidecar), sidecar_copy, move)
        session = load_session(
            video_copy,
            gps_copy,
            sidecar_copy,
            max_accuracy_m=float(cfg.ingest.max_gps_accuracy_m),
            max_gap_s=float(cfg.ingest.max_gps_gap_s),
        )
        return save_session(
            conn,
            session,
            role=role,
            include_in_map=include_in_map,
            gps_source=gps_source,
            session_id=session_id,
        )
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def delete_session_files(cfg: AppConfig, session_id: str) -> None:
    """Remove the session's upload folder and artifact folder, if they exist.

    Refuses (``ValueError``) an id that is not a plain folder name, and never
    touches a path that does not lie strictly inside the configured directory.
    """
    if not session_id or session_id in (".", "..") or any(c in session_id for c in "/\\"):
        raise ValueError(f"not a valid session id: {session_id!r}")
    for base in (Path(cfg.uploads.dir), Path(cfg.uploads.artifacts_dir)):
        target = base / session_id
        if not _inside(target, base):
            raise ValueError(f"{target} is not inside {base}")
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
