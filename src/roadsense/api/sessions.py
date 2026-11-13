"""Session endpoints: upload, list, detail, status, detections, track, reprocess, patch, delete."""

import logging
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from roadsense.api.deps import ACTIVE_JOB_STATUSES, AppState, db_conn, get_state
from roadsense.api.errors import ApiError
from roadsense.api.views import (
    all_rows,
    observation_view,
    one,
    session_report,
    session_view,
)
from roadsense.ingest import IngestError
from roadsense.ingest.gps import GpsError
from roadsense.mapstate import write_and_rebuild
from roadsense.pipeline.intake import (
    VIDEO_EXTENSIONS,
    UnsupportedFileError,
    create_session_from_files,
    delete_session_files,
)
from roadsense.storage import jobs as jobs_db
from roadsense.storage import observations as obs_db
from roadsense.storage.sessions import delete_session, load_gps_fixes

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

CHUNK = 1 << 20


def _session_row(conn, session_id: str) -> dict:
    row = one(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)))
    if row is None:
        raise ApiError(404, "not_found", f"No session with id {session_id!r}.")
    return row


def _is_active(conn, session_id: str) -> bool:
    job = jobs_db.latest_job_for_session(conn, session_id)
    return job is not None and job["status"] in ACTIVE_JOB_STATUSES


def _copy_limited(upload: UploadFile, dest: Path, used: int, limit: int) -> int:
    """Copy an upload to ``dest`` in chunks; returns the running total of bytes copied."""
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(CHUNK)
            if not chunk:
                return used
            used += len(chunk)
            if used > limit:
                raise ApiError(
                    413, "too_large", f"The upload is larger than the {limit // (1 << 20)} MB limit."
                )
            out.write(chunk)


def _session_and_job(state: AppState, session_id: str) -> dict:
    with db_conn(state) as conn:
        view = session_view(conn, _session_row(conn, session_id))
    return {"session": view, "job": view["job"]}


@router.post("/sessions", status_code=202)
def create_session(
    request: Request,
    video: UploadFile = File(...),
    gps: UploadFile | None = File(None),
    sidecar: UploadFile | None = File(None),
    role: Literal["survey", "tuning", "test", "demo"] = Form("survey"),
    gps_source: Literal["recorded", "synthetic"] = Form("recorded"),
    include_in_map: bool = Form(True),
) -> dict:
    state = get_state(request)
    cfg = state.cfg
    ext = Path(video.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ApiError(
            400,
            "unsupported_file",
            f"Unsupported video type {ext or '(no extension)'!r}; use one of "
            f"{', '.join(VIDEO_EXTENSIONS)}.",
        )
    limit = cfg.uploads.max_upload_mb * (1 << 20)
    tmp = Path(cfg.uploads.dir) / f".incoming-{uuid.uuid4().hex}"
    try:
        tmp.mkdir(parents=True)
        used = 0
        saved: dict[str, Path] = {}
        for key, upload, filename in (
            ("video", video, f"video{ext}"),
            ("gps", gps, "gps.csv"),
            ("sidecar", sidecar, "sidecar.json"),
        ):
            if upload is not None:
                saved[key] = tmp / filename
                used = _copy_limited(upload, saved[key], used, limit)
        with db_conn(state) as conn:
            try:
                session_id = create_session_from_files(
                    conn,
                    cfg,
                    video=saved["video"],
                    gps=saved.get("gps"),
                    sidecar=saved.get("sidecar"),
                    name=video.filename,
                    role=role,
                    gps_source=gps_source,
                    include_in_map=include_in_map,
                    move=True,  # the .incoming files are this request's own temporaries
                )
                conn.commit()
            except UnsupportedFileError as exc:
                raise ApiError(400, "unsupported_file", str(exc)) from exc
            except GpsError as exc:
                raise ApiError(422, "invalid_gps", str(exc)) from exc
            except IngestError as exc:
                raise ApiError(422, "invalid_video", str(exc)) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    state.runner.submit(session_id)
    return _session_and_job(state, session_id)


@router.get("/sessions")
def list_sessions(request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        rows = all_rows(conn.execute("SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC"))
        return {"sessions": [session_view(conn, s) for s in rows]}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        row = _session_row(conn, session_id)
        return {**session_view(conn, row), "report": session_report(conn, row)}


@router.get("/sessions/{session_id}/status")
def get_status(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        view = session_view(conn, _session_row(conn, session_id))
        return {"session_id": session_id, "job": view["job"]}


@router.get("/sessions/{session_id}/detections")
def get_detections(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        row = _session_row(conn, session_id)
        return {
            "session_id": session_id,
            "job_id": row["active_job_id"],
            "gps_source": row["gps_source"],
            "observations": [
                observation_view(o) for o in obs_db.list_observations(conn, session_id)
            ],
        }


@router.get("/sessions/{session_id}/track")
def get_track(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        row = _session_row(conn, session_id)
        return {
            "gps_source": row["gps_source"],
            "points": [
                [f.lat, f.lon, f.t_video_s, f.used] for f in load_gps_fixes(conn, session_id)
            ],
        }


@router.post("/sessions/{session_id}/reprocess", status_code=202)
def reprocess(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with state.job_lock:
        with db_conn(state) as conn:
            _session_row(conn, session_id)
            if _is_active(conn, session_id):
                raise ApiError(
                    409, "job_running", "This session is already queued or being processed."
                )
        state.runner.submit(session_id)
    return _session_and_job(state, session_id)


class SessionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_in_map: bool


@router.patch("/sessions/{session_id}")
def patch_session(session_id: str, body: SessionPatch, request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        _session_row(conn, session_id)

        def change(c) -> None:
            c.execute(
                "UPDATE sessions SET include_in_map = ? WHERE id = ?",
                (int(body.include_in_map), session_id),
            )

        # map_lock before any write: the job runner takes them in the same order
        with state.map_lock:
            write_and_rebuild(conn, state.cfg, state.graph, before=change)
        return session_view(conn, _session_row(conn, session_id))


@router.delete("/sessions/{session_id}")
def remove_session(session_id: str, request: Request) -> dict:
    state = get_state(request)
    with state.job_lock:
        with db_conn(state) as conn:
            _session_row(conn, session_id)
            if _is_active(conn, session_id):
                raise ApiError(
                    409, "job_running", "This session is being processed; try again when it is done."
                )
            with state.map_lock:
                write_and_rebuild(
                    conn, state.cfg, state.graph, before=lambda c: delete_session(c, session_id)
                )
    delete_session_files(state.cfg, session_id)
    return {"deleted": session_id}


@router.get("/artifacts/{path:path}")
def get_artifact(path: str, request: Request) -> FileResponse:
    base = Path(get_state(request).cfg.uploads.artifacts_dir).resolve()
    try:
        target = (base / path).resolve()
        found = bool(path) and target.is_relative_to(base) and target.is_file()
    except (OSError, ValueError):  # for example a NUL byte or an invalid Windows name
        found = False
    if not found:
        raise ApiError(404, "not_found", "No such file.")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type)
