"""FastAPI application factory.

Imported only when the ``api`` extra is installed. ``create_app`` wires the
routers, the error handlers and the static front end; the lifespan handler
prepares the database, loads the road graph, brings the map up to date and
starts the job runner. Nothing heavy (model, graph) is loaded at import time.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from roadsense import __version__
from roadsense.api import mapdata, routes, sessions, system
from roadsense.api.deps import AppState
from roadsense.api.errors import ApiError, error_response, install_error_handlers
from roadsense.config import AppConfig
from roadsense.detection.detector import DetectorLike
from roadsense.mapstate import build_is_current, load_configured_graph, write_and_rebuild
from roadsense.pipeline.runner import JobRunner
from roadsense.routing.graph import RoadGraph
from roadsense.storage import db
from roadsense.storage import jobs as jobs_db
from roadsense.storage.proclock import ProcessLock, lock_path_for

log = logging.getLogger(__name__)

# Multipart framing around the files, allowed on top of ``max_upload_mb``.
UPLOAD_OVERHEAD_BYTES = 1 << 20

# Locks whose worker thread outlived shutdown. The operating system drops a file lock
# when its handle is closed, and the handle closes when the ProcessLock object is
# garbage collected; a local variable would not survive the lifespan. Keeping a
# strong reference here holds the lock until the process exits.
_HELD_LOCKS: list[ProcessLock] = []

# How long shutdown waits for the job runner's worker thread.
DEFAULT_STOP_TIMEOUT_S = 10.0


def _default_detector_factory(settings: AppConfig) -> Callable[[], DetectorLike]:
    def factory() -> DetectorLike:
        from roadsense.detection.detector import Detector

        return Detector(settings.detection)

    return factory


def create_app(
    settings: AppConfig,
    *,
    detector_factory: Callable[[], DetectorLike] | None = None,
    graph: RoadGraph | None | Literal["auto"] = "auto",
    stop_timeout_s: float | None = DEFAULT_STOP_TIMEOUT_S,
) -> FastAPI:
    """Build the application.

    ``detector_factory`` builds the detector on the worker thread at the first
    job (default: the real ``Detector``). ``graph`` is ``"auto"`` to load
    ``routing.graph_path`` if that file exists, or a ready ``RoadGraph`` / None
    (used by tests). ``stop_timeout_s`` is how long shutdown waits for a running job.
    """
    factory = detector_factory or _default_detector_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        graph_error = None
        if graph == "auto":
            road_graph, graph_error = load_configured_graph(settings)
        else:
            road_graph = graph

        # One processing process per database. Holding this lock is what makes it
        # correct to fail left-over jobs below: no other server can own them.
        # A second server on the same database stops here with AlreadyRunningError
        # and touches nothing.
        process_lock = ProcessLock(lock_path_for(settings.db_path), role="server").acquire()
        stopped = True  # False only if the worker thread outlives runner.stop()
        try:
            conn = db.connect(settings.db_path)
            try:
                db.init_schema(conn)
                n_stale = jobs_db.fail_stale_running_jobs(conn)
                conn.commit()
                if n_stale:
                    log.warning("marked %d interrupted job(s) as failed", n_stale)
                if not build_is_current(conn, road_graph, settings):
                    conn.rollback()
                    write_and_rebuild(conn, settings, road_graph)
                    log.info("map rebuilt at start-up")
            finally:
                conn.close()

            state: AppState

            def on_done(conn: sqlite3.Connection, session_id: str, job_id: str) -> None:
                with state.map_lock:
                    write_and_rebuild(conn, settings, state.graph)

            runner = JobRunner(settings.db_path, settings, factory, on_done=on_done)
            state = AppState(settings, road_graph, graph_error, runner)
            app.state.rs = state
            runner.start()
            try:
                yield
            finally:
                stopped = runner.stop(stop_timeout_s)
        finally:
            if stopped:
                process_lock.release()
            else:
                _HELD_LOCKS.append(process_lock)  # outlive the lifespan; see _HELD_LOCKS
                log.warning(
                    "job runner did not stop in time; keeping the process lock until the "
                    "process exits so that no second server marks its job failed"
                )

    app = FastAPI(title="RoadSense", version=__version__, lifespan=lifespan)
    install_error_handlers(app)

    @app.middleware("http")
    async def revalidate_web_files(request: Request, call_next):
        """Make browsers revalidate the web interface's files on every load.

        Without this a browser may keep running an old copy of a script after
        the files change, which is confusing during a demonstration.
        """
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.middleware("http")
    async def reject_oversized_upload(request: Request, call_next):
        """Refuse a declared-too-large upload before its body is read."""
        if request.method == "POST" and request.url.path == "/api/sessions":
            declared = request.headers.get("content-length", "")
            limit = settings.uploads.max_upload_mb * (1 << 20) + UPLOAD_OVERHEAD_BYTES
            if declared.isdigit() and int(declared) > limit:
                return error_response(
                    413,
                    "too_large",
                    f"The upload is larger than the {settings.uploads.max_upload_mb} MB limit.",
                )
        return await call_next(request)

    for module in (system, sessions, mapdata, routes):
        app.include_router(module.router)

    @app.api_route(
        "/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False
    )
    def unknown_api_path(rest: str) -> None:
        raise ApiError(404, "not_found", f"No such API endpoint: /api/{rest}")

    # Mounted last so it does not shadow the /api routes.
    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    else:
        log.warning("web_dir %s does not exist; the web interface is not served", settings.web_dir)
    return app
