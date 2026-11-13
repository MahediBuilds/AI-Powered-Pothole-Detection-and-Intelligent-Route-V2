"""Shared application state and per-request database access."""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Request

from roadsense.config import AppConfig, project_root
from roadsense.pipeline.runner import JobRunner
from roadsense.routing.graph import ATTRIBUTION, RoadGraph
from roadsense.storage import db

ACTIVE_JOB_STATUSES = ("queued", "running")


@dataclass
class AppState:
    """Everything the endpoints share; created once by the lifespan handler."""

    cfg: AppConfig
    graph: RoadGraph | None
    graph_error: str | None
    runner: JobRunner
    # Map rebuilds write the whole derived tables: one at a time.
    map_lock: threading.Lock = field(default_factory=threading.Lock)
    # Makes "is a job active? then queue one / delete" atomic.
    job_lock: threading.Lock = field(default_factory=threading.Lock)
    _weights_cache: tuple[tuple[int, int], str] | None = None
    _graph_info: dict[str, Any] | None = None

    def graph_info(self) -> dict[str, Any]:
        """The ``graph`` object of ``/api/system`` (cached; the graph never changes)."""
        if self._graph_info is None:
            path = display_path(self.cfg.routing.graph_path)
            g = self.graph
            if g is None:
                info: dict[str, Any] = {"loaded": False, "path": path, "error": self.graph_error}
            else:
                info = {
                    "loaded": True,
                    "path": path,
                    "sha256": g.sha256,
                    "bbox": list(g.bbox),
                    "n_nodes": g.g.number_of_nodes(),
                    "n_edges": g.g.number_of_edges(),
                    "attribution": ATTRIBUTION,
                    "error": None,
                }
            self._graph_info = info
        return self._graph_info

    def weights_sha256(self) -> str | None:
        """SHA-256 of the weights file, hashed once per file version; None if missing."""
        import hashlib

        path = self.cfg.detection.weights
        try:
            st = path.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        if self._weights_cache is None or self._weights_cache[0] != key:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._weights_cache = (key, digest.hexdigest())
        return self._weights_cache[1]


def display_path(path: Path) -> str:
    """The path relative to the project root when inside it, else as given."""
    try:
        return Path(path).relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def get_state(request: Request) -> AppState:
    return request.app.state.rs


@contextmanager
def db_conn(state: AppState) -> Iterator[sqlite3.Connection]:
    """A connection for the calling thread only, closed afterwards.

    Used inside the endpoint function itself (never as a dependency): FastAPI
    would otherwise open and close it on different worker threads.
    """
    conn = db.connect(state.cfg.db_path)
    try:
        yield conn
    finally:
        conn.close()
