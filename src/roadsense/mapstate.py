"""Road graph loading and map rebuilding, shared by the API, the CLI and scripts.

Derived map data (potholes, coverage, road statistics) is rebuilt in full by
``rebuild_map`` and records the hash of the graph and the job of every session
it was built from. ``build_is_current`` compares that record with the database,
so a rebuild that was missed (for example because the database stayed locked)
is noticed and repeated at the next opportunity.

Demo mode. Synthetic (demonstration) sessions are left out of the map unless
demo mode is switched on; the choice is remembered in the ``meta`` table under
``include_synthetic`` so that automatic rebuilds repeat it. It is off until
someone turns it on. Even when on, synthetic data stays separate from recorded
data (see ``roadsense.geo.segments``).

Locking. ``write_and_rebuild`` is the way to rebuild from a fresh connection:
it takes SQLite's write lock up front (``BEGIN IMMEDIATE``, which waits for the
busy timeout), so the rebuild cannot fail halfway through on a lock, and it
retries a bounded number of times with a growing pause. A rebuild is one
transaction that replaces all derived rows, so a failed or repeated rebuild
never leaves partial or duplicated map data. ``rebuild_current`` is atomic with
the caller's transaction when one is open and commits by itself otherwise (see
its docstring).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any

import roadsense
from roadsense.config import AppConfig
from roadsense.geo.segments import counted_sessions, rebuild_map
from roadsense.routing.graph import GraphError, RoadGraph, load_graph
from roadsense.storage.map_queries import latest_build_id

log = logging.getLogger(__name__)

META_INCLUDE_SYNTHETIC = "include_synthetic"

# write_and_rebuild: attempts and the pause before the 2nd, 3rd, ... attempt
# (doubling). Each attempt already waits for the connection's busy timeout.
REBUILD_ATTEMPTS = 5
REBUILD_BASE_DELAY_S = 0.25


def load_configured_graph(cfg: AppConfig) -> tuple[RoadGraph | None, str | None]:
    """Load ``routing.graph_path`` if the file exists.

    Returns ``(graph, None)``, ``(None, None)`` when there is no file, or
    ``(None, reason)`` when the file is malformed (also logged).
    """
    path = cfg.routing.graph_path
    if not path.is_file():
        return None, None
    try:
        return load_graph(path), None
    except GraphError as exc:
        log.error("road graph ignored: %s", exc)
        return None, str(exc)


def get_include_synthetic(conn: sqlite3.Connection) -> bool:
    """Whether demo mode is on (default False: synthetic sessions stay out of the map)."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (META_INCLUDE_SYNTHETIC,)
    ).fetchone()
    return False if row is None else row[0] == "true"


def set_include_synthetic(conn: sqlite3.Connection, value: bool) -> None:
    """Remember the demo-mode choice. Does not commit."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (META_INCLUDE_SYNTHETIC, "true" if value else "false"),
    )


def rebuild_current(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    graph: RoadGraph | None,
    *,
    include_synthetic: bool | None = None,
) -> dict[str, Any]:
    """Rebuild the map; with ``include_synthetic`` given, remember that choice first.

    Atomic with the caller's transaction when one is open (nothing is committed
    here). On a connection with no open transaction the outermost
    ``SAVEPOINT``/``RELEASE`` inside ``rebuild_map`` commits by itself, so the
    rebuild is then self-committing. ``write_and_rebuild`` is the function to use
    from a fresh connection.
    """
    if include_synthetic is None:
        include_synthetic = get_include_synthetic(conn)
    else:
        set_include_synthetic(conn, include_synthetic)
    return rebuild_map(
        conn, graph, cfg.mapping, cfg.ingest, include_synthetic=include_synthetic
    )


def is_locked_error(exc: BaseException) -> bool:
    """True for SQLite's "database is locked / busy" errors (worth retrying)."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def write_and_rebuild(
    conn: sqlite3.Connection,
    cfg: AppConfig,
    graph: RoadGraph | None,
    *,
    include_synthetic: bool | None = None,
    before: Callable[[sqlite3.Connection], Any] | None = None,
    attempts: int = REBUILD_ATTEMPTS,
    base_delay_s: float = REBUILD_BASE_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Rebuild the map in one committed write transaction, retrying while the database is locked.

    ``before(conn)`` runs first inside the same transaction (for example the
    change that makes the rebuild necessary), so either both happen or neither.
    The connection must not be inside a transaction. A lock error is retried up
    to ``attempts`` times in total, pausing ``base_delay_s``, then twice that, and
    so on; the last one is raised. Any other error is raised at once. Whatever
    happens, nothing is left half done: every failed attempt is rolled back.
    """
    if conn.in_transaction:
        raise RuntimeError("write_and_rebuild needs a connection with no open transaction")
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts!r}")
    for attempt in range(1, attempts + 1):
        try:
            conn.execute("BEGIN IMMEDIATE")
            if before is not None:
                before(conn)
            summary = rebuild_current(conn, cfg, graph, include_synthetic=include_synthetic)
            conn.commit()
            return summary
        except BaseException as exc:
            conn.rollback()
            if not is_locked_error(exc) or attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning(
                "map rebuild: database is locked (attempt %d of %d); retrying in %.2f s",
                attempt, attempts, delay,
            )  # fmt: skip
            sleep(delay)
    raise AssertionError("unreachable")


def build_is_current(conn: sqlite3.Connection, graph: RoadGraph | None, cfg: AppConfig) -> bool:
    """Whether the latest build used this graph, these settings, this code version and every session's current job."""
    build_id = latest_build_id(conn)
    if build_id is None:
        return False
    sha, params_json, version = conn.execute(
        "SELECT graph_sha256, params_json, pipeline_version FROM builds WHERE id = ?", (build_id,)
    ).fetchone()
    if sha != (None if graph is None else graph.sha256):
        return False
    if version != roadsense.__version__:
        return False
    try:
        params = json.loads(params_json)
    except ValueError:
        return False
    if not isinstance(params, dict):
        return False
    if params.get("mapping") != cfg.mapping.model_dump(mode="json"):
        return False
    if params.get("max_gps_gap_s") != float(cfg.ingest.max_gps_gap_s):
        return False
    include = get_include_synthetic(conn)
    if bool(params.get("include_synthetic", True)) != include:
        return False
    wanted = {s[0]: s[1] for s in counted_sessions(conn, include)}
    return params.get("jobs") == wanted


def ensure_map_current(
    conn: sqlite3.Connection, cfg: AppConfig, graph: RoadGraph | None
) -> bool:
    """Rebuild the map unless the latest build is current; True if rebuilt.

    Atomic with the caller's transaction when one is open; on a connection with
    no open transaction the rebuild commits by itself (see ``rebuild_current``).
    Use ``write_and_rebuild`` from a fresh connection, especially when another
    connection may be writing.
    """
    if build_is_current(conn, graph, cfg):
        return False
    rebuild_current(conn, cfg, graph)
    return True
