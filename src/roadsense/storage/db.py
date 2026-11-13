"""SQLite connection helper and schema initialisation."""

from __future__ import annotations

import os
import sqlite3
from importlib import resources
from pathlib import Path

# Bump when schema.sql changes incompatibly. There is no migration framework:
# a database with a different version is refused.
SCHEMA_VERSION = 2


class SchemaVersionError(Exception):
    """The database was created by a different schema version."""


class GraphMismatchError(Exception):
    """The database is bound to a different road graph file."""


def connect(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open the database (creating its parent directory) with the project's pragmas."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _stored_version(conn: sqlite3.Connection) -> str | None:
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if not has_meta:
        return None
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return None if row is None else row["value"]


def _check_version(stored: str) -> None:
    if stored != str(SCHEMA_VERSION):
        raise SchemaVersionError(
            f"database schema_version is {stored!r} but this code expects "
            f"{SCHEMA_VERSION}; there is no migration, recreate the database"
        )


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the schema if missing; refuse a database with a different version."""
    stored = _stored_version(conn)
    if stored is not None:
        _check_version(stored)
    sql = resources.files("roadsense.storage").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    _check_version(_stored_version(conn))


def bind_graph(conn: sqlite3.Connection, graph_sha256: str) -> None:
    """Bind the database to one road graph file by its hash.

    The first call stores the hash in ``meta``; later calls must present the same
    hash, because stored edge ids are only meaningful for the graph they were
    computed on.
    """
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('graph_sha256', ?)",
        (graph_sha256,),
    )
    conn.commit()
    stored = conn.execute("SELECT value FROM meta WHERE key='graph_sha256'").fetchone()["value"]
    if stored != graph_sha256:
        raise GraphMismatchError(
            f"database is bound to graph {stored} but graph {graph_sha256} was given; "
            "edge ids are not comparable across graphs, use a new database"
        )
