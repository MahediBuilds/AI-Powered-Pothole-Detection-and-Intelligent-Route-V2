"""Read helpers for the derived map data (the latest build).

Every function returns plain dicts and lists that are JSON-serialisable, and
works whatever ``row_factory`` the connection uses. Bounding boxes are
``(min_lat, min_lon, max_lat, max_lon)``.
"""

from __future__ import annotations

import sqlite3

from roadsense.geo.polyline import cut_polyline
from roadsense.geo.segments import merge_intervals
from roadsense.routing.graph import RoadGraph, edge_geometry

BBox = tuple[float, float, float, float]


class MapQueryError(Exception):
    """Stored map data cannot be read against the given road graph."""


def _dicts(cur: sqlite3.Cursor) -> list[dict]:
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def latest_build_id(conn: sqlite3.Connection) -> int | None:
    """Id of the most recent map build, or None if the map was never built."""
    row = conn.execute("SELECT MAX(id) FROM builds").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def list_potholes(conn: sqlite3.Connection, bbox: BBox | None = None) -> list[dict]:
    """Mapped potholes of the latest build, optionally inside ``bbox``, by id."""
    build_id = latest_build_id(conn)
    if build_id is None:
        return []
    sql = (
        "SELECT id, build_id, lat, lon, edge_u, edge_v, edge_key, edge_offset_m,"
        " n_observations, n_sessions, confidence, gps_source, first_seen_utc, last_seen_utc"
        " FROM potholes WHERE build_id = ?"
    )
    args: list = [build_id]
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        sql += " AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
        args += [min_lat, max_lat, min_lon, max_lon]
    return _dicts(conn.execute(sql + " ORDER BY id", args))


def get_pothole(conn: sqlite3.Connection, pothole_id: int) -> dict | None:
    """One pothole with its member observations (under ``observations``), or None."""
    cur = conn.execute(
        "SELECT id, build_id, lat, lon, edge_u, edge_v, edge_key, edge_offset_m,"
        " n_observations, n_sessions, confidence, gps_source, first_seen_utc, last_seen_utc"
        " FROM potholes WHERE id = ?",
        (pothole_id,),
    )
    rows = _dicts(cur)
    if not rows:
        return None
    pothole = rows[0]
    pothole["observations"] = _dicts(
        conn.execute(
            "SELECT o.id, o.session_id, s.name AS session_name, s.gps_source AS session_gps_source,"
            " o.job_id, o.track_id, o.n_frames, o.confidence, o.first_t_video_s, o.t_video_s,"
            " o.frame_index, o.frame_w, o.frame_h, o.x1, o.y1, o.x2, o.y2, o.pothole_id,"
            " o.image_path, o.cam_lat, o.cam_lon, o.cam_accuracy_m"
            " FROM observations o JOIN sessions s ON s.id = o.session_id"
            " WHERE o.pothole_id = ? ORDER BY o.id",
            (pothole_id,),
        )
    )
    return pothole


def list_coverage(
    conn: sqlite3.Connection, graph: RoadGraph | None, bbox: BBox | None = None
) -> list[dict]:
    """Covered road stretches (union over sessions of one kind), cut from the edge geometry.

    Each item is ``{"edge_u", "edge_v", "edge_key", "start_m", "end_m", "coords",
    "source"}`` where ``coords`` is a list of ``(lat, lon)`` along the road and
    ``source`` is ``recorded`` (surveyed by a real recording) or ``synthetic`` (a
    generated demonstration trace, which is not a survey). A stretch covered by
    both kinds is listed as ``recorded`` only. With ``bbox``,
    stretches with no point inside the box are left out. Raises ``MapQueryError``
    if coverage exists but the graph is missing or lacks a stored edge (a graph
    other than the one the map was built with).
    """
    rows = conn.execute(
        "SELECT c.edge_u, c.edge_v, c.edge_key, c.start_m, c.end_m, s.gps_source"
        " FROM edge_coverage c JOIN sessions s ON s.id = c.session_id"
        " ORDER BY c.edge_u, c.edge_v, c.edge_key, c.start_m"
    ).fetchall()
    if not rows:
        return []
    if graph is None:
        raise MapQueryError("coverage is stored but no road graph was given")
    per_edge: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for u, v, k, a, b, src in rows:
        kind = "synthetic" if src == "synthetic" else "recorded"
        per_edge.setdefault((u, v, k), {"recorded": [], "synthetic": []})[kind].append((a, b))
    out = []
    for (u, v, k), kinds in per_edge.items():
        try:
            geom, length = edge_geometry(graph, u, v, k)
        except KeyError:
            raise MapQueryError(
                f"edge ({u}, {v}, {k}) is not in the road graph; the map was built with a "
                "different graph"
            ) from None
        real = merge_intervals(kinds["recorded"])
        pieces = [(a, b, "recorded") for a, b in real]
        pieces += [(a, b, "synthetic") for a, b in subtract_intervals(kinds["synthetic"], real)]
        for a, b, source in pieces:
            coords = cut_polyline(geom, length, a, b)
            if bbox is not None and not _any_inside(coords, bbox):
                continue
            out.append(
                {
                    "edge_u": u,
                    "edge_v": v,
                    "edge_key": k,
                    "start_m": a,
                    "end_m": b,
                    "coords": [[lat, lon] for lat, lon in coords],
                    "source": source,
                }
            )
    return out


def subtract_intervals(intervals, minus, tol: float = 1e-6) -> list[tuple[float, float]]:
    """The parts of ``intervals`` (merged) not covered by the sorted, merged ``minus``."""
    out = []
    for a, b in merge_intervals(intervals):
        cur = a
        for x, y in minus:
            if y <= cur or x >= b:
                continue
            if x - cur > tol:
                out.append((cur, x))
            cur = max(cur, y)
        if b - cur > tol:
            out.append((cur, b))
    return out


def coverage_totals(conn: sqlite3.Connection) -> dict:
    """Road length covered, in metres, kept apart by kind (needs no graph).

    ``surveyed_m``/``edges`` count recorded sessions; ``demo_m``/``demo_edges``
    count road that only a synthetic trace runs along.
    """
    per_edge: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for u, v, k, a, b, src in conn.execute(
        "SELECT c.edge_u, c.edge_v, c.edge_key, c.start_m, c.end_m, s.gps_source"
        " FROM edge_coverage c JOIN sessions s ON s.id = c.session_id"
    ):
        kind = "synthetic" if src == "synthetic" else "recorded"
        per_edge.setdefault((u, v, k), {"recorded": [], "synthetic": []})[kind].append((a, b))
    surveyed = demo = 0.0
    edges = demo_edges = 0
    for kinds in per_edge.values():
        real = merge_intervals(kinds["recorded"])
        only_demo = subtract_intervals(kinds["synthetic"], real)
        surveyed += sum(b - a for a, b in real)
        demo += sum(b - a for a, b in only_demo)
        edges += bool(real)
        demo_edges += bool(only_demo)
    return {"surveyed_m": surveyed, "edges": edges, "demo_m": demo, "demo_edges": demo_edges}


def _any_inside(coords, bbox: BBox) -> bool:
    min_lat, min_lon, max_lat, max_lon = bbox
    return any(min_lat <= lat <= max_lat and min_lon <= lon <= max_lon for lat, lon in coords)


def map_stats(conn: sqlite3.Connection) -> dict:
    """Totals for a dashboard.

    ``sessions_by_job_status`` counts sessions by the status of their active job
    (``"no_active_job"`` when none), ``potholes_by_gps_source`` covers the latest
    build, and ``latest_build`` is None until a map has been built.
    """
    sessions: dict[str, int] = {}
    for status, n in conn.execute(
        "SELECT COALESCE(j.status, 'no_active_job'), COUNT(*) FROM sessions s"
        " LEFT JOIN jobs j ON j.id = s.active_job_id GROUP BY 1 ORDER BY 1"
    ):
        sessions[status] = n
    n_obs, n_obs_mapped = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(pothole_id IS NOT NULL), 0) FROM observations"
    ).fetchone()
    build_id = latest_build_id(conn)
    by_source: dict[str, int] = {}
    surveyed_m = 0.0
    latest_build = None
    if build_id is not None:
        for src, n in conn.execute(
            "SELECT gps_source, COUNT(*) FROM potholes WHERE build_id = ? GROUP BY 1 ORDER BY 1",
            (build_id,),
        ):
            by_source[src] = n
        surveyed_m = float(
            conn.execute(
                "SELECT COALESCE(SUM(surveyed_m), 0) FROM edge_stats WHERE build_id = ?",
                (build_id,),
            ).fetchone()[0]
        )
        latest_build = _dicts(
            conn.execute(
                "SELECT id, created_at, pipeline_version, graph_sha256 FROM builds WHERE id = ?",
                (build_id,),
            )
        )[0]
    return {
        "sessions_by_job_status": sessions,
        "n_sessions": sum(sessions.values()),
        "n_observations": int(n_obs),
        "n_observations_mapped": int(n_obs_mapped),
        "n_potholes": sum(by_source.values()),
        "potholes_by_gps_source": by_source,
        "surveyed_km": surveyed_m / 1000.0,
        "latest_build": latest_build,
    }
