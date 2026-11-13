"""Synthetic road graphs and plain-SQL database helpers for the geo/routing tests."""

from __future__ import annotations

import json

import networkx as nx

from roadsense.geo.projection import LocalProjection
from roadsense.ingest.gps import haversine_m
from roadsense.routing.graph import RoadGraph, edge_geometry, make_road_graph
from roadsense.geo.polyline import point_at

LAT0, LON0 = 51.5, -0.12
PROJ = LocalProjection(LAT0, LON0)


def xy(x: float, y: float) -> tuple[float, float]:
    """(lat, lon) of a point x metres east and y metres north of the grid origin."""
    return PROJ.to_latlon(x, y)


def _length(geom) -> float:
    return sum(haversine_m(*p, *q) for p, q in zip(geom, geom[1:]))


def make_grid(
    rows: int = 4,
    cols: int = 4,
    block_m: float = 100.0,
    oneway: tuple[tuple[int, int], ...] = ((5, 6),),
    curved: tuple[tuple[int, int], ...] = ((9, 10),),
    parallel: tuple[tuple[int, int], ...] = ((1, 2),),
) -> RoadGraph:
    """A rows x cols grid. Node id = r * cols + c at (x = c * block, y = r * block).

    Every neighbouring pair is a two-way street (key 0) with straight geometry,
    except:
      * ``oneway`` pairs ``(a, b)``: only a -> b exists;
      * ``curved`` pairs: the street bulges 30 m to the north (+y) through two
        intermediate points, so it is longer than the straight line;
      * ``parallel`` pairs: an extra two-way edge with key 1 that bulges 40 m to
        the south (-y) through one intermediate point.
    Edge geometry is stored from u to v for each direction.
    """
    g = nx.MultiDiGraph()
    pos = {}
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            pos[n] = (c * block_m, r * block_m)
            lat, lon = xy(*pos[n])
            g.add_node(n, lat=lat, lon=lon)

    def add(u, v, key, mid_pts, both=True, name=None):
        pts = [pos[u], *mid_pts, pos[v]]
        geom = [xy(*p) for p in pts]
        g.add_edge(u, v, key, length_m=_length(geom), geometry=geom, name=name,
                   highway="residential", speed_kph=30.0)  # fmt: skip
        if both:
            rgeom = list(reversed(geom))
            g.add_edge(v, u, key, length_m=_length(rgeom), geometry=rgeom, name=name,
                       highway="residential", speed_kph=30.0)  # fmt: skip

    pairs = []
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                pairs.append((n, n + 1))
            if r + 1 < rows:
                pairs.append((n, n + cols))
    ow = set(oneway)
    for u, v in pairs:
        mids = []
        if (u, v) in curved:
            (x0, y0), (x1, y1) = pos[u], pos[v]
            mids = [
                (x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0) + 30.0),
                (x0 + 0.75 * (x1 - x0), y0 + 0.75 * (y1 - y0) + 30.0),
            ]
        add(u, v, 0, mids, both=(u, v) not in ow, name=f"street {u}-{v}")
    for u, v in parallel:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        add(u, v, 1, [((x0 + x1) / 2, (y0 + y1) / 2 - 40.0)], name=f"alley {u}-{v}")
    return make_road_graph(g, {"source": "test grid"})


def edge_point(graph: RoadGraph, u: int, v: int, key: int, offset_m: float) -> tuple[float, float]:
    """(lat, lon) of the point ``offset_m`` along the canonical edge."""
    geom, length = edge_geometry(graph, u, v, key)
    return point_at(geom, length, offset_m)


def path_points(graph: RoadGraph, nodes: list[int], spacing_m: float = 10.0):
    """Points every ``spacing_m`` along the straightest key-0 route through ``nodes``."""
    pts = []
    for a, b in zip(nodes, nodes[1:]):
        geom, length = edge_geometry(graph, a, b, 0)
        if a > b:
            geom = list(reversed(geom))
        n = max(1, int(length // spacing_m))
        for i in range(n):
            off = i * spacing_m
            pts.append(point_at(geom, length, off))
    last = graph.g.nodes[nodes[-1]]
    pts.append((last["lat"], last["lon"]))
    return pts


# --------------------------------------------------------------------------- database helpers


def add_session(
    conn,
    sid: str,
    *,
    gps_source: str = "recorded",
    include_in_map: int = 1,
    started_utc: str | None = "2026-01-01T10:00:00+00:00",
    job_status: str | None = "done",
    active: bool = True,
) -> str:
    """Insert a session and (unless ``job_status`` is None) one job; returns the job id."""
    jid = f"job-{sid}"
    conn.execute(
        "INSERT INTO sessions (id, name, video_path, started_utc, gps_source, include_in_map,"
        " created_at) VALUES (?,?,?,?,?,?,?)",
        (sid, sid, f"{sid}.mp4", started_utc, gps_source, include_in_map, "t"),
    )
    if job_status is not None:
        conn.execute(
            "INSERT INTO jobs (id, session_id, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (jid, sid, job_status, "t", "t"),
        )
        if active:
            conn.execute("UPDATE sessions SET active_job_id=? WHERE id=?", (jid, sid))
    conn.commit()
    return jid


def add_fixes(conn, sid: str, points, *, dt: float = 1.0, used: int = 1, t0: float = 0.0) -> None:
    """Insert one GPS fix per point, ``dt`` seconds apart. ``dt`` may be a list per gap."""
    t = t0
    rows = []
    for i, (lat, lon) in enumerate(points):
        rows.append((sid, i, t, lat, lon, used))
        t += dt[i] if isinstance(dt, (list, tuple)) else dt
    conn.executemany(
        "INSERT INTO gps_fixes (session_id, seq, t_video_s, lat, lon, used) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def add_obs(
    conn,
    sid: str,
    lat: float | None,
    lon: float | None,
    *,
    track: int,
    conf: float = 0.9,
    t: float = 1.0,
    job_id: str | None = None,
) -> int:
    jid = job_id or f"job-{sid}"
    cur = conn.execute(
        "INSERT INTO observations (session_id, job_id, track_id, n_frames, confidence, t_video_s,"
        " frame_index, x1, y1, x2, y2, cam_lat, cam_lon) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, jid, track, 3, conf, t, 30, 0, 0, 10, 10, lat, lon),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_build(
    conn,
    graph: RoadGraph | None,
    *,
    stats: dict | None = None,
    potholes: list | None = None,
    coverage: list | None = None,
    session: str = "route-s",
) -> int:
    """Insert derived rows directly (no map build) for router tests.

    ``stats``: {(u, v, key): (n_potholes, surveyed_m)} (length from the graph).
    ``potholes``: [(edge (u, v, key), offset_m, gps_source)]; ids are 1..n.
    ``coverage``: [(edge, start_m, end_m)].
    """
    cur = conn.execute(
        "INSERT INTO builds (created_at, graph_sha256, params_json) VALUES ('t', ?, ?)",
        (None if graph is None else graph.sha256, json.dumps({})),
    )
    bid = cur.lastrowid
    for (u, v, k), (n, s) in (stats or {}).items():
        _, length = edge_geometry(graph, u, v, k)
        conn.execute(
            "INSERT INTO edge_stats (edge_u, edge_v, edge_key, build_id, length_m, surveyed_m,"
            " n_potholes, rate_per_100m) VALUES (?,?,?,?,?,?,?,?)",
            (u, v, k, bid, length, s, n, (n / s * 100 if s else 0.0)),
        )
    for i, ((u, v, k), off, src) in enumerate(potholes or [], start=1):
        lat, lon = edge_point(graph, u, v, k, off)
        conn.execute(
            "INSERT INTO potholes (id, build_id, edge_u, edge_v, edge_key, edge_offset_m, lat, lon,"
            " n_observations, n_sessions, confidence, gps_source) VALUES (?,?,?,?,?,?,?,?,1,1,0.8,?)",
            (i, bid, u, v, k, off, lat, lon, src),
        )
    if coverage:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, name, video_path, created_at)"
            " VALUES (?, 'n', 'v', 't')",
            (session,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, session_id, status, created_at, updated_at)"
            " VALUES (?, ?, 'done', 't', 't')",
            (f"job-{session}", session),
        )
        for (u, v, k), a, b in coverage:
            conn.execute(
                "INSERT INTO edge_coverage (session_id, job_id, edge_u, edge_v, edge_key, start_m,"
                " end_m) VALUES (?,?,?,?,?,?,?)",
                (session, f"job-{session}", u, v, k, a, b),
            )
    conn.commit()
    return int(bid)
