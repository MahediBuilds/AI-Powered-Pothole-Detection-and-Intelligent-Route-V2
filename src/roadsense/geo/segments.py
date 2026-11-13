"""Map build: from raw observations and GPS traces to potholes and road coverage.

``rebuild_map`` recomputes every derived row from the raw rows, in one
transaction (``builds``, ``potholes``, ``edge_coverage``, ``edge_stats`` and the
``observations.edge_*`` / ``pothole_id`` columns). Raw rows are never touched:
original observations are not deleted or merged, only linked to a pothole.

A road graph is optional. Without one, potholes are still clustered and have
coordinates, but no edge, and no coverage or edge statistics exist.

Recorded and synthetic (demonstration) sessions are kept apart all the way
through. Their sightings are clustered separately, so a pothole is either
``recorded`` or ``synthetic`` and never a blend. ``edge_stats`` (surveyed length,
pothole count and rate per road) describes real roads and is computed from
recorded sessions only. A synthetic session still gets ``edge_coverage`` rows
and potholes, which readers tell apart through ``sessions.gps_source`` and
``potholes.gps_source``.

Surveyed road is only what the camera filmed: a session's GPS trace is cut to
the video interval ``[0, duration]`` first (see ``clip_fixes_to_video``), so
fixes logged before the first frame or after the last one survey nothing.

Known limitations (all deliberate simplifications):
  * A pothole's position is the mean *camera* position of its observations; the
    pothole itself is a few metres ahead of the camera.
  * Snapping is nearest-edge (see ``roadsense.geo.mapmatch``), not HMM matching.
  * Coverage joins two consecutive fixes on different edges only when the edges
    share a node; fixes that skip an entire short edge leave it uncovered.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import roadsense
from roadsense.config import IngestConfig, MappingConfig
from roadsense.geo.dedup import cluster_points
from roadsense.geo.mapmatch import edge_index_for
from roadsense.routing.graph import RoadGraph

# Every snapped GPS fix surveys this far either side of itself.
FIX_HALF_WIDTH_M = 5.0

Edge = tuple[int, int, int]


def merge_intervals(intervals, tol: float = 1e-6) -> list[tuple[float, float]]:
    """Union of ``(start, end)`` intervals as sorted, non-overlapping intervals."""
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _valid_position(lat, lon) -> bool:
    return (
        lat is not None
        and lon is not None
        and math.isfinite(lat)
        and math.isfinite(lon)
        and abs(lat) <= 90
        and abs(lon) <= 180
    )


def counted_sessions(conn: sqlite3.Connection, include_synthetic: bool):
    """``(id, active_job_id, started_utc, gps_source, video_duration_s)`` of each session a build counts."""
    sql = (
        "SELECT s.id, s.active_job_id, s.started_utc, s.gps_source, s.duration_s, j.summary_json"
        " FROM sessions s JOIN jobs j ON j.id = s.active_job_id AND j.session_id = s.id "
        "WHERE s.include_in_map = 1 AND j.status = 'done'"
    )
    if not include_synthetic:
        sql += " AND s.gps_source != 'synthetic'"
    out = []
    for sid, job, started, src, duration, summary in conn.execute(sql + " ORDER BY s.id").fetchall():
        out.append((sid, job, started, src, _video_duration(duration, summary)))
    return out


def _video_duration(duration, summary_json) -> float | None:
    """Length of the video in seconds: the session row, else the job report, else None."""
    candidates = [duration]
    if summary_json:
        try:
            candidates.append(json.loads(summary_json).get("video", {}).get("duration_s"))
        except (ValueError, AttributeError):
            pass
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return None


def clip_fixes_to_video(fixes, duration_s: float | None, max_gap_s: float):
    """The part of a GPS trace recorded while the video was running.

    ``fixes`` are ``(t_video_s, lat, lon)`` sorted by time, on the video clock
    (0 is the first frame). Fixes before 0 or after ``duration_s`` are dropped.
    Where the trace crosses a boundary, the position exactly at the boundary is
    interpolated between the fix outside and the fix inside, unless those two are
    more than ``max_gap_s`` apart (a dropout is not a straight line, as in
    ``position_at``). With ``duration_s`` None the end of the video is unknown
    and only the start is clipped.
    """
    t_end = math.inf if duration_s is None else float(duration_s)
    out = []
    prev = None
    for fix in fixes:
        t = fix[0]
        if 0.0 <= t <= t_end:
            if prev is not None and prev[0] < 0.0 and t > 0.0 and t - prev[0] <= max_gap_s:
                out.append(_between(prev, fix, 0.0))
            out.append((t, fix[1], fix[2]))
        elif t > t_end:
            if prev is not None and 0.0 <= prev[0] < t_end and t - prev[0] <= max_gap_s:
                out.append(_between(prev, fix, t_end))
            elif prev is not None and prev[0] < 0.0 and t - prev[0] <= max_gap_s:
                # no fix inside the video at all: the span crosses both boundaries
                out.append(_between(prev, fix, 0.0))
                out.append(_between(prev, fix, t_end))
            # a fix after the end is skipped; later fixes may still lie inside the video
        prev = fix
    return out


def _between(a, b, t: float):
    frac = (t - a[0]) / (b[0] - a[0])
    return (t, a[1] + frac * (b[1] - a[1]), a[2] + frac * (b[2] - a[2]))


def _coverage_intervals(
    snaps, times, lengths: dict[Edge, float], max_gap_s: float
) -> dict[Edge, list[tuple[float, float]]]:
    """Surveyed intervals (canonical offsets) implied by one session's snapped fixes."""
    found: dict[Edge, list[tuple[float, float]]] = defaultdict(list)

    def add(edge: Edge, a: float, b: float) -> None:
        length = lengths[edge]
        a, b = max(0.0, a), min(length, b)
        if b > a:
            found[edge].append((a, b))

    for s in snaps:
        if s is not None:
            add(s.edge, s.offset_m - FIX_HALF_WIDTH_M, s.offset_m + FIX_HALF_WIDTH_M)

    for i in range(len(snaps) - 1):
        s1, s2 = snaps[i], snaps[i + 1]
        if s1 is None or s2 is None or times[i + 1] - times[i] > max_gap_s:
            continue
        if s1.edge == s2.edge:
            add(s1.edge, min(s1.offset_m, s2.offset_m), max(s1.offset_m, s2.offset_m))
            continue
        shared = {s1.edge_u, s1.edge_v} & {s2.edge_u, s2.edge_v}
        if not shared:
            continue

        def to_node(s, node):
            return s.offset_m if node == s.edge_u else lengths[s.edge] - s.offset_m

        node = min(sorted(shared), key=lambda n: to_node(s1, n) + to_node(s2, n))
        for s in (s1, s2):
            if node == s.edge_u:
                add(s.edge, 0.0, s.offset_m)
            else:
                add(s.edge, s.offset_m, lengths[s.edge])
    return found


def rebuild_map(
    conn: sqlite3.Connection,
    graph: RoadGraph | None,
    mapping: MappingConfig,
    ingest: IngestConfig,
    *,
    include_synthetic: bool = True,
) -> dict:
    """Rebuild all derived map data from the raw rows; returns a JSON-serialisable summary.

    Counted sessions have ``include_in_map = 1`` and an ``active_job_id`` whose
    job is ``done`` (and, if ``include_synthetic`` is False, a ``gps_source`` other
    than ``synthetic``). Synthetic sessions that are counted stay separate from
    recorded ones (see the module docstring). Everything runs in one savepoint:
    on any error nothing changes. Atomic with the caller's transaction when one is
    open; on a connection with no open transaction the outermost
    ``SAVEPOINT``/``RELEASE`` commits by itself. ``roadsense.mapstate.write_and_rebuild``
    is the function to use from a fresh connection.

    The database is not bound to the graph here; call
    ``roadsense.storage.db.bind_graph`` with ``graph.sha256`` first.
    """
    index = edge_index_for(graph) if graph is not None else None
    lengths = dict(zip(index.edges, index.lengths)) if index is not None else {}

    conn.execute("SAVEPOINT rebuild_map")
    try:
        summary = _rebuild(conn, graph, index, lengths, mapping, ingest, include_synthetic)
    except BaseException:
        conn.execute("ROLLBACK TO rebuild_map")
        conn.execute("RELEASE rebuild_map")
        raise
    conn.execute("RELEASE rebuild_map")
    return summary


def _rebuild(conn, graph, index, lengths, mapping, ingest, include_synthetic) -> dict:
    sessions = counted_sessions(conn, include_synthetic)
    by_id = {sid: (job, started, src) for sid, job, started, src, _ in sessions}
    durations = {sid: duration for sid, _, _, _, duration in sessions}

    def is_synthetic(sid: str) -> bool:
        return by_id[sid][2] == "synthetic"

    # ---- clear derived data (explicitly, so it does not depend on FK cascades)
    conn.execute(
        "UPDATE observations SET edge_u=NULL, edge_v=NULL, edge_key=NULL, edge_offset_m=NULL,"
        " pothole_id=NULL"
    )
    for table in ("edge_stats", "potholes", "edge_coverage", "builds"):
        conn.execute(f"DELETE FROM {table}")

    has_synthetic = any(src == "synthetic" for _, _, _, src, _ in sessions)
    params = {
        "mapping": mapping.model_dump(mode="json"),
        "max_gps_gap_s": float(ingest.max_gps_gap_s),
        "fix_half_width_m": FIX_HALF_WIDTH_M,
        "include_synthetic": include_synthetic,
        "session_ids": [s[0] for s in sessions],
        # which job of each session this build used; ``build_is_current`` compares it
        "jobs": {s[0]: s[1] for s in sessions},
    }
    build_id = conn.execute(
        "INSERT INTO builds (created_at, pipeline_version, graph_sha256, params_json)"
        " VALUES (?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            roadsense.__version__,
            None if graph is None else graph.sha256,
            json.dumps(params, sort_keys=True),
        ),
    ).lastrowid

    # ---- observations of the counted sessions
    obs = []  # (id, session_id, confidence, t_video_s, lat, lon)
    n_without_position = 0
    for sid, (job, _, _) in by_id.items():
        for oid, conf, t, lat, lon in conn.execute(
            "SELECT id, confidence, t_video_s, cam_lat, cam_lon FROM observations"
            " WHERE session_id = ? AND job_id = ? ORDER BY id",
            (sid, job),
        ).fetchall():
            if _valid_position(lat, lon):
                obs.append((oid, sid, conf, t, lat, lon))
            else:
                n_without_position += 1

    # ---- potholes: recorded and synthetic sightings never share a pothole
    members: dict[int, list[tuple]] = defaultdict(list)
    for synthetic in (False, True):
        group = [o for o in obs if is_synthetic(o[1]) == synthetic]
        labels = cluster_points([o[4] for o in group], [o[5] for o in group], mapping.dedup_radius_m)
        base = len(members)
        for o, label in zip(group, labels):
            members[base + label].append(o)

    pothole_rows = []  # dicts, in label order
    for label in sorted(members):
        ms = members[label]
        lat = sum(m[4] for m in ms) / len(ms)
        lon = sum(m[5] for m in ms) / len(ms)
        times = []
        for m in ms:
            start = _parse_utc(by_id[m[1]][1])
            if start is not None:
                times.append(start + timedelta(seconds=m[3]))
        edge = (None, None, None, None)
        if index is not None:
            snap = index.snap(lat, lon, mapping.max_snap_m)
            if snap is not None:
                edge = (snap.edge_u, snap.edge_v, snap.edge_key, snap.offset_m)
        pothole_rows.append(
            {
                "members": ms,
                "lat": lat,
                "lon": lon,
                "sessions": sorted({m[1] for m in ms}),
                "confidence": max(m[2] for m in ms),
                "gps_source": "synthetic" if is_synthetic(ms[0][1]) else "recorded",
                "first": min(times).isoformat() if times else None,
                "last": max(times).isoformat() if times else None,
                "edge": edge,
            }
        )

    for p in pothole_rows:
        p["id"] = conn.execute(
            "INSERT INTO potholes (build_id, edge_u, edge_v, edge_key, edge_offset_m, lat, lon,"
            " n_observations, n_sessions, confidence, gps_source, first_seen_utc, last_seen_utc)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                build_id, *p["edge"], p["lat"], p["lon"], len(p["members"]), len(p["sessions"]),
                p["confidence"], p["gps_source"], p["first"], p["last"],
            ),
        ).lastrowid  # fmt: skip
        conn.executemany(
            "UPDATE observations SET pothole_id=?, edge_u=?, edge_v=?, edge_key=?, edge_offset_m=?"
            " WHERE id=?",
            [(p["id"], *p["edge"], m[0]) for m in p["members"]],
        )

    # ---- coverage (graph only)
    per_session_edge: dict[tuple[str, Edge], list[tuple[float, float]]] = defaultdict(list)
    if index is not None:
        for sid, (job, _, _) in by_id.items():
            fixes = conn.execute(
                "SELECT t_video_s, lat, lon FROM gps_fixes WHERE session_id = ? AND used = 1"
                " ORDER BY seq",
                (sid,),
            ).fetchall()
            # only the road filmed by the video counts: no pre-roll or post-roll GPS
            fixes = clip_fixes_to_video(fixes, durations[sid], float(ingest.max_gps_gap_s))
            if not fixes:
                continue
            times = [f[0] for f in fixes]
            snaps = index.snap_many([f[1] for f in fixes], [f[2] for f in fixes], mapping.max_snap_m)
            for edge, ivs in _coverage_intervals(snaps, times, lengths, ingest.max_gps_gap_s).items():
                per_session_edge[(sid, edge)].extend(ivs)
        # an edge carrying a pothole is surveyed around it, by each session that saw it
        for p in pothole_rows:
            if p["edge"][0] is None:
                continue
            edge, off = p["edge"][:3], p["edge"][3]
            length = lengths[edge]
            for sid in p["sessions"]:
                a, b = max(0.0, off - FIX_HALF_WIDTH_M), min(length, off + FIX_HALF_WIDTH_M)
                if b > a:
                    per_session_edge[(sid, edge)].append((a, b))

    # per_edge_all describes real roads: recorded sessions only
    per_edge_all: dict[Edge, list[tuple[float, float]]] = defaultdict(list)
    per_edge_demo: dict[Edge, list[tuple[float, float]]] = defaultdict(list)
    cov_rows = []
    for (sid, edge), ivs in sorted(per_session_edge.items()):
        merged = merge_intervals(ivs)
        (per_edge_demo if is_synthetic(sid) else per_edge_all)[edge].extend(merged)
        cov_rows.extend((sid, by_id[sid][0], *edge, a, b) for a, b in merged)
    conn.executemany(
        "INSERT INTO edge_coverage (session_id, job_id, edge_u, edge_v, edge_key, start_m, end_m)"
        " VALUES (?,?,?,?,?,?,?)",
        cov_rows,
    )

    # ---- edge statistics (recorded data only; synthetic data never enters them)
    n_on_edge: dict[Edge, int] = defaultdict(int)
    for p in pothole_rows:
        if p["edge"][0] is not None and p["gps_source"] == "recorded":
            n_on_edge[p["edge"][:3]] += 1
    stat_rows = []
    total_surveyed = 0.0
    for edge in sorted(set(per_edge_all) | set(n_on_edge)):
        surveyed = sum(b - a for a, b in merge_intervals(per_edge_all.get(edge, [])))
        n = n_on_edge.get(edge, 0)
        total_surveyed += surveyed
        stat_rows.append(
            (*edge, build_id, lengths[edge], surveyed, n, n / surveyed * 100.0 if surveyed else 0.0)
        )
    conn.executemany(
        "INSERT INTO edge_stats (edge_u, edge_v, edge_key, build_id, length_m, surveyed_m,"
        " n_potholes, rate_per_100m) VALUES (?,?,?,?,?,?,?,?)",
        stat_rows,
    )

    n_on_road = sum(1 for p in pothole_rows if p["edge"][0] is not None)
    n_synthetic = sum(1 for p in pothole_rows if p["gps_source"] == "synthetic")
    demo_surveyed = sum(
        b - a for ivs in per_edge_demo.values() for a, b in merge_intervals(ivs)
    )
    return {
        "build_id": int(build_id),
        "n_sessions": len(sessions),
        "n_observations": len(obs),
        "n_observations_without_position": n_without_position,
        "n_potholes": len(pothole_rows),
        "n_potholes_recorded": len(pothole_rows) - n_synthetic,
        "n_potholes_synthetic": n_synthetic,
        "n_potholes_on_road": n_on_road,
        "n_potholes_off_road": len(pothole_rows) - n_on_road,
        "n_edges_with_coverage": len(per_edge_all),
        "surveyed_m": float(total_surveyed),  # recorded sessions only
        "demo_surveyed_m": float(demo_surveyed),  # synthetic traces; not a survey of real roads
        "graph_sha256": None if graph is None else graph.sha256,
        "has_synthetic": has_synthetic,
    }
