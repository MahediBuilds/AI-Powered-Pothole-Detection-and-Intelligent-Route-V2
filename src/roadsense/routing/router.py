"""Route comparison that trades distance against estimated pothole exposure.

``plan_routes`` finds up to three distinct routes between two points and
describes each with the numbers behind it. Cost model (per directed edge)::

    estimated potholes  E = n + rate * (L - s) / 100
    edge cost             = L + avoidance * max_detour_per_pothole_m * E

``L`` is the edge length in metres, ``n`` the mapped potholes on it, ``s`` the
surveyed length and ``rate`` the potholes per 100 m assumed for road with no
survey evidence.

The assumed rate. Road that nobody has surveyed is unknown, so it is never
costed as if it had no potholes. Unless ``routing.unsurveyed_rate_per_100m``
fixes the rate, it is the observed rate shrunk towards a stated assumption::

    rate = (N * 100 + assumed_rate_per_100m * prior_strength_m) / (S + prior_strength_m)

``N`` and ``S`` are the mapped on-road potholes and the surveyed metres over the
whole map. With nothing surveyed the rate is ``assumed_rate_per_100m``; a small
survey moves it only a little (a single pothole on a few metres of road cannot
produce an extreme rate); once ``S`` is several times ``prior_strength_m`` the
observed rate dominates.

Evidence. Each route reports how much of it is surveyed and an ``evidence``
level: ``adequate`` (at least ``adequate_surveyed_fraction`` surveyed),
``partial``, or ``insufficient`` (below ``min_surveyed_fraction``). Estimates on
an ``insufficient`` route come almost entirely from the assumed rate, and the
explanation says so.

Real and demonstration data. Recorded sessions describe real roads. Synthetic
(demonstration) sessions are only used when the map was built with them
included; the plan then has ``demo_mode`` set, whatever road the routes use, and
every route says how much of it relies on demonstration data. With demo mode
off, synthetic data has no influence on any number here.

Candidates use avoidance 0 ("Shortest"), the requested value ("Your setting")
and 1 ("Strongest pothole avoidance"); candidates that follow the same edges are
merged. The labels name the setting that produced a route, not an outcome: the
strongest-avoidance route minimises the estimate above, which can differ from
the number of potholes actually on the road. Simplifications: a route starts
and ends at the nearest graph *node* to each point (not at the exact position
on a road), and durations come from speed limits or assumed speeds.

Wording rule: nothing generated here describes a road or route as free of
potholes, safe or clear, and no severity is mentioned. Counts are always
phrased as "mapped" potholes, which is what is known.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import networkx as nx

from roadsense.config import MappingConfig, RoutingConfig
from roadsense.geo.mapmatch import NodeIndex
from roadsense.geo.polyline import cut_polyline
from roadsense.geo.segments import merge_intervals
from roadsense.ingest.gps import haversine_m
from roadsense.routing.graph import RoadGraph, canonical_edge, edge_geometry
from roadsense.storage.db import GraphMismatchError
from roadsense.storage.map_queries import latest_build_id

MAX_NODE_SNAP_M = 250.0

CAVEATS = [
    "Recommendations reflect only the potholes detected and mapped so far; potholes that "
    "have not been detected or have not been surveyed are not included.",
    "Road that has not been surveyed is unknown: it may have potholes nobody has recorded. "
    "Its estimated potholes come from an assumed rate, not from observation.",
    "Potholes are detected as a single class; no severity or size is recorded.",
    "Travel times are estimates from posted or assumed speed limits.",
    "A route's estimate can be lower than another's while it passes more mapped potholes, "
    "because unsurveyed road also counts through the assumed rate.",
]
SYNTHETIC_CAVEAT = (
    "DEMONSTRATION MODE: synthetic (generated) sessions are included in these results. "
    "Potholes and surveyed road from them are for demonstration only and say nothing about "
    "real road conditions."
)

LABEL_SHORTEST = "Shortest"
LABEL_REQUESTED = "Your setting"
LABEL_STRONGEST = "Strongest pothole avoidance"

EVIDENCE_ADEQUATE = "adequate"
EVIDENCE_PARTIAL = "partial"
EVIDENCE_INSUFFICIENT = "insufficient"

# Section status along a route, strongest evidence first.
SURVEYED, DEMO, UNSURVEYED = "surveyed", "demo", "unsurveyed"


class RoutingError(Exception):
    """Base class for routing failures."""


class NoGraphError(RoutingError):
    """No road graph is loaded, so routes cannot be computed."""


class OutsideGraphError(RoutingError):
    """A requested point is too far from the road network."""


class NoRouteError(RoutingError):
    """The two points are not connected in the road graph."""


def _check_point(name: str, point: Any) -> tuple[float, float]:
    try:
        lat, lon = point
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (lat, lon) pair, got {point!r}") from None
    for v in (lat, lon):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must contain finite numbers, got {point!r}")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{name} {point!r} is outside valid latitude/longitude ranges")
    return float(lat), float(lon)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _fmt_dist(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.1f} km"


def _covered(ivs: list[tuple[float, float]], length: float) -> float:
    return sum(max(0.0, min(b, length) - max(a, 0.0)) for a, b in ivs)


class _Map:
    """Evidence of the latest build, keyed by canonical edge.

    ``real_*`` comes from recorded sessions. ``demo_*`` comes from synthetic
    sessions and is empty unless the build included them; ``demo_mode`` says
    whether any exists.
    """

    def __init__(self) -> None:
        self.real_stats: dict[tuple[int, int, int], tuple[int, float]] = {}
        self.real_cov: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        self.demo_cov: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        self.potholes: dict[tuple[int, int, int], list[dict]] = {}
        self.demo_mode = False

    def n_demo(self, edge) -> int:
        return sum(1 for p in self.potholes.get(edge, []) if p["gps_source"] != "recorded")

    def evidence(self, edge, length: float) -> tuple[int, float]:
        """``(mapped potholes, surveyed metres)`` used to cost the edge."""
        n, s = self.real_stats.get(edge, (0, 0.0))
        s = min(s, length)
        if not self.demo_mode:
            return n, s
        demo = self.demo_cov.get(edge)
        if demo:
            union = merge_intervals([*self.real_cov.get(edge, []), *demo])
            s = max(s, min(_covered(union, length), length))
        return n + self.n_demo(edge), s

    def totals(self, lengths) -> tuple[int, float]:
        """Mapped on-road potholes and surveyed metres over the whole map."""
        edges = set(self.real_stats) | (set(self.demo_cov) | set(self.potholes) if self.demo_mode else set())
        n_total, s_total = 0, 0.0
        for edge in edges:
            n, s = self.evidence(edge, lengths(edge))
            n_total += n
            s_total += s
        return n_total, s_total


def _load_map(conn: sqlite3.Connection, graph: RoadGraph, build_id: int | None) -> _Map:
    m = _Map()
    if build_id is None:
        return m
    sha = conn.execute("SELECT graph_sha256 FROM builds WHERE id = ?", (build_id,)).fetchone()[0]
    if sha is not None and sha != graph.sha256:
        raise GraphMismatchError(
            f"the map was built with graph {sha} but graph {graph.sha256} was given; "
            "rebuild the map or load the matching graph"
        )
    for u, v, k, s, n in conn.execute(
        "SELECT edge_u, edge_v, edge_key, surveyed_m, n_potholes FROM edge_stats WHERE build_id = ?",
        (build_id,),
    ):
        m.real_stats[(u, v, k)] = (int(n), float(s))
    for u, v, k, a, b, src in conn.execute(
        "SELECT c.edge_u, c.edge_v, c.edge_key, c.start_m, c.end_m, s.gps_source"
        " FROM edge_coverage c JOIN sessions s ON s.id = c.session_id"
    ):
        target = m.demo_cov if src == "synthetic" else m.real_cov
        target.setdefault((u, v, k), []).append((a, b))
    m.real_cov = {e: merge_intervals(ivs) for e, ivs in m.real_cov.items()}
    m.demo_cov = {e: merge_intervals(ivs) for e, ivs in m.demo_cov.items()}
    for pid, lat, lon, conf, src, u, v, k, off in conn.execute(
        "SELECT id, lat, lon, confidence, gps_source, edge_u, edge_v, edge_key, edge_offset_m"
        " FROM potholes WHERE build_id = ? AND edge_u IS NOT NULL ORDER BY id",
        (build_id,),
    ):
        m.potholes.setdefault((u, v, k), []).append(
            {"id": pid, "lat": lat, "lon": lon, "confidence": conf, "gps_source": src,
             "_offset": off}  # fmt: skip
        )
    # any synthetic pothole of the build counts, also one that is not on a road
    m.demo_mode = bool(m.demo_cov) or (
        conn.execute(
            "SELECT 1 FROM potholes WHERE build_id = ? AND gps_source != 'recorded' LIMIT 1",
            (build_id,),
        ).fetchone()
        is not None
    )
    return m


def _pieces(
    length: float, real: list[tuple[float, float]], demo: list[tuple[float, float]]
) -> list[tuple[float, float, str]]:
    """Split ``[0, length]`` into consecutive pieces of one status each.

    A stretch covered by a recorded session is ``surveyed``; one covered only by
    a synthetic session is ``demo``; the rest is ``unsurveyed``.
    """
    if length <= 0:
        return [(0.0, 0.0, UNSURVEYED)]
    cuts = {0.0, length}
    for a, b in (*real, *demo):
        cuts.update(x for x in (a, b) if 0.0 < x < length)
    points = sorted(cuts)

    def inside(ivs, x: float) -> bool:
        return any(a <= x <= b for a, b in ivs)

    out: list[tuple[float, float, str]] = []
    for lo, hi in zip(points, points[1:]):
        mid = (lo + hi) / 2
        status = SURVEYED if inside(real, mid) else DEMO if inside(demo, mid) else UNSURVEYED
        if out and out[-1][2] == status:
            out[-1] = (out[-1][0], hi, status)
        else:
            out.append((lo, hi, status))
    return out


def assumed_rate(
    n_total: int, surveyed_m: float, routing: RoutingConfig, *, demo: bool = False
) -> tuple[float, str]:
    """Potholes per 100 m assumed for unsurveyed road, and where the number comes from.

    With ``demo`` the totals include synthetic potholes and road, and the source
    text says so.
    """
    if routing.unsurveyed_rate_per_100m is not None:
        rate = float(routing.unsurveyed_rate_per_100m)
        source = "configured (routing.unsurveyed_rate_per_100m)"
        if rate == 0:
            source += "; at 0, unsurveyed road adds nothing to the estimate although it is unknown"
        return rate, source
    base = float(routing.assumed_rate_per_100m)
    strength = float(routing.prior_strength_m)
    rate = (n_total * 100.0 + base * strength) / (surveyed_m + strength)
    if surveyed_m <= 0:
        source = (
            f"assumption only ({base:g} per 100 m, routing.assumed_rate_per_100m): no road has "
            "been surveyed yet"
        )
    else:
        what = (
            "rate over recorded AND synthetic demonstration data"
            if demo
            else "observed rate on surveyed road"
        )
        source = (
            f"{what} ({_plural(n_total, 'mapped pothole')} over "
            f"{surveyed_m:.0f} m surveyed) blended with an assumed {base:g} per 100 m that "
            f"weighs as much as {strength:.0f} m of survey"
        )
    return rate, source


def evidence_level(surveyed_fraction: float, routing: RoutingConfig) -> str:
    if surveyed_fraction >= float(routing.adequate_surveyed_fraction):
        return EVIDENCE_ADEQUATE
    if surveyed_fraction >= float(routing.min_surveyed_fraction):
        return EVIDENCE_PARTIAL
    return EVIDENCE_INSUFFICIENT


def plan_routes(
    conn: sqlite3.Connection,
    graph: RoadGraph | None,
    origin: tuple[float, float],
    destination: tuple[float, float],
    avoidance: float,
    routing: RoutingConfig,
    mapping: MappingConfig,
) -> dict:
    """Compute 1 to 3 distinct routes and explain them; the result is JSON-serialisable.

    ``origin`` and ``destination`` are ``(lat, lon)``; ``avoidance`` is in [0, 1]
    (0 = shortest, 1 = avoid estimated potholes as strongly as configured).
    ``mapping`` is accepted for symmetry with the map build and is not used.

    Raises ``NoGraphError`` (graph is None), ``ValueError`` (bad input),
    ``OutsideGraphError`` (a point is over 250 m from any road node),
    ``NoRouteError`` (not connected) and ``GraphMismatchError`` (the stored map
    was built with a different graph).
    """
    if graph is None:
        raise NoGraphError("no road graph is loaded; routing is unavailable")
    o_lat, o_lon = _check_point("origin", origin)
    d_lat, d_lon = _check_point("destination", destination)
    if isinstance(avoidance, bool) or not isinstance(avoidance, (int, float)):
        raise ValueError(f"avoidance must be a number in [0, 1], got {avoidance!r}")
    if not math.isfinite(avoidance) or not 0.0 <= avoidance <= 1.0:
        raise ValueError(f"avoidance must be between 0 and 1, got {avoidance!r}")
    avoidance = float(avoidance)

    nodes = NodeIndex(graph)
    ends = {}
    for name, (lat, lon) in (("origin", (o_lat, o_lon)), ("destination", (d_lat, d_lon))):
        node = nodes.nearest_node(lat, lon, MAX_NODE_SNAP_M)
        if node is None:
            raise OutsideGraphError(
                f"{name} ({lat:.5f}, {lon:.5f}) is more than {MAX_NODE_SNAP_M:.0f} m from any road node"
            )
        nd = graph.g.nodes[node]
        ends[name] = {
            "requested": [lat, lon],
            "snapped": [nd["lat"], nd["lon"]],
            "node": int(node),
            "snap_distance_m": round(haversine_m(lat, lon, nd["lat"], nd["lon"]), 1),
        }

    build_id = latest_build_id(conn)
    m = _load_map(conn, graph, build_id)

    def canon_length(edge) -> float:
        try:
            return edge_geometry(graph, *edge)[1]
        except KeyError:
            return 0.0

    total_n, total_s = m.totals(canon_length)
    rate, rate_source = assumed_rate(total_n, total_s, routing, demo=m.demo_mode)
    max_detour = float(routing.max_detour_per_pothole_m)

    g = graph.g

    def expected(canon, length: float) -> float:
        n, s = m.evidence(canon, length)
        return n + rate * (length - s) / 100.0

    def edge_cost(av: float, u: int, v: int, k: int, d: dict) -> float:
        length = float(d["length_m"])
        return length + av * max_detour * expected(canonical_edge(u, v, k), length)

    def solve(av: float) -> list[tuple[int, int, int]]:
        def weight(u, v, parallel):
            return min(edge_cost(av, u, v, k, d) for k, d in parallel.items())

        try:
            node_path = nx.shortest_path(g, ends["origin"]["node"], ends["destination"]["node"],
                                         weight=weight)  # fmt: skip
        except nx.NetworkXNoPath:
            raise NoRouteError("there is no route between the origin and destination") from None
        edges = []
        for a, b in zip(node_path, node_path[1:]):
            k = min(g[a][b], key=lambda key: (edge_cost(av, a, b, key, g[a][b][key]), key))
            edges.append((a, b, k))
        return edges

    candidates = [(0.0, LABEL_SHORTEST), (avoidance, LABEL_REQUESTED), (1.0, LABEL_STRONGEST)]
    merged: list[dict] = []
    for av, label in candidates:
        edges = solve(av)
        for entry in merged:
            if entry["edges"] == edges:
                entry["labels"].append(label)
                break
        else:
            merged.append({"edges": edges, "labels": [label]})

    routes = [
        _describe(graph, e["edges"], e["labels"], m, rate, routing, ends["origin"]["snapped"])
        for e in merged
    ]
    shortest = next(r for r in routes if LABEL_SHORTEST in r["labels"])
    for r in routes:
        r["recommended"] = LABEL_REQUESTED in r["labels"]
        r["explanation"] = _explain(r, shortest)

    requested = next(r for r in routes if r["recommended"])
    notes = []
    if requested["distance_m"] > 0 and requested["evidence"] == EVIDENCE_INSUFFICIENT:
        notes.append(
            f"Only {requested['surveyed_fraction'] * 100:.0f}% of the route for your setting has "
            "been surveyed. Its pothole estimate rests mostly on the assumed rate, so treat the "
            "comparison between these routes as a rough guide."
        )
    if routing.unsurveyed_rate_per_100m is None and total_s < float(routing.prior_strength_m):
        notes.append(
            f"Only {total_s:.0f} m of road has been surveyed in total"
            f"{' (including synthetic demonstration road)' if m.demo_mode else ''}, "
            "so the rate assumed for "
            "unsurveyed road is mostly an assumption."
        )
    return {
        "origin": ends["origin"],
        "destination": ends["destination"],
        "avoidance": avoidance,
        "build_id": build_id,
        "cost_model": {
            "description": (
                "Each road segment costs its length in metres plus avoidance x "
                "max_detour_per_pothole_m x estimated potholes on it. Estimated potholes are the "
                "mapped potholes on the segment plus prior_rate_per_100m x its unsurveyed "
                "length / 100 m."
            ),
            "max_detour_per_pothole_m": max_detour,
            "prior_rate_per_100m": rate,
            "prior_source": rate_source,
            "surveyed_total_m": round(total_s, 1),
            "mapped_potholes_total": total_n,
            "includes_demo_data": m.demo_mode,
        },
        "routes": routes,
        "confidence_notes": notes,
        # Demonstration data is in play whenever the map was built with it, whichever
        # roads these particular routes use.
        "demo_mode": m.demo_mode,
        "uses_synthetic_data": m.demo_mode,
        "caveats": [SYNTHETIC_CAVEAT, *CAVEATS] if m.demo_mode else list(CAVEATS),
    }


def _describe(graph, edges, labels, m: _Map, rate, routing, start_point) -> dict:
    """Metrics, geometry and surveyed/demo/unsurveyed sections for one edge sequence."""
    g = graph.g
    distance = duration = unsurveyed_expected = canon_total = 0.0
    by_status = {SURVEYED: 0.0, DEMO: 0.0, UNSURVEYED: 0.0}
    geometry: list[list[float]] = []
    sections: list[dict] = []
    route_potholes: list[dict] = []

    def append_point(coords: list[list[float]], pt: list[float]) -> None:
        if not coords or coords[-1] != pt:
            coords.append(pt)

    for a, b, k in edges:
        d = g.edges[a, b, k]
        canon = canonical_edge(a, b, k)
        geom, length = edge_geometry(graph, a, b, k)
        reverse = a > b
        if reverse:
            geom = list(reversed(geom))
        distance += float(d["length_m"])
        duration += float(d["length_m"]) / (float(d["speed_kph"]) / 3.6)
        _, s_used = m.evidence(canon, length)
        canon_total += length
        unsurveyed_expected += rate * (length - s_used) / 100.0

        for lat, lon in geom:
            append_point(geometry, [lat, lon])

        real = m.real_cov.get(canon, [])
        demo = m.demo_cov.get(canon, []) if m.demo_mode else []
        if reverse:
            real = sorted((length - y, length - x) for x, y in real)
            demo = sorted((length - y, length - x) for x, y in demo)
        pieces = _pieces(length, real, demo)
        # edge_stats may know a surveyed length without stored intervals (older builds)
        s_real = min(m.real_stats.get(canon, (0, 0.0))[1], length)
        by_status[SURVEYED] += max(s_real, sum(hi - lo for lo, hi, st in pieces if st == SURVEYED))
        by_status[DEMO] += sum(hi - lo for lo, hi, st in pieces if st == DEMO)
        for lo, hi, status in pieces:
            coords = [[lat, lon] for lat, lon in cut_polyline(geom, length, lo, hi)]
            if sections and sections[-1]["status"] == status:
                for pt in coords:
                    append_point(sections[-1]["coords"], pt)
            else:
                sections.append({"surveyed": status == SURVEYED, "status": status, "coords": coords})

        here = sorted(m.potholes.get(canon, []), key=lambda p: p["_offset"], reverse=reverse)
        if not m.demo_mode:
            here = [p for p in here if p["gps_source"] == "recorded"]
        route_potholes.extend({k2: v for k2, v in p.items() if k2 != "_offset"} for p in here)

    if not edges:  # origin and destination are the same node
        geometry = [list(start_point)]
    surveyed = min(by_status[SURVEYED], canon_total)
    demo_m = min(by_status[DEMO], max(0.0, canon_total - surveyed))
    fraction = surveyed / canon_total if canon_total > 0 else 0.0
    n_obs = len(route_potholes)
    n_demo = sum(1 for p in route_potholes if p["gps_source"] != "recorded")
    return {
        "labels": list(labels),
        "recommended": False,
        "distance_m": round(distance, 1),
        "duration_s": round(duration, 1),
        "n_potholes_observed": n_obs,
        "n_potholes_demo": n_demo,
        "potholes": route_potholes,
        "surveyed_m": round(surveyed, 1),
        "surveyed_fraction": round(fraction, 4),
        "demo_m": round(demo_m, 1),
        "unsurveyed_m": round(max(0.0, canon_total - surveyed - demo_m), 1),
        "evidence": evidence_level(fraction, routing),
        "expected_potholes_unsurveyed": round(unsurveyed_expected, 3),
        "expected_potholes_total": round(n_obs + unsurveyed_expected, 3),
        "geometry": geometry,
        "sections": sections,
    }


def _explain(route: dict, shortest: dict) -> str:
    n = route["n_potholes_observed"]
    more_mapped = False
    if route is shortest:
        first = (
            f"This is the shortest route found ({_fmt_dist(route['distance_m'])}) and it passes "
            f"{_plural(n, 'mapped pothole')}."
        )
    else:
        diff = route["distance_m"] - shortest["distance_m"]
        base = shortest["distance_m"]
        pct = diff / base * 100.0 if base > 0 else 0.0
        if diff >= 0.5:
            length_part = f"{diff:.0f} m ({pct:.0f}%) longer than the shortest route"
        elif diff <= -0.5:
            length_part = f"{-diff:.0f} m ({-pct:.0f}%) shorter than the shortest route"
        else:
            length_part = "about the same length as the shortest route"
        delta = shortest["n_potholes_observed"] - n
        if delta > 0:
            count_part = (
                f"passes {_plural(delta, 'mapped pothole')} fewer than the shortest route "
                f"({n} versus {shortest['n_potholes_observed']})"
            )
        elif delta < 0:
            more_mapped = True
            count_part = (
                f"passes {_plural(-delta, 'mapped pothole')} more than the shortest route "
                f"({n} versus {shortest['n_potholes_observed']})"
            )
        else:
            count_part = f"passes the same number of mapped potholes as the shortest route ({n})"
        first = f"This route is {length_part} and {count_part}."
    if route["distance_m"] <= 0:
        return first
    parts = [first]
    if more_mapped:
        parts.append(
            "It is chosen at this setting because it has less unsurveyed road, where potholes are "
            f"assumed rather than seen ({route['expected_potholes_unsurveyed']:.1f} assumed here "
            f"versus {shortest['expected_potholes_unsurveyed']:.1f} on the shortest route), not "
            "because fewer potholes were seen on it."
        )
    frac = route["surveyed_fraction"]
    if frac >= 0.9995:
        parts.append("All of this route has been surveyed.")
    elif frac <= 0.0005:
        parts.append("None of this route has been surveyed, so its condition is unknown.")
    else:
        parts.append(
            f"{frac * 100:.0f}% of this route has been surveyed; the rest has not, so its "
            "condition there is unknown."
        )
    if route["demo_m"] > 0 or route["n_potholes_demo"] > 0:
        parts.append(
            f"{_fmt_dist(route['demo_m'])} of it follows a synthetic demonstration drive and "
            f"{_plural(route['n_potholes_demo'], 'pothole')} on it "
            f"{'comes' if route['n_potholes_demo'] == 1 else 'come'} from demonstration data; "
            "neither describes the real road."
        )
    if route["expected_potholes_unsurveyed"] > 0:
        parts.append(
            f"About {route['expected_potholes_unsurveyed']:.1f} further potholes are assumed for "
            "the part without a survey, from an assumed rate and not from observation."
        )
    if route["evidence"] == EVIDENCE_INSUFFICIENT and frac > 0.0005:
        parts.append("That is too little survey to judge this route on potholes.")
    return " ".join(parts)
