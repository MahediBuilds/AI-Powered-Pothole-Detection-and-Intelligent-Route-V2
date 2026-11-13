"""Road graph: in-memory structure, JSON file format and OpenStreetMap download.

The runtime never imports OSMnx: ``load_graph`` reads a JSON file written by
``save_graph`` (or by the download command below). OSMnx is imported only inside
``download_graph``.

File format (one JSON object)::

    {"meta": {...},
     "nodes": [[id, lat, lon], ...],
     "edges": [{"u": 1, "v": 2, "key": 0, "length_m": 101.3, "name": "High St",
                "highway": "residential", "speed_kph": 30.0,
                "geometry": [[lat, lon], ...]}, ...]}

Edge identity. Stored edge ids (observations, potholes, coverage, edge stats)
are undirected: ``canonical_edge(u, v, key) == (min(u, v), max(u, v), key)``.
A directed edge ``(u, v, k)`` and its reverse ``(v, u, k)`` are treated as one
physical edge, so the invariant is: for every node pair, edges with the same
``key`` in the two directions are the same physical road, and different roads
never share a key. Raw OSMnx graphs do not satisfy this (parallel edges are
numbered independently per direction), so ``pair_directions`` establishes it
when a graph enters the system (``from_osmnx`` and ``load_graph``): it matches
each forward edge with its reverse twin by geometry and re-keys where needed.
Limitation: twins are recognised by geometry only, and only within the stated
tolerances (lengths within 1 m, shapes within 3 m at vertices and midpoint);
names and other attributes are ignored, so a road whose two directions are
drawn more than 3 m apart is treated as two separate one-way roads.
``edge_geometry`` returns the canonical geometry (running from the lower to the
higher node id) and length.

Run as a command to download a graph::

    python -m roadsense.routing.graph --bbox MIN_LAT,MIN_LON,MAX_LAT,MAX_LON
        [--out data/graphs/road_graph.json]
        [--around-recordings DIR --margin-m 2000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from roadsense.geo.projection import LocalProjection
from roadsense.ingest.gps import haversine_m

LatLon = tuple[float, float]
BBox = tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon

ATTRIBUTION = "© OpenStreetMap contributors"
_FALLBACK_SPEED_KPH = 30.0

# Rough free-flow speeds by OSM highway class, used when maxspeed is missing or
# unparseable. These are assumptions for a duration estimate, not measurements.
HIGHWAY_SPEED_KPH = {
    "motorway": 100.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 50.0,
    "tertiary": 40.0,
    "unclassified": 40.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
}


class GraphError(Exception):
    """A road graph file is missing, malformed or cannot be built."""


@dataclass
class RoadGraph:
    """A road network plus the identity of the data it came from."""

    g: nx.MultiDiGraph
    sha256: str
    bbox: BBox
    meta: dict[str, Any] = field(default_factory=dict)


def canonical_edge(u: int, v: int, key: int) -> tuple[int, int, int]:
    """The undirected identity of an edge: ``u < v`` (or equal for a loop)."""
    return (u, v, key) if u <= v else (v, u, key)


def edge_geometry(graph: RoadGraph, u: int, v: int, key: int) -> tuple[list[LatLon], float]:
    """Canonical geometry (from the lower node id) and length of an edge.

    Works whichever direction(s) exist in the directed graph; raises ``KeyError``
    if neither does.
    """
    cu, cv, ck = canonical_edge(u, v, key)
    g = graph.g
    if g.has_edge(cu, cv, ck):
        # After pair_directions the reverse twin has the same geometry by construction.
        data = g.edges[cu, cv, ck]
        return list(data["geometry"]), float(data["length_m"])
    if g.has_edge(cv, cu, ck):
        data = g.edges[cv, cu, ck]
        return list(reversed(data["geometry"])), float(data["length_m"])
    raise KeyError((u, v, key))


def canonical_edges(graph: RoadGraph) -> list[tuple[int, int, int]]:
    """Every undirected edge of the graph, sorted."""
    return sorted({canonical_edge(u, v, k) for u, v, k in graph.g.edges(keys=True)})


# --------------------------------------------------------------------------- direction pairing

_TWIN_LENGTH_TOL_M = 1.0
_TWIN_SHAPE_TOL_M = 3.0
_PAIRED_HASH_TAG = ":paired-v1"


def _point_polyline_dist(px, py, x, y) -> np.ndarray:
    """Distance from each point (px, py) to the polyline with vertices (x, y)."""
    if len(x) == 1:
        return np.hypot(px - x[0], py - y[0])
    best = np.full(len(px), np.inf)
    for i in range(len(x) - 1):
        dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
        seg2 = dx * dx + dy * dy
        if seg2 <= 0:
            t = np.zeros(len(px))
        else:
            t = np.clip(((px - x[i]) * dx + (py - y[i]) * dy) / seg2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (x[i] + t * dx), py - (y[i] + t * dy)))
    return best


def _midpoint_xy(x, y) -> tuple[float, float]:
    seg = np.hypot(np.diff(x), np.diff(y))
    total = float(seg.sum())
    if total <= 0:
        return float(x[0]), float(y[0])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    i = min(int(np.searchsorted(cum, total / 2, side="right")) - 1, len(seg) - 1)
    t = (total / 2 - cum[i]) / seg[i] if seg[i] > 0 else 0.0
    return float(x[i] + t * (x[i + 1] - x[i])), float(y[i] + t * (y[i + 1] - y[i]))


def _shape_deviation_m(fwd: list[LatLon], rev_reversed: list[LatLon]) -> float:
    """Largest distance from the vertices and midpoint of either polyline to the other."""
    proj = LocalProjection(fwd[0][0], fwd[0][1])
    ax, ay = proj.to_xy(np.array([p[0] for p in fwd]), np.array([p[1] for p in fwd]))
    bx, by = proj.to_xy(
        np.array([p[0] for p in rev_reversed]), np.array([p[1] for p in rev_reversed])
    )
    worst = 0.0
    for (px, py), (qx, qy) in (((ax, ay), (bx, by)), ((bx, by), (ax, ay))):
        mx, my = _midpoint_xy(px, py)
        pts_x = np.append(px, mx)
        pts_y = np.append(py, my)
        worst = max(worst, float(_point_polyline_dist(pts_x, pts_y, qx, qy).max()))
    return worst


def _twin_deviation(fdata: dict[str, Any], rdata: dict[str, Any]) -> float | None:
    """Shape deviation if the reverse edge is a twin of the forward edge, else ``None``."""
    if abs(float(fdata["length_m"]) - float(rdata["length_m"])) > _TWIN_LENGTH_TOL_M:
        return None
    fg = [tuple(p) for p in fdata["geometry"]]
    rg = [tuple(p) for p in reversed(rdata["geometry"])]
    if fg == rg:
        return 0.0
    dev = _shape_deviation_m(fg, rg)
    return dev if dev <= _TWIN_SHAPE_TOL_M else None


def pair_directions(g: nx.MultiDiGraph) -> int:
    """Re-key edges in place so ``(u, v, k)`` and ``(v, u, k)`` are one physical road.

    For every node pair ``a < b`` with edges in both directions, a reverse edge
    is the twin of a forward edge when its reversed geometry coincides with the
    forward geometry (lengths within 1 m; shapes within 3 m at the vertices and
    midpoint). Twins are matched greedily, smallest deviation first, each edge at
    most once, ties broken by key. Twinned edges then share a key, and every
    un-twinned edge gets a key no other edge of the pair uses in either
    direction. Keys that already satisfy this are kept (an ordinary two-way road
    with key 0 both ways is untouched); forward edges always keep their keys.
    Self-loops and one-way pairs are left alone. Attribute dicts move with the
    edge unchanged. Returns the number of directed edges whose key changed.
    """
    pairs = sorted({(u, v) for u, v in g.edges() if u < v and g.has_edge(v, u)})
    changed = 0
    for a, b in pairs:
        fkeys = sorted(g[a][b])
        rkeys = sorted(g[b][a])
        cands = []
        for kf in fkeys:
            for kr in rkeys:
                dev = _twin_deviation(g.edges[a, b, kf], g.edges[b, a, kr])
                if dev is not None:
                    cands.append((dev, kf, kr))
        cands.sort()
        new_key: dict[int, int] = {}
        used_f: set[int] = set()
        for _, kf, kr in cands:
            if kf in used_f or kr in new_key:
                continue
            used_f.add(kf)
            new_key[kr] = kf
        taken = set(fkeys)
        leftovers = []
        for kr in rkeys:
            if kr in new_key:
                continue
            if kr not in taken:
                new_key[kr] = kr
                taken.add(kr)
            else:
                leftovers.append(kr)
        nxt = 0
        for kr in leftovers:
            while nxt in taken:
                nxt += 1
            new_key[kr] = nxt
            taken.add(nxt)
        moved = [kr for kr in rkeys if new_key[kr] != kr]
        if not moved:
            continue
        data = {kr: g.edges[b, a, kr] for kr in rkeys}
        for kr in rkeys:
            g.remove_edge(b, a, kr)
        for kr in rkeys:
            g.add_edge(b, a, new_key[kr], **data[kr])
        changed += len(moved)
    return changed


# --------------------------------------------------------------------------- build / serialise


def _graph_bbox(g: nx.MultiDiGraph) -> BBox:
    lats = [d["lat"] for _, d in g.nodes(data=True)]
    lons = [d["lon"] for _, d in g.nodes(data=True)]
    for _, _, d in g.edges(data=True):
        for lat, lon in d["geometry"]:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        raise GraphError("the graph has no nodes")
    return (min(lats), min(lons), max(lats), max(lons))


def _payload(g: nx.MultiDiGraph, meta: dict[str, Any]) -> dict[str, Any]:
    nodes = [[n, d["lat"], d["lon"]] for n, d in sorted(g.nodes(data=True))]
    edges = []
    for u, v, k in sorted(g.edges(keys=True)):
        d = g.edges[u, v, k]
        edges.append(
            {
                "u": u,
                "v": v,
                "key": k,
                "length_m": d["length_m"],
                "name": d.get("name"),
                "highway": d.get("highway"),
                "speed_kph": d["speed_kph"],
                "geometry": [[lat, lon] for lat, lon in d["geometry"]],
            }
        )
    return {"meta": meta, "nodes": nodes, "edges": edges}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def make_road_graph(g: nx.MultiDiGraph, meta: dict[str, Any] | None = None) -> RoadGraph:
    """Wrap an in-memory graph; ``sha256`` is the hash of its canonical JSON.

    That is the hash ``save_graph`` / ``load_graph`` give the same graph after a
    file round trip, as long as ``meta`` is unchanged.
    """
    meta = dict(meta or {})
    return RoadGraph(
        g=g,
        sha256=hashlib.sha256(_canonical_bytes(_payload(g, meta))).hexdigest(),
        bbox=_graph_bbox(g),
        meta=meta,
    )


def save_graph(graph: RoadGraph, path: str | Path) -> str:
    """Write the graph as canonical JSON (parents created); returns the file's SHA-256."""
    data = _canonical_bytes(_payload(graph.g, graph.meta))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _num(value: Any, what: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphError(f"{path}: {what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise GraphError(f"{path}: {what} must be finite, got {value!r}")
    return float(value)


def _int(value: Any, what: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphError(f"{path}: {what} must be an integer, got {value!r}")
    return value


def _geometry(raw: Any, what: str, path: Path) -> list[LatLon]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise GraphError(f"{path}: {what} geometry must be a list of at least 2 [lat, lon] points")
    out = []
    for p in raw:
        if not isinstance(p, list) or len(p) != 2:
            raise GraphError(f"{path}: {what} geometry point {p!r} is not [lat, lon]")
        out.append(
            (_num(p[0], f"{what} geometry lat", path), _num(p[1], f"{what} geometry lon", path))
        )
    return out


def load_graph(path: str | Path) -> RoadGraph:
    """Read a graph file; ``sha256`` is the hash of the file's bytes.

    Raises ``GraphError`` if the file is missing, not JSON, or structurally wrong.
    """
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise GraphError(f"{path}: cannot read road graph file: {exc}") from exc
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise GraphError(f"{path}: top level must be an object")
    for key in ("nodes", "edges"):
        if not isinstance(doc.get(key), list):
            raise GraphError(f"{path}: missing or invalid {key!r} list")
    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        raise GraphError(f"{path}: 'meta' must be an object")

    g = nx.MultiDiGraph()
    for i, node in enumerate(doc["nodes"]):
        if not isinstance(node, list) or len(node) != 3:
            raise GraphError(f"{path}: node {i} must be [id, lat, lon], got {node!r}")
        nid = _int(node[0], f"node {i} id", path)
        lat = _num(node[1], f"node {nid} lat", path)
        lon = _num(node[2], f"node {nid} lon", path)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise GraphError(f"{path}: node {nid} has out-of-range coordinates ({lat}, {lon})")
        if nid in g:
            raise GraphError(f"{path}: duplicate node id {nid}")
        g.add_node(nid, lat=lat, lon=lon)

    for i, e in enumerate(doc["edges"]):
        if not isinstance(e, dict):
            raise GraphError(f"{path}: edge {i} must be an object")
        for req in ("u", "v", "key", "length_m", "geometry"):
            if req not in e:
                raise GraphError(f"{path}: edge {i} is missing {req!r}")
        u, v, key = (_int(e[k], f"edge {i} {k}", path) for k in ("u", "v", "key"))
        for n in (u, v):
            if n not in g:
                raise GraphError(f"{path}: edge {i} refers to unknown node {n}")
        length = _num(e["length_m"], f"edge {i} length_m", path)
        if length < 0:
            raise GraphError(f"{path}: edge {i} has negative length_m {length}")
        speed = _num(e.get("speed_kph", _FALLBACK_SPEED_KPH), f"edge {i} speed_kph", path)
        if speed <= 0:
            raise GraphError(f"{path}: edge {i} has non-positive speed_kph {speed}")
        for text in ("name", "highway"):
            if e.get(text) is not None and not isinstance(e[text], str):
                raise GraphError(f"{path}: edge {i} {text} must be a string or null")
        if g.has_edge(u, v, key):
            raise GraphError(f"{path}: duplicate edge ({u}, {v}, {key})")
        g.add_edge(
            u,
            v,
            key,
            length_m=length,
            geometry=_geometry(e["geometry"], f"edge {i}", path),
            name=e.get("name"),
            highway=e.get("highway"),
            speed_kph=speed,
        )

    if not len(g):
        raise GraphError(f"{path}: the graph has no nodes")
    sha = hashlib.sha256(raw_bytes).hexdigest()
    rekeyed = pair_directions(g)
    if rekeyed:
        # The in-memory edge ids no longer match the file's, so the identity must differ.
        sha = hashlib.sha256((sha + _PAIRED_HASH_TAG).encode()).hexdigest()
        meta = {**meta, "paired_edges_rekeyed": rekeyed}
    return RoadGraph(g=g, sha256=sha, bbox=_graph_bbox(g), meta=meta)


# --------------------------------------------------------------------------- OSMnx conversion

_MAXSPEED_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*$", re.IGNORECASE)


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def parse_speed_kph(maxspeed: Any, highway: str | None, default_speed_kph: float) -> float:
    """Speed in km/h from an OSM ``maxspeed`` tag, else a default for the road class.

    Plain numbers are km/h and ``"NN mph"`` is converted; anything else
    ("signals", "none", "walk", ...) falls back to ``HIGHWAY_SPEED_KPH`` and then
    to ``default_speed_kph``.
    """
    candidates = maxspeed if isinstance(maxspeed, (list, tuple)) else [maxspeed]
    for c in candidates:
        if c is None:
            continue
        m = _MAXSPEED_RE.match(str(c))
        if m and float(m.group(1)) > 0:
            value = float(m.group(1))
            return value * 1.609344 if (m.group(2) or "").lower() == "mph" else value
    if highway:
        base = highway.removesuffix("_link")
        if base in HIGHWAY_SPEED_KPH:
            return HIGHWAY_SPEED_KPH[base]
    return float(default_speed_kph)


def from_osmnx(
    osm: nx.MultiDiGraph,
    *,
    default_speed_kph: float,
    meta: dict[str, Any] | None = None,
) -> RoadGraph:
    """Convert an OSMnx-style graph (node ``x``/``y``, edge ``length``/``geometry``).

    Pure function (no OSMnx import) so it can be tested with a hand-built graph.
    Edge geometry is the shapely ``geometry`` attribute when present (reversed if
    it runs against the edge direction), otherwise the straight line between the
    nodes. List-valued ``name``/``highway`` keep their first value.
    """
    g = nx.MultiDiGraph()
    for n, d in osm.nodes(data=True):
        g.add_node(int(n), lat=float(d["y"]), lon=float(d["x"]))
    for u, v, k, d in osm.edges(keys=True, data=True):
        u, v = int(u), int(v)
        a = (g.nodes[u]["lat"], g.nodes[u]["lon"])
        b = (g.nodes[v]["lat"], g.nodes[v]["lon"])
        line = d.get("geometry")
        if line is not None:
            geom = [(float(y), float(x)) for x, y in line.coords]
            if len(geom) >= 2 and haversine_m(*geom[0], *a) > haversine_m(*geom[0], *b):
                geom.reverse()
        else:
            geom = [a, b]
        if len(geom) < 2:
            geom = [a, b]
        length = d.get("length")
        if length is None:
            length = sum(haversine_m(*p, *q) for p, q in zip(geom, geom[1:]))
        highway = _first(d.get("highway"))
        name = _first(d.get("name"))
        g.add_edge(
            u,
            v,
            int(k),
            length_m=float(length),
            geometry=geom,
            name=None if name is None else str(name),
            highway=None if highway is None else str(highway),
            speed_kph=parse_speed_kph(d.get("maxspeed"), highway, default_speed_kph),
        )
    pair_directions(g)
    return make_road_graph(g, meta)


def download_graph(bbox: BBox, *, default_speed_kph: float) -> RoadGraph:
    """Download the drivable OpenStreetMap network inside ``bbox`` (needs internet).

    ``bbox`` is ``(min_lat, min_lon, max_lat, max_lon)``. OSMnx 2.x takes
    ``(left, bottom, right, top)`` = ``(min_lon, min_lat, max_lon, max_lat)``,
    so the order is converted here.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    if not (min_lat < max_lat and min_lon < max_lon):
        raise ValueError(f"bbox must have min < max in both axes, got {bbox}")
    import osmnx as ox  # heavy; only needed for downloading

    try:
        osm = ox.graph_from_bbox(
            (min_lon, min_lat, max_lon, max_lat), network_type="drive", simplify=True
        )
    except Exception as exc:  # osmnx raises many types (network, empty result)
        raise GraphError(f"OSMnx could not download a graph for bbox {bbox}: {exc}") from exc
    meta = {
        "bbox": list(bbox),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenStreetMap via OSMnx",
        "osmnx_version": ox.__version__,
        "attribution": ATTRIBUTION,
    }
    return from_osmnx(osm, default_speed_kph=default_speed_kph, meta=meta)


# --------------------------------------------------------------------------- CLI


def bbox_around_recordings(directory: str | Path, margin_m: float) -> tuple[BBox, int]:
    """Bounding box of every ``*_gps.csv`` fix under ``directory`` plus a margin.

    Returns the box and the number of fixes read. Rows without a numeric
    ``lat``/``lon`` are skipped.
    """
    lats: list[float] = []
    lons: list[float] = []
    for csv_path in sorted(Path(directory).rglob("*_gps.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    lat, lon = float(row["lat"]), float(row["lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(lat) and math.isfinite(lon):
                    lats.append(lat)
                    lons.append(lon)
    if not lats:
        raise GraphError(f"no GPS fixes found in *_gps.csv files under {directory}")
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * math.cos(math.radians((min(lats) + max(lats)) / 2)))
    return (min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon), len(lats)


def _parse_bbox(text: str) -> BBox:
    try:
        parts = [float(p) for p in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected MIN_LAT,MIN_LON,MAX_LAT,MAX_LON")
    return (parts[0], parts[1], parts[2], parts[3])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download an OpenStreetMap drive graph.")
    parser.add_argument("--bbox", type=_parse_bbox, help="MIN_LAT,MIN_LON,MAX_LAT,MAX_LON")
    parser.add_argument("--out", default="data/graphs/road_graph.json")
    parser.add_argument(
        "--around-recordings",
        metavar="DIR",
        help="derive the bbox from every *_gps.csv under DIR",
    )
    parser.add_argument("--margin-m", type=float, default=2000.0)
    parser.add_argument("--default-speed-kph", type=float, default=30.0)
    args = parser.parse_args(argv)
    if (args.bbox is None) == (args.around_recordings is None):
        parser.error("give exactly one of --bbox and --around-recordings")
    try:
        if args.around_recordings:
            bbox, n_fixes = bbox_around_recordings(args.around_recordings, args.margin_m)
            print(f"bbox from {n_fixes} GPS fixes + {args.margin_m:g} m margin: {bbox}")
        else:
            bbox = args.bbox
        graph = download_graph(bbox, default_speed_kph=args.default_speed_kph)
        sha = save_graph(graph, args.out)
    except (GraphError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"nodes: {graph.g.number_of_nodes()}, directed edges: {graph.g.number_of_edges()}, "
        f"undirected edges: {len(canonical_edges(graph))}\n"
        f"wrote {args.out} (sha256 {sha[:16]}...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
