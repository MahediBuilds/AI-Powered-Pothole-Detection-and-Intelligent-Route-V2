import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import LineString

from geo_fixtures import make_grid, xy
from roadsense.geo.polyline import cut_polyline, point_at
from roadsense.ingest.gps import haversine_m
from roadsense.routing import graph as G
from roadsense.routing.graph import GraphError, canonical_edge, edge_geometry


def test_canonical_edge_orders_nodes():
    assert canonical_edge(5, 2, 1) == (2, 5, 1)
    assert canonical_edge(2, 5, 1) == (2, 5, 1)
    assert canonical_edge(3, 3, 0) == (3, 3, 0)


def test_grid_fixture_shape():
    graph = make_grid()
    assert graph.g.number_of_nodes() == 16
    assert graph.g.has_edge(5, 6, 0) and not graph.g.has_edge(6, 5, 0)  # one-way
    assert graph.g.has_edge(1, 2, 1) and graph.g.has_edge(2, 1, 1)  # parallel
    assert len(graph.g.edges[9, 10, 0]["geometry"]) == 4  # curved
    assert graph.g.edges[9, 10, 0]["length_m"] > 100.0


def test_edge_geometry_is_canonical_in_both_directions():
    graph = make_grid()
    geom, length = edge_geometry(graph, 10, 9, 0)  # asked from the high end
    geom2, length2 = edge_geometry(graph, 9, 10, 0)
    assert geom == geom2 and length == length2
    assert geom[0] == (graph.g.nodes[9]["lat"], graph.g.nodes[9]["lon"])
    assert len(geom) == 4


def test_edge_geometry_for_one_way_edge_only_stored_one_way():
    graph = make_grid()
    geom, length = edge_geometry(graph, 6, 5, 0)  # only 5 -> 6 is stored
    assert geom[0] == (graph.g.nodes[5]["lat"], graph.g.nodes[5]["lon"])
    assert length == pytest.approx(100.0, abs=0.5)
    with pytest.raises(KeyError):
        edge_geometry(graph, 0, 15, 0)


def test_one_way_stored_from_higher_node_is_reversed_to_canonical():
    g = nx.MultiDiGraph()
    g.add_node(1, lat=0.0, lon=0.0)
    g.add_node(2, lat=0.0, lon=0.001)
    geom = [(0.0, 0.001), (0.0, 0.0)]  # stored 2 -> 1
    g.add_edge(2, 1, 0, length_m=111.0, geometry=geom, name=None, highway=None, speed_kph=30.0)
    graph = G.make_road_graph(g)
    cgeom, _ = edge_geometry(graph, 1, 2, 0)
    assert cgeom == [(0.0, 0.0), (0.0, 0.001)]
    assert G.canonical_edges(graph) == [(1, 2, 0)]


def test_bbox_covers_nodes_and_geometry():
    graph = make_grid()
    min_lat, min_lon, max_lat, max_lon = graph.bbox
    assert min_lat < graph.g.nodes[0]["lat"]  # parallel edge bulges south of row 0
    assert max_lat > graph.g.nodes[15]["lat"] - 1e-9
    assert min_lon <= graph.g.nodes[0]["lon"] and max_lon >= graph.g.nodes[15]["lon"]


def test_save_load_round_trip_and_hash_stability(tmp_path):
    graph = make_grid()
    p1, p2 = tmp_path / "a" / "g.json", tmp_path / "g2.json"
    sha1 = G.save_graph(graph, p1)
    sha2 = G.save_graph(graph, p2)
    assert sha1 == sha2 == graph.sha256  # in-memory hash equals the file hash
    assert p1.read_bytes() == p2.read_bytes()
    loaded = G.load_graph(p1)
    assert loaded.sha256 == sha1
    assert set(loaded.g.nodes) == set(graph.g.nodes)
    assert set(loaded.g.edges(keys=True)) == set(graph.g.edges(keys=True))
    for u, v, k in graph.g.edges(keys=True):
        a, b = graph.g.edges[u, v, k], loaded.g.edges[u, v, k]
        assert a["length_m"] == b["length_m"]
        assert a["geometry"] == b["geometry"]
        assert (a["name"], a["highway"], a["speed_kph"]) == (b["name"], b["highway"], b["speed_kph"])
    assert loaded.bbox == graph.bbox
    assert loaded.meta == graph.meta
    assert G.make_road_graph(loaded.g, loaded.meta).sha256 == sha1


def test_hash_changes_when_graph_changes():
    a = make_grid()
    b = make_grid(parallel=())
    assert a.sha256 != b.sha256


def _write(tmp_path, doc):
    path = tmp_path / "bad.json"
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return path


def _good_doc():
    return {
        "meta": {},
        "nodes": [[1, 0.0, 0.0], [2, 0.0, 0.001]],
        "edges": [
            {"u": 1, "v": 2, "key": 0, "length_m": 111.0, "geometry": [[0.0, 0.0], [0.0, 0.001]]}
        ],
    }


def test_minimal_valid_file_loads_with_defaults(tmp_path):
    graph = G.load_graph(_write(tmp_path, _good_doc()))
    d = graph.g.edges[1, 2, 0]
    assert d["speed_kph"] == 30.0 and d["name"] is None and d["highway"] is None


def test_missing_file(tmp_path):
    with pytest.raises(GraphError, match="cannot read"):
        G.load_graph(tmp_path / "nope.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.pop("nodes"), "nodes"),
        (lambda d: d.pop("edges"), "edges"),
        (lambda d: d["nodes"].append([3, 0.0]), "node 2"),
        (lambda d: d["nodes"].append([1, 0.0, 0.0]), "duplicate node"),
        (lambda d: d["nodes"].append([3, 95.0, 0.0]), "out-of-range"),
        (lambda d: d["nodes"].append(["x", 0.0, 0.0]), "integer"),
        (lambda d: d["edges"][0].pop("length_m"), "length_m"),
        (lambda d: d["edges"][0].update(length_m=-1), "negative"),
        (lambda d: d["edges"][0].update(v=99), "unknown node"),
        (lambda d: d["edges"][0].update(geometry=[[0.0, 0.0]]), "geometry"),
        (lambda d: d["edges"][0].update(geometry=[[0.0, 0.0], [1]]), "not \\[lat, lon\\]"),
        (lambda d: d["edges"][0].update(speed_kph=0), "speed_kph"),
        (lambda d: d["edges"][0].update(name=5), "name"),
        (lambda d: d["edges"].append(dict(d["edges"][0])), "duplicate edge"),
        (lambda d: d.update(meta=[]), "meta"),
        (lambda d: d.update(nodes=[], edges=[]), "no nodes"),
    ],
)
def test_malformed_documents(tmp_path, mutate, match):
    doc = _good_doc()
    mutate(doc)
    with pytest.raises(GraphError, match=match):
        G.load_graph(_write(tmp_path, doc))


def test_not_json_and_wrong_top_level(tmp_path):
    with pytest.raises(GraphError, match="not valid JSON"):
        G.load_graph(_write(tmp_path, "{nope"))
    with pytest.raises(GraphError, match="top level"):
        G.load_graph(_write(tmp_path, "[1, 2]"))


def test_non_finite_number_rejected(tmp_path):
    path = _write(tmp_path, json.dumps(_good_doc()).replace("111.0", "NaN"))
    with pytest.raises(GraphError, match="finite"):
        G.load_graph(path)


# --------------------------------------------------------------------------- OSMnx conversion


def _osm_graph():
    osm = nx.MultiDiGraph()
    osm.add_node(10, x=-0.12, y=51.5)
    osm.add_node(20, x=-0.118, y=51.5005)
    osm.add_node(30, x=-0.116, y=51.5)
    curved = LineString([(-0.12, 51.5), (-0.119, 51.5008), (-0.118, 51.5005)])
    osm.add_edge(10, 20, 0, length=123.4, geometry=curved, maxspeed="50", name=["High Street", "A1"],
                 highway=["residential", "tertiary"])  # fmt: skip
    # reverse direction stored with geometry running against the edge direction
    osm.add_edge(20, 10, 0, length=123.4, geometry=curved, maxspeed="30 mph", name="High Street",
                 highway="residential")  # fmt: skip
    osm.add_edge(20, 30, 0, length=140.0, highway="tertiary", maxspeed="signals")  # no geometry
    osm.add_edge(30, 20, 0, highway="unknown_type")  # no length, no geometry
    return osm


def test_from_osmnx_converts_attributes():
    graph = G.from_osmnx(_osm_graph(), default_speed_kph=33.0, meta={"source": "t"})
    g = graph.g
    assert g.nodes[10] == {"lat": 51.5, "lon": -0.12}
    e = g.edges[10, 20, 0]
    assert e["length_m"] == 123.4
    assert e["geometry"] == [(51.5, -0.12), (51.5008, -0.119), (51.5005, -0.118)]
    assert e["name"] == "High Street" and e["highway"] == "residential"
    assert e["speed_kph"] == 50.0


def test_from_osmnx_reverses_geometry_against_edge_direction_and_converts_mph():
    g = G.from_osmnx(_osm_graph(), default_speed_kph=33.0).g
    e = g.edges[20, 10, 0]
    assert e["geometry"][0] == (51.5005, -0.118) and e["geometry"][-1] == (51.5, -0.12)
    assert e["speed_kph"] == pytest.approx(48.28032)


def test_from_osmnx_fallbacks():
    g = G.from_osmnx(_osm_graph(), default_speed_kph=33.0).g
    straight = g.edges[20, 30, 0]
    assert straight["geometry"] == [(51.5005, -0.118), (51.5, -0.116)]
    assert straight["speed_kph"] == 40.0  # "signals" is not a number -> tertiary default
    # the two 20<->30 edges differ in length by more than 1 m (140 m given vs ~120 m computed),
    # so they are not twins and the reverse edge is re-keyed to 1
    missing = g.edges[30, 20, 1]
    assert missing["length_m"] == pytest.approx(haversine_m(51.5, -0.116, 51.5005, -0.118))
    assert missing["speed_kph"] == 33.0  # unknown highway type -> configured default
    assert missing["name"] is None


def test_from_osmnx_graph_can_be_saved_and_loaded(tmp_path):
    graph = G.from_osmnx(_osm_graph(), default_speed_kph=33.0, meta={"source": "t"})
    path = tmp_path / "osm.json"
    G.save_graph(graph, path)
    assert G.load_graph(path).sha256 == graph.sha256


@pytest.mark.parametrize(
    "tag,highway,expected",
    [
        ("50", None, 50.0),
        ("25 mph", None, 40.2336),
        ("50 km/h", None, 50.0),
        (["30", "50"], None, 30.0),
        (["signals", "40"], None, 40.0),
        ("none", "motorway", 100.0),
        (None, "primary_link", 60.0),
        (None, None, 25.0),
        ("0", "residential", 30.0),
    ],
)
def test_parse_speed(tag, highway, expected):
    assert G.parse_speed_kph(tag, highway, 25.0) == pytest.approx(expected, abs=1e-3)


def test_download_rejects_bad_bbox_without_importing_osmnx():
    with pytest.raises(ValueError):
        G.download_graph((1.0, 1.0, 0.0, 2.0), default_speed_kph=30.0)


def test_runtime_modules_do_not_import_osmnx_at_import_time():
    import subprocess
    import sys

    code = (
        "import sys, roadsense.routing.graph, roadsense.geo.mapmatch, roadsense.routing.router;"
        "sys.exit(1 if 'osmnx' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --------------------------------------------------------------------------- CLI helpers


def test_bbox_around_recordings(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x_gps.csv").write_text(
        "timestamp_sec,lat,lon\n0,51.50,-0.12\n1,51.51,-0.10\n2,bad,row\n", encoding="utf-8"
    )
    (tmp_path / "other.csv").write_text("lat,lon\n10,10\n", encoding="utf-8")  # ignored
    (min_lat, min_lon, max_lat, max_lon), n = G.bbox_around_recordings(tmp_path, 1000.0)
    assert n == 2
    assert min_lat == pytest.approx(51.50 - 1000 / 111_320)
    assert max_lat == pytest.approx(51.51 + 1000 / 111_320)
    assert min_lon < -0.12 and max_lon > -0.10
    assert haversine_m(min_lat, -0.12, 51.50, -0.12) == pytest.approx(1000.0, rel=0.01)


def test_bbox_around_recordings_without_fixes(tmp_path):
    with pytest.raises(GraphError, match="no GPS fixes"):
        G.bbox_around_recordings(tmp_path, 100.0)


def test_cli_requires_exactly_one_bbox_source():
    with pytest.raises(SystemExit):
        G.main([])


def test_cli_reports_failure_without_network(monkeypatch, tmp_path, capsys):
    def boom(bbox, *, default_speed_kph):
        raise GraphError("offline")

    monkeypatch.setattr(G, "download_graph", boom)
    code = G.main(["--bbox", "51.0,-0.2,51.1,-0.1", "--out", str(tmp_path / "g.json")])
    assert code == 1 and "offline" in capsys.readouterr().err


def test_cli_writes_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(G, "download_graph", lambda bbox, *, default_speed_kph: make_grid())
    out = tmp_path / "sub" / "g.json"
    assert G.main(["--bbox", "51.0,-0.2,51.1,-0.1", "--out", str(out)]) == 0
    assert G.load_graph(out).g.number_of_nodes() == 16
    assert "nodes: 16" in capsys.readouterr().out


# --------------------------------------------------------------------------- polyline helpers


def test_cut_polyline_whole_edge_returns_vertices():
    graph = make_grid()
    geom, length = edge_geometry(graph, 9, 10, 0)
    assert cut_polyline(geom, length, 0, length) == geom


def test_cut_polyline_partial_and_point_at():
    geom = [xy(0, 0), xy(100, 0), xy(100, 100)]
    length = 200.0
    cut = cut_polyline(geom, length, 50, 150)
    assert len(cut) == 3  # start, the corner, end
    assert cut[1] == geom[1]
    assert haversine_m(*cut[0], *xy(50, 0)) < 0.01
    assert haversine_m(*cut[-1], *xy(100, 50)) < 0.01
    assert haversine_m(*point_at(geom, length, 150), *xy(100, 50)) < 0.01
    assert len(cut_polyline(geom, length, 30, 30)) == 1
    with pytest.raises(ValueError):
        cut_polyline(geom, length, 10, 5)


def test_cut_polyline_scales_offsets_to_nominal_length():
    geom = [xy(0, 0), xy(100, 0)]
    # nominal length 50 m but polyline is 100 m: offset 25 is the middle
    assert haversine_m(*point_at(geom, 50.0, 25.0), *xy(50, 0)) < 0.01
    assert math.isclose(haversine_m(*cut_polyline(geom, 50.0, 0, 50.0)[-1], *xy(100, 0)), 0, abs_tol=0.01)


# --------------------------------------------------------------------------- pair_directions


def _len(geom):
    return sum(haversine_m(*p, *q) for p, q in zip(geom, geom[1:]))


def _pair_graph(edges):
    """Nodes 1 (0, 0) and 2 (200, 0) (+ node 3 at (400, 0)); edges = (u, v, key, mid, name)."""
    pos = {1: (0, 0), 2: (200, 0), 3: (400, 0)}
    g = nx.MultiDiGraph()
    for n, p in pos.items():
        g.add_node(n, lat=xy(*p)[0], lon=xy(*p)[1])
    for u, v, k, mids, name in edges:
        geom = [xy(*pos[u]), *[xy(*m) for m in mids], xy(*pos[v])]
        g.add_edge(u, v, k, length_m=_len(geom), geometry=geom, name=name,
                   highway="residential", speed_kph=30.0)  # fmt: skip
    return g


_A = [(100, 30)]  # road A bulges north
_B = [(100, -40)]  # road B bulges south


def _swapped():
    return _pair_graph(
        [(1, 2, 0, _A, "A"), (1, 2, 1, _B, "B"), (2, 1, 0, _B, "B"), (2, 1, 1, _A, "A")]
    )


def test_pair_directions_fixes_swapped_keys():
    g = _swapped()
    assert G.pair_directions(g) == 2
    for k in (0, 1):
        f, r = g.edges[1, 2, k], g.edges[2, 1, k]
        assert f["name"] == r["name"]
        assert list(reversed(r["geometry"])) == f["geometry"]
    graph = G.make_road_graph(g)
    for k, name in ((0, "A"), (1, "B")):
        geom, length = edge_geometry(graph, 2, 1, k)  # asked from the high end
        assert geom == g.edges[1, 2, k]["geometry"]
        assert g.edges[1, 2, k]["name"] == name
        assert length == g.edges[1, 2, k]["length_m"]


def test_different_one_way_carriageways_get_different_keys():
    g = _pair_graph([(1, 2, 0, [(100, 5)], "N"), (2, 1, 0, [(100, -9)], "S")])
    assert G.pair_directions(g) == 1
    graph = G.make_road_graph(g)
    assert len(G.canonical_edges(graph)) == 2
    assert g.edges[1, 2, 0]["name"] == "N" and g.edges[2, 1, 1]["name"] == "S"


def test_ordinary_two_way_road_untouched():
    g = _pair_graph([(1, 2, 0, _A, "A"), (2, 1, 0, _A[::-1], "A")])
    before = {(u, v, k): d for u, v, k, d in g.edges(keys=True, data=True)}
    assert G.pair_directions(g) == 0
    assert {(u, v, k): d for u, v, k, d in g.edges(keys=True, data=True)} == before


def test_same_geometry_different_names_stays_one_identity():
    g = _pair_graph([(1, 2, 0, _A, "Forward St"), (2, 1, 0, _A, "Back St")])
    assert G.pair_directions(g) == 0
    assert len(G.canonical_edges(G.make_road_graph(g))) == 1


def test_one_way_street_untouched():
    g = _pair_graph([(1, 2, 0, _A, "A"), (1, 2, 3, _B, "B")])
    assert G.pair_directions(g) == 0
    assert set(g.edges(keys=True)) == {(1, 2, 0), (1, 2, 3)}


def test_self_loops_untouched():
    g = _pair_graph([(1, 2, 0, _A, "A"), (2, 1, 0, _A, "A")])
    loop = [xy(0, 0), xy(0, 20), xy(20, 20), xy(0, 0)]
    g.add_edge(1, 1, 5, length_m=_len(loop), geometry=loop, name="loop", highway=None, speed_kph=30.0)
    g.add_edge(1, 1, 2, length_m=_len(loop), geometry=loop, name="loop", highway=None, speed_kph=30.0)
    assert G.pair_directions(g) == 0
    assert set(g.edges(1, keys=True)) >= {(1, 1, 5), (1, 1, 2)}


def test_pair_directions_idempotent_and_deterministic():
    g1, g2 = _swapped(), _swapped()
    assert G.pair_directions(g1) == 2
    assert G.pair_directions(g1) == 0
    G.pair_directions(g2)
    assert G.make_road_graph(g1).sha256 == G.make_road_graph(g2).sha256


def test_pair_directions_preserves_connectivity():
    g = _swapped()
    g.add_edge(2, 3, 0, length_m=200.0, geometry=[xy(200, 0), xy(400, 0)], name="C",
               highway="residential", speed_kph=30.0)  # fmt: skip
    g.add_edge(3, 2, 0, length_m=200.0, geometry=[xy(400, 0), xy(200, 0)], name="C",
               highway="residential", speed_kph=30.0)  # fmt: skip

    def snapshot(gr):
        edges = sorted((u, v, round(d["length_m"], 6), d["name"]) for u, v, d in gr.edges(data=True))
        paths = [
            nx.shortest_path_length(gr, s, t, weight="length_m") for s, t in ((1, 3), (3, 1), (2, 1))
        ]
        return gr.number_of_edges(), edges, paths

    before = snapshot(g)
    G.pair_directions(g)
    assert snapshot(g) == before


def test_load_graph_hash_changes_when_rekeyed_and_is_stable(tmp_path):
    path = tmp_path / "swapped.json"
    file_sha = G.save_graph(G.make_road_graph(_swapped(), {"source": "t"}), path)
    a, b = G.load_graph(path), G.load_graph(path)
    assert a.sha256 != file_sha and a.sha256 == b.sha256
    assert a.meta["paired_edges_rekeyed"] == 2
    assert a.sha256 == hashlib.sha256((file_sha + ":paired-v1").encode()).hexdigest()
    assert path.read_bytes() and hashlib.sha256(path.read_bytes()).hexdigest() == file_sha

    ok = _pair_graph([(1, 2, 0, _A, "A"), (2, 1, 0, _A[::-1], "A")])
    ok_path = tmp_path / "ok.json"
    ok_sha = G.save_graph(G.make_road_graph(ok), ok_path)
    loaded = G.load_graph(ok_path)
    assert loaded.sha256 == ok_sha and "paired_edges_rekeyed" not in loaded.meta


def test_from_osmnx_normalises_keys_and_hash_matches_saved_file(tmp_path):
    osm = nx.MultiDiGraph()
    osm.add_node(1, x=0.0, y=0.0)
    osm.add_node(2, x=0.002, y=0.0)
    a = LineString([(0.0, 0.0), (0.001, 0.0003), (0.002, 0.0)])
    b = LineString([(0.0, 0.0), (0.001, -0.0004), (0.002, 0.0)])
    osm.add_edge(1, 2, 0, length=230.0, geometry=a, name="A")
    osm.add_edge(1, 2, 1, length=240.0, geometry=b, name="B")
    osm.add_edge(2, 1, 0, length=240.0, geometry=b, name="B")
    osm.add_edge(2, 1, 1, length=230.0, geometry=a, name="A")
    graph = G.from_osmnx(osm, default_speed_kph=30.0)
    for k in (0, 1):
        assert graph.g.edges[1, 2, k]["name"] == graph.g.edges[2, 1, k]["name"]
    path = tmp_path / "o.json"
    assert G.save_graph(graph, path) == graph.sha256
    assert G.load_graph(path).sha256 == graph.sha256


_REAL_GRAPH = Path(__file__).resolve().parent.parent / "data" / "graphs" / "road_graph.json"


@pytest.mark.skipif(not _REAL_GRAPH.exists(), reason="real road graph not present")
def test_real_graph_has_consistent_edge_identity():
    graph = G.load_graph(_REAL_GRAPH)
    g = graph.g
    assert g.number_of_edges() == 7138
    for u, v, k, d in g.edges(keys=True, data=True):
        if u < v and g.has_edge(v, u, k):
            assert abs(d["length_m"] - g.edges[v, u, k]["length_m"]) <= 1.0
    names = {k: (g.edges[2006424885, 2006424889, k]["name"], g.edges[2006424889, 2006424885, k]["name"])
             for k in (0, 1)}  # fmt: skip
    assert sorted(names.values()) == [("1st Cross Road",) * 2, ("1st Main Road",) * 2]
