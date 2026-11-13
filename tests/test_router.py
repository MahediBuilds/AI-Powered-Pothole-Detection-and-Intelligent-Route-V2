import json
import re

import pytest

from geo_fixtures import (
    add_fixes, add_obs, add_session, edge_point, insert_build, make_grid, path_points, xy,
)  # fmt: skip
from roadsense.config import IngestConfig, MappingConfig, RoutingConfig
from roadsense.geo.segments import rebuild_map
from roadsense.ingest.gps import haversine_m
from roadsense.routing import router as R
from roadsense.routing.router import (
    NoGraphError, NoRouteError, OutsideGraphError, RoutingError, plan_routes,
)  # fmt: skip
from roadsense.storage import db

MAPPING = MappingConfig()
NODE = {n: (r, c) for n in range(16) for r, c in [divmod(n, 4)]}


def node_ll(n):
    r, c = NODE[n]
    return xy(c * 100.0, r * 100.0)


def cfg(prior=0.0, max_detour=300.0):
    return RoutingConfig(unsurveyed_rate_per_100m=prior, max_detour_per_pothole_m=max_detour)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "r.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def graph():
    return make_grid(parallel=())  # one-way 5 -> 6 and curved 9-10 stay


@pytest.fixture(scope="module")
def graph_alley():
    return make_grid()  # also has the parallel alley (1, 2, key 1)


def plan(conn, graph, a, b, avoidance=0.5, routing=None):
    return plan_routes(conn, graph, node_ll(a), node_ll(b), avoidance, routing or cfg(), MAPPING)


def full_stats(graph, edges, n=0):
    return {e: (n, graph.g.edges[e]["length_m"]) for e in edges}


# --------------------------------------------------------------------------- basics


def test_shortest_path_when_avoidance_is_zero(conn, graph):
    res = plan(conn, graph, 0, 3, avoidance=0.0)
    assert len(res["routes"]) == 1
    r = res["routes"][0]
    assert r["labels"] == ["Shortest", "Your setting", "Strongest pothole avoidance"]
    assert r["recommended"] is True
    assert r["distance_m"] == pytest.approx(300.0, abs=0.5)
    assert r["duration_s"] == pytest.approx(36.0, abs=0.2)  # 300 m at 30 km/h
    assert haversine_m(*r["geometry"][0], *node_ll(0)) < 0.01
    assert haversine_m(*r["geometry"][-1], *node_ll(3)) < 0.01
    assert res["origin"]["node"] == 0 and res["destination"]["node"] == 3
    assert res["avoidance"] == 0.0


def test_endpoints_snap_to_nearest_node_and_report_it(conn, graph):
    o = (node_ll(0)[0] + 0.0001, node_ll(0)[1])  # about 11 m north of node 0
    res = plan_routes(conn, graph, o, node_ll(3), 0.0, cfg(), MAPPING)
    assert res["origin"]["requested"] == list(o)
    assert res["origin"]["snapped"] == list(node_ll(0))
    assert res["origin"]["snap_distance_m"] == pytest.approx(11.1, abs=0.3)


def test_same_origin_and_destination(conn, graph):
    res = plan(conn, graph, 5, 5)
    (r,) = res["routes"]
    assert r["distance_m"] == 0 and len(r["geometry"]) == 1 and r["sections"] == []
    json.dumps(res)


def test_one_way_street_is_respected(conn, graph):
    forward = plan(conn, graph, 5, 6, avoidance=0.0)["routes"][0]
    assert forward["distance_m"] == pytest.approx(100.0, abs=0.5)
    backward = plan(conn, graph, 6, 5, avoidance=0.0)["routes"][0]
    assert backward["distance_m"] == pytest.approx(300.0, abs=0.5)  # 6-2-1-5, not 6-5


def test_curved_edge_length_and_geometry_are_used(conn, graph):
    r = plan(conn, graph, 9, 10, avoidance=0.0)["routes"][0]
    assert r["distance_m"] == pytest.approx(graph.g.edges[9, 10, 0]["length_m"], abs=0.1)
    assert r["distance_m"] > 110 and len(r["geometry"]) == 4


# --------------------------------------------------------------------------- avoiding potholes

ROW0 = [(0, 1, 0), (1, 2, 0), (2, 3, 0)]


def pothole_on_12(conn, graph, n=1):
    insert_build(
        conn, graph,
        stats={(1, 2, 0): (n, 100.0)},
        potholes=[((1, 2, 0), 50.0 + i, "recorded") for i in range(n)],
    )  # fmt: skip


def test_detours_around_potholed_edge_when_cheaper_than_penalty(conn, graph):
    pothole_on_12(conn, graph)
    res = plan(conn, graph, 0, 3, avoidance=1.0)
    shortest, fewest = res["routes"]
    assert shortest["labels"] == ["Shortest"] and shortest["distance_m"] == pytest.approx(300, abs=0.5)
    assert shortest["n_potholes_observed"] == 1
    assert fewest["labels"] == ["Your setting", "Strongest pothole avoidance"] and fewest["recommended"]
    assert fewest["distance_m"] == pytest.approx(500.0, abs=0.5)  # +200 m < 300 m penalty
    assert fewest["n_potholes_observed"] == 0 and fewest["potholes"] == []
    assert "200 m (67%) longer" in fewest["explanation"]
    assert "1 mapped pothole fewer" in fewest["explanation"] and "(0 versus 1)" in fewest["explanation"]
    assert shortest["recommended"] is False


def test_no_detour_when_detour_is_longer_than_penalty(conn, graph):
    pothole_on_12(conn, graph)
    res = plan(conn, graph, 0, 3, avoidance=1.0, routing=cfg(max_detour=150.0))  # 200 m > 150 m
    (only,) = res["routes"]
    assert only["distance_m"] == pytest.approx(300, abs=0.5) and only["n_potholes_observed"] == 1
    assert len(only["labels"]) == 3 and only["recommended"]


def test_requested_avoidance_selects_between_candidates(conn, graph):
    pothole_on_12(conn, graph)
    # direct costs 300 + a*300, detour 500: break-even at a = 2/3
    low = plan(conn, graph, 0, 3, avoidance=0.5)
    direct = [r for r in low["routes"] if r["distance_m"] < 400][0]
    assert direct["recommended"] and direct["labels"] == ["Shortest", "Your setting"]
    high = plan(conn, graph, 0, 3, avoidance=0.8)
    detour = [r for r in high["routes"] if r["distance_m"] > 400][0]
    assert detour["recommended"] and detour["labels"] == ["Your setting", "Strongest pothole avoidance"]
    assert sum(r["recommended"] for r in high["routes"]) == 1


def test_potholes_listed_in_travel_order(conn, graph):
    insert_build(
        conn, graph, stats={(1, 2, 0): (2, 100.0)},
        potholes=[((1, 2, 0), 70.0, "recorded"), ((1, 2, 0), 20.0, "recorded")],
    )  # fmt: skip
    fwd = plan(conn, graph, 0, 3, avoidance=0.0)["routes"][0]["potholes"]
    back = plan(conn, graph, 3, 0, avoidance=0.0)["routes"][0]["potholes"]
    assert [p["id"] for p in fwd] == [2, 1] and [p["id"] for p in back] == [1, 2]
    assert set(fwd[0]) == {"id", "lat", "lon", "confidence", "gps_source"}


def test_three_distinct_routes_with_parallel_edges(conn, graph_alley):
    g = graph_alley
    insert_build(
        conn, g,
        stats={
            (1, 2, 0): (3, 100.0), (1, 2, 1): (2, g.g.edges[1, 2, 1]["length_m"]),
            **full_stats(g, [(1, 5, 0), (5, 6, 0), (2, 6, 0)]),
        },
        potholes=[((1, 2, 0), 20 + 20 * i, "recorded") for i in range(3)]
        + [((1, 2, 1), 30 + 30 * i, "recorded") for i in range(2)],
    )  # fmt: skip
    res = plan(conn, g, 1, 2, avoidance=0.4, routing=cfg(max_detour=100.0))
    shortest, yours, fewest = res["routes"]
    assert shortest["labels"] == ["Shortest"] and shortest["n_potholes_observed"] == 3
    assert yours["labels"] == ["Your setting"] and yours["recommended"]
    assert yours["n_potholes_observed"] == 2  # took the alley (key 1)
    assert yours["distance_m"] == pytest.approx(g.g.edges[1, 2, 1]["length_m"], abs=0.1)
    assert fewest["labels"] == ["Strongest pothole avoidance"] and fewest["n_potholes_observed"] == 0
    assert fewest["distance_m"] == pytest.approx(300, abs=0.5)
    assert [r["recommended"] for r in res["routes"]] == [False, True, False]
    json.dumps(res)


def test_identical_candidates_are_merged_without_potholes(conn, graph):
    insert_build(conn, graph, stats={})
    res = plan(conn, graph, 0, 15, avoidance=1.0)
    assert len(res["routes"]) == 1 and len(res["routes"][0]["labels"]) == 3
    assert res["routes"][0]["recommended"]


def test_avoidance_zero_merges_shortest_and_requested_only(conn, graph):
    pothole_on_12(conn, graph)
    res = plan(conn, graph, 0, 3, avoidance=0.0)
    assert [r["labels"] for r in res["routes"]] == [
        ["Shortest", "Your setting"], ["Strongest pothole avoidance"],
    ]  # fmt: skip
    assert [r["recommended"] for r in res["routes"]] == [True, False]


# --------------------------------------------------------------------------- unsurveyed prior


def test_prior_from_config_and_unsurveyed_expectation(conn, graph):
    insert_build(conn, graph, stats={})
    res = plan(conn, graph, 0, 3, avoidance=1.0, routing=cfg(prior=2.0))
    cm = res["cost_model"]
    assert cm["prior_rate_per_100m"] == 2.0 and "configured" in cm["prior_source"]
    assert cm["max_detour_per_pothole_m"] == 300.0 and "length" in cm["description"]
    (r,) = res["routes"]
    assert r["expected_potholes_unsurveyed"] == pytest.approx(6.0, abs=0.05)  # 300 m x 2 / 100
    assert r["expected_potholes_total"] == pytest.approx(6.0, abs=0.05)
    assert r["surveyed_m"] == 0 and r["surveyed_fraction"] == 0
    assert "None of this route has been surveyed" in r["explanation"]
    assert "About 6.0 further potholes are assumed" in r["explanation"]


def test_assumed_rate_when_nothing_surveyed_and_no_config(conn, graph):
    res = plan(conn, graph, 0, 3, routing=RoutingConfig())  # unsurveyed_rate_per_100m=None
    # nothing surveyed: the rate is the assumption itself, never 0
    assert res["cost_model"]["prior_rate_per_100m"] == 1.0
    assert "assumption only" in res["cost_model"]["prior_source"]
    assert "no road has been surveyed" in res["cost_model"]["prior_source"]
    assert res["build_id"] is None
    (r,) = res["routes"]
    assert r["expected_potholes_unsurveyed"] == pytest.approx(3.0, abs=0.05)  # 300 m x 1 / 100
    assert r["evidence"] == "insufficient" and res["confidence_notes"]


def _prior_scenario(conn, graph):
    # surveyed: the direct row with 1 pothole on (1, 2), clean on (0, 1) and (2, 3)
    insert_build(
        conn, graph,
        stats={(1, 2, 0): (1, 100.0), **full_stats(graph, [(0, 1, 0), (2, 3, 0)])},
        potholes=[((1, 2, 0), 50.0, "recorded")],
    )  # fmt: skip


def test_prior_from_observed_rate_changes_chosen_route(conn, graph):
    _prior_scenario(conn, graph)
    # config prior 0: the unsurveyed side street looks free, so detour (500 < 600)
    zero = plan(conn, graph, 0, 3, avoidance=1.0, routing=cfg(prior=0.0))
    assert max(r["distance_m"] for r in zero["routes"]) == pytest.approx(500, abs=0.5)
    assert [r for r in zero["routes"] if r["recommended"]][0]["distance_m"] > 400
    # blended rate = (1 x 100 + 1.0 x 500) / (300 + 500) = 0.75 per 100 m: the detour has
    # 300 m unsurveyed -> E = 2.25, cost 500 + 675 > 600, so the direct route is recommended
    observed = plan(conn, graph, 0, 3, avoidance=1.0, routing=RoutingConfig(max_detour_per_pothole_m=300))
    cm = observed["cost_model"]
    assert cm["prior_rate_per_100m"] == pytest.approx(0.75)
    assert "observed rate" in cm["prior_source"] and "300 m surveyed" in cm["prior_source"]
    (only,) = observed["routes"]
    assert only["distance_m"] == pytest.approx(300, abs=0.5) and only["recommended"]
    assert only["surveyed_fraction"] == pytest.approx(1.0)
    assert only["expected_potholes_unsurveyed"] == 0 and only["expected_potholes_total"] == 1
    assert "All of this route has been surveyed" in only["explanation"]


def test_unsurveyed_road_counts_in_detour_numbers(conn, graph):
    _prior_scenario(conn, graph)
    res = plan(conn, graph, 0, 3, avoidance=1.0, routing=cfg(prior=1.0))
    detour_costs = [r for r in res["routes"] if r["distance_m"] > 400]
    assert detour_costs == []  # 5 unsurveyed hundreds at 1/100 m each outweigh the single pothole


# --------------------------------------------------------------------------- sections


def test_sections_alternate_and_cover_the_geometry(conn, graph):
    insert_build(
        conn, graph, stats={(0, 1, 0): (0, 40.0), (1, 2, 0): (0, 70.0)},
        coverage=[((0, 1, 0), 0.0, 40.0), ((1, 2, 0), 30.0, 100.0)],
    )  # fmt: skip
    r = plan(conn, graph, 0, 3, avoidance=0.0)["routes"][0]
    assert [s["surveyed"] for s in r["sections"]] == [True, False, True, False]
    assert r["surveyed_m"] == pytest.approx(110.0)
    assert r["surveyed_fraction"] == pytest.approx(110 / 300, abs=1e-3)
    secs = r["sections"]
    # consecutive sections share their joint point; ends match the route geometry
    for a, b in zip(secs, secs[1:]):
        assert a["coords"][-1] == b["coords"][0]
    assert secs[0]["coords"][0] == r["geometry"][0] and secs[-1]["coords"][-1] == r["geometry"][-1]
    lengths = [
        sum(haversine_m(*p, *q) for p, q in zip(s["coords"], s["coords"][1:])) for s in secs
    ]
    assert lengths == pytest.approx([40, 90, 70, 100], abs=0.5)
    assert "37% of this route has been surveyed" in r["explanation"]


def test_sections_when_travelling_against_canonical_direction(conn, graph):
    insert_build(
        conn, graph, stats={(0, 1, 0): (0, 40.0), (1, 2, 0): (0, 70.0)},
        coverage=[((0, 1, 0), 0.0, 40.0), ((1, 2, 0), 30.0, 100.0)],
    )  # fmt: skip
    r = plan(conn, graph, 3, 0, avoidance=0.0)["routes"][0]
    assert [s["surveyed"] for s in r["sections"]] == [False, True, False, True]
    assert haversine_m(*r["sections"][-1]["coords"][-1], *node_ll(0)) < 0.01
    # the surveyed piece on edge (1, 2) is the part nearest node 2 (offsets 30..100 from node 1)
    first_surveyed = r["sections"][1]["coords"]
    assert haversine_m(*first_surveyed[0], *node_ll(2)) == pytest.approx(0.0, abs=0.01)
    assert haversine_m(*first_surveyed[-1], *xy(130, 0)) < 0.05


def test_sections_include_all_vertices_of_a_curved_edge(conn, graph):
    length = graph.g.edges[9, 10, 0]["length_m"]
    insert_build(conn, graph, stats={(9, 10, 0): (0, 30.0)}, coverage=[((9, 10, 0), 20.0, 50.0)])
    r = plan(conn, graph, 9, 10, avoidance=0.0)["routes"][0]
    assert [s["surveyed"] for s in r["sections"]] == [False, True, False]
    points = {tuple(p) for s in r["sections"] for p in s["coords"]}
    assert all(tuple(p) in points for p in r["geometry"])
    total = sum(
        haversine_m(*p, *q) for s in r["sections"] for p, q in zip(s["coords"], s["coords"][1:])
    )
    assert total == pytest.approx(length, abs=0.5)


def test_unsurveyed_route_is_one_unsurveyed_section(conn, graph):
    r = plan(conn, graph, 0, 3, avoidance=0.0)["routes"][0]
    assert [s["surveyed"] for s in r["sections"]] == [False]
    assert r["sections"][0]["coords"] == r["geometry"]


# --------------------------------------------------------------------------- synthetic flag


@pytest.mark.parametrize(
    "source,expected", [("recorded", False), ("synthetic", True), ("mixed", True)]
)
def test_uses_synthetic_data_flag(conn, graph, source, expected):
    insert_build(conn, graph, stats={(1, 2, 0): (1, 100.0)}, potholes=[((1, 2, 0), 50.0, source)])
    res = plan(conn, graph, 0, 3, avoidance=0.0)
    assert res["uses_synthetic_data"] is expected
    assert (any("synthetic" in c for c in res["caveats"])) is expected


def test_demo_mode_is_a_property_of_the_build_not_of_the_route(conn, graph):
    # the only synthetic pothole is far from the route 0 -> 3; demo mode still applies
    insert_build(conn, graph, stats={}, potholes=[((14, 15, 0), 50.0, "synthetic")])
    res = plan(conn, graph, 0, 3, avoidance=0.0)
    assert res["demo_mode"] is True and res["uses_synthetic_data"] is True
    assert res["caveats"][0] == R.SYNTHETIC_CAVEAT
    assert res["caveats"][1:] == R.CAVEATS
    assert all(r["n_potholes_demo"] == 0 for r in res["routes"])


# --------------------------------------------------------------------------- end to end


def test_end_to_end_with_map_build(conn, graph):
    add_session(conn, "s1")
    add_fixes(conn, "s1", path_points(graph, [0, 1, 2, 3]))
    add_obs(conn, "s1", *edge_point(graph, 1, 2, 0, 50), track=1)
    summary = rebuild_map(conn, graph, MAPPING, IngestConfig())
    assert summary["n_potholes_on_road"] == 1
    res = plan_routes(conn, graph, node_ll(0), node_ll(3), 1.0, RoutingConfig(), MAPPING)
    # blended rate: 1 pothole over the surveyed metres, shrunk towards the assumed 1.0 per 100 m
    surveyed = summary["surveyed_m"]
    assert surveyed == pytest.approx(300, abs=15)
    assert res["cost_model"]["prior_rate_per_100m"] == pytest.approx(
        (100.0 + 1.0 * 500.0) / (surveyed + 500.0)
    )
    (only,) = res["routes"]
    assert only["distance_m"] == pytest.approx(300, abs=0.5)
    assert only["surveyed_fraction"] > 0.98 and only["n_potholes_observed"] == 1
    assert only["potholes"][0]["gps_source"] == "recorded"
    # a configured prior of zero treats the unsurveyed side street as free: now it detours
    res0 = plan_routes(conn, graph, node_ll(0), node_ll(3), 1.0, cfg(prior=0.0), MAPPING)
    rec = [r for r in res0["routes"] if r["recommended"]][0]
    assert rec["distance_m"] > 400 and rec["n_potholes_observed"] == 0
    json.dumps(res0)


# --------------------------------------------------------------------------- errors


def test_no_graph(conn):
    with pytest.raises(NoGraphError):
        plan_routes(conn, None, (51.5, -0.1), (51.5, -0.1), 0.5, cfg(), MAPPING)
    assert issubclass(NoGraphError, RoutingError)


def test_point_outside_graph_names_the_point(conn, graph):
    far = (graph.bbox[0] + 0.1, graph.bbox[1])
    with pytest.raises(OutsideGraphError, match="origin"):
        plan_routes(conn, graph, far, node_ll(3), 0.5, cfg(), MAPPING)
    with pytest.raises(OutsideGraphError, match="destination"):
        plan_routes(conn, graph, node_ll(0), far, 0.5, cfg(), MAPPING)
    near = xy(-240, 0)  # 240 m from node 0: still allowed
    plan_routes(conn, graph, near, node_ll(3), 0.5, cfg(), MAPPING)
    with pytest.raises(OutsideGraphError):
        plan_routes(conn, graph, xy(-260, 0), node_ll(3), 0.5, cfg(), MAPPING)


def test_no_route_between_disconnected_nodes(conn):
    g = make_grid(rows=1, cols=2, oneway=((0, 1),), curved=(), parallel=())
    plan_routes(conn, g, node_ll(0), node_ll(1), 0.5, cfg(), MAPPING)
    with pytest.raises(NoRouteError):
        plan_routes(conn, g, node_ll(1), node_ll(0), 0.5, cfg(), MAPPING)


@pytest.mark.parametrize(
    "origin,dest,avoidance",
    [
        ((1.0,), (1.0, 2.0), 0.5),
        ("ab", (1.0, 2.0), 0.5),
        ((float("nan"), 0.0), (1.0, 2.0), 0.5),
        ((100.0, 0.0), (1.0, 2.0), 0.5),
        ((0.0, 200.0), (1.0, 2.0), 0.5),
        ((True, 0.0), (1.0, 2.0), 0.5),
        (None, (1.0, 2.0), 0.5),
        ((1.0, 2.0), (1.0, "x"), 0.5),
        ((1.0, 2.0), (1.0, 2.0), -0.1),
        ((1.0, 2.0), (1.0, 2.0), 1.5),
        ((1.0, 2.0), (1.0, 2.0), float("nan")),
        ((1.0, 2.0), (1.0, 2.0), "high"),
        ((1.0, 2.0), (1.0, 2.0), True),
    ],
)
def test_bad_input_is_value_error(conn, graph, origin, dest, avoidance):
    with pytest.raises(ValueError):
        plan_routes(conn, graph, origin, dest, avoidance, cfg(), MAPPING)


def test_map_built_with_another_graph_is_refused(conn, graph):
    other = make_grid(rows=3, cols=3)
    insert_build(conn, other, stats={})
    with pytest.raises(db.GraphMismatchError):
        plan(conn, graph, 0, 3)


def test_build_without_graph_gives_no_stats_and_still_routes(conn, graph):
    insert_build(conn, None)
    res = plan(conn, graph, 0, 3, avoidance=0.5)
    assert res["routes"][0]["surveyed_m"] == 0


# --------------------------------------------------------------------------- wording


FORBIDDEN = re.compile(
    r"pothole[- ]free|free of (any )?potholes|\bno potholes\b|\bsafe\w*|\bclear\w*|\bsmooth\w*|"
    r"\bsever\w*|\bminor\b|\bmajor\b|\bgood condition|\bperfect\w*|\bflawless|\bwithout potholes",
    re.IGNORECASE,
)


def test_forbidden_words_scanner_catches_claims():
    for bad in ["a pothole-free road", "this route is safe", "clear of damage", "severe pothole",
                "minor damage", "no potholes here", "a smooth ride"]:  # fmt: skip
        assert FORBIDDEN.search(bad), bad


def test_generated_text_never_claims_roads_are_pothole_free_safe_or_clear(conn, graph, graph_alley):
    texts: list[str] = []

    def collect(res):
        texts.extend(r["explanation"] for r in res["routes"])
        for caveat in res["caveats"]:
            texts.append(caveat.replace("no severity or size is recorded", ""))
        caveats = res["caveats"]
        if res["demo_mode"]:
            assert caveats[0] == R.SYNTHETIC_CAVEAT
            caveats = caveats[1:]
        assert caveats[:3] == R.CAVEATS[:3]

    # nothing surveyed, with and without a prior
    for prior in (0.0, 1.5):
        for av in (0.0, 0.5, 1.0):
            collect(plan(conn, graph, 0, 15, avoidance=av, routing=cfg(prior=prior)))
    # fully surveyed route that has zero mapped potholes
    insert_build(conn, graph, stats=full_stats(graph, ROW0))
    collect(plan(conn, graph, 0, 3, avoidance=1.0))
    conn.execute("DELETE FROM builds")
    # potholes, detours, partial coverage, synthetic sources
    insert_build(
        conn, graph,
        stats={(1, 2, 0): (2, 100.0), (0, 1, 0): (0, 40.0)},
        potholes=[((1, 2, 0), 30.0, "synthetic"), ((1, 2, 0), 60.0, "mixed")],
        coverage=[((0, 1, 0), 0.0, 40.0), ((1, 2, 0), 0.0, 100.0)],
    )  # fmt: skip
    for av in (0.0, 0.3, 1.0):
        collect(plan(conn, graph, 0, 3, avoidance=av, routing=cfg(prior=0.5)))
        collect(plan(conn, graph, 3, 0, avoidance=av))
    assert len(texts) > 20
    for text in texts:
        assert not FORBIDDEN.search(text), text
    assert "0 mapped potholes" in " ".join(texts)  # counts are stated as counts, not as claims


def test_caveats_state_the_limits(conn, graph):
    res = plan(conn, graph, 0, 3)
    joined = " ".join(res["caveats"])
    assert "mapped so far" in joined
    assert "not been surveyed is unknown" in joined
    assert "single class" in joined
    json.dumps(res)


# --------------------------------------------------------------------------- demo wording


def test_demo_plan_says_figures_include_demonstration_data(conn, graph):
    insert_build(
        conn, graph,
        stats={(1, 2, 0): (1, 100.0)},
        potholes=[((1, 2, 0), 50.0, "recorded"), ((2, 3, 0), 20.0, "synthetic")],
    )  # fmt: skip
    res = plan(conn, graph, 0, 3, routing=RoutingConfig())
    cm = res["cost_model"]
    assert res["demo_mode"] is True and cm["includes_demo_data"] is True
    assert "recorded AND synthetic demonstration data" in cm["prior_source"]
    assert "observed rate on surveyed road" not in cm["prior_source"]
    assert any("including synthetic demonstration road" in n for n in res["confidence_notes"])
    for text in [cm["prior_source"], *res["confidence_notes"]]:
        assert not FORBIDDEN.search(text), text


def test_non_demo_plan_wording_is_unchanged(conn, graph):
    insert_build(
        conn, graph, stats={(1, 2, 0): (1, 100.0)}, potholes=[((1, 2, 0), 50.0, "recorded")]
    )
    res = plan(conn, graph, 0, 3, routing=RoutingConfig())
    cm = res["cost_model"]
    assert res["demo_mode"] is False and cm["includes_demo_data"] is False
    assert cm["prior_source"].startswith("observed rate on surveyed road (1 mapped pothole over")
    assert not any("synthetic" in n for n in res["confidence_notes"])
    assert "Only 100 m of road has been surveyed in total, so the rate" in " ".join(
        res["confidence_notes"]
    )
