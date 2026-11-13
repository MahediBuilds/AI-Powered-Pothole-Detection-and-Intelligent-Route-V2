import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from api_fixtures import make_api_cfg
from geo_fixtures import xy
from pipeline_fixtures import FakeDetector

from roadsense.geo.mapmatch import EdgeIndex
from roadsense.ingest.gps import haversine_m, load_gps_csv
from roadsense.routing.graph import canonical_edge, make_road_graph, save_graph
from roadsense.storage import db

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_demo.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_demo", SCRIPT)
demo = importlib.util.module_from_spec(_spec)
sys.modules["make_synthetic_demo"] = demo
_spec.loader.exec_module(demo)


def ladder_graph(n=10, spacing=50.0, rung=20.0):
    """Two parallel rails (ids 0..n-1 below, n..2n-1 above) joined by a rung at every node."""
    g = nx.MultiDiGraph()
    for i in range(n):
        g.add_node(i, **dict(zip(("lat", "lon"), xy(spacing * i, 0.0))))
        g.add_node(n + i, **dict(zip(("lat", "lon"), xy(spacing * i, rung))))

    def link(u, v, name):
        a, b = (g.nodes[u]["lat"], g.nodes[u]["lon"]), (g.nodes[v]["lat"], g.nodes[v]["lon"])
        for p, q, geom in ((u, v, [a, b]), (v, u, [b, a])):
            g.add_edge(p, q, 0, length_m=haversine_m(*a, *b), geometry=geom, name=name,
                       highway="residential", speed_kph=30.0)  # fmt: skip

    for i in range(n - 1):
        link(i, i + 1, "Bottom Road")
        link(n + i, n + i + 1, "Top Road")
    for i in range(n):
        link(i, n + i, None)
    return make_road_graph(g, {"source": "test ladder"})


def write_image(path: Path, positive: bool, shade: int = 0):
    import cv2

    image = np.full((100, 120, 3), 15 + 6 * shade, np.uint8)  # each image looks different
    if positive:
        image[30:70, 40:80] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    for i in range(6):
        write_image(root / "images" / "val" / f"pos_{i}.jpg", True, i)
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "val" / f"pos_{i}.txt").write_text("0 0.5 0.5 0.33 0.4\n")
    for i in range(4):
        write_image(root / "images" / "val" / f"neg_{i}.jpg", False, 10 + i)
        (root / "labels" / "val" / f"neg_{i}.txt").write_text("")
    # the test split must never be touched: a file there that would break any reader
    (root / "images" / "test").mkdir(parents=True)
    (root / "images" / "test" / "secret_test_image.jpg").write_bytes(b"not an image")
    (root / "labels" / "test").mkdir(parents=True)
    (root / "labels" / "test" / "secret_test_image.txt").write_text("0 0.5 0.5 0.9 0.9\n")
    return root


@pytest.fixture
def env(tmp_path, dataset):
    # a large per-pothole detour makes avoiding the potholes worthwhile on this tiny map
    cfg = make_api_cfg(tmp_path, routing={"max_detour_per_pothole_m": 1000.0})
    graph = ladder_graph()
    save_graph(graph, cfg.routing.graph_path)
    return cfg, graph, dataset


def run(env, tmp_path, **kwargs):
    cfg, _, dataset = env
    options = dict(
        seed=3, n_potholes=2, detector_factory=FakeDetector, dataset_dir=dataset,
        out_dir=tmp_path / "demo", log=lambda message: None,
    )  # fmt: skip
    options.update(kwargs)
    return demo.run_demo(cfg, **options)


# ---------------------------------------------------------------------------- pieces


def test_caption_is_burned_into_the_banner():
    frame = np.full((120, 400, 3), 90, np.uint8)
    out = demo.add_caption(frame)
    assert out is frame
    banner = frame[: demo.BANNER_H]
    assert (banner == (0, 255, 255)).all(axis=2).any()  # yellow text
    assert (banner == 0).all(axis=2).sum() > banner.shape[0] * banner.shape[1] * 0.5  # black bar
    assert (frame[demo.BANNER_H :] == 90).all()
    assert demo.CAPTION == (
        "SYNTHETIC DEMO - images from RDD2022 validation set, GPS trace generated"
    )
    demo.add_caption(np.zeros((60, 100, 3), np.uint8))  # a narrow frame still works


def test_every_rendered_frame_has_the_caption_and_blue_channel_is_clear_of_it(dataset):
    frame = demo.render_frame(dataset / "images" / "val" / "pos_0.jpg")
    assert frame.shape == (demo.FRAME_SIZE, demo.FRAME_SIZE, 3)
    assert (frame[: demo.BANNER_H] == (0, 255, 255)).all(axis=2).any()
    assert frame[: demo.BANNER_H, :, 0].max() == 0  # the caption never looks like a bright object
    assert frame[demo.BANNER_H :, :, 0].max() == 255  # the image itself is there
    with pytest.raises(demo.DemoError, match="cannot read"):
        demo.render_frame(dataset / "images" / "test" / "secret_test_image.jpg")


def test_schedule_satisfies_tracker_and_spacing_rules():
    from roadsense.config import TrackingConfig

    for tracking in (
        TrackingConfig(),
        TrackingConfig(frame_stride=1, min_hits=5, max_age_frames=10, iou_match=0.3),
        TrackingConfig(frame_stride=5, min_hits=2, max_age_frames=0),
    ):
        s = demo.make_schedule(13, tracking)
        assert s.hold >= tracking.frame_stride * tracking.min_hits
        # more processed frames than max_age without a pothole between two images
        assert s.gap >= tracking.frame_stride * (tracking.max_age_frames + 1)
        assert s.cycle / s.fps * demo.NOMINAL_SPEED_MPS >= 30.0
        segs = s.segments([Path(f"p{i}") for i in range(13)], [Path(f"f{i}") for i in range(40)])
        assert segs[0].start == 0 and segs[-1].end == s.total_frames
        assert all(a.end == b.start for a, b in zip(segs, segs[1:]))  # no holes, no overlaps
        assert sum(1 for x in segs if x.kind == "pothole") == 13
        assert sum(1 for x in segs if x.kind == "filler") == s.n_filler_segments()


def test_list_val_images_uses_validation_only(dataset):
    positives, negatives = demo.list_val_images(dataset)
    assert [p.name for p in positives] == [f"pos_{i}.jpg" for i in range(6)]
    assert [p.name for p in negatives] == [f"neg_{i}.jpg" for i in range(4)]
    assert all(p.parent.name == "val" for p in positives + negatives)
    assert not any(p.parent.name == "test" for p in positives + negatives)


def test_list_val_images_prefers_large_boxes_when_enough(dataset):
    (dataset / "labels" / "val" / "pos_0.txt").write_text("0 0.5 0.5 0.02 0.02\n")
    positives, _ = demo.list_val_images(dataset, min_positives=3)
    assert "pos_0.jpg" not in [p.name for p in positives] and len(positives) == 5
    positives, _ = demo.list_val_images(dataset, min_positives=6)
    assert len(positives) == 6  # too few large ones: all positives are used


def test_missing_dataset_is_a_clear_error(tmp_path):
    with pytest.raises(demo.DemoError, match="validation images not found"):
        demo.list_val_images(tmp_path / "nothing")


def test_choose_images_is_deterministic_and_validated(dataset):
    positives, negatives = demo.list_val_images(dataset)
    a = demo.choose_images(positives, negatives, 3, 5, seed=1)
    assert a == demo.choose_images(positives, negatives, 3, 5, seed=1)
    assert len(set(a[0])) == 3 and len(a[1]) == 5
    assert all(b != c for b, c in zip(a[1], a[1][1:]))  # no image twice in a row
    assert any(demo.choose_images(positives, negatives, 3, 5, seed=s) != a for s in range(2, 8))
    with pytest.raises(demo.DemoError, match="only 6"):
        demo.choose_images(positives, negatives, 7, 5, seed=1)
    with pytest.raises(demo.DemoError, match="pothole-free"):
        demo.choose_images(positives, [], 2, 3, seed=1)


def test_generated_gps_parses_with_the_loader_and_lies_on_the_path(env, tmp_path):
    cfg, graph, _ = env
    schedule = demo.make_schedule(2, cfg.tracking)
    (cand, *_) = demo.find_candidates(graph, schedule, seed=3)
    fixes, speed = demo.generate_fixes(cand.points, schedule.duration_s)
    path = tmp_path / "trace_gps.csv"
    demo.write_gps_csv(path, fixes)
    log = load_gps_csv(path, 25.0)
    assert log.issues == () and len(log.fixes) == len(fixes) and log.summary.n_used == len(fixes)
    assert log.fixes[0].t_video_s == pytest.approx(-1.0)
    assert log.fixes[-1].t_video_s == pytest.approx(schedule.duration_s + 1.0)
    # constant plausible speed
    assert 7.0 <= speed <= 12.0 and {f.speed_mps for f in log.fixes} == {round(speed, 2)}
    # every fix is within a metre of the chosen node path
    path_edges = {canonical_edge(a, b, 0) for a, b in zip(cand.nodes, cand.nodes[1:])}
    index = EdgeIndex(graph)
    for fix in log.fixes:
        snap = index.snap(fix.lat, fix.lon, 5.0)
        assert snap is not None and snap.distance_m < 1.0
        at_path_node = any(
            haversine_m(fix.lat, fix.lon, graph.g.nodes[n]["lat"], graph.g.nodes[n]["lon"]) < 1.0
            for n in cand.nodes
        )
        assert snap.edge in path_edges or at_path_node  # a node is shared with other edges
    # the first and last fixes are the path ends
    first, last = graph.g.nodes[cand.nodes[0]], graph.g.nodes[cand.nodes[-1]]
    assert haversine_m(log.fixes[0].lat, log.fixes[0].lon, first["lat"], first["lon"]) < 1.0
    assert haversine_m(log.fixes[-1].lat, log.fixes[-1].lon, last["lat"], last["lon"]) < 1.0
    # consecutive potholes are more than 30 m apart at this speed
    assert speed * schedule.cycle / schedule.fps >= 30.0


def test_candidates_are_deterministic_and_have_an_alternative(env):
    cfg, graph, _ = env
    schedule = demo.make_schedule(2, cfg.tracking)
    a = demo.find_candidates(graph, schedule, seed=3)
    b = demo.find_candidates(graph, schedule, seed=3)
    assert a and [(c.nodes, c.alt_ratio) for c in a] == [(c.nodes, c.alt_ratio) for c in b]
    assert all(1.0 <= c.alt_ratio <= 1.6 for c in a)
    assert [c.alt_ratio for c in a] == sorted(c.alt_ratio for c in a)
    assert all(0 < c.dest_index < len(c.nodes) - 1 for c in a)


# ---------------------------------------------------------------------------- end to end


def test_run_demo_registers_processes_and_verifies(env, tmp_path):
    cfg, graph, _ = env
    result = run(env, tmp_path)
    out = tmp_path / "demo"
    assert (out / "synthetic_demo.mp4").is_file() and (out / "synthetic_demo_gps.csv").is_file()
    sidecar = json.loads((out / "synthetic_demo.json").read_text())
    assert sidecar["format"] == "road-recorder/1" and "SYNTHETIC" in sidecar["device"]
    assert sidecar["synthetic"] is True

    assert result.n_inserted == 2 and result.n_found == 2 and result.n_extra_tracks == 0
    assert result.routes_differ is True and result.candidates_tried >= 1
    example = json.loads((out / "demo_route.json").read_text())
    assert set(example) >= {"origin", "destination", "description", "synthetic"}
    assert example["synthetic"] is True and set(example["origin"]) == {"lat", "lon"}
    assert "SYNTHETIC" in example["description"] and example["routes_differ"] is True
    assert example["origin"] != example["destination"]

    conn = db.connect(cfg.db_path)
    try:
        (s,) = conn.execute("SELECT * FROM sessions").fetchall()
        assert (s["name"], s["role"], s["gps_source"]) == ("synthetic_demo", "demo", "synthetic")
        assert s["device"].startswith("SYNTHETIC") and s["active_job_id"]
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM potholes").fetchone()[0] == 2
        assert conn.execute("SELECT graph_sha256 FROM builds").fetchone()[0] == graph.sha256
    finally:
        conn.close()
    assert (cfg.uploads.dir / result.session_id / "synthetic_demo.mp4").is_file()

    # the suggested route really changes with avoidance
    from roadsense.routing.router import plan_routes

    conn = db.connect(cfg.db_path)
    try:
        o, d = result.origin, result.destination
        fewest = plan_routes(conn, graph, (o["lat"], o["lon"]), (d["lat"], d["lon"]), 1.0,
                             cfg.routing, cfg.mapping)  # fmt: skip
        shortest = plan_routes(conn, graph, (o["lat"], o["lon"]), (d["lat"], d["lon"]), 0.0,
                               cfg.routing, cfg.mapping)  # fmt: skip
    finally:
        conn.close()
    assert fewest["uses_synthetic_data"] is True
    by_label = lambda plan, label: next(r for r in plan["routes"] if label in r["labels"])  # noqa: E731
    assert by_label(fewest, "Strongest pothole avoidance")["geometry"] != by_label(shortest, "Shortest")["geometry"]
    assert by_label(fewest, "Shortest")["geometry"] == by_label(shortest, "Shortest")["geometry"]


def test_run_demo_is_deterministic_for_a_seed_and_replaces_the_old_session(env, tmp_path):
    cfg, _, _ = env
    first = run(env, tmp_path)
    out = tmp_path / "demo"

    def digest():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                if p.name != "demo_route.json"}  # fmt: skip

    before = digest()
    route_before = json.loads((out / "demo_route.json").read_text())
    second = run(env, tmp_path)
    assert digest() == before
    assert second.nodes == first.nodes and second.origin == first.origin
    assert second.session_id != first.session_id
    route_after = json.loads((out / "demo_route.json").read_text())
    assert {k: v for k, v in route_after.items() if k != "session_id"} == {
        k: v for k, v in route_before.items() if k != "session_id"
    }
    conn = db.connect(cfg.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1  # replaced, not added
    finally:
        conn.close()
    assert not (cfg.uploads.dir / first.session_id).exists()


def test_a_different_seed_gives_a_different_video(env, tmp_path):
    run(env, tmp_path, seed=1, process=False)
    a = (tmp_path / "demo" / "synthetic_demo.mp4").read_bytes()
    run(env, tmp_path, seed=2, process=False)
    assert (tmp_path / "demo" / "synthetic_demo.mp4").read_bytes() != a


def test_no_process_registers_without_running_the_detector(env, tmp_path):
    cfg, _, _ = env

    def never():
        raise AssertionError("detector must not be built")

    result = run(env, tmp_path, process=False, detector_factory=never)
    assert result.n_found is None and result.routes_differ is None
    assert any("not processed" in n for n in result.notes)
    assert (tmp_path / "demo" / "demo_route.json").is_file()
    conn = db.connect(cfg.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        conn.close()


def test_failure_to_show_different_routes_is_reported_honestly(env, tmp_path, monkeypatch):
    cfg, _, _ = env
    monkeypatch.setattr(demo, "routes_differ", lambda *args, **kwargs: (False, "forced: same route"))
    result = run(env, tmp_path, max_candidates=2)
    assert result.routes_differ is False and result.candidates_tried == 2
    assert any("NO candidate" in n for n in result.notes)
    assert json.loads((tmp_path / "demo" / "demo_route.json").read_text())["routes_differ"] is False
    conn = db.connect(cfg.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1  # the last attempt stays
    finally:
        conn.close()
    assert [p.name for p in cfg.uploads.dir.iterdir()] == [result.session_id]


def test_missing_graph_is_a_clear_error(tmp_path, dataset):
    cfg = make_api_cfg(tmp_path)
    with pytest.raises(demo.DemoError, match="road graph is needed"):
        run((cfg, None, dataset), tmp_path)


def test_malformed_graph_is_a_clear_error(tmp_path, dataset):
    cfg = make_api_cfg(tmp_path)
    cfg.routing.graph_path.parent.mkdir(parents=True)
    cfg.routing.graph_path.write_text("{broken")
    with pytest.raises(demo.DemoError, match="not valid JSON"):
        run((cfg, None, dataset), tmp_path)


def test_missing_dataset_is_reported_by_run_demo(env, tmp_path):
    cfg, _, _ = env
    with pytest.raises(demo.DemoError, match="validation images not found"):
        run(env, tmp_path, dataset_dir=tmp_path / "no_dataset")


def test_a_graph_without_a_suitable_drive_is_reported(env, tmp_path):
    cfg, _, _ = env
    # a 3-node graph is far too small for a drive of this length
    g = nx.MultiDiGraph()
    for i in range(3):
        g.add_node(i, **dict(zip(("lat", "lon"), xy(50.0 * i, 0.0))))
    for i in range(2):
        a, b = (g.nodes[i]["lat"], g.nodes[i]["lon"]), (g.nodes[i + 1]["lat"], g.nodes[i + 1]["lon"])
        g.add_edge(i, i + 1, 0, length_m=50.0, geometry=[a, b], name=None, highway="residential", speed_kph=30.0)
        g.add_edge(i + 1, i, 0, length_m=50.0, geometry=[b, a], name=None, highway="residential", speed_kph=30.0)
    save_graph(make_road_graph(g, {}), cfg.routing.graph_path)
    with pytest.raises(demo.DemoError, match="no suitable drive"):
        run(env, tmp_path)


def test_main_reports_errors_and_success(env, tmp_path, capsys):
    cfg, _, dataset = env
    import yaml

    config = tmp_path / "app.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(cfg.data_dir),
                "db_path": str(cfg.db_path),
                "uploads": {"dir": str(cfg.uploads.dir), "artifacts_dir": str(cfg.uploads.artifacts_dir)},
                "routing": {
                    "graph_path": str(cfg.routing.graph_path),
                    "max_detour_per_pothole_m": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    # default dataset location (<data_dir>/processed/rdd2022_pothole) does not exist here
    assert demo.main(["--config", str(config), "--n-potholes", "2"], detector_factory=FakeDetector) == 1
    assert "validation images not found" in capsys.readouterr().err
    target = cfg.data_dir / "processed" / "rdd2022_pothole"
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(dataset, target)
    code = demo.main(["--config", str(config), "--n-potholes", "2", "--seed", "3"],
                     detector_factory=FakeDetector)  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "SYNTHETIC DEMONSTRATION SESSION" in out
    assert "detector found 2 of 2" in out and "routes differ  True" in out
