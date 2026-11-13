"""Regression tests: GPS coverage is limited to the interval the video was running."""

import json

import pytest

from geo_fixtures import add_fixes, add_session, edge_point, make_grid, xy
from roadsense.config import IngestConfig, MappingConfig
from roadsense.geo.segments import clip_fixes_to_video, rebuild_map
from roadsense.storage import db

MAPPING, INGEST = MappingConfig(), IngestConfig()  # snap 30 m, gap 3 s

GAP = 3.0


def fx(t, lat=None, lon=None):
    """A fix at video time ``t``; position defaults to a value that identifies the time."""
    return (t, float(t) if lat is None else lat, 2.0 * t if lon is None else lon)


# --------------------------------------------------------------------------- clip_fixes_to_video


def test_fixes_inside_the_video_are_kept_unchanged():
    fixes = [fx(0), fx(1), fx(2.5), fx(5)]
    assert clip_fixes_to_video(fixes, 5.0, GAP) == fixes


def test_the_position_at_the_start_boundary_is_interpolated():
    out = clip_fixes_to_video([(-1.0, 0.0, 0.0), (1.0, 2.0, 4.0), (2.0, 3.0, 6.0)], 10.0, GAP)
    assert out == [(0.0, 1.0, 2.0), (1.0, 2.0, 4.0), (2.0, 3.0, 6.0)]


def test_the_position_at_the_end_boundary_is_interpolated():
    out = clip_fixes_to_video([(8.0, 0.0, 0.0), (9.0, 1.0, 2.0), (11.0, 3.0, 6.0)], 10.0, GAP)
    assert out == [(8.0, 0.0, 0.0), (9.0, 1.0, 2.0), (10.0, 2.0, 4.0)]


def test_fixes_exactly_at_the_boundaries_are_not_duplicated_or_extended():
    fixes = [fx(-2), fx(0), fx(1), fx(9), fx(10), fx(12)]
    assert clip_fixes_to_video(fixes, 10.0, GAP) == [fx(0), fx(1), fx(9), fx(10)]


def test_no_fix_inside_the_video_but_trace_spans_it():
    out = clip_fixes_to_video([(-1.0, 0.0, 0.0), (3.0, 4.0, 8.0)], 2.0, 10.0)
    assert out == [(0.0, 1.0, 2.0), (2.0, 3.0, 6.0)]


def test_no_fix_inside_the_video_and_trace_ends_before_or_starts_after_it():
    assert clip_fixes_to_video([fx(-9), fx(-5)], 10.0, GAP) == []
    assert clip_fixes_to_video([fx(11), fx(15)], 10.0, GAP) == []
    assert clip_fixes_to_video([], 10.0, GAP) == []


def test_a_gap_longer_than_the_limit_across_the_start_is_not_bridged():
    # 20 s between the last fix before the video and the first inside it
    out = clip_fixes_to_video([(-10.0, 0.0, 0.0), (10.0, 2.0, 4.0), (11.0, 3.0, 6.0)], 30.0, GAP)
    assert out == [(10.0, 2.0, 4.0), (11.0, 3.0, 6.0)]


def test_a_gap_longer_than_the_limit_across_the_end_is_not_bridged():
    out = clip_fixes_to_video([(8.0, 0.0, 0.0), (9.0, 1.0, 2.0), (29.0, 5.0, 9.0)], 10.0, GAP)
    assert out == [(8.0, 0.0, 0.0), (9.0, 1.0, 2.0)]


def test_a_long_gap_over_a_trace_spanning_the_whole_video_leaves_nothing():
    assert clip_fixes_to_video([(-10.0, 0.0, 0.0), (30.0, 4.0, 8.0)], 10.0, GAP) == []


def test_the_gap_limit_is_inclusive():
    out = clip_fixes_to_video([(-1.0, 0.0, 0.0), (2.0, 3.0, 6.0)], 10.0, 3.0)
    assert out == [(0.0, 1.0, 2.0), (2.0, 3.0, 6.0)]


def test_unknown_duration_clips_only_the_start():
    out = clip_fixes_to_video([(-5.0, 0.0, 0.0), (5.0, 10.0, 20.0), (100.0, 1.0, 1.0)], None, 20.0)
    assert out == [(0.0, 5.0, 10.0), (5.0, 10.0, 20.0), (100.0, 1.0, 1.0)]
    assert clip_fixes_to_video([fx(-3), fx(-1)], None, GAP) == []


def test_a_single_fix():
    assert clip_fixes_to_video([fx(3)], 10.0, GAP) == [fx(3)]
    assert clip_fixes_to_video([fx(0)], 10.0, GAP) == [fx(0)]
    assert clip_fixes_to_video([fx(10)], 10.0, GAP) == [fx(10)]
    assert clip_fixes_to_video([fx(-1)], 10.0, GAP) == []
    assert clip_fixes_to_video([fx(11)], 10.0, GAP) == []
    assert clip_fixes_to_video([fx(3)], None, GAP) == [fx(3)]


# --------------------------------------------------------------------------- map build


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "clip.sqlite")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def graph():
    return make_grid()


def drive(conn, sid, t_start, t_end, *, x0=120.0, speed=2.0, dt=1.0, duration=None, summary=None):
    """A session whose car is at x = x0 + speed * t along the bottom row, fixes every ``dt`` s.

    With the defaults the video interval t in [0, 20] is x = 120 .. 160, inside the street
    from node 1 (x = 100) to node 2 (x = 200), so no junction effects blur the numbers.
    """
    add_session(conn, sid)
    times = []
    t = t_start
    while t <= t_end + 1e-9:
        times.append(t)
        t += dt
    add_fixes(conn, sid, [xy(x0 + speed * t, 0.0) for t in times], dt=dt, t0=t_start)
    if duration is not None:
        conn.execute("UPDATE sessions SET duration_s = ? WHERE id = ?", (duration, sid))
    if summary is not None:
        conn.execute("UPDATE jobs SET summary_json = ? WHERE session_id = ?", (json.dumps(summary), sid))
    conn.commit()


def surveyed(conn, graph):
    return rebuild_map(conn, graph, MAPPING, INGEST)["surveyed_m"]


def test_a_trace_longer_than_the_video_surveys_only_the_road_filmed(conn, graph):
    # GPS runs from 30 s before the 20 s video until 30 s after it (x = 60 .. 220, 160 m), but only
    # t in [0, 20] was filmed: x = 120 .. 160 = 40 m, plus 5 m either side of the ends.
    drive(conn, "s", -30.0, 50.0, duration=20.0)
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)
    covered = conn.execute("SELECT edge_u, edge_v, start_m, end_m FROM edge_coverage").fetchall()
    assert [(r["edge_u"], r["edge_v"]) for r in covered] == [(1, 2)]  # no other road at all
    assert covered[0]["start_m"] == pytest.approx(15.0, abs=0.5)
    assert covered[0]["end_m"] == pytest.approx(65.0, abs=0.5)


def test_the_boundaries_are_interpolated_when_no_fix_falls_on_them(conn, graph):
    # fixes at half seconds, so none lies on t = 0 or t = 20
    drive(conn, "s", -29.5, 49.5, duration=20.0)
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)


def test_without_any_duration_only_the_start_is_clipped(conn, graph):
    drive(conn, "s", -30.0, 30.0)  # x = 60 .. 180; the part before the first frame is cut
    # t in [0, 30]: x = 120 .. 180 -> 60 m + 10 m (the end of the video is unknown)
    assert surveyed(conn, graph) == pytest.approx(70.0, abs=0.5)


def test_the_job_report_supplies_the_duration_when_the_session_row_has_none(conn, graph):
    drive(conn, "s", -30.0, 50.0, summary={"video": {"duration_s": 20.0}})
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)


def test_the_session_duration_wins_over_the_job_report(conn, graph):
    drive(conn, "s", -30.0, 50.0, duration=20.0, summary={"video": {"duration_s": 60.0}})
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)


def test_a_trace_inside_the_video_is_not_shortened(conn, graph):
    drive(conn, "s", 0.0, 20.0, duration=20.0)
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)
    # and a longer video than trace: the trace is all there is
    conn.execute("UPDATE sessions SET duration_s = 100.0")
    assert surveyed(conn, graph) == pytest.approx(50.0, abs=0.5)


def test_a_dropout_across_the_start_of_the_video_is_not_bridged(conn, graph):
    # last fix 10 s before the video, next one 10 s into it: 20 s > the 3 s limit
    add_session(conn, "s")
    add_fixes(conn, "s", [xy(10, 0), xy(130, 0)], dt=20.0, t0=-10.0)
    conn.execute("UPDATE sessions SET duration_s = 20.0")
    conn.commit()
    # only the 10 m around the fix inside the video is surveyed, nothing on the road before it
    assert surveyed(conn, graph) == pytest.approx(10.0, abs=0.5)


def test_gps_gaps_inside_the_video_are_still_not_bridged(conn, graph):
    pts = [edge_point(graph, 0, 1, 0, 20), edge_point(graph, 0, 1, 0, 80)]
    add_session(conn, "close")
    add_fixes(conn, "close", pts, dt=1.0)
    add_session(conn, "gap")
    add_fixes(conn, "gap", pts, dt=10.0)
    conn.execute("UPDATE sessions SET duration_s = 60.0")
    conn.commit()
    rebuild_map(conn, graph, MAPPING, INGEST)

    def covered(sid):
        rows = conn.execute(
            "SELECT start_m, end_m FROM edge_coverage WHERE session_id = ? ORDER BY start_m", (sid,)
        ).fetchall()
        return sum(r["end_m"] - r["start_m"] for r in rows), len(rows)

    assert covered("close") == (pytest.approx(70.0, abs=0.5), 1)  # 15 .. 85 m
    assert covered("gap") == (pytest.approx(20.0, abs=0.5), 2)  # only around each fix


def test_a_video_that_the_trace_never_overlaps_surveys_nothing(conn, graph):
    drive(conn, "s", -50.0, -10.0, duration=20.0)  # all fixes before the video
    assert surveyed(conn, graph) == 0.0
    assert conn.execute("SELECT COUNT(*) FROM edge_coverage").fetchone()[0] == 0


def test_a_fix_after_the_end_followed_by_fixes_inside_does_not_drop_them():
    # one out-of-order row must not hide the later fixes that lie inside the video
    out = clip_fixes_to_video([fx(5), fx(11), fx(6), fx(7)], 10.0, GAP)
    assert out == [fx(5), fx(6), fx(7)]


def test_an_out_of_order_fix_after_the_end_adds_one_boundary_point_only():
    out = clip_fixes_to_video([fx(8), fx(12), fx(12.5), fx(9), fx(9.5)], 10.0, 5.0)
    # one crossing 8 -> 12 (boundary at 10); later fixes after the end add nothing more
    assert out.count((10.0, 10.0, 20.0)) == 1
    assert [f for f in out if f[0] != 10.0] == [fx(8), fx(9), fx(9.5)]
