import math

import pytest

from roadsense.ingest.gps import (
    GpsError,
    GpsFix,
    GpsLog,
    haversine_m,
    load_gps_csv,
    position_at,
)

HEADER = "timestamp_sec,lat,lon,accuracy_m,speed_mps,bearing_deg,altitude_m,utc_ms"
# Obviously fake coordinates (open ocean).
LAT0, LON0 = 10.0, 50.0
H3 = "timestamp_sec,lat,lon"
M_PER_DEG = 6_371_008.8 * math.pi / 180  # metres per degree along a meridian


@pytest.fixture
def write_csv(tmp_path):
    def _write(lines, header=HEADER, name="s_gps.csv"):
        path = tmp_path / name
        text = "\n".join(([header] if header is not None else []) + list(lines)) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def load(path, max_accuracy_m=25.0):
    return load_gps_csv(path, max_accuracy_m)


def codes(log):
    return [(i.line, i.code) for i in log.issues]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_recorder_header_parses_and_keeps_negative_preroll(write_csv):
    path = write_csv(
        [
            "-4.360,10.0000000,50.0000000,5.4,0.12,89.4,842.6,1792667934657",
            "0.640,10.0000100,50.0000100,4.6,0.24,297.0,842.9,1792667939656",
            "1.500,10.0000200,50.0000200,5.2,0.09,,837.2,1792667940500",
        ]
    )
    log = load(path)
    assert not log.issues
    assert log.n_rows_read == 3
    first = log.fixes[0]
    assert first == GpsFix(
        seq=0,
        t_video_s=-4.36,
        lat=10.0,
        lon=50.0,
        accuracy_m=5.4,
        speed_mps=0.12,
        bearing_deg=89.4,
        utc_ms=1792667934657,
        used=True,
    )
    assert [f.seq for f in log.fixes] == [0, 1, 2]
    assert log.fixes[2].bearing_deg is None  # blank -> None, no issue


def test_three_column_csv_is_valid(write_csv):
    path = write_csv(["0.0,10.0,50.0", "1.0,10.1,50.1"], header="timestamp_sec,lat,lon")
    log = load(path)
    assert not log.issues
    f = log.fixes[0]
    assert (f.accuracy_m, f.speed_mps, f.bearing_deg, f.utc_ms) == (None, None, None, None)
    assert f.used


def test_extra_columns_and_column_order_are_ignored(write_csv):
    path = write_csv(["1.5,99,10.0,50.0,extra"], header="timestamp_sec,junk,lat,lon,more")
    log = load(path)
    assert log.fixes[0].t_video_s == 1.5 and log.fixes[0].lat == 10.0


def test_utf8_bom_header_is_accepted(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbftimestamp_sec,lat,lon\n0,10,50\n")
    assert len(load(path).fixes) == 1


def test_missing_required_column_names_file_and_column(write_csv):
    path = write_csv(["0,10"], header="timestamp_sec,lat")
    with pytest.raises(GpsError) as exc:
        load(path)
    assert str(path) in str(exc.value) and "lon" in str(exc.value)


def test_empty_file_is_error(tmp_path):
    path = tmp_path / "empty_gps.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(GpsError, match="empty") as exc:
        load(path)
    assert str(path) in str(exc.value)


def test_missing_and_binary_file_are_errors(tmp_path):
    with pytest.raises(GpsError, match="cannot read"):
        load(tmp_path / "nope.csv")
    binary = tmp_path / "bin.csv"
    binary.write_bytes(b"\xff\xfe\x00\x80\x81\x82")
    with pytest.raises(GpsError) as exc:
        load(binary)
    assert str(binary) in str(exc.value)


def test_header_only_file_loads_with_no_fixes(write_csv):
    log = load(write_csv([]))
    assert log.fixes == () and log.n_rows_read == 0
    assert log.summary.n_used == 0 and log.summary.t_first_s is None


def test_bad_max_accuracy_is_rejected(write_csv):
    path = write_csv(["0,10,50"])
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_accuracy_m"):
            load(path, bad)


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (",10.0,50.0,5,1,1,1,1", "timestamp_missing"),
        ("abc,10.0,50.0,5,1,1,1,1", "timestamp_invalid"),
        ("nan,10.0,50.0,5,1,1,1,1", "timestamp_invalid"),
        ("inf,10.0,50.0,5,1,1,1,1", "timestamp_invalid"),
        ("1,,50.0,5,1,1,1,1", "lat_missing"),
        ("1,north,50.0,5,1,1,1,1", "lat_invalid"),
        ("1,nan,50.0,5,1,1,1,1", "lat_invalid"),
        ("1,10.0,,5,1,1,1,1", "lon_missing"),
        ("1,10.0,east,5,1,1,1,1", "lon_invalid"),
        ("1,10.0,-inf,5,1,1,1,1", "lon_invalid"),
        ("1,90.5,50.0,5,1,1,1,1", "lat_out_of_range"),
        ("1,-91,50.0,5,1,1,1,1", "lat_out_of_range"),
        ("1,10.0,180.5,5,1,1,1,1", "lon_out_of_range"),
        ("1,10.0,-181,5,1,1,1,1", "lon_out_of_range"),
        ("1,0,0,5,1,1,1,1", "null_island"),
        ("1,0.0,0.0,,,,,", "null_island"),
    ],
)
def test_every_rejection_rule_records_issue_and_load_succeeds(write_csv, row, code):
    path = write_csv(["0.0,10.0,50.0,5,1,1,1,1", row, "2.0,10.0,50.1,5,1,1,1,1"])
    log = load(path)
    assert len(log.fixes) == 2
    assert log.n_rows_read == 3
    assert len(log.issues) == 1
    issue = log.issues[0]
    assert issue.line == 3  # header is line 1
    assert issue.code == code
    assert issue.row_dropped
    assert issue.raw == row
    assert log.summary.rejected_by_reason == {code: 1}


def test_boundary_coordinates_are_valid(write_csv):
    path = write_csv(["0,90,180", "1,-90,-180", "2,0,1", "3,1,0"], header=H3)
    log = load(path)
    assert not log.issues and len(log.fixes) == 4


def test_line_numbers_count_physical_lines_including_blank(write_csv):
    path = write_csv(["0,10,50", "", "1,bad,50"], header=H3)
    log = load(path)
    assert codes(log) == [(4, "lat_invalid")]
    assert log.n_rows_read == 2  # the blank line is not a row


def test_blank_optional_field_is_none_without_issue(write_csv):
    log = load(write_csv(["0,10,50,,,,,"]))
    f = log.fixes[0]
    assert (f.accuracy_m, f.speed_mps, f.bearing_deg, f.utc_ms) == (None, None, None, None)
    assert not log.issues


@pytest.mark.parametrize(
    ("row", "field", "code"),
    [
        ("0,10,50,abc,1,90,0,5", "accuracy_m", "accuracy_m_invalid"),
        ("0,10,50,nan,1,90,0,5", "accuracy_m", "accuracy_m_invalid"),
        ("0,10,50,-1,1,90,0,5", "accuracy_m", "accuracy_m_invalid"),
        ("0,10,50,5,xyz,90,0,5", "speed_mps", "speed_mps_invalid"),
        ("0,10,50,5,-0.5,90,0,5", "speed_mps", "speed_mps_invalid"),
        ("0,10,50,5,inf,90,0,5", "speed_mps", "speed_mps_invalid"),
        ("0,10,50,5,1,361,0,5", "bearing_deg", "bearing_deg_invalid"),
        ("0,10,50,5,1,-1,0,5", "bearing_deg", "bearing_deg_invalid"),
        ("0,10,50,5,1,north,0,5", "bearing_deg", "bearing_deg_invalid"),
        ("0,10,50,5,1,90,0,12.5", "utc_ms", "utc_ms_invalid"),
        ("0,10,50,5,1,90,0,soon", "utc_ms", "utc_ms_invalid"),
    ],
)
def test_bad_optional_field_becomes_none_with_issue_and_fix_kept(write_csv, row, field, code):
    log = load(write_csv([row]))
    assert len(log.fixes) == 1
    assert getattr(log.fixes[0], field) is None
    assert len(log.issues) == 1
    issue = log.issues[0]
    assert (issue.line, issue.code, issue.row_dropped) == (2, code, False)
    # the other fields survive
    assert log.fixes[0].lat == 10.0 and log.fixes[0].lon == 50.0
    assert log.summary.rejected_by_reason == {}


def test_bearing_bounds_are_inclusive(write_csv):
    log = load(write_csv(["0,10,50,5,1,0,0,5", "1,10,50,5,1,360,0,5"]))
    assert not log.issues
    assert [f.bearing_deg for f in log.fixes] == [0.0, 360.0]


def test_utc_ms_accepts_integral_float_notation(write_csv):
    log = load(write_csv(["0,10,50,5,1,1,0,1.792667934657e12"]))
    assert log.fixes[0].utc_ms == 1792667934657 and not log.issues


def test_short_row_is_kept_and_flagged(write_csv):
    log = load(write_csv(["0,10,50,5"]))
    assert len(log.fixes) == 1 and log.fixes[0].accuracy_m == 5.0
    assert codes(log) == [(2, "column_count")]
    assert not log.issues[0].row_dropped


def test_unsorted_input_is_sorted_and_flagged(write_csv):
    path = write_csv(["2,10.2,50", "0,10.0,50", "1,10.1,50"], header=H3)
    log = load(path)
    assert [f.t_video_s for f in log.fixes] == [0.0, 1.0, 2.0]
    assert [f.lat for f in log.fixes] == [10.0, 10.1, 10.2]
    assert [f.seq for f in log.fixes] == [0, 1, 2]
    assert [i.code for i in log.issues] == ["unsorted"]
    assert not log.issues[0].row_dropped


def test_sorted_input_has_no_unsorted_issue(write_csv):
    assert not load(write_csv(["0,10,50", "1,10,50.1"], header=H3)).issues


def test_duplicate_timestamp_keeps_more_accurate_fix(write_csv):
    path = write_csv(["1,10.0,50.0,9,1,1,0,5", "1,10.5,50.5,3,1,1,0,5", "2,10.6,50.6,5,1,1,0,5"])
    log = load(path)
    assert len(log.fixes) == 2
    assert log.fixes[0].lat == 10.5 and log.fixes[0].accuracy_m == 3.0
    assert [(i.line, i.code, i.row_dropped) for i in log.issues] == [(2, "duplicate_timestamp", True)]
    assert log.summary.rejected_by_reason == {"duplicate_timestamp": 1}
    assert [f.seq for f in log.fixes] == [0, 1]


def test_duplicate_with_unknown_accuracy_counts_as_worst(write_csv):
    log = load(write_csv(["1,10.0,50.0", "1,10.5,50.5,40"], header="timestamp_sec,lat,lon,accuracy_m"))
    assert log.fixes[0].lat == 10.5  # known (even if poor) beats unknown
    log2 = load(write_csv(["1,10.0,50.0,7", "1,10.5,50.5,"], header="timestamp_sec,lat,lon,accuracy_m"))
    assert log2.fixes[0].lat == 10.0


def test_duplicate_tie_keeps_first_in_file(write_csv):
    log = load(write_csv(["1,10.0,50.0,5", "1,10.5,50.5,5"], header="timestamp_sec,lat,lon,accuracy_m"))
    assert log.fixes[0].lat == 10.0


def test_accuracy_filter_boundary(write_csv):
    path = write_csv(
        ["0,10,50,24.99", "1,10,50.1,25.0", "2,10,50.2,25.01"],
        header="timestamp_sec,lat,lon,accuracy_m",
    )
    log = load(path, max_accuracy_m=25.0)
    assert [f.used for f in log.fixes] == [True, True, False]
    assert log.summary.n_fixes == 3 and log.summary.n_used == 2


def test_unknown_accuracy_stays_usable_and_is_counted(write_csv):
    path = write_csv(["0,10,50,", "1,10,50.1,30", "2,10,50.2,abc"], header="timestamp_sec,lat,lon,accuracy_m")
    log = load(path, max_accuracy_m=25.0)
    assert [f.used for f in log.fixes] == [True, False, True]
    assert log.summary.n_unknown_accuracy == 2
    assert log.summary.median_accuracy_m is None  # no used fix has a known accuracy


def test_summary_numbers(write_csv):
    lines = [
        "-2.0,10.000,50.0,4,0,,0,1",
        "-1.0,10.001,50.0,6,0,,0,1",  # 111.2 m north
        "0.5,10.002,50.0,30,0,,0,1",  # rejected (accuracy 30)
        "2.5,10.003,50.0,10,0,,0,1",
        "bad,10.0,50.0,1,0,,0,1",  # row rejected
    ]
    path = write_csv(lines)
    log = load(path, max_accuracy_m=25.0)
    s = log.summary
    assert s.n_rows_read == 5
    assert s.n_fixes == 4 and s.n_used == 3
    assert s.rejected_by_reason == {"timestamp_invalid": 1}
    assert (s.t_first_s, s.t_last_s) == (-2.0, 2.5)
    assert s.median_accuracy_m == 6.0  # used accuracies 4, 6, 10
    assert s.max_gap_s == pytest.approx(3.5)  # -1.0 -> 2.5 skips the rejected fix
    expected = 0.003 * M_PER_DEG  # used fixes 10.000 -> 10.001 -> 10.003, straight north
    assert s.path_length_m == pytest.approx(expected, rel=0.005)


def test_summary_with_single_used_fix(write_csv):
    s = load(write_csv(["0,10,50,5"], header="timestamp_sec,lat,lon,accuracy_m")).summary
    assert s.max_gap_s is None and s.path_length_m == 0.0 and s.median_accuracy_m == 5.0


def test_haversine_matches_hand_computed_values():
    assert haversine_m(10, 50, 10, 50) == 0.0
    assert haversine_m(10.0, 50.0, 10.001, 50.0) == pytest.approx(0.001 * M_PER_DEG, rel=0.005)
    # one degree of longitude at the equator equals one degree of meridian
    assert haversine_m(0.0, 50.0, 0.0, 51.0) == pytest.approx(M_PER_DEG, rel=0.005)
    # at 60 degrees north a degree of longitude is half as long
    assert haversine_m(60.0, 20.0, 60.0, 21.0) == pytest.approx(M_PER_DEG * 0.5, rel=0.005)
    # antimeridian: 0.2 degrees apart, not 359.8
    assert haversine_m(0.0, 179.9, 0.0, -179.9) == pytest.approx(0.2 * M_PER_DEG, rel=0.005)


def test_gps_log_rejects_unsorted_fixes():
    a = GpsFix(0, 1.0, 10, 50, None, None, None, None, True)
    b = GpsFix(1, 1.0, 10, 50, None, None, None, None, True)
    with pytest.raises(ValueError, match="strictly increasing"):
        GpsLog(fixes=(a, b), issues=(), n_rows_read=2)


# ---------------------------------------------------------------------------
# position_at
# ---------------------------------------------------------------------------


def make_log(*rows):
    """rows: (t, lat, lon, accuracy, speed, bearing, used) - trailing items optional."""
    fixes = []
    for seq, row in enumerate(rows):
        t, lat, lon, *rest = row
        acc, speed, bearing, used = (list(rest) + [None, None, None, True][len(rest) :])[:4]
        fixes.append(GpsFix(seq, t, lat, lon, acc, speed, bearing, None, used))
    return GpsLog(fixes=tuple(fixes), issues=(), n_rows_read=len(fixes))


GAP, EXTRA = 3.0, 1.0


def at(log, t):
    return position_at(log, t, GAP, EXTRA)


def test_exact_fix_time_returns_that_fix():
    log = make_log((0.0, 10.0, 50.0, 5.0, 2.0, 90.0), (1.0, 10.1, 50.1, 7.0, 4.0, 100.0))
    p = at(log, 1.0)
    assert (p.lat, p.lon, p.accuracy_m, p.speed_mps, p.bearing_deg) == (10.1, 50.1, 7.0, 4.0, 100.0)
    assert not p.extrapolated and p.gap_s == 0.0


def test_midpoint_interpolation_hand_computed():
    log = make_log((0.0, 10.0, 50.0, 5.0, 2.0, 80.0), (2.0, 10.4, 50.2, 9.0, 6.0, 100.0))
    p = at(log, 1.0)
    assert p.lat == pytest.approx(10.2)
    assert p.lon == pytest.approx(50.1)
    assert p.speed_mps == pytest.approx(4.0)
    assert p.bearing_deg == pytest.approx(90.0)
    assert p.accuracy_m == 9.0  # the worse of 5 and 9
    assert p.gap_s == 2.0 and not p.extrapolated
    q = at(log, 0.5)  # quarter of the way
    assert q.lat == pytest.approx(10.1) and q.lon == pytest.approx(50.05)


def test_time_inside_preroll_interpolates_between_negative_fixes():
    log = make_log((-4.0, 10.0, 50.0, 5.0), (-3.0, 10.1, 50.0, 5.0), (0.5, 10.2, 50.0, 5.0))
    p = at(log, -3.5)
    assert p.lat == pytest.approx(10.05) and not p.extrapolated


def test_before_first_and_after_last_within_window_clamp():
    log = make_log((1.0, 10.0, 50.0, 5.0, 1.0, 10.0), (2.0, 10.1, 50.1, 6.0, 2.0, 20.0))
    before = at(log, 0.0)  # exactly max_extrapolation_s before
    assert (before.lat, before.lon, before.accuracy_m) == (10.0, 50.0, 5.0)
    assert before.extrapolated and before.gap_s == 0.0
    after = at(log, 3.0)
    assert (after.lat, after.lon) == (10.1, 50.1) and after.extrapolated


def test_beyond_extrapolation_window_is_none():
    log = make_log((1.0, 10.0, 50.0, 5.0), (2.0, 10.1, 50.1, 5.0))
    assert at(log, 1.0 - 1.001) is None
    assert at(log, 2.0 + 1.001) is None


def test_zero_extrapolation_allows_exact_ends_only():
    log = make_log((1.0, 10.0, 50.0), (2.0, 10.1, 50.1))
    assert position_at(log, 0.999, GAP, 0.0) is None
    assert position_at(log, 1.0, GAP, 0.0) is not None
    assert position_at(log, 2.001, GAP, 0.0) is None


def test_gap_larger_than_max_gap_is_none():
    log = make_log((0.0, 10.0, 50.0), (3.5, 10.1, 50.1))
    assert position_at(log, 1.0, 3.0, EXTRA) is None
    assert position_at(log, 1.0, 3.5, EXTRA) is not None  # gap equal to the limit is fine
    assert position_at(log, 0.0, 3.0, EXTRA) is not None  # exact fixes still answer


def test_rejected_fixes_are_skipped():
    log = make_log(
        (0.0, 10.0, 50.0, 5.0),
        (1.0, 11.0, 51.0, 99.0, None, None, False),  # rejected: must not pull the answer
        (2.0, 10.2, 50.2, 5.0),
    )
    p = at(log, 1.0)
    assert p.lat == pytest.approx(10.1) and p.lon == pytest.approx(50.1)
    assert p.gap_s == 2.0


def test_rejected_fixes_widen_gap_past_limit():
    log = make_log(
        (0.0, 10.0, 50.0, 5.0),
        (2.0, 11.0, 51.0, 99.0, None, None, False),
        (4.0, 10.2, 50.2, 5.0),
    )
    assert position_at(log, 1.0, 3.0, EXTRA) is None  # remaining gap is 4 s
    assert position_at(log, 1.0, 4.0, EXTRA) is not None


def test_rejected_fix_exact_time_is_not_returned():
    log = make_log((0.0, 10.0, 50.0), (1.0, 11.0, 51.0, 99.0, None, None, False), (2.0, 10.2, 50.2))
    assert at(log, 1.0).lat == pytest.approx(10.1)


def test_no_usable_fixes_is_none():
    assert at(make_log(), 0.0) is None
    only_rejected = make_log((0.0, 10.0, 50.0, 99.0, None, None, False))
    assert at(only_rejected, 0.0) is None


def test_single_usable_fix_clamps_within_window():
    log = make_log((5.0, 10.0, 50.0, 4.0))
    assert at(log, 5.5).extrapolated
    assert at(log, 7.0) is None


def test_bearing_wraps_through_north_not_south():
    log = make_log((0.0, 10.0, 50.0, 5.0, 1.0, 350.0), (1.0, 10.1, 50.0, 5.0, 1.0, 10.0))
    p = at(log, 0.5)
    assert min(p.bearing_deg, 360.0 - p.bearing_deg) == pytest.approx(0.0, abs=1e-9)
    p2 = at(log, 0.25)  # 350 + 0.25*20 = 355
    assert p2.bearing_deg == pytest.approx(355.0)
    # and the reverse direction
    back = make_log((0.0, 10.0, 50.0, 5.0, 1.0, 10.0), (1.0, 10.1, 50.0, 5.0, 1.0, 350.0))
    assert at(back, 0.75).bearing_deg == pytest.approx(355.0)


def test_bearing_falls_back_to_course_over_ground():
    # 0.001 degrees north is about 111 m, so the course is due north
    north = make_log((0.0, 10.0, 50.0, 5.0, 1.0, None), (1.0, 10.001, 50.0, 5.0, 1.0, 90.0))
    b = at(north, 0.5).bearing_deg
    assert min(b, 360.0 - b) == pytest.approx(0.0, abs=1e-6)
    east = make_log((0.0, 0.5, 50.0, 5.0, 1.0, 10.0), (1.0, 0.5, 50.001, 5.0, 1.0, None))
    assert at(east, 0.5).bearing_deg == pytest.approx(90.0, abs=0.01)
    south_west = make_log((0.0, 10.001, 50.001), (1.0, 10.0, 50.0))
    assert 180.0 < at(south_west, 0.5).bearing_deg < 270.0


def test_bearing_is_none_when_fixes_are_within_one_metre():
    # 0.000005 degrees is about 0.56 m
    log = make_log((0.0, 10.0, 50.0, 5.0, 0.0, None), (1.0, 10.000005, 50.0, 5.0, 0.0, None))
    assert at(log, 0.5).bearing_deg is None
    far = make_log((0.0, 10.0, 50.0), (1.0, 10.00002, 50.0))  # about 2.2 m
    assert at(far, 0.5).bearing_deg is not None


def test_speed_is_none_if_either_is_none():
    log = make_log((0.0, 10.0, 50.0, 5.0, 1.0, 0.0), (1.0, 10.1, 50.0, 5.0, None, 0.0))
    assert at(log, 0.5).speed_mps is None


def test_accuracy_is_worse_of_two_and_none_only_if_both_none():
    a = make_log((0.0, 10.0, 50.0, 8.0), (1.0, 10.1, 50.0, 3.0))
    assert at(a, 0.5).accuracy_m == 8.0
    b = make_log((0.0, 10.0, 50.0, None), (1.0, 10.1, 50.0, 3.0))
    assert at(b, 0.5).accuracy_m == 3.0
    c = make_log((0.0, 10.0, 50.0), (1.0, 10.1, 50.0))
    assert at(c, 0.5).accuracy_m is None


def test_interpolation_across_antimeridian_takes_short_way():
    log = make_log((0.0, 10.0, 179.9), (1.0, 10.0, -179.9))
    p = at(log, 0.5)
    assert abs(abs(p.lon) - 180.0) < 1e-9


def test_invalid_query_arguments_raise():
    log = make_log((0.0, 10.0, 50.0))
    with pytest.raises(ValueError):
        position_at(log, float("nan"), GAP, EXTRA)
    with pytest.raises(ValueError):
        position_at(log, 0.0, 0.0, EXTRA)
    with pytest.raises(ValueError):
        position_at(log, 0.0, GAP, -1.0)


def test_many_queries_use_precomputed_index():
    n = 2000
    log = make_log(*[(float(i), 10.0 + i * 1e-5, 50.0, 5.0) for i in range(n)])
    first = log._used_times
    for k in range(5000):
        p = at(log, (k % (n - 1)) + 0.5)
        assert p is not None
    assert log._used_times is first  # computed once, not per query
    assert at(log, 100.5).lat == pytest.approx(10.0 + 100.5 * 1e-5)
