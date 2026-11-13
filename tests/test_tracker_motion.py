"""Motion-aware matching: predicted boxes, source-frame ageing, and unchanged slow behaviour."""

import random

import pytest

from roadsense.detection.detector import Detection
from roadsense.tracking.tracker import IouTracker, MotionTracker, iou


def box(x1, y1, x2, y2, conf=0.9):
    return Detection(x1, y1, x2, y2, conf, 0, "pothole")


def vertical_box(y, x=100, w=40, h=20):
    return box(x, y, x + w, y + h)


def run(tracker, frames, stride=1):
    """``frames``: detection lists of consecutive analysed frames, ``stride`` source frames apart."""
    ended = []
    for i, dets in enumerate(frames):
        ended += tracker.update(dets, i * stride, i * stride / 30.0)
    return ended + tracker.flush()


def last_box_matches(a, b, threshold=0.2):
    """What the previous tracker asked: does ``b`` overlap the LAST box ``a`` enough?"""
    value = iou(a, b)
    return value > 0.0 and value >= threshold


def constant_velocity_frames(n, step=15, y0=10, h=30):
    return [[vertical_box(y0 + step * k, h=h)] for k in range(n)]


# --------------------------------------------------------------------------- fast objects


def test_box_moving_faster_than_its_height_is_one_track():
    # A 20 px tall box that accelerates down the image. From step 6 on it moves more
    # than its own height between analysed frames, so the previous last-box IoU test
    # gives 0 there and would start a new one-frame track every time.
    ys = [10 + 1.5 * k * k for k in range(11)]
    jumps = [b - a for a, b in zip(ys, ys[1:])]
    assert max(jumps) > 20
    assert iou(vertical_box(ys[-2]), vertical_box(ys[-1])) == 0.0
    assert not last_box_matches(vertical_box(ys[-2]), vertical_box(ys[-1]))

    ended = run(MotionTracker(0.2, 15, 3), [[vertical_box(y)] for y in ys])
    assert len(ended) == 1
    assert ended[0].n_hits == len(ys) and ended[0].confirmed and ended[0].track_id == 1


def test_track_survives_misses_and_continues_at_predicted_position():
    t = MotionTracker(0.2, 15, 3)
    for k, dets in enumerate(constant_velocity_frames(3)):  # y = 10, 25, 40: 15 px per frame
        t.update(dets, k, k * 0.1)
    for k in range(3, 6):  # three missed frames, fewer than the limit
        assert t.update([], k, k * 0.1) == []
    reappeared = vertical_box(40 + 4 * 15, h=30)  # it kept moving
    assert not last_box_matches(vertical_box(40, h=30), reappeared)  # previous test: IoU 0
    assert t.update([reappeared], 6, 0.6) == []
    (live,) = t.live_tracks
    assert live.track_id == 1 and live.n_hits == 4 and live.last.detection == reappeared


def test_track_ends_when_the_gap_exceeds_the_age_limit():
    t = MotionTracker(0.2, 3, 3)
    for k, dets in enumerate(constant_velocity_frames(3)):
        t.update(dets, k, k * 0.1)  # the last point is at source frame 2
    assert [t.update([], k, k * 0.1) for k in (3, 4, 5)] == [[], [], []]  # 1 to 3 frames old
    ended = t.update([], 6, 0.6)  # 4 source frames since the last point > 3
    assert [tr.track_id for tr in ended] == [1] and ended[0].confirmed
    # an object where the old track would have been predicted is a NEW track now
    t.update([vertical_box(10 + 15 * 7, h=30)], 7, 0.7)
    assert [tr.track_id for tr in t.live_tracks] == [2]


def test_objects_appearing_one_after_another_in_different_places_are_not_merged():
    a = [[vertical_box(10 + 12 * k, x=50, h=30)] for k in range(5)]
    b = [[vertical_box(200 + 12 * k, x=400, h=30)] for k in range(5)]
    # B shows up within the age limit of A, but far from where A is predicted to be
    ended = run(MotionTracker(0.2, 15, 1), a + [[]] * 4 + b)
    assert [(tr.track_id, tr.n_hits) for tr in ended] == [(1, 5), (2, 5)]
    assert ended[0].last.detection.x1 == 50 and ended[1].first.detection.x1 == 400


def test_two_objects_visible_together_keep_two_ids_when_their_paths_cross():
    # A moves right and B moves left along one row and pass through each other.
    # Detections are listed left to right, so after the crossing the last-box IoU is
    # a tie that would swap the identities; the velocity prediction keeps them.
    frames = []
    for k in range(21):
        xs = sorted([10.0 * k, 200.0 - 10.0 * k])
        frames.append([box(x, 100, x + 40, 140) for x in xs])
    ended = run(MotionTracker(0.2, 15, 3), frames)
    assert len(ended) == 2 and all(tr.n_hits == 21 for tr in ended)
    paths = [[p.detection.x1 for p in tr.points] for tr in ended]
    rising, falling = sorted(paths, key=lambda xs: xs[-1], reverse=True)
    assert rising == sorted(rising) and falling == sorted(falling, reverse=True)
    assert (rising[-1], falling[-1]) == (200, 0)


# --------------------------------------------------------------------------- slow objects


def old_last_box_tracker(frames, iou_match, max_age):
    """The previous algorithm (last-box IoU, age in analysed frames) as a test oracle.

    Returns the points of every track as a list of (frame, x1, y1, x2, y2).
    """
    live, done = [], []  # live: [points, misses]
    for f, dets in enumerate(frames):
        pairs = sorted(
            (-iou(tr[0][-1][1], d), ti, di)
            for ti, tr in enumerate(live)
            for di, d in enumerate(dets)
            if iou(tr[0][-1][1], d) > 0 and iou(tr[0][-1][1], d) >= iou_match
        )
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            live[ti][0].append((f, dets[di]))
            live[ti][1] = 0
        keep = []
        for ti, tr in enumerate(live):
            if ti not in used_t:
                tr[1] += 1
                if tr[1] > max_age:
                    done.append(tr[0])
                    continue
            keep.append(tr)
        keep += [[[(f, d)], 0] for di, d in enumerate(dets) if di not in used_d]
        live = keep
    tracks = done + [tr[0] for tr in live]
    return sorted([(f, d.x1, d.y1, d.x2, d.y2) for f, d in pts] for pts in tracks)


def test_slow_and_stationary_objects_behave_as_before():
    rng = random.Random(7)
    frames = []
    for f in range(60):
        dets = []
        for cx, cy, present in ((50, 50, f < 45), (200, 80, 10 <= f < 50), (120, 140, f % 7 != 3)):
            if present:
                jx, jy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
                dets.append(box(cx + jx, cy + jy, cx + jx + 30, cy + jy + 30))
        frames.append(dets)
    ended = run(MotionTracker(0.2, 3, 3), frames)
    got = sorted(
        [(p.frame_index, p.detection.x1, p.detection.y1, p.detection.x2, p.detection.y2)
         for p in tr.points]
        for tr in ended
    )  # fmt: skip
    assert got == old_last_box_tracker(frames, 0.2, 3)
    assert sum(tr.confirmed for tr in ended) == 3

    (only,) = run(MotionTracker(0.2, 5, 3), [[box(0, 0, 20, 20)]] * 12)  # stationary
    assert only.n_hits == 12


def test_a_new_track_is_matched_exactly_by_iou_with_its_first_box():
    t = MotionTracker(0.2, 15, 1)
    t.update([box(0, 0, 10, 10)], 0, 0.0)
    t.update([box(8, 0, 18, 10)], 1, 0.1)  # IoU 2/18 < 0.2: no velocity yet, so a new track
    assert [tr.track_id for tr in t.live_tracks] == [1, 2]
    t = MotionTracker(0.2, 15, 1)
    t.update([box(0, 0, 10, 10)], 0, 0.0)
    t.update([box(6, 0, 16, 10)], 1, 0.1)  # IoU 4/16 = 0.25
    assert [tr.track_id for tr in t.live_tracks] == [1]


# --------------------------------------------------------------------------- ageing


def test_age_is_measured_in_source_frames_at_any_stride():
    # The same video: an object visible in source frames 0..30, then gone.
    def ended_at(stride):
        t = MotionTracker(0.2, 15, 1)
        out = []
        for f in range(0, 80, stride):
            dets = [box(10, 10, 50, 50)] if f <= 30 else []
            out += [(f, tr) for tr in t.update(dets, f, f / 30.0)]
        return out

    for stride in (1, 3):
        ((end_frame, track),) = ended_at(stride)
        last = track.last.frame_index
        assert last == 30 - 30 % stride
        # ended on the first analysed frame more than 15 source frames after its last point
        assert last + 15 < end_frame <= last + 15 + stride
    assert [f for f, _ in ended_at(1)] == [46] and [f for f, _ in ended_at(3)] == [48]


def test_slow_stride_keeps_the_previous_gap_tolerance():
    # max_age_frames 5 at stride 3 is 15 source frames: five missed analysed frames
    # still continue the track (as with the previous analysed-frame ageing), six do not.
    d = box(10, 10, 50, 50)
    (track,) = run(MotionTracker(0.2, 15, 3), [[d], [d]] + [[]] * 5 + [[d]], stride=3)
    assert track.n_hits == 3
    ended = run(MotionTracker(0.2, 15, 3), [[d], [d]] + [[]] * 6 + [[d]], stride=3)
    assert len(ended) == 2


# --------------------------------------------------------------------------- misc


def test_alias_and_velocity_window():
    assert IouTracker is MotionTracker
    # The velocity comes from the last 5 points back, not from the whole history: an
    # object that moved right for a long time and then stopped is predicted to stay put.
    t = MotionTracker(0.2, 15, 1)
    for k in range(12):
        t.update([box(10.0 * k, 0, 10.0 * k + 40, 40)], k, k * 0.1)
    for k in range(12, 20):
        t.update([box(110, 0, 150, 40)], k, k * 0.1)
    (track,) = t.live_tracks
    assert track.n_hits == 20
    assert t.predicted_box(track, 20).x1 == pytest.approx(110)


def test_prediction_is_a_translation_and_size_is_kept():
    t = MotionTracker(0.2, 15, 1)
    t.update([box(0, 0, 10, 10)], 0, 0.0)
    t.update([box(2, 2, 14, 10)], 1, 0.1)  # centre moves (+3, +1) per frame; size changes
    (track,) = t.live_tracks
    p = t.predicted_box(track, 3)  # two frames later
    assert (p.x1, p.y1, p.x2, p.y2) == (2 + 6, 2 + 2, 14 + 6, 10 + 2)
    assert t.predicted_box(track, 1) == track.last.detection
