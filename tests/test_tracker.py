import pytest

from roadsense.detection.detector import Detection
from roadsense.tracking.tracker import MotionTracker, iou


def box(x1, y1, x2, y2, conf=0.9):
    return Detection(x1, y1, x2, y2, conf, 0, "pothole")


def run(tracker, frames):
    """frames: detection lists of consecutive source frames (stride 1). Returns all ended tracks.

    The tracker's age limit counts SOURCE frames, so at stride 1 a limit of N allows N
    missed frames. Motion, stride and ageing are covered in test_tracker_motion.py.
    """
    ended = []
    for i, dets in enumerate(frames):
        ended += tracker.update(dets, i, i * 0.1)
    ended += tracker.flush()
    return ended


# --------------------------------------------------------------------------- iou


def test_iou_identical_is_one():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_and_touching_is_zero():
    assert iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0
    assert iou(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    # intersection 50, union 150
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_is_symmetric_and_handles_zero_area():
    a, b = box(0, 0, 10, 10), box(3, 4, 12, 9)
    assert iou(a, b) == pytest.approx(iou(b, a))
    assert iou(box(5, 5, 5, 5), box(5, 5, 5, 5)) == 0.0


# --------------------------------------------------------------------------- tracking


def test_moving_object_is_one_track():
    t = MotionTracker(0.2, 3, 3)
    frames = [[box(10 + 4 * i, 20, 50 + 4 * i, 60)] for i in range(8)]
    ended = run(t, frames)
    assert len(ended) == 1
    assert ended[0].n_hits == 8 and ended[0].confirmed
    assert ended[0].track_id == 1


def test_two_separated_objects_are_two_tracks():
    t = MotionTracker(0.2, 3, 3)
    frames = [[box(10, 10, 40, 40), box(200, 100, 240, 140)] for _ in range(5)]
    ended = run(t, frames)
    assert len(ended) == 2
    assert {tr.track_id for tr in ended} == {1, 2}
    assert all(tr.n_hits == 5 and tr.confirmed for tr in ended)


def test_gap_within_max_age_keeps_track():
    t = MotionTracker(0.2, 2, 3)
    d = box(10, 10, 40, 40)
    frames = [[d], [d], [], [], [d], [d]]  # 2 missed source frames == the age limit
    ended = run(t, frames)
    assert len(ended) == 1 and ended[0].n_hits == 4


def test_gap_longer_than_max_age_splits_track():
    t = MotionTracker(0.2, 2, 2)
    d = box(10, 10, 40, 40)
    frames = [[d], [d], [], [], [], [d], [d]]  # 3 missed source frames > the age limit
    ended = []
    returned_at = []
    for i, dets in enumerate(frames):
        out = t.update(dets, i, i * 0.1)
        if out:
            returned_at.append(i)
        ended += out
    ended += t.flush()
    assert returned_at == [4]  # the first track ends on the third missed frame
    assert len(ended) == 2
    assert [tr.track_id for tr in ended] == [1, 2]
    assert all(tr.n_hits == 2 for tr in ended)


def test_max_age_zero_ends_track_on_first_miss():
    t = MotionTracker(0.2, 0, 1)
    d = box(10, 10, 40, 40)
    assert t.update([d], 0, 0.0) == []
    ended = t.update([], 1, 0.1)
    assert len(ended) == 1 and ended[0].n_hits == 1


def test_min_hits_rejects_flicker():
    t = MotionTracker(0.2, 1, 3)
    d = box(10, 10, 40, 40)
    flicker = box(300, 300, 330, 330)
    frames = [[d, flicker], [d], [d], [d]]
    ended = run(t, frames)
    by_id = {tr.track_id: tr for tr in ended}
    assert by_id[1].confirmed and by_id[1].n_hits == 4
    assert not by_id[2].confirmed and by_id[2].n_hits == 1


def test_matching_is_greedy_and_one_to_one():
    t = MotionTracker(0.1, 3, 1)
    t.update([box(0, 0, 10, 10)], 0, 0.0)
    # Two detections both overlap the track; only the better one may continue it.
    better, worse = box(1, 0, 11, 10), box(6, 0, 16, 10)
    t.update([worse, better], 1, 0.1)
    live = t.live_tracks
    assert len(live) == 2
    assert live[0].track_id == 1 and live[0].last.detection == better
    assert live[1].track_id == 2 and live[1].last.detection == worse
    assert live[0].n_hits == 2 and live[1].n_hits == 1


def test_below_threshold_overlap_starts_new_track():
    t = MotionTracker(0.5, 3, 1)
    t.update([box(0, 0, 10, 10)], 0, 0.0)
    t.update([box(5, 0, 15, 10)], 1, 0.1)  # IoU 1/3 < 0.5
    assert [tr.track_id for tr in t.live_tracks] == [1, 2]


def test_ids_are_deterministic():
    frames = [
        [box(10, 10, 40, 40), box(200, 100, 240, 140)],
        [box(12, 10, 42, 40)],
        [box(14, 10, 44, 40), box(100, 200, 130, 230)],
        [],
        [],
    ]

    def ids():
        ended = run(MotionTracker(0.2, 1, 1), frames)
        return [(tr.track_id, tr.n_hits, tr.first.frame_index) for tr in ended]

    assert ids() == ids()
    assert sorted(i for i, *_ in ids()) == [1, 2, 3]


def test_flush_ends_all_live_tracks_and_empties_tracker():
    t = MotionTracker(0.2, 5, 2)
    t.update([box(0, 0, 10, 10), box(100, 100, 120, 120)], 0, 0.0)
    t.update([box(0, 0, 10, 10)], 1, 0.1)
    ended = t.flush()
    assert [(tr.track_id, tr.confirmed) for tr in ended] == [(1, True), (2, False)]
    assert t.live_tracks == [] and t.flush() == []


def test_best_is_highest_confidence_ties_to_later():
    t = MotionTracker(0.2, 3, 1)
    t.update([box(0, 0, 10, 10, 0.5)], 0, 0.0)
    t.update([box(0, 0, 10, 10, 0.9)], 1, 0.1)
    t.update([box(0, 0, 10, 10, 0.9)], 2, 0.2)
    t.update([box(0, 0, 10, 10, 0.4)], 3, 0.3)
    (track,) = t.flush()
    assert track.best.frame_index == 2
    assert track.first.frame_index == 0 and track.last.frame_index == 3


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        MotionTracker(1.5, 1, 1)
    with pytest.raises(ValueError):
        MotionTracker(0.2, -1, 1)
    with pytest.raises(ValueError):
        MotionTracker(0.2, 1, 0)
