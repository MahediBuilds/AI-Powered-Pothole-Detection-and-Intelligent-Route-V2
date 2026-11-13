import math

import cv2
import numpy as np
import pytest
from ingest_fixtures import H, W, decode_frame_number, write_synthetic_video

from roadsense.ingest import video as video_mod
from roadsense.ingest.video import (
    VideoError,
    iter_frames,
    probe_video,
    read_timeline,
)


@pytest.fixture
def make_video(tmp_path):
    def _make(n_frames=20, fps=20.0):
        return write_synthetic_video(tmp_path, n_frames, fps)

    return _make


def test_probe_values(make_video):
    path = make_video(n_frames=20, fps=20.0)
    info = probe_video(path)
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(20.0)
    assert info.frame_count == 20
    assert info.duration_s == pytest.approx(1.0)
    assert isinstance(info.codec, str) and len(info.codec) == 4


def test_probe_missing_file_names_path(tmp_path):
    path = tmp_path / "nope.mp4"
    with pytest.raises(VideoError, match="not found") as exc:
        probe_video(path)
    assert str(path) in str(exc.value)


def test_probe_non_video_file_names_path(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(VideoError) as exc:
        probe_video(path)
    assert str(path) in str(exc.value)


def test_probe_empty_file_names_path(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(VideoError) as exc:
        probe_video(path)
    assert str(path) in str(exc.value)


def test_frames_stride_one_count_order_and_content(make_video):
    path = make_video(n_frames=20)
    frames = list(iter_frames(path))
    assert [f.index for f in frames] == list(range(20))
    assert [decode_frame_number(f.image) for f in frames] == list(range(20))
    first = frames[0].image
    assert first.shape == (H, W, 3) and first.dtype == np.uint8  # BGR, native size


def test_frames_do_not_share_a_buffer(make_video):
    frames = list(iter_frames(make_video(n_frames=6)))
    assert len({decode_frame_number(f.image) for f in frames}) == 6


def test_stride_three_returns_every_third_frame(make_video):
    path = make_video(n_frames=20)
    frames = list(iter_frames(path, stride=3))
    assert [f.index for f in frames] == [0, 3, 6, 9, 12, 15, 18]
    assert [decode_frame_number(f.image) for f in frames] == [0, 3, 6, 9, 12, 15, 18]


def test_timestamps_match_index_over_fps_and_increase(make_video):
    fps = 20.0
    path = make_video(n_frames=20, fps=fps)
    frames = list(iter_frames(path))
    times = [f.t_video_s for f in frames]
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert all(b > a for a, b in zip(times, times[1:]))
    for f in frames:
        assert abs(f.t_video_s - f.index / fps) < 0.5 / fps
    # stride keeps the timestamp of the frame returned, not a recomputed one
    strided = list(iter_frames(path, stride=4))
    assert [f.t_video_s for f in strided] == [times[f.index] for f in strided]


def test_timeline_reports_source_and_covers_every_frame(make_video):
    path = make_video(n_frames=15)
    timeline = read_timeline(path)
    assert timeline.source == "pts"
    assert len(timeline.times_s) == 15 and timeline.times_s[0] == 0.0


def test_supplied_timeline_avoids_second_pass(make_video, monkeypatch):
    path = make_video(n_frames=10)
    timeline = read_timeline(path)
    monkeypatch.setattr(video_mod, "read_timeline", lambda p: pytest.fail("timeline re-read"))
    assert len(list(iter_frames(path, timeline=timeline))) == 10


def test_stale_timeline_is_an_error(make_video):
    path = make_video(n_frames=10)
    short = video_mod.Timeline("pts", tuple(i / 20 for i in range(5)))
    with pytest.raises(VideoError, match="more frames than the timeline"):
        list(iter_frames(path, timeline=short))
    long = video_mod.Timeline("pts", tuple(i / 20 for i in range(12)))
    with pytest.raises(VideoError, match="timeline holds 12"):
        list(iter_frames(path, timeline=long))


def test_early_break_releases_the_capture(make_video):
    path = make_video(n_frames=20)
    for frame in iter_frames(path):
        if frame.index == 2:
            break
    path.unlink()  # raises PermissionError on Windows if the file is still open
    assert not path.exists()


def test_closing_generator_releases_the_capture(make_video):
    path = make_video(n_frames=20)
    gen = iter_frames(path)
    next(gen)
    gen.close()
    path.unlink()
    assert not path.exists()


def test_exception_in_consumer_releases_the_capture(make_video):
    path = make_video(n_frames=20)
    with pytest.raises(RuntimeError, match="boom"):
        for frame in iter_frames(path):
            if frame.index == 3:
                raise RuntimeError("boom")
    path.unlink()
    assert not path.exists()


def test_exhausted_iteration_releases_the_capture(make_video):
    path = make_video(n_frames=5)
    list(iter_frames(path))
    path.unlink()


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, "2"])
def test_bad_stride_raises_immediately(make_video, stride):
    path = make_video(n_frames=5)
    with pytest.raises(ValueError, match="stride"):
        iter_frames(path, stride)  # not even iterated


def test_iter_frames_missing_file_names_path(tmp_path):
    path = tmp_path / "nope.mp4"
    with pytest.raises(VideoError) as exc:
        next(iter_frames(path))
    assert str(path) in str(exc.value)


def test_iter_frames_non_video_file_names_path(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoError) as exc:
        next(iter_frames(path))
    assert str(path) in str(exc.value)


# ---------------------------------------------------------------------------
# Timestamp source decision
# ---------------------------------------------------------------------------

P = video_mod.Path("x.mp4")


def test_pts_used_when_strictly_increasing_and_shifted_to_zero():
    t = video_mod._choose_timeline([500.0, 533.0, 600.0, 700.0], 30.0, P)
    assert t.source == "pts"
    assert t.times_s == pytest.approx((0.0, 0.033, 0.1, 0.2))


def test_variable_frame_rate_pts_are_kept_not_replaced_by_index_over_fps():
    t = video_mod._choose_timeline([0.0, 33.0, 66.0, 132.0, 165.0], 30.0, P)
    assert t.source == "pts"
    assert t.times_s[3] == pytest.approx(0.132) and t.times_s[3] != pytest.approx(3 / 30.0)


@pytest.mark.parametrize(
    "pts",
    [
        [0.0, 0.0, 0.0, 0.0],  # stuck at zero
        [0.0, 33.0, 0.0, 99.0],  # goes backwards
        [0.0, 33.0, 33.0, 66.0],  # repeated
        [0.0, 33.0, float("nan"), 99.0],
        [0.0, 33.0, float("inf"), 99.0],
    ],
)
def test_unusable_pts_fall_back_to_index_over_fps_for_whole_file(pts):
    t = video_mod._choose_timeline(pts, 25.0, P)
    assert t.source == "index_fps"
    assert t.times_s == pytest.approx(tuple(i / 25.0 for i in range(len(pts))))


def test_single_frame_pts_is_accepted():
    assert video_mod._choose_timeline([0.0], 30.0, P).source == "pts"


def test_unusable_pts_and_no_fps_is_error():
    with pytest.raises(VideoError, match="x.mp4"):
        video_mod._choose_timeline([0.0, 0.0], 0.0, P)
    with pytest.raises(VideoError, match="no decodable frames"):
        video_mod._choose_timeline([], 30.0, P)


def test_end_to_end_fallback_when_decoder_reports_zero_pts(make_video, monkeypatch):
    path = make_video(n_frames=12, fps=20.0)

    class ZeroPts:
        def __init__(self, *args):
            self._cap = real(*args)

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_POS_MSEC else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    real = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", ZeroPts)
    timeline = read_timeline(path)
    assert timeline.source == "index_fps"
    frames = list(iter_frames(path, timeline=timeline))
    times = [f.t_video_s for f in frames]
    assert times == pytest.approx([i / 20.0 for i in range(12)])
    assert all(b > a for a, b in zip(times, times[1:]))
    assert not any(math.isnan(t) for t in times)
