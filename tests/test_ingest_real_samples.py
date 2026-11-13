"""Optional smoke tests on real recorder output. They only READ the samples and
are skipped when the sample directory is absent. Nothing from the samples is
copied into the project."""

from pathlib import Path

import pytest

from roadsense.ingest.gps import position_at
from roadsense.ingest.session import load_session
from roadsense.ingest.video import iter_frames, read_timeline

# The original project now lives in "Old Project/" next to "New Project/" (reorganised 2026-11-13).
SAMPLES_DIR = (
    Path(__file__).resolve().parents[3]
    / "Old Project"
    / "Exisiting Implementation"
    / "recordings"
    / "recordings"
)
MAX_ACC, MAX_GAP, MAX_EXTRAPOLATION = 25.0, 3.0, 1.0

pytestmark = pytest.mark.skipif(
    not SAMPLES_DIR.is_dir(), reason=f"real sample recordings not found at {SAMPLES_DIR}"
)

# Samples are either flat (<dir>/<stem>.mp4) or one folder per session.
SAMPLE_VIDEOS = sorted(SAMPLES_DIR.glob("**/road_*.mp4")) if SAMPLES_DIR.is_dir() else []


def _id(path):
    return path.stem


@pytest.fixture(params=SAMPLE_VIDEOS, ids=_id)
def sample(request):
    return load_session(request.param, max_accuracy_m=MAX_ACC, max_gap_s=MAX_GAP)


def test_every_sample_session_loads(sample):
    assert (sample.video.width, sample.video.height) == (1280, 720)
    assert sample.video.fps > 0 and sample.video.duration_s > 0
    assert sample.gps is not None and sample.gps_path.is_file()
    assert sample.sidecar_path is not None


def test_frame_timestamps_are_strictly_increasing_and_span_the_video(sample):
    timeline = read_timeline(sample.video_path)
    times = timeline.times_s
    assert timeline.source in ("pts", "index_fps")
    assert times[0] == 0.0
    assert all(b > a for a, b in zip(times, times[1:]))
    assert abs(times[-1] - sample.video.duration_s) < 1.0
    assert sample.video.frame_count is None or len(times) == sample.video.frame_count
    # decoding agrees with the timeline (every 60th frame is enough to prove it)
    got = [(f.index, f.t_video_s) for f in iter_frames(sample.video_path, stride=60, timeline=timeline)]
    assert got and all(t == times[i] for i, t in got)


def test_gps_summary_is_sane(sample):
    s = sample.gps.summary
    csv_rows = [ln for ln in sample.gps_path.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    dropped = sum(1 for i in sample.gps.issues if i.row_dropped)
    assert s.n_rows_read == len(csv_rows)
    assert s.n_fixes == len(csv_rows) - dropped
    assert 0 <= s.n_used <= s.n_fixes
    assert s.n_used > 0
    assert s.t_first_s <= s.t_last_s
    assert s.median_accuracy_m is None or 0 < s.median_accuracy_m <= MAX_ACC
    assert s.path_length_m >= 0
    assert [f.seq for f in sample.gps.fixes] == list(range(s.n_fixes))


def test_position_at_video_start_thanks_to_preroll_unless_data_does_not_support_it(sample):
    used = [f.t_video_s for f in sample.gps.fixes if f.used]
    if not used:
        pytest.skip("no usable GPS fixes in this sample")
    pos = position_at(sample.gps, 0.0, MAX_GAP, MAX_EXTRAPOLATION)
    before = [t for t in used if t <= 0.0]
    after = [t for t in used if t >= 0.0]
    if before and after:
        supported = after[0] - before[-1] <= MAX_GAP
    else:  # outside the trace: only the extrapolation window can help
        supported = (0.0 - before[-1] if before else after[0]) <= MAX_EXTRAPOLATION
    if supported:
        assert pos is not None
        assert -90 <= pos.lat <= 90 and -180 <= pos.lon <= 180
    else:
        # A sparse or poor-accuracy sample: the correct answer is an explicit "no position".
        assert pos is None
