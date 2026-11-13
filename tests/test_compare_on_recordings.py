"""Pure-logic test for scripts/compare_on_recordings.py (no ultralytics / video needed)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("compare_on_recordings", Path(__file__).resolve().parents[1] / "scripts" / "compare_on_recordings.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def test_disagreement_frames_only_one_model_ordered_by_score_and_capped():
    a = [[0.9], [], [0.5], [0.4], []]
    b = [[0.8], [0.7], [], [0.6], []]
    # frame 0 and 3 are agreed detections, 4 agreed empty; 1 (only B, .7) and 2 (only A, .5) disagree
    assert cr.disagreement_frames(a, b) == [(1, "B", 0.7), (2, "A", 0.5)]
    many_a = [[0.1 + i / 100] for i in range(20)]
    assert len(cr.disagreement_frames(many_a, [[] for _ in many_a], 12)) == 12
    assert cr.disagreement_frames(many_a, [[] for _ in many_a], 3)[0][2] == __import__("pytest").approx(0.29)
