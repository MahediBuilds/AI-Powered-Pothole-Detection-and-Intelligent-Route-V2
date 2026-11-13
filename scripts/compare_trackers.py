"""Replay saved per-frame detections through the previous and the current tracker.

Review tool, not part of the application. It needs no inference: it reads the
detections saved by an earlier review (``data/review_20261112/tracking/<name>_s1.json``,
every frame, detector confidence 0.25) and groups them into tracks twice:

* PREVIOUS behaviour: a private copy of the old matcher (last-box IoU, ageing in
  analysed frames) at stride 3, IoU 0.2, age 5 analysed frames, 3 hits.
* NEW behaviour: ``roadsense.tracking.tracker.MotionTracker`` with the settings the
  pipeline picks for the recording from its GPS speed (``plan_tracking``).

For each recording it prints the number of analysed frames, tracks, confirmed
tracks, one-detection tracks, the median and maximum duration of confirmed tracks
and, with ``--ranges``, the frame range and detection count of every confirmed
track. There is no ground truth here: this measures track CONTINUITY only, not
detection accuracy.

    python scripts/compare_trackers.py [--ranges] [NAME ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadsense.config import load_app_config  # noqa: E402
from roadsense.detection.detector import Detection  # noqa: E402
from roadsense.ingest.gps import load_gps_csv  # noqa: E402
from roadsense.pipeline.process import TrackingPlan, plan_tracking  # noqa: E402
from roadsense.tracking.tracker import MotionTracker, iou  # noqa: E402

TRACKING_DIR = ROOT / "data" / "review_20261112" / "tracking"
RECORDINGS = [
    "road_20261013_180932",
    "road_20261013_180405",
    "road_20261013_180439",
    "road_20261013_173922",
    "road_20261013_165247",
]

# (frame_index, t_video_s, detections) for every frame of the video
Frames = list[tuple[int, float, list[Detection]]]


@dataclass
class Result:
    """One ended track, reduced to what the comparison reports."""

    first_frame: int
    last_frame: int
    first_t: float
    last_t: float
    n_hits: int
    confirmed: bool

    @property
    def duration_s(self) -> float:
        return self.last_t - self.first_t


def load_detections(name: str) -> Frames:
    data = json.loads((TRACKING_DIR / f"{name}_s1.json").read_text(encoding="utf-8"))
    return [
        (int(f), float(t), [Detection(x1, y1, x2, y2, c, 0, "pothole") for _, x1, y1, x2, y2, c in dets])
        for f, t, dets in data["frames"]
    ]


def find_gps(name: str) -> Path:
    (path,) = (ROOT / "data" / "recordings").glob(f"**/{name}/{name}_gps.csv")
    return path


def plan_for(name: str, frames: Frames, cfg) -> TrackingPlan:
    log = load_gps_csv(find_gps(name), float(cfg.ingest.max_gps_accuracy_m))
    return plan_tracking(cfg.tracking, list(log.fixes), frames[-1][1])


# --------------------------------------------------------------------------- previous behaviour


@dataclass
class _OldTrack:
    points: list[tuple[int, float, Detection]]
    misses: int = 0
    ended: bool = field(default=False)


def replay_previous(frames: Frames, stride=3, iou_match=0.2, max_age_frames=5, min_hits=3) -> list[Result]:
    """PREVIOUS BEHAVIOUR, kept only for this comparison (copy of the old IouTracker).

    A detection is matched to a track only by IoU with the track's LAST box, greedily
    (highest IoU first); a track ends after more than ``max_age_frames`` consecutive
    analysed frames without a match.
    """
    live: list[_OldTrack] = []
    results: list[Result] = []

    def finish(track: _OldTrack) -> None:
        p = track.points
        results.append(Result(p[0][0], p[-1][0], p[0][1], p[-1][1], len(p), len(p) >= min_hits))

    analysed = 0
    for index, t, dets in frames:
        if index % stride:
            continue
        analysed += 1
        pairs = []
        for ti, track in enumerate(live):
            for di, det in enumerate(dets):
                value = iou(track.points[-1][2], det)
                if value > 0.0 and value >= iou_match:
                    pairs.append((-value, ti, di))
        pairs.sort()
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            live[ti].points.append((index, t, dets[di]))
            live[ti].misses = 0
        keep = []
        for ti, track in enumerate(live):
            if ti not in used_t:
                track.misses += 1
                if track.misses > max_age_frames:
                    finish(track)
                    continue
            keep.append(track)
        keep += [_OldTrack([(index, t, d)]) for di, d in enumerate(dets) if di not in used_d]
        live = keep
    for track in live:
        finish(track)
    results.sort(key=lambda r: (r.first_frame, r.last_frame))
    return results


# --------------------------------------------------------------------------- new behaviour


def replay_new(frames: Frames, cfg, plan: TrackingPlan):
    """Run the current tracker. Returns (results, assignments).

    ``assignments`` lists, per analysed frame, ``(frame_index, [(track_id, detection)])``
    for the detections that belong to a track, and ``confirmed`` maps track id -> bool.
    """
    tracker = MotionTracker(float(cfg.tracking.iou_match), plan.max_age_source_frames, plan.min_hits)
    ended = []
    assignments = []
    for index, t, dets in frames:
        if index % plan.frame_stride:
            continue
        ended += tracker.update(dets, index, t)
        assignments.append(
            (index, [(tr.track_id, tr.last.detection) for tr in tracker.live_tracks if tr.last.frame_index == index])
        )
    ended += tracker.flush()
    results = [
        Result(tr.first.frame_index, tr.last.frame_index, tr.first.t_video_s, tr.last.t_video_s,
               tr.n_hits, tr.confirmed)
        for tr in sorted(ended, key=lambda tr: tr.track_id)
    ]  # fmt: skip
    confirmed = {tr.track_id: tr.confirmed for tr in ended}
    return results, assignments, confirmed


# --------------------------------------------------------------------------- report


def summarise(results: list[Result], n_analysed: int) -> dict:
    confirmed = [r for r in results if r.confirmed]
    durations = [r.duration_s for r in confirmed]
    return {
        "analysed": n_analysed,
        "tracks": len(results),
        "confirmed": len(confirmed),
        "single": sum(1 for r in results if r.n_hits == 1),
        "median_s": statistics.median(durations) if durations else float("nan"),
        "max_s": max(durations) if durations else float("nan"),
        "ranges": ", ".join(
            f"{r.first_frame}-{r.last_frame}({r.n_hits})" for r in sorted(confirmed, key=lambda r: r.first_frame)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="recording names (default: the five review recordings)")
    parser.add_argument("--ranges", action="store_true", help="list confirmed tracks (frame range, detections)")
    args = parser.parse_args(argv)
    cfg = load_app_config(ROOT / "configs" / "app.yaml")

    rows = []
    for name in args.names or RECORDINGS:
        frames = load_detections(name)
        plan = plan_for(name, frames, cfg)
        old = replay_previous(frames)
        new, _, _ = replay_new(frames, cfg, plan)
        n_old = sum(1 for f, *_ in frames if f % 3 == 0)
        n_new = sum(1 for f, *_ in frames if f % plan.frame_stride == 0)
        rows.append((name, plan, summarise(old, n_old), summarise(new, n_new)))

    header = (
        f"{'recording':<22}{'version':<9}{'mode':<9}{'speed':>6}{'stride':>7}{'hits':>5}{'frames':>7}"
        f"{'tracks':>7}{'conf':>5}{'1-det':>6}{'med s':>7}{'max s':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, plan, old, new in rows:
        speed = "-" if plan.upper_quartile_speed_mps is None else f"{plan.upper_quartile_speed_mps:.1f}"  # decides the mode
        for label, s, stride, hits, mode in (
            ("previous", old, 3, 3, ""),
            ("new", new, plan.frame_stride, plan.min_hits, plan.mode),
        ):
            print(
                f"{name[5:]:<22}{label:<9}{mode:<9}{speed if label == 'new' else '':>6}{stride:>7}{hits:>5}"
                f"{s['analysed']:>7}{s['tracks']:>7}{s['confirmed']:>5}{s['single']:>6}"
                f"{s['median_s']:>7.2f}{s['max_s']:>7.2f}"
            )
    if args.ranges:
        for name, _, old, new in rows:
            print(f"\n{name}\n  previous confirmed: {old['ranges'] or '-'}\n  new confirmed:      {new['ranges'] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
