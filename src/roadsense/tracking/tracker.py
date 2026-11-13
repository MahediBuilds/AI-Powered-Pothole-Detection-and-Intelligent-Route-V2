"""Greedy motion-aware tracker: groups per-frame detections of one pothole into a track.

A pothole grows and drifts across the image as the camera moves. At walking
pace it keeps overlapping with itself between analysed frames, and comparing a
detection with the track's last box is enough. At vehicle speed it can move
further between two analysed frames than its own height, so the last box and the
next detection do not overlap at all. Each live track therefore predicts where
its box is now: the last box translated by the track's velocity times the number
of source frames elapsed. A detection continues the track when its IoU with the
predicted box is at least ``iou_match``. The size of the box is not predicted.

The velocity is the centre displacement per source frame between the track's
last point and the point up to ``VELOCITY_WINDOW`` points earlier (first to last
of that window, which smooths detector jitter). A track with one point has zero
velocity, so a new track is matched by plain IoU with its first box. Matching is
greedy and one-to-one (highest IoU first; ties by track, then detection order).
There is no fallback on centre distance alone: it merged unrelated detections.

Ageing is measured in SOURCE frames (``Detection`` frame indices), so it means
the same time at any frame stride. A track that is still unmatched on a frame
where ``frame_index - last.frame_index > max_age_source_frames`` has ended.
Ended tracks are returned by ``update`` (and by ``flush`` at the end of the
video) with ``confirmed`` set: a track with fewer than ``min_hits`` detections is
flicker, returned unconfirmed so the caller can count it instead of losing it.

Track ids are consecutive integers from 1 in order of creation; given the same
input the output is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from roadsense.detection.detector import Detection


VELOCITY_WINDOW = 5  # the velocity spans at most this many points before the last one


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two boxes (0.0 when either has no area)."""
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    t_video_s: float
    detection: Detection


@dataclass
class Track:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    confirmed: bool = False  # set when the track ends: n_hits >= min_hits

    @property
    def n_hits(self) -> int:
        return len(self.points)

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def best(self) -> TrackPoint:
        """Highest-confidence point; on a tie the later one."""
        best = self.points[0]
        for p in self.points[1:]:
            if p.detection.confidence >= best.detection.confidence:
                best = p
        return best


class MotionTracker:
    """See the module docstring. ``max_age_source_frames`` counts frames of the video."""

    def __init__(self, iou_match: float, max_age_source_frames: int, min_hits: int) -> None:
        if not 0.0 <= iou_match <= 1.0:
            raise ValueError(f"iou_match must be in [0, 1], got {iou_match!r}")
        if max_age_source_frames < 0:
            raise ValueError(f"max_age_source_frames must be >= 0, got {max_age_source_frames!r}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits!r}")
        self.iou_match = iou_match
        self.max_age_source_frames = max_age_source_frames
        self.min_hits = min_hits
        self._live: list[Track] = []
        self._next_id = 1

    @property
    def live_tracks(self) -> list[Track]:
        """Tracks that have not ended yet, in creation order."""
        return list(self._live)

    @staticmethod
    def predicted_box(track: Track, frame_index: int) -> Detection:
        """The track's last box moved by its velocity to ``frame_index``."""
        last = track.last
        first = track.points[max(0, len(track.points) - 1 - VELOCITY_WINDOW)]
        span = last.frame_index - first.frame_index
        if span <= 0:
            return last.detection
        a, b = first.detection, last.detection
        elapsed = frame_index - last.frame_index
        dx = ((b.x1 + b.x2) - (a.x1 + a.x2)) / 2 / span * elapsed
        dy = ((b.y1 + b.y2) - (a.y1 + a.y2)) / 2 / span * elapsed
        return replace(b, x1=b.x1 + dx, x2=b.x2 + dx, y1=b.y1 + dy, y2=b.y2 + dy)

    def update(
        self, detections: list[Detection], frame_index: int, t_video_s: float
    ) -> list[Track]:
        """Feed one analysed frame; return the tracks that ended on it."""
        pairs = []
        for ti, track in enumerate(self._live):
            predicted = self.predicted_box(track, frame_index)
            for di, det in enumerate(detections):
                value = iou(predicted, det)
                if value > 0.0 and value >= self.iou_match:
                    # sort key: highest IoU first; ties by track then detection order
                    pairs.append((-value, ti, di))
        pairs.sort()

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._live[ti].points.append(TrackPoint(frame_index, t_video_s, detections[di]))

        ended: list[Track] = []
        still_live: list[Track] = []
        for ti, track in enumerate(self._live):
            age = frame_index - track.last.frame_index  # source frames since its last point
            if ti not in matched_tracks and age > self.max_age_source_frames:
                ended.append(self._finish(track))
            else:
                still_live.append(track)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                still_live.append(
                    Track(self._next_id, [TrackPoint(frame_index, t_video_s, det)])
                )
                self._next_id += 1
        self._live = still_live
        return ended

    def flush(self) -> list[Track]:
        """End every live track (call at the end of the video)."""
        ended = [self._finish(t) for t in self._live]
        self._live = []
        return ended

    def _finish(self, track: Track) -> Track:
        track.confirmed = track.n_hits >= self.min_hits
        return track


IouTracker = MotionTracker  # the name used before the tracker became motion-aware
