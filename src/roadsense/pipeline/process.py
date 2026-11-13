"""Per-session video pipeline: frames -> detections -> tracks -> observations.

``process_session`` reads a stored session (video path and GPS fixes from the
database), runs the detector on every ``frame_stride``-th frame, groups the
detections into tracks, and stores one observation per confirmed track together
with an annotated image of the track's best detection.

The frame stride and track settings depend on how fast the camera moved: a
session whose median GPS speed is at least ``tracking.vehicle_speed_mps`` is a
vehicle recording (every frame is analysed, tracks need ``vehicle_min_hits``
detections); anything slower, or without usable GPS, uses the walking settings.
The chosen mode is recorded in the report under ``tracking``.

The best frame of each live track is encoded to its annotated JPEG as soon as it
is known, and only the bytes are kept, so memory does not grow with full frames.

Positions come only from the GPS log through ``position_at``. When it has no
answer (no GPS, a dropout, outside the trace) the observation keeps NULL
``cam_lat/cam_lon/cam_accuracy_m``; a coordinate is never invented.

All observations of a job and the ``done`` status (with ``sessions.active_job_id``)
are written in one transaction. On failure the job is marked ``failed``, no
observation of that job remains, its image folder is removed, and a
``PipelineError`` is raised. Small progress updates are committed about once a
second so another connection can poll them. While the video is being probed and
its timeline read the job has stage ``reading video`` and no progress (None).

Frames the detector fails on are skipped and counted. With ``f`` the share of
analysed frames that failed, the report's ``outcome`` is ``complete`` when
``f <= detection.partial_frame_error_fraction`` (isolated failures, counted in a
warning), ``partial`` when ``f`` is above that and at most
``detection.max_frame_error_fraction`` (the job is ``done`` and keeps its
observations, with a prominent warning that potholes in the failed frames are
missing), and above that the job fails. ``partial`` is only an outcome of a
``done`` job: the ``jobs.status`` column keeps its four values and has no schema
change. ``api.views.job_view`` exposes the outcome.
"""

from __future__ import annotations

import shutil
import sqlite3
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from roadsense.config import AppConfig, TrackingConfig
from roadsense.detection.detector import Detection, DetectorError, DetectorLike
from roadsense.ingest.gps import GpsFix, GpsLog, haversine_m, position_at
from roadsense.ingest.video import iter_frames, probe_video, read_timeline
from roadsense.storage import jobs as jobs_db
from roadsense.storage import observations as obs_db
from roadsense.storage.sessions import load_gps_fixes
from roadsense.tracking.tracker import MotionTracker, Track

PIPELINE_VERSION = "1"
PROGRESS_INTERVAL_S = 1.0  # at most one progress write per this many seconds
FIRST_FRAMES_CHECKED = 30  # the job fails if the detector fails on all of these
MAX_IMAGE_SIDE = 1280
JPEG_QUALITY = 85

READING_STAGE = "reading video"  # stage while the video is probed and its timeline read
MIN_RECORDED_SPEEDS = 3  # fewer recorded GPS speeds than this: derive speeds from positions

ProgressCallback = Callable[[str, float], None]


class PipelineError(Exception):
    """A session could not be processed; the job has been marked failed."""


def _session_row(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    return None if row is None else {d[0]: v for d, v in zip(cur.description, tuple(row))}


def annotated_jpeg(image: Any, det: Detection) -> bytes:
    """JPEG bytes of the frame with the box and confidence drawn on it.

    The longest side is at most ``MAX_IMAGE_SIDE``; the box is scaled with it.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        image = image.copy()
    x1, y1, x2, y2 = (round(v * scale) for v in (det.x1, det.y1, det.x2, det.y2))
    color = (0, 0, 255)
    thickness = max(2, round(max(image.shape[:2]) / 400))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    label = f"{det.class_name} {det.confidence:.2f}"
    font_scale = max(0.5, max(image.shape[:2]) / 1200)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    top = max(0, y1 - th - base - 2)
    cv2.rectangle(image, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise PipelineError("cannot encode the annotated image as JPEG")
    return bytes(buf.tobytes())


@dataclass(frozen=True)
class TrackingPlan:
    """Frame stride and track settings chosen for one session."""

    mode: str  # "vehicle", "walking", or "default" (no usable GPS speed)
    median_speed_mps: float | None  # reported only
    upper_quartile_speed_mps: float | None  # decides the mode
    frame_stride: int
    min_hits: int
    max_age_source_frames: int

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "median_speed_mps": self.median_speed_mps,
            "upper_quartile_speed_mps": self.upper_quartile_speed_mps,
            "frame_stride": self.frame_stride,
            "min_hits": self.min_hits,
            "max_age_source_frames": self.max_age_source_frames,
        }


def session_speeds(fixes: list[GpsFix], duration_s: float) -> list[float]:
    """GPS speed samples (m/s) of the used fixes inside the video; empty if none.

    Uses the recorded ``speed_mps`` of fixes with ``0 <= t_video_s <= duration_s``
    (pre-roll and post-roll fixes describe time the camera was not recording). With
    fewer than ``MIN_RECORDED_SPEEDS`` of them, speeds are derived from consecutive
    used fixes in the interval (distance over time); recorded speeds are the last
    resort.
    """
    inside = [f for f in fixes if f.used and 0.0 <= f.t_video_s <= duration_s]
    recorded = [f.speed_mps for f in inside if f.speed_mps is not None]
    if len(recorded) >= MIN_RECORDED_SPEEDS:
        return [float(v) for v in recorded]
    derived = [
        haversine_m(a.lat, a.lon, b.lat, b.lon) / (b.t_video_s - a.t_video_s)
        for a, b in zip(inside, inside[1:])
        if b.t_video_s > a.t_video_s
    ]
    return [float(v) for v in (derived or recorded)]


def session_median_speed(fixes: list[GpsFix], duration_s: float) -> float | None:
    """Median of :func:`session_speeds` (m/s), or None when there is nothing to go on."""
    speeds = session_speeds(fixes, duration_s)
    return float(statistics.median(speeds)) if speeds else None


def session_upper_quartile_speed(fixes: list[GpsFix], duration_s: float) -> float | None:
    """75th percentile of :func:`session_speeds` (m/s), or None without samples."""
    return _upper_quartile(session_speeds(fixes, duration_s))


def _upper_quartile(speeds: list[float]) -> float | None:
    if not speeds:
        return None
    if len(speeds) == 1:
        return float(speeds[0])
    return float(statistics.quantiles(speeds, n=4)[2])


def plan_tracking(cfg: TrackingConfig, fixes: list[GpsFix], duration_s: float) -> TrackingPlan:
    """Choose the stride and thresholds for a session from its GPS speed.

    The mode follows the UPPER QUARTILE of the speed samples, not the median: a ride
    that starts with a long stop, or stop-and-go traffic, has a low median but is still
    a vehicle recording, and processing it as a walking one would fragment tracks.
    Calling a slow session a vehicle one only costs compute.
    """
    speeds = session_speeds(fixes, duration_s)
    median = float(statistics.median(speeds)) if speeds else None
    speed = _upper_quartile(speeds)
    if speed is not None and speed >= cfg.vehicle_speed_mps:
        mode, stride, min_hits = "vehicle", cfg.vehicle_frame_stride, cfg.vehicle_min_hits
    else:
        mode = "default" if speed is None else "walking"
        stride, min_hits = cfg.frame_stride, cfg.min_hits
    # max_age_frames means analysed frames at the SLOW stride in both modes: the same time.
    return TrackingPlan(
        mode,
        None if median is None else round(median, 3),
        None if speed is None else round(speed, 3),
        stride,
        min_hits,
        cfg.max_age_frames * cfg.frame_stride,
    )


class _BestFrame(NamedTuple):
    """A track's best detection, already drawn and encoded, with the frame's size."""

    jpeg: bytes
    width: int
    height: int


class _Run:
    """Mutable state of one pipeline run (kept out of the function signature)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session: dict[str, Any],
        detector: DetectorLike,
        cfg: AppConfig,
        job_id: str,
        progress: ProgressCallback | None,
    ) -> None:
        self.conn = conn
        self.session = session
        self.session_id: str = session["id"]
        self.detector = detector
        self.cfg = cfg
        self.job_id = job_id
        self.progress = progress
        self.tracker = MotionTracker(
            float(cfg.tracking.iou_match),
            cfg.tracking.max_age_frames * cfg.tracking.frame_stride,
            cfg.tracking.min_hits,
        )
        self.best_frames: dict[int, _BestFrame] = {}  # live track id -> its annotated best frame
        self.rows: list[dict[str, Any]] = []
        self.n_tracks = 0
        self.n_rejected = 0
        self.log: GpsLog | None = None
        self.out_dir = Path(cfg.uploads.artifacts_dir) / self.session_id / job_id
        self._last_progress = float("-inf")

    # -- progress -----------------------------------------------------------

    def report_progress(self, stage: str, fraction: float, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < PROGRESS_INTERVAL_S:
            return
        self._last_progress = now
        jobs_db.set_progress(self.conn, self.job_id, stage, fraction)
        self.conn.commit()
        if self.progress is not None:
            self.progress(stage, fraction)

    # -- tracks -------------------------------------------------------------

    def note_best_frames(self, frame_index: int, image: Any) -> None:
        """Encode the annotated frame of every live track whose best point is this frame.

        The full frame is not kept: only the JPEG bytes and the frame size, replacing
        the track's earlier best.
        """
        h, w = image.shape[:2]
        for track in self.tracker.live_tracks:
            if track.best.frame_index == frame_index:
                self.best_frames[track.track_id] = _BestFrame(
                    annotated_jpeg(image, track.best.detection), w, h
                )

    def finish_tracks(self, tracks: list[Track]) -> None:
        for track in tracks:
            self.n_tracks += 1
            kept = self.best_frames.pop(track.track_id, None)
            if not track.confirmed:
                self.n_rejected += 1
                continue
            if kept is None:  # cannot happen: every best point's frame is encoded
                raise PipelineError(f"internal error: no frame kept for track {track.track_id}")
            self.rows.append(self._observation(track, kept))

    def _observation(self, track: Track, kept: _BestFrame) -> dict[str, Any]:
        last = track.last
        det = last.detection

        rel = f"{self.session_id}/{self.job_id}/track_{track.track_id}.jpg"
        path = Path(self.cfg.uploads.artifacts_dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(kept.jpeg)

        pos = None
        if self.log is not None:
            pos = position_at(
                self.log,
                last.t_video_s,
                float(self.cfg.ingest.max_gps_gap_s),
                float(self.cfg.ingest.max_gps_extrapolation_s),
            )
        return {
            "track_id": track.track_id,
            "n_frames": track.n_hits,
            "confidence": max(p.detection.confidence for p in track.points),
            "first_t_video_s": track.first.t_video_s,
            "t_video_s": last.t_video_s,
            "frame_index": last.frame_index,
            "frame_w": kept.width,
            "frame_h": kept.height,
            "image_path": rel,
            "x1": det.x1,
            "y1": det.y1,
            "x2": det.x2,
            "y2": det.y2,
            "cam_lat": None if pos is None else pos.lat,
            "cam_lon": None if pos is None else pos.lon,
            "cam_accuracy_m": None if pos is None else pos.accuracy_m,
        }


def _params(cfg: AppConfig) -> dict[str, Any]:
    return {
        "detection": cfg.detection.model_dump(mode="json"),
        "tracking": cfg.tracking.model_dump(mode="json"),
        "ingest": cfg.ingest.model_dump(mode="json"),
    }


def _execute(run: _Run) -> dict[str, Any]:
    started = time.monotonic()
    session, cfg = run.session, run.cfg
    warnings: list[str] = list(getattr(run.detector, "warnings", []))

    # GPS
    fixes = load_gps_fixes(run.conn, run.session_id)
    gps_info = {
        "source": session["gps_source"],
        "n_fixes": len(fixes),
        "n_used": sum(1 for f in fixes if f.used),
    }
    if fixes:
        run.log = GpsLog(fixes=tuple(fixes), issues=(), n_rows_read=len(fixes))
    if not fixes:
        warnings.append("no GPS data: detections cannot be placed on the map")
    elif gps_info["n_used"] == 0:
        warnings.append("no usable GPS fixes: detections cannot be placed on the map")
    if session["gps_source"] == "synthetic":
        warnings.append("GPS trace is synthetic (demonstration data), not a real recording")

    # Video
    video_path = Path(session["video_path"])
    video = probe_video(video_path)
    timeline = read_timeline(video_path)
    frames_total = len(timeline.times_s)
    plan = plan_tracking(cfg.tracking, fixes, timeline.times_s[-1])
    stride = plan.frame_stride
    run.tracker = MotionTracker(
        float(cfg.tracking.iou_match), plan.max_age_source_frames, plan.min_hits
    )

    run.report_progress("detecting", 0.0, force=True)
    n_processed = n_errors = n_detections = 0
    for frame in iter_frames(video_path, stride, timeline):
        n_processed += 1
        try:
            detections = run.detector.detect(frame.image)
        except DetectorError:
            n_errors += 1
            if n_processed == FIRST_FRAMES_CHECKED and n_errors == n_processed:
                raise PipelineError(
                    f"the detector failed on each of the first {FIRST_FRAMES_CHECKED} frames"
                ) from None
            continue
        n_detections += len(detections)
        ended = run.tracker.update(detections, frame.index, frame.t_video_s)
        run.note_best_frames(frame.index, frame.image)
        run.finish_tracks(ended)
        run.report_progress("detecting", (frame.index + 1) / frames_total)
    run.finish_tracks(run.tracker.flush())

    if n_processed > 0 and n_errors == n_processed:
        raise PipelineError(f"the detector failed on every one of the {n_processed} frames")
    outcome = "complete"
    if n_errors:
        fraction = n_errors / n_processed
        percent = f"{100 * fraction:.1f}%"
        max_fraction = float(cfg.detection.max_frame_error_fraction)
        if fraction > max_fraction:
            raise PipelineError(
                f"{n_errors} of {n_processed} analysed frames ({percent}) could not be analysed, "
                f"more than the {100 * max_fraction:g}% allowed; no results were stored"
            )
        if fraction > float(cfg.detection.partial_frame_error_fraction):
            outcome = "partial"
            warnings.insert(
                0,
                f"{n_errors} of {n_processed} analysed frames ({percent}) could not be analysed; "
                "potholes in those frames are missing from the results.",
            )
        else:
            warnings.append(f"{n_errors} of {n_processed} processed frame(s) could not be analysed")

    rows = sorted(run.rows, key=lambda r: r["track_id"])
    n_with = sum(1 for r in rows if r["cam_lat"] is not None)
    n_without = len(rows) - n_with
    if n_without and fixes:
        warnings.append(
            f"{n_without} observation(s) have no camera position "
            "(GPS gap too long or outside the trace)"
        )

    report: dict[str, Any] = {
        "session_id": run.session_id,
        "session_name": session["name"],
        "job_id": run.job_id,
        "pipeline_version": PIPELINE_VERSION,
        "video": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": frames_total,
            "duration_s": video.duration_s,
            "timeline_source": timeline.source,
        },
        "outcome": outcome,
        "frame_stride": stride,
        "tracking": plan.report(),
        "frames_total": frames_total,
        "frames_processed": n_processed,
        "n_frame_errors": n_errors,
        "n_detections": n_detections,
        "n_tracks": run.n_tracks,
        "n_tracks_confirmed": len(rows),
        "n_tracks_rejected_short": run.n_rejected,
        "n_observations_with_position": n_with,
        "n_observations_without_position": n_without,
        "gps": gps_info,
        "detector": run.detector.info(),
        "elapsed_s": round(time.monotonic() - started, 3),
        "warnings": warnings,
    }

    # One transaction: observations, done status, active job.
    run.report_progress("saving", 1.0, force=True)
    obs_db.insert_observations(run.conn, run.session_id, run.job_id, rows)
    jobs_db.mark_done(run.conn, run.job_id, report)
    run.conn.commit()
    if run.progress is not None:
        run.progress("done", 1.0)
    return report


def process_session(
    conn: sqlite3.Connection,
    session_id: str,
    detector: DetectorLike,
    cfg: AppConfig,
    *,
    job_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Detect potholes in a stored session and return the report (also ``summary_json``).

    Creates a job when ``job_id`` is None (otherwise uses that ``queued`` job),
    runs the pipeline, and commits. Raises ``PipelineError`` if the session does
    not exist or the run fails; in the latter case the job is ``failed`` with the
    error text and has no observations. ``progress(stage, fraction)`` is called
    when progress is written.
    """
    session = _session_row(conn, session_id)
    if session is None:
        raise PipelineError(f"no such session: {session_id}")

    out_dir = None
    try:
        if job_id is None:
            job_id = jobs_db.create_job(conn, session_id)
        jobs_db.mark_running(
            conn,
            job_id,
            stage=READING_STAGE,
            progress=None,
            pipeline_version=PIPELINE_VERSION,
            model_sha256=getattr(detector, "weights_sha256", None),
            params=_params(cfg),
        )
        conn.commit()
        run = _Run(conn, session, detector, cfg, job_id, progress)
        out_dir = run.out_dir
        return _execute(run)
    except Exception as exc:
        message = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
        try:
            conn.rollback()
            if job_id is not None:
                jobs_db.mark_failed(conn, job_id, message)
                conn.commit()
        except Exception:  # the database itself is failing; report the original error
            pass
        if out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(message) from exc
