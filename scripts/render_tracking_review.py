"""Render annotated tracking videos and contact sheets from saved detections.

Review tool. Draws the saved per-frame detections (see ``compare_trackers.py``),
grouped by the CURRENT tracker with the settings the pipeline picks for the
recording, on the original video: confirmed tracks are coloured per id with a
short trail, unconfirmed tracks are grey and marked with an ``x``. No inference
is run. A video holds the analysed frames only, played at ``fps / stride``.

    python scripts/render_tracking_review.py [--out DIR]

Default output: ``data/review_20261113/tracking/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_trackers import ROOT, load_detections, plan_for, replay_new  # noqa: E402

from roadsense.config import load_app_config  # noqa: E402
from roadsense.ingest.video import iter_frames, probe_video, read_timeline  # noqa: E402

VIDEOS = ["road_20261013_180932", "road_20261013_173922"]
# (recording, first frame, last frame, step, file name)
SHEETS = [
    ("road_20261013_180932", 757, 790, 3, "sheet_180932_f757-790.jpg"),
    ("road_20261013_180932", 1068, 1086, 3, "sheet_180932_f1068-1086.jpg"),
    ("road_20261013_165247", 262, 280, 2, "sheet_165247_f262-280.jpg"),
    ("road_20261013_165247", 350, 360, 1, "sheet_165247_f350-360.jpg"),
]
TRAIL_POINTS = 40
OUT_W = 960


def video_path(name: str) -> Path:
    (path,) = (ROOT / "data" / "recordings").glob(f"**/{name}.mp4")
    return path


def color_of(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(v) for v in rng.integers(70, 256, 3))


def annotate(image, index, t, assigned, confirmed, history, scale):
    img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for track_id, det in assigned:
        ok = confirmed[track_id]
        color = color_of(track_id) if ok else (150, 150, 150)
        x1, y1, x2, y2 = (int(round(v * scale)) for v in (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if ok else 1)
        cv2.putText(img, f"{track_id}{'' if ok else 'x'} {det.confidence:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  # fmt: skip
        if ok:
            pts = [(int(cx * scale), int(cy * scale)) for f, cx, cy in history[track_id] if f <= index]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts[-TRAIL_POINTS:], np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {index}  t={t:.2f}s", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    return img


def render(name: str, cfg, out_dir: Path, write_video: bool) -> None:
    frames = load_detections(name)
    plan = plan_for(name, frames, cfg)
    _, assignments, confirmed = replay_new(frames, cfg, plan)
    by_index = dict(assignments)
    history: dict[int, list[tuple[int, float, float]]] = {}
    for index, assigned in assignments:
        for track_id, d in assigned:
            history.setdefault(track_id, []).append((index, (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2))

    path = video_path(name)
    info = probe_video(path)
    scale = OUT_W / info.width
    size = (OUT_W, round(info.height * scale))
    writer = None
    if write_video:
        out = out_dir / f"{name}_new_tracker.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps / plan.frame_stride, size)
        if not writer.isOpened():
            raise SystemExit(f"cannot open a video writer for {out}")
    sheet_frames = {s[4]: set(range(s[1], s[2] + 1, s[3])) for s in SHEETS if s[0] == name}
    tiles: dict[str, list[np.ndarray]] = {fname: [] for fname in sheet_frames}
    timeline = read_timeline(path)
    for frame in iter_frames(path, plan.frame_stride, timeline):
        if frame.index not in by_index:
            continue
        img = annotate(frame.image, frame.index, frame.t_video_s, by_index[frame.index], confirmed, history, scale)
        if writer is not None:
            writer.write(img)
        for fname, indices in sheet_frames.items():
            if frame.index in indices:
                tiles[fname].append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    if writer is not None:
        writer.release()
        print(f"wrote {out}  ({plan.mode}, stride {plan.frame_stride})")
    for fname, items in tiles.items():
        if not items:
            continue
        cols = 3
        while len(items) % cols:
            items.append(np.zeros_like(items[0]))
        rows = [np.hstack(items[i : i + cols]) for i in range(0, len(items), cols)]
        cv2.imwrite(str(out_dir / fname), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {out_dir / fname}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "review_20261113" / "tracking")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_app_config(ROOT / "configs" / "app.yaml")
    for name in dict.fromkeys(VIDEOS + [s[0] for s in SHEETS]):
        render(name, cfg, args.out, write_video=name in VIDEOS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
