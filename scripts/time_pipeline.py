"""Wall-clock time of the real pipeline (real detector) on recordings: old vs new stride.

Review tool. For each recording and each variant it registers the recording in a
THROWAWAY database and upload/artifact folder under ``--work`` (default
``data/review_20261113/timing``; it never touches ``data/roadsense.sqlite``) and
times ``process_session`` with the trained detector, which is loaded and warmed up
once beforehand.

* ``old``: every session uses the previous settings, stride 3 and 3 hits (for a
  vehicle recording: ``vehicle_frame_stride=3``, ``vehicle_min_hits=3``).
* ``new``: the shipped ``configs/app.yaml`` (vehicle sessions: stride 1, 6 hits).

Both variants run the current tracker and the current image handling, so the
comparison isolates the stride and thresholds. The variants alternate, ``--repeat``
times each.

    python scripts/time_pipeline.py [--repeat N] [--work DIR] NAME [NAME ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadsense.config import TrackingConfig, load_app_config  # noqa: E402
from roadsense.pipeline.intake import create_session_from_files  # noqa: E402
from roadsense.pipeline.process import process_session  # noqa: E402
from roadsense.storage import db  # noqa: E402


def find_video(name: str) -> Path:
    (path,) = (ROOT / "data" / "recordings").glob(f"**/{name}/{name}.mp4")
    return path


def variant_config(base, variant: str, work: Path):
    tracking = base.tracking
    if variant == "old":
        tracking = TrackingConfig(**{**base.tracking.model_dump(), "vehicle_frame_stride": 3, "vehicle_min_hits": 3})
    uploads = base.uploads.model_copy(update={"dir": work / "uploads", "artifacts_dir": work / "artifacts"})
    return base.model_copy(update={"tracking": tracking, "uploads": uploads, "db_path": work / "timing.sqlite"})


def run_once(name: str, variant: str, base, detector, work: Path) -> dict:
    run_dir = work / f"{variant}_{name}"
    if run_dir.exists():
        shutil.rmtree(run_dir)  # only ever our own scratch folder
    cfg = variant_config(base, variant, run_dir)
    run_dir.mkdir(parents=True)
    video = find_video(name)
    conn = db.connect(cfg.db_path)
    try:
        db.init_schema(conn)
        sid = create_session_from_files(
            conn, cfg, video=video, gps=video.with_name(f"{name}_gps.csv"), sidecar=video.with_name(f"{name}.json")
        )
        conn.commit()
        t0 = time.perf_counter()
        report = process_session(conn, sid, detector, cfg)
        wall = time.perf_counter() - t0
    finally:
        conn.close()
    return {
        "wall_s": wall,
        "elapsed_s": report["elapsed_s"],
        "mode": report["tracking"]["mode"],
        "stride": report["frame_stride"],
        "processed": report["frames_processed"],
        "total": report["frames_total"],
        "tracks": report["n_tracks"],
        "confirmed": report["n_tracks_confirmed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--work", type=Path, default=ROOT / "data" / "review_20261113" / "timing")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    if ROOT / "data" / "roadsense.sqlite" in (work, *work.parents):  # pragma: no cover
        raise SystemExit("refusing to work inside the real database")
    base = load_app_config(ROOT / "configs" / "app.yaml")

    from roadsense.detection.detector import Detector

    t = time.perf_counter()
    detector = Detector(base.detection)
    constructed = time.perf_counter() - t
    import numpy as np

    t = time.perf_counter()
    detector.detect(np.zeros((720, 1280, 3), np.uint8))
    print(f"detector construct {constructed:.2f}s, warm-up inference {time.perf_counter() - t:.2f}s, "
          f"device {detector.device}", flush=True)

    for name in args.names:
        for rep in range(args.repeat):
            for variant in ("old", "new"):
                r = run_once(name, variant, base, detector, work)
                print(
                    f"{name} run {rep + 1} {variant:<3} mode={r['mode']:<8} stride={r['stride']} "
                    f"analysed {r['processed']}/{r['total']} frames  process_session wall {r['wall_s']:.1f}s "
                    f"(report elapsed {r['elapsed_s']:.1f}s)  tracks {r['tracks']} confirmed {r['confirmed']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
