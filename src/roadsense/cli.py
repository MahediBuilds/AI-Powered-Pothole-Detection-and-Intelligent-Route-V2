"""Command line: add recordings, rebuild the map, list sessions, reset the demo state.

    python -m roadsense.cli add VIDEO [--gps CSV] [--sidecar JSON] ...
    python -m roadsense.cli add-dir DIR
    python -m roadsense.cli rebuild-map [--exclude-synthetic]
    python -m roadsense.cli list
    python -m roadsense.cli reset --yes

``add`` and ``add-dir`` register the files through the same intake code as the
web upload and, unless ``--no-process`` is given, run the detector in this
process and rebuild the map. ``add``, ``add-dir`` and ``reset`` take the
database's process lock, so they refuse to run while the web server (or another
such command) is using the same database. ``rebuild-map`` and ``list`` are safe
at any time.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from roadsense.config import AppConfig, ConfigError, load_app_config, project_root
from roadsense.ingest import IngestError
from roadsense.mapstate import load_configured_graph, write_and_rebuild
from roadsense.pipeline.intake import ROLES, VIDEO_EXTENSIONS, create_session_from_files
from roadsense.pipeline.process import PipelineError, process_session
from roadsense.storage import db
from roadsense.storage.proclock import AlreadyRunningError, ProcessLock, lock_path_for

DetectorFactory = Callable[[], Any]

EXIT_ERROR = 1
EXIT_USAGE = 2


class _Context:
    """The loaded config, the road graph and a detector built on first use."""

    def __init__(self, cfg: AppConfig, detector_factory: DetectorFactory | None) -> None:
        self.cfg = cfg
        self._factory = detector_factory
        self._detector: Any = None
        self.graph, graph_error = load_configured_graph(cfg)
        if graph_error:
            print(f"warning: road graph ignored: {graph_error}", file=sys.stderr)
        elif self.graph is None:
            print(
                f"note: no road graph at {cfg.routing.graph_path}; potholes will not be put on roads",
                file=sys.stderr,
            )

    def detector(self) -> Any:
        if self._detector is None:
            if self._factory is not None:
                self._detector = self._factory()
            else:
                from roadsense.detection.detector import Detector

                self._detector = Detector(self.cfg.detection)
        return self._detector

    def connect(self):
        conn = db.connect(self.cfg.db_path)
        db.init_schema(conn)
        return conn


# ---------------------------------------------------------------------------- add


def find_videos(directory: Path) -> list[Path]:
    """Every video file below ``directory`` (recursively), sorted by path."""
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _sibling(video: Path, suffix: str) -> Path | None:
    candidate = video.with_name(f"{video.stem}{suffix}")
    return candidate if candidate.is_file() else None


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"  processed {report['frames_processed']} of {report['frames_total']} frames; "
        f"{report['n_tracks_confirmed']} pothole(s) tracked "
        f"({report['n_observations_with_position']} with a GPS position, "
        f"{report['n_observations_without_position']} without)"
    )
    for warning in report["warnings"]:
        print(f"  warning: {warning}")


def _add_video(ctx: _Context, conn, video: Path, args: argparse.Namespace) -> str | None:
    """Register (and unless ``--no-process`` process) one video; returns its session id."""
    gps = Path(args.gps) if args.gps else _sibling(video, "_gps.csv")
    sidecar = Path(args.sidecar) if args.sidecar else _sibling(video, ".json")
    if args.synthetic_gps and gps is None:
        print(f"error: {video}: --synthetic-gps needs a GPS file", file=sys.stderr)
        return None
    try:
        session_id = create_session_from_files(
            conn,
            ctx.cfg,
            video=video,
            gps=gps,
            sidecar=sidecar,
            role=args.role,
            gps_source="synthetic" if args.synthetic_gps else None,
            include_in_map=not args.no_map,
        )
        conn.commit()
    except (IngestError, OSError) as exc:
        conn.rollback()
        print(f"error: {video}: {exc}", file=sys.stderr)
        return None
    gps_note = "no GPS file" if gps is None else f"GPS {gps.name}"
    print(f"added {video.name} as session {session_id} ({gps_note})")
    if args.no_process:
        return session_id
    try:
        _print_report(process_session(conn, session_id, ctx.detector(), ctx.cfg))
    except PipelineError as exc:
        print(f"error: processing {video.name} failed: {exc}", file=sys.stderr)
        return None
    return session_id


def _add_videos(ctx: _Context, videos: Sequence[Path], args: argparse.Namespace) -> int:
    try:
        lock = ProcessLock(lock_path_for(ctx.cfg.db_path), role="command line").acquire()
    except AlreadyRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        return _add_videos_locked(ctx, videos, args)
    finally:
        lock.release()


def _add_videos_locked(ctx: _Context, videos: Sequence[Path], args: argparse.Namespace) -> int:
    conn = ctx.connect()
    n_ok = n_failed = 0
    try:
        for video in videos:
            try:
                ok = _add_video(ctx, conn, video, args) is not None
            except Exception as exc:  # detector could not be built, or a bug: stop cleanly
                print(f"error: {video.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                n_failed += 1
                break
            n_ok += ok
            n_failed += not ok
        if n_ok and not args.no_process:
            conn.rollback()  # write_and_rebuild needs a connection with no open transaction
            summary = write_and_rebuild(conn, ctx.cfg, ctx.graph)
            print(
                f"map rebuilt: {summary['n_potholes']} pothole(s) "
                f"({summary['n_potholes_on_road']} on a road), "
                f"{summary['surveyed_m']:.0f} m surveyed"
            )
    finally:
        conn.close()
    print(f"{n_ok} added, {n_failed} failed")
    return EXIT_ERROR if n_failed else 0


def cmd_add(ctx: _Context, args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.is_file():
        print(f"error: {video}: video file not found", file=sys.stderr)
        return EXIT_ERROR
    return _add_videos(ctx, [video], args)


def cmd_add_dir(ctx: _Context, args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"error: {directory}: not a directory", file=sys.stderr)
        return EXIT_ERROR
    videos = find_videos(directory)
    if not videos:
        print(f"error: no video files ({', '.join(VIDEO_EXTENSIONS)}) found under {directory}", file=sys.stderr)
        return EXIT_ERROR
    return _add_videos(ctx, videos, args)


# ---------------------------------------------------------------------------- other commands


def cmd_rebuild_map(ctx: _Context, args: argparse.Namespace) -> int:
    conn = ctx.connect()
    try:
        summary = write_and_rebuild(
            conn, ctx.cfg, ctx.graph, include_synthetic=args.include_synthetic
        )
    finally:
        conn.close()
    print(
        f"map rebuilt (build {summary['build_id']}): {summary['n_sessions']} session(s), "
        f"{summary['n_potholes']} pothole(s) ({summary['n_potholes_on_road']} on a road), "
        f"{summary['surveyed_m']:.0f} m surveyed by recorded sessions; "
        f"demo mode {'ON: ' + str(summary['n_potholes_synthetic']) + ' synthetic pothole(s) shown, kept apart from recorded data' if summary['has_synthetic'] else 'off or no synthetic session'}"
    )
    return 0


def cmd_list(ctx: _Context, args: argparse.Namespace) -> int:
    conn = ctx.connect()
    try:
        rows = conn.execute(
            "SELECT s.id, s.name, s.role, s.gps_source, s.include_in_map,"
            " (SELECT status FROM jobs WHERE session_id = s.id"
            "  ORDER BY created_at DESC, rowid DESC LIMIT 1) AS status,"
            " (SELECT COUNT(*) FROM observations o WHERE o.job_id = s.active_job_id) AS n_obs,"
            " (SELECT COUNT(*) FROM observations o WHERE o.job_id = s.active_job_id"
            "  AND o.cam_lat IS NOT NULL AND o.cam_lon IS NOT NULL) AS n_pos"
            " FROM sessions s ORDER BY s.created_at, s.rowid"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print("no sessions")
        return 0
    print(f"{'id':32}  {'name':26}  {'role':7}  {'gps':9}  {'status':7}  {'potholes':>8}  {'positioned':>10}  map")
    for r in rows:
        print(
            f"{r['id']:32}  {r['name'][:26]:26}  {r['role']:7}  {r['gps_source']:9}  "
            f"{r['status'] or '-':7}  {r['n_obs']:>8}  {r['n_pos']:>10}  "
            f"{'yes' if r['include_in_map'] else 'no'}"
        )
    return 0


def _protected(path: Path, cfg: AppConfig) -> bool:
    """True if deleting ``path`` would remove the project, the data directory or the web files."""
    path = path.resolve()
    keep = [project_root(), cfg.data_dir, cfg.web_dir, cfg.runs_dir, cfg.db_path.parent]
    return any(path == k.resolve() or path in k.resolve().parents for k in keep)


def cmd_reset(ctx: _Context, args: argparse.Namespace) -> int:
    cfg = ctx.cfg
    if not args.yes:
        print(
            "refusing to reset: this deletes the database, all uploads and all artifacts.\n"
            f"  database: {cfg.db_path}\n  uploads:  {cfg.uploads.dir}\n  artifacts: {cfg.uploads.artifacts_dir}\n"
            "Run again with --yes to confirm.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    for directory in (cfg.uploads.dir, cfg.uploads.artifacts_dir):
        if _protected(Path(directory), cfg):
            print(f"error: refusing to delete {directory}: it contains project files", file=sys.stderr)
            return EXIT_ERROR
    try:  # never delete the database under a running server
        lock = ProcessLock(lock_path_for(cfg.db_path), role="command line").acquire()
    except AlreadyRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:  # the lock is held for the whole deletion; the lock file itself is kept
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{cfg.db_path}{suffix}")
            if path.is_file():
                path.unlink()
                print(f"deleted {path}")
        for directory in (cfg.uploads.dir, cfg.uploads.artifacts_dir):
            if Path(directory).is_dir():
                shutil.rmtree(directory)
                print(f"deleted {directory}")
    finally:
        lock.release()
    print("reset done")
    return 0


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=str(project_root() / "configs" / "app.yaml"),
        help="app config YAML (default: configs/app.yaml)",
    )

    add_opts = argparse.ArgumentParser(add_help=False)
    add_opts.add_argument("--role", choices=ROLES, default="survey")
    add_opts.add_argument(
        "--synthetic-gps", action="store_true", help="declare the GPS trace generated, not recorded"
    )
    add_opts.add_argument("--no-map", action="store_true", help="do not include the session in the map")
    add_opts.add_argument(
        "--no-process", action="store_true", help="only register the files; do not run the detector"
    )

    parser = argparse.ArgumentParser(
        prog="python -m roadsense.cli",
        description="RoadSense command line. Processing runs in this process: do not use it on a "
        "database while the web server is processing jobs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", parents=[common, add_opts], help="register and process one recording")
    p.add_argument("video")
    p.add_argument("--gps", help="GPS CSV (default: VIDEO's <stem>_gps.csv if present)")
    p.add_argument("--sidecar", help="recorder JSON (default: VIDEO's <stem>.json if present)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser(
        "add-dir", parents=[common, add_opts], help="add every video found under a folder"
    )
    p.add_argument("directory")
    p.set_defaults(func=cmd_add_dir, gps=None, sidecar=None)

    p = sub.add_parser("rebuild-map", parents=[common], help="rebuild potholes and coverage")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--include-synthetic", dest="include_synthetic", action="store_const", const=True,
        help="turn demo mode on: show synthetic (demonstration) sessions, kept apart from recorded data",
    )  # fmt: skip
    group.add_argument(
        "--exclude-synthetic", dest="include_synthetic", action="store_const", const=False,
        help="turn demo mode off: leave out sessions with a synthetic GPS trace",
    )  # fmt: skip
    p.set_defaults(func=cmd_rebuild_map, include_synthetic=None)

    p = sub.add_parser("list", parents=[common], help="list sessions with status and counts")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser(
        "reset", parents=[common], help="delete the database, uploads and artifacts (needs --yes)"
    )
    p.add_argument("--yes", action="store_true", help="confirm the deletion")
    p.set_defaults(func=cmd_reset)
    return parser


def main(argv: Sequence[str] | None = None, *, detector_factory: DetectorFactory | None = None) -> int:
    """Run the CLI; ``detector_factory`` replaces the real detector (used by tests)."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_app_config(Path(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return args.func(_Context(cfg, detector_factory), args)


if __name__ == "__main__":
    raise SystemExit(main())
