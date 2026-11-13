"""System endpoints: health, system description, dashboard statistics, demo route example."""

import json
from typing import Any

from fastapi import APIRouter, Request

from roadsense import __version__
from roadsense.api.deps import AppState, db_conn, display_path, get_state
from roadsense.api.errors import ApiError
from roadsense.api.routes import cost_model_settings
from roadsense.mapstate import get_include_synthetic
from roadsense.storage import map_queries as mq

router = APIRouter(prefix="/api")

ARCHITECTURE = "YOLOv8s detection, single class 'pothole'"
JOB_STATUSES = ("done", "running", "queued", "failed")
GPS_SOURCES = ("recorded", "synthetic", "none")
POTHOLE_SOURCES = ("recorded", "synthetic", "mixed")  # "mixed" is never produced any more


def demo_status(conn) -> dict[str, Any]:
    """Whether demonstration (synthetic) data exists and whether the map shows it.

    ``mode`` is the remembered switch, ``sessions`` the number of synthetic
    sessions stored, and ``active`` is true when the switch is on and the map
    therefore contains synthetic data. The interface shows a permanent notice
    while ``active`` is true.
    """
    mode = get_include_synthetic(conn)
    n = conn.execute("SELECT COUNT(*) FROM sessions WHERE gps_source = 'synthetic'").fetchone()[0]
    counted = conn.execute(
        "SELECT COUNT(*) FROM sessions s JOIN jobs j ON j.id = s.active_job_id"
        " WHERE s.gps_source = 'synthetic' AND s.include_in_map = 1 AND j.status = 'done'"
    ).fetchone()[0]
    return {"mode": mode, "sessions": int(n), "active": bool(mode and counted)}


def _loaded_device(state: AppState) -> tuple[bool, str | None]:
    """Whether the worker has built a detector, and its device. Never builds one."""
    detector = state.runner.detector
    if detector is None:
        return False, None
    device = getattr(detector, "device", None)
    if device is None:
        try:
            device = detector.info().get("device")
        except Exception:  # a broken info() must not break /api/system
            device = None
    return True, None if device is None else str(device)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/system")
def system(request: Request) -> dict[str, Any]:
    state = get_state(request)
    cfg = state.cfg
    loaded, device = _loaded_device(state)
    with db_conn(state) as conn:
        include_synthetic = get_include_synthetic(conn)
        demo = demo_status(conn)
    weights_sha = state.weights_sha256()
    return {
        "version": __version__,
        "model": {
            "weights": display_path(cfg.detection.weights),
            "exists": weights_sha is not None,
            "sha256": weights_sha,
            "loaded": loaded,
            "device": device,
            "architecture": ARCHITECTURE,
        },
        "graph": state.graph_info(),
        "settings": {
            "detection": {
                "conf": cfg.detection.conf,
                "iou": cfg.detection.iou,
                "imgsz": cfg.detection.imgsz,
                "partial_frame_error_fraction": cfg.detection.partial_frame_error_fraction,
                "max_frame_error_fraction": cfg.detection.max_frame_error_fraction,
            },
            "tracking": {
                "frame_stride": cfg.tracking.frame_stride,
                "iou_match": cfg.tracking.iou_match,
                "max_age_frames": cfg.tracking.max_age_frames,
                "min_hits": cfg.tracking.min_hits,
                "vehicle_speed_mps": cfg.tracking.vehicle_speed_mps,
                "vehicle_frame_stride": cfg.tracking.vehicle_frame_stride,
                "vehicle_min_hits": cfg.tracking.vehicle_min_hits,
            },
            "mapping": {
                "dedup_radius_m": cfg.mapping.dedup_radius_m,
                "max_snap_m": cfg.mapping.max_snap_m,
            },
            "ingest": {
                "max_gps_accuracy_m": cfg.ingest.max_gps_accuracy_m,
                "max_gps_gap_s": cfg.ingest.max_gps_gap_s,
                "max_gps_extrapolation_s": cfg.ingest.max_gps_extrapolation_s,
            },
            "routing": cost_model_settings(state),
        },
        "include_synthetic": include_synthetic,
        "demo": demo,
    }


@router.get("/stats")
def stats(request: Request) -> dict[str, Any]:
    state = get_state(request)
    with db_conn(state) as conn:
        by_status = dict.fromkeys(JOB_STATUSES, 0)
        for status, n in conn.execute(
            "SELECT j.status, COUNT(*) FROM sessions s JOIN jobs j ON j.id = ("
            "  SELECT id FROM jobs WHERE session_id = s.id ORDER BY created_at DESC, rowid DESC LIMIT 1)"
            " GROUP BY j.status"
        ):
            by_status[status] = n
        by_gps = dict.fromkeys(GPS_SOURCES, 0)
        for source, n in conn.execute("SELECT gps_source, COUNT(*) FROM sessions GROUP BY 1"):
            by_gps[source] = n
        total_sessions = sum(by_gps.values())

        n_obs, n_pos = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(o.cam_lat IS NOT NULL AND o.cam_lon IS NOT NULL), 0) "
            "FROM observations o JOIN sessions s ON s.id = o.session_id AND s.active_job_id = o.job_id"
        ).fetchone()

        build_id = mq.latest_build_id(conn)
        pot_by_source = dict.fromkeys(POTHOLE_SOURCES, 0)
        on_road = 0
        build = None
        # recorded and synthetic coverage are separate numbers, never one total
        cover = mq.coverage_totals(conn)
        demo = demo_status(conn)
        if build_id is not None:
            for source, n in conn.execute(
                "SELECT gps_source, COUNT(*) FROM potholes WHERE build_id = ? GROUP BY 1",
                (build_id,),
            ):
                pot_by_source[source] = n
            on_road = conn.execute(
                "SELECT COUNT(*) FROM potholes WHERE build_id = ? AND edge_u IS NOT NULL",
                (build_id,),
            ).fetchone()[0]
            created_at, graph_sha, params = conn.execute(
                "SELECT created_at, graph_sha256, params_json FROM builds WHERE id = ?",
                (build_id,),
            ).fetchone()
            build = {
                "id": build_id,
                "created_at": created_at,
                "has_graph": graph_sha is not None,
                "include_synthetic": bool(json.loads(params).get("include_synthetic", True)),
            }
    pot_total = sum(pot_by_source.values())
    n_demo_potholes = pot_total - pot_by_source["recorded"]
    return {
        "sessions": {"total": total_sessions, "by_status": by_status, "by_gps_source": by_gps},
        "observations": {
            "total": int(n_obs),
            "with_position": int(n_pos),
            "without_position": int(n_obs) - int(n_pos),
        },
        "potholes": {
            "total": pot_total,
            "by_gps_source": pot_by_source,
            "on_road": int(on_road),
            "off_road": pot_total - int(on_road),
        },
        # surveyed_m and edges describe real roads (recorded sessions only)
        "coverage": {
            "surveyed_m": float(cover["surveyed_m"]),
            "edges": int(cover["edges"]),
            "demo_m": float(cover["demo_m"]),
            "demo_edges": int(cover["demo_edges"]),
        },
        "demo": {**demo, "potholes": int(n_demo_potholes)},
        "build": build,
    }


@router.get("/demo/route-example")
def demo_route_example(request: Request) -> Any:
    """The prepared demonstration route, only while its synthetic session is on the map.

    The file outlives a database reset and a deleted session, so it is checked
    against the database: without the session it names, or with demo mode off,
    the example would lead nowhere and is reported as absent (404).
    """
    state = get_state(request)
    path = state.cfg.data_dir / "demo" / "demo_route.json"
    try:
        example = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ApiError(404, "not_found", "No demonstration route has been prepared.") from None
    except (OSError, ValueError) as exc:
        raise ApiError(404, "not_found", f"The demonstration route file is unreadable: {exc}") from exc
    session_id = example.get("session_id") if isinstance(example, dict) else None
    with db_conn(state) as conn:
        usable = (
            session_id is not None
            and get_include_synthetic(conn)
            and conn.execute(
                "SELECT 1 FROM sessions s JOIN jobs j ON j.id = s.active_job_id"
                " WHERE s.id = ? AND s.gps_source = 'synthetic' AND s.include_in_map = 1"
                " AND j.status = 'done'",
                (session_id,),
            ).fetchone()
            is not None
        )
    if not usable:
        raise ApiError(
            404,
            "not_found",
            "The demonstration route is not available: its synthetic session is not on the map "
            "(demo mode is off, or the session was removed).",
        )
    return example
