"""Helpers for the API, CLI and intake tests: temporary config, GPS files, upload and polling."""

import time
from pathlib import Path

from geo_fixtures import xy

from roadsense.config import AppConfig

# Frames 0..11 hold one bright box, so the fake detector tracks one pothole that is
# last seen at t = 0.9 s (30 frames at 10 fps).
VIDEO_FRAMES = 30


def make_api_cfg(tmp_path: Path, **sections) -> AppConfig:
    """An AppConfig whose database, uploads, artifacts and graph all live under ``tmp_path``."""
    data = {
        "data_dir": str(tmp_path / "data"),
        "db_path": str(tmp_path / "data" / "app.sqlite"),
        "uploads": {
            "dir": str(tmp_path / "data" / "uploads"),
            "artifacts_dir": str(tmp_path / "data" / "artifacts"),
        },
        "routing": {"graph_path": str(tmp_path / "data" / "graph.json")},
        "detection": {"weights": str(tmp_path / "weights.pt")},
    }
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return AppConfig.model_validate(data)


def grid_fixes(n=6, t0=-1.0, x0=50.0, speed=20.0, y=0.0):
    """Fixes every second moving east along y = ``y`` of the geo_fixtures grid."""
    return [(t0 + i, *xy(x0 + speed * (t0 + i), y)) for i in range(n)]


def write_gps_csv(path: Path, fixes, accuracy=5.0) -> Path:
    lines = ["timestamp_sec,lat,lon,accuracy_m"]
    lines += [f"{t:.3f},{lat:.7f},{lon:.7f},{accuracy}" for t, lat, lon in fixes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def post_session(client, video: Path, gps: Path | None = None, sidecar: Path | None = None, **fields):
    """POST /api/sessions as a browser would; ``fields`` become form fields."""
    files = {"video": (video.name, video.read_bytes(), "video/mp4")}
    if gps is not None:
        files["gps"] = (gps.name, gps.read_bytes(), "text/csv")
    if sidecar is not None:
        files["sidecar"] = (sidecar.name, sidecar.read_bytes(), "application/json")
    data = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in fields.items()}
    return client.post("/api/sessions", files=files, data=data)


def wait_job(client, session_id: str, timeout: float = 30.0) -> dict:
    """Poll the status endpoint until the job is done or failed; returns the job object."""
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/api/sessions/{session_id}/status").json()["job"]
        if job["status"] in ("done", "failed"):
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job still {job['status']} after {timeout} s")
        time.sleep(0.05)
