"""Map endpoints: potholes, coverage and rebuild."""

import math

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from roadsense.api.deps import db_conn, get_state
from roadsense.api.errors import ApiError
from roadsense.api.views import observation_view, pothole_view
from roadsense.mapstate import write_and_rebuild
from roadsense.storage import map_queries as mq

router = APIRouter(prefix="/api")

POTHOLE_NOTE = (
    "Positions are the camera position when the pothole was last seen; "
    "the pothole is a few metres ahead."
)


def parse_bbox(text: str | None) -> mq.BBox | None:
    """``min_lat,min_lon,max_lat,max_lon`` -> tuple; ``ApiError`` 422 when malformed."""
    if text is None or not text.strip():
        return None
    bad = ApiError(
        422,
        "invalid_input",
        "bbox must be 'min_lat,min_lon,max_lat,max_lon' with finite numbers and min <= max.",
    )
    try:
        values = [float(p) for p in text.split(",")]
    except ValueError:
        raise bad from None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise bad
    min_lat, min_lon, max_lat, max_lon = values
    if min_lat > max_lat or min_lon > max_lon:
        raise bad
    return (min_lat, min_lon, max_lat, max_lon)


@router.get("/potholes")
def list_potholes(request: Request, bbox: str | None = Query(None)) -> dict:
    box = parse_bbox(bbox)
    with db_conn(get_state(request)) as conn:
        return {
            "potholes": [pothole_view(p) for p in mq.list_potholes(conn, box)],
            "note": POTHOLE_NOTE,
        }


@router.get("/potholes/{pothole_id}")
def get_pothole(pothole_id: int, request: Request) -> dict:
    with db_conn(get_state(request)) as conn:
        p = mq.get_pothole(conn, pothole_id)
        if p is None:
            raise ApiError(404, "not_found", f"No pothole with id {pothole_id}.")
        observations = []
        for o in p["observations"]:
            view = observation_view(o)
            view["session_id"] = o["session_id"]
            view["session_name"] = o["session_name"]
            observations.append(view)
        return {**pothole_view(p), "observations": observations}


@router.get("/coverage")
def get_coverage(request: Request, bbox: str | None = Query(None)) -> dict:
    box = parse_bbox(bbox)
    state = get_state(request)
    if state.graph is None:
        return {"stretches": [], "surveyed_m": 0.0, "demo_stretches": [], "demo_m": 0.0}
    with db_conn(state) as conn:
        stretches = mq.list_coverage(conn, state.graph, box)
    real = [s for s in stretches if s["source"] == "recorded"]
    demo = [s for s in stretches if s["source"] == "synthetic"]
    return {
        # surveyed road: filmed by a recorded session
        "stretches": [s["coords"] for s in real],
        "surveyed_m": sum(s["end_m"] - s["start_m"] for s in real),
        # road a synthetic (demonstration) trace was generated along; not a survey
        "demo_stretches": [s["coords"] for s in demo],
        "demo_m": sum(s["end_m"] - s["start_m"] for s in demo),
    }


class RebuildBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_synthetic: bool | None = None


@router.post("/map/rebuild")
def rebuild(request: Request, body: RebuildBody | None = None) -> dict:
    state = get_state(request)
    choice = None if body is None else body.include_synthetic
    with db_conn(state) as conn:
        with state.map_lock:
            summary = write_and_rebuild(conn, state.cfg, state.graph, include_synthetic=choice)
    return summary
