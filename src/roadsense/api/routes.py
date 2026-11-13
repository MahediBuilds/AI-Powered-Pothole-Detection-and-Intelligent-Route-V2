"""Routing endpoints: availability info and route planning."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from roadsense.api.deps import AppState, db_conn, get_state
from roadsense.api.errors import ApiError
from roadsense.mapstate import write_and_rebuild
from roadsense.routing.router import (
    NoGraphError,
    NoRouteError,
    OutsideGraphError,
    plan_routes,
)
from roadsense.storage import map_queries as mq
from roadsense.storage.db import GraphMismatchError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: Point
    destination: Point
    avoidance: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)


def cost_model_settings(state: AppState) -> dict:
    r = state.cfg.routing
    return {
        "max_detour_per_pothole_m": r.max_detour_per_pothole_m,
        "assumed_rate_per_100m": r.assumed_rate_per_100m,
        "prior_strength_m": r.prior_strength_m,
        "unsurveyed_rate_per_100m": r.unsurveyed_rate_per_100m,
        "min_surveyed_fraction": r.min_surveyed_fraction,
        "adequate_surveyed_fraction": r.adequate_surveyed_fraction,
    }


@router.get("/routes/info")
def routes_info(request: Request) -> dict:
    state = get_state(request)
    with db_conn(state) as conn:
        build = conn.execute("SELECT MAX(id) FROM builds").fetchone()[0]
        on_road = demo_on_road = 0
        if build is not None:
            on_road, demo_on_road = conn.execute(
                "SELECT COALESCE(SUM(gps_source = 'recorded'), 0),"
                " COALESCE(SUM(gps_source != 'recorded'), 0)"
                " FROM potholes WHERE build_id = ? AND edge_u IS NOT NULL",
                (build,),
            ).fetchone()
        cover = mq.coverage_totals(conn)
    return {
        "available": state.graph is not None,
        "graph": state.graph_info(),
        # real roads only (recorded sessions)
        "coverage": {"surveyed_m": float(cover["surveyed_m"]), "edges": int(cover["edges"])},
        "potholes_on_road": int(on_road),
        # demonstration data, present only in demo mode
        "demo": {
            "active": bool(cover["demo_m"] or demo_on_road),
            "demo_m": float(cover["demo_m"]),
            "potholes_on_road": int(demo_on_road),
        },
        "cost_model": cost_model_settings(state),
    }


@router.post("/routes")
def post_routes(body: RouteRequest, request: Request) -> dict:
    state = get_state(request)
    if state.graph is None:
        raise ApiError(
            409,
            "no_graph",
            "No road network is loaded, so routes cannot be computed. Download one with "
            "'python -m roadsense.routing.graph' and restart the server.",
        )
    origin = (body.origin.lat, body.origin.lon)
    destination = (body.destination.lat, body.destination.lon)

    def plan() -> dict:
        with db_conn(state) as conn:
            return plan_routes(
                conn, state.graph, origin, destination, body.avoidance,
                state.cfg.routing, state.cfg.mapping,
            )  # fmt: skip

    try:
        try:
            return plan()
        except GraphMismatchError:
            # The stored map was built with another graph: rebuild once and retry.
            log.warning("map was built with a different road graph; rebuilding")
            with db_conn(state) as conn:
                with state.map_lock:
                    write_and_rebuild(conn, state.cfg, state.graph)
            return plan()
    except NoGraphError as exc:
        raise ApiError(409, "no_graph", str(exc)) from exc
    except OutsideGraphError as exc:
        raise ApiError(422, "outside_graph", str(exc)) from exc
    except NoRouteError as exc:
        raise ApiError(404, "no_route", str(exc)) from exc
    except ValueError as exc:
        raise ApiError(422, "invalid_input", str(exc)) from exc
