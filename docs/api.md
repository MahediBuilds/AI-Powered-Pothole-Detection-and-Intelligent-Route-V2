# HTTP API

Served by `python -m roadsense.api` (default `http://127.0.0.1:8000`). The web interface in `src/roadsense/web/static/` uses only these endpoints. All bodies are JSON unless stated. Coordinates are WGS84 decimal degrees; a bounding box is `min_lat,min_lon,max_lat,max_lon`.

Errors always have this shape, with a fitting HTTP status:

```json
{"error": {"code": "no_graph", "message": "No road network is loaded. ..."}}
```

| Status | `code` values |
|---|---|
| 400 | `bad_request`, `unsupported_file` |
| 404 | `not_found`, `no_route` |
| 409 | `no_graph`, `job_running` |
| 413 | `too_large` |
| 422 | `invalid_video`, `invalid_gps`, `invalid_input`, `outside_graph` |
| 500 | `internal` |

## System

### `GET /api/health`
`{"status": "ok", "version": "0.1.0"}`

### `GET /api/system`
What the server is running with. Never triggers a model load.

```json
{
  "version": "0.1.0",
  "model": {"weights": "runs/train/baseline/weights/best.pt", "exists": true, "sha256": "2aab…", "loaded": false,
            "device": null, "architecture": "YOLOv8s detection, single class 'pothole'"},
  "graph": {"loaded": true, "path": "data/graphs/road_graph.json", "sha256": "…", "bbox": [12.9, 77.5, 13.0, 77.6],
            "n_nodes": 1234, "n_edges": 3210, "attribution": "© OpenStreetMap contributors"},
  "settings": {"detection": {"conf": 0.25, "iou": 0.5, "imgsz": 640, "partial_frame_error_fraction": 0.02, "max_frame_error_fraction": 0.25},
               "tracking": {"frame_stride": 3, "iou_match": 0.2, "max_age_frames": 5, "min_hits": 3,
                            "vehicle_speed_mps": 2.5, "vehicle_frame_stride": 1, "vehicle_min_hits": 6},
               "mapping": {"dedup_radius_m": 8.0, "max_snap_m": 30.0}, "ingest": {"max_gps_accuracy_m": 25.0, "max_gps_gap_s": 3.0, "max_gps_extrapolation_s": 1.0},
               "routing": {"max_detour_per_pothole_m": 300.0, "assumed_rate_per_100m": 1.0, "prior_strength_m": 500.0,
                           "unsurveyed_rate_per_100m": null, "min_surveyed_fraction": 0.2, "adequate_surveyed_fraction": 0.8}},
  "include_synthetic": false,
  "demo": {"mode": false, "sessions": 1, "active": false}
}
```

`include_synthetic` is demo mode (the remembered switch; off until switched on). `demo.mode` is the same switch, `demo.sessions` the number of synthetic sessions stored, and `demo.active` is true when the switch is on and at least one synthetic session is counted in the map. The interface shows its permanent banner while `demo.active` is true. The `routing` settings are the same six keys as `cost_model` in `/api/routes/info`.

### `GET /api/stats`
Dashboard totals.

```json
{
  "sessions": {"total": 6, "by_status": {"done": 5, "running": 1, "queued": 0, "failed": 0}, "by_gps_source": {"recorded": 5, "synthetic": 1, "none": 0}},
  "observations": {"total": 14, "with_position": 12, "without_position": 2},
  "potholes": {"total": 9, "by_gps_source": {"recorded": 6, "synthetic": 3, "mixed": 0}, "on_road": 8, "off_road": 1},
  "coverage": {"surveyed_m": 412.5, "edges": 7, "demo_m": 310.0, "demo_edges": 4},
  "demo": {"mode": true, "sessions": 1, "active": true, "potholes": 3},
  "build": {"id": 3, "created_at": "2026-11-12T08:00:00+00:00", "has_graph": true, "include_synthetic": true}
}
```

`coverage.surveyed_m` and `edges` describe real roads (recorded sessions only). `demo_m` and `demo_edges` count road that only a synthetic trace runs along; they are never added to `surveyed_m`. `demo` is as in `/api/system`, plus `potholes`, the mapped potholes that are not `recorded`. `by_gps_source.mixed` is kept for compatibility and is always 0.

## Sessions

A session is one uploaded video with its optional GPS trace. Processing runs in the background, one job at a time.

### `POST /api/sessions` (multipart/form-data)

| Field | Required | Meaning |
|---|---|---|
| `video` | yes | `.mp4`, `.mov`, `.avi` or `.mkv` |
| `gps` | no | GPS CSV as written by the recorder app (`timestamp_sec,lat,lon,…`) |
| `sidecar` | no | recorder JSON sidecar |
| `role` | no | `survey` (default), `tuning`, `test`, `demo` |
| `gps_source` | no | `recorded` (default) or `synthetic`. Declare `synthetic` for a generated trace. Without a `gps` file it is stored as `none`. A `sidecar` that is a JSON object with `"synthetic": true`, given together with a `gps` file, forces `synthetic` whatever this field says. |
| `include_in_map` | no | `true` (default) or `false` |

Returns `202` with `{"session": <session>, "job": <job>}`. The files are moved (not copied again) from a temporary folder into `data/uploads/<session id>/`. Sessions with `gps_source` `synthetic` are demonstration data and appear in the map only while demo mode is on.

### Session object

```json
{
  "id": "9f2c…", "name": "road_20261022_164858", "role": "survey", "gps_source": "recorded", "include_in_map": true,
  "created_at": "…", "started_utc": "…", "duration_s": 20.1, "device": "Pixel 8",
  "gps": {"n_fixes": 21, "n_used": 20, "median_accuracy_m": 4.8},
  "job": {"id": "…", "status": "running", "stage": "detecting", "progress": 0.42, "error": null, "outcome": null, "updated_at": "…"},
  "n_observations": 3, "n_with_position": 3
}
```

`job.status` is `queued`, `running`, `done` or `failed`. `job.stage` is null while queued, then `reading video` (the video is probed and its timeline read; `progress` is `null` during this stage), `detecting` (`progress` 0 to 1), `saving`, and `done` when finished. `job.outcome` is `complete` or `partial` for a `done` job and `null` otherwise. `partial` means the detector failed on more than `detection.partial_frame_error_fraction` (0.02) of the analysed frames, up to `detection.max_frame_error_fraction` (0.25): the job is `done` and keeps its detections, but potholes in the failed frames are missing, and the report's `warnings` say so. Above 0.25 the job is `failed`. `n_observations` counts tracked potholes of the latest finished job.

### `GET /api/sessions`
`{"sessions": [<session>, …]}`, newest first.

### `GET /api/sessions/{id}`
`<session>` plus `"report"`: the detection report of the latest finished job (or `null`), including frame counts, track counts, GPS association counts, detector settings and `warnings`. Keys that matter when reading it: `outcome` (`complete` or `partial`); `frame_stride` and `tracking` (`mode` `vehicle`, `walking` or `default` when no GPS speed exists, `median_speed_mps`, `frame_stride`, `min_hits`, `max_age_source_frames`); `n_frame_errors`; `n_tracks`, `n_tracks_confirmed`, `n_tracks_rejected_short`; `n_observations_with_position` and `n_observations_without_position`; `video` (including `duration_s`); `gps`; `elapsed_s`. Reports of jobs run before 2026-11-12 may lack `outcome` and `tracking`.

### `GET /api/sessions/{id}/status`
`{"session_id": "…", "job": <job>}`. For polling.

### `GET /api/sessions/{id}/detections`

```json
{
  "session_id": "…", "job_id": "…", "gps_source": "recorded",
  "observations": [
    {"id": 12, "track_id": 3, "n_frames": 7, "confidence": 0.81, "first_t_video_s": 4.2, "t_video_s": 5.0, "frame_index": 150,
     "box": [412.0, 380.5, 560.2, 441.0], "frame_w": 1280, "frame_h": 720,
     "position": {"lat": 12.97, "lon": 77.59, "accuracy_m": 4.8}, "pothole_id": 5,
     "image_url": "/api/artifacts/9f2c…/a1b2…/track_3.jpg"}
  ]
}
```

One entry per tracked pothole, not per frame. `position` is the camera position when the pothole was last seen, or `null` when the GPS data does not support one. `image_url` is the frame of the best detection with the box drawn.

### `GET /api/sessions/{id}/track`
`{"gps_source": "recorded", "points": [[lat, lon, t_video_s, used], …]}`

### `POST /api/sessions/{id}/reprocess`
Queues a new job. `409 job_running` if one is queued or running.

### `PATCH /api/sessions/{id}`
Body `{"include_in_map": false}`. Rebuilds the map.

### `DELETE /api/sessions/{id}`
Deletes the session, its jobs, observations, uploaded files and images, then rebuilds the map. `409 job_running` while it is being processed.

### `GET /api/artifacts/{path}`
An annotated detection image (JPEG).

## Map

### `GET /api/potholes?bbox=min_lat,min_lon,max_lat,max_lon`
`bbox` is optional.

```json
{
  "potholes": [
    {"id": 5, "lat": 12.97, "lon": 77.59, "confidence": 0.81, "n_observations": 2, "n_sessions": 2, "gps_source": "recorded",
     "on_road": true, "first_seen_utc": "…", "last_seen_utc": "…"}
  ],
  "note": "Positions are the camera position when the pothole was last seen; the pothole is a few metres ahead."
}
```

One entry per mapped pothole after spatial deduplication. `gps_source` is `recorded` or `synthetic`; synthetic potholes are demonstration data and are never clustered with recorded ones (`mixed` is no longer produced). Synthetic potholes are listed only while demo mode is on.

### `GET /api/potholes/{id}`
The pothole plus `"observations"`: the original sightings grouped into it (same shape as in `/detections`, with `session_id` and `session_name`).

### `GET /api/coverage?bbox=…`
`{"stretches": [[[lat, lon], …], …], "surveyed_m": 412.5, "demo_stretches": [[[lat, lon], …], …], "demo_m": 310.0}`. `stretches` is road that a processed recorded session filmed (a GPS trace is cut to the video interval first). Everything else is unsurveyed: unknown, not pothole-free. `demo_stretches` is road a synthetic (generated) trace runs along; it is not a survey, is empty with demo mode off, and a stretch also covered by a recorded session is listed only in `stretches`. Both are empty when no road network is loaded.

### `POST /api/map/rebuild`
Body `{"include_synthetic": true}` (optional; the choice is remembered; `true` turns demo mode on, `false` off, and omitting it keeps the current choice). Rebuilds potholes, coverage and road statistics from the stored observations, in one write transaction that is retried up to 5 times with a doubling pause while the database is locked. The map is also rebuilt automatically after every finished job, and at server start when the stored build is out of date.

Returns the build summary:

```json
{"build_id": 4, "n_sessions": 6, "n_observations": 14, "n_observations_without_position": 2,
 "n_potholes": 9, "n_potholes_recorded": 6, "n_potholes_synthetic": 3,
 "n_potholes_on_road": 8, "n_potholes_off_road": 1, "n_edges_with_coverage": 7,
 "surveyed_m": 412.5, "demo_surveyed_m": 310.0, "graph_sha256": "…", "has_synthetic": true}
```

`surveyed_m` is recorded sessions only; `demo_surveyed_m` is the synthetic trace and is not a survey.

## Routing

### `GET /api/routes/info`
```json
{"available": true, "graph": "<as in /api/system>",
 "coverage": {"surveyed_m": 412.5, "edges": 7},
 "potholes_on_road": 8,
 "demo": {"active": true, "demo_m": 310.0, "potholes_on_road": 3},
 "cost_model": {"max_detour_per_pothole_m": 300.0, "assumed_rate_per_100m": 1.0, "prior_strength_m": 500.0,
                "unsurveyed_rate_per_100m": null, "min_surveyed_fraction": 0.2, "adequate_surveyed_fraction": 0.8}}
```

`available` is false when no road network is loaded. `coverage` and `potholes_on_road` count real roads and recorded potholes only. `demo` describes synthetic data in the latest build: `active` is true when it has synthetic coverage or potholes on a road. `cost_model` holds the configured settings (not the rate in use; that is in the plan from `POST /api/routes`).

### `POST /api/routes`
Body `{"origin": {"lat": 12.97, "lon": 77.59}, "destination": {"lat": 12.98, "lon": 77.60}, "avoidance": 0.5}`. `avoidance` runs from 0 (shortest distance) to 1 (strongest pothole avoidance).

Returns the plan produced by `roadsense.routing.router.plan_routes`: one to three routes (labels `Shortest`, `Your setting`, `Strongest pothole avoidance`; routes that follow the same roads are merged and carry several labels). A label names the setting that produced the route, not an outcome. The strongest-avoidance route minimises the estimated potholes, so it can pass more mapped potholes than the shortest route when that avoids unsurveyed road.

Plan keys: `origin`, `destination` (requested point, snapped node, `snap_distance_m`), `avoidance`, `build_id`, `routes`, `cost_model`, `confidence_notes`, `demo_mode`, `uses_synthetic_data` (same value as `demo_mode`) and `caveats`.

- `demo_mode` is true whenever the map was built with demonstration data, whichever roads the routes use. The first entry of `caveats` is then the demonstration caveat.
- `confidence_notes` is a list of strings: one when the route for the user's setting has `insufficient` evidence, one when less road than `prior_strength_m` has been surveyed in total (counting demo road when demo mode is on). It can be empty.

Each route has `labels`, `recommended` (true for the one labelled `Your setting`), `distance_m`, `duration_s`, `n_potholes_observed` (mapped potholes on it; with demo mode off, recorded only), `n_potholes_demo` (how many of those come from synthetic data), `potholes`, `surveyed_m`, `surveyed_fraction`, `demo_m` (length that follows a synthetic drive and no recorded one), `unsurveyed_m`, `evidence`, `expected_potholes_unsurveyed`, `expected_potholes_total` (observed plus assumed; an estimate), `geometry`, `sections` and `explanation`. `surveyed_m + demo_m + unsurveyed_m` is the length of the route's roads.

`evidence` is `adequate` (`surveyed_fraction` at least `adequate_surveyed_fraction`, 0.8), `partial`, or `insufficient` (below `min_surveyed_fraction`, 0.2). `surveyed_fraction` counts recorded survey only. On an `insufficient` route the estimate rests almost entirely on the assumed rate, and the explanation says so.

Errors: `409 no_graph`, `422 outside_graph` (a point is too far from any road in the loaded network), `404 no_route`, `422 invalid_input`.

Each route section is `{"status": "surveyed", "surveyed": true, "coords": [[lat, lon], …]}`. `status` is `surveyed` (covered by a recorded session), `demo` (covered only by a synthetic session; present only in demo mode) or `unsurveyed`. `surveyed` is true only for `surveyed`. Consecutive pieces with the same status are joined.

`cost_model` has `description`, `max_detour_per_pothole_m`, `prior_rate_per_100m` (potholes per 100 m assumed on unsurveyed road), `prior_source` (plain-language origin of that rate, stating when it is only an assumption), `surveyed_total_m` and `mapped_potholes_total` (the totals `N` and `S` of the formula), and `includes_demo_data`. The cost of a road segment is `length + avoidance x max_detour_per_pothole_m x estimated potholes`, with estimated potholes = mapped potholes + `prior_rate_per_100m` x unsurveyed length / 100 m. `prior_rate_per_100m` is `(N x 100 + assumed_rate_per_100m x prior_strength_m) / (S + prior_strength_m)` unless `unsurveyed_rate_per_100m` fixes it. The defaults are assumptions, not measurements. With demo mode on, `N` and `S` include synthetic potholes and road.

### `GET /api/demo/route-example`
`{"origin": {"lat": …, "lon": …}, "destination": {…}, "description": "…", "synthetic": true, "routes_differ": true, "session_id": "…"}` when `scripts/make_synthetic_demo.py` has prepared an example and its synthetic session is on the map. The example is demonstration data built on a generated GPS trace.

`404 not_found` when no example file exists or it is unreadable, and when it is not usable: demo mode is off, or the session it names no longer exists, is not synthetic, is excluded from the map or has no finished job. The file outlives a database reset, so it is checked against the database on every request.
