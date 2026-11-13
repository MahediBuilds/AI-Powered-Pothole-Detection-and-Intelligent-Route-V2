-- Application schema. Every statement must be idempotent (IF NOT EXISTS).
-- The schema version is held in the code (roadsense.storage.db.SCHEMA_VERSION)
-- and recorded in meta under the key 'schema_version'.
--
-- Raw tables:     sessions, gps_fixes, jobs, observations (except the columns below)
-- Derived tables: builds, potholes, edge_coverage, edge_stats, and the columns
--                 observations.edge_* and observations.pothole_id
--   Derived data is rebuilt in full by each map build from the raw rows of each
--   session's active_job_id where include_in_map = 1, against the road graph
--   loaded at that time (builds.graph_sha256; NULL when no graph is loaded).
--   Edge ids (u, v, key) are only meaningful for the graph of the latest build.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    video_path        TEXT NOT NULL,
    gps_path          TEXT,
    started_utc       TEXT,
    duration_s        REAL,
    device            TEXT,
    median_accuracy_m REAL,
    role              TEXT NOT NULL DEFAULT 'survey'
                      CHECK (role IN ('survey', 'tuning', 'test', 'demo')),
    -- where the GPS trace came from: a real recording, a generated (synthetic)
    -- trace used only for demonstration, or no trace at all
    gps_source        TEXT NOT NULL DEFAULT 'recorded'
                      CHECK (gps_source IN ('recorded', 'synthetic', 'none')),
    include_in_map    INTEGER NOT NULL DEFAULT 1 CHECK (include_in_map IN (0, 1)),
    -- latest successful job; plain column, no FK (jobs references sessions)
    active_job_id     TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gps_fixes (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    t_video_s   REAL NOT NULL,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    accuracy_m  REAL,
    speed_mps   REAL,
    bearing_deg REAL,
    utc_ms      INTEGER,
    used        INTEGER NOT NULL DEFAULT 1 CHECK (used IN (0, 1)),
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    status           TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    stage            TEXT,
    progress         REAL,
    error            TEXT,
    pipeline_version TEXT,
    model_sha256     TEXT,
    params_json      TEXT,
    summary_json     TEXT,          -- the session's detection report, set when done
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS builds (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT NOT NULL,
    pipeline_version TEXT,
    graph_sha256     TEXT,          -- NULL: built without a road graph
    params_json      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS potholes (
    id             INTEGER PRIMARY KEY,
    build_id       INTEGER NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
    -- canonical undirected edge (u < v), offset from u; NULL when not on a known road
    edge_u         INTEGER,
    edge_v         INTEGER,
    edge_key       INTEGER,
    edge_offset_m  REAL,
    -- mean CAMERA position of the grouped observations (the pothole is a few metres ahead)
    lat            REAL NOT NULL,
    lon            REAL NOT NULL,
    n_observations INTEGER NOT NULL,
    n_sessions     INTEGER NOT NULL,
    confidence     REAL NOT NULL,   -- highest detection confidence among the observations
    gps_source     TEXT NOT NULL CHECK (gps_source IN ('recorded', 'synthetic', 'mixed')),
    first_seen_utc TEXT,
    last_seen_utc  TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id             INTEGER PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id         TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    track_id       INTEGER NOT NULL,
    n_frames       INTEGER NOT NULL,   -- detections in the track
    confidence     REAL NOT NULL,      -- highest confidence in the track
    first_t_video_s REAL,              -- when the track was first seen
    t_video_s      REAL NOT NULL,      -- when the track was last seen
    frame_index    INTEGER NOT NULL,   -- frame of the last detection
    frame_w        INTEGER,
    frame_h        INTEGER,
    image_path     TEXT,               -- annotated frame of the best detection, relative to artifacts_dir
    x1             REAL NOT NULL,
    y1             REAL NOT NULL,
    x2             REAL NOT NULL,
    y2             REAL NOT NULL,
    -- CAMERA position at t_video_s, not the pothole's
    cam_lat        REAL,
    cam_lon        REAL,
    cam_accuracy_m REAL,
    -- canonical undirected edge (u < v), offset measured from u; NULL if unmatched
    edge_u         INTEGER,
    edge_v         INTEGER,
    edge_key       INTEGER,
    edge_offset_m  REAL,
    pothole_id     INTEGER REFERENCES potholes(id) ON DELETE SET NULL,
    UNIQUE (job_id, track_id)
);

CREATE TABLE IF NOT EXISTS edge_coverage (
    id         INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    edge_u     INTEGER NOT NULL,
    edge_v     INTEGER NOT NULL,
    edge_key   INTEGER NOT NULL,
    start_m    REAL NOT NULL,
    end_m      REAL NOT NULL,
    CHECK (start_m <= end_m)
);

CREATE TABLE IF NOT EXISTS edge_stats (
    edge_u        INTEGER NOT NULL,
    edge_v        INTEGER NOT NULL,
    edge_key      INTEGER NOT NULL,
    build_id      INTEGER NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
    length_m      REAL NOT NULL,
    surveyed_m    REAL NOT NULL,
    n_potholes    INTEGER NOT NULL,
    rate_per_100m REAL NOT NULL,
    PRIMARY KEY (edge_u, edge_v, edge_key)
);

CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_observations_edge ON observations(edge_u, edge_v, edge_key);
CREATE INDEX IF NOT EXISTS idx_observations_pothole ON observations(pothole_id);
CREATE INDEX IF NOT EXISTS idx_edge_coverage_edge ON edge_coverage(edge_u, edge_v, edge_key);
CREATE INDEX IF NOT EXISTS idx_edge_coverage_session ON edge_coverage(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_potholes_edge ON potholes(edge_u, edge_v, edge_key);
