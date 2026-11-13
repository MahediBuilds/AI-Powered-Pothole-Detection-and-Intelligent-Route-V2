# Design decisions

Short record of decisions that later work depends on. Add to it when a decision changes.

## Locked by the project owner (2026-11-11)

- Forward-facing phone or dashcam footage only.
- Single-class pothole box detection. No segmentation and no severity classification.
- RDD2022 is the initial training set. Our own forward-facing footage is reserved for independent testing. Fine-tuning on our own footage is optional and must use separate sessions.
- Stack: Python, PyTorch, Ultralytics, OpenCV, FastAPI, SQLite in WAL mode, Shapely, PyProj, OSMnx, NetworkX, Leaflet with vanilla JavaScript, pytest.
- No Android app, Docker, Kubernetes, PostGIS or microservices.

## Structure

- **One package, `roadsense`, in a `src` layout.** Sub-packages follow the pipeline stages: `data`, `training`, `ingest`, `detection`, `tracking`, `geo`, `routing`, `storage`, `api`.
- **Heavy libraries are optional extras** (`ml`, `geo`, `api`) and are imported only inside the functions that need them. The core tests run with PyYAML, pydantic, Pillow, NumPy and pytest alone.
- **Configuration is YAML validated by pydantic.** Unknown keys and wrong types are errors. Relative paths resolve against the project root, not the working directory.

## Dataset conversion

See `rdd2022_dataset.md` for the verified format.

- **Only `D40` is a pothole.** Every other label is counted and treated as "not a pothole".
- **An image with any invalid pothole box is excluded whole.** Dropping just the box would turn a pothole image into a false negative.
- **Negatives are capped in the training split only** (default two per positive image). Validation and test keep their natural mix so false positives are measured honestly.
- **Splits group near-duplicate frames by perceptual hash, never by individual image.** Images within 8 bits of each other on a 64-bit difference hash are linked, and each linked group goes whole to one split. Quotas are filled by image count, so the one large group per subset stays in training.
- **Why not blocks of consecutive ids.** That was the first design. The scan of the real data showed ids are shuffled, so it leaked like a random split. The block strategy is kept only for datasets whose ids follow capture order. Evidence is in `rdd2022_dataset.md`.
- **Validation and test keep the natural positive rate (about 11%); training is capped at 33%.** Validation precision and mAP are therefore lower in absolute terms than the training mix would suggest, and are not comparable to it.
- **No measurable scene bias in validation and test.** They are drawn from images with no close neighbour, but an independent review found box sizes, image sizes and negative kinds match training closely.
- **Residual overlap is a stated limitation.** No validation or test image has a training image within 8 bits. About 64% have one at 9 to 12 bits; the pairs inspected were similar layouts, not the same scene. Same-route correlation is still plausible, so RDD2022 scores are probably optimistic for unseen roads.
- **Per-country validation and test cells are small.** Only India and Japan have more than 100 pothole images per split; United_States has 8 in validation.
- **Every labelled source image gets one manifest row with a status.** Nothing is dropped silently.
- **Output is deterministic.** Two runs on the same input give byte-identical manifests and labels.
- **Norway and China_Drone are off by default.** Reasons are in `rdd2022_dataset.md`.

## Training

- **Baseline model is YOLOv8s box detection (`yolov8s.pt`)** at 640 px, seed 42, up to 100 epochs, early stopping with patience 20. Decided by the project owner on 2026-11-11.
- **The config first named `yolo11s.pt`; that was changed before any training run.** No YOLO11 weights were trained or evaluated. Dataset, split, augmentation and all other hyperparameters are unchanged.
- **Detection only.** The config rejects weights named for another task (`-seg`, `-cls`, `-pose`, `-obb`), and the entrypoint refuses to train a loaded model whose task is not `detect`.
- **Ultralytics is pinned to 8.4.210.** Its automatic optimiser choice and weight-decay scaling depend on the version; the points below were read from that version's `engine/trainer.py`.
- **Batch size is a per-machine setting, capped at 64.** The config says 16 (fits an 8 GB card). `--batch` and `--workers` override it on other machines and are recorded in `run_config.json`.
    - 16, 32 and 64 give the same recipe: effective batch 64 through gradient accumulation, unchanged weight decay, and the same automatic optimiser (more than 10,000 iterations on 8,220 images for 100 epochs).
    - Batch sizes that do not divide 64, and any above 64, change the weight decay. From 83 upwards the iteration count drops to 10,000 or fewer and the automatic optimiser switches to AdamW.
    - 16 and 64 are still not bit-identical: BatchNorm sees 16 or 64 images per step, warmup takes more, smaller steps at 16, and the worker count changes the augmentation random streams.
- **Worker count is limited by host RAM, not by CPU threads.** Ultralytics starts twice as many validation workers as training workers (capped at the CPU count), each holding batches of twice the training batch. On a 16-thread host with a 31 GB memory limit, `--workers 12` at batch 64 was killed by the container's memory limit during the first validation pass (peak 28.9 GB); `--workers 6` ran. The worker count changes the augmentation random streams but no hyperparameter.
- **An out-of-memory error in the first epoch makes Ultralytics halve the batch with only a warning.** `run_config.json` would still show the requested value, so check the training log for "Reducing to batch".
- **The entrypoint always tells Ultralytics to reuse the run directory.** It creates that directory itself to write `run_config.json`, and Ultralytics would otherwise rename the run to `baseline-2`. A local two-epoch run on a 128-image subset found this on 2026-11-11; that run was a check of the entrypoint, not an experiment, and its outputs were deleted.
- The training entrypoint never evaluates the test split. Model selection uses the validation split only.
- The resolved configuration and the dataset hashes are written to the run directory before training starts.
- Vertical flip augmentation is off, because road scenes are not vertically symmetric.

## Baseline run (2026-11-11)

Outputs are in `runs/train/baseline/`; the instance's logs are in `data/logs/vast_20261111/remote_logs/`.

- **Where:** one rented RTX 5090 (32 GB), PyTorch 2.14.1 with CUDA 13.0, Ultralytics 8.4.210, Python 3.12.
- **Data:** the dataset was rebuilt on the instance from figshare and was byte-identical to the local one (manifest SHA-256 `60459e8c…0e3e`, same labels and file names, all 19 verification checks passed).
- **Settings:** `configs/train_baseline.yaml` with `--batch 64 --workers 6`. The automatic optimiser chose MuSGD with learning rate 0.01, as predicted. The batch was not reduced during the run.
- **Length:** early stopping ended the run after 76 epochs (20 min 12 s of training). The best epoch was 56.
- **Validation at the best epoch** (2,784 images, 477 boxes, about 11% of images positive): precision 0.581, recall 0.457, mAP50 0.464, mAP50-95 0.203.
- **After epoch 56 the validation losses rise while the training losses keep falling,** so the model had started to overfit; stopping there was appropriate.
- **These are in-domain RDD2022 validation numbers.** They say nothing yet about our own forward-facing footage, and the RDD2022 test split has not been evaluated.
- **A first attempt with 12 workers was killed by the memory limit before completing an epoch** (see Training). Its log is kept as `train_attempt1_oom_workers12.log`.
- `best.pt` SHA-256: `2aabdd131d1664089e06e3219fabcaed4cc96610fa4603ff165429f92696d798`.

## Application pipeline (2026-11-11)

Upload, then detect and track, then place on the map, then route. Each stage is a small module; the HTTP contract is in `api.md`.

### Detection and tracking

- **The detector wrapper only accepts our checkpoint.** It refuses weights that are not a detection model with exactly the class `pothole`, so generic pretrained weights cannot be used by mistake. The model is loaded once and reused.
- **GPU when available, CPU otherwise.** `detection.device: auto` picks GPU 0 if CUDA works.
- **Every third frame is analysed** (`tracking.frame_stride`), about 10 frames per second for 30 fps video. *Superseded for vehicle recordings on 2026-11-12: every frame is analysed there. See "Reliability fixes after independent review".*
- **Tracking is a greedy IoU tracker with no motion model.** A track continues while a detection overlaps its last box (IoU at least 0.2) and survives five analysed frames without a match. This replaces the ByteTrack plan: it needs no extra dependency and is enough to stop one pothole being counted once per frame. *Superseded on 2026-11-12: the tracker now predicts the box from the track's velocity and ages tracks in source frames. Why the old one failed at vehicle speed is recorded below.*
- **A pothole sighting is a track with at least three detections.** Shorter tracks are counted in the report as rejected, not stored. On the old walking-pace clips most raw detections are rejected this way. *Still true for walking and no-speed sessions; vehicle sessions need six (2026-11-12).*
- **One row per sighting, not per frame.** `observations` stores the track's best confidence, its first and last times, the last box, and an annotated image of the best frame.
- **All thresholds are provisional** and were not tuned on any test data.

### GPS association

- **A sighting is placed at the camera position when the pothole was last seen,** using the existing `position_at` with the ingestion thresholds. The pothole itself is a few metres ahead; no offset is applied.
- **No usable GPS means no position.** The sighting is kept and shown, but never placed on the map.
- **Every session records where its GPS came from:** `recorded`, `synthetic` or `none` (`sessions.gps_source`). A generated trace must be declared synthetic on upload.

### Map build

- **Derived data is rebuilt in full after every finished job.** Potholes, surveyed road, per-road statistics and the links from sightings to potholes are all recomputed from the stored sightings and GPS fixes. A database is therefore no longer bound to one road graph; each build records the graph hash it used (and, since 2026-11-12, the jobs, mapping settings and code version).
- **Deduplication is by distance.** ~~Sightings within 8 m of each other, in any session, are one mapped pothole (single linkage).~~ *Superseded 2026-11-12: greedy complete linkage; every pair in one mapped pothole is at most 8 m apart.* The original sightings are kept and linked to it. This replaces the planned deduplication along road edges; it also works with no road network loaded.
- **Road matching is nearest-road snapping within 30 m,** not whole-trace map matching. Near junctions and parallel roads a point can land on the wrong road.
- **Surveyed road is where a processed session's GPS trace ran.** Anything else is unsurveyed. *Refined 2026-11-12: only the part of the trace inside the video interval counts.*
- **A mapped pothole is labelled by the GPS source of its sightings:** `recorded`, `synthetic` or `mixed`. ~~Synthetic sessions can be excluded from the build with one switch.~~ *Superseded 2026-11-12: `mixed` is never produced (synthetic and recorded sightings are clustered apart) and demo mode is off by default.*

### Road network and routing

- **The road network is OpenStreetMap, downloaded once with OSMnx** and stored as a plain JSON file, so the server does not need OSMnx at run time.
- **Cost of a road segment** = length in metres + avoidance × 300 m × expected potholes. Avoidance is the user's slider from 0 to 1; 300 m is `routing.max_detour_per_pothole_m`.
- **Expected potholes** = mapped potholes on the segment + an assumed rate × its unsurveyed length. ~~The assumed rate is the rate observed on surveyed road unless one is configured.~~ *Superseded 2026-11-12: the observed rate is shrunk towards a stated assumption.* Unsurveyed road is never treated as clean.
- **Up to three routes are returned:** shortest, the user's setting, and strongest avoidance, with identical ones merged. Each carries distance, an estimated time, mapped potholes passed, the surveyed share, an evidence level, and a plain-language explanation. (The third route was labelled "Fewest expected potholes" until 2026-11-12.)
- **Routes start and end at the nearest road junction,** and a point more than 250 m from the network is refused.
- **No severity anywhere.** The detector has one class.

### Schema version 2

- `sessions.gps_source`, and `demo` as a session role.
- `jobs.summary_json` holds the detection report.
- `observations` gained the first-seen time, the frame size and the annotated image path.
- `potholes` may have no road edge, and carries `confidence` and `gps_source`.
- `builds.graph_sha256` may be empty (built without a road network).
- A version 1 database is refused; there was no stored data to migrate.

### Known limitations

- **The model has only been validated on RDD2022.** On our old ground-facing walking clips its confident detections include pale patches that are not potholes. The 2026-11-12 review adds manholes, road markings, stains and tyre marks.
- **Positions are camera positions,** so a mapped pothole is a few metres behind its true place, plus GPS error (about 5 m on the samples).
- **Two different potholes closer than 8 m can merge into one, and one pothole seen with larger GPS error can split.** Complete linkage no longer chains distant sightings together.
- **Travel times are estimates** from speed limits or road-type defaults.
- **One job runs at a time,** and jobs still queued or running when the server stops are marked failed at the next start. That is still true and is now safe, because it happens only while the server holds the database's process lock (see 2026-11-12).

## Evaluation (planned)

- Standard Ultralytics validation settings, with no confidence pre-filter.
- Evaluating a held-out test split needs an explicit flag and appends to a ledger, so the number of test runs is auditable.
- Headline numbers come from our own forward-facing test sessions: detector metrics, false positives per km, and per-pothole recall against a manual count.
- Thresholds are tuned on a separate tuning session and frozen before the first test run.

## Ingestion

- **A session is three sibling files:** `<stem>.mp4`, `<stem>_gps.csv` and an optional `<stem>.json` sidecar, as written by the recorder app.
- **Frame timestamps come from the decoder's presentation timestamps,** checked once per file. If they are not strictly increasing, the whole file falls back to frame index divided by frame rate. The two sources are never mixed.
- **GPS rows are never dropped silently.** Each rejected row is recorded with its line number and reason.
- **`position_at` returns no position rather than a guess.** That applies when there are no usable fixes, when the surrounding fixes are further apart than the gap limit, or when the time is beyond the log by more than the extrapolation limit.
- **Ingestion thresholds are provisional:** accuracy 25 m, gap 3 s, extrapolation 1 s. They must be fixed on a tuning session before any test-set evaluation. One walking-pace sample already sits on the edge of all three.
- **Not stored yet:** GPS row issues, coverage, warnings and the timestamp source are lost after a session is saved. Add columns when tracking and map matching need them.

## Storage

- One SQLite file in WAL mode with foreign keys on.
- **Raw tables:** `sessions`, `gps_fixes`, `jobs`, `observations`.
- **Derived data:** `builds`, `potholes`, `edge_coverage`, `edge_stats`, and the road-edge and pothole links on `observations`. It is rebuilt in full by each map build, which records the parameters and graph hash used.
- **Edge ids are only meaningful for the graph of the latest build.** Routing refuses a build made with a different graph; the server rebuilds the map when the graph changes. (The first design bound a database to one graph file; see "Application pipeline".)
- Road edges are stored undirected in canonical form (`u < v`), with the offset measured from `u`.
- `observations` stores the camera position when the pothole was last seen, not the pothole's own position. The columns are named `cam_lat` and `cam_lon` to make that explicit.

## Still open

- How much scene-level overlap remains between RDD2022 splits beyond what the hash catches.
- Whether an RDD2022-only model transfers to our phone footage (decides whether fine-tuning is needed).
- Detection, tracking, deduplication and routing thresholds. They exist in `configs/app.yaml` but are provisional and untuned; no tuning session has been run.

## Reliability fixes after independent review (2026-11-12)

An independent review of the application found the problems below. Each entry gives the decision and the reason. Statements elsewhere in this file that these replace are marked as superseded.

### Real and demo data

- **Demo mode (`meta.include_synthetic`) is off until switched on.** Reason: synthetic sessions are generated and must not shape anything a reader could take for real. Switch: Dashboard, `rebuild-map --include-synthetic`, or the demo script. With it off, synthetic sessions are not in the map.
- **Synthetic and recorded sightings are clustered separately.** A pothole is `recorded` or `synthetic`, never `mixed`. Reason: a blend has no honest label. (`/api/stats` still lists a `mixed` key; it is always 0.)
- **`edge_stats` and every "surveyed road" figure use recorded sessions only.** Synthetic coverage is returned and drawn separately (`demo_stretches`, `demo_m`), and a stretch covered by both kinds counts as recorded. Reason: a generated drive is not a survey.
- **With demo mode on, a permanent banner is shown and every route plan has `demo_mode: true` and the demonstration caveat,** whichever roads the routes use. Synthetic potholes and road also enter the totals behind the assumed rate, and the plan says so. Reason: a plan can be influenced by demonstration data through that rate even on roads it never touches.
- **A recorder sidecar with `"synthetic": true` forces `gps_source = synthetic`** when a GPS file is given. Reason: the demo script's output must not be mistaken for a real recording through a missing form field.

### GPS and video timing

- **GPS times are on the video clock, and a trace is cut to `[0, video duration]` before coverage is computed.** The position at each boundary is interpolated, unless the two fixes around it are further apart than `ingest.max_gps_gap_s`. Reason: fixes logged before the first frame or after the last one describe road the camera did not film, so they must not count as surveyed.

### Map rebuild reliability

- **A rebuild runs in one write transaction taken up front (`BEGIN IMMEDIATE`), retried up to 5 times with a doubling pause (0.25 s first) when the database is locked.** Reason: a rebuild that met a lock halfway could leave the map stale with no sign of it.
- **The job runner retries a failed post-job rebuild while idle** (every 3 s, up to 10 times). A later successful rebuild also settles it.
- **Each build records the jobs, mapping settings and code version it used, and the server rebuilds at start when any of them, the road graph, or demo mode differs.** Reason: a missed rebuild is now noticed and repeated.

### One processing process per database

- **The server holds an operating-system lock on `<db>.lock`.** A second server on the same database refuses to start and changes nothing. CLI `add`, `add-dir`, `reset` and the demo script take the same lock and are refused while the server runs, and the server will not start while one of them runs. `rebuild-map` and `list` are safe at any time. Reason: jobs live in an in-memory queue, so a second process would mistake the first one's `queued` and `running` rows for crash leftovers and fail them. The lock belongs to an open file handle, so the operating system releases it however the process ends; no stale-lock clean-up is needed.

### Job outcome

- **A job where the detector failed on a share of analysed frames is `complete` up to `detection.partial_frame_error_fraction` (0.02), `done` with `outcome: "partial"` up to `detection.max_frame_error_fraction` (0.25), and `failed` above that.** `partial` is an outcome of a `done` job; the status values are unchanged and the schema is untouched. Reason: a job that lost a large share of its frames should not look the same as a clean one, and one that lost most of them should not store results.

### Tracking

- **The tracker matches a detection to the track's predicted box** (last box moved by the track's recent velocity, over at most 5 earlier points; the box size is not predicted). **Track age is measured in source frames,** so it means the same time at any stride. The age limit is `max_age_frames x frame_stride` = 15 source frames in both modes. Reason: at vehicle speed a pothole moves further between analysed frames than its own height, so the last box and the next detection did not overlap and one object became a string of one-frame tracks. A fallback on centre distance alone was rejected: it merged unrelated detections.
- **Vehicle sessions** (upper-quartile GPS speed inside the video at least `tracking.vehicle_speed_mps`, 2.5 m/s) **analyse every frame and need `vehicle_min_hits` = 6 detections;** other sessions keep every third frame and 3 detections. Reason: analysing every frame yields more detections per pothole, so the hit threshold is raised with it.
- **Evidence is track continuity, not accuracy** (replay of saved detections with `scripts/compare_trackers.py`; no ground truth). On `road_20261013_180932` one defect that used to be discarded as one-frame tracks is now one track of 19 detections, and one object split by a 10-frame gap is now one track of 25; confirmed tracks went from 5 to 7. The two clean-road motorcycle recordings stay at 0 confirmed tracks. Walking recording `road_20261013_173922` keeps its 9.
- **Cost:** on the RTX 4060 laptop GPU the 60 s vehicle clip takes about 19 s instead of about 8 s; walking clips are unchanged (about 6 s).
- **Known weakness:** on `road_20261013_165247` (vehicle, strong camera movement) the new settings confirm 3 tracks, on tyre marks, where the old settings confirmed none. Consistent detector false positives cannot be removed by tracking.

### Deduplication

- **Greedy complete linkage replaces single linkage.** Pairs within `mapping.dedup_radius_m` (8 m) are merged in order of distance, and two groups merge only if every cross pair is within the radius. Every mapped pothole therefore fits in a disc 8 m across. Reason: single linkage chained a line of nearby sightings into one pothole. On the review database (50 sightings, 13 real recordings) this gives 15 mapped potholes with a largest cluster 8.0 m across, where chain linkage gave 11 with a largest cluster 15.8 m across. Limits: the result is deterministic but not guaranteed to be the fewest clusters; positions are camera positions with about 5 to 10 m GPS error, so two potholes closer than 8 m can still merge and one pothole can still split.

### Route scoring

- **Cost of a road segment = length + avoidance x `max_detour_per_pothole_m` (300 m) x estimated potholes;** estimated potholes = mapped potholes + assumed rate x unsurveyed length / 100 m.
- **The assumed rate is the observed rate shrunk towards an assumption:** `(N x 100 + assumed_rate_per_100m x prior_strength_m) / (S + prior_strength_m)`, defaults 1.0 per 100 m and 500 m. `N` and `S` are the mapped on-road potholes and surveyed metres. With nothing surveyed it equals the assumption; a small survey cannot give an extreme rate. `unsurveyed_rate_per_100m`, if set, fixes the rate. Reason: the observed rate alone is undefined or zero when little is surveyed, which costs unknown road as if it had no potholes, and a few metres of survey could give an extreme rate. Both defaults are assumptions, not measurements.
- **Each route reports `evidence`:** `adequate` (at least 80% surveyed), `partial`, `insufficient` (under 20%). The plan carries `confidence_notes`. Thresholds are `routing.adequate_surveyed_fraction` and `routing.min_surveyed_fraction`, and are unvalidated.
- **Labels are "Shortest", "Your setting", "Strongest pothole avoidance".** They name the setting that produced a route, not an outcome. Reason: the strongest-avoidance route minimises the estimate and can pass MORE mapped potholes than the shortest route when it avoids unsurveyed road. An earlier test found this in 73 of 576 trips under the previous scoring. It has not been re-measured under the current scoring.

### Road identity

- **Edge keys are normalised on load (`pair_directions`) by matching the two directions of a road on geometry** (lengths within 1 m, shapes within 3 m). Reason: OpenStreetMap graphs number parallel roads between the same two junctions independently per direction, so two different roads could share one canonical identity. When a file needs this, the graph's hash changes and the map is rebuilt once. In the project's graph 6 directed edges were re-keyed. Limit: twins are recognised by geometry only.

### Uploads and memory

- **An uploaded file is moved into its session folder instead of being copied a second time** (web upload only; the CLI copies). **The pipeline keeps one annotated JPEG per live track instead of full frames.** Reason: disk use and memory grew with video size.

### Interface

- Demo banner; the demo drive is a dashed purple line distinct from surveyed road; route cards show survey coverage, an evidence chip and "assumed" and "estimated" wording; the demo-example button disappears when the demo session is not on the map; jobs show "reading video" before detection and "done (partial)" where applicable.

### Known limitations after these fixes

- The detector's validation scores are modest (precision 0.581, recall 0.457 on RDD2022 validation) and it fires on patches, manholes, road markings, stains and tyre marks.
- A qualitative look at 50 confirmed tracks under the previous tracker and deduplication found 8 plausible potholes, 11 other road damage, 18 clear false positives and 13 unclear. These are subjective visual categories, not precision or recall.
- The project's real recordings cover about 1,020 m on seven disconnected stretches, and no two surveyed streets join the same junctions. A real-world route comparison cannot be shown; the route demonstration uses synthetic data.
- Vehicle recordings take about 2.4 times longer to process.
- Repeat sightings of one pothole can be split into several mapped potholes: every pair in a cluster must be within `mapping.dedup_radius_m`, so GPS scatter larger than the radius splits a pothole, more likely with more passes over the same road (a simulation with 3 m position error and 10 sightings per pothole gave about 2.5 clusters per real pothole). The radius trades this against merging neighbouring potholes.
- Motion tracking assumes about 30 fps video. At 10 to 15 fps, or for very small boxes at about 50 km/h, a new track's second detection may not overlap its first and the pothole is lost as short tracks. All the project's recordings are 30 fps.
- The tracking mode is chosen once per session from the upper-quartile GPS speed, not per stretch of road.
- Every threshold and the assumed rate are provisional and untuned.
