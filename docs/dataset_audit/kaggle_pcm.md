# Audit: Kaggle "Road Damage Dataset: Potholes, Cracks and Manholes"
Source: lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes. Downloaded unauthenticated (194,168,092 bytes) to `data/external/kaggle_pcm/ds.zip`; extracted to `data/external/kaggle_pcm_extracted/data/`. Machine-readable results: `kaggle_pcm.json`. Scripts: `scripts/dataset_audit/audit_kaggle_pcm.py`, `contact_sheet.py`. Contact sheets: `kaggle_pcm/*.jpg`.

## Documentation (README quotes)
- Classes: 0 Pothole, 1 Crack, 2 Manhole. No definition of "manhole" (all covers vs damaged only) is given in the README. Visually, manholes include flush, intact covers (see contact sheets).
- Capture: "GoPro and smartphone cameras mounted on vehicles at ~1.2-1.5 m height, angled 10-15 deg relative to road plane. Original resolution 1920x1080 (GoPro), 1280x720 (smartphone). Downsampled to 640x360." "Region X and Y" (location not named). "February-March 2025", 20-50 km/h, "severe blur, occlusion, or nighttime conditions are excluded".
- Licence: the README and the zip contain NO licence text and no LICENSE file. MIT is only what the Kaggle API metadata reportedly says (not verified in the bundle). Citation: https://doi.org/10.5281/zenodo.17834373.
- Split: none provided. Review/QA process: not documented.

## Measured facts
- 2009 images, all 640x360; 4737 boxes: pothole 1261 (795 images), crack 2519 (1375), manhole 957 (759). Images with no annotation: 0 (no negatives; every image has >=1 box). Boxes/image median 2, max 13.
- Annotation type: the "polygons" in `labels/` are 4-point axis-aligned rectangles (0 non-rectangular rows); YOLO and COCO boxes derive from them. COCO `segmentation` is empty for all 4737 annotations.
- Box size (px at 640x360, sqrt(area) median / p5 / p95): pothole 29.6 / 13.9 / 74.6 (median w 45, h 20; 721/1261 under 32 px); manhole 32.5 / 15.9 / 57.6 (median 55x19); crack 57.1 / 24.7 / 146.8 (max 640 wide, one with height 0).
- Validity: COCO 45 boxes out of bounds, 1 zero-area; YOLO files 26 out-of-bounds (tolerance 0.002), 1 zero-area. 0 duplicate boxes, 0 missing/extra images or label files.
- Exact duplicates (SHA-256): 5 pairs. Near-duplicates (64-bit dHash, Hamming <=6): 933 pairs, 143 clusters containing 425 images, largest cluster 48 images.
- File names: 1695 `vlcsnap-YYYY-MM-DD-HHhMMmSSsmmm` (VLC snapshots from video), 102 `YYYYMMDD_HHMMSS`, 212 with no timestamp (`vlcsnap-00001..214`). Dates: 2025-02-16/18/19/23/26 and 03-16. Of 1797 timestamped images: gap<=2 s -> 999 groups (663 singletons); <=5 s -> 470; <=10 s -> 243 (max 85 images); <=30 s -> 75; <=60 s -> 43 (max 304). Images are video-derived with heavy temporal clustering; a random split would leak. Snapshot time may be VLC snapshot time, not capture time (not verified).

## Visual review (random_0/1, manhole_only_0, pothole_only_0)
- Viewpoint: forward-facing dashcam, orange bonnet visible at bottom of most frames; a minority are handheld close-ups (e.g. 20250219_164746, 20250223_144136).
- Looks like Italian urban/suburban roads (signage, vehicles) - inference, not stated.
- Label quality: pothole boxes are mostly tiny (10-20 px tall) and often loose; some "potholes" are patches, puddles or gravel/dirt-lot areas; crack boxes are very large and vague (whole road regions, vertical slivers); some pothole/crack/manhole boxes overlap in the same spot; some tiny potholes are near-invisible at this resolution. Manholes are generally well localised.


## Licence and annotation review (2026-11-12)

Scope: decide whether the dataset can feed a single-class "pothole" detector, using licence evidence and a per-annotation exclusion-only review. All outputs are under `docs/dataset_audit/kaggle_pcm/` (scripts: `scripts/dataset_audit/review_kaggle_pcm_{render,apply,frames}.py`). Nothing in the dataset or in `data/` was modified; no box was added, moved, resized or relabelled.

### Licence evidence
- Raw Kaggle API response (fetched 2026-11-12, HTTP 200): `kaggle_pcm/kaggle_api_metadata.json`. Fields: `licenseName: "MIT"`, `hasLicenseName: true`; owner/creator "Lorenzo Arcioni" (`lorenzoarcioni`); `currentVersionNumberNullable: 4`; `lastUpdated: 2026-02-10T15:26:14Z`; `downloadCount: 11349`; `totalBytes: 194786682`; title "Road Damage Dataset: Potholes, Cracks and Manholes"; subtitle "A curated set of 2,000+ YOLO-annotated road surface images".
- Full description (verbatim; the original has line breaks between sentences): "This dataset provides real-world road surface images collected in urban and rural areas around Rome and Sacrofano (Italy) using two low-cost devices: a GoPro HERO7 mounted on a moving vehicle and a Samsung Galaxy A14 smartphone for stationary captures. All images are 640x360. It aims to support realistic, device-independent road damage detection. Each image is annotated in YOLO format with three damage classes: Class 0 - Pothole, Class 1 - Crack, Class 2 - Manhole. Unlike most existing datasets that omit or misclassify manholes, this dataset explicitly includes them to improve model robustness and reduce false positives in real-world scenarios. The collection ensures diversity in lighting, perspective, and pavement texture, enabling reliable training and cross-device generalization for AI-based road inspection systems. This dataset is a mirror of the dataset archived on Zenodo (DOI: https://doi.org/10.5281/zenodo.17834373 ). The Zenodo record represents the authoritative version for citation. Please cite our work!"
- Bundled README (`data/README.md`): the only citation/licence-related text is the "Citation" section: "If you use this dataset, please cite our work: https://doi.org/10.5281/zenodo.17834373". The README and zip contain no licence text and no LICENSE file.
- Reading: MIT is declared on the Kaggle listing only; the description says the Kaggle copy mirrors a Zenodo record that is "authoritative". The Zenodo record's own licence was NOT checked here and should be confirmed before any redistribution or publication. The authors ask for the DOI to be cited.

### Method
1. `review/pothole_sheet_01..43.jpg` (1261 boxes) and `review/manhole_sheet_01..32.jpg` (957 boxes): square crop centred on the box, side = max(96 px, 5 x the larger box side) clipped to the frame, upscaled to 256 px, 1 px magenta outline, 30 tiles per sheet (6 x 5), tile id (`P0001..`, `M0001..`) plus box size as caption. Mapping tile id to image, annotation index (order within the COCO file for that image), COCO annotation id and box: `review/id_map.csv`.
2. Every sheet was viewed and a one-letter verdict per tile was recorded (`review/verdict_codes_pothole.txt`, `review/verdict_codes_manhole.txt`; one line per sheet, tile order). `review_kaggle_pcm_apply.py` turns these into `review/annotation_review.csv` (image, ann index, class, box, verdict, reason, tile id).
3. Full-frame pass: every image still POSITIVE or NEGATIVE after step 2 (894 images) was viewed on `review/frames/frames_01..45.jpg` (384x216 tiles, 20 per sheet; pothole boxes magenta, manhole boxes cyan; ids in `frames/frame_id_map.csv`). Flags were recorded in `frames/frame_flag_codes.txt` and `review/image_flags.csv`. Images already excluded in step 2 were not re-viewed full-frame.
4. Decision policy (as specified): EXCLUDE if any pothole box is reject/unclear, any manhole is damaged_or_sunken/unclear, a pothole box is invalid or out of bounds by more than 2 px, `unlabelled_defect` is flagged, or the scene is unpaved/handheld close-up; POSITIVE = at least one pothole box, all ok (only pothole boxes kept; crack/manhole boxes dropped to background); NEGATIVE = no pothole boxes and every manhole intact (crack-only and intact-manhole images count as negatives). ASSUMPTION: that the dataset annotators labelled every pothole is not documented, and the `unlabelled_defect` flags show it does not fully hold. Outputs: `review/image_decisions.csv`, `review/kept_pothole_boxes.csv`, `review/bbox_clipping_log.csv` (1 box clipped by 1.0 px: 20250223_144310.jpg ann 3), `review/review_summary.json`, `review/time_group_summary.json`.

### Results
Annotation verdicts:

| class | ok / intact | reject / damaged_or_sunken | unclear | total |
|---|---|---|---|---|
| pothole | 107 | 155 | 999 | 1261 |
| manhole | 307 | 12 | 638 | 957 |

Pothole rejects by reason: repair patch 54, unpaved/gravel/dirt track 42, puddle/water without visible depression 23, unpaved car park/dirt lot 19, manhole/drain cover or frame 12, road marking/tyre mark/shadow 5.

Image decisions (2009 images): POSITIVE 42, NEGATIVE 815 (161 contain at least one intact manhole), EXCLUDE 1152. Exclusion by first applicable reason (images can have several): pothole_unclear 650, manhole_unclear 364, pothole_reject 93, unlabelled_defect 23 (6 of them lower-confidence, "same spot as a boxed neighbouring frame"), unpaved_or_handheld_closeup 14, manhole_damaged_or_sunken 7, pothole_box_oob_gt2px 1.

Kept pothole boxes: 59 (of 1261). Box size sqrt(area) at 640x360: min 18.7, p25 30.0, median 38.3, p75 48.6, max 300.3 px; 18 are below 32 px, 34 are 32-64 px, 7 are 64 px or more, none below 16 px.

Time-sequence groups of kept images (same timestamp parsing as the earlier audit, computed over all timestamped images). 740 kept images carry a timestamp; 117 (1 POSITIVE, 116 NEGATIVE) are numbered `vlcsnap-NNNNN` files with no timestamp and cannot be grouped. At a 10 s gap: 243 groups in total, 152 contain at least one kept image (21 with a positive, 143 with a negative, 12 with both); the largest group holds 52 kept images (next 45, 29, 24, 24); 51 groups have a single kept image. At a 60 s gap: 43 groups, 34 with kept images (11 with a positive, 33 with a negative, 10 with both); largest 116 kept images (next 91, 85, 64, 53). Kept images by date (positive, negative): 02-16 (0, 1); 02-18 (5, 317); 02-19 (21, 186); 02-23 (7, 2); 02-26 (8, 108); 03-16 (0, 85). A leakage-safe split must group by sequence (and treat the untimestamped files as one block or find another grouping). No split was created.

### Limits of this review (read before relying on it)
- Done by an AI model, not a human, looking at small upscaled crops (256 px tiles, 6 per row, displayed at reduced resolution), one pass, no second reviewer, no inter-rater measurement. Verdicts on small or low-contrast tiles are low-confidence by nature. The "unclear" bucket (79% of pothole boxes, 67% of manholes) reflects the 640x360 resolution and the many boxes under 32 px, not a quality score of the dataset.
- Exclusion-only: it does not repair boxes (many are loose or mis-sized), does not add missing potholes, and is not a re-annotation. The POSITIVE set (42 images, 59 boxes) is small and skewed toward larger, clearer potholes on the rural road sequence; it is unlikely to be enough alone for training.
- The 815 NEGATIVE images rest on the assumption above. The full-frame pass used 384x216 tiles, so small unlabelled potholes will have been missed; the `unlabelled_defect` count (23) is a lower bound. Manhole "intact" means no visible damage at crop resolution.
- The ok/reject boundaries (patch vs pothole, puddle in a depression vs puddle on intact surface) are judgement calls; patches and puddle-filled holes were rejected conservatively.
- Tile verdicts were typed per sheet from visual inspection; a transcription slip in a one-letter code is possible even though line lengths and totals (1261 and 957) were checked.
- Licence: the Zenodo record's licence was not verified.
