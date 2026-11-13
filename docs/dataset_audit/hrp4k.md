# Audit: HRP4K (Zenodo record 21941931, v1.01)
Record: CC BY 4.0 (per Zenodo metadata). v1.01 description: "Corrects an incomplete upload in v1.0; the full image set is now included." (v1.0, record 17522874, 8.66 GB zip, carries no such note itself; v1.01 zip 12,795,844,822 bytes.) Server honours HTTP Range (206). Only ~123 MB of images + ~6.4 MB annotations + 2.2 MB central directory were transferred; the full archive was NOT downloaded. Results: `hrp4k.json` (annotation audit), `hrp4k_samples.json` (sample images), `data/external/hrp4k/central_directory.json` (full listing). Scripts: `scripts/dataset_audit/{remote_zip,hrp4k_list,hrp4k_fetch,audit_hrp4k,hrp4k_samples}.py`. Sheets: `hrp4k/sheet_0..3.jpg`.

## Archive structure (central directory)
12019 entries: 6003 .jpg, 6003 .txt, 3 .json (train.json, valid.json, test.json, COCO-style). Layout `HRP4K/{train,valid,test}/{images,labels}/N.jpg|txt`. NO README, licence, yaml or class file inside the zip. Images: train 4203, valid 900, test 900 (~2.1 MB median each, ~13 GB total).
## Annotations (all fetched and audited)
- Single class "pothole" (id 0). YOLO txt (class xc yc w h) and COCO json agree (image/box counts identical). Boxes only; COCO `segmentation` empty.
- Boxes: train 5259, valid 1037, test 921. Empty images: train 1400/4203, valid 300/900, test 300/900 (exactly one third each; listed in JSON with zero annotations; whether they are verified true negatives is not documented).
- JSON image size: all 3840x2160 (also confirmed on 39 sampled files).
- Box size at native 4K (sqrt area median / p5 / p95): train 59/17/312 px, valid 65/21/219, test 63/19/274; median box ~100x33 px (about 2.6% x 1.5% of image). At 640x360 resize median ~10 px sqrt-area, p5 ~3 px.
- Validity: 0 out-of-bounds, 0 zero-area, 0 duplicate rows, 0 missing/extra files.
## Split / sequence
- Predefined train/valid/test (4203/900/900); how it was made is not documented. File numbers: train 0-4202, valid 20000-20899, test 30000-30899, each contiguous. The numbering is per-split, so cross-split leakage cannot be checked from names. Names do not encode video sequences. On 6 adjacent-number pairs from 3 train positions, dHash Hamming was 20-32 of 64 (not near-duplicates), so adjacent numbers are not obviously consecutive frames; between 39 sampled images the minimum Hamming was 8. Near-duplicate/leakage across splits NOT verified (would need full download). Capture timestamps/EXIF: none found in sampled files.
## Visual review (39 images: 18 train, 6 valid, 4 test + adjacency triples)
- Forward-facing vehicle camera, wide shots; Chinese road signage, daytime (sunny/overcast), summer vegetation, urban arterials, crossings, rural lanes. Windshield haze/glare bands visible in some frames. Capture location/camera not stated in the bundle.
- Label quality: many boxes are minute (10-20 px at 4K, visible only on zoom); some labelled objects are marginal (patched asphalt, a box beside a manhole cover in 633.jpg); an unlabelled manhole in an empty image (3239.jpg); one frame (3077) has ~10 boxes along the bottom edge in rough/cracked surface. Overall plausible but small-object dominated.

---
# Full audit (2026-11-12)
Full v1.01 archive downloaded and audited locally. Scripts: `scripts/dataset_audit/hrp4k_download.py` (parallel range download), `hrp4k_full_extract.py`, `hrp4k_hashing.py`, `hrp4k_full_integrity.py`, `hrp4k_full_dupes.py`, `hrp4k_full_crosssheet.py`, `hrp4k_full_sheets.py`. Machine-readable results: `hrp4k/full_integrity.json`, `hrp4k/full_dupes.json`, `hrp4k/full_hashes.json` (per-image sha256/dHash/pHash) and the `full_audit_2026_11_12` key of `hrp4k.json`. Sheets in `hrp4k/` (`leakage_pairs_*.jpg`, `empties_*.jpg`, `positives_*.jpg`, `crossdataset_closest_*.jpg`, `review_sheet_index.json`). Zip kept at `data/external/hrp4k/HRP4K.zip`, extracted to `data/external/hrp4k/extracted/` (only train/valid/test images, labels and the 3 JSON files; the archive contained no other member).

## Download and integrity
- Size 12,795,844,822 bytes and md5 b819f48e1abd90d1f9fef7acc6a2f69c match the Zenodo API. `zipfile.testzip()` (CRC of every member) passed. 12019 members, 0 unexpected names.
- 6003 images, all decode fully (PIL verify + load): 0 failures. All 3840x2160, JPEG/RGB. EXIF Orientation present on only 3 images (values 1, 1, 8).
- Every image has a label file and vice versa in all splits (0 missing).
- Labels: 0 non-zero class ids, 0 coordinates outside [0,1], 0 boxes extending outside the image, 0 zero-area, 0 duplicate rows. Boxes train 5259 / valid 1037 / test 921; empty label files 1400 / 300 / 300 (all 0 bytes).
- COCO JSON vs txt: same image sets, same per-image box counts, max coordinate difference 4.5e-13 px. 0 iscrowd, 0 segmentations.
- Metadata: no camera make/model, no datetime, no GPS in any of the 6003 files (JFIF only; 3 files carry EXIF with Orientation/dpi, 1 has XMP).
- Empties vs positives: median file size 2.17 MB vs 2.23 MB (train), 2.06 vs 2.08 (valid), 2.14 vs 2.10 (test). Empties are spread evenly over file numbers (train decile counts 126-163 of 4203; mod-3 counts 451/462/487, so not a fixed-stride pick); longest run of consecutive empty file numbers is 7.

## Duplicates and leakage
Cluster definition is the one in `src/roadsense/data/splits.py` (single linkage, Hamming <= T on 64-bit hash); the dHash recipe is copied from `src/roadsense/data/rdd2022.py`. pHash = grey 32x32, DCT, 8x8 low block, median threshold.
- SHA-256 exact duplicates: 0 groups.
- 64-bit dHash is nearly uninformative on this data: at T=8 one component holds 4250 of 6003 images; at T=12 one component holds 5627. Median nearest-neighbour distance is 6 bits train-to-train, 7 valid-to-train, 7 test-to-train. Cross-split counts under dHash therefore mostly reflect chaining of similar-looking road frames, not duplicates.

| Hash | T | clusters >=2 | within train / valid / test (clusters) | cross-split clusters | valid imgs sharing a cluster with train | test imgs sharing a cluster with train |
|---|---|---|---|---|---|---|
| dHash | 4 | 214 | 139 / 29 / 22 | 24 | 218 | 270 |
| dHash | 8 | 88 | 55 / 15 / 9 | 9 (largest component 4250 imgs) | 575 | 636 |
| dHash | 12 | 16 | 9 / 2 / 2 | 3 (largest 5627 imgs) | 814 | 837 |
| pHash | 4 | 171 | 129 / 22 / 19 | 1 (15 imgs) | 1 | 1 |
| pHash | 8 | 430 | 294 / 54 / 50 | 32 | 51 | 76 |
| pHash | 12 | 277 | 163 / 35 / 28 | 51 (largest component 3193 imgs) | 438 | 486 |

- Visual same-scene check: 150 random valid+test images (seed 0), 3 nearest train images each by pHash (min pHash distance to train: median 14 bits; 1 image at 4 bits, 9-10 at 6). 15 sheets `leakage_pairs_00..14.jpg` (row = query, then 3 neighbours), all viewed. Result: 0 of 150 queries show the same frame or same moment among their 3 nearest train images. About 4 queries show the same road corridor or road type (e.g. valid/20488 vs train/653, test/30207 vs train/784 and 3079, test/30825 vs train/815 and 3150); these were judged different locations or unclear, not counted as leakage. The check covers only the 3 nearest pHash neighbours of 150 queries (about 17 percent of valid+test), not all pairs.
- Within-train near-duplicates are visible on the sheets: several train frames of the same junction or road section sit at file numbers far apart (e.g. train 2121/3361/3503, 2574/3822, 1100/2334, 1764/239/3749, 1411/1546/2714, 3404/1223), so file numbers are not temporal order and train is internally redundant.

## Cross-dataset overlap
Compared by sha256, dHash and pHash against `data/processed/rdd2022_pothole` (13,788 images incl. 934 China_MotorBike, all 512x512) and `data/external/kaggle_pcm_extracted` (2009 images, 640x360).
- sha256 matches: 0 and 0. pHash <= 4 pairs: 0 and 0. pHash <= 8: 32 pairs with RDD2022, 4 with Kaggle PCM; no pair has both dHash and pHash <= 8. The 12 closest pairs of each of the three comparisons (`crossdataset_closest_*.jpg`) were viewed: none shows the same scene (RDD Japan/India roadside lanes; China_MotorBike downward close-ups of road surface; Kaggle PCM Rome dashcam frames with an orange bonnet).
- Limitation: China_MotorBike images are 512x512 views with a different (close, downward) geometry; a global hash cannot detect a crop of an HRP4K frame. No local-feature (SIFT/ORB) matching was run.

## Empty-label images (120 random empties, seed 0, 20 sheets `empties_00..19.jpg`)
Tiles show the lower 60 percent of the frame (from y = 0.40 H) at 1280 px wide; 97.0 percent of labelled box centres lie at y > 0.4 H and the minimum box centre is at 0.31 H, so a pothole above that line could be missed. All 20 sheets viewed. Manual counts (subjective boundaries):
- clearly visible unlabelled pothole or depression: 0
- hard-negative content (manhole or drain grate, patch, seam, hairline or heavy cracking, oil or tyre stains, worn markings, strong shadow): 51
- plain road, no confounder: 69
- ambiguous: 0. Closest to ambiguous, none judged a pothole: heavy cracking in train 1712, 2784 and valid 20173; dark patches or patch seams in train 3945, 2322 and 2268.
Potholes under about 10 px at 1280 wide would not be visible on these tiles.

## Positive label quality (120 random positives, seed 0, 20 sheets `positives_00..19.jpg`)
Same crop, boxes drawn with 5 px padding; all 20 sheets viewed. Counts are boxes, approximate and subjective (tiny boxes cannot be judged at 1280 px; boxes cut by the crop line counted when partly visible); about 212 boxes judged.
- clearly a pothole, depression or spall: about 92
- on something else (manhole cover, drain grate or kerb edge, repaired patch, transverse crack or joint, marking, over-large box spanning a patch): about 22 (e.g. train 349 manhole, 94 manhole ring, 98 arrow marking, 1380 transverse joint with 3 boxes, 1279 large box along a crack, 601 and 880 drain at a cone)
- ambiguous (tiny, under about 15 px at 1280 wide, or indistinct): about 98
- visible potholes with no box: none judged unambiguous. Unboxed patches, dark stains and cracks occur in several images (372, 1993, 30449, 3472) and were not counted as missed potholes.

## Box size vs resolution (7217 boxes; sqrt(w*h) in pixels at the stated image width)
| width | median | p5 | p95 | share < 8 px | share < 16 px |
|---|---|---|---|---|---|
| 3840 | 60.3 | 17.2 | 291.6 | 0.04% | 3.7% |
| 1920 | 30.1 | 8.6 | 145.8 | 3.7% | 23.2% |
| 1280 | 20.1 | 5.7 | 97.2 | 13.6% | 39.3% |
| 960 | 15.1 | 4.3 | 72.9 | 23.2% | 52.9% |
| 640 | 10.0 | 2.9 | 48.6 | 39.3% | 70.4% |

## Not verified
How the splits were made (no metadata); capture date, camera or GPS (none in files); local-feature overlap with RDD2022 China_MotorBike; labels outside the sampled 240 images (manual counts are samples, not totals); potholes in the top 40 percent of frames; licence text inside the archive (none shipped; CC BY 4.0 comes from the Zenodo record only).

