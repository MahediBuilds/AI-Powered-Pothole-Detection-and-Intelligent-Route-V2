# RDD2022 dataset specification

Everything here was checked on 2026-11-11. Each fact is tagged with how it was established:

- **[META]** read from the figshare record or the official GitHub README.
- **[ZIP]** read from the archive's own directory listing over HTTP Range requests (no image data downloaded).
- **[XML]** computed by parsing all 38,385 annotation files, fetched as a few MB of byte ranges.
- **[UNVERIFIED]** not confirmed from a primary source.

The image files themselves have not been inspected yet. The converter verifies them at conversion time.

## Source and licence

| Item | Value |
|---|---|
| Title | RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022 [META] |
| DOI | 10.6084/m9.figshare.21431547.v1 [META] |
| File | `RDD2022_released_through_CRDDC2022.zip`, 13,264,172,619 bytes [META] |
| MD5 | `b62bd51d2ffcfaa76c60f234f0cc2bb3` [META] |
| Download URL | `https://ndownloader.figshare.com/files/38030910` (Range requests work) [ZIP] |
| Code and README | `https://github.com/sekilab/RoadDamageDetector` [META] |
| Data article | Arya et al., "RDD2022: A multi-national image dataset for automatic Road Damage Detection", arXiv 2209.08538 [META] |

**Licence discrepancy.** The figshare record says CC BY 4.0. The GitHub README says the images are under CC BY-SA 4.0. We follow the stricter one: attribute the authors, and share any redistributed derivative of the images under CC BY-SA 4.0. Cite both the figshare record and the data article in the report.

**Dead links.** The per-country S3 links in the GitHub README returned HTTP 403 on 2026-11-11. Figshare is the only working source we found.

## Archive structure

The outer zip holds seven nested zips, all stored uncompressed, so each one is a contiguous byte range of the outer file and can be downloaded on its own. [ZIP]

| Nested entry | Bytes |
|---|---|
| `RDD2022/China_Drone.zip` | 160,193,541 |
| `RDD2022/China_MotorBike.zip` | 192,030,116 |
| `RDD2022/Czech.zip` | 257,117,218 |
| `RDD2022/India.zip` | 526,709,145 |
| `RDD2022/Japan.zip` | 1,072,598,857 |
| `RDD2022/Norway.zip` | 10,611,186,814 |
| `RDD2022/United_States.zip` | 444,335,958 |

Inside each nested zip [ZIP]:

```
<Country>/train/images/<Country>_NNNNNN.jpg
<Country>/train/annotations/xmls/<Country>_NNNNNN.xml
<Country>/test/images/<Country>_NNNNNN.jpg      (no annotations; China_Drone has no test folder)
```

`NNNNNN` is a six-digit zero-padded number. Every training image has an XML file and every XML file has an image. [ZIP]

## Counts per subset

| Subset | Labelled images | Unlabelled test images | Images with D40 | D40 boxes | Zero-object XMLs | Declared image size |
|---|---|---|---|---|---|---|
| India | 7,706 | 1,959 | 1,530 | 3,187 | 3,921 | 720x720 |
| Japan | 10,506 | 2,627 | 1,390 | 2,243 | 794 | 600x600 (319 files at other square sizes) |
| Czech | 2,829 | 709 | 154 | 197 | 1,757 | 600x600 |
| United_States | 4,805 | 1,200 | 116 | 135 | 0 | 640x640 |
| China_MotorBike | 1,977 | 500 | 164 | 235 | 0 | 512x512 |
| Norway | 8,161 | 2,040 | 256 | 461 | 5,247 | about 4040x2035 |
| China_Drone | 2,401 | 0 | 64 | 86 | 5 | 512x512 |
| **Total** | **38,385** | **9,035** | **3,674** | **6,544** | | |

Image counts are [ZIP]; the rest are [XML].

## Annotation format

PASCAL VOC XML. [META, XML]

- Root `<annotation>` with `<folder>`, `<filename>`, `<size>`, optional `<segmented>`, and zero or more `<object>`.
- `<size>` is always present with `<width>` and `<height>`. Element order varies, and Norway's `<depth/>` is empty.
- Each `<object>` has `<name>` and `<bndbox>` with `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>` in pixels.
- Coordinates are integers except in Norway, where they are floats such as `1138.46`.
- `<pose>`, `<truncated>`, `<difficult>`, `<occluded>` and `<attributes>` are optional and missing on some objects.

Known anomalies [XML]: one degenerate box (`Japan_001265`, label D20, zero width), two boxes with a side under 2 px, and no box that exceeds the declared image size.

## Labels

Official label map: `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole. [META] The challenge page says annotations for other categories "may be ignored". [META]

**`D40` is the only pothole label.** Other strings found in the XMLs [XML]:

| Subset | Extra labels |
|---|---|
| India | D44, D01, D43, D11, D50, and one `D0w0` |
| Japan | D44, D50, D43 |
| China_MotorBike | Repair |
| China_Drone | Repair, Block crack |

The meaning of D01, D11, D43, D44 and D50 comes from earlier RDD releases and is [UNVERIFIED] for RDD2022. The converter treats every non-D40 label as "not a pothole" and counts it.

## What this means for the converter

- **Subsets used by default:** India, Japan, Czech, United_States, China_MotorBike. That is 27,823 labelled images, 3,354 of them with potholes, and about 2.5 GB to download.
- **China_Drone is excluded.** The subset name indicates drone imagery, which is not a forward-facing view. The exact viewpoint is [UNVERIFIED].
- **Norway is off by default.** It is 10.6 GB for 256 pothole images, and at 640 px its very wide frames shrink potholes to a few pixels. It can be enabled in the config.
- **Negatives.** A labelled image with no D40 object is treated as pothole-free. This rests on an assumption: that annotators marked every pothole in every labelled image. D40 appears in every subset, which supports it, but exhaustive annotation is [UNVERIFIED]. Some negatives will therefore contain unmarked potholes.
- **Unlabelled test images are never negatives.** They have no annotations at all.
- **No sequence identifiers exist.** Country and numeric id are the only grouping. [ZIP]
- **Ids are not in capture order.** Measured on the downloaded images (see below).
- **Split strategy.** The converter groups near-duplicate frames by perceptual hash and assigns whole groups to a split. Splitting by blocks of consecutive ids, the first design, does not work on this dataset.
- **The RDD2022 test split is a secondary check.** The project's final test set is our own forward-facing footage, which never passes through this converter.

## Leakage findings on the downloaded data (2026-11-11)

Measured on the 27,823 labelled images of the five default subsets with a 64-bit difference hash (`scripts/leakage_audit.py`). Raw outputs are in `data/logs/`.

- **Images with adjacent ids are no more alike than random pairs.** The median hash distance is 25 to 31 bits in every subset for both. So ids are shuffled relative to capture order.
- **The dataset does contain consecutive frames of the same drive.** Example: `India_003424` and `India_002200` show the same road a few metres apart.
- **A split by id block therefore leaks like a random per-image split.** With 200-id blocks, the share of validation and test images that had a training image within 4 bits was:

| Subset | Validation | Test |
|---|---|---|
| India | 17.6% | 17.8% |
| China_MotorBike | 9.5% | 14.1% |
| Czech | 2.3% | 1.0% |
| Japan | 1.1% | 0.4% |
| United_States | 1.0% | 1.0% |

- **One exact duplicate pair exists:** `United_States_001623` and `United_States_003917` are byte-identical files.
- **Near-duplicates chain into very unequal groups.** At 8 bits, single-linkage grouping gives one large group per subset (4,926 of India's 7,706 images) and thousands of single images. Large groups stay in training; validation and test are filled from the rest.
- **Limit of the method.** A 64-bit hash of a road scene is coarse. Unrelated scenes with the same layout can be merged, which only makes groups larger. Frames of the same road taken further apart can be missed, so some scene-level overlap between splits remains.
- **Image files are otherwise clean.** No corrupt or missing files, no invalid pothole boxes, and every image matches its declared size. Two India images carry a non-default EXIF orientation; both are pothole-free.
