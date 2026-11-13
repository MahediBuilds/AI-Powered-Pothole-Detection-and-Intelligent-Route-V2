"""Investigation 1: categorise the candidate's false positives on HRP4K validation (validation only).

Modes
  prepare   threshold (recomputed: pooled-val max-F1 of candidate@1280), mechanical split of FPs, review crops +
            review sheets, baseline@640 mechanical counts.
  finalize  merge the visual labels in fp_categories/labels/*.csv (one row per reviewed item: id,category,confidence,note),
            write fp_review.csv, category tables with bootstrap shares, breakdowns and per-category contact sheets.

    .venv/Scripts/python.exe scripts/val_investigations/fp_categories.py prepare
    .venv/Scripts/python.exe scripts/val_investigations/fp_categories.py finalize

The categories are a single AI visual pass (not ground truth). Nothing here changes labels or recomputes precision as if
the judgements were truth; a clearly labelled 'if accepted' sensitivity line is the only use.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fp_mech as M  # noqa: E402

OUTD = C.OUT / "fp_categories"
CATEGORIES = ["manhole_or_utility_cover", "drain_or_grate", "patch_or_repair", "road_marking", "shadow_or_stain_or_wet",
              "crack_or_surface_wear", "plausible_unlabelled_pothole", "debris_or_object", "road_edge_or_offroad", "unclear"]
CONFIDENCES = ["high", "medium", "low"]
MAX_REVIEW = 350
SAMPLE_SEED = 0
BANDS = [("thr-0.4", None, 0.4), ("0.4-0.6", 0.4, 0.6), (">0.6", 0.6, None)]
TILE, BAND = 400, 90
PER_SHEET = 6


def pooled_threshold(model, imgsz, pool):
    from ensemble import maxf1_threshold
    P = C.build_P(pool, C.load_model_preds(model, imgsz, pool), iou_thrs=np.array([0.5]))
    return maxf1_threshold(P, np.ones(len(pool.stems)))


def mechanical_table(hv, preds, thr):
    rows = []
    for stem in hv.stems:
        pb, pc = preds[stem]
        gb = hv.gts[stem][0]
        for r in M.mechanical_split(pb, pc, gb, thr):
            b = pb[r["idx"]]
            rows.append(dict(image=stem, x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                             conf=r["conf"], cls=r["cls"], max_iou=r["max_iou"], gt_index=r["gt_index"],
                             empty_image=int(len(gb) == 0), n_gt=int(len(gb))))
    return rows


def counts_of(rows):
    out = {k: 0 for k in ("tp", "duplicate", "loc", "bg")}
    for r in rows:
        out[r["cls"]] += 1
    out["fp_total"] = out["duplicate"] + out["loc"] + out["bg"]
    out["fp_on_empty_images"] = sum(1 for r in rows if r["cls"] != "tp" and r["empty_image"])
    out["detections"] = len(rows)
    return out


# ----------------------------------------------------------------------------- rendering
def render_tile(hv, row, item_id):
    import cv2
    img = cv2.imread(str(C.guard(hv.path_of[row["image"]])))
    H, W = img.shape[:2]
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    bw, bh = x2 - x1, y2 - y1
    S = int(round(max(3 * max(bw, bh), 320)))
    S = min(S, W, H) if min(W, H) >= 160 else S
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X0 = int(round(min(max(cx - S / 2, 0), W - S))); Y0 = int(round(min(max(cy - S / 2, 0), H - S)))
    crop = img[Y0:Y0 + S, X0:X0 + S].copy()
    sc = TILE / S
    crop = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)

    def box(b, col, th):
        p1 = (int(round((b[0] - X0) * sc)), int(round((b[1] - Y0) * sc)))
        p2 = (int(round((b[2] - X0) * sc)), int(round((b[3] - Y0) * sc)))
        cv2.rectangle(crop, p1, p2, col, th)
    for g in hv.gts[row["image"]][0]:
        box(g, (0, 200, 0), 2)           # GT: green (BGR)
    box((x1, y1, x2, y2), (0, 0, 255), 2)  # detection: red
    tile = np.full((TILE + BAND, TILE, 3), 30, np.uint8)
    tile[:TILE] = crop
    th_w = 150
    th_h = int(round(th_w * H / W))
    th = cv2.resize(img, (th_w, th_h), interpolation=cv2.INTER_AREA)
    k = th_w / W
    cv2.rectangle(th, (int(X0 * k), int(Y0 * k)), (int((X0 + S) * k), int((Y0 + S) * k)), (0, 255, 255), 1)
    cv2.rectangle(th, (int(x1 * k), int(y1 * k)), (max(int(x2 * k), int(x1 * k) + 3), max(int(y2 * k), int(y1 * k) + 3)), (0, 0, 255), 2)
    th = th[:BAND - 6]
    tile[TILE + 3:TILE + 3 + th.shape[0], 3:3 + th_w] = th
    cv2.putText(tile, f"#{item_id} {row['cls']}", (th_w + 12, TILE + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, f"conf {row['conf']:.2f}  box {int(bw)}x{int(bh)}px", (th_w + 12, TILE + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(tile, f"{'EMPTY image' if row['empty_image'] else str(row['n_gt']) + ' GT in image'}", (th_w + 12, TILE + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(tile, "red=detection green=GT", (th_w + 12, TILE + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return tile


def make_sheet(tiles, path, cols=3):
    import cv2
    rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols]
        r += [np.full_like(tiles[0], 30)] * (cols - len(r))
        rows.append(np.concatenate(r, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


# ----------------------------------------------------------------------------- prepare
def prepare():
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "crops").mkdir(exist_ok=True)
    (OUTD / "review_sheets").mkdir(exist_ok=True)
    (OUTD / "labels").mkdir(exist_ok=True)
    C.verify_checkpoint("baseline"); C.verify_checkpoint("candidate")
    pool = C.load_pool()
    hv = C.ValSet("hrp4k_val")
    gmap = dict(zip(hv.stems, hv.groups))
    out = {}
    for model, sz, tag in (("candidate", 1280, "candidate@1280"), ("baseline", 640, "baseline@640")):
        thr, f1 = pooled_threshold(model, sz, pool)
        preds = C.load_model_preds(model, sz, pool)
        rows = mechanical_table(hv, preds, thr)
        for r in rows:
            r["group"] = gmap[r["image"]]
        out[tag] = dict(pooled_maxF1_threshold=thr, pooled_maxF1=f1, counts_hrp4k_val=counts_of(rows))
        if tag == "candidate@1280":
            cand_rows = rows
    C.write_json(OUTD / "mechanical_split_counts.json", out)

    fps = [r for r in cand_rows if r["cls"] in ("bg", "loc")]
    n_all = len(fps)
    fps.sort(key=lambda r: (r["image"], -r["conf"], r["x1"]))
    sampled = False
    if len(fps) > MAX_REVIEW:
        rng = np.random.default_rng(SAMPLE_SEED)
        pick = sorted(rng.choice(len(fps), MAX_REVIEW, replace=False))
        fps = [fps[i] for i in pick]
        sampled = True
    for i, r in enumerate(fps):
        r["id"] = i + 1
    items = []
    for i in range(0, len(fps), PER_SHEET):
        tiles = []
        for r in fps[i:i + PER_SHEET]:
            t = render_tile(hv, r, r["id"])
            cp = OUTD / "crops" / f"fp_{r['id']:03d}.png"
            import cv2
            cv2.imwrite(str(cp), t)
            r["crop"] = str(cp.relative_to(C.ROOT)).replace("\\", "/")
            tiles.append(t)
        make_sheet(tiles, OUTD / "review_sheets" / f"sheet_{i // PER_SHEET + 1:02d}.png")
    cols = ["id", "image", "group", "x1", "y1", "x2", "y2", "conf", "cls", "max_iou", "gt_index", "empty_image", "n_gt", "crop"]
    with open(OUTD / "fp_items.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in fps:
            wr.writerow([f"{r[c]:.3f}" if isinstance(r[c], float) else r[c] for c in cols])
    meta = dict(threshold=out["candidate@1280"]["pooled_maxF1_threshold"], n_bg_plus_loc=n_all, n_reviewed=len(fps),
                sampled=sampled, sample_seed=SAMPLE_SEED, n_sheets=(len(fps) + PER_SHEET - 1) // PER_SHEET, per_sheet=PER_SHEET)
    C.write_json(OUTD / "prepare_meta.json", meta)
    C.progress(f"fp_categories: prepared {len(fps)} items for review (bg+loc total {n_all}, sampled={sampled}), "
               f"{meta['n_sheets']} sheets in fp_categories/review_sheets/")
    print(json.dumps(out, indent=2), json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------- finalize
def load_labels():
    lab = {}
    for f in sorted((OUTD / "labels").glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                i = int(r["id"])
                if i in lab:
                    raise SystemExit(f"duplicate label for id {i} ({f.name})")
                if r["category"] not in CATEGORIES:
                    raise SystemExit(f"bad category {r['category']!r} for id {i}")
                if r["confidence"] not in CONFIDENCES:
                    raise SystemExit(f"bad confidence for id {i}")
                lab[i] = r
    return lab


def finalize():
    import cv2
    meta = json.loads((OUTD / "prepare_meta.json").read_text())
    items = list(csv.DictReader(open(OUTD / "fp_items.csv", newline="")))
    lab = load_labels()
    missing = [int(r["id"]) for r in items if int(r["id"]) not in lab]
    if missing:
        raise SystemExit(f"{len(missing)} items without a visual label, e.g. {missing[:10]}")
    extra = set(lab) - {int(r["id"]) for r in items}
    if extra:
        raise SystemExit(f"labels for unknown ids {sorted(extra)[:10]}")
    hv = C.ValSet("hrp4k_val")
    for r in items:
        L = lab[int(r["id"])]
        r.update(category=L["category"], category_confidence=L["confidence"], note=L["note"])
    cols = ["id", "image", "x1", "y1", "x2", "y2", "conf", "cls", "empty_image", "category", "category_confidence", "note", "crop"]
    with open(OUTD / "fp_review.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "box_x1y1x2y2", "conf", "mechanical_class", "empty_image", "category", "category_confidence", "note", "crop_path", "id", "group"])
        for r in items:
            wr.writerow([r["image"], f"{float(r['x1']):.1f},{float(r['y1']):.1f},{float(r['x2']):.1f},{float(r['y2']):.1f}", r["conf"],
                         r["cls"], r["empty_image"], r["category"], r["category_confidence"], r["note"], r["crop"], r["id"], r["group"]])

    cat = np.array([r["category"] for r in items])
    conf = np.array([float(r["conf"]) for r in items])
    empty = np.array([int(r["empty_image"]) for r in items]).astype(bool)
    mech = np.array([r["cls"] for r in items])
    groups = np.array([r["group"] for r in items])
    # group bootstrap: resample all HRP4K-val groups (same resampling scheme as the rest of the project), weight each item
    # by its image's group weight; share = weighted count of the category / weighted number of reviewed items
    gl = sorted(set(hv.groups))
    pos = {g: i for i, g in enumerate(gl)}
    gi = np.array([pos[g] for g in groups])
    rng = np.random.default_rng(0)
    G = len(gl)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(1000)])[:, gi].astype(float)  # (1000, n_items)

    def table(mask):
        n = int(mask.sum())
        out = []
        for c in CATEGORIES:
            m = mask & (cat == c)
            k = int(m.sum())
            tot = W[:, mask].sum(1)
            share_b = np.where(tot > 0, W[:, m].sum(1) / np.maximum(tot, 1e-12), np.nan)
            out.append(dict(category=c, n=k, share=k / n if n else float("nan"), ci95=C.pctl(share_b) if n else None))
        return n, out

    def band_mask(lo, hi):
        m = np.ones(len(items), bool)
        if lo is not None:
            m &= conf >= lo
        if hi is not None:
            m &= conf < hi
        return m

    res = dict(meta=meta, overall=dict(zip(("n", "table"), table(np.ones(len(items), bool)))),
               empty_images_only=dict(zip(("n", "table"), table(empty))),
               by_mechanical_class={k: dict(zip(("n", "table"), table(mech == k))) for k in ("bg", "loc")},
               by_conf_band={name: dict(zip(("n", "table"), table(band_mask(lo, hi)))) for name, lo, hi in BANDS})
    conf_cnt = {}
    for c in CATEGORIES:
        conf_cnt[c] = {k: int(((cat == c) & np.array([r["category_confidence"] == k for r in items])).sum()) for k in CONFIDENCES}
    res["visual_confidence_by_category"] = conf_cnt
    same = np.array([r["note"].startswith("[same-object]") for r in items])
    res["same_object_localisation_items"] = dict(
        n=int(same.sum()), by_mechanical_class={k: int((same & (mech == k)).sum()) for k in ("bg", "loc")},
        note="items whose detection lies on the very object a ground-truth box also covers (localisation/duplicate-box effect); "
             "they carry category 'unclear' because none of the listed categories names that situation")
    res["unclear_excluding_same_object"] = int(((cat == "unclear") & ~same).sum())
    mc = json.loads((OUTD / "mechanical_split_counts.json").read_text())
    res["mechanical"] = mc
    n_pl = int((cat == "plausible_unlabelled_pothole").sum())
    cc = mc["candidate@1280"]["counts_hrp4k_val"]
    tp, fp = cc["tp"], cc["fp_total"]
    res["if_accepted_sensitivity"] = dict(
        label="SENSITIVITY ONLY, NOT A MEASURED PRECISION: counts the AI-judged 'plausible_unlabelled_pothole' items as if they were "
              "correct detections. Single unverified visual pass; labels are unchanged.",
        n_plausible_unlabelled=n_pl, tp=tp, fp=fp, precision_as_measured=tp / (tp + fp),
        precision_if_plausible_accepted=(tp + n_pl) / (tp + fp),
        note=("The review covers all bg + loc false positives" if not meta["sampled"] else "sample only") + "; duplicates are not categorised")
    C.write_json(OUTD / "category_summary.json", res)

    # contact sheets per category
    (OUTD / "contact_sheets").mkdir(exist_ok=True)
    for c in CATEGORIES:
        ids = [r for r in items if r["category"] == c]
        for k in range(0, len(ids), 12):
            tiles = [cv2.imread(str(C.ROOT / r["crop"])) for r in ids[k:k + 12]]
            tiles = [cv2.resize(t, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA) for t in tiles]
            make_sheet(tiles, OUTD / "contact_sheets" / f"{c}_{k // 12 + 1:02d}.png", cols=4)
    write_md(res, items)
    C.progress("fp_categories: finalize done (fp_review.csv, category_summary.json, contact_sheets/)")


def write_md(res, items):
    def tab(t):
        L = ["| category | n | share | 95% CI (group bootstrap) |", "|---|---|---|---|"]
        for r in t["table"]:
            L.append(f"| {r['category']} | {r['n']} | {r['share']:.3f} | " + (f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if r["ci95"] else "n/a") + " |")
        return L
    mc = res["mechanical"]
    L = ["# False-positive categories, candidate@1280 on HRP4K validation", "",
         f"Threshold {mc['candidate@1280']['pooled_maxF1_threshold']:.4f} (pooled-val max-F1, recomputed). Categories are a SINGLE AI visual pass: not ground truth.", "",
         "## Mechanical split (no visual judgement)", "", "| model | threshold | detections | TP | FP total | (a) localisation 0.1<=IoU<0.5 | (b) duplicate | (c) background | FP on empty images |", "|---|---|---|---|---|---|---|---|---|"]
    for tag in ("candidate@1280", "baseline@640"):
        c = mc[tag]["counts_hrp4k_val"]
        L.append(f"| {tag} | {mc[tag]['pooled_maxF1_threshold']:.4f} | {c['detections']} | {c['tp']} | {c['fp_total']} | {c['loc']} | {c['duplicate']} | {c['bg']} | {c['fp_on_empty_images']} |")
    L += ["", f"## Visual categories, all reviewed items (a)+(c), n = {res['overall']['n']}", ""] + tab(res["overall"])
    so = res["same_object_localisation_items"]
    L += ["", f"Of the 'unclear' items, {so['n']} are tagged [same-object] (box on an object a GT box also covers; by mechanical class {so['by_mechanical_class']}); "
              f"the other {res['unclear_excluding_same_object']} 'unclear' items are tiny or ambiguous marks."]
    L += ["", f"## Empty (no-GT) images only, n = {res['empty_images_only']['n']}", ""] + tab(res["empty_images_only"])
    for k, v in res["by_conf_band"].items():
        L += ["", f"## Confidence band {k}, n = {v['n']}", ""] + tab(v)
    s = res["if_accepted_sensitivity"]
    L += ["", "## " + s["label"], "", f"TP {s['tp']}, FP {s['fp']}, measured precision {s['precision_as_measured']:.3f}; "
          f"if the {s['n_plausible_unlabelled']} 'plausible_unlabelled_pothole' items were accepted: {s['precision_if_plausible_accepted']:.3f}."]
    (OUTD / "CATEGORY_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"prepare": prepare, "finalize": finalize}[sys.argv[1]]()
