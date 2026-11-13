"""HRP4K full audit stage 1: integrity, labels, COCO comparison, EXIF, hashes, box sizes, empties-vs-positives stats.
usage: hrp4k_full_integrity.py EXTRACTED_DIR OUT_DIR   (reads EXTRACTED_DIR/HRP4K, writes OUT_DIR/full_*.json)"""
import sys, os, json, collections, statistics
import numpy as np
from PIL import Image, ExifTags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrp4k_hashing import hash_dir

def main():
    root, out = os.path.join(sys.argv[1], "HRP4K"), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    SPL = ["train", "valid", "test"]
    R = {"splits": {}}
    paths = []
    imgs = {}
    labs = {}
    for s in SPL:
        imgs[s] = sorted(f[:-4] for f in os.listdir(f"{root}/{s}/images"))
        labs[s] = sorted(f[:-4] for f in os.listdir(f"{root}/{s}/labels"))
        paths += [f"{root}/{s}/images/{n}.jpg" for n in imgs[s]]
        R["splits"][s] = {"images": len(imgs[s]), "labels": len(labs[s]),
                          "n_images_without_label": len(set(imgs[s]) - set(labs[s])),
                          "n_labels_without_image": len(set(labs[s]) - set(imgs[s])),
                          "images_without_label": sorted(set(imgs[s]) - set(labs[s]))[:20],
                          "labels_without_image": sorted(set(labs[s]) - set(imgs[s]))[:20]}
    H = hash_dir(paths, workers=int(os.environ.get("W", "6")))
    for h in H:
        p = h["path"].replace("\\", "/").split("/")
        h["split"], h["name"] = p[-3], p[-1][:-4]
        h["path"] = "/".join(p[-4:])
    json.dump(H, open(f"{out}/full_hashes.json", "w"))
    byk = {(h["split"], h["name"]): h for h in H}
    R["decode_failures"] = [h["path"] for h in H if not h["ok"]]
    R["resolution"] = dict(collections.Counter(f'{h["w"]}x{h["h"]}' for h in H if h["ok"]))
    R["modes_formats"] = dict(collections.Counter(f'{h["fmt"]}/{h["mode"]}' for h in H if h["ok"]))
    R["exif_orientation"] = dict(collections.Counter(str(h["exif_orient"]) for h in H if h["ok"]))

    bad = collections.Counter()
    badex = collections.defaultdict(list)
    boxes = {}
    for s in SPL:
        for n in labs[s]:
            rows = []
            seen = set()
            for ln, line in enumerate(open(f"{root}/{s}/labels/{n}.txt").read().splitlines()):
                if not line.strip():
                    continue
                t = line.split()

                def flag(k):
                    bad[k] += 1
                    if len(badex[k]) < 5:
                        badex[k].append(f"{s}/{n}:{ln}")
                if len(t) != 5:
                    flag("not_5_fields")
                    continue
                try:
                    c = int(t[0])
                    v = list(map(float, t[1:]))
                except ValueError:
                    flag("unparsable")
                    continue
                if c != 0:
                    flag("class_not_0")
                x, y, w, h = v
                if not all(0 <= a <= 1 for a in v):
                    flag("coord_outside_0_1")
                if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                    flag("box_extends_outside_image")
                if w <= 0 or h <= 0:
                    flag("zero_area")
                if tuple(t) in seen:
                    flag("duplicate_row")
                seen.add(tuple(t))
                rows.append((x, y, w, h))
            boxes[(s, n)] = rows
    R["label_problems"] = dict(bad)
    R["label_problem_examples"] = dict(badex)
    R["boxes"] = {s: sum(len(boxes[(s, n)]) for n in labs[s]) for s in SPL}
    R["empty_labels"] = {s: sum(1 for n in labs[s] if not boxes[(s, n)]) for s in SPL}
    R["empty_label_files_zero_bytes"] = {s: sum(1 for n in labs[s] if os.path.getsize(f"{root}/{s}/labels/{n}.txt") == 0) for s in SPL}

    cmp = {}
    for s in SPL:
        j = json.load(open(f"{root}/{s}.json"))
        id2f = {i["id"]: i["file_name"][:-4] for i in j["images"]}
        sz = {i["file_name"][:-4]: (i["width"], i["height"]) for i in j["images"]}
        ann = collections.defaultdict(list)
        for a in j["annotations"]:
            ann[id2f[a["image_id"]]].append(a)
        r = {"coco_images": len(j["images"]), "coco_annotations": len(j["annotations"]), "categories": j.get("categories"),
             "coco_files_not_in_dir": sorted(set(id2f.values()) - set(imgs[s]))[:10],
             "dir_files_not_in_coco": sorted(set(imgs[s]) - set(id2f.values()))[:10],
             "images_count_mismatch": 0, "images_box_mismatch_gt_1px": 0, "max_abs_px_diff": 0.0, "size_mismatch_vs_real": 0,
             "iscrowd_nonzero": sum(1 for a in j["annotations"] if a.get("iscrowd")),
             "nonempty_segmentation": sum(1 for a in j["annotations"] if a.get("segmentation"))}
        ex = []
        for n in imgs[s]:
            W, Hh = sz.get(n, (3840, 2160))
            hh = byk[(s, n)]
            if hh["ok"] and (hh["w"], hh["h"]) != (W, Hh):
                r["size_mismatch_vs_real"] += 1
            a = sorted([tuple(x["bbox"]) for x in ann.get(n, [])])
            b = sorted([((x - w / 2) * W, (y - h / 2) * Hh, w * W, h * Hh) for x, y, w, h in boxes[(s, n)]])
            if len(a) != len(b):
                r["images_count_mismatch"] += 1
                ex.append(n)
                continue
            d = max([max(abs(p - q) for p, q in zip(u, v)) for u, v in zip(a, b)] or [0])
            r["max_abs_px_diff"] = max(r["max_abs_px_diff"], d)
            if d > 1:
                r["images_box_mismatch_gt_1px"] += 1
                ex.append(n)
        r["mismatch_examples"] = ex[:10]
        cmp[s] = r
    R["coco_vs_txt"] = cmp

    allb = [(s, n, x, y, w, h) for (s, n), rows in boxes.items() for x, y, w, h in rows]
    W0, H0 = 3840, 2160
    WIDTHS = (3840, 1920, 1280, 960, 640)


    def sqrt_area(sub, width):
        k = width / W0
        return np.array([np.sqrt(w * W0 * h * H0) * k for _, _, _, _, w, h in sub])


    def st(sa, width):
        return {"width": width, "n": int(len(sa)), "median_sqrt_area_px": float(np.median(sa)), "p5": float(np.percentile(sa, 5)),
                "p95": float(np.percentile(sa, 95)), "share_lt_8px": float((sa < 8).mean()), "share_lt_16px": float((sa < 16).mean())}


    R["box_size_by_width"] = [st(sqrt_area(allb, w), w) for w in WIDTHS]
    R["box_size_by_width_per_split"] = {s: [st(sqrt_area([b for b in allb if b[0] == s], w), w) for w in WIDTHS] for s in SPL}
    yc = np.array([b[3] for b in allb])
    R["box_yc_distribution"] = {"share_yc_gt_0.5": float((yc > 0.5).mean()), "share_yc_gt_0.4": float((yc > 0.4).mean()),
                                "min": float(yc.min()), "p1": float(np.percentile(yc, 1)), "p5": float(np.percentile(yc, 5))}
    R["boxes_per_positive_image"] = {s: dict(sorted(collections.Counter(len(boxes[(s, n)]) for n in labs[s] if boxes[(s, n)]).items())) for s in SPL}

    E = {}
    for s in SPL:
        for grp, sel in (("empty", lambda r: not r), ("positive", lambda r: bool(r))):
            names = [n for n in imgs[s] if sel(boxes[(s, n)])]
            fs = [byk[(s, n)]["size"] for n in names]
            nums = [int(n) for n in names]
            E[f"{s}/{grp}"] = {"n": len(names), "filesize_median": statistics.median(fs), "filesize_mean": statistics.mean(fs),
                               "filesize_p10": float(np.percentile(fs, 10)), "filesize_p90": float(np.percentile(fs, 90)),
                               "number_median": statistics.median(nums), "number_min": min(nums), "number_max": max(nums)}
        seq = [0 if boxes[(s, n)] else 1 for n in imgs[s]]
        runs = []
        cur = 1
        for a, b in zip(seq, seq[1:]):
            if a == b:
                cur += 1
            else:
                runs.append((a, cur))
                cur = 1
        runs.append((seq[-1], cur))
        E[f"{s}/runs"] = {"n_runs": len(runs), "max_empty_run": max([l for v, l in runs if v == 1] or [0]),
                          "max_pos_run": max([l for v, l in runs if v == 0] or [0]),
                          "empty_by_number_mod3": dict(collections.Counter(int(n) % 3 for n in imgs[s] if not boxes[(s, n)])),
                          "empty_decile_counts": [int(sum(seq[int(len(seq) * q / 10):int(len(seq) * (q + 1) / 10)])) for q in range(10)]}
    R["empties_vs_positives"] = E

    EX = collections.Counter()
    models = collections.Counter()
    dts = []
    info_keys = collections.Counter()
    for p in paths:
        with Image.open(p) as im:
            ex = im.getexif()
            for k in ex:
                EX[ExifTags.TAGS.get(k, str(k))] += 1
            if ex.get_ifd(0x8825):
                EX["GPS_ifd_nonempty"] += 1
            models[str((ex.get(271), ex.get(272)))] += 1
            for k in im.info:
                info_keys[k] += 1
            d = ex.get(306) or ex.get_ifd(0x8769).get(36867)
            if d:
                dts.append(d)
    R["exif"] = {"images": len(paths), "tag_counts": dict(EX), "make_model": dict(models), "n_with_datetime": len(dts),
                 "datetime_min": min(dts) if dts else None, "datetime_max": max(dts) if dts else None, "jpeg_info_keys": dict(info_keys)}
    json.dump(R, open(f"{out}/full_integrity.json", "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("splits", "decode_failures", "resolution", "label_problems", "boxes", "empty_labels", "exif")}, indent=1)[:3500])



if __name__ == "__main__":
    main()
