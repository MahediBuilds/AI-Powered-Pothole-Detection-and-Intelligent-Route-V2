"""Near-duplicate leakage audit for a converted (or scanned) RDD2022 manifest.

Read-only with respect to the dataset. Computes a 64-bit difference hash (dHash) per image and reports:

A. Whether numeric id order looks like capture order: Hamming distance between images whose ids
   differ by 1, compared with random same-country pairs.
B. Near-duplicates across splits among INCLUDED images: for every val/test image, the closest
   train image of the same country, and likewise val against test.

Usage:
    python scripts/leakage_audit.py <manifest.csv> <raw_dir> <out.json>

A dHash distance of 0-4 bits (of 64) means the two images are visually near-identical. Road scenes
resemble each other, so distances around 10 and above are not evidence of duplication.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NEAR = 4      # bits: near-identical
SIMILAR = 8   # bits: very similar


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def popcount(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "frac_le_%d" % NEAR: round(float((a <= NEAR).mean()), 4),
        "frac_le_%d" % SIMILAR: round(float((a <= SIMILAR).mean()), 4),
    }


def main() -> int:
    manifest_path, raw_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    rows = [r for r in rows if r["image_relpath"]]
    print("hashing %d images" % len(rows), flush=True)
    for i, r in enumerate(rows):
        r["_h"] = dhash(raw_dir / r["image_relpath"])
        r["_id"] = int(r["numeric_id"])
        if (i + 1) % 5000 == 0:
            print("  %d" % (i + 1), flush=True)

    rng = random.Random(0)
    by_country = defaultdict(list)
    for r in rows:
        by_country[r["country"]].append(r)

    report = {"manifest": manifest_path.name, "n_images_hashed": len(rows), "thresholds_bits": [NEAR, SIMILAR],
              "id_order": {}, "cross_split": {}, "examples": []}

    # A. id order vs capture order (all labelled images, regardless of status)
    for country, items in sorted(by_country.items()):
        items.sort(key=lambda r: r["_id"])
        adjacent = [bin(a["_h"] ^ b["_h"]).count("1") for a, b in zip(items, items[1:]) if b["_id"] - a["_id"] == 1]
        rand = []
        for _ in range(min(20000, len(items) * 5)):
            a, b = rng.sample(items, 2)
            rand.append(bin(a["_h"] ^ b["_h"]).count("1"))
        report["id_order"][country] = {"adjacent_ids": quantiles(adjacent), "random_pairs": quantiles(rand)}

    # B. cross-split near duplicates among included images
    examples = []
    for country, items in sorted(by_country.items()):
        inc = [r for r in items if r["status"] == "included"]
        split_rows = {s: [r for r in inc if r["split"] == s] for s in ("train", "val", "test")}
        out = {}
        for query, ref in (("val", "train"), ("test", "train"), ("val", "test")):
            q, t = split_rows[query], split_rows[ref]
            if not q or not t:
                continue
            ref_hashes = np.array([r["_h"] for r in t], dtype=np.uint64)
            mins, id_gaps_near, n_near, n_similar = [], [], 0, 0
            for r in q:
                d = popcount(ref_hashes ^ np.uint64(r["_h"]))
                j = int(d.argmin())
                dist = int(d[j])
                mins.append(dist)
                if dist <= SIMILAR:
                    n_similar += 1
                if dist <= NEAR:
                    n_near += 1
                    gap = abs(r["_id"] - t[j]["_id"])
                    id_gaps_near.append(gap)
                    examples.append({"country": country, "query": r["source_id"], "query_split": query,
                                     "match": t[j]["source_id"], "match_split": ref, "bits": dist, "id_gap": gap,
                                     "query_positive": r["is_positive"], "match_positive": t[j]["is_positive"]})
            gaps = Counter("1" if g == 1 else "2-10" if g <= 10 else "11-200" if g <= 200 else ">200" for g in id_gaps_near)
            out["%s_vs_%s" % (query, ref)] = {
                "n_query": len(q), "n_ref": len(t),
                "n_near_le_%d" % NEAR: n_near, "n_similar_le_%d" % SIMILAR: n_similar,
                "frac_near": round(n_near / len(q), 4),
                "min_distance": quantiles(mins),
                "id_gap_of_near_matches": dict(gaps),
            }
        report["cross_split"][country] = out
    examples.sort(key=lambda e: (e["bits"], e["country"], e["query"]))
    report["n_near_examples"] = len(examples)
    report["examples"] = examples[:400]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    slim = {k: v for k, v in report.items() if k != "examples"}
    print(json.dumps(slim, indent=1, sort_keys=True))
    print("first examples:")
    for e in examples[:25]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
