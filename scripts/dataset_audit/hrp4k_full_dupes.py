"""HRP4K full audit stage 2: exact + near duplicates, cross-split leakage, nearest-train pHash pairs, cross-dataset overlap.
Cluster definition = src/roadsense/data/splits.py similarity_components: single linkage, Hamming(64-bit hash) <= T.
usage: hrp4k_full_dupes.py HASHES_DIR EXTRACTED_DIR OUT_DIR RDD_HASH.json KAGGLE_HASH.json
(reads full_hashes.json from HASHES_DIR; writes OUT_DIR/full_dupes.json, leakage_pairs_*.jpg, leakage_pairs_index.json)"""
import sys, os, json, collections, random
import numpy as np
from PIL import Image, ImageDraw

M1, M2, M4, H01 = (np.uint64(x) for x in (0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x0101010101010101))


def popcount(x):
    x = x - ((x >> np.uint64(1)) & M1)
    x = (x & M2) + ((x >> np.uint64(2)) & M2)
    x = (x + (x >> np.uint64(4))) & M4
    return (x * H01) >> np.uint64(56)


def dist_matrix(a, b):
    return popcount(a[:, None] ^ b[None, :]).astype(np.int16)


def components(arr, T):
    n = len(arr)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s in range(0, n, 500):
        d = dist_matrix(arr[s:s + 500], arr)
        r, c = np.nonzero(d <= T)
        for i, j in zip((r + s).tolist(), c.tolist()):
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def thumb(path, w, h, label):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.draft("RGB", (w * 2, h * 2))
        im = im.resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 14], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    return im


def main():
    hdir, ext, out, rddf, kagf = sys.argv[1:6]
    H = json.load(open(f"{hdir}/full_hashes.json"))
    N = len(H)
    split = np.array([h["split"] for h in H])
    names = [f'{h["split"]}/{h["name"]}' for h in H]
    R = {}
    # exact
    byh = collections.defaultdict(list)
    for i, h in enumerate(H):
        byh[h["sha256"]].append(i)
    groups = [g for g in byh.values() if len(g) > 1]
    R["exact_sha256"] = {"groups": len(groups), "images_in_groups": sum(len(g) for g in groups),
                         "groups_spanning_splits": sum(1 for g in groups if len({H[i]["split"] for i in g}) > 1),
                         "examples": [[names[i] for i in g] for g in groups[:10]]}
    for kind in ("dhash", "phash"):
        arr = np.array([int(h[kind], 16) for h in H], dtype=np.uint64)
        R[kind] = {}
        for T in (4, 8, 12):
            lab = np.array(components(arr, T))
            cl = collections.defaultdict(list)
            for i, l in enumerate(lab):
                cl[l].append(i)
            multi = {l: m for l, m in cl.items() if len(m) > 1}
            within = {s: {"clusters": 0, "images": 0} for s in ("train", "valid", "test")}
            cross = {"clusters": 0, "images": 0, "valid_images_in_cross": 0, "test_images_in_cross": 0, "train_images_in_cross": 0, "pattern": collections.Counter()}
            biggest = max((len(m) for m in cl.values()), default=1)
            for l, m in multi.items():
                sp = {split[i] for i in m}
                if len(sp) == 1:
                    s = next(iter(sp))
                    within[s]["clusters"] += 1
                    within[s]["images"] += len(m)
                else:
                    cross["clusters"] += 1
                    cross["images"] += len(m)
                    cross["pattern"]["+".join(sorted(sp))] += 1
                    for i in m:
                        cross[f"{split[i]}_images_in_cross"] += 1
            cross["pattern"] = dict(cross["pattern"])
            # clusters that are mixed vs total multi
            R[kind][f"T{T}"] = {"n_clusters_total": len(cl), "n_clusters_size_ge2": len(multi), "images_in_size_ge2": sum(len(m) for m in multi.values()),
                                "largest_cluster": biggest, "within_split": within, "cross_split": cross}
            # a valid/test image counts as leaked if its cluster has a train image
            lk = {"valid": 0, "test": 0}
            for l, m in multi.items():
                if any(split[i] == "train" for i in m):
                    for i in m:
                        if split[i] in lk:
                            lk[split[i]] += 1
            R[kind][f"T{T}"]["valid_test_images_sharing_cluster_with_train"] = lk
        # nearest-train distance for valid/test images
        tr = np.where(split == "train")[0]
        nn = {}
        for s in ("valid", "test"):
            idx = np.where(split == s)[0]
            d = dist_matrix(arr[idx], arr[tr])
            m = d.min(axis=1)
            nn[s] = {"min_dist_to_train_hist": {str(k): int((m == k).sum()) for k in range(0, 33) if (m == k).any()}, "median": float(np.median(m)),
                     "share_le_8": float((m <= 8).mean()), "share_le_12": float((m <= 12).mean()), "share_le_16": float((m <= 16).mean())}
        R[kind]["nearest_train_distance"] = nn
        # baseline: nearest-neighbour distance train->train (excluding self) for context
        d = np.concatenate([dist_matrix(arr[tr[s:s + 500]], arr[tr]) for s in range(0, len(tr), 500)])
        np.fill_diagonal(d, 99)
        m = d.min(axis=1)
        R[kind]["nearest_train_to_train"] = {"median": float(np.median(m)), "share_le_8": float((m <= 8).mean()), "share_le_12": float((m <= 12).mean())}
    # nearest-neighbour pHash pairs for 150 valid+test queries
    ph = np.array([int(h["phash"], 16) for h in H], dtype=np.uint64)
    tr = np.where(split == "train")[0]
    vt = np.where(split != "train")[0]
    rnd = random.Random(0)
    q = sorted(rnd.sample(list(vt), 150))
    D = dist_matrix(ph[q], ph[tr])
    items = []
    for k, qi in enumerate(q):
        o = np.argsort(D[k], kind="stable")[:3]
        items.append({"query": names[qi], "neighbours": [{"name": names[tr[j]], "phash_dist": int(D[k, j]),
                      "dhash_dist": int(bin(int(H[qi]["dhash"], 16) ^ int(H[tr[j]]["dhash"], 16)).count("1"))} for j in o]})
    json.dump(items, open(f"{out}/leakage_pairs_index.json", "w"), indent=1)
    PER = 10
    TW, TH = 400, 225
    for s in range(0, 150, PER):
        sheet = Image.new("RGB", (4 * TW, PER * TH), (20, 20, 20))
        for r, it in enumerate(items[s:s + PER]):
            sp, nm = it["query"].split("/")
            sheet.paste(thumb(f"{ext}/HRP4K/{sp}/images/{nm}.jpg", TW, TH, f'Q {it["query"]}'), (0, r * TH))
            for c, nb in enumerate(it["neighbours"]):
                sp2, nm2 = nb["name"].split("/")
                sheet.paste(thumb(f"{ext}/HRP4K/{sp2}/images/{nm2}.jpg", TW, TH, f'{nb["name"]} p{nb["phash_dist"]} d{nb["dhash_dist"]}'), ((c + 1) * TW, r * TH))
        sheet.save(f"{out}/leakage_pairs_{s // PER:02d}.jpg", quality=85)
    # cross-dataset overlap
    ds = {"rdd2022_pothole": json.load(open(rddf)), "kaggle_pcm": json.load(open(kagf))}
    dh = np.array([int(h["dhash"], 16) for h in H], dtype=np.uint64)
    X = {}
    for dn, L in ds.items():
        L = [r for r in L if r["ok"]]
        lsha = collections.defaultdict(list)
        for r in L:
            lsha[r["sha256"]].append(r["path"])
        exact = [(names[i], lsha[H[i]["sha256"]][:2]) for i in range(N) if H[i]["sha256"] in lsha]
        od = np.array([int(r["dhash"], 16) for r in L], dtype=np.uint64)
        op = np.array([int(r["phash"], 16) for r in L], dtype=np.uint64)
        res = {"n_other": len(L), "exact_sha256_matches": len(exact), "exact_examples": exact[:5]}
        for T in (4, 8, 12):
            dd = np.concatenate([dist_matrix(dh[s:s + 500], od) for s in range(0, N, 500)])
            pp = np.concatenate([dist_matrix(ph[s:s + 500], op) for s in range(0, N, 500)])
            res[f"dhash_le_{T}_pairs"] = int((dd <= T).sum())
            res[f"hrp_images_with_dhash_match_le_{T}"] = int((dd <= T).any(axis=1).sum())
            res[f"phash_le_{T}_pairs"] = int((pp <= T).sum())
            res[f"both_dhash_and_phash_le_{T}_pairs"] = int(((dd <= T) & (pp <= T)).sum())
        res["min_dhash_dist"] = int(dd.min())
        res["min_phash_dist"] = int(pp.min())
        # China_MotorBike specifically
        if dn == "rdd2022_pothole":
            cm = [i for i, r in enumerate(L) if "China_MotorBike" in r["path"]]
            res["china_motorbike_n"] = len(cm)
            res["china_motorbike_min_dhash_dist"] = int(dd[:, cm].min())
            res["china_motorbike_min_phash_dist"] = int(pp[:, cm].min())
            res["china_motorbike_pairs_dhash_le_12"] = int((dd[:, cm] <= 12).sum())
            res["china_motorbike_resolutions"] = dict(collections.Counter(f'{L[i]["w"]}x{L[i]["h"]}' for i in cm))
        res["other_resolutions_top"] = dict(collections.Counter(f'{r["w"]}x{r["h"]}' for r in L).most_common(5))
        close = np.argwhere((dd <= 8) & (pp <= 8))
        res["closest_pairs_examples"] = [[names[i], L[j]["path"].split("data/")[-1], int(dd[i, j]), int(pp[i, j])] for i, j in close[:10]]
        X[dn] = res
    R["cross_dataset"] = X
    json.dump(R, open(f"{out}/full_dupes.json", "w"), indent=1)
    print(json.dumps(R, indent=1)[:6000])


if __name__ == "__main__":
    main()
