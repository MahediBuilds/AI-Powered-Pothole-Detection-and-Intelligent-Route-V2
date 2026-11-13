"""Render the closest HRP4K <-> other-dataset pairs (by pHash, tie-break dHash) side by side.
usage: hrp4k_full_crosssheet.py EXTRACTED_DIR OUT_DIR RDD_HASH.json KAGGLE_HASH.json HRP_HASHES.json"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from hrp4k_full_dupes import dist_matrix


def main():
    ext, out, rddf, kagf, hf = sys.argv[1:6]
    H = json.load(open(hf))
    ph = np.array([int(h["phash"], 16) for h in H], dtype=np.uint64)
    dh = np.array([int(h["dhash"], 16) for h in H], dtype=np.uint64)
    for dn, f, sub in (("rdd2022", rddf, None), ("rdd2022_chinamotorbike", rddf, "China_MotorBike"), ("kaggle_pcm", kagf, None)):
        L = [r for r in json.load(open(f)) if r["ok"] and (sub is None or sub in r["path"])]
        op = np.array([int(r["phash"], 16) for r in L], dtype=np.uint64)
        od = np.array([int(r["dhash"], 16) for r in L], dtype=np.uint64)
        P = np.concatenate([dist_matrix(ph[s:s + 500], op) for s in range(0, len(H), 500)])
        D = np.concatenate([dist_matrix(dh[s:s + 500], od) for s in range(0, len(H), 500)])
        score = P.astype(np.int32) * 100 + D
        flat = np.argsort(score, axis=None)[:12]
        rows = [np.unravel_index(k, score.shape) for k in flat]
        TW, TH = 480, 270
        sheet = Image.new("RGB", (2 * TW, 12 * TH), (20, 20, 20))
        info = []
        for r, (i, j) in enumerate(rows):
            a = Image.open(f'{ext}/HRP4K/{H[i]["split"]}/images/{H[i]["name"]}.jpg')
            a.draft("RGB", (TW * 2, TH * 2))
            a = a.convert("RGB").resize((TW, TH))
            b = Image.open(L[j]["path"]).convert("RGB").resize((TW, TH))
            for im, lab in ((a, f'HRP {H[i]["split"]}/{H[i]["name"]}'), (b, f'{L[j]["path"].replace(chr(92), "/").split("/")[-1]} p{P[i, j]} d{D[i, j]}')):
                ImageDraw.Draw(im).text((3, 3), lab, fill=(255, 255, 0))
            sheet.paste(a, (0, r * TH))
            sheet.paste(b, (TW, r * TH))
            info.append([f'{H[i]["split"]}/{H[i]["name"]}', L[j]["path"].replace("\\", "/").split("/")[-1], int(P[i, j]), int(D[i, j])])
        sheet.save(f"{out}/crossdataset_closest_{dn}.jpg", quality=85)
        print(dn, info)


if __name__ == "__main__":
    main()
