"""HRP4K full audit stage 3: review sheets for 120 random empty-label and 120 random positive images (seed 0).
Each tile = lower half of the image (y in [0.5H, H]) at 1280 px wide (1280x360), 6 tiles stacked per sheet (1280x2160).
Positives get their boxes drawn (padded outline so the box location is visible without hiding the object).
usage: hrp4k_full_sheets.py EXTRACTED_DIR OUT_DIR [crop_top_fraction=0.5]"""
import sys, os, json, random
from PIL import Image, ImageDraw


def labels(p):
    return [list(map(float, l.split())) for l in open(p).read().splitlines() if l.strip()]


def tile(ext, split, name, draw_boxes, top):
    with Image.open(f"{ext}/HRP4K/{split}/images/{name}.jpg") as im:
        im = im.convert("RGB")
        W, H = im.size
    k = 1280 / W
    th = int(H * (1 - top) * k)
    t = im.resize((1280, int(H * k)), Image.LANCZOS).crop((0, int(H * k * top), 1280, int(H * k * top) + th))
    d = ImageDraw.Draw(t)
    n = 0
    if draw_boxes:
        for _, xc, yc, w, h in labels(f"{ext}/HRP4K/{split}/labels/{name}.txt"):
            x0, y0, x1, y1 = (xc - w / 2) * 1280, (yc - h / 2) * H * k - H * k * top, (xc + w / 2) * 1280, (yc + h / 2) * H * k - H * k * top
            d.rectangle([x0 - 5, y0 - 5, x1 + 5, y1 + 5], outline=(255, 0, 0), width=2)
            n += 1
    d.rectangle([0, 0, 330, 16], fill=(0, 0, 0))
    d.text((4, 2), f"{split}/{name}  boxes={n}" if draw_boxes else f"{split}/{name}", fill=(255, 255, 0))
    d.rectangle([0, th - 2, 1280, th], fill=(0, 200, 255))
    return t


def main():
    ext, out = sys.argv[1], sys.argv[2]
    top = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    rnd = random.Random(0)
    pools = {"empties": [], "positives": []}
    for s in ("train", "valid", "test"):
        for f in sorted(os.listdir(f"{ext}/HRP4K/{s}/labels")):
            pools["empties" if not labels(f"{ext}/HRP4K/{s}/labels/{f}") else "positives"].append((s, f[:-4]))
    index = {}
    for kind, pool in pools.items():
        pick = sorted(rnd.sample(pool, 120))
        rnd.shuffle(pick)
        for sidx in range(20):
            grp = pick[sidx * 6:(sidx + 1) * 6]
            tiles = [tile(ext, s, n, kind == "positives", top) for s, n in grp]
            sheet = Image.new("RGB", (1280, sum(t.height for t in tiles)))
            y = 0
            for t in tiles:
                sheet.paste(t, (0, y))
                y += t.height
            sheet.save(f"{out}/{kind}_{sidx:02d}.jpg", quality=88)
            index[f"{kind}_{sidx:02d}.jpg"] = [f"{s}/{n}" for s, n in grp]
    json.dump(index, open(f"{out}/review_sheet_index.json", "w"), indent=1)
    print("sheets written", len(index))


if __name__ == "__main__":
    main()
