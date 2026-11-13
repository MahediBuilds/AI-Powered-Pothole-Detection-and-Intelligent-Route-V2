"""Render per-annotation review sheets for Kaggle PCM (pothole + manhole boxes). Read-only on dataset."""
import csv, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data/external/kaggle_pcm_extracted/data'
OUT = ROOT / 'docs/dataset_audit/kaggle_pcm/review'
OUT.mkdir(parents=True, exist_ok=True)
coco = json.load(open(D / 'annotations_coco.json'))
imgs = {i['id']: i['file_name'] for i in coco['images']}
per = {}
for a in coco['annotations']:
    per.setdefault(a['image_id'], []).append(a)
rows = []  # (series, image, idx, ann_id, cls, x,y,w,h)
cname = {0: 'pothole', 1: 'crack', 2: 'manhole'}
for iid in sorted(per, key=lambda k: imgs[k]):
    for idx, a in enumerate(per[iid]):
        rows.append((imgs[iid], idx, a['id'], cname[a['category_id']], *a['bbox']))
T = 256; COLS = 6; PER = 30; CAP = 22
font = ImageFont.load_default(size=15)

def crop(fn, x, y, w, h):
    im = Image.open(D / 'images' / fn).convert('RGB')
    W, H = im.size
    cx, cy = x + w / 2, y + h / 2
    side = min(max(96, 5 * max(w, h)), min(W, H))
    x0 = min(max(cx - side / 2, 0), W - side); y0 = min(max(cy - side / 2, 0), H - side)
    c = im.crop((int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side)))).resize((T, T), Image.LANCZOS)
    s = T / side
    ImageDraw.Draw(c).rectangle([(x - x0) * s, (y - y0) * s, (x + w - x0) * s, (y + h - y0) * s], outline=(255, 0, 255), width=1)
    return c

mapping = []
for series, cls, pre in (('pothole', 'pothole', 'P'), ('manhole', 'manhole', 'M')):
    sel = [r for r in rows if r[3] == cls]
    for n, r in enumerate(sel, 1):
        mapping.append((f'{pre}{n:04d}', series, r[0], r[1], r[2], r[3], *[round(v, 2) for v in r[4:]]))
    for s in range(0, len(sel), PER):
        chunk = sel[s:s + PER]
        sheet = Image.new('RGB', (COLS * T, ((len(chunk) + COLS - 1) // COLS) * (T + CAP)), (30, 30, 30))
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            gx, gy = (k % COLS) * T, (k // COLS) * (T + CAP)
            sheet.paste(crop(r[0], *r[4:]), (gx, gy + CAP))
            dr.text((gx + 3, gy + 2), f'{pre}{s + k + 1:04d}  {r[4+2]:.0f}x{r[4+3]:.0f}', fill=(255, 255, 0), font=font)
        sheet.save(OUT / f'{series}_sheet_{s // PER + 1:02d}.jpg', quality=88)
with open(OUT / 'id_map.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'series', 'image', 'ann_index', 'coco_ann_id', 'class', 'x', 'y', 'w', 'h']); w.writerows(mapping)
print(len(mapping))
