"""Render full-frame review sheets (20 frames/sheet) for the images still POSITIVE/NEGATIVE after the per-box review."""
import csv, json, collections
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'docs/dataset_audit/kaggle_pcm/review'
D = ROOT / 'data/external/kaggle_pcm_extracted/data/images'
OUT = REV / 'frames'
OUT.mkdir(exist_ok=True)
rows = [r for r in csv.DictReader(open(REV / 'image_decisions.csv')) if r['decision'] in ('POSITIVE', 'NEGATIVE')]
rows.sort(key=lambda r: r['image'])
coco = json.load(open(ROOT / 'data/external/kaggle_pcm_extracted/data/annotations_coco.json'))
fid = {i['file_name']: i['id'] for i in coco['images']}
boxes = collections.defaultdict(list)
for a in coco['annotations']:
    if a['category_id'] in (0, 2): boxes[a['image_id']].append((a['category_id'], a['bbox']))
TW, TH, COLS, PER, CAP = 384, 216, 4, 20, 18
font = ImageFont.load_default(size=14)
m = []
for s in range(0, len(rows), PER):
    chunk = rows[s:s + PER]
    sheet = Image.new('RGB', (COLS * TW, ((len(chunk) + COLS - 1) // COLS) * (TH + CAP)), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    for k, r in enumerate(chunk):
        gx, gy = (k % COLS) * TW, (k // COLS) * (TH + CAP)
        tile = Image.open(D / r['image']).convert('RGB').resize((TW, TH), Image.LANCZOS)
        td = ImageDraw.Draw(tile)
        for c, (bx, by, bw, bh) in boxes[fid[r['image']]]:
            sc = TW / 640
            td.rectangle([bx * sc, by * sc, (bx + bw) * sc, (by + bh) * sc], outline=(255, 0, 255) if c == 0 else (0, 255, 255), width=1)
        sheet.paste(tile, (gx, gy + CAP))
        tid = f'F{s + k + 1:04d}'
        dr.text((gx + 3, gy + 1), f"{tid} {r['decision'][:3]} p{r['n_pothole']} m{r['n_manhole']} c{r['n_crack']}", fill=(255, 255, 0), font=font)
        m.append((tid, r['image'], r['decision']))
    sheet.save(OUT / f'frames_{s // PER + 1:02d}.jpg', quality=88)
with open(OUT / 'frame_id_map.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'image', 'decision_before_frame_pass']); w.writerows(m)
print(len(rows), (len(rows) + PER - 1) // PER)
