"""Render per-box review sheets for HRP4K (read-only on dataset)."""
import csv, json, re
from multiprocessing import Pool
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data/external/hrp4k/extracted/HRP4K'
OUT = ROOT / 'docs/dataset_audit/hrp4k/box_review'
T, CAP, COLS, ROWS = 288, 22, 5, 4
PER = COLS * ROWS


def num(p):
    m = re.findall(r'\d+', p.stem)
    return int(m[-1]) if m else 0


def collect():
    rows = []  # split, image, idx, cx, cy, w, h (normalised)
    for split in ('train', 'valid', 'test'):
        for lp in sorted((D / split / 'labels').glob('*.txt'), key=lambda p: (num(p), p.name)):
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            img = next((lp.with_suffix(e).name for e in ('.jpg', '.jpeg', '.png')
                        if (D / split / 'images' / lp.with_suffix(e).name).exists()), None)
            for i, l in enumerate(lines):
                rows.append((split, img, i, *map(float, l[1:5])))
    return rows


def render_sheet(job):
    split, sheet_no, chunk = job  # chunk: list of (tid, row)
    font = ImageFont.load_default(size=15)
    sheet = Image.new('RGB', (COLS * T, ((len(chunk) + COLS - 1) // COLS) * (T + CAP)), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    cache = {}
    for k, (tid, r) in enumerate(chunk):
        _, img, _, cx, cy, w, h = r
        if img not in cache:
            cache = {img: Image.open(D / split / 'images' / img).convert('RGB')}
        im = cache[img]
        W, H = im.size
        bw, bh, bx, by = w * W, h * H, cx * W, cy * H
        side = max(3 * max(bw, bh), 320)
        x0, y0 = bx - side / 2, by - side / 2
        # crop() pads with black outside the image
        c = im.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((T, T), Image.LANCZOS)
        s = T / side
        d = ImageDraw.Draw(c)
        d.rectangle([(bx - bw / 2 - x0) * s, (by - bh / 2 - y0) * s, (bx + bw / 2 - x0) * s, (by + bh / 2 - y0) * s],
                    outline=(255, 0, 255), width=1)
        gx, gy = (k % COLS) * T, (k // COLS) * (T + CAP)
        sheet.paste(c, (gx, gy + CAP))
        dr.text((gx + 3, gy + 2), tid, fill=(255, 255, 0), font=font)
    name = f'{split}_{sheet_no:04d}.jpg'
    sheet.save(OUT / name, quality=85)
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    # image sizes
    sizes = {}
    idmap, jobs, sheets = [], [], {}
    n = 0
    for split in ('train', 'valid', 'test'):
        sr = [r for r in rows if r[0] == split]
        sheets[split] = []
        for s in range(0, len(sr), PER):
            chunk = []
            sheet_no = s // PER + 1
            name = f'{split}_{sheet_no:04d}.jpg'
            for k, r in enumerate(sr[s:s + PER]):
                n += 1
                tid = f'T{n:05d}'
                chunk.append((tid, r))
                key = (split, r[1])
                if key not in sizes:
                    sizes[key] = Image.open(D / split / 'images' / r[1]).size
                W, H = sizes[key]
                idmap.append([tid, name, k // COLS, k % COLS, split, r[1], r[2],
                              f'{r[3]:.6f} {r[4]:.6f} {r[5]:.6f} {r[6]:.6f}', round(r[5] * W, 1), round(r[6] * H, 1)])
            sheets[split].append({'sheet': name, 'first_id': chunk[0][0], 'last_id': chunk[-1][0], 'n': len(chunk)})
            jobs.append((split, sheet_no, chunk))
    with Pool(12) as p:
        p.map(render_sheet, jobs, chunksize=4)
    with open(OUT / 'id_map.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tile_id', 'sheet_file', 'row', 'col', 'split', 'image_file', 'box_index', 'box_norm_cx_cy_w_h',
                    'box_w_px', 'box_h_px'])
        w.writerows(idmap)
    json.dump({'totals': {'boxes': n, 'sheets': sum(len(v) for v in sheets.values()),
                          **{f'sheets_{k}': len(v) for k, v in sheets.items()}}, 'splits': sheets},
              open(OUT / 'sheets_index.json', 'w'), indent=1)
    # batches: contiguous over train, valid, test sheet order
    allsheets = [(s['sheet'], s['first_id'], s['last_id']) for sp in ('train', 'valid', 'test') for s in sheets[sp]]
    nb, tot = 4, len(allsheets)
    base, extra = divmod(tot, nb)
    batches, pos = {}, 0
    tiles_by_sheet = {}
    for row in idmap:
        tiles_by_sheet.setdefault(row[1], []).append(row[0])
    for i, X in enumerate('ABCD'):
        cnt = base + (1 if i < extra else 0)
        part = allsheets[pos:pos + cnt]; pos += cnt
        batches[X] = {'n_sheets': len(part), 'sheets': [p[0] for p in part]}
        tids = [t for p in part for t in tiles_by_sheet[p[0]]]
        batches[X]['n_tiles'] = len(tids)
        with open(OUT / f'verdicts_batch_{X}.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['tile_id', 'verdict', 'reason'])
            w.writerows([[t, '', ''] for t in tids])
    json.dump(batches, open(OUT / 'batches.json', 'w'), indent=1)
    print(n, {k: len(v) for k, v in sheets.items()}, {k: (v['n_sheets'], v['n_tiles']) for k, v in batches.items()})


if __name__ == '__main__':
    main()
