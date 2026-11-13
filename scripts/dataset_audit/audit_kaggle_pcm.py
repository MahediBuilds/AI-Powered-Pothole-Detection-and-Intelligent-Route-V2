"""Audit of Kaggle road-damage (potholes/cracks/manholes) dataset. Run with: python -I audit_kaggle_pcm.py <data_dir> <out_json>
Read-only on data_dir. Does not execute anything shipped with the dataset."""
import sys, os, json, re, hashlib, hashlib, collections, datetime
import numpy as np
from PIL import Image

data = sys.argv[1]; out = sys.argv[2]
coco = json.load(open(os.path.join(data, 'annotations_coco.json')))
cats = {c['id']: c['name'] for c in coco['categories']}
imgs = {i['id']: i for i in coco['images']}
files = sorted(os.listdir(os.path.join(data, 'images')))
R = {}
R['n_images_dir'] = len(files); R['n_images_coco'] = len(imgs); R['n_annotations'] = len(coco['annotations'])
R['categories'] = cats
R['resolution_coco'] = collections.Counter(f"{i['width']}x{i['height']}" for i in imgs.values())
# actual resolutions
actual = collections.Counter(); mism = 0
for i in imgs.values():
    p = os.path.join(data, 'images', i['file_name'])
    if os.path.exists(p):
        with Image.open(p) as im:
            actual[f"{im.size[0]}x{im.size[1]}"] += 1
            if im.size != (i['width'], i['height']): mism += 1
R['resolution_actual'] = actual; R['coco_size_mismatch'] = mism
anns = coco['annotations']
R['annotation_keys'] = sorted({k for a in anns for k in a})
R['segmentation_present'] = sum(1 for a in anns if a.get('segmentation'))
seg_len = collections.Counter(len(a['segmentation'][0]) if a.get('segmentation') else 0 for a in anns)
R['segmentation_len_counter'] = seg_len
# is bbox derived from polygon?
derived = 0; nd = 0
for a in anns:
    s = a.get('segmentation')
    if s:
        xs = s[0][0::2]; ys = s[0][1::2]
        bb = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        if np.allclose(bb, a['bbox'], atol=1.0): derived += 1
        else: nd += 1
R['bbox_equals_polygon_envelope'] = derived; R['bbox_differs_from_polygon_envelope'] = nd
# polygons axis-aligned rectangles?
rect = 0
for a in anns:
    s = a.get('segmentation')
    if s and len(s[0]) == 8:
        xs = s[0][0::2]; ys = s[0][1::2]
        if len(set(np.round(xs, 1))) <= 2 and len(set(np.round(ys, 1))) <= 2: rect += 1
R['polygons_that_are_axis_aligned_rectangles'] = rect
# per class
per = {}
perimg = collections.defaultdict(set)
for a in anns: perimg[a['category_id']].add(a['image_id'])
def stats(v):
    v = np.array(v); return {'min': float(v.min()), 'p5': float(np.percentile(v, 5)), 'median': float(np.median(v)), 'mean': float(v.mean()), 'p95': float(np.percentile(v, 95)), 'max': float(v.max())} if len(v) else {}
for cid, name in cats.items():
    aa = [a for a in anns if a['category_id'] == cid]
    W = [a['bbox'][2] for a in aa]; H = [a['bbox'][3] for a in aa]
    fa = [a['bbox'][2] * a['bbox'][3] / (imgs[a['image_id']]['width'] * imgs[a['image_id']]['height']) for a in aa if a['image_id'] in imgs]
    fw = [a['bbox'][2] / imgs[a['image_id']]['width'] for a in aa if a['image_id'] in imgs]
    sqrt_area = [np.sqrt(a['bbox'][2] * a['bbox'][3]) for a in aa]
    per[name] = {'instances': len(aa), 'images': len(perimg[cid]), 'w_px': stats(W), 'h_px': stats(H), 'sqrt_area_px': stats(sqrt_area), 'area_frac': stats(fa), 'w_frac': stats(fw),
                 'n_lt_16px_side': sum(1 for w, h in zip(W, H) if min(w, h) < 16), 'n_lt_32sqrt': sum(1 for s in sqrt_area if s < 32)}
R['per_class'] = per
annimg = collections.Counter(a['image_id'] for a in anns)
R['images_with_no_annotations_coco'] = sum(1 for i in imgs if annimg[i] == 0)
R['annotations_per_image'] = stats(list(annimg[i] for i in imgs))
# validity
oob = zero = dup = missing_img = 0; seen = set()
for a in anns:
    if a['image_id'] not in imgs: missing_img += 1; continue
    im = imgs[a['image_id']]; x, y, w, h = a['bbox']
    if x < -0.5 or y < -0.5 or x + w > im['width'] + 0.5 or y + h > im['height'] + 0.5: oob += 1
    if w <= 0 or h <= 0: zero += 1
    k = (a['image_id'], a['category_id'], tuple(round(v, 1) for v in a['bbox']))
    if k in seen: dup += 1
    seen.add(k)
R['validity'] = {'out_of_bounds': oob, 'zero_or_negative_area': zero, 'duplicate_boxes_same_class': dup, 'ann_missing_image_id': missing_img,
                 'coco_images_missing_on_disk': sum(1 for i in imgs.values() if i['file_name'] not in set(files)),
                 'disk_images_not_in_coco': len(set(files) - {i['file_name'] for i in imgs.values()})}
# YOLO / polygon label files
def read_lbl(d):
    res = {}
    for f in os.listdir(os.path.join(data, d)):
        rows = [l.split() for l in open(os.path.join(data, d, f)).read().splitlines() if l.strip()]
        res[f[:-4]] = rows
    return res
Y = read_lbl('labels-YOLO'); P = read_lbl('labels')
R['yolo_label_files'] = len(Y); R['polygon_label_files'] = len(P)
R['yolo_empty_files'] = sum(1 for v in Y.values() if not v); R['polygon_empty_files'] = sum(1 for v in P.values() if not v)
stems = {f[:-4] for f in files}
R['images_without_yolo_file'] = len(stems - set(Y)); R['yolo_files_without_image'] = len(set(Y) - stems)
R['yolo_instances'] = collections.Counter(int(r[0]) for v in Y.values() for r in v)
R['polygon_instances'] = collections.Counter(int(r[0]) for v in P.values() for r in v)
R['yolo_rows_bad_len'] = sum(1 for v in Y.values() for r in v if len(r) != 5)
R['yolo_oob'] = sum(1 for v in Y.values() for r in v if len(r) == 5 and (float(r[1]) - float(r[3]) / 2 < -0.002 or float(r[1]) + float(r[3]) / 2 > 1.002 or float(r[2]) - float(r[4]) / 2 < -0.002 or float(r[2]) + float(r[4]) / 2 > 1.002))
R['yolo_zero_area'] = sum(1 for v in Y.values() for r in v if len(r) == 5 and (float(r[3]) <= 0 or float(r[4]) <= 0))
R['yolo_images_with_class'] = {c: sum(1 for v in Y.values() if any(int(r[0]) == c for r in v)) for c in (0, 1, 2)}
R['yolo_images_pothole_and_manhole'] = sum(1 for v in Y.values() if {0, 2} <= {int(r[0]) for r in v})
# polygon shapes: axis aligned?
nonrect = 0
for v in P.values():
    for r in v:
        c = list(map(float, r[1:])); xs = c[0::2]; ys = c[1::2]
        if len(set(np.round(xs, 3))) > 2 or len(set(np.round(ys, 3))) > 2: nonrect += 1
R['polygon_label_rows_not_axis_aligned'] = nonrect
# duplicates
sha = collections.defaultdict(list); hashes = {}
def dhash(im, n=8):
    g = np.asarray(im.convert('L').resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()
for f in files:
    p = os.path.join(data, 'images', f)
    sha[hashlib.sha256(open(p, 'rb').read()).hexdigest()].append(f)
    with Image.open(p) as im: hashes[f] = dhash(im, 8)
R['exact_duplicate_groups'] = [v for v in sha.values() if len(v) > 1]
H = np.array([hashes[f] for f in files]); N = len(files)
ham = (H[:, None, :] != H[None, :, :]).sum(-1)
iu = np.triu_indices(N, 1)
pairs = [(files[i], files[j], int(ham[i, j])) for i, j in zip(*iu) if ham[i, j] <= 6]
R['near_dup_pairs_dhash64_ham_le6'] = len(pairs)
# union-find clusters
par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
idx = {f: k for k, f in enumerate(files)}
for a, b, _ in pairs: par[fd(idx[a])] = fd(idx[b])
cl = collections.Counter(fd(i) for i in range(N))
R['near_dup_clusters'] = sum(1 for v in cl.values() if v > 1); R['images_in_near_dup_clusters'] = sum(v for v in cl.values() if v > 1)
R['largest_near_dup_cluster'] = max(cl.values())
R['near_dup_pair_examples'] = pairs[:20]
for t in (3, 10):
    R[f'near_dup_pairs_dhash_ham_le{t}'] = int(((ham <= t)[iu]).sum())
# cross-check: near dup pairs with different split? (n/a no split)
# timestamps
pat1 = re.compile(r'^(\d{8})_(\d{6})\.jpg$'); pat2 = re.compile(r'^vlcsnap[-_](\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})\.jpg$')
ts = []; nots = []
for f in files:
    m = pat1.match(f)
    if m: ts.append((datetime.datetime.strptime(m[1] + m[2], '%Y%m%d%H%M%S'), f, 'phone_name')); continue
    m = pat2.match(f)
    if m:
        a = list(map(int, m.groups())); ts.append((datetime.datetime(a[0], a[1], a[2], a[3], a[4], a[5], a[6] * 1000), f, 'vlcsnap')); continue
    nots.append(f)
ts.sort()
R['files_with_timestamp'] = len(ts); R['files_without_timestamp'] = len(nots)
R['files_without_timestamp_pattern'] = collections.Counter(re.sub(r'\d', '#', f) for f in nots)
R['timestamp_kinds'] = collections.Counter(k for _, _, k in ts)
R['dates'] = collections.Counter(t.strftime('%Y-%m-%d') for t, _, _ in ts)
for gap in (2, 5, 10, 30, 60):
    groups = 0; sizes = []; cur = 0; prev = None
    for t, f, k in ts:
        if prev is None or (t - prev).total_seconds() > gap:
            if cur: sizes.append(cur)
            cur = 0
        cur += 1; prev = t
    sizes.append(cur)
    R[f'sequences_gap_{gap}s'] = {'n_groups': len(sizes), 'size_median': float(np.median(sizes)), 'size_max': max(sizes), 'singletons': sum(1 for s in sizes if s == 1)}
# numbered vlcsnap: consecutive numbers
nums = sorted(int(re.search(r'(\d+)', f)[1]) for f in nots)
R['numbered_vlcsnap_range'] = [nums[0], nums[-1]] if nums else None
# class mix at pothole-only images etc.
R['images_pothole_only'] = sum(1 for v in Y.values() if v and {int(r[0]) for r in v} == {0})
R['images_manhole_only'] = sum(1 for v in Y.values() if v and {int(r[0]) for r in v} == {2})
R['images_crack_only'] = sum(1 for v in Y.values() if v and {int(r[0]) for r in v} == {1})
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    raise TypeError(type(o))
json.dump(R, open(out, 'w'), indent=1, default=conv)
print(json.dumps({k: v for k, v in R.items() if k not in ('near_dup_pair_examples',)}, indent=1, default=conv))
