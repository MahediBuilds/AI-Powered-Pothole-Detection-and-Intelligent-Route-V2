"""Audit HRP4K annotations (already fetched). Usage: python -I audit_hrp4k.py <ann_root(HRP4K dir)> <central_directory.json> <out_json>"""
import sys, os, json, collections, re
import numpy as np
root, cdj, out = sys.argv[1:4]
cd = json.load(open(cdj))['entries']
R = {'splits': {}}
def stats(v):
    v = np.array(v, float)
    return {} if len(v) == 0 else {'min': v.min(), 'p5': np.percentile(v, 5), 'median': np.median(v), 'mean': v.mean(), 'p95': np.percentile(v, 95), 'max': v.max()}
allnums = {}
for s in ['train', 'valid', 'test']:
    J = json.load(open(os.path.join(root, f'{s}.json')))
    r = {'json_keys': list(J.keys()), 'categories': J.get('categories'), 'n_images_json': len(J['images']), 'n_anns_json': len(J['annotations'])}
    imgs = {i['id']: i for i in J['images']}
    r['resolution_json'] = collections.Counter(f"{i['width']}x{i['height']}" for i in J['images'])
    r['ann_keys'] = sorted({k for a in J['annotations'] for k in a})
    r['segmentation_nonempty'] = sum(1 for a in J['annotations'] if a.get('segmentation'))
    imgs_zip = sorted(e['name'].split('/')[-1][:-4] for e in cd if e['name'].startswith(f'HRP4K/{s}/images/') and e['name'].endswith('.jpg'))
    lab_zip = sorted(e['name'].split('/')[-1][:-4] for e in cd if e['name'].startswith(f'HRP4K/{s}/labels/') and e['name'].endswith('.txt'))
    r['n_images_zip'] = len(imgs_zip); r['n_labels_zip'] = len(lab_zip); r['image_label_stem_mismatch'] = len(set(imgs_zip) ^ set(lab_zip))
    nums = sorted(int(x) for x in imgs_zip); allnums[s] = nums
    r['filename_number_range'] = [nums[0], nums[-1]]
    r['json_images_not_in_zip'] = len({i['file_name'][:-4] for i in J['images']} - set(imgs_zip))
    r['zip_images_not_in_json'] = len(set(imgs_zip) - {i['file_name'][:-4] for i in J['images']})
    # txt labels
    L = {}
    for st in lab_zip:
        L[st] = [l.split() for l in open(os.path.join(root, s, 'labels', st + '.txt')).read().splitlines() if l.strip()]
    r['txt_empty_files'] = sum(1 for v in L.values() if not v)
    r['txt_instances'] = sum(len(v) for v in L.values())
    r['txt_class_ids'] = collections.Counter(int(x[0]) for v in L.values() for x in v)
    r['txt_row_len'] = collections.Counter(len(x) for v in L.values() for x in v)
    bx = [list(map(float, x[1:5])) for v in L.values() for x in v if len(x) == 5]
    w = np.array([b[2] for b in bx]); h = np.array([b[3] for b in bx])
    W, H = 3840, 2160
    r['box_w_frac'] = stats(w); r['box_h_frac'] = stats(h); r['box_area_frac'] = stats(w * h)
    r['box_w_px_at_native'] = stats(w * W); r['box_h_px_at_native'] = stats(h * H); r['box_sqrt_area_px_native'] = stats(np.sqrt(w * h * W * H))
    r['box_sqrt_area_px_at_640w'] = stats(np.sqrt(w * h) * 640 * np.sqrt(2160 / 3840) ** 0 * np.sqrt(1) if False else np.sqrt(w * h * 640 * 360))
    r['oob'] = sum(1 for b in bx if b[0] - b[2] / 2 < -0.001 or b[0] + b[2] / 2 > 1.001 or b[1] - b[3] / 2 < -0.001 or b[1] + b[3] / 2 > 1.001)
    r['zero_area'] = sum(1 for b in bx if b[2] <= 0 or b[3] <= 0)
    dup = 0
    for v in L.values():
        seen = set()
        for x in v:
            k = tuple(round(float(t), 4) for t in x)
            if k in seen: dup += 1
            seen.add(k)
    r['duplicate_rows'] = dup
    r['instances_per_image_nonempty'] = stats([len(v) for v in L.values() if v])
    r['images_with_n_boxes'] = collections.Counter(min(len(v), 10) for v in L.values())
    # json vs txt consistency
    ann_by_img = collections.Counter(a['image_id'] for a in J['annotations'])
    name2id = {i['file_name'][:-4]: i['id'] for i in J['images']}
    r['json_images_with_no_ann'] = sum(1 for i in imgs if ann_by_img[i] == 0)
    mism = 0; jx = []
    for a in J['annotations']:
        jx.append(a['bbox'])
    cats = collections.Counter(a['category_id'] for a in J['annotations']); r['json_cat_counts'] = cats
    # compare counts per image json vs txt
    diffc = 0
    for st, v in L.items():
        iid = name2id.get(st)
        if iid is None: continue
        if ann_by_img[iid] != len(v): diffc += 1
    r['images_where_json_count_ne_txt_count'] = diffc
    # json bbox validity (pixel)
    oobj = zeroj = 0
    for a in J['annotations']:
        im = imgs.get(a['image_id'])
        if im is None: continue
        x, y, bw, bh = a['bbox']
        if x < -0.5 or y < -0.5 or x + bw > im['width'] + 0.5 or y + bh > im['height'] + 0.5: oobj += 1
        if bw <= 0 or bh <= 0: zeroj += 1
    r['json_oob'] = oobj; r['json_zero_area'] = zeroj
    r['images_missing_label_in_json_but_has_txt_boxes'] = sum(1 for st, v in L.items() if v and st not in name2id)
    r['empty_txt_in_json'] = sum(1 for st, v in L.items() if not v and st in name2id)
    r['nonempty_txt_n'] = sum(1 for v in L.values() if v)
    R['splits'][s] = r
# split interleaving / gap analysis
for s, n in allnums.items():
    gaps = np.diff(n)
    R['splits'][s]['number_gap_stats'] = {'n_consecutive_gap1': int((gaps == 1).sum()), 'median_gap': float(np.median(gaps)), 'max_gap': int(gaps.max())}
R['all_numbers_contiguous_ranges'] = {s: [n[0], n[-1], len(n)] for s, n in allnums.items()}
allset = set(); [allset.update(n) for n in allnums.values()]
R['union_numbers'] = [min(allset), max(allset), len(allset)]
# which numbers in 0..max are missing
R['missing_numbers_train_range_count'] = (allnums['train'][-1] - allnums['train'][0] + 1) - len(allnums['train'])
# adjacency across splits: nearest cross-split neighbour distance
tr = np.array(allnums['train']);
for s in ('valid', 'test'):
    n = np.array(allnums[s]); d = np.abs(n[:, None] - tr[None, :]).min(1)
    R['splits'][s]['dist_to_nearest_train_number'] = {'median': float(np.median(d)), 'min': int(d.min()), 'n_adjacent_le2': int((d <= 2).sum())}
R['n_images_total'] = sum(len(n) for n in allnums.values())
json.dump(R, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
