"""Turn the recorded per-sheet verdict codes into annotation_review.csv and image_decisions.csv.

Inputs (all under docs/dataset_audit/kaggle_pcm/review/): id_map.csv, verdict_codes_pothole.txt,
verdict_codes_manhole.txt, optional image_flags.csv (image,flag,note) from the full-frame pass.
Policy is exactly the one in the task brief. Read-only on the dataset.
"""
import csv, json, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'docs/dataset_audit/kaggle_pcm/review'
D = ROOT / 'data/external/kaggle_pcm_extracted/data'
W, H = 640, 360
PC = {'o': ('ok', 'clearly a pothole/depression in paved road'),
      'p': ('reject', 'puddle / water on surface, no visible depression'),
      'a': ('reject', 'repair patch (not an open defect)'),
      'g': ('reject', 'unpaved/gravel/dirt track surface'),
      'c': ('reject', 'unpaved car park / dirt lot'),
      'm': ('reject', 'manhole/drain cover or its frame, not a pothole'),
      's': ('reject', 'road marking / tyre mark / shadow, not a defect'),
      'u': ('unclear', 'too small, blurred or low-contrast to judge')}
MC = {'i': ('intact', 'flush cover/grate, surrounding asphalt looks sound'),
      'd': ('damaged_or_sunken', 'surrounding asphalt visibly broken/cracked around the cover'),
      'u': ('unclear', 'too small, blurred or low-contrast to judge')}


def load_codes(fn):
    lines = [l for l in open(REV / fn).read().split() if l]
    lines.sort(key=lambda l: int(l.split(':')[0]))
    return ''.join(l.split(':')[1] for l in lines)


pc, mc = load_codes('verdict_codes_pothole.txt'), load_codes('verdict_codes_manhole.txt')
idmap = list(csv.DictReader(open(REV / 'id_map.csv')))
assert len(pc) == sum(r['series'] == 'pothole' for r in idmap)
assert len(mc) == sum(r['series'] == 'manhole' for r in idmap)
ann = []
pi = mi = 0
for r in idmap:
    if r['series'] == 'pothole':
        v, why = PC[pc[pi]]; pi += 1
    else:
        v, why = MC[mc[mi]]; mi += 1
    ann.append(dict(id=r['id'], image=r['image'], ann_index=int(r['ann_index']), cls=r['class'],
                    x=float(r['x']), y=float(r['y']), w=float(r['w']), h=float(r['h']), verdict=v, reason=why))
with open(REV / 'annotation_review.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image', 'ann_index', 'class', 'box_xywh', 'verdict', 'reason', 'tile_id'])
    for a in ann:
        w.writerow([a['image'], a['ann_index'], a['cls'], f"{a['x']},{a['y']},{a['w']},{a['h']}", a['verdict'], a['reason'], a['id']])

coco = json.load(open(D / 'annotations_coco.json'))
fname = {i['id']: i['file_name'] for i in coco['images']}
byimg = collections.defaultdict(list)
for a in coco['annotations']:
    byimg[a['image_id']].append(a)
verd = {(a['image'], a['ann_index']): a for a in ann}
flags = {}
if (REV / 'image_flags.csv').exists():
    for r in csv.DictReader(open(REV / 'image_flags.csv')):
        flags.setdefault(r['image'], []).append(r['flag'])


def clip_box(x, y, w, h):
    """Return (clipped box or None, max clip px)."""
    x0, y0, x1, y1 = x, y, x + w, y + h
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    clip = max(cx0 - x0, cy0 - y0, x1 - cx1, y1 - cy1, 0.0)
    if w <= 0.5 or h <= 0.5 or cx1 - cx0 <= 0.5 or cy1 - cy0 <= 0.5:
        return None, clip
    return (cx0, cy0, cx1 - cx0, cy1 - cy0), clip


rows = []; kept_boxes = []; clip_log = []
for iid in sorted(byimg, key=lambda k: fname[k]):
    fn = fname[iid]; anns = byimg[iid]
    reasons = []
    pot = [(k, a) for k, a in enumerate(anns) if a['category_id'] == 0]
    man = [(k, a) for k, a in enumerate(anns) if a['category_id'] == 2]
    mv = []
    for k, a in pot:
        v = verd[(fn, k)]
        if v['verdict'] == 'reject': reasons.append('pothole_reject')
        if v['verdict'] == 'unclear': reasons.append('pothole_unclear')
    for k, a in man:
        v = verd[(fn, k)]['verdict']; mv.append(v)
        if v == 'damaged_or_sunken': reasons.append('manhole_damaged_or_sunken')
        if v == 'unclear': reasons.append('manhole_unclear')
    fixed = []
    for k, a in pot:
        cb, clip = clip_box(*a['bbox'])
        if cb is None:
            reasons.append('pothole_box_invalid_unfixable')
        elif clip > 1e-3:
            if clip <= 2.0: clip_log.append((fn, k, clip, a['bbox'], cb))
            else: reasons.append('pothole_box_oob_gt2px')
        fixed.append(cb)
    for fl in flags.get(fn, []): reasons.append(fl)
    reasons = sorted(set(reasons))
    if reasons: dec = 'EXCLUDE'
    elif pot: dec = 'POSITIVE'
    else: dec = 'NEGATIVE'
    n_int = sum(v == 'intact' for v in mv)
    rows.append(dict(image=fn, decision=dec, n_pothole=len(pot), n_manhole=len(man), n_crack=len(anns) - len(pot) - len(man),
                     negative_has_intact_manhole=int(dec == 'NEGATIVE' and n_int > 0), reasons=';'.join(reasons)))
    if dec == 'POSITIVE':
        for (k, a), cb in zip(pot, fixed):
            kept_boxes.append((fn, k, cb))

with open(REV / 'image_decisions.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(REV / 'kept_pothole_boxes.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['image', 'ann_index', 'x', 'y', 'w', 'h'])
    for fn, k, cb in kept_boxes: w.writerow([fn, k] + [round(v, 2) for v in cb])
with open(REV / 'bbox_clipping_log.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['image', 'ann_index', 'max_clip_px', 'orig_xywh', 'clipped_xywh'])
    for r in clip_log: w.writerow([r[0], r[1], round(r[2], 3), r[3], r[4]])

S = {}
S['annotation_verdicts'] = {c: dict(collections.Counter(a['verdict'] for a in ann if a['cls'] == c)) for c in ('pothole', 'manhole')}
S['reject_reasons'] = dict(collections.Counter(a['reason'] for a in ann if a['verdict'] == 'reject'))
S['image_decisions'] = dict(collections.Counter(r['decision'] for r in rows))
ex = collections.Counter()
for r in rows:
    for x in r['reasons'].split(';'):
        if x: ex[x] += 1
S['exclusion_reason_counts_(images_can_have_several)'] = dict(ex)
S['kept_pothole_boxes'] = len(kept_boxes)
S['negatives_with_intact_manhole'] = sum(r['negative_has_intact_manhole'] for r in rows)
if kept_boxes:
    sq = np.array([np.sqrt(cb[2] * cb[3]) for _, _, cb in kept_boxes])
    S['kept_box_sqrt_area_px'] = {'n': len(sq), 'min': float(sq.min()), 'p25': float(np.percentile(sq, 25)), 'median': float(np.median(sq)),
                                  'p75': float(np.percentile(sq, 75)), 'max': float(sq.max()), 'lt16': int((sq < 16).sum()), 'lt32': int((sq < 32).sum()),
                                  'ge32_lt64': int(((sq >= 32) & (sq < 64)).sum()), 'ge64': int((sq >= 64).sum())}
S['clipped_boxes_logged'] = len(clip_log)
S['flags_used'] = dict(collections.Counter(f for v in flags.values() for f in v))
json.dump(S, open(REV / 'review_summary.json', 'w'), indent=1)
print(json.dumps(S, indent=1))
