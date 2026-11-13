"""Analyse fetched HRP4K sample images. Usage: python -I hrp4k_samples.py <HRP4K_root_with_samples> <annotations_root> <out_sheet_dir> <out_json>"""
import sys, os, json, glob
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_sheet import sheet
Image.MAX_IMAGE_PIXELS = None
sroot, aroot, outd, outj = sys.argv[1:5]
os.makedirs(outd, exist_ok=True)
items = []; info = []; H = {}
def dh(im, n=8):
    g = np.asarray(im.convert('L').resize((n + 1, n), Image.LANCZOS), dtype=np.int16); return (g[:, 1:] > g[:, :-1]).flatten()
for p in sorted(glob.glob(os.path.join(sroot, '*', 'images', '*.jpg')), key=lambda p: (p.split(os.sep)[-3], int(os.path.basename(p)[:-4]))):
    s = p.split(os.sep)[-3]; n = os.path.basename(p)[:-4]
    boxes = [list(map(float, l.split())) for l in open(os.path.join(aroot, s, 'labels', n + '.txt')) if l.strip()]
    with Image.open(p) as im:
        info.append({'split': s, 'name': n, 'size': im.size, 'bytes': os.path.getsize(p), 'n_boxes': len(boxes), 'exif': bool(im.getexif())})
        H[(s, int(n))] = dh(im)
    items.append({'image': p, 'boxes': boxes})
adj = []
for (s, n), h in H.items():
    if (s, n + 1) in H: adj.append({'split': s, 'a': n, 'b': n + 1, 'dhash_ham': int((h != H[(s, n + 1)]).sum())})
rand_pairs = []
keys = list(H)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        if abs(keys[i][1] - keys[j][1]) > 50 or keys[i][0] != keys[j][0]:
            rand_pairs.append(int((H[keys[i]] != H[keys[j]]).sum()))
json.dump({'samples': info, 'adjacent_pairs': adj, 'non_adjacent_hamming_min_median': [min(rand_pairs), float(np.median(rand_pairs))]}, open(outj, 'w'), indent=1)
print(adj, min(rand_pairs), np.median(rand_pairs), {tuple(i['size']) for i in info})
for k in range(0, len(items), 12):
    sheet([(i['image'], i['boxes'], os.path.basename(i['image'])) for i in items[k:k + 12]], os.path.join(outd, f'sheet_{k // 12}.jpg'), cols=3, tw=640, label=True)
