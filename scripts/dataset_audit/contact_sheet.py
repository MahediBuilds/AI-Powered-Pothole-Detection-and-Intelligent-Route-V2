"""Contact sheet builder. Usage: python -I contact_sheet.py <mode> ...
 kaggle <data_dir> <out_dir>   : random / manhole-only / pothole-only sheets (YOLO labels, 3 classes)
 hrp <items_json> <out_dir>    : items_json = [{"image": path, "boxes": [[cls,xc,yc,w,h],...]}]
Seeded; read-only on inputs."""
import sys, os, json, random
from PIL import Image, ImageDraw
COL = {0: (255, 40, 40), 1: (255, 200, 0), 2: (40, 160, 255)}
def sheet(items, out, cols=6, tw=320, label=True):
    rows = (len(items) + cols - 1) // cols
    th = tw * 9 // 16
    S = Image.new('RGB', (cols * tw, rows * th), (30, 30, 30))
    for k, (path, boxes, name) in enumerate(items):
        with Image.open(path) as im:
            im = im.convert('RGB'); W, H = im.size
            im.thumbnail((tw, th)); sx = im.size[0] / W; sy = im.size[1] / H
        d = ImageDraw.Draw(im)
        for c, xc, yc, w, h in boxes:
            x0 = (xc - w / 2) * W * sx; y0 = (yc - h / 2) * H * sy; x1 = (xc + w / 2) * W * sx; y1 = (yc + h / 2) * H * sy
            d.rectangle([x0, y0, x1, y1], outline=COL.get(int(c), (0, 255, 0)), width=2)
        if label: d.text((3, 3), name[-22:], fill=(255, 255, 255))
        S.paste(im, ((k % cols) * tw, (k // cols) * th))
    S.save(out, quality=88)
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'kaggle':
        data, outd = sys.argv[2], sys.argv[3]; os.makedirs(outd, exist_ok=True)
        random.seed(0)
        names = sorted(f[:-4] for f in os.listdir(os.path.join(data, 'labels-YOLO')))
        L = {n: [list(map(float, l.split())) for l in open(os.path.join(data, 'labels-YOLO', n + '.txt')) if l.strip()] for n in names}
        sets = {'random': names, 'manhole_only': [n for n in names if L[n] and {int(r[0]) for r in L[n]} == {2}], 'pothole_only': [n for n in names if L[n] and {int(r[0]) for r in L[n]} == {0}]}
        for k, pool in sets.items():
            for s in range(2 if k == 'random' else 1):
                pick = random.sample(pool, 24)
                sheet([(os.path.join(data, 'images', n + '.jpg'), L[n], n) for n in pick], os.path.join(outd, f'{k}_{s}.jpg'))
                print(k, s, pick)
    elif mode == 'hrp':
        items = json.load(open(sys.argv[2])); outd = sys.argv[3]; os.makedirs(outd, exist_ok=True)
        per = 12
        for s in range(0, len(items), per):
            sheet([(i['image'], i['boxes'], os.path.basename(i['image'])) for i in items[s:s + per]], os.path.join(outd, f'sheet_{s // per}.jpg'), cols=3, tw=640)
