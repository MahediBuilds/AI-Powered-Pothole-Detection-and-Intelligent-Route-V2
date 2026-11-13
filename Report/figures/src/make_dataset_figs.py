"""Builds the dataset and error-analysis figures for Chapter 6 from existing project files.
Inputs are read-only; outputs are written only into the report figures folder.
Run: python figures/src/make_dataset_figs.py   (from the report root)"""
import csv, os, random
from PIL import Image, ImageDraw, ImageFont

NI = r"<PROJECT_ROOT>/New Implementation"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FONT = r"C:\Windows\Fonts\arialbd.ttf"

def save_jpg(img, name, maxw):
    img = img.convert("RGB")
    if img.width > maxw:
        img = img.resize((maxw, round(img.height * maxw / img.width)), Image.LANCZOS)
    img.save(os.path.join(OUT, name), quality=86, optimize=True)

# HRP4K audit sheet (original images with the published boxes)
save_jpg(Image.open(NI + r"\docs\dataset_audit\hrp4k\sheet_0.jpg"), "ch6_hrp4k_examples.jpg", 1600)
# Giordani et al. dataset audit sheets
save_jpg(Image.open(NI + r"\docs\dataset_audit\kaggle_pcm\pothole_only_0.jpg"), "ch6_giordani_potholes.jpg", 1600)
save_jpg(Image.open(NI + r"\docs\dataset_audit\kaggle_pcm\manhole_only_0.jpg"), "ch6_giordani_manholes.jpg", 1600)

# False-positive category sheets (candidate, HRP4K validation images)
cs = NI + r"\runs\eval\val_investigations_v1\fp_categories\contact_sheets" + "\\"
save_jpg(Image.open(cs + "patch_or_repair_01.png"), "ch6_fp_patch_repair.jpg", 960)
a = Image.open(cs + "manhole_or_utility_cover_01.png").convert("RGB")
b = Image.open(cs + "plausible_unlabelled_pothole_01.png").convert("RGB")
gap = 16
canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height + gap), (255, 255, 255))
canvas.paste(a, (0, 0)); canvas.paste(b, (0, a.height + gap))
save_jpg(canvas, "ch6_fp_cover_unlabelled.jpg", 960)

# RDD2022 example sheet: positive TRAIN images of the five used subsets, boxes from the converted labels
random.seed(7)
root = NI + r"\data\processed\rdd2022_pothole"
rows = [r for r in csv.DictReader(open(root + r"\manifest.csv", newline=""))
        if r["status"] == "included" and r["split"] == "train" and r["is_positive"] == "1"]
tiles = []
for country in ["India", "Japan", "Czech", "United_States", "China_MotorBike"]:
    pool = [r for r in rows if r["country"] == country]
    pool.sort(key=lambda r: r["source_id"])
    tiles += random.sample(pool, 2 if country != "United_States" else 1) if country != "China_MotorBike" else random.sample(pool, 1)
tiles = tiles[:8]
T = 400
font = ImageFont.truetype(FONT, 22)
sheet = Image.new("RGB", (4 * T, 2 * T), (255, 255, 255))
for i, r in enumerate(tiles):
    im = Image.open(os.path.join(root, "images", "train", r["source_id"] + ".jpg")).convert("RGB")
    d = ImageDraw.Draw(im)
    for line in open(os.path.join(root, "labels", "train", r["source_id"] + ".txt")):
        p = line.split()
        if len(p) < 5: continue
        cx, cy, w, h = [float(x) for x in p[1:5]]
        W, H = im.size
        d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], outline=(255, 30, 30), width=max(2, W // 200))
    im = im.resize((T, T), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 190, 28], fill=(0, 0, 0))
    d.text((4, 2), r["country"].replace("_", " "), fill=(255, 255, 255), font=font)
    sheet.paste(im, ((i % 4) * T, (i // 4) * T))
save_jpg(sheet, "ch6_rdd2022_examples.jpg", 1600)
print("tiles:", [t["source_id"] for t in tiles])
