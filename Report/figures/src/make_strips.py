"""Re-composes the hand-picked three-model review strips so the panel labels are legible at text width.
Each source strip (2880x642) holds a black header and three 960 px panels: earlier implementation,
baseline, candidate. The header is removed, the panels are kept unchanged and a larger label bar is added.
The original frames and boxes are not altered. Inputs are read-only; outputs go to the report figures folder."""
import os
from PIL import Image, ImageDraw, ImageFont

SRC = r"<PROJECT_ROOT>/Three_Model_Comparison\review_sheets\interesting_cases" + "\\"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 46)
HEADER_PX = 103   # black header height in the source strips (measured on the strips)
LABELS = ["Earlier implementation", "Baseline", "Candidate"]
COLS = [(168, 60, 168), (30, 130, 30), (200, 100, 0)]   # label text colours (dark enough on white)

def strip(fname):
    im = Image.open(SRC + fname).convert("RGB")
    assert im.size == (2880, 642), im.size
    panels = [im.crop((i * 960, HEADER_PX, (i + 1) * 960, 642)) for i in range(3)]
    h = panels[0].height
    bar = 64
    out = Image.new("RGB", (2880 + 2 * 12, bar + h), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for i, p in enumerate(panels):
        x = i * (960 + 12)
        out.paste(p, (x, bar))
        d.text((x + 10, 8), LABELS[i], fill=COLS[i], font=FONT)
    return out

def stack(files, name):
    rows = [strip(f) for f in files]
    gap = 24
    W = rows[0].width
    H = sum(r.height for r in rows) + gap * (len(rows) - 1)
    c = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        c.paste(r, (0, y)); y += r.height + gap
    c = c.resize((2000, round(H * 2000 / W)), Image.LANCZOS)
    c.save(os.path.join(OUT, name), quality=82, optimize=True)

stack(["C03_pothole_round_sunken_20261013_173922_f1053.jpg", "C04_pothole_round_sunken_20261013_173922_f1073.jpg"], "ch6_strips_round_pothole.jpg")
stack(["C05_manhole_cover_20261013_173205_f963.jpg", "C11_round_cover_20261013_174130_f3.jpg"], "ch6_strips_manhole_cover.jpg")
stack(["C17_old_whole_frame_box_20261013_180932_f363.jpg"], "ch6_strip_wholeframe.jpg")
stack(["C01_pothole_water_filled_20261022_165007_f356.jpg", "C14_puddle_or_wet_patch_20261022_164858_f465.jpg"], "appA_strips_water_wet.jpg")
stack(["C08_repaired_strip_20261022_164947_f281.jpg"], "appA_strip_repaired.jpg")
