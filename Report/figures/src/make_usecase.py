"""Use-case diagram (matplotlib). Output: ../ch4_usecase.pdf and preview/ch4_usecase.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 10})
W, H = 5.77, 7.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

bx0, bx1 = 1.35, 4.42          # system boundary
ax.add_patch(Rectangle((bx0, 0.08), bx1 - bx0, H - 0.16, fill=False, lw=1.2, ec="#444444"))
ax.text((bx0 + bx1) / 2, H - 0.3, "Pothole mapping and routing system", ha="center", va="center", fontsize=10, weight="bold")
cx = (bx0 + bx1) / 2
ew, eh = 2.75, 0.46

web = [  # (label, y)
 ("Upload a video and GPS log", 7.05),
 ("Follow job progress", 6.45),
 ("Browse detections of a session", 5.85),
 ("Reprocess, exclude or delete" + chr(10) + "a session", 5.2),
 ("View potholes and surveyed road", 4.55),
 ("Switch demo mode, rebuild the map", 3.95),
 ("Plan routes between two points", 3.35),
 ("Read system status and statistics", 2.75),
]
ax.plot([bx0, bx1], [2.35, 2.35], ls=":", c="#888888", lw=0.8)
ax.text(bx0 + 0.08, 2.22, "command-line interface", fontsize=10, style="italic", color="#444444", va="center")
cli = [("Add recordings (file or folder)", 1.78), ("List sessions", 1.3), ("Rebuild the map", 0.82), ("Reset the database (needs --yes)", 0.36)]
pos = {}
for lab, y in web + cli:
    h = eh + (0.12 if "\n" in lab else 0)
    ax.add_patch(Ellipse((cx, y), ew, h, fc="white", ec="#444444", lw=1))
    ax.text(cx, y, lab, ha="center", va="center", fontsize=10)
    pos[lab] = (cx, y)

def actor(x, y, label):
    ax.add_patch(plt.Circle((x, y + 0.34), 0.09, fc="white", ec="#222222", lw=1.2))
    ax.plot([x, x], [y + 0.25, y - 0.05], c="#222222", lw=1.2)
    ax.plot([x - 0.17, x + 0.17], [y + 0.17, y + 0.17], c="#222222", lw=1.2)
    ax.plot([x, x - 0.14], [y - 0.05, y - 0.3], c="#222222", lw=1.2)
    ax.plot([x, x + 0.14], [y - 0.05, y - 0.3], c="#222222", lw=1.2)
    ax.text(x, y - 0.46, label, ha="center", va="top", fontsize=10)

ax_user = (0.6, 4.9); ax_client = (W - 0.6, 4.9); ax_cli = (0.6, 1.0)
actor(*ax_user, "User\n(web UI)")
actor(*ax_client, "API client")
actor(*ax_cli, "User at the\ncommand line")
for lab, y in web:
    ax.plot([ax_user[0] + 0.2, cx - ew / 2 + 0.05 * 0 + (0.0)], [ax_user[1] + 0.1, y], c="#555555", lw=0.7, zorder=0)
for lab, y in web:
    ax.plot([ax_client[0] - 0.2, cx + ew / 2], [ax_client[1] + 0.1, y], c="#555555", lw=0.7, zorder=0)
for lab, y in cli:
    ax.plot([ax_cli[0] + 0.2, cx - ew / 2], [ax_cli[1] + 0.1, y], c="#555555", lw=0.7, zorder=0)
fig.savefig("../ch4_usecase.pdf")
fig.savefig("preview/ch4_usecase.png", dpi=110)
