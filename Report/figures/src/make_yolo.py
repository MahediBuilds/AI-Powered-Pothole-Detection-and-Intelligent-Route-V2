# YOLOv8s structure drawn from ultralytics/cfg/models/v8/yolov8.yaml (scale s: depth 0.33, width 0.5)
from uml import *
from matplotlib.patches import Rectangle, FancyArrowPatch
c = Canvas(5.77, 6.95)
ax = c.ax
LHH = 0.172
def box(key, x, ytop, w, lines, fill="white", bold=False):
    h = len(lines) * LHH + 0.1
    ax.add_patch(Rectangle((x, ytop - h), w, h, fc=fill, ec="#444444", lw=1.0, zorder=2))
    for i, t in enumerate(lines):
        ax.text(x + w / 2, ytop - 0.05 - LHH / 2 - i * LHH + 0.01, t, ha="center", va="center", fontsize=10, zorder=4, weight="bold" if (bold and i == 0) else "normal")
    c.boxes[key] = (x, ytop - h, w, h)
W = 1.45
XA, XB, XC, XD = 0.04, 1.72, 3.40, 5.0
ax.text(XA + W / 2, 6.78, "Backbone", ha="center", fontsize=10, weight="bold")
ax.text(XB + W / 2, 6.78, "Neck: top-down", ha="center", fontsize=10, weight="bold")
ax.text(XC + W / 2, 6.78, "Neck: bottom-up", ha="center", fontsize=10, weight="bold")
ax.text(XD + 0.36, 6.78, "Head", ha="center", fontsize=10, weight="bold")
box("in", XA, 6.6, W, ["Input 640×640×3"], "#e6e6e6")
box("a1", XA, 6.0, W, ["Conv 3×3 s2, 32", "P1/2"])
box("a2", XA, 5.35, W, ["Conv 3×3 s2, 64", "P2/4"])
box("a3", XA, 4.7, W, ["C2f ×1, 64"])
box("a4", XA, 4.1, W, ["Conv s2 + C2f ×2", "128 ch, P3/8"], "#f2f2f2")
box("a5", XA, 3.0, W, ["Conv s2 + C2f ×2", "256 ch, P4/16"], "#f2f2f2")
box("a6", XA, 1.9, W, ["Conv s2 + C2f ×1", "512 ch, P5/32"])
box("a7", XA, 0.95, W, ["SPPF k=5, 512 ch"], "#f2f2f2")
box("b4", XB, 3.0, W, ["Upsample, concat", "C2f ×1, 256 ch"])
box("b3", XB, 4.1, W, ["Upsample, concat", "C2f ×1, 128 ch", "(P3 output)"], "#e9e9e9")
box("c4", XC, 3.0, W, ["Conv s2, concat", "C2f ×1, 256 ch", "(P4 output)"], "#e9e9e9")
box("c5", XC, 1.9, W, ["Conv s2, concat", "C2f ×1, 512 ch", "(P5 output)"], "#e9e9e9")
box("det", XD, 4.1, 0.73, ["Detect", "3 scales", "P3, P4,", "P5"], "#d9d9d9", bold=True)
def arr(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.0, ec="#333333", fc="#333333",
                                 ls="--" if dashed else "-", shrinkA=0, shrinkB=0, zorder=1))
pt = c.pt
for a, b in [("in", "a1"), ("a1", "a2"), ("a2", "a3"), ("a3", "a4"), ("a4", "a5"), ("a5", "a6"), ("a6", "a7")]:
    arr(pt(a, "b"), pt(b, "t"))
arr(pt("a7", "r", 0.5), (XB + 0.2, pt("a7", "r", 0.5)[1]))
arr((XB + 0.2, pt("a7", "r", 0.5)[1]), (XB + 0.2, c.boxes["b4"][1]))
arr(pt("b4", "t", 0.5), pt("b3", "b", 0.5))
arr(pt("a5", "r", 0.5), pt("b4", "l", 0.5), dashed=True)
arr(pt("a4", "r", 0.5), pt("b3", "l", 0.5), dashed=True)
arr(pt("b3", "b", 0.9), pt("c4", "t", 0.1)) if False else None
arr(pt("b3", "r", 0.5), (XC + 0.3, pt("b3", "r", 0.5)[1]))
arr((XC + 0.3, pt("b3", "r", 0.5)[1]), pt("c4", "t", 0.2))
arr(pt("b4", "r", 0.5), pt("c4", "l", 0.5), dashed=True)
arr(pt("c4", "b", 0.5), pt("c5", "t", 0.5))
arr(pt("a7", "r", 0.5), pt("c5", "b", 0.2)) if False else None
# SPPF -> concat in c5 (dashed), routed below the neck
yb = c.boxes["a7"][1] + c.boxes["a7"][3] / 2
ax.plot([XA + W, XC + 0.4], [yb - 0.0, yb - 0.0], c="#333333", lw=1.0, ls="--", zorder=1)
arr((XC + 0.4, yb), (XC + 0.4, c.boxes["c5"][1]), dashed=True)
# outputs to detect
arr(pt("b3", "r", 0.2), (XD, c.boxes["b3"][1] + c.boxes["b3"][3] * 0.2)) if False else None
# route outputs on the right side of columns
ytopdet = c.boxes["det"][1] + c.boxes["det"][3]
arr((XB + W, c.boxes["b3"][1] + 0.12), (XD, c.boxes["b3"][1] + 0.12)) if False else None
for key, yy in (("c4", None), ("c5", None)):
    pass
# P3 output: from b3 top-right to detect via a line above column C
yp3 = c.boxes["b3"][1] + c.boxes["b3"][3] - 0.1
ax.plot([XB + W, XC + W + 0.15], [yp3 + 0.0, yp3 + 0.0], c="#333333", lw=1.0, zorder=1)
arr((XC + W + 0.15, yp3), (XD, yp3))
arr(pt("c4", "r", 0.5), (XD + 0.3, pt("c4", "r", 0.5)[1]))
ax.plot([XD + 0.3, XD + 0.3], [pt("c4", "r", 0.5)[1], c.boxes["det"][1]], c="#333333", lw=1.0, zorder=1)
ax.plot([XC + W, XD + 0.55], [pt("c5", "r", 0.5)[1], pt("c5", "r", 0.5)[1]], c="#333333", lw=1.0, zorder=1)
arr((XD + 0.55, pt("c5", "r", 0.5)[1]), (XD + 0.55, c.boxes["det"][1]))
# legend
ax.text(0.05, 0.42, "Dashed arrows: feature maps passed on for concatenation.", fontsize=10, va="center")
ax.text(0.05, 0.2, "s2: stride 2. C2f: cross-stage partial block. Channels for scale s.", fontsize=10, va="center")
c.save("ch5_yolov8s") if False else c.save("ch4_yolov8s")
