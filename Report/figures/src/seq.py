"""Minimal sequence-diagram helper (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 10})
LH = 0.172

class Seq:
    def __init__(self, w, h, lanes, xs, box_w=1.0):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.x = dict(zip(lanes, xs))
        self.y = h - 0.08
        self.box_w = box_w
        top = self.y
        self.head_h = 0.5
        for name, x in self.x.items():
            self.ax.add_patch(Rectangle((x - box_w / 2, top - self.head_h), box_w, self.head_h, fc="#d9d9d9", ec="#444444", lw=1.0, zorder=3))
            self.ax.text(x, top - self.head_h / 2, name, ha="center", va="center", fontsize=10, weight="bold", zorder=4, linespacing=1.0)
        self.y = top - self.head_h - 0.18
        self.life_top = top - self.head_h
        self.calls = []

    def finish(self, name):
        for x in self.x.values():
            self.ax.plot([x, x], [self.life_top, self.y + 0.1], c="#777777", lw=0.8, ls=(0, (4, 3)), zorder=0)
        self.fig.savefig(f"../{name}.pdf"); self.fig.savefig(f"preview/{name}.png", dpi=110)

    def msg(self, a, b, text, ret=False, lines=1, align="near"):
        """Arrow from lane a to lane b with text above it (left-aligned next to the leftmost lane)."""
        xa, xb = self.x[a], self.x[b]
        y = self.y - (lines - 1) * LH - 0.05
        yy = y - 0.12
        ax = self.ax
        ax.add_patch(FancyArrowPatch((xa, yy), (xb, yy), arrowstyle="->" if ret else "-|>", mutation_scale=11, lw=1.0,
                                     ec="#333333", fc="#333333", ls="--" if ret else "-", shrinkA=0, shrinkB=0, zorder=2))
        left = min(xa, xb)
        if left > 3.3:
            tx, hal = max(xa, xb) + 0.45, "right"
        else:
            tx, hal = left + 0.07, "left"
        ax.text(tx, yy + 0.04 + (lines - 1) * LH / 2 + (lines > 1) * 0.0 + 0.0, text, ha=hal, va="bottom", fontsize=10, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.8), linespacing=1.05)
        self.y = yy - 0.14 - (0.0)
        return yy

    def selfmsg(self, a, text, lines=1, width=0.5):
        x = self.x[a]
        y = self.y - 0.06
        h = 0.25
        ax = self.ax
        ax.plot([x, x + width, x + width], [y, y, y - h], c="#333333", lw=1.0, zorder=2)
        ax.add_patch(FancyArrowPatch((x + width, y - h), (x, y - h), arrowstyle="-|>", mutation_scale=11, lw=1.0, ec="#333333", fc="#333333", shrinkA=0, shrinkB=0, zorder=2))
        ax.text(x + width + 0.06, y - h / 2, text, ha="left", va="center", fontsize=10, zorder=5, bbox=dict(fc="white", ec="none", pad=0.8), linespacing=1.05)
        self.y = y - h - 0.1

    def gap(self, d): self.y -= d

    def frame(self, label, y0, y1, x0=0.1, x1=None, shade=False):
        x1 = x1 or (self.w - 0.1)
        self.ax.add_patch(Rectangle((x0, y1), x1 - x0, y0 - y1, fill=False, ec="#666666", lw=0.9, zorder=1))
        self.ax.text(x0 + 0.04, y0 - 0.02, label, ha="left", va="top", fontsize=10, style="italic", zorder=5,
                     bbox=dict(fc="white", ec="none", pad=0.5))

    def note(self, text, x, w, lines):
        h = lines * LH + 0.1
        y = self.y - h
        self.ax.add_patch(Rectangle((x, y), w, h, fc="#f2f2f2", ec="#666666", lw=0.9, zorder=3))
        self.ax.text(x + 0.05, y + h / 2, text, ha="left", va="center", fontsize=10, zorder=4, linespacing=1.05)
        self.y = y - 0.1

def block(s, label, lines, x0=0.08, x1=None, fill="#f7f7f7"):
    """A framed block (loop or note) with a label and text lines; advances the cursor."""
    x1 = x1 or (s.w - 0.08)
    h = (len(lines) + (1 if label else 0)) * LH + 0.12
    y1 = s.y - h
    s.ax.add_patch(Rectangle((x0, y1), x1 - x0, h, fc=fill, ec="#666666", lw=0.9, zorder=2))
    yy = s.y - 0.06 - LH / 2
    if label:
        s.ax.text(x0 + 0.06, yy, label, ha="left", va="center", fontsize=10, style="italic", weight="bold", zorder=4); yy -= LH
    for t in lines:
        s.ax.text(x0 + 0.06, yy, t, ha="left", va="center", fontsize=10, zorder=4); yy -= LH
    s.y = y1 - 0.1
