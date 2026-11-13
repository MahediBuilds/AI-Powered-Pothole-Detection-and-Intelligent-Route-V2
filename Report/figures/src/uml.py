"""Tiny UML helper for matplotlib diagrams (class boxes and arrows)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 10})
LH = 0.172   # line height in inches (10 pt text)

class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.boxes = {}

    def cls(self, key, x, ytop, w, name, stereo="", attrs=(), meths=()):
        """Draw a class box with its top-left corner at (x, ytop); returns (x, y, w, h) with y the bottom."""
        lines_n = 1 + (1 if stereo else 0)
        ha = max(len(attrs), 0); hm = max(len(meths), 0)
        pad = 0.06
        h = lines_n * LH + (ha * LH + pad) + (hm * LH + pad) + 0.04
        if ha == 0: h += 0.06
        if hm == 0: h += 0.06
        y = ytop - h
        ax = self.ax
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#444444", lw=1.0, zorder=2))
        hh = lines_n * LH + 0.04
        ax.add_patch(Rectangle((x, ytop - hh), w, hh, fc="#d9d9d9", ec="#444444", lw=1.0, zorder=3))
        yy = ytop - 0.02 - LH / 2
        if stereo:
            ax.text(x + w / 2, yy, "<<%s>>" % stereo, ha="center", va="center", fontsize=10, style="italic", zorder=4); yy -= LH
        ax.text(x + w / 2, yy, name, ha="center", va="center", fontsize=10, weight="bold", zorder=4)
        yy = ytop - hh - pad / 2 - LH / 2 + 0.01
        for a in attrs:
            ax.text(x + 0.05, yy, a, ha="left", va="center", fontsize=10, zorder=4); yy -= LH
        sep = ytop - hh - (ha * LH + pad if ha else 0.06 + pad * 0)
        ax.plot([x, x + w], [sep, sep], c="#444444", lw=1.0, zorder=3)
        yy = sep - pad / 2 - LH / 2 + 0.01
        for m in meths:
            ax.text(x + 0.05, yy, m, ha="left", va="center", fontsize=10, zorder=4); yy -= LH
        self.boxes[key] = (x, y, w, h)
        return self.boxes[key]

    def pt(self, key, side, frac=0.5):
        x, y, w, h = self.boxes[key]
        return {"l": (x, y + h * frac), "r": (x + w, y + h * frac), "t": (x + w * frac, y + h), "b": (x + w * frac, y)}[side]

    def arrow(self, p, q, kind="assoc", label=None, lpos=0.5, loff=(0, 0), dashed=False, via=None, ha="center"):
        ax = self.ax
        pts = [p] + (via or []) + [q]
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] if kind in ("assoc", "dep") else xs[:-1], ys[:-1], c="#444444", lw=1.0, ls="--" if dashed else "-", zorder=1)
        # last segment with head
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        if kind == "assoc" or kind == "dep":
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if kind == "assoc" else "->", mutation_scale=11,
                                         lw=1.0, ec="#444444", fc="#444444", ls="--" if dashed else "-", shrinkA=0, shrinkB=0, zorder=1))
        elif kind == "inherit":
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.0, ec="#444444",
                                         fc="white", ls="--" if dashed else "-", shrinkA=0, shrinkB=0, zorder=1))
        elif kind in ("comp", "aggr"):
            # diamond at the start (whole) end
            self._diamond(p, pts[1] if len(pts) > 2 else q, filled=(kind == "comp"))
            ax.plot([x0, x1], [y0, y1], c="#444444", lw=1.0, zorder=1)
        if label:
            # place label along the whole polyline at fraction lpos of first->last
            lx = p[0] + (q[0] - p[0]) * lpos + loff[0]; ly = p[1] + (q[1] - p[1]) * lpos + loff[1]
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=10, bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)

    def _diamond(self, p, toward, filled):
        import math
        dx, dy = toward[0] - p[0], toward[1] - p[1]
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L
        s = 0.09
        nx_, ny_ = -uy * s * 0.6, ux * s * 0.6
        pts = [p, (p[0] + ux * s + nx_, p[1] + uy * s + ny_), (p[0] + ux * 2 * s, p[1] + uy * 2 * s), (p[0] + ux * s - nx_, p[1] + uy * s - ny_)]
        self.ax.add_patch(Polygon(pts, closed=True, fc="#444444" if filled else "white", ec="#444444", lw=1.0, zorder=3))

    def save(self, name):
        self.fig.savefig(f"../{name}.pdf"); self.fig.savefig(f"preview/{name}.png", dpi=110)
