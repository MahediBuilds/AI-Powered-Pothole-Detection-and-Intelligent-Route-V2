"""Plots drawn only from numbers already stored in the project's evaluation outputs. Nothing is smoothed,
re-estimated or recomputed.
 1. Paired differences (candidate minus baseline) with 95% group-bootstrap intervals, parsed from
    New Implementation/runs/eval/unified_v1_y8s_1280/REPORT.md, section 3a.
 2. Detections per recording for the three models, read from Three_Model_Comparison/summary.csv.
Run with the system python from the report root."""
import csv, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPORT = r"<PROJECT_ROOT>/New Implementation\runs\eval\unified_v1_y8s_1280\REPORT.md"
SUMMARY = r"<PROJECT_ROOT>/Three_Model_Comparison\summary.csv"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

# ---- 1. paired differences, section 3a ----
text = open(REPORT, encoding="utf-8").read()
sec = text.split("## 3a.")[1].split("## 3b.")[0]
row = re.compile(r"^\| ([^|]+?) \| [^|]+ \| [^|]+ \| (-?[0-9.]+) \[(-?[0-9.]+), (-?[0-9.]+)\]( \*)? \|$")
sets, cur = {}, None
for line in sec.splitlines():
    m = re.match(r"^## (rdd2022 test|hrp4k test|kaggle probe)  ", line)
    if m:
        cur = m.group(1); sets[cur] = {}; continue
    if line.startswith("Manhole probe"):
        cur = "kaggle probe"
    r = row.match(line)
    if r and cur:
        sets[cur][r.group(1)] = (float(r.group(2)), float(r.group(3)), float(r.group(4)))

panels = [("rdd2022 test", "RDD2022 test"), ("hrp4k test", "HRP4K test"), ("kaggle probe", "Manhole probe set")]
metrics = [("mAP50", "mAP$_{50}$"), ("mAP50-95", "mAP$_{50-95}$"), ("Precision", "Precision"), ("Recall", "Recall"),
           ("F1", "F1"), ("FP / image", "FP per image")]
extra = ("manhole hit share, IoU>=0.3", "Manhole hit share")
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.5))
for ax, (key, title) in zip(axes, panels):
    items = [(lab, sets[key][k]) for k, lab in metrics]
    if key == "kaggle probe":
        items.append((extra[1], sets[key][extra[0]]))
    for i, (lab, (d, lo, hi)) in enumerate(items):
        y = len(items) - 1 - i
        excl = (lo > 0) or (hi < 0)
        ax.plot([lo, hi], [y, y], color="#1b4f72", lw=1.6)
        ax.plot([d], [y], "o", ms=5.5, mfc="#1b4f72" if excl else "white", mec="#1b4f72", mew=1.3)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for lab, _ in reversed(items)])
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("candidate minus baseline")
    if key != "rdd2022 test":
        ax.tick_params(labelleft=False) if False else None
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ch6_paired_differences.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- 2. detections per recording ----
rows = list(csv.DictReader(open(SUMMARY, newline="")))
recs = []
for r in rows:
    if r["recording"] not in recs and r["recording"].startswith("road_"):
        recs.append(r["recording"])
data = {m: {} for m in ("old", "baseline", "candidate")}
for r in rows:
    if r["model"] in data and r["recording"] in recs:
        data[r["model"]][r["recording"]] = int(r["total_detections"])
recs_sorted = sorted(recs)
colors = {"old": "#CC79A7", "baseline": "#009E73", "candidate": "#E69F00"}
names = {"old": "Earlier implementation", "baseline": "Baseline", "candidate": "Candidate"}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
h = 0.26
for j, m in enumerate(("old", "baseline", "candidate")):
    ys = [len(recs_sorted) - 1 - i + (1 - j) * h for i in range(len(recs_sorted))]
    ax.barh(ys, [data[m][r] for r in recs_sorted], height=h, color=colors[m], label=names[m])
ax.set_yticks(range(len(recs_sorted)))
ax.set_yticklabels([r.replace("road_", "") for r in reversed(recs_sorted)])
ax.set_xlabel("Detections over all frames of the recording")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ch6_recording_counts.png"), dpi=200)
tot = {m: sum(data[m].values()) for m in data}
print(sets["kaggle probe"].keys()); print("totals", tot, len(recs_sorted))
