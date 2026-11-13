"""Corrected grouping for kaggle_probe_v1 -> groups_v2.csv (does not touch other files).

Parses ``vlcsnap-YYYY-MM-DD-HHhMMmSSsmmm`` and ``YYYYMMDD_HHMMSS`` names, sorts by time and starts a
new group when the gap to the previous image exceeds 60 s. Unparseable names (e.g. ``vlcsnap-00004``,
a frame counter without a clock) get their own group and are flagged ``no_timestamp=1``.
"""
import csv
import datetime as dt
import re
import sys
from pathlib import Path

GAP_S = 60.0
VLC = re.compile(r"^vlcsnap-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})")
PHONE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def parse_ts(name):
    m = VLC.match(name)
    try:
        if m:
            y, mo, d, h, mi, s, ms = map(int, m.groups())
            return dt.datetime(y, mo, d, h, mi, s, ms * 1000)
        m = PHONE.match(name)
        if m:
            return dt.datetime(*map(int, m.groups()))
    except ValueError:
        return None
    return None


def group_names(names, gap_s=GAP_S):
    """-> {name: (group_id, no_timestamp)}"""
    parsed = sorted(((parse_ts(n), n) for n in names if parse_ts(n) is not None))
    out, gi, prev = {}, -1, None
    for ts, n in parsed:
        if prev is None or (ts - prev).total_seconds() > gap_s:
            gi += 1
        out[n] = (f"t{gi:04d}", 0); prev = ts
    for k, n in enumerate(sorted(n for n in names if parse_ts(n) is None)):
        out[n] = (f"u{k:04d}", 1)
    return out


def main(root):
    root = Path(root)
    with open(root / "manifest.csv", newline="") as f:
        names = [r["image"] for r in csv.DictReader(f)]
    g = group_names(names)
    with open(root / "groups_v2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "group", "no_timestamp"])
        for n in sorted(names):
            w.writerow([n, *g[n]])
    from collections import Counter
    c = Counter(v[0] for v in g.values())
    print(f"images {len(names)} groups {len(c)} largest {c.most_common(1)} unparseable {sum(v[1] for v in g.values())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/kaggle_probe_v1")
