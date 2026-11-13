"""Spatial de-duplication of observations by diameter-bounded clustering.

Sightings of one pothole are grouped so that every pair of points inside a
cluster is at most ``radius_m`` apart (exact great-circle distance), so no two
sightings of one pothole are more than ``radius_m`` apart. Unlike single
linkage, a chain of sightings each a few metres from the next is cut into
several clusters instead of becoming one long "pothole".

Rule (greedy complete linkage). Every pair of points within ``radius_m`` is a
candidate, found with a grid of cell size ``radius_m`` so work stays near linear
in sparse data. Candidates are sorted by distance ascending, ties broken by
``(i, j)``. Going through them in that order, the clusters of the two points
are merged only if every cross pair of their members is within ``radius_m``.
The original sightings are never changed, only labelled.

Limits, stated plainly:

* A greedy merge is order-dependent in principle (a different merge order can
  give a different, equally valid partition); the fixed sort order makes the
  result deterministic, not optimal. It is not guaranteed to find the fewest
  clusters.
* Two real potholes closer together than ``radius_m`` can still be merged.
* One pothole seen with GPS error larger than ``radius_m`` can still be split.
* Because every pair must be within the radius, repeat sightings of ONE pothole
  whose GPS positions scatter by more than the radius are split into several
  potholes, and this gets more likely as the number of passes over the same road
  grows (a simulation with 3 m position error and 10 sightings per pothole gave
  about 2.5 clusters per real pothole). The radius is a trade-off between that
  and merging neighbouring potholes.
"""

from __future__ import annotations

import math

import numpy as np

from roadsense.geo.projection import LocalProjection
from roadsense.ingest.gps import EARTH_RADIUS_M


def cluster_points(lats, lons, radius_m: float) -> list[int]:
    """Cluster labels ``0..k-1`` for the points, one per input point.

    Labels are numbered by the smallest member index of each cluster (the first
    point is always in cluster 0), so the result is deterministic and does not
    depend on hashing order. Zero or one point is fine. Raises ``ValueError`` for
    a negative radius, mismatched lengths or non-finite coordinates.
    """
    lat = np.asarray(lats, dtype=float).ravel()
    lon = np.asarray(lons, dtype=float).ravel()
    if lat.shape != lon.shape:
        raise ValueError("lats and lons must have the same length")
    if not (math.isfinite(radius_m) and radius_m >= 0):
        raise ValueError(f"radius_m must be a finite number >= 0, got {radius_m}")
    n = len(lat)
    if n == 0:
        return []
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("coordinates must be finite")

    # Grid cells slightly larger than the radius absorb projection distortion;
    # the haversine check below makes the final decision.
    proj = LocalProjection(float(lat.mean()), float(lon.mean()))
    x, y = proj.to_xy(lat, lon)
    cell = max(radius_m * 1.05, 1e-6)
    cx = np.floor(x / cell).astype(np.int64)
    cy = np.floor(y / cell).astype(np.int64)
    cells: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cells.setdefault((int(cx[i]), int(cy[i])), []).append(i)

    phi = np.radians(lat)
    lam = np.radians(lon)
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        cand: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(j for j in cells.get((int(cx[i]) + dx, int(cy[i]) + dy), ()) if j > i)
        if not cand:
            continue
        j = np.array(cand)
        h = (
            np.sin((phi[j] - phi[i]) / 2) ** 2
            + np.cos(phi[i]) * np.cos(phi[j]) * np.sin((lam[j] - lam[i]) / 2) ** 2
        )
        dist = 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(h)))
        keep = dist <= radius_m
        pairs.extend((float(d), i, int(k)) for d, k in zip(dist[keep], j[keep]))

    pairs.sort()  # distance, then i, then j
    within = {(i, j) for _, i, j in pairs}  # every pair at most radius_m apart

    cluster = list(range(n))  # point -> cluster id
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    # Refused merges: cluster id -> ids of clusters known not to merge with it. A refusal
    # stays true until a cluster changes, and a merged cluster inherits the union of both
    # parts' refusals (a bad pair stays bad), so the member scan runs once per cluster pair.
    refused: dict[int, set[int]] = {i: set() for i in range(n)}
    for _, i, j in pairs:
        ci, cj = cluster[i], cluster[j]
        if ci == cj or cj in refused[ci]:
            continue
        if not all(
            (min(a, b), max(a, b)) in within for a in members[ci] for b in members[cj]
        ):
            refused[ci].add(cj)
            refused[cj].add(ci)
            continue
        keep_id, drop_id = (ci, cj) if len(members[ci]) >= len(members[cj]) else (cj, ci)
        for m in members[drop_id]:
            cluster[m] = keep_id
        members[keep_id].extend(members.pop(drop_id))
        gone = refused.pop(drop_id)
        for r in gone:  # clusters that refused the absorbed one now refuse the merged one
            refused[r].discard(drop_id)
            refused[r].add(keep_id)
        refused[keep_id] |= gone

    labels: dict[int, int] = {}
    out = []
    for i in range(n):
        c = cluster[i]
        if c not in labels:
            labels[c] = len(labels)
        out.append(labels[c])
    return out
