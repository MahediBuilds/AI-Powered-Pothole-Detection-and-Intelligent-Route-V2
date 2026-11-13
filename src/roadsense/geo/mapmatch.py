"""Nearest-edge snapping of points to the road graph.

This is NOT HMM map matching. Each point is snapped on its own to the closest
point on the closest road edge, with no use of the trace's order, heading or
connectivity. It can therefore pick the wrong one of two close parallel roads,
and a point near a junction may snap to a side street. ``max_dist_m`` bounds how
far a point may be from a road before it is left unmatched (None).

``EdgeIndex`` holds every undirected edge as a polyline in local metres and
checks a query against all segments with vectorised numpy, in chunks so memory
stays bounded (adequate for graphs of some tens of thousands of segments).
Ties go to the segment that comes first in canonical-edge order (sorted
``(u, v, key)``), so results are deterministic.

Offsets are nominal metres from the lower node id ``u`` along the edge, scaled
so that the end of the polyline is exactly ``length_m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadsense.geo.projection import LocalProjection
from roadsense.routing.graph import RoadGraph, canonical_edges, edge_geometry

# Upper bound on points x segments handled per numpy step (about 32 MB per array).
_CHUNK_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class Snap:
    """A point projected onto an edge (canonical ``edge_u < edge_v`` identity)."""

    edge_u: int
    edge_v: int
    edge_key: int
    offset_m: float  # distance along the edge from edge_u, within [0, length_m]
    distance_m: float  # distance from the original point to the snapped point
    lat: float  # the snapped point
    lon: float

    @property
    def edge(self) -> tuple[int, int, int]:
        return (self.edge_u, self.edge_v, self.edge_key)


def edge_index_for(graph: RoadGraph) -> "EdgeIndex":
    """The ``EdgeIndex`` of a graph, built once and kept on the graph object.

    A loaded graph never changes, and building the index is the slow part of a
    map rebuild, so it is done once per graph instead of once per rebuild.
    """
    index = getattr(graph, "_edge_index", None)
    if index is None:
        index = EdgeIndex(graph)
        graph._edge_index = index  # type: ignore[attr-defined]
    return index


class NodeIndex:
    """Nearest graph node lookup (brute force over all nodes)."""

    def __init__(self, graph: RoadGraph) -> None:
        self._ids = sorted(graph.g.nodes)
        lats = np.array([graph.g.nodes[n]["lat"] for n in self._ids])
        lons = np.array([graph.g.nodes[n]["lon"] for n in self._ids])
        self._proj = LocalProjection(float(lats.mean()), float(lons.mean()))
        self._x, self._y = self._proj.to_xy(lats, lons)

    def nearest_node(self, lat: float, lon: float, max_dist_m: float) -> int | None:
        """Id of the nearest node within ``max_dist_m``, else None (ties: lowest id)."""
        px, py = self._proj.to_xy(lat, lon)
        d = np.hypot(self._x - px, self._y - py)
        i = int(np.argmin(d))
        return int(self._ids[i]) if d[i] <= max_dist_m else None


class EdgeIndex:
    """Snap points to the nearest edge of a ``RoadGraph``."""

    def __init__(self, graph: RoadGraph) -> None:
        self.graph = graph
        self.nodes = NodeIndex(graph)
        self._proj = self.nodes._proj
        self.edges = canonical_edges(graph)
        self.lengths: list[float] = []
        ax, ay, bx, by, owner, start, sscale = [], [], [], [], [], [], []
        for idx, (u, v, k) in enumerate(self.edges):
            geom, length = edge_geometry(graph, u, v, k)
            self.lengths.append(length)
            x, y = self._proj.to_xy(
                np.array([p[0] for p in geom]), np.array([p[1] for p in geom])
            )
            seg = np.hypot(np.diff(x), np.diff(y))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(cum[-1])
            # nominal metres per polyline metre; 1.0 for a degenerate polyline
            scale = length / total if total > 0 else 1.0
            ax.append(x[:-1])
            ay.append(y[:-1])
            bx.append(x[1:])
            by.append(y[1:])
            owner.append(np.full(len(seg), idx, dtype=np.int64))
            start.append(cum[:-1] * scale)
            sscale.append(np.full(len(seg), scale))
        if ax:
            self._ax, self._ay = np.concatenate(ax), np.concatenate(ay)
            self._bx, self._by = np.concatenate(bx), np.concatenate(by)
            self._owner = np.concatenate(owner)
            self._start = np.concatenate(start)
            self._sscale = np.concatenate(sscale)
        else:
            self._ax = self._ay = self._bx = self._by = np.zeros(0)
            self._owner = np.zeros(0, dtype=np.int64)
            self._start = self._sscale = np.zeros(0)
        self._dx = self._bx - self._ax
        self._dy = self._by - self._ay
        self._len2 = self._dx**2 + self._dy**2

    def nearest_node(self, lat: float, lon: float, max_dist_m: float) -> int | None:
        """Id of the nearest graph node within ``max_dist_m``, else None."""
        return self.nodes.nearest_node(lat, lon, max_dist_m)

    def snap(self, lat: float, lon: float, max_dist_m: float) -> Snap | None:
        """Snap one point; None if no edge is within ``max_dist_m`` metres."""
        return self.snap_many([lat], [lon], max_dist_m)[0]

    def snap_many(self, lats, lons, max_dist_m: float) -> list[Snap | None]:
        """Snap many points; the result is aligned with the input (None = unmatched)."""
        lats = np.asarray(lats, dtype=float).ravel()
        lons = np.asarray(lons, dtype=float).ravel()
        if lats.shape != lons.shape:
            raise ValueError("lats and lons must have the same length")
        n = len(lats)
        out: list[Snap | None] = [None] * n
        n_seg = len(self._ax)
        if n == 0 or n_seg == 0:
            return out
        px, py = self._proj.to_xy(lats, lons)
        step = max(1, _CHUNK_ELEMENTS // n_seg)
        for lo in range(0, n, step):
            qx = px[lo : lo + step, None]
            qy = py[lo : lo + step, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                t = ((qx - self._ax) * self._dx + (qy - self._ay) * self._dy) / self._len2
            t = np.where(self._len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
            cx = self._ax + t * self._dx
            cy = self._ay + t * self._dy
            d2 = (qx - cx) ** 2 + (qy - cy) ** 2
            best = np.argmin(d2, axis=1)  # first minimum -> deterministic ties
            rows = np.arange(len(best))
            dist = np.sqrt(d2[rows, best])
            bt = t[rows, best]
            sx, sy = cx[rows, best], cy[rows, best]
            slat, slon = self._proj.to_latlon(sx, sy)
            for j in range(len(best)):
                if not dist[j] <= max_dist_m:  # also rejects NaN
                    continue
                s = int(best[j])
                eidx = int(self._owner[s])
                u, v, k = self.edges[eidx]
                seglen = float(np.sqrt(self._len2[s]))
                offset = float(self._start[s] + bt[j] * seglen * self._sscale[s])
                offset = min(max(offset, 0.0), self.lengths[eidx])
                out[lo + j] = Snap(u, v, k, offset, float(dist[j]), float(slat[j]), float(slon[j]))
        return out
