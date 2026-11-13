from uml import *
c = Canvas(5.77, 6.45)
X1, X2, X3 = 0.04, 2.0, 3.96
W = 1.74
r1, r2, r3, r4 = 6.4, 4.65, 2.8, 1.15
c.cls("cfg", X1, r1, W, "AppConfig", "pydantic", ["ingest, detection,", "tracking, mapping,", "routing, uploads"], ["(unknown keys are", " errors)"])
c.cls("state", X2, r1, W, "AppState", "dataclass", ["cfg, graph, runner", "map_lock, job_lock"], ["graph_info()", "weights_sha256()"])
c.cls("lock", X3, r1, W, "ProcessLock", "", ["path, role"], ["acquire()", "release()"])
c.cls("map", X1, r2, W, "Map build", "functions", ["dedup.cluster_points", "segments.rebuild_map", "mapstate.", "  write_and_rebuild"], [])
c.cls("rg", X2, r2, W, "RoadGraph", "dataclass", ["g: MultiDiGraph", "sha256, bbox, meta"], [])
c.cls("plan", X3, r2, W, "plan_routes", "function", ["module routing.router"], ["plan_routes(...)", "assumed_rate(...)", "evidence_level(...)"])
c.cls("ei", X1, r3, W, "EdgeIndex", "", ["graph, nodes, edges"], ["snap(lat, lon,", "  max_dist_m)"])
c.cls("ni", X2, r3, W, "NodeIndex", "", [], ["nearest_node(lat, lon,", "  max_dist_m)"])
c.cls("err", X3, r3, W, "RoutingError", "exception", [], ["NoGraphError", "OutsideGraphError", "NoRouteError"])
c.cls("snap", X1, r4, W, "Snap", "dataclass", ["edge_u, edge_v, edge_key", "offset_m, distance_m", "lat, lon"], [])
c.cls("lp", X2, r4, W, "LocalProjection", "", ["lat0, lon0"], ["to_xy(lat, lon)"])
A = c.arrow
A(c.pt("state", "l", 0.5), c.pt("cfg", "r", 0.5), "assoc")
A(c.pt("state", "b", 0.5), c.pt("rg", "t", 0.5), "assoc", "graph", 0.5, (0.3, 0))
A(c.pt("map", "r", 0.5), c.pt("rg", "l", 0.5), "assoc")
A(c.pt("plan", "l", 0.5), c.pt("rg", "r", 0.5), "assoc")
A(c.pt("map", "b", 0.5), c.pt("ei", "t", 0.5), "assoc")
A(c.pt("ei", "r", 0.5), c.pt("ni", "l", 0.5), "comp")
A(c.pt("ei", "t", 0.85), c.pt("rg", "b", 0.2), "assoc", via=[(c.pt("ei","t",0.85)[0], 3.45+0.0)] if False else None)
A(c.pt("ei", "b", 0.5), c.pt("snap", "t", 0.5), "assoc", "returns", 0.5, (0.35, 0))
A(c.pt("ni", "b", 0.5), c.pt("lp", "t", 0.5), "assoc", "uses", 0.5, (0.25, 0))
A(c.pt("plan", "b", 0.5), c.pt("err", "t", 0.5), "assoc", "raises", 0.5, (0.35, 0), dashed=True)
A(c.pt("plan", "b", 0.12), c.pt("ni", "r", 0.5), "assoc")
c.save("ch4_class_service")
