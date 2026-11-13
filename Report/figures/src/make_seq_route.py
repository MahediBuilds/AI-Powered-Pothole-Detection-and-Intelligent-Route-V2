from seq import *
lanes = ["Web\nbrowser", "FastAPI\n(routes)", "plan_routes", "SQLite", "RoadGraph\n(networkx)"]
xs = [0.58, 1.7, 2.82, 4.0, 5.19]
s = Seq(5.77, 6.75, lanes, xs, box_w=1.05)
B, A, R, D, G = lanes
s.msg(B, A, "1  POST /api/routes: origin, destination,\n   avoidance in [0, 1]", lines=2)
s.msg(A, R, "2  plan_routes(graph, origin, destination, avoidance)", lines=1)
s.msg(R, G, "3  nearest node of each end point (limit 250 m)")
s.msg(G, R, "4  two nodes", ret=True)
s.msg(R, D, "5  latest build: edge_stats, edge_coverage,\n   potholes", lines=2)
s.msg(D, R, "6  rows", ret=True)
block(s, "", ["7  totals N (mapped on-road potholes) and S", "   (surveyed metres); assumed rate by shrinkage"], x0=2.6, x1=5.7)
yt = s.y
s.y -= LH + 0.06
s.msg(R, G, "8  Dijkstra with edge cost L + a × 300 m × E", lines=1)
s.msg(G, R, "9  edge sequence", ret=True)
s.frame("loop: a = 0, requested value, 1", yt, s.y + 0.08, x0=2.55, x1=5.72)
block(s, "", ["10  merge identical routes; describe each:", "    distance, time, surveyed share, evidence"], x0=2.6, x1=5.7)
s.msg(R, A, "11  plan: 1 to 3 routes, cost model, caveats", ret=True)
s.msg(A, B, "12  200 OK, JSON", ret=True)
s.gap(0.05)
block(s, "Errors returned in place of steps 11 and 12", ["no graph loaded: 409 no_graph", "end point over 250 m from a node: 422 outside_graph", "no connection: 404 no_route; bad input: 422 invalid_input"], x0=0.08, x1=5.69, fill="#ffffff")
s.finish("ch4_seq_route")
