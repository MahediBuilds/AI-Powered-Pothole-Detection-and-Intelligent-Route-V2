from seq import *
lanes = ["Web\nbrowser", "FastAPI", "JobRunner\nthread", "Pipeline\n(detect, track)", "SQLite"]
xs = [0.58, 1.7, 2.82, 4.0, 5.19]
s = Seq(5.77, 6.4, lanes, xs, box_w=1.05)
B, A, J, P, D = lanes
s.msg(B, A, "1  POST /api/sessions: video, GPS CSV,\n   sidecar (multipart)", lines=2)
s.msg(A, D, "2  intake: store session and all GPS fixes")
s.msg(A, J, "3  submit(session_id)")
s.msg(J, D, "4  create job row (status queued)")
s.msg(A, B, "5  202 Accepted: session and job", ret=True)
s.gap(0.06)
s.msg(J, P, "6  process_session(job)")
s.msg(P, D, "7  job running, stage 'reading video'")
block(s, "loop: each analysed frame", ["8  detect(frame), update MotionTracker;", "   keep best frame of each live track"], x0=2.6, x1=5.7)
s.msg(P, D, "9  one transaction: observations stored, job done")
s.msg(P, J, "10  report (complete or partial)", ret=True)
s.msg(J, D, "11  on_done: full map rebuild (clusters,\n    snapping, coverage, edge statistics)", lines=2)
s.gap(0.06)
yt = s.y
s.y -= LH + 0.06
s.msg(B, A, "12  GET /api/sessions/{id}/status")
s.msg(A, D, "13  read job row")
s.msg(A, B, "14  status, stage, progress", ret=True)
s.frame("loop: while the job is queued or running (every 1 s)", yt, s.y + 0.08)
s.finish("ch4_seq_upload")
