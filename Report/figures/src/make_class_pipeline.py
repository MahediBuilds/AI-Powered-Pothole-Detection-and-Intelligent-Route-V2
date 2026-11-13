from uml import *
c = Canvas(5.77, 6.55)
X1, X2, X3 = 0.04, 2.0, 3.96
W = 1.74
top = 6.5
# col 2
c.cls("proc", X2, top, W, "process_session", "function", ["module pipeline.process"], ["process_session(...)", "plan_tracking(...)"])
c.cls("trk", X2, 4.6, W, "MotionTracker", "", ["iou_match", "max_age_source_frames", "min_hits"], ["update(dets, frame, t)", "flush()", "predicted_box(...)"])
c.cls("track", X2, 2.8, W, "Track", "dataclass", ["track_id, confirmed", "points: list"], ["n_hits, first, last,", "best"])
c.cls("tp", X2, 1.2, W, "TrackPoint", "dataclass", ["frame_index", "t_video_s", "detection"], [])
# col 1
c.cls("jr", X1, top, W, "JobRunner", "", ["-queue: Queue", "-detector"], ["+start(), +stop()", "+submit(session_id)", "+wait_idle()"])
c.cls("dl", X1, 4.6, W, "DetectorLike", "Protocol", ["weights_sha256"], ["detect(image)", "info()"])
c.cls("det", X1, 3.05, W, "Detector", "", ["-conf, -iou, -imgsz", "-device"], ["detect(image)", "detect_file(path)"])
c.cls("dn", X1, 1.2, W, "Detection", "dataclass", ["x1, y1, x2, y2", "confidence", "class_id, class_name"], [])
# col 3
c.cls("plan", X3, top, W, "TrackingPlan", "dataclass", ["mode", "frame_stride, min_hits", "max_age_source_frames"], [])
c.cls("pos", X3, 4.6, W, "position_at", "function", ["module ingest.gps"], ["position_at(log, t,", "  max_gap_s,", "  max_extrapolation_s)"])
c.cls("gps", X3, 2.8, W, "GpsLog", "dataclass", ["fixes: tuple[GpsFix]", "issues"], ["used_fixes"])
A = c.arrow
A(c.pt("jr", "r", 0.5), c.pt("proc", "l", 0.5), "assoc")
A(c.pt("proc", "r", 0.5), c.pt("plan", "l", 0.5), "assoc", dashed=True)
A(c.pt("proc", "b", 0.5), c.pt("trk", "t", 0.5), "assoc", "creates", 0.5, (0.42, 0))
A(c.pt("trk", "b", 0.5), c.pt("track", "t", 0.5), "comp", "1..*", 0.5, (0.3, 0))
A(c.pt("track", "b", 0.5), c.pt("tp", "t", 0.5), "comp", "1..*", 0.5, (0.3, 0))
A(c.pt("tp", "l", 0.5), c.pt("dn", "r", 0.5), "assoc", None)
A(c.pt("jr", "b", 0.5), c.pt("dl", "t", 0.5), "aggr", "owns", 0.5, (0.3, 0))
A(c.pt("det", "t", 0.5), c.pt("dl", "b", 0.5), "inherit", None, dashed=True)
A(c.pt("det", "b", 0.5), c.pt("dn", "t", 0.5), "assoc", "returns", 0.5, (0.35, 0))
A(c.pt("proc", "r", 0.2), c.pt("pos", "l", 0.5), "assoc")
A(c.pt("pos", "b", 0.5), c.pt("gps", "t", 0.5), "assoc", "reads", 0.5, (0.3, 0))
A(c.pt("proc", "l", 0.2), c.pt("dl", "r", 0.8), "assoc")
c.save("ch4_class_pipeline")
