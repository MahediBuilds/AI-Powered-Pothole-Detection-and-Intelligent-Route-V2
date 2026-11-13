"""Per-session processing: video -> detections -> tracks -> stored observations."""

from roadsense.pipeline.process import PIPELINE_VERSION, PipelineError, process_session

__all__ = ["PIPELINE_VERSION", "PipelineError", "process_session"]
