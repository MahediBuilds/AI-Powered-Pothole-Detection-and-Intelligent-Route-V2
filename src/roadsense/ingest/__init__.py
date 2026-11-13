"""Ingestion: turn a recorded session (video, GPS CSV, optional sidecar) into
frames with trustworthy timestamps and a camera position for any video time.

``gps`` loads and queries the GPS trace, ``video`` decodes frames, and
``session`` ties the sibling files of one recording together. All three share
the video clock: ``t_video_s`` is seconds from the first video frame, and GPS
fixes recorded before the first frame have negative times.
"""


class IngestError(Exception):
    """A session file is missing, unreadable, or unusable."""
