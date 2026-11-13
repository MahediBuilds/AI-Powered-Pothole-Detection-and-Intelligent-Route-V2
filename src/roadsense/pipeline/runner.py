"""Background job runner: one worker thread processes uploaded sessions in order.

One GPU means one job at a time. ``submit`` creates a ``queued`` job row at once
and returns its id; the worker thread picks jobs up in submission order.

Threading rules: SQLite connections are never shared. ``submit`` opens a
short-lived connection of its own, and the worker thread opens one for its whole
life. The detector is built lazily on the worker thread by ``detector_factory``,
once, and reused for all later jobs; if the factory raises, that job fails with
the message and the factory is tried again on the next job.

``on_done`` (the map rebuild) runs after each successful job. If it fails, the
job stays ``done`` (its detections are stored) and the step is tried again while
the worker is idle, every ``RETRY_INTERVAL_S`` up to ``MAX_POST_RETRIES`` times;
a later job's successful ``on_done`` also settles it, because a map rebuild is a
full rebuild. Only when the retries run out is a warning added to the job's
report. The map build also records which jobs it used, so a rebuild still
missing after that is made at the next server start.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roadsense.config import AppConfig
from roadsense.detection.detector import DetectorLike
from roadsense.pipeline.process import PipelineError, process_session
from roadsense.storage import db
from roadsense.storage import jobs as jobs_db

log = logging.getLogger(__name__)

OnDone = Callable[[sqlite3.Connection, str, str], None]

RETRY_INTERVAL_S = 3.0
MAX_POST_RETRIES = 10


class JobRunner:
    def __init__(
        self,
        db_path: str | Path,
        cfg: AppConfig,
        detector_factory: Callable[[], DetectorLike],
        on_done: OnDone | None = None,
    ) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._factory = detector_factory
        self._on_done = on_done
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._cond = threading.Condition()
        self._pending = 0  # submitted jobs not yet finished (or dropped by stop)
        self._detector: DetectorLike | None = None
        # (session_id, job_id, retries left, last error) of an on_done still owed
        self._owed: tuple[str, str, int, str] | None = None
        self.retry_interval_s = RETRY_INTERVAL_S
        self.max_post_retries = MAX_POST_RETRIES

    # -- public -------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread (no effect if it is already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._work, name="job-runner", daemon=True)
        self._thread.start()

    def submit(self, session_id: str) -> str:
        """Create a ``queued`` job for the session, enqueue it, and return its id.

        Raises ``sqlite3.IntegrityError`` if the session does not exist.
        """
        conn = db.connect(self._db_path)
        try:
            job_id = jobs_db.create_job(conn, session_id)
            conn.commit()
        finally:
            conn.close()
        with self._cond:
            self._pending += 1
        self._queue.put((session_id, job_id))
        return job_id

    def stop(self, timeout: float | None = 10.0) -> bool:
        """Stop after the current job; jobs still waiting stay ``queued`` in the database.

        Returns True if the worker thread has exited (or was never started), False
        if it is still running after ``timeout`` seconds.
        """
        self._stopping.set()
        self._queue.put(None)
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def detector(self) -> DetectorLike | None:
        """The detector once the worker has built it, else None (never builds one)."""
        return self._detector

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every submitted job has finished; False if the timeout ran out."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    # -- worker -------------------------------------------------------------

    def _work(self) -> None:
        conn = db.connect(self._db_path)
        try:
            while True:
                try:
                    item = self._queue.get(
                        timeout=self.retry_interval_s if self._owed is not None else None
                    )
                except queue.Empty:
                    self._retry_owed(conn)
                    continue
                if item is None:
                    return
                try:
                    if not self._stopping.is_set():
                        self._run_job(conn, *item)
                except Exception:  # last resort: the worker must survive any job
                    log.exception("job runner: unexpected error on job %s", item[1])
                finally:
                    with self._cond:
                        self._pending -= 1
                        self._cond.notify_all()
        finally:
            conn.close()

    def _run_job(self, conn: sqlite3.Connection, session_id: str, job_id: str) -> None:
        if self._detector is None:
            try:
                self._detector = self._factory()
            except Exception as exc:
                self._fail(conn, job_id, f"detector could not be loaded: {exc}")
                return
        try:
            process_session(conn, session_id, self._detector, self._cfg, job_id=job_id)
        except PipelineError:
            return  # process_session has marked the job failed
        except Exception as exc:  # defensive: process_session wraps its own errors
            self._fail(conn, job_id, f"{type(exc).__name__}: {exc}")
            return
        if self._on_done is not None:
            self._post(conn, session_id, job_id, self.max_post_retries)

    def _post(self, conn: sqlite3.Connection, session_id: str, job_id: str, retries: int) -> None:
        """Run ``on_done``; on failure keep it owed, or warn once the retries are used up."""
        try:
            self._on_done(conn, session_id, job_id)
            conn.commit()
            self._owed = None
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            try:
                conn.rollback()
            except Exception:  # pragma: no cover - the connection itself is broken
                pass
        if retries > 0:
            log.warning("job runner: on_done failed for job %s (%s); will retry", job_id, message)
            self._owed = (session_id, job_id, retries - 1, message)
            return
        log.error("job runner: on_done failed for job %s and retries ran out: %s", job_id, message)
        self._owed = None
        try:
            self._add_warning(
                conn, job_id,
                f"post-processing failed: {message}. The map may not show this session yet; "
                "it is rebuilt at the next change or restart.",
            )  # fmt: skip
        except Exception:
            log.exception("job runner: could not record the warning on job %s", job_id)

    def _retry_owed(self, conn: sqlite3.Connection) -> None:
        if self._owed is None or self._stopping.is_set():
            return
        session_id, job_id, retries, _ = self._owed
        self._post(conn, session_id, job_id, retries)

    @staticmethod
    def _fail(conn: sqlite3.Connection, job_id: str, message: str) -> None:
        conn.rollback()
        jobs_db.mark_failed(conn, job_id, message)
        conn.commit()

    @staticmethod
    def _add_warning(conn: sqlite3.Connection, job_id: str, warning: str) -> None:
        conn.rollback()
        job: dict[str, Any] | None = jobs_db.get_job(conn, job_id)
        summary = json.loads(job["summary_json"]) if job and job["summary_json"] else {}
        summary.setdefault("warnings", []).append(warning)
        jobs_db.update_summary(conn, job_id, summary)
        conn.commit()
