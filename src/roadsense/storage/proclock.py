"""One job-processing process per database.

Jobs live in an in-memory queue, so the process that queued a job is the only
one that can run it. Two processes working on the same database would each
assume the other's ``queued``/``running`` rows were left behind by a crash.
``ProcessLock`` prevents that: the server (and the command line while it
processes recordings) holds an operating-system lock on ``<database>.lock`` for
as long as it runs.

The lock belongs to an open file handle, so the operating system releases it
when the process ends, however it ends. That makes it a liveness test with no
heartbeat and no stale-lock clean-up: if the lock can be taken, no other
processing process is alive, and any job still marked ``queued`` or ``running``
really is orphaned.

The lock file is never deleted (deleting it while another process is opening
it would race); a small ``.info`` file next to it names the holder for error
messages only.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class AlreadyRunningError(Exception):
    """Another process is already processing jobs on this database."""


def lock_path_for(db_path: str | os.PathLike[str]) -> Path:
    return Path(f"{db_path}.lock")


class ProcessLock:
    """Exclusive, non-blocking, crash-safe lock on a file."""

    def __init__(self, path: str | os.PathLike[str], role: str = "server") -> None:
        self.path = Path(path)
        self.role = role
        self._handle = None

    @property
    def info_path(self) -> Path:
        return Path(f"{self.path}.info")

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "ProcessLock":
        """Take the lock or raise ``AlreadyRunningError`` naming the holder. Never waits."""
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")  # noqa: SIM115 - kept open for the life of the lock
        try:
            _lock(handle)
        except OSError:
            handle.close()
            raise AlreadyRunningError(
                f"another RoadSense process is already working on this database{self._holder()}. "
                "Stop it first, or use a different database (db_path in the config)."
            ) from None
        self._handle = handle
        try:
            self.info_path.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "role": self.role,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass  # the info file is only for messages
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            _unlock(handle)
        except OSError:
            pass
        handle.close()
        try:
            self.info_path.unlink()
        except OSError:
            pass

    def _holder(self) -> str:
        try:
            info = json.loads(self.info_path.read_text(encoding="utf-8"))
            return f" ({info['role']}, process id {info['pid']}, started {info['started_utc']})"
        except (OSError, ValueError, KeyError, TypeError):
            return ""

    def __enter__(self) -> "ProcessLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


if sys.platform == "win32":
    import msvcrt

    def _lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
