"""Run the web application: ``python -m roadsense.api [--config configs/app.yaml]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from roadsense.config import ConfigError, load_app_config, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m roadsense.api",
        description="Serve the RoadSense API and web interface (one process, one worker). "
        "'roadsense.cli add', 'add-dir' and 'reset' (and the demo script) are refused while "
        "this server runs; 'rebuild-map' and 'list' are safe at any time.",
    )
    parser.add_argument(
        "--config", default=str(project_root() / "configs" / "app.yaml"), help="app config YAML"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_app_config(Path(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Fail early and readably if another server (or a CLI run) is using this database;
    # the application takes the same lock for good when it starts.
    from roadsense.storage.proclock import AlreadyRunningError, ProcessLock, lock_path_for

    try:
        ProcessLock(lock_path_for(settings.db_path)).acquire().release()
    except AlreadyRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    from roadsense.api.app import create_app

    print(f"RoadSense: http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
