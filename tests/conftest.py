from pathlib import Path

import pytest
import yaml


@pytest.fixture
def write_yaml(tmp_path):
    """Write a mapping to a yaml file in tmp_path and return its path."""

    def _write(name: str, data) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def pothole_dataset_yaml(tmp_path):
    """A valid single-class dataset yaml with non-empty train and val image directories."""
    ds_dir = tmp_path / "dataset"
    for split in ("train", "val"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "images" / split / "img.jpg").write_bytes(b"x")
    path = ds_dir / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(ds_dir), "train": "images/train", "val": "images/val", "names": {0: "pothole"}}
        ),
        encoding="utf-8",
    )
    return path


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: uses the real trained detector checkpoint (deselect with -m 'not slow')"
    )


def pytest_collection_modifyitems(config, items):
    """Run tests that import ultralytics last.

    Some older tests assert that ultralytics is not yet imported (the training
    dry run must not need it). Importing it is irreversible within a process, so
    the detector tests and real-model tests go to the end. The sort is stable.
    """

    def imports_ultralytics(item):
        return item.get_closest_marker("slow") is not None or item.module.__name__ == "test_detector"

    items.sort(key=imports_ultralytics)


@pytest.fixture
def one_pothole_video(tmp_path):
    """A 30-frame video in which the fake detector tracks one pothole (last seen at 0.9 s)."""
    from pipeline_fixtures import moving_object, write_object_video

    return write_object_video(tmp_path, 30, [moving_object(0, 11)])


@pytest.fixture
def make_client(tmp_path):
    """Factory for TestClients of a real app (lifespan running) on a temporary config.

    ``make_client(graph=..., detector_factory=..., raise_server_exceptions=True,
    **config_sections)``; the client has ``.cfg``. The fake detector is the default.
    """
    from contextlib import ExitStack

    from api_fixtures import make_api_cfg
    from fastapi.testclient import TestClient
    from pipeline_fixtures import FakeDetector

    from roadsense.api.app import create_app

    stack = ExitStack()

    def make(*, graph=None, detector_factory=None, raise_server_exceptions=True, **sections):
        cfg = make_api_cfg(tmp_path, **sections)
        app = create_app(cfg, detector_factory=detector_factory or FakeDetector, graph=graph)
        client = stack.enter_context(
            TestClient(app, raise_server_exceptions=raise_server_exceptions)
        )
        client.cfg = cfg
        return client

    yield make
    stack.close()
