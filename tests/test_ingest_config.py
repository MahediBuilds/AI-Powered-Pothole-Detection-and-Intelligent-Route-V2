import pytest

from roadsense.config import AppConfig, ConfigError, IngestConfig, load_app_config, project_root

ROOT = project_root()


def test_defaults_when_ingest_section_absent(write_yaml):
    cfg = load_app_config(write_yaml("a.yaml", {"data_dir": "d"}))
    assert cfg.ingest == IngestConfig()
    assert (cfg.ingest.max_gps_accuracy_m, cfg.ingest.max_gps_gap_s) == (25.0, 3.0)
    assert cfg.ingest.max_gps_extrapolation_s == 1.0
    assert AppConfig().ingest == IngestConfig()


def test_shipped_app_yaml_has_ingest_section_with_expected_values():
    cfg = load_app_config(ROOT / "configs" / "app.yaml")
    assert cfg.ingest.max_gps_accuracy_m == 25.0
    assert cfg.ingest.max_gps_gap_s == 3.0
    assert cfg.ingest.max_gps_extrapolation_s == 1.0
    text = (ROOT / "configs" / "app.yaml").read_text(encoding="utf-8")
    assert "ingest:" in text and "PROVISIONAL" in text


def test_partial_ingest_section_keeps_other_defaults(write_yaml):
    cfg = load_app_config(write_yaml("a.yaml", {"ingest": {"max_gps_gap_s": 5}}))
    assert cfg.ingest.max_gps_gap_s == 5
    assert cfg.ingest.max_gps_accuracy_m == 25.0


def test_extrapolation_may_be_zero(write_yaml):
    cfg = load_app_config(write_yaml("a.yaml", {"ingest": {"max_gps_extrapolation_s": 0}}))
    assert cfg.ingest.max_gps_extrapolation_s == 0


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ({"max_gps_accuracy_m": 0}, "max_gps_accuracy_m"),
        ({"max_gps_accuracy_m": -5.0}, "max_gps_accuracy_m"),
        ({"max_gps_accuracy_m": ".inf"}, "max_gps_accuracy_m"),
        ({"max_gps_accuracy_m": "25"}, "max_gps_accuracy_m"),
        ({"max_gps_accuracy_m": True}, "max_gps_accuracy_m"),
        ({"max_gps_gap_s": 0}, "max_gps_gap_s"),
        ({"max_gps_gap_s": -1}, "max_gps_gap_s"),
        ({"max_gps_gap_s": ".nan"}, "max_gps_gap_s"),
        ({"max_gps_extrapolation_s": -0.1}, "max_gps_extrapolation_s"),
        ({"max_gps_extrapolation_s": ".inf"}, "max_gps_extrapolation_s"),
        ({"max_gps_extrapolation_s": None}, "max_gps_extrapolation_s"),
        ({"max_gps_speed": 3}, "max_gps_speed"),
    ],
)
def test_ingest_validation_names_file_and_field(write_yaml, section, field):
    path = write_yaml("a.yaml", {"ingest": section})
    with pytest.raises(ConfigError) as exc:
        load_app_config(path)
    assert str(path) in str(exc.value)
    assert f"ingest.{field}" in str(exc.value)
