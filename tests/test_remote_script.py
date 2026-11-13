"""Static checks of scripts/vast_remote_unified.sh (the lifecycle was run on Linux with a fake trainer)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vast_remote_unified.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")


def test_script_uses_lf_line_endings():
    assert b"\r\n" not in SCRIPT.read_bytes()  # a CRLF script fails on Linux: 'set: pipefail: invalid option name'


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_script_parses():
    assert subprocess.run(["bash", "-n"], input=SCRIPT.read_bytes(), capture_output=True).returncode == 0


def test_train_stage_passes_a_computed_time_cap():
    assert "time_cap_hours" in TEXT and "--time-hours \"$hours\"" in TEXT and "--time-hours $hours" in TEXT
    assert "($TOTAL_LIMIT_MIN - $1 - $TIME_MARGIN_MIN) / 60" in TEXT
    assert re.search(r'TIME_CAP_H="\$\{TIME_CAP_H:-2\.5\}"', TEXT)
    assert re.search(r'TIME_MARGIN_MIN="\$\{TIME_MARGIN_MIN:-12\}"', TEXT)
    assert re.search(r'TOTAL_LIMIT_MIN="\$\{TOTAL_LIMIT_MIN:-200\}"', TEXT)


def test_watchdog_does_not_start_python_every_poll():
    body = TEXT[TEXT.index("stage_watchdog() {"):TEXT.index('stage="${1:-}"')]
    assert "export_new_epochs" in body and "export_checkpoints" not in body
    helper = TEXT[TEXT.index("export_new_epochs() {"):TEXT.index("stage_pack() {")]
    assert "--only-epochs" in helper and "nice -n 10" in helper
    assert helper.index("need") < helper.index("export_checkpoints.py")  # the cheap shell check comes first


def test_selfdestruct_is_opt_in_documented_and_never_earlier_than_the_hard_limit():
    assert 'SELFDESTRUCT_AT_MIN="${SELFDESTRUCT_AT_MIN:-205}"' in TEXT
    assert '[ "${SELFDESTRUCT:-0}" = 1 ] || return 0' in TEXT  # off unless SELFDESTRUCT=1
    assert '"$SELFDESTRUCT_AT_MIN" -lt "$TOTAL_LIMIT_MIN"' in TEXT  # refuses an earlier deadline
    assert "vastai destroy instance \"$CONTAINER_ID\" --api-key \"$CONTAINER_API_KEY\"" in TEXT
    assert "selfdestruct --check" in TEXT and "--cancel" in TEXT
    # the key is never echoed
    assert not re.search(r'echo[^\n]*\$\{?CONTAINER_API_KEY\}?(?!\w)[^\n]*\n', TEXT.replace("${#CONTAINER_API_KEY}", ""))
    # the watchdog destroys only after packing
    wd = TEXT[TEXT.index("stage_watchdog() {"):TEXT.index('stage="${1:-}"')]
    assert wd.count("stage_pack\n            wd_after_pack") == 2 and "selfdestruct_now" not in wd
