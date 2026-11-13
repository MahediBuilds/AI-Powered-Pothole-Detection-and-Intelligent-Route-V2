"""Tests for scripts/select_checkpoint.py (the rule only; no model evaluation is run)."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "select_checkpoint", Path(__file__).resolve().parents[1] / "scripts" / "select_checkpoint.py"
)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

HEADER = "epoch,time,metrics/mAP50(B),metrics/mAP50-95(B)\n"


def make_run(tmp_path, names, fitness_rows):
    weights = tmp_path / "weights"
    weights.mkdir()
    for n in names:
        (weights / n).write_bytes(n.encode())
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(HEADER + "".join(f"{e},{e},{m50},{m5095}\n" for e, m50, m5095 in fitness_rows))
    return weights, csv_path


def make_eval_yaml(tmp_path, name, list_name):
    (tmp_path / "lists").mkdir(exist_ok=True)
    (tmp_path / "lists" / list_name).write_text("images/a.jpg\n")
    path = tmp_path / name
    path.write_text(f"train: lists/{list_name}\nval: lists/{list_name}\ntest: lists/{list_name}\nnames:\n  0: pothole\n")
    return path


def args(tmp_path, weights, csv_path, scores, **kw):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores))
    rdd = make_eval_yaml(tmp_path, "rdd2022_val.yaml", "rdd2022_val.txt")
    hrp = make_eval_yaml(tmp_path, "hrp4k_val.yaml", "hrp4k_val.txt")
    return sc.argparse.Namespace(weights_dir=weights, results_csv=csv_path, out=tmp_path / "out",
                                 rdd_yaml=rdd, hrp_yaml=hrp, imgsz=1280, batch=8, device="0",
                                 scores_json=scores_path, **kw)


ROWS = [(1, 0.30, 0.10), (2, 0.40, 0.15), (3, 0.45, 0.20), (4, 0.44, 0.19), (5, 0.43, 0.18), (6, 0.42, 0.17)]


def S(rdd, hrp):
    return {"rdd2022": rdd, "hrp4k": hrp}


def test_highest_mean_of_per_source_map50_wins(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch0.pt", "epoch2.pt", "epoch4.pt", "best.pt", "last.pt"], ROWS)
    scores = {"epoch0.pt": S(0.30, 0.20), "epoch2.pt": S(0.50, 0.40),  # mean 0.45
              "epoch4.pt": S(0.60, 0.20),                              # mean 0.40, best RDD only
              "best.pt": S(0.50, 0.40), "last.pt": S(0.44, 0.40)}
    result = sc.run(args(tmp_path, weights, csv_path, scores))
    assert result["chosen"]["name"] == "epoch2.pt"
    assert result["chosen"]["mean_map50"] == pytest.approx(0.45)
    assert result["reads_test_lists"] is False
    assert (tmp_path / "out" / "selection.json").is_file()


def test_tie_goes_to_lower_epoch_even_if_listed_later(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch1.pt", "epoch3.pt", "epoch5.pt"], ROWS)
    scores = {"epoch1.pt": S(0.40, 0.30), "epoch3.pt": S(0.40001, 0.30001), "epoch5.pt": S(0.40, 0.30)}
    assert sc.run(args(tmp_path, weights, csv_path, scores))["chosen"]["name"] == "epoch1.pt"


def test_difference_beyond_tie_resolution_is_not_a_tie(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch1.pt", "epoch3.pt"], ROWS)
    scores = {"epoch1.pt": S(0.40, 0.30), "epoch3.pt": S(0.4003, 0.3003)}
    assert sc.run(args(tmp_path, weights, csv_path, scores))["chosen"]["name"] == "epoch3.pt"


def test_best_and_last_are_placed_at_their_epoch_and_epochn_wins_on_equal_epoch(tmp_path):
    # fitness peaks at epoch index 2 (row 3); last is index 5
    weights, csv_path = make_run(tmp_path, ["epoch2.pt", "best.pt", "last.pt"], ROWS)
    cands = {c["name"]: c for c in sc.candidate_epochs(weights, csv_path)}
    assert cands["best.pt"]["epoch"] == 2 and cands["last.pt"]["epoch"] == 5 and cands["epoch2.pt"]["epoch"] == 2
    scores = {n: S(0.5, 0.5) for n in cands}
    assert sc.run(args(tmp_path, weights, csv_path, scores))["chosen"]["name"] == "epoch2.pt"


def test_last_wins_only_if_strictly_better(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch2.pt", "last.pt"], ROWS)
    scores = {"epoch2.pt": S(0.5, 0.5), "last.pt": S(0.51, 0.5)}
    assert sc.run(args(tmp_path, weights, csv_path, scores))["chosen"]["name"] == "last.pt"


def test_resumed_run_with_repeated_epoch_rows(tmp_path):
    rows = ROWS[:3] + [(3, 0.50, 0.30)] + ROWS[3:]  # epoch 3 appears twice; the later row wins
    assert sc.results_epochs(make_run(tmp_path, ["last.pt"], rows)[1])[2] == pytest.approx(0.30)  # fitness is mAP50-95 alone in Ultralytics 8.4.210


def test_identical_weights_are_evaluated_once(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch2.pt", "best.pt", "last.pt"], ROWS)
    calls = []

    def evaluate(cand, source):
        calls.append((cand["name"], source))
        return 0.5

    cands = sc.candidate_epochs(weights, csv_path)
    same = {"epoch2.pt": "A", "best.pt": "A", "last.pt": "B"}
    chosen, rows = sc.select(cands, evaluate, dedupe=lambda p: same[p.name])
    assert sorted(calls) == [("best.pt", "hrp4k"), ("best.pt", "rdd2022"), ("last.pt", "hrp4k"), ("last.pt", "rdd2022")]
    assert {r["name"]: r["same_weights_as"] for r in rows}["epoch2.pt"] == "best.pt"
    assert chosen["name"] == "epoch2.pt"  # same score, same epoch as best.pt: epochN.pt wins


@pytest.mark.parametrize("name", ["rdd2022_test.yaml", "test_eval.yaml"])
def test_test_yaml_names_are_refused(tmp_path, name):
    path = make_eval_yaml(tmp_path, name, "rdd2022_val.txt")
    with pytest.raises(sc.SelectionError, match="test"):
        sc.assert_val_only(path)


def test_yaml_pointing_at_a_test_list_is_refused(tmp_path):
    path = make_eval_yaml(tmp_path, "eval.yaml", "rdd2022_test.txt")
    with pytest.raises(sc.SelectionError, match="test location"):
        sc.assert_val_only(path)
    (tmp_path / "lists2" / "test").mkdir(parents=True)
    path.write_text("train: lists2/test/a.txt\nval: lists2/test/a.txt\nnames:\n  0: pothole\n")
    with pytest.raises(sc.SelectionError, match="test location"):
        sc.assert_val_only(path)


def test_attest_is_not_mistaken_for_test(tmp_path):
    path = make_eval_yaml(tmp_path, "latest_val.yaml", "greatest_val.txt")
    assert sc.assert_val_only(path)  # 'latest' and 'greatest' contain the letters but are not test lists


def test_run_refuses_test_yaml_before_doing_anything(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch0.pt"], ROWS)
    a = args(tmp_path, weights, csv_path, {"epoch0.pt": S(0.1, 0.1)})
    a.rdd_yaml = make_eval_yaml(tmp_path, "rdd2022_test.yaml", "rdd2022_test.txt")
    with pytest.raises(sc.SelectionError):
        sc.run(a)
    assert not (tmp_path / "out" / "selection.json").exists()


def test_unexpected_weight_file_refused(tmp_path):
    weights, csv_path = make_run(tmp_path, ["epoch0.pt", "notes.pt"], ROWS)
    with pytest.raises(sc.SelectionError, match="unexpected"):
        sc.candidate_epochs(weights, csv_path)


def test_eval_command_is_val_only_at_1280(tmp_path):
    cmd = sc.eval_command(Path("c.pt"), Path("d.yaml"), Path("o"), 1280, 8, "0")
    assert cmd[cmd.index("--split") + 1] == "val" and cmd[cmd.index("--imgsz") + 1] == "1280"
    assert cmd[cmd.index("--mode") + 1] == "select" and "test" not in " ".join(cmd).replace("select", "")


def test_default_eval_yamls_if_present_are_val_only():
    for path in (sc.DEFAULT_RDD_YAML, sc.DEFAULT_HRP_YAML):
        if path.is_file():
            sc.assert_val_only(path)


def test_read_map50_reads_threshold_free_block(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"threshold_free": {"mAP50": 0.4321, "mAP50_95": 0.2}}))
    assert sc.read_map50(tmp_path) == pytest.approx(0.4321)


def test_best_pt_holds_the_last_epoch_tied_on_map50_95(tmp_path):
    # mAP50-95 ties at the top in rows 2 and 4 (indices 1 and 3); Ultralytics rewrites best.pt on equality,
    # so best.pt is the LAST tied epoch. mAP50 differs and must not matter (fitness is mAP50-95 alone).
    rows = [(1, 0.30, 0.10), (2, 0.60, 0.25), (3, 0.40, 0.20), (4, 0.35, 0.25), (5, 0.45, 0.22)]
    weights, csv_path = make_run(tmp_path, ["best.pt", "last.pt"], rows)
    cands = {c["name"]: c for c in sc.candidate_epochs(weights, csv_path)}
    assert cands["best.pt"]["epoch"] == 3 and cands["last.pt"]["epoch"] == 4


def test_csv_rounding_noise_does_not_break_the_tie(tmp_path):
    rows = [(1, 0.3, 0.250004), (2, 0.3, 0.24), (3, 0.3, 0.25)]
    weights, csv_path = make_run(tmp_path, ["best.pt"], rows)
    assert sc.candidate_epochs(weights, csv_path)[0]["epoch"] == 2


def test_best_pt_epoch_is_taken_from_identical_weights_when_available(tmp_path):
    weights, csv_path = make_run(tmp_path, ["best.pt", "epoch1.pt", "epoch4.pt"], ROWS)
    cands = sc.candidate_epochs(weights, csv_path)
    fingerprints = {"best.pt": "X", "epoch1.pt": "X", "epoch4.pt": "Y"}
    sc.refine_best_epoch(cands, lambda path: fingerprints[path.name])
    assert {c["name"]: c["epoch"] for c in cands}["best.pt"] == 1


def _val_yaml_with_list(tmp_path, body):
    (tmp_path / "lists").mkdir(exist_ok=True)
    (tmp_path / "lists" / "rdd2022_val.txt").write_text(body)
    path = tmp_path / "ok_name.yaml"
    path.write_text("train: lists/rdd2022_val.txt\nval: lists/rdd2022_val.txt\nnames:\n  0: pothole\n")
    return path


def test_list_content_with_a_test_component_is_refused(tmp_path):
    path = _val_yaml_with_list(tmp_path, "./../images/rdd2022/val/a.jpg\n./../images/rdd2022/test/b.jpg\n")
    with pytest.raises(sc.SelectionError, match="test component"):
        sc.assert_val_only(path)
    for bad in ("images/hrp4k/Test/x.jpg", "C:/data/test/x.jpg", "a" + chr(92) + "test" + chr(92) + "x.jpg"):
        path = _val_yaml_with_list(tmp_path, bad + "\n")
        with pytest.raises(sc.SelectionError, match="test component"):
            sc.assert_val_only(path)


def test_list_content_check_accepts_val_lists_and_test_like_names(tmp_path):
    path = _val_yaml_with_list(
        tmp_path, "./../images/rdd2022/val/test_001.jpg\n./../images/hrp4k/val/latest/hrp4k_val_1.jpg\n")
    assert sc.assert_val_only(path)  # a file NAME containing "test" is not a test directory
