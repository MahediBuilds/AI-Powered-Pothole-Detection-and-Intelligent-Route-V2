"""Pre-declared checkpoint selection for the unified_v1 run. Local, after the cloud run.

    python scripts/select_checkpoint.py --weights-dir runs/cloud/unified_v1/weights \
        --results-csv runs/cloud/unified_v1/results.csv --out runs/select/unified_v1 \
        [--rdd-yaml data/processed/unified_v1/eval/rdd2022_val.yaml] \
        [--hrp-yaml data/processed/unified_v1/eval/hrp4k_val.yaml] [--imgsz 1280] [--batch 8]

THE RULE (declared before the run; do not edit after seeing results):
  1. Candidates: every checkpoint in --weights-dir: epochN.pt (saved every 5 epochs; N is the 0-based
     epoch index, so epoch4.pt is the 5th epoch), best.pt and last.pt.
  2. Score: the mean of two mAP50 values, each measured separately at imgsz 1280 with
     scripts/eval_checkpoint.py on a VALIDATION list: RDD2022 val and HRP4K val.
  3. Pick the highest score. Scores that agree to 4 decimals are tied; a tie goes to the LOWER epoch
     (best.pt and last.pt are placed at the epoch they hold and, on an equal epoch, the epochN.pt file
     wins, so the choice is deterministic). Ultralytics 8.4.210 picks best.pt on FITNESS = mAP50-95
     alone (weights [0, 0, 0, 1], not the older 0.1*mAP50 + 0.9*mAP50-95) and, because it rewrites
     best.pt whenever best_fitness == fitness, best.pt holds the LAST epoch among those tied at the
     top. best.pt is therefore placed at the last epoch within 1e-5 of the maximum mAP50-95 in
     results.csv (the csv keeps 5 significant digits), unless its weights are identical to an
     epochN.pt, in which case that epoch is used.
  4. Test lists are never read: only val yaml/list files are accepted (refused if a path or a listed
     file name contains "test", or if ANY image path inside a list has a directory component named
     "test"), and the evaluation split is fixed to val.
Checkpoints holding identical weights (typically best.pt and one epochN.pt) are evaluated once.

Writes OUT/selection.json (rule, every candidate's scores, the choice, sha256) and prints a table.
``--scores-json FILE`` supplies precomputed scores {name: {"rdd2022": x, "hrp4k": y}} instead of
running the evaluations (used by the tests). Existing OUT/<name>/<source>/metrics.json files for the
same checkpoint sha256 are reused, so an interrupted selection can be re-run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_checkpoint.py"
DEFAULT_RDD_YAML = ROOT / "data" / "processed" / "unified_v1" / "eval" / "rdd2022_val.yaml"
DEFAULT_HRP_YAML = ROOT / "data" / "processed" / "unified_v1" / "eval" / "hrp4k_val.yaml"
SOURCES = ("rdd2022", "hrp4k")
TIE_DECIMALS = 4
RULE_TEXT = (
    "highest mean of per-source val mAP50 (RDD2022 val, HRP4K val; imgsz 1280); "
    "ties (equal to 4 decimals) go to the lower epoch; test lists are never read"
)


class SelectionError(Exception):
    pass


# ----------------------------------------------------------------------------- guards
def _is_test_name(text: str) -> bool:
    return re.search(r"(^|[^a-z])test([^a-z]|$)", text.lower()) is not None


def assert_list_has_no_test_paths(list_path: Path) -> None:
    """Content check: refuse a list in which any image path has a directory component named ``test``."""
    if not list_path.is_file():
        return
    for line in list_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip().replace(chr(92), "/")
        if entry and any(part.lower() == "test" for part in Path(entry).parts[:-1]):
            raise SelectionError(f"{list_path}: image path has a test component: {entry}")


def assert_val_only(yaml_path: Path) -> list[Path]:
    """Refuse any dataset yaml that is, or points at, a test list, and any list whose CONTENT has a
    test path component. Returns the list files it names."""
    if _is_test_name(yaml_path.name):
        raise SelectionError(f"refusing a test dataset yaml: {yaml_path}")
    ds = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = (yaml_path.parent / ds["path"]).resolve() if ds.get("path") else yaml_path.parent.resolve()
    lists = []
    for key in ("train", "val", "test"):
        for entry in ([ds[key]] if isinstance(ds.get(key), str) else ds.get(key) or []):
            if _is_test_name(Path(entry).name) or "test" in (p.lower() for p in Path(entry).parts[:-1]):
                raise SelectionError(f"{yaml_path}: '{key}' entry refers to a test location: {entry}")
            lists.append((root / entry).resolve())
    for list_path in lists:
        assert_list_has_no_test_paths(list_path)
    return lists


# ----------------------------------------------------------------------------- candidates
FITNESS_TIE_TOL = 1e-5  # results.csv keeps 5 significant digits


def results_epochs(results_csv: Path) -> dict[int, float]:
    """{0-based epoch: Ultralytics 8.4.210 fitness (= mAP50-95)} from results.csv.

    A resumed run may repeat an epoch (the last row wins)."""
    rows: dict[int, float] = {}
    with open(results_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader((ln.replace(" ", "") if i == 0 else ln for i, ln in enumerate(fh)))
        for r in reader:
            e = int(float(r["epoch"])) - 1
            rows[e] = float(r["metrics/mAP50-95(B)"])
    if not rows:
        raise SelectionError(f"no epochs in {results_csv}")
    return rows


def candidate_epochs(weights_dir: Path, results_csv: Path) -> list[dict]:
    fitness = results_epochs(results_csv)
    top = max(fitness.values())
    # Ultralytics rewrites best.pt whenever best_fitness == fitness, so on a tie the LAST tied epoch holds it.
    best_epoch = max(e for e, f in fitness.items() if f >= top - FITNESS_TIE_TOL)
    last_epoch = max(fitness)
    out = []
    for path in sorted(weights_dir.glob("*.pt")):
        m = re.fullmatch(r"epoch(\d+)\.pt", path.name)
        if m:
            epoch, kind = int(m.group(1)), 0
        elif path.name == "best.pt":
            epoch, kind = best_epoch, 1
        elif path.name == "last.pt":
            epoch, kind = last_epoch, 1
        else:
            raise SelectionError(f"unexpected file in the weights directory: {path.name}")
        out.append({"name": path.name, "path": path, "epoch": epoch, "kind": kind})
    if not out:
        raise SelectionError(f"no checkpoints in {weights_dir}")
    return out


def refine_best_epoch(cands: list[dict], fingerprint: Callable) -> None:
    """If best.pt's weights equal an epochN.pt's, label best.pt with that epoch (more exact than results.csv)."""
    best = next((c for c in cands if c["name"] == "best.pt"), None)
    if best is None:
        return
    fp = fingerprint(best["path"])
    for c in cands:
        if c["kind"] == 0 and fingerprint(c["path"]) == fp:
            best["epoch"] = c["epoch"]
            return


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_fingerprint(path: Path) -> str:
    """Hash of the model tensors only (metadata such as epoch or date is ignored)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    h = hashlib.sha256()
    for key, tensor in sorted(model.float().state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- evaluation
def eval_command(ckpt: Path, data_yaml: Path, out: Path, imgsz: int, batch: int, device: str) -> list[str]:
    """The ONE place that knows scripts/eval_checkpoint.py's command line."""
    return [sys.executable, str(EVAL_SCRIPT), "--mode", "select", "--checkpoint", str(ckpt),
            "--data", str(data_yaml), "--split", "val", "--imgsz", str(imgsz), "--batch", str(batch),
            "--device", device, "--out", str(out), "--no-plots"]


def read_map50(out: Path) -> float:
    """The ONE place that knows where eval_checkpoint.py writes mAP50 (metrics.json, threshold_free)."""
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return float(metrics["threshold_free"]["mAP50"])


def make_evaluator(out_dir: Path, yamls: dict[str, Path], imgsz: int, batch: int, device: str) -> Callable:
    def evaluate(cand: dict, source: str) -> float:
        out = out_dir / "evals" / Path(cand["name"]).stem / source
        cfg = out / "eval_config.json"
        sha = sha256_file(cand["path"])
        if (out / "metrics.json").is_file() and cfg.is_file():
            if json.loads(cfg.read_text(encoding="utf-8")).get("checkpoint_sha256") == sha:
                return read_map50(out)
        out.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(eval_command(cand["path"], yamls[source], out, imgsz, batch, device),
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise SelectionError(f"evaluation failed for {cand['name']} on {source}:\n{res.stderr[-2000:]}")
        return read_map50(out)

    return evaluate


# ----------------------------------------------------------------------------- the rule
def choose(rows: list[dict]) -> dict:
    """Apply the rule to rows with ``scores`` {source: mAP50}, ``epoch`` and ``kind`` (0 epochN, 1 best/last)."""
    if not rows:
        raise SelectionError("no candidates")
    for r in rows:
        r["mean_map50"] = sum(r["scores"][s] for s in SOURCES) / len(SOURCES)
        r["tie_key"] = round(r["mean_map50"], TIE_DECIMALS)
    return min(rows, key=lambda r: (-r["tie_key"], r["epoch"], r["kind"], r["name"]))


def select(cands: list[dict], evaluate: Callable, dedupe: Callable | None = None) -> tuple[dict, list[dict]]:
    groups: dict[str, dict] = {}
    for c in cands:
        key = dedupe(c["path"]) if dedupe else c["name"]
        groups.setdefault(key, c)  # first (sorted) name evaluates; identical weights share its scores
    scores_by_key: dict[str, dict] = {}
    for key, c in groups.items():
        scores_by_key[key] = {s: evaluate(c, s) for s in SOURCES}
        print(f"{c['name']:>12}  " + "  ".join(f"{s} {scores_by_key[key][s]:.4f}" for s in SOURCES), flush=True)
    rows = []
    for c in cands:
        key = dedupe(c["path"]) if dedupe else c["name"]
        rows.append({"name": c["name"], "epoch": c["epoch"], "kind": c["kind"],
                     "scores": scores_by_key[key], "same_weights_as": groups[key]["name"],
                     "path": str(c["path"])})
    return choose(rows), rows


def run(args: argparse.Namespace) -> dict:
    yamls = {"rdd2022": Path(args.rdd_yaml), "hrp4k": Path(args.hrp_yaml)}
    listed = {s: assert_val_only(p) for s, p in yamls.items()}
    cands = candidate_epochs(Path(args.weights_dir), Path(args.results_csv))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedupe = None
    if args.scores_json:
        table = json.loads(Path(args.scores_json).read_text(encoding="utf-8"))
        evaluate = lambda cand, source: float(table[cand["name"]][source])  # noqa: E731
    else:
        evaluate = make_evaluator(out_dir, yamls, args.imgsz, args.batch, args.device)
        dedupe = weights_fingerprint
        refine_best_epoch(cands, weights_fingerprint)
    chosen, rows = select(cands, evaluate, dedupe)
    result = {
        "rule": RULE_TEXT, "tie_decimals": TIE_DECIMALS, "imgsz": args.imgsz,
        "eval_data": {s: str(p) for s, p in yamls.items()},
        "eval_lists_sha256": {s: [sha256_file(p) for p in lst if p.is_file()] for s, lst in listed.items()},
        "reads_test_lists": False,
        "chosen": {"name": chosen["name"], "epoch_0based": chosen["epoch"],
                   "epoch_1based": chosen["epoch"] + 1, "mean_map50": chosen["mean_map50"],
                   "scores": chosen["scores"], "path": chosen["path"],
                   "sha256": sha256_file(Path(chosen["path"])) if Path(chosen["path"]).is_file() else None},
        "candidates": sorted(rows, key=lambda r: (r["epoch"], r["kind"], r["name"])),
    }
    (out_dir / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nCHOSEN: {chosen['name']} (epoch {chosen['epoch'] + 1}), mean mAP50 {chosen['mean_map50']:.4f}")
    print(f"written: {out_dir / 'selection.json'}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rdd-yaml", type=Path, default=DEFAULT_RDD_YAML)
    ap.add_argument("--hrp-yaml", type=Path, default=DEFAULT_HRP_YAML)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--scores-json", type=Path, help="precomputed scores instead of running evaluations")
    ns = ap.parse_args(argv)
    try:
        run(ns)
    except (SelectionError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
