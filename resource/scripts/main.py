#!/usr/bin/env python3
"""Single entry point for the availability calibration pipeline."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent          # scripts
ROOT = SCRIPTS_DIR.parent                               # repo root (…/resource); results/ paths resolve here
PY = sys.executable
sys.path.insert(0, str(SCRIPTS_DIR))
import _core


def _stage(script: str, *args):
    cmd = [PY, str(SCRIPTS_DIR / script), *[str(a) for a in args]]
    print("  $ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


# ══ The general method chain (real / user data) ═══════════════════════════════════════════════
def calibrate(run_dir: str, groups: str, rows: str, *,
              seeds: int, resims: int, figures: bool = False) -> None:
    """The chain: feature extraction, Q⁺ training, diagnosis, calibration, evaluation."""
    _stage("estimation.py", "--run-dir", run_dir, "--real", "--n-real-seeds", seeds,
           "--real-rows", rows, "--real-groups", groups)
    _stage("diagnosis.py", "--run-dir", run_dir, "--real", "--n-real-seeds", seeds)
    _stage("calibration.py", "--run-dir", run_dir, "--real",
           "--n-real-seeds", seeds, "--n-real-runs", resims)
    # Aggregate per-seed metrics into real_per_seed.csv (what --report-only reads).
    _stage("evaluation.py", "--run-dir", run_dir, "--n-real-seeds", seeds)
    if figures:
        _stage("reports.py", "--run-dir", run_dir, "--n-real-seeds", seeds)
    print("\n[calibrate] CED/CWD improvement% (median over seeds):", flush=True)
    _stage("evaluation.py", "--run-dir", run_dir, "--report-only")


# ══ Source providers: build (groups, rows) + the calibration manifest under run_dir ═══════════
# Manifest (candidate to config_json) that build_real_groups() produces.
REAL_SOURCE_MANIFEST = ROOT / "results" / "assets" / "real_source" / "real_common" / "manifest.json"


def source_from_real(run_dir: str) -> tuple[str, str]:
    """Reproduction preset: build real groups + manifest from the regenerated ρ (copies the manifest into <run>/configs/real/ for calibration --real)."""
    print("  building the real group cache via _core.build_real_groups() …", flush=True)
    groups, rows = _core.build_real_groups(run_id="repro_real_groups")
    dst = ROOT / run_dir / "configs" / "real" / "manifest.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_SOURCE_MANIFEST, dst)
    return str(groups), str(rows)


# ══ Entry points ══════════════════════════════════════════════════════════════════════════════
def run_real(seeds: int, resims: int) -> None:
    """Reproduction preset over calibrate(): regenerated-rho real source."""
    print(f"\n=== Real-world reproduction ({seeds} seeds x {resims} re-sims) ===", flush=True)
    run = "results/real"
    groups, rows = source_from_real(run)
    calibrate(run, groups, rows, seeds=seeds, resims=resims, figures=True)
    print(f"[real] figure data: {run}/report/", flush=True)


def run_synthetic(seeds: int, resims: int, logs: str) -> None:
    """Synthetic reproduction harness (generates its own ground truth)."""
    print(f"\n=== Synthetic reproduction ({seeds} seeds x {resims} runs/setting) ===", flush=True)
    run = "results/synthetic"
    os.environ["SYNTH_LOG_STORE"] = logs
    _stage("preparation.py", "--run-dir", run, "--synthetic",
           "--n-synth-seeds", seeds, "--n-synth-runs", resims)
    _stage("estimation.py", "--run-dir", run, "--synthetic", "--n-synth-seeds", seeds)
    _stage("diagnosis.py", "--run-dir", run, "--synthetic", "--n-synth-seeds", seeds)
    _stage("calibration.py", "--run-dir", run, "--synthetic",
           "--n-synth-seeds", seeds, "--n-synth-runs", resims)
    _stage("evaluation.py", "--run-dir", run, "--n-synth-seeds", seeds)
    _stage("reports.py", "--run-dir", run, "--n-synth-seeds", seeds)
    print("\n[synthetic] CED/CWD/CTD improvement% (median over seeds):", flush=True)
    _stage("evaluation.py", "--run-dir", run, "--report-only")
    print(f"[synthetic] figure data: {run}/report/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--synthetic", action="store_true", help="synthetic validation harness")
    ap.add_argument("--real", action="store_true", help="real path with ρ regenerated from the shipped inputs")
    ap.add_argument("--seeds", type=int, default=10,
                    help="calibration and report seeds (default 10, the reported run)")
    ap.add_argument("--resims", type=int, default=5,
                    help="simulation runs per setting, and P_cal re-sims per seed (default 5)")
    ap.add_argument("--logs", help="where the synthetic logs are, data/synthetic/logs for "
                                   "the ones this repository ships")
    args = ap.parse_args()

    if not (args.synthetic or args.real):
        ap.error("pass --synthetic or --real to reproduce the paper")
    if args.synthetic:
        if not args.logs:
            ap.error("--synthetic needs --logs, for instance --logs data/synthetic/logs")
        run_synthetic(seeds=args.seeds, resims=args.resims, logs=args.logs)
    if args.real:
        run_real(seeds=args.seeds, resims=args.resims)
    print("\n[main] done.", flush=True)


if __name__ == "__main__":
    main()
