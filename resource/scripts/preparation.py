#!/usr/bin/env python3
"""Load the frozen synthetic configs and run the baseline simulation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _core import (
    run_seeds,
    synth_log_store,
    ROOT,
    log,
    resolve_cfg,
    run_extract,
    simulate_synthetic_config,
)

# Overridable so an experiment can use its own config set and log store without
# touching the shipped ones.
SYNTH_CONFIGS   = Path(os.environ.get("SYNTH_CONFIGS",   ROOT / "data" / "synthetic" / "configs_deviating_3_of_5"))
# Resolved per call, so an unset SYNTH_LOG_STORE fails where it is used.


def load_synth_configs(run_dir: Path) -> list[dict]:
    """Return the synth manifest items and copy the manifest into run_dir/configs/synth/ for the downstream stages."""
    src = SYNTH_CONFIGS / "manifest.json"
    if not src.exists():
        sys.exit(f"[ERROR] {src} not found. The synthetic configs ship with the repository; "
                 f"restore them with `git checkout -- data/synthetic/`.")
    dst = run_dir / "configs" / "synth"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "manifest.json")
    return json.loads(src.read_text(encoding="utf-8"))["manifest"]


# ── Baseline synthetic simulation ──────────────────────────────────────────────
def run_simulation_and_extract(out_seed_dir: Path, manifest_items, num_runs: int,
                                bin_minutes: int = 60) -> None:
    """Simulate {P,P_TRUE} for all scenarios and extract rho profiles under out_seed_dir."""
    rows = []
    for item in manifest_items:
        rows.extend(
            simulate_synthetic_config(
                resolve_cfg(item["config_json"]),
                out_seed_dir / "logs",
                num_runs,
                {"P", "P_TRUE"},
            )
        )
    # At the sim-dir root, outside logs/, which is what the store persists.
    log_manifest = out_seed_dir / "log_manifest.csv"
    pd.DataFrame(rows).to_csv(log_manifest, index=False)

    manifest_json = out_seed_dir / "manifest.json"
    manifest_json.write_text(json.dumps({"manifest": manifest_items}, indent=2), encoding="utf-8")
    # extract runs as subprocess to avoid argparse global-state collision
    run_extract(manifest_json, log_manifest, out_seed_dir, bin_minutes)


def _place_dir(src: Path, dst: Path) -> None:
    """Copy src dir to dst via .tmp+rename so a crash can't leave a half-written entry."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    t = dst.with_name(dst.name + ".tmp")
    if t.exists():
        shutil.rmtree(t)
    shutil.copytree(src, t)
    if dst.exists():
        shutil.rmtree(dst)
    os.rename(t, dst)


def simulate_synth_baseline(run_dir: Path, manifest_items: list[dict], num_runs: int,
                             seeds: list[int], bin_minutes: int = 60,
                             store_tag: str = "") -> None:
    """Simulate the seed-independent baseline once (reuse if present), then fan the logs and the extracted rho into each seed dir."""
    logs_dir = synth_log_store() / store_tag
    rho_csv  = logs_dir / "rho_profiles.csv"

    if rho_csv.exists():   # written last, so its presence means the store is complete
        log(f"reuse stored baseline simulation ({store_tag})")
    else:
        tmp = run_dir / "simulation" / "synth" / "_baseline_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        log(f"simulate baseline ({num_runs} runs/scenario)")
        run_simulation_and_extract(tmp, manifest_items, num_runs, bin_minutes=bin_minutes)
        _place_dir(tmp / "logs" / "synthetic", logs_dir)
        shutil.copy2(tmp / "rho_profiles.csv", rho_csv)
        shutil.rmtree(tmp)

    for s in seeds:   # feature extraction reads per-seed
        sd = run_dir / "simulation" / "synth" / f"seed{s:03d}"
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True, exist_ok=True)
        shutil.copytree(logs_dir, sd / "logs",
                        ignore=shutil.ignore_patterns("rho_profiles.csv"))
        shutil.copy2(rho_csv, sd / "rho_profiles.csv")
    log(f"baseline -> {len(seeds)} seed dir(s)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",            required=True,
                    help="Root directory for all pipeline artifacts.")
    ap.add_argument("--synthetic",          action="store_true",
                    help="Load the synthetic configs and run the baseline simulation.")
    ap.add_argument("--bin-minutes",        type=int,   default=60,
                    help="Time-bin resolution in minutes (must match the frozen configs).")
    ap.add_argument("--n-synth-seeds",      type=int, default=10)
    ap.add_argument("--n-synth-runs",       type=int, default=5,
                    help="Simulation runs per scenario per state.")
    args = ap.parse_args()
    if not args.synthetic:
        ap.error("Pass --synthetic. Preparation builds the synthetic side; the real side "
                 "starts from the shipped observed log.")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = load_synth_configs(run_dir)
    store_tag = f"n{args.n_synth_runs}_bin{args.bin_minutes}_queueing"
    seeds = run_seeds(args.n_synth_seeds)
    log(f"Baseline (seed-independent) for {args.n_synth_seeds} seeds, "
        f"{args.n_synth_runs} runs/scenario, bin_minutes={args.bin_minutes}, sim-store [{store_tag}]")
    simulate_synth_baseline(run_dir, manifest_items, args.n_synth_runs, seeds,
                            bin_minutes=args.bin_minutes, store_tag=store_tag)

    log("Preparation complete.")


if __name__ == "__main__":
    main()
