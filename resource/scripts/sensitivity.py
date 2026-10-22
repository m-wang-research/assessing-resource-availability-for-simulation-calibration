#!/usr/bin/env python3
"""Run the synthetic chain repeatedly, varying one factor at a time."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # …/resource
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))
import _core
TEMPLATE = ROOT / "data" / "synthetic" / "configs_deviating_3_of_5"
CONFIG_ROOT = ROOT / "data" / "synthetic"              # the configs_deviating_* sets ship from here
WORK = ROOT / "results" / "sensitivity_arrival_rate"   # results/ is gitignored
RESOURCE_WORK = ROOT / "results" / "sensitivity_deviating_resources"
SUPP = ROOT.parent / "supplementary"

REPORTED = [0.25, 0.5, 0.75, 1.0, 1.25]

SCENARIOS = _core.SCENARIOS
DEVIATION = ["S1-low-uniform", "S1-high-uniform", "S2-early-strong", "S2-late-strong"]
CONTROL = ["S0-same", "S3-busy", "S3-idle"]

BASE_ENV = _core.reported_env()

def tag_of(multiplier: float) -> str:
    return f"{multiplier:g}x"


def build_stages(seeds: int, runs: int) -> list[tuple[str, list[str]]]:
    n, r = str(seeds), str(runs)
    return [
        ("preparation.py", ["--synthetic", "--n-synth-seeds", n, "--n-synth-runs", r]),
        ("estimation.py", ["--synthetic", "--n-synth-seeds", n]),
        ("diagnosis.py", ["--synthetic", "--n-synth-seeds", n]),
        ("calibration.py", ["--synthetic", "--n-synth-seeds", n, "--n-synth-runs", r]),
        ("evaluation.py", ["--n-synth-seeds", n, "--n-real-seeds", "0"]),
    ]


def write_configs(multiplier: float, dest: Path) -> None:
    """All seven settings at this arrival multiplier, availability copied verbatim."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy2(TEMPLATE / "p_true_surface.csv", dest / "p_true_surface.csv")

    items = []
    for scen in SCENARIOS:
        (dest / scen).mkdir()
        cfg = json.loads((TEMPLATE / scen / "config.json").read_text())
        own_rate = float(cfg["arrival_rate_per_hour"])       # keeps S3-busy/S3-idle relative
        cfg["arrival_rate_per_hour"] = own_rate * multiplier
        cfg["arrival_lambda"] = own_rate * multiplier
        cfg["workload_level"] = f"{cfg.get('workload_level', 'baseline')}@{tag_of(multiplier)}"
        cfg["scenario_description"] = (
            f"{cfg.get('scenario_description', '').rstrip()} "
            f"System load scaled to {multiplier:g}x: arrival rate {own_rate * multiplier:g} per hour.")
        cfg["p_surface_csv"] = str((dest / scen / "p_surface.csv").resolve())
        cfg["p_true_surface_csv"] = str((dest / "p_true_surface.csv").resolve())
        (dest / scen / "config.json").write_text(json.dumps(cfg, indent=1))
        shutil.copy2(TEMPLATE / scen / "p_surface.csv", dest / scen / "p_surface.csv")
        items.append({"target": "synthetic", "scenario_id": scen,
                      "scenario_type": cfg.get("scenario_type", ""),
                      "workload_level": cfg["workload_level"],
                      "config_json": str((dest / scen / "config.json").resolve()),
                      "state": "P"})

    (dest / "manifest.json").write_text(json.dumps({"manifest": items}, indent=1))


def launch(tag: str, cfg_dir: Path, work: Path, seeds: int, runs: int,
           log_dir: Path) -> tuple[subprocess.Popen, Path]:
    """One step: the synthetic chain with SYNTH_CONFIGS pointed at cfg_dir."""
    run_dir = work / f"run_{tag}"
    store = work / f"store_{tag}"
    shutil.rmtree(run_dir, ignore_errors=True)

    env = dict(os.environ)
    env.update(BASE_ENV)
    env["SYNTH_CONFIGS"] = str(cfg_dir)
    env["SYNTH_LOG_STORE"] = str(store)   # one per step: the store tag carries no load
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    rel = run_dir.relative_to(ROOT)
    chain = " && ".join(
        f"{sys.executable} {SCRIPTS / s} --run-dir {rel} " + " ".join(a)
        for s, a in build_stages(seeds, runs))
    log = log_dir / f"{tag}.log"
    proc = subprocess.Popen(["/bin/sh", "-c", chain], cwd=ROOT, env=env,
                            stdout=log.open("w"), stderr=subprocess.STDOUT)
    return proc, log


def launch_workload(multiplier: float, seeds: int, runs: int,
                    log_dir: Path) -> tuple[subprocess.Popen, Path]:
    """One arrival rate: its configuration set is written under that step's own work dir."""
    tag = tag_of(multiplier)
    cfg_dir = WORK / f"configs_{tag}"
    write_configs(multiplier, cfg_dir)
    return launch(tag, cfg_dir, WORK, seeds, runs, log_dir)


def launch_resources(n: int, seeds: int, runs: int,
                     log_dir: Path) -> tuple[subprocess.Popen, Path]:
    """One number of deviating resources: its configuration set ships, so nothing is written."""
    return launch(f"{n}_of_5", CONFIG_ROOT / f"configs_deviating_{n}_of_5", RESOURCE_WORK,
                  seeds, runs, log_dir)

# ── Supplementary material ─────────────────────────────────────────────────────
def collect(multipliers: list[float]) -> None:
    """Write the sweep's per-seed results and its observable into supplementary/data."""
    per_seed, observable = [], []
    for m in multipliers:
        tag = tag_of(m)
        run = WORK / f"run_{tag}"
        synth = pd.read_csv(run / "evaluation" / "synth_per_seed.csv")
        emp = pd.read_csv(run / "evaluation" / "emp_baseline_per_seed.csv")
        emp = emp[["seed", "scenario", "D_P_emp", "D_P_cal", "D_P_cal_working_hours"]]
        d = synth.merge(emp, on=["seed", "scenario"], how="left", validate="one_to_one")
        d.insert(0, "load", m)
        per_seed.append(d)

        rate = {s: float(json.loads((TEMPLATE / s / "config.json").read_text())
                         ["arrival_rate_per_hour"]) * m for s in SCENARIOS}
        for f in sorted((run / "extraction" / "synth").glob("seed*/synthetic_row_features.csv")):
            x = pd.read_csv(f, usecols=["scenario", "weekday", "time_bin", "rho_obs", "p_true"])
            x = x[(x.weekday < 5) & (x.time_bin >= 8) & (x.time_bin < 16)]
            for scen, g in x.groupby("scenario"):
                observable.append({
                    "load": m, "scenario": scen, "seed": int(f.parent.name[len("seed"):]),
                    "arrival_rate_per_hour": rate[scen],
                    "timeslots": int(len(g)),
                    "no_start_timeslots": int((g.rho_obs <= 1e-3).sum()),
                    "rho_obs_mean": float(g.rho_obs.mean()),
                    "p_true_mean": float(g.p_true.mean()),
                    "rho_obs_over_p_true": float(g.rho_obs.mean() / g.p_true.mean()),
                })

    dest = SUPP / "data" / "synthetic" / "sensitivity"
    dest.mkdir(parents=True, exist_ok=True)
    ps = pd.concat(per_seed, ignore_index=True)
    ob = pd.DataFrame(observable).sort_values(["load", "scenario", "seed"])
    ps.to_csv(dest / "workload_per_seed.csv", index=False)
    ob.to_csv(dest / "workload_observable.csv", index=False)
    print(f"wrote {dest / 'workload_per_seed.csv'}  ({len(ps)} rows)")
    print(f"wrote {dest / 'workload_observable.csv'}  ({len(ob)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    factor = ap.add_mutually_exclusive_group(required=True)
    factor.add_argument("--workload", action="store_true",
                        help="scale every scenario's arrival rate, one run per multiplier")
    factor.add_argument("--resources", action="store_true",
                        help="deviate on one to five of the five resources, one run each")
    ap.add_argument("--multipliers", type=float, nargs="+", default=REPORTED,
                    help="with --workload: the arrival-rate multipliers")
    ap.add_argument("--deviating", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="with --resources: how many of the five resources deviate")
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-synth-runs", type=int, default=5)
    ap.add_argument("--max-parallel", type=int, default=5,
                    help="runs in flight at once; each one is single-threaded")
    ap.add_argument("--collect", action="store_true",
                    help="with --workload: write the workload CSVs from finished runs")
    args = ap.parse_args()

    if args.collect:
        if args.resources:
            ap.error("--collect covers the arrival-rate runs; "
                     "the resource runs are collected by collect_supplementary.py")
        collect(args.multipliers)
        return

    if args.resources:
        work, items = RESOURCE_WORK, list(args.deviating)
        started_from = "resource"
    else:
        work, items = WORK, list(args.multipliers)
        started_from = "workload"
    work.mkdir(parents=True, exist_ok=True)
    log_dir = work / "logs"
    log_dir.mkdir(exist_ok=True)

    def launch_item(item):
        if args.resources:
            return launch_resources(item, args.n_synth_seeds, args.n_synth_runs, log_dir)
        return launch_workload(item, args.n_synth_seeds, args.n_synth_runs, log_dir)

    def label_of(item) -> str:
        return f"{item} of five deviating" if args.resources else f"{item:g}x"

    pending = list(items)
    running: list[tuple[object, subprocess.Popen, Path]] = []
    done: list[tuple[object, int]] = []
    began = time.time()

    while pending or running:
        while pending and len(running) < args.max_parallel:
            item = pending.pop(0)
            proc, log = launch_item(item)
            running.append((item, proc, log))
            print(f"[{started_from}] {label_of(item)} started -> {log.relative_to(ROOT)}", flush=True)
        time.sleep(10)
        for entry in list(running):
            item, proc, log = entry
            if proc.poll() is not None:
                running.remove(entry)
                done.append((item, proc.returncode))
                mins = (time.time() - began) / 60
                state = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{started_from}] {label_of(item)} {state} after {mins:.1f} min", flush=True)

    print(f"\n[{started_from}] all runs finished")
    for item, rc in sorted(done, key=str):
        tag = f"{item}_of_5" if args.resources else tag_of(item)
        print(f"  {label_of(item):22} {'ok' if rc == 0 else f'FAILED rc={rc}'}  "
              f"{work / f'run_{tag}' / 'evaluation' / 'synth_per_seed.csv'}")
    if any(rc != 0 for _, rc in done):
        sys.exit(1)


if __name__ == "__main__":
    main()
