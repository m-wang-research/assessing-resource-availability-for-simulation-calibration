#!/usr/bin/env python3
"""Synthetic simulation. Turns a scenario config and availability surface into an event log."""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_SP = os.environ.get("SIMPN_PATH")
if _SP and _SP not in sys.path:
    sys.path.insert(0, _SP)


_ROOT = Path(__file__).resolve().parents[1]     # resource


def resolve_cfg(p: str | Path) -> Path:
    """Resolve a manifest/config path."""
    p = Path(p)
    return p if p.is_absolute() else _ROOT / p


# ── The queueing engine: a real Petri net on SimPN ─────────────────────────────
def generate_queueing(cfg, state, rng, resources, lookup, start, weeks, mean_service,
                      n_act, activity_names, case_prefix):
    """Sequential n_act-activity generator in which cases queue for resources."""
    from simpn.simulator import SimProblem, SimToken

    total_min = weeks * 7 * 24 * 60
    past_horizon = total_min * 10      # parks a resource that has no shift left to serve
    on: dict = {}

    def is_on(r, absday, b):
        """Whether r is on shift in hour b of absday, drawn once per timeslot."""
        key = (r, absday, b)
        if key not in on:
            weekday = (start + timedelta(days=absday)).weekday()
            draw = random.Random(f"{case_prefix}:availability:{r}:{absday}:{b}").random()
            on[key] = draw < lookup.get((r, weekday, b), 0.0)
        return on[key]

    def next_on(r, minute):
        """Earliest minute >= `minute` at which r is on shift, or None past the horizon."""
        m = max(0.0, float(minute))
        for _ in range(weeks * 7 * 24 + 2):
            h = int(m // 60)
            if h * 60 >= total_min:
                return None
            if is_on(r, h // 24, h % 24):
                return m
            m = (h + 1) * 60.0
        return None

    problem = SimProblem()
    pool = problem.add_var("resources")
    flows = [problem.add_var(f"flow{i}") for i in range(n_act + 1)]
    # The arrivals wait here rather than on flow0.
    arrivals = problem.add_var("arrivals")

    def release(c):
        return [SimToken(c)]

    problem.add_event([arrivals], [flows[0]], release, name="release")

    log: list[tuple] = []

    def make_task(idx, name):
        busy = problem.add_var(f"busy_{idx}")

        def start(case, resource):
            """A guard cannot carry this rule: SimPN evaluates guards while it enumerates bindings, before the clock reaches the firing time."""
            now = problem.clock
            begin = next_on(resource, now)
            if begin is not None and begin <= now:
                # keyed by case and activity, so a rate sweep does not also redraw durations
                duration = (random.Random(f"{case_prefix}:service:{case}:{name}")
                            .expovariate(1.0 / mean_service))
                return [SimToken((case, resource, duration), delay=duration), None, None]
            # off shift: the case returns to its place and may take another resource
            wait = past_horizon if begin is None else begin - now
            return [None, SimToken(case), SimToken(resource, delay=wait)]

        def complete(b):
            case, resource, duration = b
            done_at = problem.clock
            log.append((case, name, done_at - duration, done_at, resource))
            back = next_on(resource, done_at)
            wait = past_horizon if back is None else back - done_at
            return [SimToken(case), SimToken(resource, delay=wait)]

        problem.add_event([flows[idx], pool], [busy, flows[idx], pool], start,
                          name=f"start_{name}_{idx}")
        problem.add_event([busy], [flows[idx + 1], pool], complete,
                          name=f"complete_{name}_{idx}")

    for i in range(n_act):
        make_task(i, activity_names[i] if i < len(activity_names) else f"act{i + 1}")

    for r in resources:
        first = next_on(r, 0.0)
        pool.put(r, time=0.0 if first is None else first)

    n_case = 0
    for day in range(weeks * 7):
        weekday = (start + timedelta(days=day)).weekday()
        for b in range(24):
            for _ in range(poisson(arrival_rate(cfg, state, weekday, b), rng)):
                arrivals.put(f"{case_prefix}_{n_case}", time=day * 1440 + b * 60 + rng.randrange(60))
                n_case += 1

    problem.simulate(total_min)

    log.sort(key=lambda e: (int(e[0].rsplit("_", 1)[1]), e[2]))
    return [{
        "case_id": case,
        "activity": name,
        "start_time": (start + timedelta(minutes=sm)).isoformat(),
        "end_time": (start + timedelta(minutes=em)).isoformat(),
        "resource": r,
    } for (case, name, sm, em, r) in log]


# ── Synthetic and real (Prosimos) engine, re-exported by _core ─────────────────

PROSIMOS = Path(os.environ.get(
    "PROSIMOS_ROOT",
    str(Path(__file__).resolve().parents[2]
        / "external" / "probabilistic_resource_calendars" / "Resource_Availability" / "Prosimos"),
))

def load_manifest(path: Path, target: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["manifest"]
    return [r for r in rows if target == "both" or r.get("target") == target]

def simulate_synthetic_config(cfg_path: Path, output_dir: Path, num_runs: int, states: set[str], sim_seed: int | None = None) -> list[dict]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    p = pd.read_csv(resolve_cfg(cfg["p_surface_csv"]))
    p_true = pd.read_csv(resolve_cfg(cfg["p_true_surface_csv"]))
    surfaces = {"P": p, "P_TRUE": p_true}
    if str(cfg.get("p_cal_surface_csv", "")).strip():
        surfaces["P_CAL"] = pd.read_csv(cfg["p_cal_surface_csv"]).rename(columns={"p_cal": "p"})
    rows = []
    resources = sorted(p["resource"].astype(str).unique())
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)
    weeks = int(cfg["weeks"])
    mean_service = float(cfg["mean_service_minutes"])
    n_act = int(cfg.get("n_activities", 2))
    activity_names = cfg.get("activity_names") or (
        ["Preparation", "Processing"] if n_act == 2 else [f"act{i + 1}" for i in range(n_act)]
    )
    for state, surface in surfaces.items():
        if state not in states:
            continue
        lookup = {(str(r.resource), int(r.weekday), int(r.time_bin)): float(getattr(r, "p", getattr(r, "p_true", 0.0))) for r in surface.itertuples(index=False)}
        for run in range(num_runs):
            salt = "" if sim_seed is None else f"SYNTH{sim_seed}:"
            rng = random.Random(f"{salt}{cfg['scenario_id']}:{state}:{run}")
            case_prefix = f"{cfg['scenario_id']}_{state}_{run}"
            out_rows = generate_queueing(cfg, state, rng, resources, lookup, start, weeks, mean_service, n_act, activity_names, case_prefix)
            out = output_dir / "synthetic" / cfg["scenario_id"] / state
            out.mkdir(parents=True, exist_ok=True)
            log_path = out / f"run{run}.csv"
            pd.DataFrame(
                out_rows,
                columns=["case_id", "activity", "start_time", "end_time", "resource"],
            ).to_csv(log_path, index=False)
            rows.append({"target": "synthetic", "scenario_id": cfg["scenario_id"], "state": state, "run_idx": run, "log_csv": str(log_path)})
    return rows

def arrival_rate(cfg: dict, state: str, weekday: int, time_bin: int) -> float:
    base = float(cfg["arrival_rate_per_hour"])
    by_state = cfg.get("arrival_rate_by_state", {}) or {}
    if state in by_state:
        base = float(by_state[state])
    multipliers = cfg.get("arrival_multiplier_by_state", {}) or {}
    base *= float(multipliers.get(state, 1.0))
    profiles = cfg.get("arrival_profile_by_state", {}) or {}
    profile = profiles.get(state) or profiles.get("*")
    if profile:
        base *= float(profile[int(time_bin) % len(profile)])
    weekday_multipliers = cfg.get("arrival_weekday_multiplier_by_state", {}) or {}
    weekday_profile = weekday_multipliers.get(state) or weekday_multipliers.get("*")
    if weekday_profile:
        base *= float(weekday_profile[int(weekday) % len(weekday_profile)])
    return max(0.0, base)

def poisson(lam: float, rng: random.Random) -> int:
    """Knuth's method. lam = 0 exits the loop before it runs, hence the floor at zero."""
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return max(0, k - 1)

def is_fuzzy(json_path: Path) -> bool:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return any("availability_probabilities" in c for c in payload.get("resource_calendars", []))

def run_prosimos(real_dir: Path, json_path: Path, output_dir: Path, num_runs: int, total_cases: int, sim_seed: int | None = None, tag: str = "") -> list[dict]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    # The Prosimos subprocess runs with cwd=PROSIMOS, so every path handed to it
    # must be absolute (a relative --out makes json_path relative here).
    real_dir = real_dir.resolve()
    json_path = json_path.resolve()
    start_ts = pd.to_datetime(pd.read_csv(real_dir / "training_csv_log.csv", usecols=["start_time"])["start_time"], utc=True, format="mixed").min().isoformat()
    fuzzy = is_fuzzy(json_path)
    stats_dir = output_dir / "stats"           # run_simulation's stats output; unused downstream
    stats_dir.mkdir(exist_ok=True)
    for run in range(num_runs):
        log_path   = output_dir / f"{json_path.stem}_l{run}.csv"
        stats_path = stats_dir / f"{json_path.stem}_l{run}.csv"
        seed_prelude = ""
        if sim_seed is not None:
            s = int(hashlib.md5(f"{sim_seed}:{tag}:{run}".encode()).hexdigest()[:8], 16)
            seed_prelude = f"import numpy, random; numpy.random.seed({s}); random.seed({s}); "
        # run_simulation(bpmn, json, total_cases, is_fuzzy, stat_out_path, log_out_path, starting_at)
        code = (
            "import sys, datetime; "
            + seed_prelude
            + f"sys.path.insert(0, {str(PROSIMOS)!r}); "
            "from bpdfr_simulation_engine.simulation_engine import run_simulation; "
            f"run_simulation({str(real_dir / 'bpmn_model.bpmn')!r}, {str(json_path)!r}, {int(total_cases)!r}, {bool(fuzzy)!r}, "
            f"{str(stats_path)!r}, {str(log_path)!r}, datetime.datetime.fromisoformat({start_ts!r}))"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(PROSIMOS))
        logs.append({"target": "real", "state": json_path.stem, "run_idx": run, "log_csv": str(log_path)})
    return logs

def simulate_real_config(cfg_path: Path, output_dir: Path, num_runs: int, states: set[str], sim_seed: int | None = None) -> list[dict]:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    real_dir = Path(cfg["observed_log"]).parent
    rows = []
    if "P" in states:
        for idx, path in enumerate(sorted(glob.glob(cfg["input_log_glob"]))[:num_runs]):
            rows.append({"target": "real", "candidate_id": cfg["candidate_id"], "state": "P", "run_idx": idx, "log_csv": path})
    if "P_CAL" in states and str(cfg.get("p_cal_json", "")).strip():
        made = run_prosimos(real_dir, Path(cfg["p_cal_json"]),
                            output_dir / "real" / cfg["candidate_id"] / "P_CAL",
                            num_runs, coverage_total_cases(real_dir),
                            sim_seed=sim_seed, tag=f"{cfg['candidate_id']}:P_CAL")
        # run_prosimos names the state after the json it ran; say which state it was.
        for r in made:
            r["candidate_id"] = cfg["candidate_id"]
            r["state"] = "P_CAL"
        rows.extend(made)
    return rows

def observed_cases(real_dir: Path) -> int:
    """Cases in the observed log."""
    return int(pd.read_csv(real_dir / "training_csv_log.csv", usecols=["case_id"])["case_id"].nunique())

REAL_COVERAGE_FACTOR = 2.0

def coverage_total_cases(real_dir: Path) -> int:
    """``observed_cases`` scaled so simulated arrivals cover the validation window."""
    return int(math.ceil(observed_cases(real_dir) * REAL_COVERAGE_FACTOR))
