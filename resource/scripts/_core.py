#!/usr/bin/env python3
"""Shared constants, helpers, and math for the pipeline stages."""
from __future__ import annotations

import argparse
import glob
import os
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

# ── Project layout (all paths derived from this file's location) ───────────────
SCRIPTS_DIR = Path(__file__).resolve().parent   # resource/scripts
ROOT        = SCRIPTS_DIR.parent                 # resource
PROJECT     = ROOT.parent                        # repo root
_LOG_DISTANCE_ROOT = PROJECT / "external" / "log-distance-measures" / "src"
if _LOG_DISTANCE_ROOT.exists() and str(_LOG_DISTANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOG_DISTANCE_ROOT))

# ── Artifact paths ─────────────────────────────────────────────────────────────
DEFAULT_OBS_LOG      = ROOT / "data/real/logs/BPI_2012/training_csv_log.csv"


def windows_from_scheme(scheme, bin_minutes: int = 60) -> dict[str, list[int]]:
    """{group_type: [bin indices]} for scheme; uncovered bins go to "other"."""
    if 60 % bin_minutes:
        raise ValueError(f"bin_minutes={bin_minutes} does not divide an hour")
    bph = 60 // bin_minutes          # bins per hour (1 at 60-min, 4 at 15-min, ...)
    bpd = 1440 // bin_minutes

    win: dict[str, list[int]] = {}
    explicit: set[int] = set()
    for name, h_start, h_end in scheme:  # h_end inclusive
        win[name] = [h * bph + sub for h in range(h_start, h_end + 1)
                     for sub in range(bph) if h * bph + sub < bpd]
        explicit.update(win[name])
    win["other"] = [b for b in range(bpd) if b not in explicit]
    return win


# Real log grouping: uniform 4h groups from 07:00.
REAL_4H_FROM7_SCHEME = [("h07", 7, 10), ("h11", 11, 14), ("h15", 15, 18)]   # [07,11)[11,15)[15,19)

# ── Scenarios ──────────────────────────────────────────────────────────────────
# Same set in SCENARIOS and SCEN_ORDER; only the S1 pair is swapped (iteration vs display order).
SCENARIOS = os.environ.get(
    "SCENARIOS",
    "S0-same,S1-low-uniform,S1-high-uniform,"
    "S2-early-strong,S2-late-strong,S3-busy,S3-idle").split(",")
SCEN_ORDER = ["S0-same", "S1-high-uniform", "S1-low-uniform",
              "S2-early-strong", "S2-late-strong", "S3-busy", "S3-idle"]
CONTROLS = {"S0-same", "S3-busy", "S3-idle"}
SCEN_SHORT = {
    "S0-same": "S0-same",
    "S1-low-uniform": "S1-low", "S1-high-uniform": "S1-high",
    "S2-early-strong": "S2-early", "S2-late-strong": "S2-late",
    "S3-busy": "S3-busy", "S3-idle": "S3-idle",
}


def run_extract(manifest_json, log_manifest, output_root, bin_minutes, target: str = "synthetic") -> None:
    """Run the extract stage as an isolated subprocess."""
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "_core.py"), "--entry", "extract",
         "--target", target,
         "--manifest-json", str(manifest_json),
         "--log-manifest-csv", str(log_manifest),
         "--output-root", str(output_root),
         "--bin-minutes", str(bin_minutes)],
        check=True, cwd=str(ROOT),
    )


# ── Approach hyperparameters ───────────────────────────────────────────────────
SEED_META = 10           # the run seeds are drawn from this one constant


def run_seeds(n: int) -> list[int]:
    """The n run seeds, drawn from SEED_META so the set is reproducible from one number."""
    return [int(x) for x in np.random.default_rng(SEED_META).integers(0, 2**32, n)]


def reported_env() -> dict[str, str]:
    """The reported synthetic configuration as environment variables."""
    return {
        "SIM_WINDOW":         SIM_WINDOW,
        "RUN_AGGREGATOR":     RUN_AGGREGATOR,
        "SYNTH_OBS_RUN":      str(SYNTH_OBS_RUN),
        "SYNTH_OBS_RUNS":     str(SYNTH_OBS_RUNS),
        "SYNTH_TRAIN_WINDOW": ",".join(SYNTH_TRAIN_WINDOW),
        "SYNTH_VAL_WINDOW":   ",".join(SYNTH_VAL_WINDOW),
        "SCENARIOS":          ",".join(SCENARIOS),
    }
MIN_MARGIN     = 1e-9      # group-majority label margin
MAJORITY_SHARE = 0.5       # group majority threshold

CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function":      "Logloss",
    "eval_metric":        "AUC",
    "iterations":         180,
    "depth":              4,
    "learning_rate":      0.05,
    "random_seed":        42,   # overridden per rep
    "verbose":            False,
    "thread_count":       2,
    "auto_class_weights": "Balanced",
    "allow_writing_files": False,  # no catboost_info/ training-log dir
}

# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Busy ratio ─────────────────────────────────────────────────────────────────
def add_busy_to_rows(rows: pd.DataFrame,
                         busy_by_unit: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add busy_ratio to rows, taking each unit_id's own lookup."""
    out = rows.copy()
    out["busy_ratio"] = 0.0
    parts = [merge_busy_into_rows(sub, busy_by_unit.get(str(unit), pd.DataFrame()))
             for unit, sub in out.groupby("unit_id")]
    return pd.concat(parts, axis=0).sort_index() if parts else out


def merge_busy_into_rows(rows: pd.DataFrame, busy: pd.DataFrame) -> pd.DataFrame:
    """Merge a busy_ratio lookup into rows. Idempotent; empty busy gives zeros."""
    out = rows.copy()
    if busy.empty:
        out["busy_ratio"] = 0.0
        return out
    rb = busy.copy()
    rb["resource"] = rb["resource"].astype(str)
    rb["weekday"]  = rb["weekday"].astype(int)
    rb["time_bin"] = rb["time_bin"].astype(int)
    for col in ["resource", "weekday", "time_bin"]:
        out[col] = out[col].astype(rb[col].dtype)
    out = out.drop(columns=["busy_ratio"], errors="ignore").merge(
        rb[["resource", "weekday", "time_bin", "busy_ratio"]],
        on=["resource", "weekday", "time_bin"], how="left",
    )
    out["busy_ratio"] = out["busy_ratio"].fillna(0.0)
    return out


# ── Group-majority label ───────────────────────────────────────────────────────
def build_group_majority_label(rows: pd.DataFrame) -> pd.Series:
    """Y+ per group: at least MAJORITY_SHARE of the group's rows have rho_obs > p."""
    is_pos = (rows["rho_obs"].astype(float) - rows["p"].astype(float)) > MIN_MARGIN
    df  = pd.DataFrame({"group_id": rows["group_id"].astype(str),
                         "is_pos":   is_pos.astype(int)})
    agg = df.groupby("group_id")["is_pos"].agg(["sum", "count"])
    share = agg["sum"] / agg["count"].clip(lower=1)
    return (share >= MAJORITY_SHARE).astype(int)


# ── Real group/row build inputs ────────────────────────────────────────────────
# Shipped: the observed BPI-2012 log + the per-candidate Prosimos base json.
_DATA_BPI = ROOT / "data/real/logs/BPI_2012"
_REAL_GROUPS_BUILD_ROOT = ROOT / "results" / "assets" / "real_groups_build"
_REAL_CONFIG_DIR   = ROOT / "data/real/configs"      # per-candidate prosimos_base.json
_REAL_PSURFACE_DIR = ROOT / "results/assets/real_p_surface"     # p_surface, derived from the base json (regenerable)
_REAL_P_LOGS       = ROOT / "results/assets/real_p_logs"        # seeded P (uncalibrated) reruns (regenerable)
_REAL_P_SEED = 0        # fixes the P-baseline reruns so ρ_sim is bit-reproducible
_REAL_P_RUNS = 5        # reruns per candidate for the P baseline

_WEEKDAY_IDX = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
                "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}


def p_surface_from_prosimos(base_json: Path, bin_minutes: int = 60) -> pd.DataFrame:
    """Flatten a Prosimos resource_calendars json to the per-(resource, weekday, time_bin) p grid. Assumes hourly fuzzy_intervals (the discovery granule); bins not covered stay 0."""
    d = json.loads(Path(base_json).read_text(encoding="utf-8"))
    bpd = 1440 // bin_minutes
    bph = 60 // bin_minutes
    rows = []
    for cal in d["resource_calendars"]:
        res = str(cal["id"]).replace("_timetable", "")
        grid = {(wd, b): 0.0 for wd in range(7) for b in range(bpd)}
        for ap in cal["availability_probabilities"]:
            wd = _WEEKDAY_IDX[ap["week_day"]]
            for iv in ap["fuzzy_intervals"]:
                b = int(iv["begin_time"][:2]) * bph
                if 0 <= b < bpd:
                    grid[(wd, b)] = float(iv["probability"])
        rows.extend({"resource": res, "weekday": wd, "time_bin": b, "p": p}
                    for (wd, b), p in grid.items())
    return pd.DataFrame(rows)


def ensure_p_surface(cid: str, bin_minutes: int = 60) -> Path:
    """Return the candidate's p_surface.csv, deriving it from prosimos_base.json into results/assets if absent."""
    out = _REAL_PSURFACE_DIR / cid / "p_surface.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        p_surface_from_prosimos(_REAL_CONFIG_DIR / cid / "prosimos_base.json",
                                bin_minutes).to_csv(out, index=False)
    return out

# ── Real group/row caches ──────────────────────────────────────────────────────
def real_candidate_ids() -> list[str]:
    """Shipped real candidates: the per-candidate prosimos_base.json dirs."""
    return sorted(p.name for p in _REAL_CONFIG_DIR.iterdir()
                  if (p / "prosimos_base.json").exists())


def ensure_real_p_logs(cid: str, num_runs: int = _REAL_P_RUNS, seed: int = _REAL_P_SEED) -> list[Path]:
    """Return the candidate's P (uncalibrated) baseline event logs, simulating them from the shipped prosimos_base.json into results/assets if absent."""
    out = _REAL_P_LOGS / cid
    existing = sorted(out.glob("prosimos_base_l*.csv"))
    if len(existing) >= num_runs:
        return existing[:num_runs]
    out.mkdir(parents=True, exist_ok=True)
    logs = run_prosimos(_DATA_BPI, _REAL_CONFIG_DIR / cid / "prosimos_base.json", out,
                        num_runs, coverage_total_cases(_DATA_BPI),
                        sim_seed=seed, tag=f"{cid}:P")
    return [Path(r["log_csv"]) for r in logs]


def _assemble_real_source(dest: Path) -> Path:
    """Assemble a real source_run at dest (never mutates a frozen source) from shipped inputs only: ρ_obs from the observed BPI log (train window), ρ_sim from seeded reruns of the shipped prosimos_base.json."""
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "real_common" / "real" / "configs").mkdir(parents=True)
    frames, items = [], []
    for cid in real_candidate_ids():
        resources = sorted(pd.read_csv(ensure_p_surface(cid))["resource"].astype(str).unique())
        obs = rho_from_logs([_DATA_BPI / "training_csv_log.csv"], 60, resources,
                            week_window=REAL_TRAIN_WINDOW, pool_workweek=False)
        sim = rho_from_logs(ensure_real_p_logs(cid), 60, resources,
                            week_window=sim_window(REAL_TRAIN_WINDOW), pool_workweek=False)
        for state, prof in (("OBS", obs), ("P", sim)):
            fr = prof[["resource", "weekday", "time_bin", "rho", "support_days", "run_count"]].copy()
            fr["target"] = "real"
            fr["unit_id"] = cid
            fr["state"] = state
            frames.append(fr)
        cd = dest / "real_common" / "real" / "configs" / cid
        cd.mkdir(parents=True, exist_ok=True)
        cfg = {"candidate_id": cid, "bin_minutes": 60,
               "base_json": str(_REAL_CONFIG_DIR / cid / "prosimos_base.json"),
               "p_surface_csv": str(ensure_p_surface(cid)),
               "observed_log": str(_DATA_BPI / "training_csv_log.csv"),
               "input_log_glob": str(_REAL_P_LOGS / cid / "prosimos_base_l*.csv")}
        json.dump(cfg, open(cd / "config.json", "w"), indent=1)
        items.append({"target": "real", "candidate_id": cid,
                      "config_json": str(cd / "config.json"), "state": "P"})
    pd.concat(frames, ignore_index=True).to_csv(dest / "real_common" / "rho_profiles.csv", index=False)
    json.dump({"manifest": items}, open(dest / "real_common" / "manifest.json", "w"), indent=1)
    return dest


def build_real_groups(run_id: str | None = None):
    """Regenerate real group and row caches from the regenerated ρ (ρ_obs from the observed log, ρ_sim from seeded reruns). Returns (groups_csv, rows_csv)."""
    run_id = run_id or "selfcontained"
    source = _assemble_real_source(ROOT / "results" / "assets" / "real_source")
    log("[_sources] building real groups from regenerated ρ")
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "_core.py"), "--entry", "real_groups",
         "--source-run", str(source), "--run-id", run_id],
        check=True,
    )
    out = _REAL_GROUPS_BUILD_ROOT / run_id
    return (out / "group_features.csv", out / "real_log_row_level_updates.csv")


OUT = ROOT / "results" / "_pipeline_scratch"

sys.path.insert(0, str(SCRIPTS_DIR))
from simulation import (
    load_manifest, simulate_synthetic_config, simulate_real_config,
    run_prosimos, coverage_total_cases, resolve_cfg,
)


REAL_TRAIN_WINDOW = tuple(os.environ.get("REAL_TRAIN_WINDOW", "2012-01-09,2012-02-05").split(","))

REAL_VAL_WINDOW = tuple(os.environ.get("REAL_VAL_WINDOW", "2012-02-06,2012-03-04").split(","))

# rho_obs comes from the training weeks of one P_TRUE run; the reference is that run's
# held-out weeks.
SYNTH_TRAIN_WINDOW = tuple(os.environ.get(
    "SYNTH_TRAIN_WINDOW", "2020-01-06,2020-02-02").split(","))   # 4 weeks, Mon .. Sun
SYNTH_VAL_WINDOW = tuple(os.environ.get(
    "SYNTH_VAL_WINDOW", "2020-02-03,2020-03-01").split(","))     # 4 weeks, Mon .. Sun
# Which shipped run plays the deployment's log, and how many runs pool into rho_obs.
SYNTH_OBS_RUN = int(os.environ.get("SYNTH_OBS_RUN", "0"))
SYNTH_OBS_RUNS = int(os.environ.get("SYNTH_OBS_RUNS", "1"))

# The window the real simulation is cut to: "train", "active" (without the wind-down
# tail) or "full".
SIM_WINDOW = os.environ.get("SIM_WINDOW", "train").lower()

_sim_active = os.environ.get("REAL_SIM_ACTIVE_WINDOW", "2011-10-01,2012-04-30")
REAL_SIM_ACTIVE_WINDOW = None if _sim_active.lower() == "none" else tuple(_sim_active.split(","))


def sim_window(train_window: tuple[str, str] | None) -> tuple[str, str] | None:
    """Window rho_sim is estimated on, given the side's training window."""
    if SIM_WINDOW == "train":
        return train_window
    if SIM_WINDOW == "active":
        return REAL_SIM_ACTIVE_WINDOW
    return None

def synth_obs_run_paths(paths: list[Path]) -> list[Path]:
    """Keep SYNTH_OBS_RUNS runs starting at SYNTH_OBS_RUN, so rho_obs comes from as many realisations as the protocol allows (one, to mirror the real side's single log)."""
    want = [f"run{SYNTH_OBS_RUN + i}" for i in range(max(SYNTH_OBS_RUNS, 1))]
    picked = [p for p in paths if Path(p).stem in want]
    return picked or (paths[:max(SYNTH_OBS_RUNS, 1)] if paths else [])


def synth_log_store() -> Path:
    """Where the synthetic logs live, from SYNTH_LOG_STORE."""
    value = os.environ.get("SYNTH_LOG_STORE")
    if not value:
        raise SystemExit("SYNTH_LOG_STORE is not set. Point it at the synthetic logs, "
                         "data/synthetic/logs for the ones this repository ships.")
    return Path(value)


def synth_val_reference(scenario: str) -> Path | None:
    """The shipped P_TRUE run SYNTH_OBS_RUN for `scenario` -- the same realisation whose training weeks produced rho_obs, so calibration and scoring share one reality."""
    root = synth_log_store()
    stores = sorted(d for d in root.glob("*")
                    if (d / scenario / "P_TRUE").is_dir())
    if len(stores) != 1:
        return None
    hit = stores[0] / scenario / "P_TRUE" / f"run{SYNTH_OBS_RUN}.csv"
    return hit if hit.exists() else None


# format="ISO8601": these logs mix whole-second and microsecond stamps.
def filter_log_window(df: pd.DataFrame, window: tuple[str, str] | None,
                      time_col: str = "start_time") -> pd.DataFrame:
    """Keep rows whose ``time_col`` falls in the inclusive [start, end] day window."""
    if window is None or df.empty or time_col not in df.columns:
        return df
    lo = pd.Timestamp(window[0], tz="UTC")
    hi = pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)  # end day inclusive
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True, format="ISO8601")
    return df[(t >= lo) & (t < hi)].copy()


def norm_log(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=["resource", "start_time", "end_time"])
    if df.empty:
        return pd.DataFrame(columns=["resource", "start_time", "end_time"])
    if "completion_time" in df.columns and "end_time" not in df.columns:
        df = df.rename(columns={"completion_time": "end_time"})
    if "task" in df.columns and "activity" not in df.columns:
        df = df.rename(columns={"task": "activity"})
    for c in ["start_time", "end_time"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="ISO8601")
    df = df.dropna(subset=["resource", "start_time", "end_time"])
    df = df[df["resource"].astype(str).str.strip().ne("")]
    if "activity" in df.columns:
        df = df[~df["activity"].astype(str).str.lower().isin(["arrive", "complete"])]
    return df.copy()

# How a seed's re-simulations collapse into its value, written into the metrics.
RUN_AGGREGATOR = os.environ.get("RUN_AGGREGATOR", "median").lower()
if RUN_AGGREGATOR not in ("median", "mean"):
    raise ValueError(f"RUN_AGGREGATOR={RUN_AGGREGATOR!r}; expected 'median' or 'mean'")


def aggregate_runs(values):
    """Collapse the per-re-simulation values of one seed. See RUN_AGGREGATOR."""
    return float(np.mean(values) if RUN_AGGREGATOR == "mean" else np.median(values))


def paired_improvement_pct(before, after):
    """Improvement in percent for one seed: subtract within a run, then aggregate."""
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    ok = np.isfinite(b) & np.isfinite(a) & (b > 0)
    if not ok.any():
        return float("nan")
    return aggregate_runs(100.0 * (b[ok] - a[ok]) / b[ok])


def weekday_class(weekday: pd.Series | np.ndarray, pool_workweek: bool) -> np.ndarray:
    """Mon-Fri one class and Sat+Sun another when pooling; otherwise each weekday its own."""
    wd = np.asarray(weekday, dtype=int)
    return np.where(wd < 5, 0, 1) if pool_workweek else wd


def occupancy_per_cell(log: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    """Minutes each (resource, date, time_bin) cell is covered by a busy interval (events exploded across touched bins, overlap clipped and summed)."""
    cols = ["resource", "date", "time_bin", "covered_min"]
    valid = log[log["end_time"] > log["start_time"]]
    if valid.empty:
        return pd.DataFrame(columns=cols)
    bin_ns = bin_minutes * 60 * 1_000_000_000
    bins_per_day = 1440 // bin_minutes
    s_ns = valid["start_time"].astype("int64").to_numpy()
    e_ns = valid["end_time"].astype("int64").to_numpy()
    res = valid["resource"].astype(str).to_numpy()
    first_bin = s_ns // bin_ns
    last_bin = (e_ns - 1) // bin_ns  # inclusive
    n_bins = (last_bin - first_bin + 1).astype(np.int64)
    res_rep = np.repeat(res, n_bins)
    fb_rep = np.repeat(first_bin, n_bins)
    s_rep = np.repeat(s_ns, n_bins)
    e_rep = np.repeat(e_ns, n_bins)
    offsets = np.concatenate([np.arange(k, dtype=np.int64) for k in n_bins]) if len(n_bins) else np.array([], dtype=np.int64)
    abs_bin = fb_rep + offsets
    bin_start_ns = abs_bin * bin_ns
    bin_end_ns = bin_start_ns + bin_ns
    ov_ns = np.minimum(e_rep, bin_end_ns) - np.maximum(s_rep, bin_start_ns)
    ov_min = np.clip(ov_ns, 0, None) / 60_000_000_000.0
    occ = pd.DataFrame({
        "resource": res_rep,
        "date": pd.to_datetime(bin_start_ns, utc=True).date.astype(str),
        "time_bin": (abs_bin % bins_per_day).astype(int),
        "covered_min": ov_min,
    })
    return occ.groupby(["resource", "date", "time_bin"], as_index=False)["covered_min"].sum()

def rho_from_logs(paths: list[Path], bin_minutes: int, resources: list[str] | None = None,
                  week_window: tuple[str, str] | None = None, *,
                  pool_workweek: bool) -> pd.DataFrame:
    daily_frames = []
    for run_idx, path in enumerate(paths):
        log = norm_log(path)
        log = filter_log_window(log, week_window)
        if log.empty:
            continue
        log["resource"] = log["resource"].astype(str)
        if resources is None:
            use_resources = sorted(log["resource"].unique())
        else:
            use_resources = sorted(map(str, resources))
            log = log[log["resource"].isin(use_resources)]
        start = log["start_time"].min().floor("D")
        end = log["start_time"].max().floor("D")
        dates = pd.DataFrame({"date_ts": pd.date_range(start, end, freq="D")})
        dates["date"] = dates["date_ts"].dt.date.astype(str)
        dates["weekday"] = dates["date_ts"].dt.weekday.astype(int)
        grid = pd.MultiIndex.from_product([use_resources, dates["date"], range(1440 // bin_minutes)], names=["resource", "date", "time_bin"]).to_frame(index=False)
        grid = grid.merge(dates[["date", "weekday"]], on="date", how="left")
        starts = log.copy()
        starts["date"] = starts["start_time"].dt.date.astype(str)
        starts["weekday"] = starts["start_time"].dt.weekday.astype(int)
        starts["time_bin"] = ((starts["start_time"].dt.hour * 60 + starts["start_time"].dt.minute) // bin_minutes).astype(int)
        counts = starts.groupby(["resource", "date", "weekday", "time_bin"], as_index=False).size().rename(columns={"size": "start_count"})
        daily = grid.merge(counts, on=["resource", "date", "weekday", "time_bin"], how="left")
        occ = occupancy_per_cell(log, bin_minutes)
        daily = daily.merge(occ, on=["resource", "date", "time_bin"], how="left")
        daily["covered_min"] = pd.to_numeric(daily["covered_min"], errors="coerce").fillna(0.0)
        daily["run_idx"] = run_idx
        daily["started"] = pd.to_numeric(daily["start_count"], errors="coerce").fillna(0).gt(0).astype(float)
        # Spare-capacity occurrence: not fully occupied, or a start was observed.
        daily["has_spare"] = (daily["started"] > 0) | (daily["covered_min"] < bin_minutes - 1e-6)
        daily_frames.append(daily)
    if not daily_frames:
        return pd.DataFrame(columns=["resource", "weekday", "time_bin", "rho", "support_days", "run_count"])
    daily = pd.concat(daily_frames, ignore_index=True)
    keys = ["resource", "weekday", "time_bin"]
    full_keys = daily[keys].drop_duplicates()
    spare = daily[daily["has_spare"]].copy()
    # rho = spare-capacity occurrences with a start over all of them, on the pooled
    # weekday class and broadcast back to its weekdays.
    spare["wclass"] = weekday_class(spare["weekday"], pool_workweek)
    pooled_keys = ["resource", "wclass", "time_bin"]
    agg = spare.groupby(pooled_keys, as_index=False).agg(
        rho=("started", "mean"),
        support_days=("date", "nunique"),
        run_count=("run_idx", "nunique"),
    )
    out = full_keys.copy()
    out["wclass"] = weekday_class(out["weekday"], pool_workweek)
    out = out.merge(agg, on=pooled_keys, how="left").drop(columns=["wclass"])
    out["support_days"] = out["support_days"].fillna(0).astype(int)
    out["run_count"] = out["run_count"].fillna(0).astype(int)
    return out


def map_synth_rows_to_groups(synth_rows: pd.DataFrame, synth_groups: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, b in synth_groups.iterrows():
        bs, be = int(b["bin_start"]), int(b["bin_end"])
        bm = b.get("bin_members", "")
        if isinstance(bm, str) and bm.strip():
            bins = [int(x) for x in bm.replace("\n", ",").split(",") if x.strip().isdigit()]
        else:
            bins = list(range(bs, be + 1))
        for tb in bins:
            pieces.append((str(b["unit_id"]), str(b["resource"]), int(b["weekday"]),
                           int(tb), str(b["group_id"])))
    mp = pd.DataFrame(pieces, columns=["unit_id", "resource", "weekday", "time_bin", "group_id"])
    mp = mp.drop_duplicates(subset=["unit_id", "resource", "weekday", "time_bin"])
    sr = synth_rows.copy()
    sr["unit_id"] = sr["unit_id"].astype(str)
    sr["resource"] = sr["resource"].astype(str)
    sr["weekday"] = sr["weekday"].astype(int)
    sr["time_bin"] = sr["time_bin"].astype(int)
    return sr.merge(mp, on=["unit_id", "resource", "weekday", "time_bin"], how="left")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rho_diff"] = out["rho_obs"] - out["rho_sim"]
    out["abs_rho_diff"] = out["rho_diff"].abs()
    out = out.sort_values(["unit_id", "resource", "weekday", "time_bin"]).reset_index(drop=True)
    key = ["unit_id", "resource", "weekday"]
    for col in ["rho_diff", "rho_obs", "rho_sim", "p"]:
        out[f"{col}_prev_bin"] = out.groupby(key)[col].shift(1).fillna(0.0)
        out[f"{col}_next_bin"] = out.groupby(key)[col].shift(-1).fillna(0.0)
        out[f"{col}_local_mean"] = (
            out.groupby(key)[col]
            .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
            .fillna(0.0)
        )
    out["rho_obs_resource_mean"] = out.groupby(["unit_id", "resource"])["rho_obs"].transform("mean")
    out["rho_sim_resource_mean"] = out.groupby(["unit_id", "resource"])["rho_sim"].transform("mean")
    out["rho_obs_bin_mean"] = out.groupby(["unit_id", "weekday", "time_bin"])["rho_obs"].transform("mean")
    out["rho_sim_bin_mean"] = out.groupby(["unit_id", "weekday", "time_bin"])["rho_sim"].transform("mean")
    out["rho_diff_same_bin_weekday_mean"] = out.groupby(["unit_id", "resource", "time_bin"])["rho_diff"].transform("mean")
    out["resource_total_starts_obs"] = out.groupby(["unit_id", "resource"])["rho_obs"].transform("sum")
    out["resource_total_starts_sim"] = out.groupby(["unit_id", "resource"])["rho_sim"].transform("sum")
    out["rho_sim_run_std"] = np.sqrt((out["rho_sim"] * (1.0 - out["rho_sim"])) / out["run_count_sim"].clip(lower=1.0))
    return out.fillna(0.0)

def build_feature_frame(manifest: list[dict[str, Any]], rho: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for item in manifest:
        if item["target"] != "synthetic":
            continue
        cfg = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        unit = item["scenario_id"]
        p = p_surface(cfg, truth=False)
        pt = p_surface(cfg, truth=True).rename(columns={"p": "p_true"})
        obs = rho_state(rho, unit, "P_TRUE", "rho_obs")
        sim = rho_state(rho, unit, "P", "rho_sim")
        frame = p.merge(pt, on=["resource", "weekday", "time_bin"], how="left")
        frame = frame.merge(obs, on=["resource", "weekday", "time_bin"], how="left")
        frame = frame.merge(sim, on=["resource", "weekday", "time_bin"], how="left")
        for col in ["rho_obs", "rho_sim", "support_days_obs", "support_days_sim", "run_count_sim", "p_true"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        frame["unit_id"] = unit
        frame["scenario"] = unit.rsplit("_", 1)[0]
        frames.append(frame)
    return add_features(pd.concat(frames, ignore_index=True))

def p_surface(cfg: dict[str, Any], truth: bool = False) -> pd.DataFrame:
    path = resolve_cfg(cfg["p_true_surface_csv"] if truth else cfg["p_surface_csv"])
    frame = pd.read_csv(path).rename(columns={"p_true": "p"})
    return frame[["resource", "weekday", "time_bin", "p"]]

def rho_state(rho: pd.DataFrame, unit_id: str, state: str, name: str) -> pd.DataFrame:
    rows = rho[(rho["unit_id"].astype(str) == str(unit_id)) & (rho["state"].astype(str) == state)].copy()
    return rows[["resource", "weekday", "time_bin", "rho", "support_days", "run_count"]].rename(
        columns={"rho": name, "support_days": f"support_days_{name.removeprefix('rho_')}", "run_count": f"run_count_{name.removeprefix('rho_')}"}
    )


def add_reliability_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["excess_over_p"] = (out["rho_obs"] - out["p"]).clip(lower=0.0)
    out["support_count"] = pd.to_numeric(out.get("support_days_obs", 0), errors="coerce").fillna(0.0)
    out = out.sort_values(["unit_id", "resource", "weekday", "time_bin"]).reset_index(drop=True)
    key = ["unit_id", "resource", "weekday"]
    pos = out["excess_over_p"].gt(0)
    out["pos_prev"] = pos.groupby([out[k] for k in key]).shift(1).fillna(False).astype(int)
    out["pos_next"] = pos.groupby([out[k] for k in key]).shift(-1).fillna(False).astype(int)
    out["neighbor_positive_evidence"] = out["pos_prev"] + out["pos_next"]
    groups = []
    for _, g in out.groupby(key, sort=False):
        vals = g["excess_over_p"].gt(0).to_numpy()
        lengths = np.zeros(len(vals), dtype=int)
        start = 0
        while start < len(vals):
            end = start + 1
            while end < len(vals) and vals[end] == vals[start]:
                end += 1
            if vals[start]:
                lengths[start:end] = end - start
            start = end
        groups.extend(lengths.tolist())
    out["contiguous_positive_run_length"] = groups
    out["resource_day_positive_evidence_rate"] = out.groupby(["unit_id", "resource", "weekday"])["excess_over_p"].transform(lambda s: float(s.gt(0).mean()))
    denom = out.groupby(["unit_id", "resource", "time_bin"])["weekday"].transform("nunique").clip(lower=1)
    out["recurrence_across_weeks"] = out.groupby(["unit_id", "resource", "time_bin"])["excess_over_p"].transform(lambda s: float(s.gt(0).sum())) / denom
    return out.fillna(0.0)


def real_feature_frame(manifest: list[dict[str, Any]], rho: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for item in manifest:
        cfg = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        unit = item["candidate_id"]
        feat = real_features_for_unit(cfg, unit, rho)
        frames.append(feat)
    frame = pd.concat(frames, ignore_index=True)
    frame["scenario"] = frame["unit_id"]
    frame["scenario_type"] = "real_log"
    frame["scenario_factor"] = "real_candidate"
    frame["workload_level"] = "real"
    return add_reliability_features(frame)

def real_features_for_unit(cfg: dict[str, Any], unit: str, rho: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(resolve_cfg(cfg["p_surface_csv"]))[["resource", "weekday", "time_bin", "p"]]
    obs = rho[(rho["unit_id"].astype(str) == unit) & (rho["state"].eq("OBS"))][["resource", "weekday", "time_bin", "rho", "support_days"]].rename(columns={"rho": "rho_obs", "support_days": "support_days_obs"})
    sim_p = rho[(rho["unit_id"].astype(str) == unit) & (rho["state"].eq("P"))][["resource", "weekday", "time_bin", "rho", "support_days"]].rename(columns={"rho": "rho_sim", "support_days": "support_days_sim"})
    out = p.merge(obs, on=["resource", "weekday", "time_bin"], how="left").merge(sim_p, on=["resource", "weekday", "time_bin"], how="left")
    for c in ["rho_obs", "rho_sim", "support_days_obs", "support_days_sim"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["rho_diff"] = out["rho_obs"] - out["rho_sim"]
    out["abs_rho_diff"] = out["rho_diff"].abs()
    out["unit_id"] = unit
    out["p_true"] = np.nan
    return out


def add_resource_context(groups: pd.DataFrame) -> pd.DataFrame:
    """Two per-resource Q+ features: resource_mean_support, resource_activity_level."""
    out = groups.copy()
    group = ["unit_id", "resource"]
    out["resource_mean_support"] = out.groupby(group)["mean_support_count"].transform("mean")
    out["resource_activity_level"] = out.groupby(group)["sum_rho_obs"].transform("sum")
    return out.fillna(0.0)

def aggregate_group(group: pd.DataFrame, unit: str, resource: str, weekday: int, group_type: str, start: int, end: int) -> dict[str, Any]:
    """Group-level aggregation: the four Q+ features, sum_rho_obs, keys, has_behavior_evidence, and mean_rho_diff/sum_rho_diff/group_length."""
    rho_obs = group["rho_obs"].astype(float)
    rho_diff = group["rho_diff"].astype(float)
    support = group["support_count"].astype(float)
    sum_rho_obs = float(rho_obs.sum())
    sum_rho_diff = float(rho_diff.sum())
    return {
        "unit_id": unit,
        "scenario": group.get("scenario", pd.Series([unit])).iloc[0],
        "resource": str(resource),
        "weekday": weekday,
        "group_type": group_type,
        "bin_start": int(start),
        "bin_end": int(end),
        "group_id": f"{unit}|{resource}|{weekday}|{group_type}|{start}-{end}",
        "group_length": int(len(group)),
        "mean_support_count": float(support.mean()),
        "mean_rho_diff": float(rho_diff.mean()),
        "sum_rho_diff": sum_rho_diff,
        "sum_rho_obs": sum_rho_obs,
        "neighbor_positive_count": int(group.get("neighbor_positive_evidence", pd.Series(0, index=group.index)).sum()),
        "mean_recurrence_across_weeks": float(group["recurrence_across_weeks"].astype(float).mean()),
        "mean_resource_day_positive_evidence_rate": float(group["resource_day_positive_evidence_rate"].astype(float).mean()),
        "has_behavior_evidence": int((sum_rho_diff > 0) or (sum_rho_obs > 0)),
    }

def ensure_row_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "support_count" not in out.columns:
        out["support_count"] = pd.to_numeric(out.get("support_days_obs", 0), errors="coerce").fillna(0)
    if "excess_over_p" not in out.columns:
        out["excess_over_p"] = (out["rho_obs"] - out["p"]).clip(lower=0.0)
    if "rho_diff" not in out.columns:
        out["rho_diff"] = out["rho_obs"] - out["rho_sim"]
    if "recurrence_across_weeks" not in out.columns:
        out["recurrence_across_weeks"] = 0.0
    if "resource_day_positive_evidence_rate" not in out.columns:
        out["resource_day_positive_evidence_rate"] = 0.0
    if "scenario" not in out.columns:
        out["scenario"] = out.get("unit_id", "unknown")
    return out


def real_rows(source: Path) -> pd.DataFrame:
    manifest_json = source / "real_common" / "manifest.json"
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))["manifest"]
    rho = pd.read_csv(source / "real_common" / "rho_profiles.csv")
    rows = real_feature_frame(manifest, rho)
    rows["abs_rho_diff"] = rows["rho_diff"].abs()
    return rows


def _build_units(rows: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Group rows into windowed groups + resource-context features."""
    rows = ensure_row_features(rows)
    groups = []
    for (unit, res, weekday), g in rows.sort_values(["unit_id", "resource", "weekday", "time_bin"]).groupby(["unit_id", "resource", "weekday"], dropna=False):
        available = set(g["time_bin"].astype(int).tolist())
        for name, bins in windows.items():
            use_bins = [b for b in bins if b in available]
            if not use_bins:
                continue
            group = g[g["time_bin"].astype(int).isin(use_bins)].copy()
            item = aggregate_group(group, unit, str(res), int(weekday), name, int(min(use_bins)), int(max(use_bins)))
            item["bin_members"] = ",".join(map(str, use_bins))
            item["is_discontiguous"] = int(name == "other")
            groups.append(item)
    out = pd.DataFrame(groups)
    if out.empty:
        return out
    out = add_resource_context(out)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def get_windows(bin_minutes: int = 60) -> dict[str, list[int]]:
    """Synthetic grouping: 2-hour daytime groups over 08-16 + a night 'other' group."""
    bins_per_day = 1440 // bin_minutes
    start_hour = lambda b: (b * bin_minutes) / 60.0
    win: dict[str, list[int]] = {}
    for s in range(8, 16, 2):
        win[f"d{s:02d}"] = [b for b in range(bins_per_day) if s <= start_hour(b) < s + 2]
    win["other"] = [b for b in range(bins_per_day) if not any(b in v for v in win.values())]
    return win


def add_busy_to_groups(groups: pd.DataFrame, busy_long: pd.DataFrame) -> pd.DataFrame:
    """Attach mean_busy_ratio per group by averaging busy_ratio over its bin_members."""
    out = groups.copy()
    out["mean_busy_ratio"] = 0.0
    if busy_long is None or busy_long.empty:
        return out
    key = ["resource", "weekday", "time_bin"]
    busy_lookup = busy_long.set_index(key)["busy_ratio"].to_dict()
    vals = []
    for b in out.itertuples(index=False):
        members = [int(x) for x in str(getattr(b, "bin_members")).split(",") if str(x).strip()]
        if not members:
            vals.append(0.0); continue
        res = str(b.resource); wd = int(b.weekday)
        rs = [busy_lookup.get((res, wd, tb), 0.0) for tb in members]
        vals.append(float(np.mean(rs)) if rs else 0.0)
    out["mean_busy_ratio"] = vals
    return out

def add_busy_to_synth_groups(synth_groups: pd.DataFrame, busy_by_unit: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = synth_groups.copy()
    out["mean_busy_ratio"] = 0.0
    parts = []
    for unit, sub in out.groupby("unit_id"):
        bl = busy_by_unit.get(str(unit), pd.DataFrame())
        attached = add_busy_to_groups(sub, bl)
        parts.append(attached)
    return pd.concat(parts, axis=0).sort_index() if parts else out


def busy_per_resource_weekday_bin(log_paths: list[Path]) -> pd.DataFrame:
    """Per (resource, weekday, time_bin), fraction of week-days with an event ongoing at the bin start (start_time <= bin_start <= end_time)."""
    if not log_paths:
        return pd.DataFrame(columns=["resource", "weekday", "time_bin", "busy_ratio"])
    per_run_means = []
    for p in log_paths:
        df = pd.read_csv(p, usecols=["start_time", "end_time", "resource"])
        if df.empty:
            continue
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, format="mixed", errors="coerce")
        df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True, format="mixed", errors="coerce")
        df = df.dropna(subset=["start_time", "end_time"])
        if df.empty:
            continue
        df["resource"] = df["resource"].astype(str)

        day_min = df["start_time"].dt.floor("D").min()
        day_max = df["end_time"].dt.floor("D").max()
        all_days = pd.date_range(start=day_min, end=day_max, freq="D", tz="UTC")

        # Each event covers hourly bins floor(start)..floor(end): bin (day, hour) is covered iff
        # start_time <= bin_start <= end_time.
        starts_floor = df["start_time"].dt.floor("h")
        ends_floor   = df["end_time"].dt.floor("h")
        n_bins = ((ends_floor - starts_floor) / pd.Timedelta(hours=1)).astype(int) + 1
        rows_resource = np.repeat(df["resource"].values, n_bins.values)
        offsets = np.concatenate([np.arange(n) for n in n_bins.values])
        starts_floor_rep = np.repeat(starts_floor.values, n_bins.values)
        bin_ts = pd.to_datetime(starts_floor_rep, utc=True) + pd.to_timedelta(offsets, unit="h")
        bin_ts = pd.DatetimeIndex(bin_ts)
        # Enforce bin_start <= end_time strictly (floor != ceil on the second boundary).
        ends_rep = np.repeat(df["end_time"].values, n_bins.values)
        ends_rep = pd.to_datetime(ends_rep, utc=True)
        mask_valid = bin_ts <= ends_rep
        rows_resource = rows_resource[mask_valid]
        bin_ts = bin_ts[mask_valid]

        ev = pd.DataFrame({
            "resource": rows_resource,
            "day": bin_ts.floor("D"),
            "hour": bin_ts.hour.astype(int),
        }).drop_duplicates()

        # Fraction of days (per resource, weekday, hour) with an ongoing event.
        ev["weekday"] = ev["day"].dt.weekday.astype(int)
        # Denominator: distinct days per weekday across the log span.
        weekday_day_count = pd.Series(all_days.weekday).value_counts().to_dict()

        busy_counts = ev.groupby(["resource", "weekday", "hour"]).size().rename("busy_count").reset_index()
        busy_counts["denom"] = busy_counts["weekday"].map(weekday_day_count).fillna(1)
        busy_counts["busy_ratio_run"] = busy_counts["busy_count"] / busy_counts["denom"].clip(lower=1)
        per_run_means.append(busy_counts[["resource", "weekday", "hour", "busy_ratio_run"]])

    if not per_run_means:
        return pd.DataFrame(columns=["resource", "weekday", "time_bin", "busy_ratio"])
    cat = pd.concat(per_run_means, ignore_index=True)
    out = cat.groupby(["resource", "weekday", "hour"], as_index=False)["busy_ratio_run"].mean()
    out = out.rename(columns={"hour": "time_bin", "busy_ratio_run": "busy_ratio"})
    out["resource"] = out["resource"].astype(str)
    out["weekday"] = out["weekday"].astype(int)
    out["time_bin"] = out["time_bin"].astype(int)
    return out


REAL_WINDOWS = windows_from_scheme(REAL_4H_FROM7_SCHEME)


def add_group_id(rows: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Attach each real row's group_id (first matching window)."""
    out = rows.copy()
    out["group_id"] = pd.Series(np.nan, index=out.index, dtype="object")
    for b in groups.itertuples(index=False):
        members = {int(x) for x in str(getattr(b, "bin_members")).split(",") if str(x).strip()}
        mask = (
            out["unit_id"].astype(str).eq(str(b.unit_id))
            & out["resource"].astype(str).eq(str(b.resource))
            & out["weekday"].astype(int).eq(int(b.weekday))
            & out["time_bin"].astype(int).isin(members)
        )
        out.loc[mask & out["group_id"].isna(), "group_id"] = b.group_id  # first matching window wins
    return out

def make_updates(rows: pd.DataFrame, groups: pd.DataFrame, target: str) -> pd.DataFrame:
    """Attach each real row to its aggregation group and tag the target."""
    out = add_group_id(rows, groups)
    out["target"] = target
    return out


def _real_groups_main() -> None:
    ap = argparse.ArgumentParser(description="Build the real group/row-update caches from a source run.")
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=_REAL_GROUPS_BUILD_ROOT)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    run_id = args.run_id or pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    real_groups = _build_units(real_rows(args.source_run), REAL_WINDOWS)
    real_groups.to_csv(out_dir / "group_features.csv", index=False)
    real_updates = make_updates(real_rows(args.source_run), real_groups, "real_log")
    real_updates.to_csv(out_dir / "real_log_row_level_updates.csv", index=False)
    print(json.dumps({"out_dir": str(out_dir),
                      "group_features": str(out_dir / "group_features.csv")}, indent=2))


def _simulate_main() -> None:
    ap = argparse.ArgumentParser(description="Run standalone synthetic simulator or direct Prosimos real reruns.")
    ap.add_argument("--manifest-json", type=Path, default=OUT / "manifest.json")
    ap.add_argument("--target", choices=["synthetic", "real", "both"], default="both")
    ap.add_argument("--state", action="append", default=[], help="P, P_TRUE, P_CAL. Default: P and P_TRUE for synthetic, P for real.")
    ap.add_argument("--output-root", type=Path, default=OUT)
    ap.add_argument("--num-runs", type=int, default=10)
    ap.add_argument("--sim-seed", type=int, default=None,
                    help="Seed the real Prosimos reruns (bit-reproducible); folds into the per-run seed. Unset = non-deterministic.")
    args = ap.parse_args()
    states = set(args.state) if args.state else set()
    rows = []
    for item in load_manifest(args.manifest_json, args.target):
        if item["target"] == "synthetic":
            rows.extend(simulate_synthetic_config(resolve_cfg(item["config_json"]), args.output_root / "logs", args.num_runs, states or {"P", "P_TRUE"}))
        elif item["target"] == "real":
            rows.extend(simulate_real_config(resolve_cfg(item["config_json"]), args.output_root / "logs", args.num_runs, states or {"P"}, sim_seed=args.sim_seed))
    out = args.output_root / "logs" / "log_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(rows)
    if out.exists():
        old_rows = pd.read_csv(out)
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
        keys = [c for c in ["target", "scenario_id", "candidate_id", "state", "run_idx", "log_csv"] if c in new_rows.columns]
        new_rows = new_rows.drop_duplicates(subset=keys, keep="last")
    new_rows.to_csv(out, index=False)
    print(json.dumps({"log_manifest_csv": str(out), "rows": len(rows)}, indent=2))


def _manifest_log_paths(logs: pd.DataFrame, id_col: str, key: str, state: str) -> list[Path]:
    """The log-manifest rows for one unit in one state, as paths."""
    if logs.empty:
        return []
    hit = logs.loc[(logs.get(id_col) == key) & (logs.get("state") == state), "log_csv"]
    return [Path(x) for x in hit.tolist()]


def _extract_main() -> None:
    ap = argparse.ArgumentParser(description="Extract rho profiles: fraction of spare-capacity occurrences (weekday-bin dates that were not fully occupied) with at least one start.")
    ap.add_argument("--manifest-json", type=Path, default=OUT / "manifest.json")
    ap.add_argument("--log-manifest-csv", type=Path, default=OUT / "logs" / "log_manifest.csv")
    ap.add_argument("--target", choices=["synthetic", "real", "both"], default="both")
    ap.add_argument("--output-root", type=Path, default=OUT)
    ap.add_argument("--bin-minutes", type=int, default=60)
    args = ap.parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))["manifest"]
    logs = pd.read_csv(args.log_manifest_csv) if args.log_manifest_csv.exists() else pd.DataFrame()
    rows = []
    for item in manifest:
        if args.target != "both" and item["target"] != args.target:
            continue
        cfg = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        key = item.get("scenario_id") or item.get("candidate_id")
        resources = sorted(pd.read_csv(resolve_cfg(cfg["p_surface_csv"]))["resource"].astype(str).unique())
        if item["target"] == "synthetic":
            for state in ["P_TRUE", "P", "P_CAL"]:
                paths = _manifest_log_paths(logs, "scenario_id", key, state)
                # P_TRUE stands in for the observation: one run, training weeks. P and P_CAL
                # use every run over the full span.
                window = sim_window(SYNTH_TRAIN_WINDOW)
                if state == "P_TRUE":
                    paths = synth_obs_run_paths(paths)
                    window = SYNTH_TRAIN_WINDOW
                if paths:
                    prof = rho_from_logs(paths, args.bin_minutes, resources, week_window=window, pool_workweek=True)
                    prof["target"] = "synthetic"; prof["unit_id"] = key; prof["state"] = state
                    rows.append(prof)
        else:
            obs = rho_from_logs([Path(cfg["observed_log"])], args.bin_minutes, resources,
                                week_window=REAL_TRAIN_WINDOW, pool_workweek=False)
            obs["target"] = "real"; obs["unit_id"] = key; obs["state"] = "OBS"
            rows.append(obs)
            for state in ["P", "P_CAL"]:
                paths = _manifest_log_paths(logs, "candidate_id", key, state)
                if state == "P" and not paths:
                    paths = [Path(x) for x in glob.glob(cfg["input_log_glob"])]
                if paths:
                    prof = rho_from_logs(paths, args.bin_minutes, resources,
                                         week_window=sim_window(REAL_TRAIN_WINDOW), pool_workweek=False)
                    prof["target"] = "real"; prof["unit_id"] = key; prof["state"] = state
                    rows.append(prof)
    out = args.output_root / "rho_profiles.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(json.dumps({"rho_profiles_csv": str(out), "frames": len(rows)}, indent=2))


# Subprocess entry-point dispatch (--entry simulate|extract|real_groups).
if __name__ == "__main__":
    _ENTRIES = {"simulate": _simulate_main, "extract": _extract_main, "real_groups": _real_groups_main}
    if "--entry" in sys.argv:
        _i = sys.argv.index("--entry")
        _name = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
        _ENTRIES[_name]()
    else:
        _real_groups_main()
