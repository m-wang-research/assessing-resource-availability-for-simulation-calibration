#!/usr/bin/env python3
"""Aggregate the per seed calibration metrics into summary CSVs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _core import (
    run_seeds,
    SCENARIOS,
    SCEN_ORDER,
    SCEN_SHORT,
    aggregate_runs,
    log,
    paired_improvement_pct,
)


# ── Behavioral-distance metrics (wrappers over log_distance_measures) ──────────
import datetime as _dt
from log_distance_measures.config import DistanceMetric, EventLogIDs
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance

# Wasserstein distances compare distribution shape; volume is reported as throughput.
LDM_METRIC = "wasserstein"

_LDM_IDS = EventLogIDs(
    case="case_id",
    activity="task",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)

_LDM_BIN_SIZE = _dt.timedelta(minutes=1)

# --- Fixed unavailable periods, for cycle time -------------------------------------------
# CTD runs on a clock that stands still while the calendar is closed. CWD and CED keep
# elapsed time. Hours of the week, index 0 being Monday 00:00, the grid p_{r,tau} uses.
HOURS_PER_WEEK = 168
_WEEK_ORIGIN = pd.Timestamp("2020-01-06T00:00:00+00:00")   # a Monday: hour-of-week index 0


def hours_of_week(*, workdays, open_hours) -> tuple[int, ...]:
    """The hour-of-week indices *outside* `open_hours` on `workdays` -- i.e."""
    open_set = {d * 24 + h for d in workdays for h in open_hours}
    return tuple(i for i in range(HOURS_PER_WEEK) if i not in open_set)


def parse_hours_of_week(spec: str) -> tuple[int, ...]:
    """Parse "0-7,16-23,120-167" into hour-of-week indices. Empty string -> ()."""
    out: set[int] = set()
    for part in (x.strip() for x in spec.split(",") if x.strip()):
        lo, _, hi = part.partition("-")
        rng = range(int(lo), int(hi) + 1) if hi else range(int(lo), int(lo) + 1)
        for i in rng:
            if not 0 <= i < HOURS_PER_WEEK:
                raise ValueError(f"hour-of-week {i} out of range in {spec!r}")
            out.add(i)
    return tuple(sorted(out))


# The synthetic calendar: 08:00-16:00, Monday to Friday, so nights and weekends are closed.
SYNTH_CLOSED_HOURS = (
    parse_hours_of_week(os.environ["SYNTH_CLOSED_HOURS"])
    if "SYNTH_CLOSED_HOURS" in os.environ
    else hours_of_week(workdays=range(5), open_hours=range(8, 16))
)

# BPI-2012 declares no calendar, so the real side is plain elapsed time.
REAL_CLOSED_HOURS: tuple[int, ...] = ()


def skip_closed_hours(frame: pd.DataFrame, closed: tuple[int, ...]) -> pd.DataFrame:
    """`frame` with its timestamps remapped onto a clock that stands still during `closed`."""
    if not closed:
        return frame
    open_mask = np.ones(HOURS_PER_WEEK, dtype=float)
    open_mask[list(closed)] = 0.0
    # Open minutes accumulated at the start of each hour of the week.
    prefix = np.concatenate([[0.0], np.cumsum(open_mask) * 60.0])

    def remap(stamps: pd.Series) -> pd.Series:
        idx = pd.DatetimeIndex(pd.to_datetime(stamps, errors="coerce", utc=True, format="ISO8601"))
        hours = np.asarray((idx - _WEEK_ORIGIN).total_seconds()) / 3600.0
        week, hour_of_week = np.divmod(np.floor(hours).astype(np.int64), HOURS_PER_WEEK)
        within = (hours - np.floor(hours)) * 60.0 * open_mask[hour_of_week]
        elapsed = week * prefix[HOURS_PER_WEEK] + prefix[hour_of_week] + within
        return _WEEK_ORIGIN + pd.to_timedelta(elapsed, unit="m")

    out = frame.copy()
    for col in ("start_time", "end_time"):
        out[col] = remap(out[col])
    return out


def normalize_log_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "completion_time" in out.columns and "end_time" not in out.columns:
        out = out.rename(columns={"completion_time": "end_time"})
    if "activity" in out.columns and "task" not in out.columns:
        out = out.rename(columns={"activity": "task"})
    for column in ["start_time", "end_time", "enable_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce", utc=True, format="ISO8601")
    return out


# format="ISO8601": these logs mix whole-second and microsecond stamps.
def load_event_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename_map = {}
    if "completion_time" in frame.columns and "end_time" not in frame.columns:
        rename_map["completion_time"] = "end_time"
    if "activity" in frame.columns and "task" not in frame.columns:
        rename_map["activity"] = "task"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    required = {"case_id", "task", "start_time", "end_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing behavior columns in {path}: {sorted(missing)}")
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce", utc=True, format="ISO8601")
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce", utc=True, format="ISO8601")
    if "enable_time" in frame.columns:
        frame["enable_time"] = pd.to_datetime(frame["enable_time"], errors="coerce", utc=True, format="ISO8601")
    return frame.dropna(subset=["case_id", "task", "start_time", "end_time"]).copy()


def service_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Resource-executed service events (drops arrive/complete + resource-less rows)."""
    work = frame.copy()
    work["task"] = work["task"].astype(str)
    resource = work.get("resource", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    return work[resource.ne("") & ~work["task"].str.lower().isin(["arrive", "complete"])].copy()


def _ldm_ready_log(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "resource" not in out.columns:
        out["resource"] = ""
    out["task"] = out["task"].astype(str)
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True, format="ISO8601")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce", utc=True, format="ISO8601")
    return out.dropna(subset=["case_id", "task", "start_time", "end_time"]).copy()


def _ldm_distance(reference_log: pd.DataFrame, candidate_log: pd.DataFrame, metric_fn,
                  *, extract=None, metric=None, **kwargs) -> float:
    """Adapter over log-distance-measures: optionally select events via `extract`, then run metric_fn un-normalised under `metric` (default Wasserstein)."""
    ref = extract(reference_log) if extract else reference_log
    cand = extract(candidate_log) if extract else candidate_log
    if ref.empty or cand.empty:
        return float("nan")
    return float(metric_fn(
        _ldm_ready_log(ref), _LDM_IDS,
        _ldm_ready_log(cand), _LDM_IDS,
        metric=metric or DistanceMetric.WASSERSTEIN, normalize=False, **kwargs))


def cycle_time_distance(reference_log: Path, candidate_log: Path,
                        *, closed_hours: tuple[int, ...] = ()) -> float:
    """CTD alone, so a second reading on a different set of closed hours is cheap."""
    ref = skip_closed_hours(normalize_log_timestamps(load_event_log(reference_log)), closed_hours)
    cand = skip_closed_hours(normalize_log_timestamps(load_event_log(candidate_log)), closed_hours)
    return _ldm_distance(ref, cand, cycle_time_distribution_distance,
                         bin_size=_LDM_BIN_SIZE, metric=DistanceMetric.WASSERSTEIN)


def compute_behavior_distances(
    reference_log: Path,
    candidate_log: Path,
    *,
    closed_hours: tuple[int, ...] = (),
) -> dict[str, float]:
    """CTD, CWD and CED between two logs, nan where one cannot be computed."""
    try:
        from log_distance_measures.circadian_event_distribution import (
            circadian_event_distribution_distance,
        )

        reference = normalize_log_timestamps(load_event_log(reference_log))
        candidate = normalize_log_timestamps(load_event_log(candidate_log))

        ctd = _ldm_distance(
            skip_closed_hours(reference, closed_hours),
            skip_closed_hours(candidate, closed_hours),
            cycle_time_distribution_distance,
            bin_size=_LDM_BIN_SIZE, metric=DistanceMetric.WASSERSTEIN,
        )
        cwd = _ldm_distance(
            reference, candidate, circadian_workforce_distribution_distance,
            extract=service_events, metric=DistanceMetric.WASSERSTEIN,
        )
        try:
            ced = float(circadian_event_distribution_distance(
                _ldm_ready_log(reference), _LDM_IDS,
                _ldm_ready_log(candidate), _LDM_IDS,
                metric=DistanceMetric.WASSERSTEIN, normalize=False,
            ))
        except Exception as err:
            log(f"[WARN] CED failed ({reference_log.name} -> {candidate_log.name}): {err}")
            ced = float("nan")

        out = {"CTD_wass": float(ctd), "CWD_wass": float(cwd), "CED_wass": ced}
        out.update({name: out[f"{name}_wass"] for name in ("CTD", "CWD", "CED")})
        return out
    except Exception as err:
        log(f"[WARN] compute_behavior_distances failed "
            f"({reference_log.name} -> {candidate_log.name}): {err}")
        return {k: float("nan") for k in
                ("CTD", "CWD", "CED", "CTD_wass", "CWD_wass", "CED_wass")}


def summarise(per_seed: pd.DataFrame,
               group_cols: list[str],
               metric_cols: list[str]) -> pd.DataFrame:
    """Mean ± std over seeds for each group; reports n_seeds."""
    result_rows = []
    for keys, sub in per_seed.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        for m in metric_cols:
            if m not in sub.columns:
                row[f"{m}_mean"] = float("nan"); row[f"{m}_std"] = float("nan")
                continue
            vals = sub[m].dropna().values.astype(float)
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{m}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        result_rows.append(row)
    return pd.DataFrame(result_rows)

SYNTH_METRIC_COLS = [
    "D_P_before", "D_P_after", "D_P_improvement",
    "D_rho_high_gap_0.01_improvement",
    "D_rho_updated_row_improvement",
    "D_rho_support_weighted_improvement",
    "ctd_improvement_pct", "cwd_improvement_pct", "ced_improvement_pct",
    "ctd_skipped_improvement_pct",
    "update_mass", "n_pos_updated", "n_neg_updated", "harmful_update_rate",
    # Direction quality: a revision can point the right way and still overshoot.
    "n_harmful_updates", "over_correction_rate",
    "signed_direction_accuracy",
    "signed_direction_accuracy_pos", "signed_direction_accuracy_neg",
]


# ── Synth aggregation ──────────────────────────────────────────────────────────
def _read_csv_safe(p: Path) -> pd.DataFrame:
    """read_csv that tolerates missing or empty (0-byte / header-less) files."""
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _surface_metrics(r: pd.Series) -> dict[str, Any]:
    """One setting's surface-level numbers, as calibration wrote them."""
    out: dict[str, Any] = {
        "D_P_before":      float(r["D_P_before"]),
        "D_P_after":       float(r["D_P_after"]),
        "D_P_improvement": float(r["D_P_before"]) - float(r["D_P_after"]),
    }
    counts = ("n_updated_rows", "n_pos_updated", "n_neg_updated", "n_harmful_updates")
    for col in ("n_updated_rows", "n_pos_updated", "n_neg_updated", "update_mass",
                "harmful_update_rate", "n_harmful_updates", "over_correction_rate",
                "signed_direction_accuracy", "signed_direction_accuracy_pos",
                "signed_direction_accuracy_neg"):
        out[col] = int(r.get(col, 0)) if col in counts else float(r.get(col, float("nan")))
    return out


def _behavior_metrics(pm_s: pd.DataFrame) -> dict[str, Any]:
    """One setting's behavioral distances, collapsed over its re-simulations."""
    out: dict[str, Any] = {}
    for metric in ("ctd", "cwd", "ced", "ctd_skipped"):
        b, a = f"{metric}_before", f"{metric}_after"
        if b not in pm_s.columns or a not in pm_s.columns:
            continue
        out[b] = aggregate_runs(pm_s[b].dropna().values)
        out[a] = aggregate_runs(pm_s[a].dropna().values)
        out[f"{metric}_improvement_pct"] = paired_improvement_pct(pm_s[b].values, pm_s[a].values)
    return out


def collect_synth_per_seed(run_dir: Path, seeds: range) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for s in seeds:
        cal_dir = run_dir / "calibration" / "synth" / f"seed{s:03d}"
        sm = _read_csv_safe(cal_dir / "synth_metrics.csv")
        if sm.empty:
            log(f"[SKIP] synth seed {s:03d}: no synth_metrics.csv")
            continue
        pm = _read_csv_safe(cal_dir / "process_metrics.csv")
        has_pm = not pm.empty and "scenario" in pm.columns

        for scenario in SCENARIOS:
            row: dict[str, Any] = {"seed": s, "scenario": scenario}
            sm_r = sm[sm["scenario"].eq(scenario)]
            if not sm_r.empty:
                row.update(_surface_metrics(sm_r.iloc[0]))
            pm_s = pm[pm["scenario"].eq(scenario)] if has_pm else pm
            if not pm_s.empty:
                row.update(_behavior_metrics(pm_s))
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ── Real aggregation ───────────────────────────────────────────────────────────
def _runs_improved(pr: pd.DataFrame) -> dict[str, int]:
    """How many re-simulations each measure improved on, out of the runs on record."""
    out: dict[str, int] = {"n_runs": int(len(pr))}
    for metric in ("CTD", "CWD", "CED"):
        b, a = f"{metric}_before", f"{metric}_after"
        if b in pr.columns and a in pr.columns:
            bv = pr[b].astype(float).values
            av = pr[a].astype(float).values
            ok = np.isfinite(bv) & np.isfinite(av) & (bv > 0)
            out[f"{metric}_runs_improved"] = int((av[ok] < bv[ok]).sum())
    return out


def collect_real_per_seed(run_dir: Path, seeds: range) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for s in seeds:
        sdir = run_dir / "calibration" / "real" / f"seed{s:03d}"
        cand_metrics = sorted(sdir.glob("*/metrics.csv"))   # seed/<candidate_id>/metrics.csv
        if not cand_metrics:
            log(f"[SKIP] real seed {s:03d}: no metrics.csv found")
            continue
        for mp in cand_metrics:
            df = pd.read_csv(mp)
            if df.empty:
                continue
            rec = df.iloc[0].to_dict()
            rec.setdefault("candidate_id", mp.parent.name)
            # per_run_metrics.csv holds one row per re-simulation; its counts carry
            # "k of n re-simulations improved".
            pr_path = mp.parent / "per_run_metrics.csv"
            if pr_path.exists():
                rec.update(_runs_improved(pd.read_csv(pr_path)))
            all_rows.append(rec)
    return pd.DataFrame(all_rows)


# ── The P^emp empirical-filling baseline ───────────────────────────────────────
def compute_emp_baseline_dp(run_dir: Path, seeds: range,
                              synth_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D_P(P)/D_P(P^cal)/D_P(P^emp) per scenario (P^emp = clip(rho_obs,0,1); row-level, no re-sim)."""
    per_seed_rows: list[dict[str, Any]] = []
    for s in seeds:
        row_feat_path = run_dir / "extraction" / "synth" / f"seed{s:03d}" / "row_features_mapped.csv"
        if not row_feat_path.exists():
            continue
        f = pd.read_csv(row_feat_path, low_memory=False)
        p     = f["p"].astype(float)
        pt    = f["p_true"].astype(float)
        p_emp = f["rho_obs"].astype(float).clip(0.0, 1.0)
        f = f.assign(_before=(pt - p).abs(), _emp=(pt - p_emp).abs())
        agg_cols = {"D_P_before": ("_before", "mean"), "D_P_emp": ("_emp", "mean")}

        # The calibrated setting, and the same distance over the working hours.
        pcal_path = run_dir / "calibration" / "synth" / f"seed{s:03d}" / "p_cal_long.csv"
        if pcal_path.exists():
            keys = ["unit_id", "resource", "weekday", "time_bin"]
            pc = pd.read_csv(pcal_path)
            f = f.merge(pc, on=keys, how="left")
            f["_cal"] = (f["p_true"].astype(float) - f["p_cal"].astype(float)).abs()
            working = (f["weekday"] < 5) & (f["time_bin"] >= 8) & (f["time_bin"] < 16)
            f["_cal_work"] = f["_cal"].where(working)
            agg_cols["D_P_cal"] = ("_cal", "mean")
            agg_cols["D_P_cal_working_hours"] = ("_cal_work", "mean")

        g = f.groupby("scenario").agg(**agg_cols).reset_index()
        g["seed"] = s
        per_seed_rows.append(g)

    if not per_seed_rows:
        return pd.DataFrame(), pd.DataFrame()
    perseed = pd.concat(per_seed_rows, ignore_index=True)
    agg = perseed.groupby("scenario").agg(
        D_P_before_check=("D_P_before", "mean"),
        D_P_emp_mean=("D_P_emp", "mean"),
        D_P_emp_std=("D_P_emp", "std"),
    ).reset_index()

    nc = synth_summary.set_index("scenario")

    out: list[dict[str, Any]] = []
    for sc in SCEN_ORDER:
        if sc not in nc.index:
            continue
        a = agg[agg["scenario"].eq(sc)]
        if a.empty:
            continue
        a = a.iloc[0]
        out.append({
            "scenario":       sc,
            "scenario_short": SCEN_SHORT.get(sc, sc),
            "D_P_P":          float(nc.loc[sc, "D_P_before_mean"]),
            "D_P_Pcal_mean":  float(nc.loc[sc, "D_P_after_mean"]),
            "D_P_Pcal_std":   float(nc.loc[sc, "D_P_after_std"]),
            "D_P_Pemp_mean":  float(a["D_P_emp_mean"]),
            "D_P_Pemp_std":   float(a["D_P_emp_std"]),
        })
    lead = [c for c in ("seed", "scenario") if c in perseed.columns]
    perseed = perseed[lead + [c for c in perseed.columns if c not in lead]]
    return pd.DataFrame(out), perseed.sort_values(lead).reset_index(drop=True)


# ── Grouping-metric reader (--report-only) ─────────────────────────────────────
# improvement% is the paired per-run figure from the per_seed CSVs, at its median over seeds.
def _synth_improvement(run_dir: Path):
    d = _read_csv_safe(run_dir / "evaluation" / "synth_per_seed.csv")
    if d.empty or "scenario" not in d.columns:
        return None
    rows = []
    for met in ("cwd", "ced", "ctd", "ctd_skipped"):
        pcol, bcol, acol = f"{met}_improvement_pct", f"{met}_before", f"{met}_after"
        if pcol not in d:
            continue
        for scen, g in d.groupby("scenario"):
            v = g[pcol].dropna().values
            if not len(v):
                continue
            rows.append({"scenario": scen, "metric": met.upper(),
                         "imp%": round(float(np.median(v)), 1),
                         "wins": f"{int((v > 0).sum())}/{len(v)}",
                         "before": round(float(np.median(g[bcol].dropna())), 3) if bcol in g else float("nan"),
                         "after": round(float(np.median(g[acol].dropna())), 3) if acol in g else float("nan")})
    return pd.DataFrame(rows)


def _real_improvement(run_dir: Path):
    # Per candidate: the seed's paired improvement%, then its median over seeds.
    d = _read_csv_safe(run_dir / "evaluation" / "real_per_seed.csv")
    if d.empty or "candidate_id" not in d.columns:
        return None
    rows = []
    for met in ("CWD", "CED", "CTD"):
        pcol, bcol, acol = f"{met}_improvement_pct", f"{met}_before", f"{met}_after"
        if pcol not in d.columns:
            continue
        for cid, g in d.groupby("candidate_id"):
            v = g[pcol].dropna().values
            if not len(v):
                continue
            rows.append({"candidate": str(cid), "metric": met,
                         "imp%": round(float(np.median(v)), 1),
                         "wins": f"{int((v > 0).sum())}/{len(v)}",
                         "before": round(float(np.median(g[bcol].dropna())), 3) if bcol in g else float("nan"),
                         "after": round(float(np.median(g[acol].dropna())), 3) if acol in g else float("nan")})
    return pd.DataFrame(rows)


def _print_block(title: str, frame: pd.DataFrame, index: str) -> None:
    """improvement%, seeds improved, and the absolute levels the percentage came from."""
    print(title)
    print(frame.pivot(index=index, columns="metric", values="imp%").to_string())
    print("  seeds improved:")
    print(frame.pivot(index=index, columns="metric", values="wins").to_string())
    print("  median distance before -> after:")
    lv = frame.assign(level=frame.apply(lambda r: f"{r['before']:g} -> {r['after']:g}", axis=1))
    print(lv.pivot(index=index, columns="metric", values="level").to_string())


def report_improvement(run_dir: Path) -> None:
    s = _synth_improvement(run_dir)
    if s is not None and not s.empty:
        _print_block("[SYNTH] improvement% (paired per run, median within seed then over seeds)",
                     s, "scenario")
    r = _real_improvement(run_dir)
    if r is not None and not r.empty:
        _print_block("[REAL] improvement% (paired per run, median within seed then over seeds)",
                     r, "candidate")
    if (s is None or s.empty) and (r is None or r.empty):
        print("(no synth_per_seed.csv or real_per_seed.csv found — run the full evaluation first)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",       required=True)
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-real-seeds",  type=int, default=10)
    ap.add_argument("--report-only",   action="store_true",
                    help="Print the improvement%% of a completed run and exit (was grouping_metric.py).")
    args = ap.parse_args()

    run_dir  = Path(args.run_dir)
    if args.report_only:
        report_improvement(run_dir)
        return
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    synth_seeds = run_seeds(args.n_synth_seeds)
    real_seeds  = run_seeds(args.n_real_seeds)

    # ── Synthetic ──────────────────────────────────────────────────────────────
    log("Collecting synthetic per-seed metrics …")
    synth_ps = collect_synth_per_seed(run_dir, synth_seeds)
    synth_ps.to_csv(eval_dir / "synth_per_seed.csv", index=False)
    log(f"synth_per_seed.csv: {len(synth_ps)} rows")

    synth_sum = pd.DataFrame()
    if not synth_ps.empty:
        metric_cols = [c for c in SYNTH_METRIC_COLS if c in synth_ps.columns]
        # feeds compute_emp_baseline_dp; not written to disk (nothing reads it).
        synth_sum   = summarise(synth_ps, ["scenario"], metric_cols)

    # ── Real ──────────────────────────────────────────────────────────────────
    log("Collecting real per-seed metrics …")
    real_ps = collect_real_per_seed(run_dir, real_seeds)
    real_ps.to_csv(eval_dir / "real_per_seed.csv", index=False)
    log(f"real_per_seed.csv: {len(real_ps)} rows")

    # ── The P^emp empirical baseline ───────────────────────────────────────────
    emp_df = pd.DataFrame()
    if not synth_sum.empty:
        log("Computing emp_baseline_dp …")
        emp_df, emp_ps = compute_emp_baseline_dp(run_dir, synth_seeds, synth_sum)
        if not emp_df.empty:
            emp_df.to_csv(eval_dir / "emp_baseline_dp.csv", index=False)
            log(f"emp_baseline_dp.csv: {len(emp_df)} rows")
        if not emp_ps.empty:
            emp_ps.to_csv(eval_dir / "emp_baseline_per_seed.csv", index=False)
            log(f"emp_baseline_per_seed.csv: {len(emp_ps)} rows")

    log("Evaluation complete.")


if __name__ == "__main__":
    main()
