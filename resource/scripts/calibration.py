#!/usr/bin/env python3
"""Build P_cal, re-simulate, and score the rho distance and behavioral improvement."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core
from _core import (
    resolve_cfg,
    ROOT,
    DEFAULT_OBS_LOG,
    log,
    merge_busy_into_rows,
    paired_improvement_pct,
    SCENARIOS,
    SYNTH_VAL_WINDOW,
    synth_val_reference,
    filter_log_window,
)
from evaluation import (  # metric defs live in evaluation
    compute_behavior_distances,
    cycle_time_distance,
    REAL_CLOSED_HOURS,
    SYNTH_CLOSED_HOURS,
)



# ── Real availability-JSON writers ─────────────────────────────────────────────

WEEKDAY = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]

RULE_ID = "p_cal"   # names the calibrated availability json: <candidate>_<RULE_ID>.json


def time_string(bin_id: int, bin_minutes: int) -> str:
    m = int(bin_id) * int(bin_minutes)
    if m >= 24 * 60:
        return "23:59:59"
    return f"{m // 60:02d}:{m % 60:02d}:00"

def fuzzy_intervals(day: pd.DataFrame, bin_minutes: int) -> list[dict[str, Any]]:
    """One weekday's nonzero probabilities, as Prosimos fuzzy intervals."""
    out = []
    for row in day.sort_values("time_bin").itertuples(index=False):
        p = float(row.prob)
        if p > 0:
            b = int(row.time_bin)
            out.append({"begin_time": time_string(b, bin_minutes),
                        "end_time": time_string(b + 1, bin_minutes),
                        "probability": p})
    return out

def write_real_json(base_json: Path, surface: pd.DataFrame, prob_col: str, out_json: Path, bin_minutes: int) -> None:
    payload = json.loads(base_json.read_text(encoding="utf-8"))
    work = surface.copy()
    work["resource"] = work["resource"].astype(str)
    work["prob"] = pd.to_numeric(work[prob_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["weekday"] = work["weekday"].astype(int)
    by_res = {r: g for r, g in work.groupby("resource")}
    for cal in payload.get("resource_calendars", []) or []:
        group = by_res.get(str(cal.get("id", "")).removesuffix("_timetable"))
        if group is None:
            continue
        cal["availability_probabilities"] = [
            {"week_day": WEEKDAY[d],
             "fuzzy_intervals": fuzzy_intervals(group[group["weekday"].eq(d)], bin_minutes)}
            for d in range(7)
        ]
        cal.pop("time_periods", None)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_real_manifest_for_rule(updates: pd.DataFrame, base_manifest: list[dict[str, Any]], rule: dict[str, Any], rule_dir: Path) -> Path:
    manifest = json.loads(json.dumps(base_manifest))
    by_unit = {item["candidate_id"]: item for item in manifest if item["target"] == "real"}
    for unit, rows in updates.groupby("unit_id"):
        item = by_unit[unit]
        cfg = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        udir = rule_dir / "configs" / str(unit)
        udir.mkdir(parents=True, exist_ok=True)
        pcal = udir / "p_cal_surface.csv"
        rows[["resource", "weekday", "time_bin", "p_cal"]].to_csv(pcal, index=False)
        cfg["p_cal_surface_csv"] = str(pcal)
        pcal_json = udir / f"{unit}_{rule['rule_id']}.json"
        write_real_json(Path(cfg["base_json"]), rows, "p_cal", pcal_json, int(cfg["bin_minutes"]))
        cfg["p_cal_json"] = str(pcal_json)
        cfg_path = udir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        item["config_json"] = str(cfg_path)
        item["rule_id"] = rule["rule_id"]
    path = rule_dir / "manifest_calibrated.json"
    path.write_text(json.dumps({"manifest": manifest}, indent=2), encoding="utf-8")
    return path


# ── The calibration rule ───────────────────────────────────────────────────────
def diagnosed_signed_change(
        p:               pd.Series,
        g:               pd.Series,
        q_pos:           pd.Series,
        theta_pos:         float,
        theta_neg_q:     float,
        theta_neg_delta: float,
        g_group:         pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The diagnosed signed change, per availability probability."""
    p     = p.astype(float);  g = g.astype(float);  q_pos = q_pos.astype(float)
    g_gate = g if g_group is None else g_group.astype(float)

    sel_pos = g.gt(0) & q_pos.ge(theta_pos)
    sel_neg = g_gate.lt(-theta_neg_delta) & q_pos.lt(theta_neg_q)

    dp = pd.Series(np.where(sel_pos, q_pos.values * g.values, 0.0),
                    index=p.index).clip(lower=0.0)
    dn = pd.Series(np.where(sel_neg, -g.values, 0.0),
                    index=p.index).clip(lower=0.0)

    delta     = dp - dn
    p_cal     = (p + delta).clip(0.0, 1.0)
    delta = p_cal - p
    return delta, dp, dn


def build_p_cal_surface(rows_mapped: pd.DataFrame, delta: pd.Series) -> pd.DataFrame:
    df = rows_mapped[["unit_id", "resource", "weekday", "time_bin", "p"]].copy()
    df["p_cal"] = (df["p"].astype(float) + delta.astype(float).values).clip(0.0, 1.0)
    return df[["unit_id", "resource", "weekday", "time_bin", "p_cal"]]


def write_pcal_manifest(p_cal_long: pd.DataFrame,
                         base_manifest: list[dict[str, Any]],
                         method_dir: Path) -> Path:
    """Per-unit calibrated config JSONs + combined manifest_calibrated.json."""
    method_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(base_manifest))   # deep copy
    by_unit  = {item["scenario_id"]: item
                for item in manifest if item.get("target") == "synthetic"}
    for unit, unit_rows in p_cal_long.groupby("unit_id"):
        if unit not in by_unit:
            continue
        item     = by_unit[unit]
        cfg      = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        udir     = method_dir / "configs" / str(unit)
        udir.mkdir(parents=True, exist_ok=True)
        pcal_csv = udir / "p_cal_surface.csv"
        unit_rows[["resource", "weekday", "time_bin", "p_cal"]].to_csv(pcal_csv, index=False)
        cfg["p_cal_surface_csv"] = str(pcal_csv)
        cfg_path = udir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        item["config_json"] = str(cfg_path)
    manifest_path = method_dir / "manifest_calibrated.json"
    manifest_path.write_text(json.dumps({"manifest": manifest}, indent=2), encoding="utf-8")
    return manifest_path


# ── Per-rep delta computation ──────────────────────────────────────────────────
def compute_delta(
        rows_mapped:     pd.DataFrame,
        groups:          pd.DataFrame,
        q_pos_blk:       pd.Series,
        theta_pos:         float,
        theta_neg_q:     float,
        theta_neg_delta: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply the calibration rule for one rep; return (delta, dp, dn) at row level."""
    group_ids  = groups["group_id"].astype(str).values
    row_blk_id = rows_mapped["group_id"].astype(str)

    q_pos_lookup = pd.Series(q_pos_blk.reindex(group_ids).fillna(0.0).values, index=group_ids)
    q_pos_row    = row_blk_id.map(q_pos_lookup).fillna(0.0).astype(float)

    p_row = rows_mapped["p"].astype(float)
    g_row = rows_mapped["rho_diff"].astype(float)
    # Mean difference over each row's aggregation group, the unit theta_neg_delta is set in.
    g_grp = g_row.groupby(row_blk_id).transform("mean")

    return diagnosed_signed_change(
        p_row, g_row, q_pos_row, theta_pos, theta_neg_q, theta_neg_delta, g_group=g_grp)



# ── Seeded RNG ─────────────────────────────────────────────────────────────────



# ── Windowing helpers ──────────────────────────────────────────────────────────
def write_windowed_copy(src: Path, window: tuple[str, str] | None, dst: Path) -> Path:
    df = pd.read_csv(src)
    df = filter_log_window(df, window)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return dst


def _window_each(paths: list[Path], window, dst: Path) -> list[Path]:
    """Window each log to `window` and keep them separate, one file per run."""
    out: list[Path] = []
    for p in paths:
        try:
            wp = write_windowed_copy(
                Path(p), window, dst.with_name(dst.stem + f"_{Path(p).stem}.csv"))
        except Exception as err:
            log(f"[WARN] {Path(p).name}: could not window to {window}, scoring the "
                f"whole log instead ({err})")
            wp = Path(p)
        if Path(wp).exists():
            out.append(Path(wp))
    return out


# ── Re-simulation ──────────────────────────────────────────────────────────────
def simulate_and_extract_pcal(method_dir: Path, manifest_path: Path,
                                num_runs: int,
                                bin_minutes: int = 60,
                                states: set[str] | None = None,
                                skip_pcal_scenarios: set[str] | None = None,
                                sim_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate each synthetic scenario then extract. skip_pcal_scenarios skips no-update (P_CAL≡P) re-sims."""
    if states is None:
        states = {"P_CAL"}
    skip_pcal_scenarios = skip_pcal_scenarios or set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest"]
    rows: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("target") != "synthetic":
            continue
        item_states = set(states)
        if item.get("scenario_id") in skip_pcal_scenarios:
            item_states.discard("P_CAL")
            stale = method_dir / "logs" / "synthetic" / str(item["scenario_id"]) / "P_CAL"
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
        if not item_states:
            continue
        log(f"  re-sim {item.get('scenario_id')}: "
            f"{'/'.join(sorted(item_states))} x {num_runs} runs")
        rows.extend(
            _core.simulate_synthetic_config(
                resolve_cfg(item["config_json"]),
                method_dir / "logs",
                num_runs,
                item_states,
                sim_seed=sim_seed,
            )
        )
    log_manifest = method_dir / "logs" / "log_manifest.csv"
    log_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(log_manifest, index=False)

    _core.run_extract(manifest_path, log_manifest, method_dir, bin_minutes)
    rho_pcal = pd.read_csv(method_dir / "rho_profiles.csv")
    logs     = pd.read_csv(log_manifest)
    return rho_pcal, logs


# ── Metric helpers ─────────────────────────────────────────────────────────────
def synth_metrics(rows: pd.DataFrame, delta_col: str, dp_col: str | None,
                   dn_col: str | None) -> pd.DataFrame:
    p = rows["p"].astype(float)
    pt = rows["p_true"].astype(float)
    delta = rows[delta_col].astype(float)
    p_cal = (p + delta).clip(0, 1)
    upd = delta.abs().gt(1e-12)
    pos_upd = (delta > 1e-12)
    neg_upd = (delta < -1e-12)
    needed = pt - p
    needed_pos = (needed > 1e-9)
    needed_neg = (needed < -1e-9)
    correct_dir = ((pos_upd & needed_pos) | (neg_upd & needed_neg)) & upd
    harmful = ((delta * needed) < -1e-12) & upd
    over_corr = (np.sign(delta) == np.sign(needed)) & (delta.abs() > needed.abs() + 1e-9) & upd
    out = []
    for s in SCENARIOS:
        idx = rows.index[rows["scenario"].eq(s)]
        if len(idx) == 0:
            continue
        n_upd = int(upd.loc[idx].sum())
        n_pos = int(pos_upd.loc[idx].sum())
        n_neg = int(neg_upd.loc[idx].sum())
        sda_overall = float(correct_dir.loc[idx].sum() / max(n_upd, 1)) if n_upd else float("nan")
        neg_idx = idx[neg_upd.loc[idx].values]
        pos_idx = idx[pos_upd.loc[idx].values]
        sda_neg = (float(correct_dir.loc[neg_idx].sum() / max(n_neg, 1))
                    if n_neg else float("nan"))
        sda_pos = (float(correct_dir.loc[pos_idx].sum() / max(n_pos, 1))
                    if n_pos else float("nan"))
        dp_mass = float(rows.loc[idx, dp_col].sum()) if dp_col and dp_col in rows.columns else float(delta.loc[idx].clip(lower=0).sum())
        dn_mass = float(rows.loc[idx, dn_col].sum()) if dn_col and dn_col in rows.columns else float((-delta.loc[idx].clip(upper=0)).sum())
        out.append({
            "scenario": s, "n_rows": int(len(idx)),
            "D_P_before": float((pt.loc[idx] - p.loc[idx]).abs().mean()),
            "D_P_after": float((pt.loc[idx] - p_cal.loc[idx]).abs().mean()),
            "D_P_change": float((pt.loc[idx] - p_cal.loc[idx]).abs().mean() - (pt.loc[idx] - p.loc[idx]).abs().mean()),
            "n_updated_rows": n_upd,
            "n_pos_updated": n_pos,
            "n_neg_updated": n_neg,
            "update_mass": float(delta.loc[idx].abs().sum()),
            "positive_update_mass": float(delta.loc[idx].clip(lower=0).sum()),
            "negative_update_mass": float((-delta.loc[idx].clip(upper=0)).sum()),
            "delta_pos_component_mass": dp_mass,
            "delta_neg_component_mass": dn_mass,
            "harmful_update_rate": float(harmful.loc[idx].sum() / max(n_upd, 1)) if n_upd else 0.0,
            "n_harmful_updates": int(harmful.loc[idx].sum()),
            "over_correction_rate": float(over_corr.loc[idx].sum() / max(n_upd, 1)) if n_upd else 0.0,
            "signed_direction_accuracy": sda_overall,
            "signed_direction_accuracy_pos": sda_pos,
            "signed_direction_accuracy_neg": sda_neg,
        })
    return pd.DataFrame(out)


def _score_pair(item: tuple[str, int, str, str, str]) -> dict[str, Any]:
    """Score one (scenario, run) pair against the holdout reference."""
    scenario, run_idx, ref_s, b_s, a_s = item
    ref, b, a = Path(ref_s), Path(b_s), Path(a_s)
    d_before = compute_behavior_distances(ref, b)
    d_after = compute_behavior_distances(ref, a)
    row: dict[str, Any] = {
        "scenario": scenario, "run": run_idx,
        "ctd_skipped_before": cycle_time_distance(ref, b, closed_hours=SYNTH_CLOSED_HOURS),
        "ctd_skipped_after": cycle_time_distance(ref, a, closed_hours=SYNTH_CLOSED_HOURS),
    }
    for key in ("CTD", "CWD", "CED",
                "CTD_wass", "CWD_wass", "CED_wass"):
        row[f"{key.lower()}_before"] = d_before[key]
        row[f"{key.lower()}_after"] = d_after[key]
    return row


def compute_synth_process_metrics(
    base_log_manifest: pd.DataFrame,
    pcal_log_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """CTD/CWD/CED before/after per scenario, on the synthetic holdout."""
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    def _runs(manifest_df: pd.DataFrame, scenario: str, state: str) -> list[Path]:
        subset = manifest_df[
            manifest_df["scenario_id"].eq(scenario) & manifest_df["state"].eq(state)
        ]
        out = []
        for p in subset["log_csv"].tolist():
            q = Path(p)
            out.append(q if q.is_absolute() else ROOT / q)
        return out

    def _windowed(src: Path, dst: Path) -> Path | None:
        try:
            frame = filter_log_window(pd.read_csv(src), SYNTH_VAL_WINDOW)
        except Exception as err:
            log(f"[WARN] {src.name}: could not window to {SYNTH_VAL_WINDOW} ({err})")
            return None
        if frame.empty:
            return None
        frame.to_csv(dst, index=False)
        return dst

    scenarios = sorted(base_log_manifest["scenario_id"].unique())
    tmp = Path(tempfile.mkdtemp())
    items: list[tuple[str, int, str, str, str]] = []
    rows: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            shipped = synth_val_reference(scenario)
            if shipped is None:
                log(f"[WARN] synth process metrics: no shipped reference for {scenario}; skipping")
                continue
            before_runs = _runs(base_log_manifest, scenario, "P")
            after_runs  = _runs(pcal_log_manifest, scenario, "P_CAL")
            if not before_runs:
                log(f"[WARN] synth process metrics: no P logs for {scenario}; skipping")
                continue
            if not after_runs:
                # No-update scenario (P_CAL≡P, re-sim skipped): omit the before/after row.
                log(f"synth process metrics: no P_CAL logs for {scenario} "
                    f"(P_CAL≡P / skipped); omitting before/after")
                continue

            log(f"  process metrics {scenario}: {len(before_runs)} before / "
                f"{len(after_runs)} after runs")
            ref = _windowed(shipped, tmp / f"{scenario}_ref.csv")
            if ref is None:
                log(f"[WARN] synth process metrics: empty holdout reference for {scenario}; skipping")
                continue
            for run_idx, (b_src, a_src) in enumerate(zip(before_runs, after_runs)):
                b = _windowed(b_src, tmp / f"{scenario}_before{run_idx}.csv")
                a = _windowed(a_src, tmp / f"{scenario}_after{run_idx}.csv")
                if b is None or a is None:
                    continue
                items.append((scenario, run_idx, str(ref), str(b), str(a)))

        # Every (scenario, run) pair is independent; map preserves input order.
        if items:
            want = int(os.environ.get("RESIM_WORKERS", "0") or 0)
            workers = want if want > 0 else min((os.cpu_count() or 1), 8)
            workers = max(1, min(workers, len(items)))
            log(f"  scoring {len(items)} (scenario, run) pairs on {workers} worker(s)")
            if workers == 1:
                rows = [_score_pair(it) for it in items]
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(_score_pair, items))
            log(f"  scored {len(rows)} (scenario, run) pairs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return pd.DataFrame(rows)

# ── Synthetic calibration ──────────────────────────────────────────────────────
def delta_for_seed(rows: pd.DataFrame, groups: pd.DataFrame, est_dir: Path,
                   thr_df: pd.DataFrame, theta_pos: float):
    """The revision from the seed's Q+ scores, with its two branch masses."""
    q_pos_df = pd.read_csv(est_dir / "q_pos.csv")
    q_pos_blk = pd.Series(q_pos_df["q_pos"].values,
                          index=q_pos_df["group_id"].astype(str).values)
    thr = thr_df.iloc[0]
    return compute_delta(rows, groups, q_pos_blk, theta_pos,
                         float(thr["theta_neg_q"]), float(thr["theta_neg_delta"]))


def calibrate_synth_seed(seed: int, run_dir: Path, num_runs: int,
                           manifest_items: list[dict],
                           bin_minutes: int = 60) -> None:
    ext_dir  = run_dir / "extraction"  / "synth" / f"seed{seed:03d}"
    est_dir  = run_dir / "estimation"  / "synth" / f"seed{seed:03d}"
    diag_dir = run_dir / "diagnosis"   / "synth" / f"seed{seed:03d}"
    cal_dir  = run_dir / "calibration" / "synth" / f"seed{seed:03d}"
    cal_dir.mkdir(parents=True, exist_ok=True)

    rows_mapped = pd.read_csv(ext_dir / "row_features_mapped.csv", low_memory=False)
    groups      = pd.read_csv(ext_dir / "group_features.csv", low_memory=False)
    thr_df      = pd.read_csv(diag_dir / "neg_thresholds.csv")
    theta_pos_cfg = json.loads((est_dir / "theta_pos.json").read_text(encoding="utf-8"))
    theta_pos     = float(theta_pos_cfg["theta_pos"])

    groups["group_id"] = groups["group_id"].astype(str)

    delta, dp, dn = delta_for_seed(rows_mapped, groups, est_dir, thr_df, theta_pos)

    delta_df = pd.DataFrame({"delta": delta.values, "dp": dp.values, "dn": dn.values},
                              index=rows_mapped.index)
    delta_df.to_csv(cal_dir / "delta_rows.csv", index=False)

    p_cal_long = build_p_cal_surface(rows_mapped, delta)
    p_cal_long.to_csv(cal_dir / "p_cal_long.csv", index=False)
    mpath = write_pcal_manifest(p_cal_long, manifest_items, cal_dir)

    # One salt for P/P_CAL. P_TRUE is not re-simulated here.
    log(f"seed {seed:03d}: re-simulating P/P_CAL ({num_runs} runs/scenario)")

    # P_CAL≡P scenarios: skip the re-sim.
    update_mass = pd.Series(delta.abs().values).groupby(
        rows_mapped["unit_id"].values).sum()
    skip_pcal = {str(u) for u, m in update_mass.items() if m <= 1e-9}
    if skip_pcal:
        log(f"seed {seed:03d}: no-update scenarios (P_CAL≡P, skipped): "
            f"{sorted(skip_pcal)}")

    _rho_pcal, pcal_logs = simulate_and_extract_pcal(
        cal_dir, mpath, num_runs,
        bin_minutes=bin_minutes,
        states={"P", "P_CAL"},
        skip_pcal_scenarios=skip_pcal,
        sim_seed=seed)

    # Candidates come from the salted re-sim; the reference is the shipped holdout.
    log(f"seed {seed:03d}: scoring process metrics")
    proc = compute_synth_process_metrics(pcal_logs, pcal_logs)
    proc.to_csv(cal_dir / "process_metrics.csv", index=False)

    rows_mapped["_d"]  = delta.values
    rows_mapped["_dp"] = dp.values
    rows_mapped["_dn"] = dn.values
    sm = synth_metrics(rows_mapped, "_d", "_dp", "_dn")
    sm.to_csv(cal_dir / "synth_metrics.csv", index=False)


# ── Real calibration ───────────────────────────────────────────────────────────
def calibrate_real_seed(seed: int, run_dir: Path,
                          candidate_id: str,
                          real_rows_df: pd.DataFrame,
                          real_groups:  pd.DataFrame,
                          base_manifest: list[dict],
                          val_obs_path:  Path,
                          num_runs: int = 1) -> dict[str, Any]:
    """Calibrate one candidate for one seed. Invariant: real_rows_df/real_groups/base_manifest already filtered to this candidate_id so re-sim runs only this setting."""
    est_dir  = run_dir / "estimation"  / "real" / f"seed{seed:03d}"
    diag_dir = run_dir / "diagnosis"   / "real" / f"seed{seed:03d}"
    cal_dir  = run_dir / "calibration" / "real" / f"seed{seed:03d}" / candidate_id
    cal_dir.mkdir(parents=True, exist_ok=True)

    theta_pos_cfg = json.loads((est_dir / "theta_pos.json").read_text(encoding="utf-8"))
    theta_pos     = float(theta_pos_cfg["theta_pos"])
    thr_df      = pd.read_csv(diag_dir / "neg_thresholds.csv")
    real_groups["group_id"] = real_groups["group_id"].astype(str)

    rows = real_rows_df.copy()
    rows["group_id"] = rows["group_id"].astype(str)

    delta, dp, dn = delta_for_seed(rows, real_groups, est_dir, thr_df, theta_pos)

    delta_df = pd.DataFrame({"delta": delta.values, "dp": dp.values, "dn": dn.values},
                              index=rows.index)
    delta_df.to_csv(cal_dir / "delta_rows.csv", index=False)

    rows["delta_p_hat"] = delta.values
    rows["p_cal"]       = (rows["p"].astype(float) + rows["delta_p_hat"]).clip(0, 1)
    rows["update_applied"] = rows["delta_p_hat"].abs().gt(1e-12).astype(int)
    rows["rule_id"]   = RULE_ID; rows["rule_name"] = RULE_ID; rows["rule_role"] = "capacity_adjusted"
    rule = {"rule_id": RULE_ID}
    man_path = write_real_manifest_for_rule(rows, base_manifest, rule, cal_dir)

    log(f"real seed {seed:03d}: [{candidate_id}] re-simulating P_CAL ({num_runs} runs)")
    shutil.rmtree(cal_dir / "logs", ignore_errors=True)
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "_core.py"), "--entry", "simulate",
           "--target", "real",
           "--manifest-json",  str(man_path.resolve()),
           "--output-root",    str(cal_dir.resolve()),
           "--num-runs", str(num_runs), "--state", "P_CAL",
           "--sim-seed", str(seed)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    log_manifest = cal_dir / "logs" / "log_manifest.csv"
    pcal_log_df  = pd.read_csv(log_manifest)
    pcal_log     = pcal_log_df[pcal_log_df["state"].eq("P_CAL")]

    cfg       = json.loads(resolve_cfg(base_manifest[0]["config_json"]).read_text())
    resources = sorted(pd.read_csv(resolve_cfg(cfg["p_surface_csv"]))["resource"].astype(str).unique())
    bin_mins  = int(cfg.get("bin_minutes", 60))
    log_paths = [Path(p) for p in pcal_log["log_csv"].tolist()]
    rho_after = _core.rho_from_logs(
        log_paths, bin_mins, resources,
        week_window=_core.sim_window(_core.REAL_TRAIN_WINDOW), pool_workweek=False).rename(
        columns={"rho": "rho_after"})
    rho_after.to_csv(cal_dir / "rho_after.csv", index=False)

    rows2 = rows.copy()
    rows2 = rows2.merge(rho_after[["resource", "weekday", "time_bin", "rho_after"]],
                         on=["resource", "weekday", "time_bin"], how="left")
    rows2["rho_after"] = rows2["rho_after"].fillna(0.0)
    rows2["abs_err"]   = (rows2["rho_obs"].astype(float) - rows2["rho_after"]).abs()
    g_abs    = rows2["rho_diff"].astype(float).abs()
    hg_mask  = g_abs.ge(0.01)
    act_mask = rows2["delta_p_hat"].abs().gt(1e-12)
    sup      = rows2.get("support_days_obs", pd.Series(1.0, index=rows2.index)).astype(float).clip(lower=0)
    D_high   = float(rows2.loc[hg_mask, "abs_err"].mean()) if hg_mask.any() else float("nan")
    D_act    = float(rows2.loc[act_mask, "abs_err"].mean()) if act_mask.any() else float("nan")
    D_sup    = float((rows2["abs_err"] * sup).sum() / sup.sum()) if float(sup.sum()) > 0 else float(rows2["abs_err"].mean())

    _nan = {k: float("nan") for k in
            ("CTD", "CWD", "CED", "CTD_wass", "CWD_wass", "CED_wass")}

    after_runs: list[Path] = []
    try:
        after_paths = [Path(p) for p in pcal_log["log_csv"].tolist()]
        after_runs = _window_each(
            after_paths, _core.REAL_VAL_WINDOW,
            cal_dir / "after_valwin_concat.csv")
    except Exception as e:
        log(f"real seed {seed:03d}: could not resolve P_CAL logs: {e}")

    before_runs: list[Path] = []
    try:
        cfg_before = json.loads(resolve_cfg(base_manifest[0]["config_json"]).read_text())
        p_sim_files = sorted(glob.glob(str(cfg_before.get("input_log_glob", ""))))
        if p_sim_files:
            before_runs = _window_each(
                [Path(p) for p in p_sim_files], _core.REAL_VAL_WINDOW,
                cal_dir / "before_valwin_concat.csv")
    except Exception as e:
        log(f"real seed {seed:03d}: could not resolve P logs: {e}")

    # One distance per re-simulation, paired by index.
    _keys = ("CTD", "CWD", "CED", "CTD_wass", "CWD_wass", "CED_wass")
    per_run: list[dict] = []
    if len(before_runs) != len(after_runs):
        log(f"real seed {seed:03d}: [{candidate_id}] {len(before_runs)} before vs "
            f"{len(after_runs)} after runs; pairing the first {min(len(before_runs), len(after_runs))}")
    n_pairs = min(len(before_runs), len(after_runs))
    for i, (fb, fa) in enumerate(zip(before_runs, after_runs)):
        # Real: BPI-2012 declares no calendar, so nothing is subtracted -- see
        # evaluation.REAL_CLOSED_HOURS.
        log(f"real seed {seed:03d}: [{candidate_id}] distances for run "
            f"{i + 1}/{n_pairs}")
        db = compute_behavior_distances(val_obs_path, fb, closed_hours=REAL_CLOSED_HOURS)
        da = compute_behavior_distances(val_obs_path, fa, closed_hours=REAL_CLOSED_HOURS)
        per_run.append({"seed": seed, "candidate_id": candidate_id, "run": i,
                        **{f"{k}_before": db.get(k, float("nan")) for k in _keys},
                        **{f"{k}_after": da.get(k, float("nan")) for k in _keys}})
    if per_run:
        pr = pd.DataFrame(per_run)
        pr.to_csv(cal_dir / "per_run_metrics.csv", index=False)
        d_before = {k: _core.aggregate_runs(pr[f"{k}_before"].values) for k in _keys}
        d_after  = {k: _core.aggregate_runs(pr[f"{k}_after"].values)  for k in _keys}
        # The reported figure, never recomputed from the absolute levels above.
        d_pct = {k: paired_improvement_pct(pr[f"{k}_before"].values, pr[f"{k}_after"].values)
                 for k in _keys}
    else:
        log(f"real seed {seed:03d}: [{candidate_id}] no usable re-simulation pair")
        d_before, d_after, d_pct = dict(_nan), dict(_nan), dict(_nan)

    metrics: dict[str, Any] = {
        "seed": seed, "candidate_id": candidate_id,
        "D_rho_high_gap": D_high, "D_rho_action_local": D_act,
        "D_rho_support_weighted": D_sup,
        "CTD_before":  d_before["CTD"],
        "CTD_after":   d_after["CTD"],
        "CWD_before":  d_before["CWD"],
        "CWD_after":   d_after["CWD"],
        "CED_before":  d_before["CED"],
        "CED_after":   d_after["CED"],
        **{f"{k}_{ph}": (d_before if ph == "before" else d_after)[k]
           for k in ("CTD_wass", "CWD_wass", "CED_wass")
           for ph in ("before", "after")},
        **{f"{k}_improvement_pct": d_pct[k] for k in _keys},
        "run_aggregator": _core.RUN_AGGREGATOR,
        "n_runs_scored": len(per_run),
        "update_mass": float(delta.abs().sum()),
        "n_active_rows": int(act_mask.sum()),
        "n_pos": int((delta > 1e-12).sum()),
        "n_neg": int((delta < -1e-12).sum()),
    }
    pd.DataFrame([metrics]).to_csv(cal_dir / "metrics.csv", index=False)
    log(f"real seed {seed:03d}: [{candidate_id}] calibrated")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def load_real_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                             dict[str, dict[str, Any]], dict[str, Path]]:
    """What the real driver needs: the extracted rows and groups, the manifest entry of each setting, and that setting's observed reference on the test weeks."""
    ext_dir = run_dir / "extraction" / "real"
    ext_rows = pd.read_csv(ext_dir / "row_features.csv", low_memory=False).reset_index(drop=True)
    ext_groups = pd.read_csv(ext_dir / "group_features.csv", low_memory=False).reset_index(drop=True)
    base_manifest = json.loads(
        (run_dir / "configs" / "real" / "manifest.json").read_text())["manifest"]
    by_cand = {str(e["candidate_id"]): e
               for e in base_manifest if e.get("target") == "real"}

    # busy_ratio: extraction usually provides it; recompute from the observed log if absent.
    if "busy_ratio" not in ext_rows.columns:
        cfg0 = json.loads(resolve_cfg(base_manifest[0]["config_json"]).read_text())
        obs0 = Path(cfg0.get("observed_log", DEFAULT_OBS_LOG))
        ext_rows = merge_busy_into_rows(ext_rows, _core.busy_per_resource_weekday_bin([obs0]))

    val_obs_by_cand: dict[str, Path] = {}
    for cand_id, entry in by_cand.items():
        cfg = json.loads(resolve_cfg(entry["config_json"]).read_text())
        obs_log = Path(cfg.get("observed_log", DEFAULT_OBS_LOG))
        val_obs_by_cand[cand_id] = write_windowed_copy(
            obs_log, _core.REAL_VAL_WINDOW,
            run_dir / "calibration" / "real" / f"_val_obs_{cand_id}.csv")
    return ext_rows, ext_groups, by_cand, val_obs_by_cand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",       required=True)
    ap.add_argument("--synthetic",     action="store_true")
    ap.add_argument("--real",          action="store_true")
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-real-seeds",  type=int, default=10)
    ap.add_argument("--n-synth-runs",  type=int, default=5,
                    help="Synthetic P_CAL re-simulation runs per scenario.")
    ap.add_argument("--n-real-runs",   type=int, default=5,
                    help="Real P_CAL re-simulation runs per (seed, candidate); "
                         "concatenated before computing one distance per seed.")
    ap.add_argument("--bin-minutes",   type=int, default=60,
                    help="Time-bin resolution in minutes (must match earlier stages).")
    args = ap.parse_args()

    if not args.synthetic and not args.real:
        ap.error("Specify at least one of --synthetic or --real.")

    run_dir  = Path(args.run_dir)

    if args.synthetic:
        synth_manifest_path = run_dir / "configs" / "synth" / "manifest.json"
        manifest_items = json.loads(
            synth_manifest_path.read_text(encoding="utf-8"))["manifest"]
        for i, s in enumerate(_core.run_seeds(args.n_synth_seeds)):
            log(f"synth seed {i + 1}/{args.n_synth_seeds} (seed {s:03d}): starting")
            try:
                calibrate_synth_seed(s, run_dir, args.n_synth_runs, manifest_items,
                                      bin_minutes=args.bin_minutes)
            except Exception as e:
                log(f"[ERROR] synth seed {s:03d}: {e}")
                log(traceback.format_exc())

    if args.real:
        ext_rows, ext_groups, by_cand, val_obs_by_cand = load_real_inputs(run_dir)
        for i, s in enumerate(_core.run_seeds(args.n_real_seeds)):
            log(f"real seed {i + 1}/{args.n_real_seeds} (seed {s:03d}): "
                f"{len(by_cand)} candidate(s)")
            for cand_id, entry in by_cand.items():
                rows_cand = ext_rows[
                    ext_rows["unit_id"].astype(str).eq(cand_id)
                ].reset_index(drop=True)
                groups_cand = ext_groups[
                    ext_groups["unit_id"].astype(str).eq(cand_id)
                ].reset_index(drop=True)
                if rows_cand.empty or groups_cand.empty:
                    log(f"[WARN] real seed {s:03d} [{cand_id}]: "
                        f"no extraction rows/groups; skipping")
                    continue
                try:
                    calibrate_real_seed(
                        s, run_dir, cand_id, rows_cand, groups_cand,
                        [entry], val_obs_by_cand[cand_id],
                        num_runs=args.n_real_runs)
                except Exception as e:
                    log(f"[ERROR] real seed {s:03d} [{cand_id}]: {e}")
                    log(traceback.format_exc())

    log("Calibration complete.")


if __name__ == "__main__":
    main()
