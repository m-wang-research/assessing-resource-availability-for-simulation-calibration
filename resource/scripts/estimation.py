#!/usr/bin/env python3
"""Build the row and group features, then train the Q+ availability scorer on them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _core
from _core import (
    _build_units,
    get_windows,
    resolve_cfg,
    busy_per_resource_weekday_bin,
    DEFAULT_OBS_LOG,
    add_reliability_features,
    add_busy_to_rows,
    build_feature_frame,
    log,
    merge_busy_into_rows,
    CATBOOST_PARAMS,
    build_group_majority_label,
)



def scenario_metadata(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in manifest:
        cfg = json.loads(resolve_cfg(item["config_json"]).read_text(encoding="utf-8"))
        rows.append({
            "scenario": item["scenario_id"],
            "scenario_type": item.get("scenario_type"),
            "workload_level": item.get("workload_level"),
            "arrival_lambda": cfg.get("arrival_lambda", cfg.get("arrival_rate_per_hour")),
            "weeks": cfg.get("weeks"),
            "resources": len(cfg.get("resources", [])),
            "description": cfg.get("scenario_description", ""),
        })
    return pd.DataFrame(rows)


def busy_for_synth_seed(common_dir: Path, manifest_items: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Per-unit busy from P_TRUE logs under common_dir/logs/<unit>/P_TRUE/."""
    out = {}
    for it in manifest_items:
        if it.get("target") != "synthetic":
            continue
        unit = it["scenario_id"]
        ptrue_dir = common_dir / "logs" / unit / "P_TRUE"
        log_paths = sorted(ptrue_dir.glob("run*.csv")) if ptrue_dir.exists() else []
        out[unit] = busy_per_resource_weekday_bin(log_paths)
    return out


def build_features_and_groups(out_seed_dir: Path, manifest_items, base_rho: pd.DataFrame,
                               bin_minutes: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic_row_features.csv and group_features.csv."""
    meta = scenario_metadata(manifest_items)
    rows = build_feature_frame(manifest_items, base_rho)
    rows = rows.merge(
        meta[["scenario", "scenario_type", "workload_level"]],
        left_on="unit_id", right_on="scenario", how="left",
    ).drop(columns=["scenario_y"]).rename(columns={"scenario_x": "scenario"})
    rows = add_reliability_features(rows)
    rows.to_csv(out_seed_dir / "synthetic_row_features.csv", index=False)

    groups = _build_units(rows, get_windows(bin_minutes))
    groups = groups.merge(
        meta[["scenario", "scenario_type", "workload_level"]],
        left_on="unit_id", right_on="scenario", how="left",
        suffixes=("", "_meta"),
    )
    groups.to_csv(out_seed_dir / "group_features.csv", index=False)
    return rows, groups



# ── Synthetic extraction (one seed) ────────────────────────────────────────────
def extract_one_synth_seed(seed: int, run_dir: Path,
                             manifest_items: list[dict],
                             bin_minutes: int = 60) -> None:
    sim_seed_dir = run_dir / "simulation" / "synth" / f"seed{seed:03d}"
    ext_seed_dir = run_dir / "extraction" / "synth" / f"seed{seed:03d}"
    ext_seed_dir.mkdir(parents=True, exist_ok=True)

    rho_path = sim_seed_dir / "rho_profiles.csv"
    if not rho_path.exists():
        raise FileNotFoundError(f"rho_profiles.csv not found: {rho_path}\n"
                                f"Run preparation.py --synthetic first.")
    base_rho = pd.read_csv(rho_path)

    rows, groups = build_features_and_groups(
        ext_seed_dir, manifest_items, base_rho, bin_minutes=bin_minutes)

    # busy from P_TRUE logs (sim logs live in the simulation dir)
    busy_by_unit = busy_for_synth_seed(sim_seed_dir, manifest_items)
    groups = _core.add_busy_to_synth_groups(groups, busy_by_unit)
    rows   = add_busy_to_rows(rows, busy_by_unit)

    # map rows to groups so downstream has group_id
    rows_mapped = _core.map_synth_rows_to_groups(
        rows, groups).reset_index(drop=True)

    # overwrite raw versions written above with enriched ones
    rows_mapped.to_csv(ext_seed_dir / "row_features_mapped.csv", index=False)
    groups.to_csv(ext_seed_dir / "group_features.csv", index=False)

    log(f"seed {seed:03d}: extraction done — {len(rows_mapped)} rows, {len(groups)} groups")


# ── Real extraction (one-time) ─────────────────────────────────────────────────
def extract_real(run_dir: Path,
                  real_rows_path: Path, real_groups_path: Path,
                  obs_log_path: Path) -> None:
    ext_real_dir = run_dir / "extraction" / "real"
    ext_real_dir.mkdir(parents=True, exist_ok=True)

    real_rows = pd.read_csv(real_rows_path, low_memory=False).reset_index(drop=True)
    real_groups = pd.read_csv(real_groups_path, low_memory=False).reset_index(drop=True)

    # busy_ratio from the observed BPI-2012 log
    real_rows = merge_busy_into_rows(
        real_rows, busy_per_resource_weekday_bin([obs_log_path]))
    real_rows.to_csv(ext_real_dir / "row_features.csv", index=False)

    # real groups are pre-computed; copy as-is
    real_groups.to_csv(ext_real_dir / "group_features.csv", index=False)
    log(f"Real extraction done — {len(real_rows)} rows, {len(real_groups)} groups")


# ── Main ───────────────────────────────────────────────────────────────────────


GROUP_FEATURES = [
    "mean_support_count",
    "mean_recurrence_across_weeks",
    "mean_resource_day_positive_evidence_rate",
    "neighbor_positive_count",
    "resource_activity_level",
    "resource_mean_support",
]


def train_and_score_qpos(groups: pd.DataFrame,
                           rows_mapped: pd.DataFrame,
                           seed: int,
                           ) -> tuple[list[pd.Series], list[dict[str, Any]]]:
    """Train Q+ on the groups, score the same groups, set theta_pos. Returns (q_pos, trace)."""
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split

    for col in GROUP_FEATURES:
        if col not in groups.columns:
            groups[col] = 0.0
    X_all     = groups[GROUP_FEATURES].astype(float)
    score_ids = groups["group_id"].astype(str).values

    # Y+ group label: group-majority(rho_obs > p)
    mapped_ok = rows_mapped.dropna(subset=["group_id"])
    group_ids    = groups["group_id"].astype(str).values
    y_pos_by_blk = build_group_majority_label(mapped_ok)
    y_pos = pd.Series(
        y_pos_by_blk.reindex(group_ids).fillna(0).astype(int).values,
        index=groups.index,
    )

    # split stratified by Y+, random if degenerate
    try:
        idx_tr, _ = train_test_split(
            groups.index, test_size=0.30, random_state=seed,
            stratify=y_pos if y_pos.nunique() >= 2 else None,
        )
    except ValueError:
        idx_tr, _ = train_test_split(
            groups.index, test_size=0.30, random_state=seed)
    X_tr   = X_all.loc[idx_tr]
    y_tr_p = y_pos.loc[idx_tr]

    # κ⁺ = global positive-labeling rate from training data (Elkan & Noto 2008): fraction of
    # groups with an observed positive label. Unlabeled groups enter as background with weight κ⁺.
    kappa_pos = float(y_tr_p.mean())
    kappa_pos = max(kappa_pos, 1e-3)   # floor to avoid near-zero weight
    w = np.where(y_tr_p.values > 0, 1.0, kappa_pos)

    params = dict(CATBOOST_PARAMS); params["random_seed"] = seed

    if y_tr_p.nunique() < 2:
        q_pos = pd.Series(float(y_tr_p.mean()), index=X_all.index)
    else:
        mp = CatBoostClassifier(**params)
        mp.fit(X_tr, y_tr_p, sample_weight=w)  # biased PU: unlabeled weight = κ⁺
        q_pos = pd.Series(mp.predict_proba(X_all)[:, 1], index=X_all.index)

    theta_pos = float(np.quantile(q_pos.values, 0.75))
    trace = {
            "seed":        seed,
            "theta_pos":     theta_pos,
            "q_pos_min":   float(np.min(q_pos.values)),
            "q_pos_q25":   float(np.quantile(q_pos.values, 0.25)),
            "q_pos_q50":   float(np.quantile(q_pos.values, 0.50)),
            "q_pos_q75":   theta_pos,
            "q_pos_max":   float(np.max(q_pos.values)),
            "n_groups":    int(len(q_pos)),
            "n_y_pos":     int(y_pos.sum()),
    }
    return pd.Series(q_pos.values, index=score_ids, name="q_pos"), trace


def save_estimation_artifacts(est_dir: Path,
                                q_pos: pd.Series,
                                trace: dict[str, Any]) -> None:
    est_dir.mkdir(parents=True, exist_ok=True)
    q_pos.reset_index().rename(columns={"index": "group_id"}).to_csv(
        est_dir / "q_pos.csv", index=False)
    (est_dir / "theta_pos.json").write_text(
        json.dumps({"theta_pos": trace["theta_pos"]}, indent=2), encoding="utf-8")


def _estimate_seed(target: str, seed: int, run_dir: Path) -> None:
    """Train Q+ on one seed's extracted groups (target = synth|real), score them, save."""
    ext_dir = run_dir / "extraction" / target / (f"seed{seed:03d}" if target == "synth" else "")
    est_dir = run_dir / "estimation" / target / f"seed{seed:03d}"

    groups = pd.read_csv(ext_dir / "group_features.csv", low_memory=False).reset_index(drop=True)
    rows_name = "row_features_mapped.csv" if target == "synth" else "row_features.csv"
    rows_mapped = pd.read_csv(ext_dir / rows_name, low_memory=False)

    q_pos, trace = train_and_score_qpos(groups.copy(), rows_mapped, seed)
    save_estimation_artifacts(est_dir, q_pos, trace)
    log(f"{target} seed {seed:03d}: Q+ trained — "
        f"theta_pos={trace['theta_pos']:.4f}, n_y_pos={trace['n_y_pos']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",       required=True)
    ap.add_argument("--synthetic",     action="store_true")
    ap.add_argument("--real",          action="store_true")
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-real-seeds",  type=int, default=10)
    ap.add_argument("--bin-minutes",   type=int, default=60,
                    help="Time-bin resolution in minutes (must match preparation.py).")
    ap.add_argument("--real-rows",   help="with --real: the extracted row features")
    ap.add_argument("--real-groups", help="with --real: the extracted group features")
    args = ap.parse_args()

    if not args.synthetic and not args.real:
        ap.error("Specify at least one of --synthetic or --real.")
    if args.real and not (args.real_rows and args.real_groups):
        ap.error("--real needs --real-rows and --real-groups; main.py --real builds both "
                 "with _core.build_real_groups().")

    run_dir = Path(args.run_dir)

    if args.synthetic:
        synth_manifest_path = run_dir / "configs" / "synth" / "manifest.json"
        if not synth_manifest_path.exists():
            sys.exit(f"[ERROR] {synth_manifest_path} not found. "
                     f"Run preparation.py --synthetic first.")
        manifest_items = json.loads(
            synth_manifest_path.read_text(encoding="utf-8"))["manifest"]

        for s in _core.run_seeds(args.n_synth_seeds):
            extract_one_synth_seed(s, run_dir, manifest_items, bin_minutes=args.bin_minutes)

    if args.real:
        extract_real(
            run_dir,
            real_rows_path=Path(args.real_rows),
            real_groups_path=Path(args.real_groups),
            obs_log_path=DEFAULT_OBS_LOG,
        )

    log("Feature extraction complete.")

    if args.synthetic:
        for s in _core.run_seeds(args.n_synth_seeds):
            _estimate_seed("synth", s, run_dir)

    if args.real:
        for s in _core.run_seeds(args.n_real_seeds):
            _estimate_seed("real", s, run_dir)

    log("Training complete.")


if __name__ == "__main__":
    main()
