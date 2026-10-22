#!/usr/bin/env python3
"""Compute the thresholds the decrease branch of the calibration rule needs (see neg_rule_thresholds)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _core import log, run_seeds


def neg_rule_thresholds(rows: pd.DataFrame, groups: pd.DataFrame,
                        q_pos_group: pd.Series) -> tuple[float, float]:
    """The two thresholds that gate a decrease in the calibration rule."""
    blk_index    = pd.Index(groups["group_id"].astype(str).values)
    rid          = rows["group_id"].astype(str)
    group_mean_g = (rows.assign(_bid=rid)
                     .groupby("_bid")["rho_diff"].mean()
                     .reindex(blk_index, fill_value=0.0))
    q_pos_group  = q_pos_group.reindex(blk_index, fill_value=0.0)

    theta_neg_q = float(np.quantile(q_pos_group.values, 0.25))
    neg_g = (-group_mean_g[group_mean_g < 0]).values
    # no negative group at all: no decrease can be selected
    theta_neg_delta = float(np.quantile(neg_g, 0.75)) if len(neg_g) else float("inf")
    return theta_neg_q, theta_neg_delta


def diagnose(seed: int, scope: str, run_dir: Path) -> None:
    """scope: 'synth' or 'real'."""
    ext_dir = run_dir / "extraction" / scope / (f"seed{seed:03d}" if scope == "synth" else "")
    est_dir = run_dir / "estimation"  / scope / f"seed{seed:03d}"
    diag_dir = run_dir / "diagnosis"  / scope / f"seed{seed:03d}"
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows_name = "row_features_mapped.csv" if scope == "synth" else "row_features.csv"
    rows   = pd.read_csv(ext_dir / rows_name, low_memory=False)
    groups = pd.read_csv(ext_dir / "group_features.csv", low_memory=False)

    # drop rows without a group mapping
    rows = rows.dropna(subset=["group_id"]).copy()
    rows["group_id"] = rows["group_id"].astype(str)
    groups["group_id"] = groups["group_id"].astype(str)

    q_pos_path = est_dir / "q_pos.csv"
    if not q_pos_path.exists():
        raise FileNotFoundError(f"{q_pos_path} not found. Run estimation.py first.")
    q_pos_df  = pd.read_csv(q_pos_path)
    q_pos_blk = pd.Series(
        q_pos_df["q_pos"].values,
        index=q_pos_df["group_id"].astype(str).values,
        name="q_pos",
    )
    theta_neg_q, theta_neg_delta = neg_rule_thresholds(rows, groups, q_pos_blk)
    pd.DataFrame([{"theta_neg_q": theta_neg_q, "theta_neg_delta": theta_neg_delta}]).to_csv(
        diag_dir / "neg_thresholds.csv", index=False)

    log(f"{scope} seed {seed:03d}: negative-rule thresholds "
        f"theta_neg_q={theta_neg_q:.4f}, theta_neg_delta={theta_neg_delta:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",       required=True)
    ap.add_argument("--synthetic",     action="store_true")
    ap.add_argument("--real",          action="store_true")
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-real-seeds",  type=int, default=10)
    args = ap.parse_args()

    if not args.synthetic and not args.real:
        ap.error("Specify at least one of --synthetic or --real.")

    run_dir = Path(args.run_dir)

    if args.synthetic:
        for s in run_seeds(args.n_synth_seeds):
            diagnose(s, "synth", run_dir)

    if args.real:
        for s in run_seeds(args.n_real_seeds):
            diagnose(s, "real", run_dir)

    log("Diagnosis complete.")


if __name__ == "__main__":
    main()
