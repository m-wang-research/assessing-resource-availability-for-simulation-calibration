#!/usr/bin/env python3
"""Write the series behind the figures of a run as CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _core import (
    run_seeds,
    CONTROLS,
    SCEN_ORDER,
    log,
)


# ── Shared ─────────────────────────────────────────────────────────────────────
# The working day the settings and the observed fractions are read over.
_HOURS = np.arange(5, 20)

# The settings the settings series carries.
SETTINGS_SCENARIOS = ("S1-high-uniform", "S1-low-uniform", "S2-early-strong", "S2-late-strong")

# The controls the observed fractions are read for.
OBSERVED_SCENARIOS = ("S0-same", "S3-busy", "S3-idle")


def _write(path: Path, frame: pd.DataFrame) -> None:
    """Write one figure's series."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _distance_frame(labels: list[str], series: dict[str, list], key: str) -> pd.DataFrame:
    """CTD, CWD and CED before and after, one row per label and metric."""
    rows = []
    for i, lab in enumerate(labels):
        for met in ("ctd", "cwd", "ced"):
            rows.append({key: lab, "metric": met,
                         "before": series[f"{met}_b"][i],
                         "before_lo": series[f"{met}_blo"][i],
                         "before_hi": series[f"{met}_bhi"][i],
                         "after": series[f"{met}_a"][i],
                         "after_lo": series[f"{met}_alo"][i],
                         "after_hi": series[f"{met}_ahi"][i]})
    return pd.DataFrame(rows)


# ── Synthetic series ───────────────────────────────────────────────────────────
def _distance_to_ptrue(run_dir: Path) -> pd.DataFrame:
    """Mean |p - p_true| of P, P^cal and P^emp per setting."""
    emp_df = pd.read_csv(run_dir / "evaluation" / "emp_baseline_dp.csv").set_index("scenario")
    scen = [s for s in SCEN_ORDER if s in emp_df.index]
    cols = ["D_P_P", "D_P_Pcal_mean", "D_P_Pcal_std", "D_P_Pemp_mean"]
    return emp_df.loc[scen, cols].reset_index()


def _revision_counts(run_dir: Path, n_seeds: int, scen: list[str]) -> tuple[list[float], list[float]]:
    """Median over seeds of the probabilities raised and lowered, per setting."""
    ext_base = run_dir / "extraction" / "synth"
    cal_base = run_dir / "calibration" / "synth"
    scen_raise: dict[str, list] = {s: [] for s in scen}
    scen_lower: dict[str, list] = {s: [] for s in scen}

    for s in run_seeds(n_seeds):
        feat_s = pd.read_csv(ext_base / f"seed{s:03d}" / "synthetic_row_features.csv",
                             low_memory=False,
                             usecols=["unit_id", "resource", "weekday", "time_bin", "p", "scenario"])
        pcal_s = pd.read_csv(cal_base / f"seed{s:03d}" / "p_cal_long.csv",
                             low_memory=False)
        feat_s["resource"] = feat_s["resource"].astype(str)
        pcal_s["resource"] = pcal_s["resource"].astype(str)
        m = feat_s.merge(pcal_s, on=["unit_id", "resource", "weekday", "time_bin"], how="inner")
        m["delta"] = m["p_cal"] - m["p"]
        for sc in scen:
            sub_sc = m[m["scenario"] == sc]
            scen_raise[sc].append(int((sub_sc["delta"] > 1e-9).sum()))
            scen_lower[sc].append(int((sub_sc["delta"] < -1e-9).sum()))

    return [np.median(scen_raise[sc]) for sc in scen], [np.median(scen_lower[sc]) for sc in scen]


def _process_series(run_dir: Path, n_seeds: int, scen: list[str]) -> tuple[list[str], dict[str, list]]:
    """CTD, CWD and CED before and after, as the median over every re-simulation of every seed with interquartile offsets, for the settings that carry a deviation."""
    cal_base = run_dir / "calibration" / "synth"
    dev_scen = [sc for sc in scen if sc not in CONTROLS]
    pm_all = []
    for s in run_seeds(n_seeds):
        pm_path = cal_base / f"seed{s:03d}" / "process_metrics.csv"
        if pm_path.exists():
            pm = pd.read_csv(pm_path, low_memory=False)
            pm["seed"] = s
            pm_all.append(pm)
    pm_df = pd.concat(pm_all, ignore_index=True) if pm_all else pd.DataFrame()

    def _median_iqr_for_scenario(sc: str, col: str):
        """Median over the runs of every seed, with the interquartile range as the bar."""
        if col not in pm_df.columns:
            return 0.0, 0.0, 0.0
        v = pm_df.loc[pm_df["scenario"] == sc, col].dropna()
        if v.empty:
            return 0.0, 0.0, 0.0
        med = float(v.median())
        return (med,
                max(0.0, med - float(v.quantile(0.25))),
                max(0.0, float(v.quantile(0.75)) - med))

    series = {k: [] for k in ("ctd_b", "ctd_blo", "ctd_bhi", "ctd_a", "ctd_alo", "ctd_ahi",
                              "cwd_b", "cwd_blo", "cwd_bhi", "cwd_a", "cwd_alo", "cwd_ahi",
                              "ced_b", "ced_blo", "ced_bhi", "ced_a", "ced_alo", "ced_ahi")}
    valid_dev = []
    for sc in dev_scen:
        if pm_df.empty or sc not in set(pm_df["scenario"]):
            continue
        for met in ("ctd", "cwd", "ced"):
            # CWD and CED carry the metric mode in the column name; CTD does not.
            bcol = f"{met}_before" if met == "ctd" else f"{met}_wass_before"
            acol = f"{met}_after" if met == "ctd" else f"{met}_wass_after"
            mb, blo, bhi = _median_iqr_for_scenario(sc, bcol)
            ma, alo, ahi = _median_iqr_for_scenario(sc, acol)
            series[f"{met}_b"].append(mb); series[f"{met}_blo"].append(blo); series[f"{met}_bhi"].append(bhi)
            series[f"{met}_a"].append(ma); series[f"{met}_alo"].append(alo); series[f"{met}_ahi"].append(ahi)
        valid_dev.append(sc)
    return valid_dev, series


def _rho_profile(run_dir: Path) -> pd.DataFrame:
    """Observed, simulated and calibrated start fraction over the day for one cell."""
    ext_base = run_dir / "extraction" / "synth"
    cal_base = run_dir / "calibration" / "synth"
    seed0 = f"seed{run_seeds(1)[0]:03d}"
    feat0 = pd.read_csv(ext_base / seed0 / "synthetic_row_features.csv", low_memory=False)
    rho0  = pd.read_csv(cal_base / seed0 / "rho_profiles.csv", low_memory=False)
    rho0_cal = rho0[rho0["state"] == "P_CAL"].copy()

    # The example is an S1-low setting, drawn from a resource whose availability is wrong.
    s1low = [s for s in feat0["scenario"].unique() if str(s).startswith("S1-low")]
    pick_scen = s1low[0] if s1low else \
                next(s for s in SCEN_ORDER if s not in CONTROLS and s in feat0["scenario"].values)
    sub = feat0[feat0["scenario"] == pick_scen].copy()
    sub["resource"] = sub["resource"].astype(str)
    if "p_true" in sub.columns:
        wrong_res = sub.loc[(sub["p"].astype(float) - sub["p_true"].astype(float)).abs() > 1e-9, "resource"].unique()
        if len(wrong_res):
            sub = sub[sub["resource"].isin(wrong_res)]
    # Cell selection: observed activity, preferring cells calibration improves.
    rc_all = rho0_cal[rho0_cal["unit_id"] == pick_scen][["resource", "weekday", "time_bin", "rho"]].copy()
    rc_all["resource"] = rc_all["resource"].astype(str)
    psel = sub.merge(rc_all.rename(columns={"rho": "rho_cal"}),
                     on=["resource", "weekday", "time_bin"], how="left")
    gsel = psel.groupby(["resource", "weekday"])
    gap_mass = gsel.apply(lambda d: (d.rho_obs - d.rho_sim).abs().sum(), include_groups=False)
    obs_mass = gsel["rho_obs"].sum()
    improve = gsel.apply(lambda d: ((d.rho_obs - d.rho_sim).abs()
                                    - (d.rho_obs - d.rho_cal).abs()).sum(), include_groups=False)
    if (obs_mass > 0).any():
        elig = obs_mass[obs_mass >= obs_mass[obs_mass > 0].quantile(0.50)].index
    else:
        elig = obs_mass.index
    score = (gap_mass + 2.0 * improve.clip(lower=0))
    score = score[score.index.isin(elig)]
    res_pick, wd_pick = score.idxmax()
    cell = sub[(sub["resource"] == res_pick) & (sub["weekday"] == wd_pick)].sort_values("time_bin")
    rc = rho0_cal[(rho0_cal["unit_id"] == pick_scen) &
                  (rho0_cal["resource"].astype(str) == str(res_pick)) &
                  (rho0_cal["weekday"] == wd_pick)].sort_values("time_bin")

    parts = [pd.DataFrame({"series": "obs", "time_bin": cell["time_bin"].to_numpy(),
                           "rho": cell["rho_obs"].to_numpy()}),
             pd.DataFrame({"series": "sim", "time_bin": cell["time_bin"].to_numpy(),
                           "rho": cell["rho_sim"].to_numpy()})]
    if not rc.empty:
        parts.append(pd.DataFrame({"series": "cal", "time_bin": rc["time_bin"].to_numpy(),
                                   "rho": rc["rho"].to_numpy()}))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "weekday", int(wd_pick))
    out.insert(0, "resource", str(res_pick))
    out.insert(0, "scenario", pick_scen)
    return out


def _observed_fractions(run_dir: Path,
                        scenarios: tuple[str, ...] = OBSERVED_SCENARIOS) -> pd.DataFrame:
    """Observed start fractions over a working day against P_true, for the controls."""
    f = pd.read_csv(run_dir / "extraction" / "synth" / f"seed{run_seeds(1)[0]:03d}" / "synthetic_row_features.csv",
                    usecols=["scenario", "weekday", "time_bin", "rho_obs", "p_true"], low_memory=False)
    day = f[f["weekday"] == 0]
    pt = (day[day["scenario"] == scenarios[0]].groupby("time_bin")["p_true"].mean()
          .reindex(_HOURS, fill_value=0.0))
    parts = [pd.DataFrame({"series": "p_true", "scenario": scenarios[0],
                           "time_bin": _HOURS, "value": pt.to_numpy()})]
    for sc in scenarios:
        r = day[day["scenario"] == sc].groupby("time_bin")["rho_obs"].mean().reindex(_HOURS, fill_value=0.0)
        parts.append(pd.DataFrame({"series": "rho_obs", "scenario": sc,
                                   "time_bin": _HOURS, "value": r.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def _settings_series(config_dirs: list[Path]) -> pd.DataFrame:
    """The assessed availability settings over a working day, per configuration set."""
    def monday_by_hour(frame: pd.DataFrame, col: str) -> np.ndarray:
        day = frame[frame["weekday"] == 0].groupby("time_bin")[col].mean()
        return day.reindex(_HOURS, fill_value=0.0).to_numpy()

    parts = []
    for cfg in config_dirs:
        parts.append(pd.DataFrame(
            {"config": cfg.name, "series": "p_true", "scenario": "", "time_bin": _HOURS,
             "value": monday_by_hour(pd.read_csv(cfg / "p_true_surface.csv"), "p_true")}))
        for scen in SETTINGS_SCENARIOS:
            parts.append(pd.DataFrame(
                {"config": cfg.name, "series": "p", "scenario": scen, "time_bin": _HOURS,
                 "value": monday_by_hour(pd.read_csv(cfg / scen / "p_surface.csv"), "p")}))
    return pd.concat(parts, ignore_index=True)


# ── Real series ────────────────────────────────────────────────────────────────
def _candidates_by_angle(candidate_ids: list[str]) -> list[str]:
    """The candidate ids by ascending discovery angle (…_an<angle>)."""
    def _angle_from_id(cid: str) -> float:
        try:
            return float(str(cid).split("_an")[-1])
        except Exception:
            return float("inf")
    return sorted(candidate_ids, key=_angle_from_id)


def _real_series(run_dir: Path, n_seeds: int) -> dict[str, pd.DataFrame]:
    """The series behind the four panels of the real-log figure."""
    cal_base  = run_dir / "calibration" / "real"
    ext_rows  = pd.read_csv(run_dir / "extraction" / "real" / "row_features.csv",
                            low_memory=False)
    ext_rows["resource"] = ext_rows["resource"].astype(str)

    # ── Discover candidate_ids present in the calibration outputs ─────────────
    cand_ids = {m.parent.name for s in run_seeds(n_seeds)
                for m in (cal_base / f"seed{s:03d}").glob("*/metrics.csv")}
    if not cand_ids:
        cand_ids = set(ext_rows["unit_id"].astype(str).unique())
    cands = _candidates_by_angle(sorted(cand_ids))

    # ── Per-candidate per-seed metrics (n_pos/n_neg, CTD/CWD before/after) ─────
    rec: list[dict] = []
    for s in run_seeds(n_seeds):
        for c in cands:
            mp = cal_base / f"seed{s:03d}" / c / "metrics.csv"
            if mp.exists():
                d = pd.read_csv(mp).iloc[0].to_dict()
                d["candidate_id"] = c
                rec.append(d)
    met = pd.DataFrame(rec)

    def _per_candidate(col: str, agg: str) -> list[float]:
        out = []
        for c in cands:
            sub = met[met["candidate_id"] == c] if not met.empty else pd.DataFrame()
            if sub.empty or col not in sub.columns:
                out.append(0.0)
            else:
                v = (sub[col].median() if agg == "median"
                     else sub[col].mean() if agg == "mean"
                     else sub[col].std())
                out.append(float(v))
        return out

    # Distances come from per_run_metrics.csv's explicit Wasserstein columns.
    pr_all = []
    for s in run_seeds(n_seeds):
        for pr_path in sorted((cal_base / f"seed{s:03d}").glob("*/per_run_metrics.csv")):
            pr = pd.read_csv(pr_path, low_memory=False)
            pr["seed"] = s
            pr_all.append(pr)
    pr_df = pd.concat(pr_all, ignore_index=True) if pr_all else pd.DataFrame()
    if pr_df.empty:   # otherwise panel (d) draws a full set of zero-height bars
        raise ValueError(f"{run_dir} has no per_run_metrics.csv")

    def _median_iqr_per_candidate(col: str):
        """Median over the runs of every seed, with the interquartile range as the bar."""
        meds, lo, hi = [], [], []
        for c in cands:
            v = (pr_df.loc[pr_df["candidate_id"] == c, col].dropna()
                 if (not pr_df.empty and col in pr_df.columns) else pd.Series([], dtype=float))
            if v.empty:
                meds.append(0.0); lo.append(0.0); hi.append(0.0)
            else:
                m = float(v.median()); meds.append(m)
                lo.append(max(0.0, m - float(v.quantile(0.25))))
                hi.append(max(0.0, float(v.quantile(0.75)) - m))
        return meds, lo, hi

    revisions = pd.DataFrame({"candidate_id": cands,
                              "raised": _per_candidate("n_pos", "median"),
                              "lowered": _per_candidate("n_neg", "median")})

    dist: dict[str, list] = {}
    for met_name in ("ctd", "cwd", "ced"):
        for state, tag in (("b", "before"), ("a", "after")):
            meds, lo, hi = _median_iqr_per_candidate(f"{met_name.upper()}_wass_{tag}")
            dist[f"{met_name}_{state}"] = meds
            dist[f"{met_name}_{state}lo"] = lo
            dist[f"{met_name}_{state}hi"] = hi

    # ── Representative candidate (the smallest angle) for panels (a)/(c) ───────
    rep_cand = cands[0]
    base_rep = ext_rows[ext_rows["unit_id"].astype(str) == rep_cand].reset_index(drop=True)
    obs_sim  = (base_rep[["resource", "weekday", "time_bin",
                          "rho_obs", "rho_sim", "support_days_obs"]]
                .rename(columns={"support_days_obs": "support_days"}))

    # Calibrated rho for rep candidate, pooled over seeds
    cal_frames = []
    for s in run_seeds(n_seeds):
        ra_path = cal_base / f"seed{s:03d}" / rep_cand / "rho_after.csv"
        if ra_path.exists():
            ra = pd.read_csv(ra_path)
            ra["resource"] = ra["resource"].astype(str)
            cal_frames.append(ra)
    if cal_frames:
        allr = pd.concat(cal_frames, ignore_index=True)
        cal_pooled = (allr.groupby(["resource", "weekday", "time_bin"], as_index=False)["rho_after"]
                      .mean().rename(columns={"rho_after": "rho_cal"}))
    else:
        cal_pooled = pd.DataFrame(columns=["resource", "weekday", "time_bin", "rho_cal"])
    prof = obs_sim.merge(cal_pooled, on=["resource", "weekday", "time_bin"], how="left")

    # Per-row delta for rep candidate (delta_rows row-aligned to base_rep by position)
    dlong_frames = []
    for s in run_seeds(n_seeds):
        dr_path = cal_base / f"seed{s:03d}" / rep_cand / "delta_rows.csv"
        if dr_path.exists():
            dr = pd.read_csv(dr_path)
            if len(dr) == len(base_rep):
                dr = dr.copy()
                dr["resource"] = base_rep["resource"].values
                dr["weekday"]  = base_rep["weekday"].values
                dr["time_bin"] = base_rep["time_bin"].values
                dr["seed"]     = s
                dlong_frames.append(dr[["resource", "weekday", "time_bin", "delta", "seed"]])
    dlong = pd.concat(dlong_frames, ignore_index=True) if dlong_frames else pd.DataFrame()

    rho_cols = ["candidate_id", "resource", "weekday", "series", "time_bin", "rho"]
    rho_profile = pd.DataFrame(columns=rho_cols)
    if not prof.empty and "rho_obs" in prof.columns and "rho_sim" in prof.columns:
        prof["gap"] = (prof.rho_obs - prof.rho_sim).abs() * prof.support_days.fillna(1)
        pick_src = prof[prof.resource.astype(str) != "NOT_SET"]
        cell_support = pick_src.groupby(["resource", "weekday"])["support_days"].sum()
        gmass = pick_src.groupby(["resource", "weekday"])["gap"].sum()
        # Guard against degenerate cells: rho_obs ~0 all day but rho_sim spikes.
        obs_mass = pick_src.groupby(["resource", "weekday"])["rho_obs"].sum()
        # how far calibration moves rho toward rho_obs (positive = improvement)
        if "rho_cal" in pick_src.columns:
            improve = (pick_src.assign(
                imp=((pick_src.rho_obs - pick_src.rho_sim).abs()
                     - (pick_src.rho_obs - pick_src.rho_cal).abs())
                    * pick_src.support_days.fillna(1))
                .groupby(["resource", "weekday"])["imp"].sum())
        else:
            improve = pd.Series(0.0, index=gmass.index)
        rough = (pick_src.sort_values("time_bin")
                 .groupby(["resource", "weekday"])["rho_sim"]
                 .apply(lambda s: np.abs(np.diff(s.to_numpy())).sum() if len(s) > 1 else 0.0))
        # eligible: enough recorded support AND real observed activity
        elig_support = cell_support >= cell_support.quantile(0.75)
        elig_obs = obs_mass >= obs_mass[obs_mass > 0].quantile(0.50) if (obs_mass > 0).any() else obs_mass > -1
        elig = cell_support[elig_support & elig_obs].index
        if len(elig) == 0:
            elig = cell_support[elig_support].index
        # prefer cells where calibration visibly improves the fit
        score = ((gmass + 2.0 * improve.clip(lower=0)) / (1.0 + rough))
        score = score[score.index.isin(elig)]
        res_pick, wd_pick = score.idxmax()
        cell = prof[(prof.resource == res_pick) & (prof.weekday == wd_pick)].sort_values("time_bin")
        parts = [pd.DataFrame({"series": "obs", "time_bin": cell.time_bin.to_numpy(),
                               "rho": cell.rho_obs.to_numpy()}),
                 pd.DataFrame({"series": "sim", "time_bin": cell.time_bin.to_numpy(),
                               "rho": cell.rho_sim.to_numpy()})]
        if "rho_cal" in cell.columns and cell.rho_cal.notna().any():
            parts.append(pd.DataFrame({"series": "cal", "time_bin": cell.time_bin.to_numpy(),
                                       "rho": cell.rho_cal.to_numpy()}))
        rho_profile = pd.concat(parts, ignore_index=True)
        rho_profile.insert(0, "weekday", int(wd_pick))
        rho_profile.insert(0, "resource", str(res_pick))
        rho_profile.insert(0, "candidate_id", rep_cand)

    response = pd.DataFrame(columns=["candidate_id", "resource", "gap", "resp", "nchg"])
    if not dlong.empty:
        gap_res  = (prof.assign(ag=(prof.rho_obs - prof.rho_sim).abs())
                    .groupby("resource")["ag"].mean())
        resp_res = dlong.assign(ad=dlong.delta.abs()).groupby("resource")["ad"].mean()
        n_seeds_dlong = max(dlong["seed"].nunique(), 1) if "seed" in dlong.columns else 1
        nchg_res = (dlong.assign(chg=dlong.delta.abs() > 1e-9)
                    .groupby("resource")["chg"].sum() / n_seeds_dlong)
        res_df = pd.concat([gap_res.rename("gap"), resp_res.rename("resp"),
                            nchg_res.rename("nchg")], axis=1).dropna()
        response = res_df.rename_axis("resource").reset_index()
        response.insert(0, "candidate_id", rep_cand)

    return {"rho_profile": rho_profile,
            "revisions": revisions,
            "response": response,
            "distances": _distance_frame(cands, dist, "candidate_id")}


# ── Writers ────────────────────────────────────────────────────────────────────
def write_synthetic(run_dir: Path, n_seeds: int, out_dir: Path,
                    settings_configs: list[Path] | None = None) -> None:
    """The series behind the synthetic figure, its distances, its observed fractions."""
    dp = _distance_to_ptrue(run_dir)
    scen = list(dp["scenario"])
    raise_v, lower_v = _revision_counts(run_dir, n_seeds, scen)
    valid_dev, series = _process_series(run_dir, n_seeds, scen)
    if not valid_dev:
        raise ValueError("no seed has process_metrics.csv")

    _write(out_dir / "synthetic_rho_profile.csv", _rho_profile(run_dir))
    _write(out_dir / "synthetic_revisions.csv",
           pd.DataFrame({"scenario": scen, "raised": raise_v, "lowered": lower_v}))
    _write(out_dir / "synthetic_distance_to_ptrue.csv", dp)
    _write(out_dir / "synthetic_distances.csv", _distance_frame(valid_dev, series, "scenario"))
    _write(out_dir / "synthetic_observed_fractions.csv", _observed_fractions(run_dir))
    if settings_configs:
        _write(out_dir / "synthetic_settings.csv", _settings_series(settings_configs))


def write_real(run_dir: Path, n_seeds: int, out_dir: Path) -> None:
    """The series behind the real-log figure."""
    frames = _real_series(run_dir, n_seeds)
    _write(out_dir / "real_rho_profile.csv", frames["rho_profile"])
    _write(out_dir / "real_revisions.csv", frames["revisions"])
    _write(out_dir / "real_response.csv", frames["response"])
    _write(out_dir / "real_distances.csv", frames["distances"])


def write_run_series(run_dir, n_synth_seeds=10, n_real_seeds=10,
                     settings_configs: list[Path] | None = None) -> None:
    """Write the synth and real series independently (whichever data the run holds; the other is skipped)."""
    out_dir = Path(run_dir) / "report"
    for name, fn, n in (("synthetic",
                         lambda rd, k, od: write_synthetic(rd, k, od, settings_configs), n_synth_seeds),
                        ("real", write_real, n_real_seeds)):
        try:
            fn(Path(run_dir), n, out_dir)
        except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError) as e:
            log(f"[report-series] skipped {name} series — run lacks {name} data ({type(e).__name__})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-synth-seeds", type=int, default=10)
    ap.add_argument("--n-real-seeds",  type=int, default=10)
    ap.add_argument("--settings-configs", nargs="+", type=Path,
                    help="also write the availability settings of these configuration sets")
    args = ap.parse_args()

    write_run_series(Path(args.run_dir), args.n_synth_seeds, args.n_real_seeds,
                     args.settings_configs)
    print("Done.")


if __name__ == "__main__":
    main()
