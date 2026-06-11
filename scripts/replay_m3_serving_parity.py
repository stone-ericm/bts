"""Serving-parity replay for audit M3: decision-level cost of stale inference lookups.

Production serving (`_build_feature_lookups`) reads each entity's feature at its
most recent PLAYED date; the shift(1) training contract means that value excludes
the entity's latest game — serving is one played-date stale, everywhere, always
(audit 2026-06-09 [P1]; sized in scripts/spotcheck_m3_staleness.py: bpm 0.73 std,
hr_7g 0.38 std). The backtest scores feature ROWS directly, so no backtest number
has ever seen this gap — it is pure serving-path degradation.

This replay measures the decision-level cost. For each fold season S:
  1. Train the production single model + 12-model blend on seasons < S
     (bts.model.predict.train_model / train_blend — the real functions).
  2. Slate = actual starters (lineup_position 1-9), one row per (batter, game),
     pitcher = the batter's first pitcher faced (starter proxy).
  3. Score each slate twice with identical machinery:
       FRESH arm: feature value as-of the slate date  (post-fix serving)
       STALE arm: value at the entity's previous played date (current serving)
     Both arms use last-non-NaN fallback (ffill), matching the lookups'
     dropna().last() semantics; only freshness differs.
  4. Compare: top-1/top-3 daily hit rate, game-level AUC, pick divergence,
     paired day-bootstrap CI on the top-1 delta.

Run: uv run python scripts/replay_m3_serving_parity.py [--folds 2024 2025 2026]
"""

import argparse
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from bts.features.compute import compute_all_features, FEATURE_COLS, STATCAST_COLS, TRAIN_START_YEAR
from bts.model.predict import train_model, train_blend

PRIOR_RATE = 0.2195
PA_EST = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}
STARTER_PAS = 2.5

# Serving feature -> lookup key columns (mirrors _build_feature_lookups)
KEY_GROUPS = {
    ("batter_id",): [
        "batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g",
        "batter_whiff_60g", "batter_count_tendency_30g", "batter_gb_hit_rate",
        "batter_barrel_rate_30g", "batter_hard_hit_rate_30g",
        "batter_sweet_spot_rate_30g", "batter_avg_ev_30g", "batter_avg_velo_faced_30g",
    ],
    ("batter_id", "pitch_hand"): ["platoon_hr"],
    ("pitcher_id",): [
        "pitcher_hr_30g", "pitcher_entropy_30g", "pitcher_catcher_framing",
        "pitcher_avg_velo_30g", "pitcher_avg_spin_30g",
        "pitcher_avg_extension_30g", "pitcher_break_total_30g",
    ],
    ("batter_id", "pitcher_id"): ["batter_pitcher_shrunk_hr"],
    ("venue_id",): ["park_factor"],
    ("opp_pitching_team_id",): ["opp_bullpen_hr_30g"],
}
# Slate-supplied either way (live feed / computed from current date) — same both arms.
PASSTHROUGH = ["weather_temp", "days_rest"]


def _bpm_arm_values(df_feat: pd.DataFrame, slate: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """bpm fresh/stale via as-of join, not exact-date merge.

    The generic level-frame path only works when the (batter, pitcher, slate
    date) row materialized — i.e. the pair actually met that day. When the
    slate's starter proxy and reality diverge (~10% of rows), an exact-date
    merge silently flattens BOTH arms to the prior, while production serving
    would return the pair's last-meeting value (caught by the parity check,
    max|diff| 0.145). First-principles recompute, golden-checked against the
    pipeline's own column, then merge_asof to the last meeting strictly
    before the slate date: STALE = that meeting's stored (shift-style) value
    = what production .last() serves; FRESH = the value including that
    meeting's outcomes = what fixed serving would deliver.
    """
    K = 10
    daily = (
        df_feat.groupby(["batter_id", "pitcher_id", "date"])
        .agg(h=("is_hit", "sum"), n=("is_hit", "count"))
        .reset_index()
        .sort_values("date", kind="mergesort")
    )
    g = daily.groupby(["batter_id", "pitcher_id"], sort=False)
    daily["cum_h"] = g["h"].cumsum()
    daily["cum_n"] = g["n"].cumsum()
    daily["stored"] = (PRIOR_RATE * K + daily["cum_h"] - daily["h"]) / (K + daily["cum_n"] - daily["n"])
    daily["incl"] = (PRIOR_RATE * K + daily["cum_h"]) / (K + daily["cum_n"])

    # Golden check: recomputed stored values must equal the pipeline's column
    chk = df_feat[["batter_id", "pitcher_id", "date", "batter_pitcher_shrunk_hr"]].merge(
        daily[["batter_id", "pitcher_id", "date", "stored"]],
        on=["batter_id", "pitcher_id", "date"], how="inner",
    )
    worst = (chk["batter_pitcher_shrunk_hr"] - chk["stored"]).abs().max()
    assert worst < 1e-9, f"bpm recompute diverges from pipeline: {worst}"

    left = slate[["batter_id", "pitcher_id", "date"]].reset_index().sort_values("date", kind="mergesort")
    m = pd.merge_asof(
        left, daily[["batter_id", "pitcher_id", "date", "stored", "incl"]],
        on="date", by=["batter_id", "pitcher_id"], allow_exact_matches=False,
    ).set_index("index")
    fresh_vals = m["incl"].reindex(slate.index).fillna(PRIOR_RATE)
    stale_vals = m["stored"].reindex(slate.index).fillna(PRIOR_RATE)
    return fresh_vals, stale_vals


def build_arm_values(df_feat: pd.DataFrame, slate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fresh, stale) copies of slate with arm-specific feature values."""
    fresh, stale = slate.copy(), slate.copy()
    fresh["batter_pitcher_shrunk_hr"], stale["batter_pitcher_shrunk_hr"] = _bpm_arm_values(df_feat, slate)
    for keys, cols in KEY_GROUPS.items():
        if keys == ("batter_id", "pitcher_id"):
            continue  # handled by _bpm_arm_values (as-of join, not exact-date)
        keys = list(keys)
        cols = [c for c in cols if c in df_feat.columns and df_feat[c].notna().any()]
        if not cols:
            continue
        if any(k not in df_feat.columns for k in keys):
            continue
        lvl = (
            df_feat.dropna(subset=keys)[keys + ["date"] + cols]
            .groupby(keys + ["date"], as_index=False)
            .first()
            .sort_values("date", kind="mergesort")
        )
        grp = lvl.groupby(keys, sort=False)
        for col in cols:
            lvl[f"_fresh_{col}"] = grp[col].transform(lambda s: s.ffill())
            lvl[f"_stale_{col}"] = grp[col].transform(lambda s: s.ffill().shift(1))
        merge_cols = keys + ["date"] + [f"_fresh_{c}" for c in cols] + [f"_stale_{c}" for c in cols]
        for arm_df, prefix in ((fresh, "_fresh_"), (stale, "_stale_")):
            m = arm_df.merge(lvl[merge_cols], on=keys + ["date"], how="left")
            for col in cols:
                arm_df[col] = m[f"{prefix}{col}"].to_numpy()
    return fresh, stale


def score_arm(arm: pd.DataFrame, model, blend: dict, league_reliever_hr: float,
              league_entropy: float) -> pd.Series:
    """Mirror predict()'s blend scoring: per-model p_game averaged per slot."""
    est = arm["lineup_position"].map(PA_EST).fillna(4.0)
    reliever_pas = (est - STARTER_PAS).clip(lower=0)
    starter_pas = est - reliever_pas

    feat = arm[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    for col in STATCAST_COLS:
        arm[col] = pd.to_numeric(arm[col], errors="coerce")

    # Single-model reliever estimate (pitcher quality -> league average)
    feat_rel = feat.copy()
    feat_rel["pitcher_hr_30g"] = league_reliever_hr
    feat_rel["pitcher_entropy_30g"] = league_entropy
    valid = feat.notna().any(axis=1)
    p_rel = pd.Series(np.nan, index=arm.index)
    p_rel[valid] = model.predict_proba(feat_rel[valid])[:, 1]

    per_model_pgame = []
    for name, (bmodel, bcols) in blend.items():
        bfeat = pd.concat([feat, arm[[c for c in bcols if c not in FEATURE_COLS]]], axis=1)[bcols]
        bvalid = bfeat.notna().any(axis=1)
        p_st = pd.Series(np.nan, index=arm.index)
        if bvalid.any():
            p_st[bvalid] = bmodel.predict_proba(bfeat[bvalid])[:, 1]
        p_game = 1 - ((1 - p_st) ** starter_pas * (1 - p_rel) ** reliever_pas)
        per_model_pgame.append(p_game)

    stacked = pd.concat(per_model_pgame, axis=1)
    return stacked.mean(axis=1, skipna=True)


def fold_metrics(slate: pd.DataFrame, label: str) -> dict:
    """Daily top-k hit rates + AUC for one arm's scores in slate[f'p_{label}'].

    Caller must pass a slate already restricted to rows where BOTH arms have
    scores (joint candidate mask), so the arms rank identical pools.
    """
    col = f"p_{label}"
    day = slate.sort_values(col, ascending=False)
    top1 = day.groupby("date").head(1)
    top3 = day.groupby("date").head(3)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(day["actual_hit"], day[col])
    return {
        "top1": top1["actual_hit"].mean(), "n_days": top1["date"].nunique(),
        "top3": top3["actual_hit"].mean(), "auc": auc,
        "top1_picks": top1.set_index("date")[["batter_id", "actual_hit"]],
    }


def parity_check_stale_arm(df_feat: pd.DataFrame, slate_stale: pd.DataFrame,
                           check_dates: list, n_sample: int = 200) -> None:
    """Golden test: the STALE arm must equal what production's
    _build_feature_lookups serves when built on data strictly before D.
    (Past feature rows are invariant to truncation — verified by the
    spot-check's 0-mismatch sanity — so truncating df_feat replicates the
    production frame for that morning.)"""
    from bts.model.predict import _build_feature_lookups
    rng = np.random.default_rng(7)
    worst = 0.0
    n_checked = 0
    for d in check_dates:
        lk = _build_feature_lookups(df_feat[df_feat["date"] < d])
        rows = slate_stale[slate_stale["date"] == d]
        if rows.empty:
            continue
        rows = rows.sample(min(n_sample, len(rows)), random_state=rng.integers(1 << 30))
        for _, r in rows.iterrows():
            for col in KEY_GROUPS[("batter_id",)]:
                v_lk = lk["batter"].get(col, {}).get(r["batter_id"])
                v_arm = r.get(col)
                if v_lk is not None and pd.notna(v_arm):
                    worst = max(worst, abs(v_lk - v_arm)); n_checked += 1
            v_lk = lk.get("batter_pitcher_hr", {}).get((r["batter_id"], r["pitcher_id"]))
            v_arm = r.get("batter_pitcher_shrunk_hr")
            if v_lk is not None and pd.notna(v_arm):
                worst = max(worst, abs(v_lk - v_arm)); n_checked += 1
            v_lk = lk.get("pitcher_hr", {}).get(r["pitcher_id"])
            v_arm = r.get("pitcher_hr_30g")
            if v_lk is not None and pd.notna(v_arm):
                worst = max(worst, abs(v_lk - v_arm)); n_checked += 1
    print(f"  parity check (STALE arm vs production lookups, {len(check_dates)} dates, "
          f"{n_checked} values): max|diff| = {worst:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--seasons-from", type=int, default=2017)
    args = ap.parse_args()

    proc = Path("data/processed")
    seasons = [int(p.stem.split("_")[1]) for p in sorted(proc.glob("pa_*.parquet"))]
    seasons = [s for s in seasons if s >= args.seasons_from]
    df = pd.concat([pd.read_parquet(proc / f"pa_{s}.parquet") for s in seasons], ignore_index=True)
    print(f"loaded {len(df)} PA rows, seasons {min(seasons)}-{max(seasons)}", flush=True)

    df_feat = compute_all_features(df)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    print("features computed", flush=True)

    all_days: list = []
    for fold in args.folds:
        train_df = df_feat[(df_feat["season"] >= TRAIN_START_YEAR) & (df_feat["season"] < fold)]
        season_df = df_feat[df_feat["season"] == fold]
        if season_df.empty or train_df.empty:
            print(f"fold {fold}: no data, skipping")
            continue

        print(f"\n=== fold {fold}: train {TRAIN_START_YEAR}-{fold-1} "
              f"({len(train_df)} PAs), replay {fold} ({season_df['date'].nunique()} days) ===", flush=True)

        model = train_model(train_df)
        blend = train_blend(train_df)
        print("  trained single + 12-model blend", flush=True)

        league_reliever_hr = float(train_df["is_hit"].mean())
        league_entropy = float(train_df["pitcher_entropy_30g"].mean())

        # Slate: actual starters (lineup 1-9), one row per (batter, game).
        # Opposing starter proxy: the modal pitcher faced by that game-side's
        # batters (the starter faces the most batters; PA row order within a
        # game is not guaranteed after compute's sort, so "first faced" isn't).
        starters = season_df[season_df["lineup_position"].between(1, 9)]
        slate = starters.groupby(["game_pk", "batter_id"], as_index=False).first()
        side_starter = (
            starters.groupby(["game_pk", "is_home"])["pitcher_id"]
            .agg(lambda s: s.mode().iat[0])
            .rename("starter_pid")
            .reset_index()
        )
        slate = slate.merge(side_starter, on=["game_pk", "is_home"], how="left")
        slate["pitcher_id"] = slate["starter_pid"].fillna(slate["pitcher_id"])
        hand_map = df_feat.dropna(subset=["pitch_hand"]).groupby("pitcher_id")["pitch_hand"].last()
        slate["pitch_hand"] = slate["pitcher_id"].map(hand_map)
        outcome = season_df.groupby(["game_pk", "batter_id"])["is_hit"].max().rename("actual_hit")
        slate = slate.merge(outcome, on=["game_pk", "batter_id"], how="left")
        slate = slate.dropna(subset=["actual_hit"])
        print(f"  slate: {len(slate)} batter-games", flush=True)

        fresh, stale = build_arm_values(df_feat, slate)

        # Merge coverage for the sparse composite keys (Codex review point:
        # exact-(key,date) misses fall back to prior in BOTH arms, which would
        # suppress the measured bpm staleness cost — so quantify the miss rate).
        bpm_lvl = df_feat[["batter_id", "pitcher_id", "date"]].drop_duplicates()
        bpm_hit = slate.merge(bpm_lvl, on=["batter_id", "pitcher_id", "date"], how="left", indicator=True)
        print(f"  bpm (batter,pitcher,date) merge coverage: "
              f"{(bpm_hit['_merge'] == 'both').mean():.1%} of slate rows", flush=True)

        fresh["p_fresh"] = score_arm(fresh, model, blend, league_reliever_hr, league_entropy)
        stale["p_stale"] = score_arm(stale, model, blend, league_reliever_hr, league_entropy)
        slate["p_fresh"] = fresh["p_fresh"]
        slate["p_stale"] = stale["p_stale"]

        # Joint candidate mask: both arms rank the same pool (Codex review).
        eval_slate = slate.dropna(subset=["p_fresh", "p_stale"])
        dropped = len(slate) - len(eval_slate)
        if dropped:
            print(f"  joint mask dropped {dropped} rows scored by only one arm", flush=True)

        mf = fold_metrics(eval_slate, "fresh")
        ms = fold_metrics(eval_slate, "stale")
        print(f"  FRESH (post-fix):  top1={mf['top1']:.4f}  top3={mf['top3']:.4f}  auc={mf['auc']:.4f}  days={mf['n_days']}")
        print(f"  STALE (current):   top1={ms['top1']:.4f}  top3={ms['top3']:.4f}  auc={ms['auc']:.4f}  days={ms['n_days']}")

        # Pick divergence + per-fold paired day bootstrap on top-1 delta
        j = mf["top1_picks"].join(ms["top1_picks"], lsuffix="_f", rsuffix="_s", how="inner")
        j["fold"] = fold
        all_days.append(j)
        diverged = (j["batter_id_f"] != j["batter_id_s"]).mean()
        d = (j["actual_hit_f"] - j["actual_hit_s"]).to_numpy(dtype=float)
        rng = np.random.default_rng(20260611)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        fo = (j["actual_hit_f"] == 1) & (j["actual_hit_s"] == 0)
        so = (j["actual_hit_s"] == 1) & (j["actual_hit_f"] == 0)
        print(f"  top-1 pick differs on {diverged:.1%} of days; "
              f"delta top1 = {d.mean():+.4f}  (95% CI [{lo:+.4f}, {hi:+.4f}])")
        print(f"  discordant days: fresh-only-hit={int(fo.sum())}  stale-only-hit={int(so.sum())}", flush=True)

        # Golden parity test: STALE arm vs the real production lookups
        fold_dates = sorted(eval_slate["date"].unique())
        check = [fold_dates[len(fold_dates) // 4], fold_dates[len(fold_dates) // 2],
                 fold_dates[3 * len(fold_dates) // 4]]
        parity_check_stale_arm(df_feat, stale, check)

    # --- Pooled cross-fold analysis (fold-stratified bootstrap + sign test) ---
    if all_days:
        pooled = pd.concat(all_days)
        d = (pooled["actual_hit_f"] - pooled["actual_hit_s"]).to_numpy(dtype=float)
        rng = np.random.default_rng(20260611)
        by_fold = [g[1][["actual_hit_f", "actual_hit_s"]].to_numpy(dtype=float)
                   for g in pooled.groupby("fold")]
        boots = []
        for _ in range(10000):
            tot = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in by_fold])
            boots.append((tot[:, 0] - tot[:, 1]).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        fo = int(((pooled["actual_hit_f"] == 1) & (pooled["actual_hit_s"] == 0)).sum())
        so = int(((pooled["actual_hit_s"] == 1) & (pooled["actual_hit_f"] == 0)).sum())
        from scipy.stats import binomtest
        p_sign = binomtest(fo, fo + so, 0.5).pvalue if fo + so else float("nan")
        print(f"\n=== POOLED ({len(pooled)} days, {len(all_days)} folds) ===")
        print(f"  delta top1 = {d.mean():+.4f}  (fold-stratified 95% CI [{lo:+.4f}, {hi:+.4f}])")
        print(f"  discordant days: fresh-only-hit={fo}, stale-only-hit={so} "
              f"(effective n; sign-test p={p_sign:.3f})")


if __name__ == "__main__":
    main()
