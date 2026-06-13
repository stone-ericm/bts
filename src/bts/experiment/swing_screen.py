"""Stage-1 screen arm registry for the Statcast swing campaign.

Each arm = production FEATURE_COLS + the arm's swing columns. Base rolling
columns are produced by bts.features.swing (attach step in the driver);
derived columns (drifts, interactions, controls) are built here so every
definition is registry-local and testable.

Pre-registered inventory per the spec/plan: baseline, 18 single-variant arms
across families P/B/T/S/M, 6 omnibus arms, 3 controls. Family verdicts and
selection rules live in scripts/swing_screen_report.py.

Omnibus/control builders materialize only variants whose source columns are
present in the frame (the production driver frame carries all of them; small
test frames need not).

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_P = "pitcher"
_B = "batter"

# derived variant name -> (required source columns, builder)
_DERIVED = {
    "t_intercept_drift": (
        [f"{_B}_intercept_y_7g", f"{_B}_intercept_y_60g"],
        lambda pa: pa[f"{_B}_intercept_y_7g"] - pa[f"{_B}_intercept_y_60g"],
    ),
    "t_miss_drift": (
        [f"{_B}_miss_dist_7g", f"{_B}_miss_dist_60g"],
        lambda pa: pa[f"{_B}_miss_dist_7g"] - pa[f"{_B}_miss_dist_60g"],
    ),
    "s_swinglen_drift": (
        [f"{_B}_swing_len_7g", f"{_B}_swing_len_60g"],
        lambda pa: pa[f"{_B}_swing_len_7g"] - pa[f"{_B}_swing_len_60g"],
    ),
    "m_high_alignment": (
        [f"{_B}_whiff_high_share_30g", f"{_P}_whiff_high_share_30g"],
        lambda pa: pa[f"{_B}_whiff_high_share_30g"] * pa[f"{_P}_whiff_high_share_30g"],
    ),
    "m_high_mismatch": (
        [f"{_B}_whiff_high_share_30g", f"{_P}_whiff_high_share_30g"],
        lambda pa: (pa[f"{_B}_whiff_high_share_30g"] - pa[f"{_P}_whiff_high_share_30g"]).abs(),
    ),
}

# variant arm name -> source (a plain attached column, or a _DERIVED key)
_SINGLE_VARIANTS = {
    # P family
    "p_miss_7g": f"{_P}_miss_dist_7g",
    "p_miss_15g": f"{_P}_miss_dist_15g",
    "p_miss_30g": f"{_P}_miss_dist_30g",
    "p_miss_60g": f"{_P}_miss_dist_60g",
    "p_miss_std_30g": f"{_P}_miss_std_30g",
    "p_high_share_30g": f"{_P}_whiff_high_share_30g",
    # B family
    "b_miss_7g": f"{_B}_miss_dist_7g",
    "b_miss_15g": f"{_B}_miss_dist_15g",
    "b_miss_30g": f"{_B}_miss_dist_30g",
    "b_miss_60g": f"{_B}_miss_dist_60g",
    "b_miss_std_30g": f"{_B}_miss_std_30g",
    # T family (derived)
    "t_intercept_drift": "t_intercept_drift",
    "t_miss_drift": "t_miss_drift",
    # S family
    "s_swinglen_drift": "s_swinglen_drift",
    "s_attack_std_30g": f"{_B}_attack_std_30g",
    "s_attack_angle_30g": f"{_B}_attack_angle_30g",
    # M family (derived)
    "m_high_alignment": "m_high_alignment",
    "m_high_mismatch": "m_high_mismatch",
}

_FAMILY_MEMBERS = {
    "P": ["p_miss_7g", "p_miss_15g", "p_miss_30g", "p_miss_60g",
          "p_miss_std_30g", "p_high_share_30g"],
    "B": ["b_miss_7g", "b_miss_15g", "b_miss_30g", "b_miss_60g", "b_miss_std_30g"],
    "T": ["t_intercept_drift", "t_miss_drift"],
    "S": ["s_swinglen_drift", "s_attack_std_30g", "s_attack_angle_30g"],
    "M": ["m_high_alignment", "m_high_mismatch"],
}

ARMS = (
    ["baseline"]
    + list(_SINGLE_VARIANTS)
    + [f"omni_{f}" for f in _FAMILY_MEMBERS]
    + ["omni_ALL", "ctl_mask_only", "ctl_permuted",
       "ctl_sentinel_gross", "ctl_sentinel_soft", "ctl_sentinel_m3"]
)

FAMILY_OF = {"baseline": "baseline"}
for fam, members in _FAMILY_MEMBERS.items():
    for m in members:
        FAMILY_OF[m] = fam
    FAMILY_OF[f"omni_{fam}"] = "omnibus"
FAMILY_OF["omni_ALL"] = "omnibus"
for c in ("ctl_mask_only", "ctl_permuted", "ctl_sentinel_gross",
          "ctl_sentinel_soft", "ctl_sentinel_m3"):
    FAMILY_OF[c] = "control"


def _sources_present(pa: pd.DataFrame, variant: str) -> bool:
    src = _SINGLE_VARIANTS[variant]
    if src in _DERIVED:
        return all(c in pa.columns for c in _DERIVED[src][0])
    return src in pa.columns


def _materialize(pa: pd.DataFrame, variant: str) -> pd.Series:
    src = _SINGLE_VARIANTS[variant]
    if src in _DERIVED:
        return _DERIVED[src][1](pa)
    return pa[src]


def _build_variants(pa: pd.DataFrame, variants: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = pa.copy()
    cols = []
    for variant in variants:
        if not _sources_present(pa, variant):
            continue
        frame[variant] = _materialize(pa, variant)
        cols.append(variant)
    if not cols:
        raise ValueError("no variant source columns present in frame")
    return frame, cols


def _with_avail_flags(pa: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Availability flags for every materializable variant (amendment #2:
    these live in the BASELINE and every arm, so candidate deltas measure
    value beyond coverage/roster information)."""
    frame, cols = _build_variants(pa, list(_SINGLE_VARIANTS))
    flag_cols = []
    for c in cols:
        frame[f"has_{c}"] = frame[c].notna()
        flag_cols.append(f"has_{c}")
    return frame.drop(columns=cols), flag_cols


def build_arm_frame(
    arm: str, pa: pd.DataFrame, permute_seed: int = 13,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, swing_cols) for an arm. frame = pa + the arm's columns;
    swing_cols are ADDED to production FEATURE_COLS by the runner. Every arm
    (baseline included) carries the availability flags."""
    base_frame, flag_cols = _with_avail_flags(pa)

    if arm == "baseline":
        return base_frame, flag_cols

    if arm in _SINGLE_VARIANTS:
        frame, cols = _build_variants(base_frame, [arm])
        return frame, flag_cols + cols

    if arm.startswith("omni_") and arm != "omni_ALL":
        fam = arm.split("_", 1)[1]
        frame, cols = _build_variants(base_frame, _FAMILY_MEMBERS[fam])
        return frame, flag_cols + cols

    if arm == "omni_ALL":
        frame, cols = _build_variants(base_frame, list(_SINGLE_VARIANTS))
        return frame, flag_cols + cols

    if arm == "ctl_mask_only":
        # NaN pattern preserved, value content destroyed (constant where
        # present) — isolates the LightGBM NaN-routing channel beyond flags.
        frame, cols = _build_variants(base_frame, list(_SINGLE_VARIANTS))
        mask_cols = []
        for c in cols:
            name = f"mask_{c}"
            frame[name] = np.where(frame[c].notna(), 1.0, np.nan)
            mask_cols.append(name)
        return frame.drop(columns=cols), flag_cols + mask_cols

    if arm == "ctl_permuted":
        # Shuffle feature values WITHIN DATE (Codex r3/r5: preserve the day's
        # marginal distribution, BREAK the entity->outcome link). Permuting
        # within batter_id was WRONG — it preserved each batter's mean and so
        # leaked batter-quality signal (a null arm showed +0.028, caught on the
        # first-seed sanity 2026-06-13). Within-date breaks that cleanly.
        frame, cols = _build_variants(base_frame, list(_SINGLE_VARIANTS))
        rng = np.random.default_rng(permute_seed)
        perm_cols = []
        for c in cols:
            name = f"perm_{c}"
            frame[name] = (
                frame.groupby("date")[c]
                .transform(lambda s: s.sample(frac=1, random_state=rng.integers(1 << 30)).to_numpy())
            )
            perm_cols.append(name)
        return frame.drop(columns=cols), flag_cols + perm_cols

    if arm == "ctl_sentinel_gross":
        # GROSS PLUMBING CANARY = same-day game outcome (the label). Proven to
        # explode (auc->1.0) 2026-06-13: it is the definitive "can the harness
        # see a leak" test. The earlier same-day-whiff-count sentinel was too
        # weak at GAME granularity (a batter whiffs AND singles in one game),
        # which is why it didn't inflate — the harness was never broken.
        if "ORACLE_game_hit" not in pa.columns:
            raise ValueError("ctl_sentinel_gross requires ORACLE_game_hit (driver attaches)")
        return base_frame, flag_cols + ["ORACLE_game_hit"]

    if arm == "ctl_sentinel_soft":
        # CALIBRATED SOFT-ORACLE (Codex round-5): a graded leak tuned to ~+0.005
        # daily rank-AUC, so the screen's POWER at the candidate effect size is
        # directly tested — not just the trivial full oracle. Driver attaches.
        if "SOFT_ORACLE" not in pa.columns:
            raise ValueError("ctl_sentinel_soft requires SOFT_ORACLE (driver attaches)")
        return base_frame, flag_cols + ["SOFT_ORACLE"]

    if arm == "ctl_sentinel_m3":
        if "M3LEAK_batter_miss_dist_30g" not in pa.columns:
            raise ValueError("ctl_sentinel_m3 requires M3LEAK_batter_miss_dist_30g (driver attaches)")
        return base_frame, flag_cols + ["M3LEAK_batter_miss_dist_30g"]

    raise KeyError(f"unknown arm: {arm}")


# --- screen runner -----------------------------------------------------------

PA_EST = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}


def _slate_for_season(pa: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (batter, game) for actual starters (lineup 1-9); outcome =
    any hit in that game. Mirrors the validated replay_m3_serving_parity
    construction."""
    sdf = pa[(pa["season"] == season) & (pa["lineup_position"].between(1, 9))]
    slate = sdf.groupby(["game_pk", "batter_id"], as_index=False).first()
    outcome = sdf.groupby(["game_pk", "batter_id"])["is_hit"].max().rename("actual_hit")
    slate = slate.drop(columns=["is_hit"]).merge(outcome, on=["game_pk", "batter_id"], how="left")
    return slate.dropna(subset=["actual_hit"])


def run_screen_arm(
    arm: str,
    pa: pd.DataFrame,
    train_seasons: tuple,
    screen_season: int,
    seed: int,
    base_cols: list | None = None,
    lgb_overrides: dict | None = None,
    out_dir=None,
    permute_seed: int = 13,
    train_extra_through: str | None = None,
    screen_start: str | None = None,
) -> dict:
    """Train one LightGBM on train_seasons, score screen_season slates,
    return + persist the per-arm metric payload.

    train_extra_through/screen_start implement the 2026-06-12 spec amendment:
    early screen-season dates (through train_extra_through) join TRAINING so
    swing features have learnable coverage; the slate starts at screen_start.
    """
    import json as _json
    from pathlib import Path as _Path

    import lightgbm as lgb

    from bts.health.slate_auc import _rank_auc
    from bts.model.predict import FEATURE_COLS, LGB_PARAMS
    from bts.validate.slate_rank import daily_ndcg  # noqa: F401 (used below)

    frame, swing_cols = build_arm_frame(arm, pa, permute_seed=permute_seed)
    cols = (base_cols if base_cols is not None else FEATURE_COLS) + swing_cols

    train_mask = frame["season"].isin(train_seasons)
    if train_extra_through is not None:
        train_mask |= (frame["season"] == screen_season) & (
            frame["date"] <= pd.Timestamp(train_extra_through)
        )
    train = frame[train_mask]
    params = {**LGB_PARAMS, **(lgb_overrides or {}),
              "deterministic": True, "force_row_wise": True}
    model = lgb.LGBMClassifier(**params, random_state=seed)
    X = train[cols]
    mask = X.notna().any(axis=1) & train["is_hit"].notna()
    model.fit(X[mask], train["is_hit"][mask])

    slate = _slate_for_season(frame, screen_season)
    if screen_start is not None:
        slate = slate[slate["date"] >= pd.Timestamp(screen_start)]
    p_pa = model.predict_proba(slate[cols])[:, 1]
    est = slate["lineup_position"].map(PA_EST).fillna(4.0)
    slate = slate.assign(p_game=1 - (1 - p_pa) ** est)

    days = []
    for d, day in slate.groupby("date"):
        v = daily_ndcg(day, "p_game", k=10)
        if not np.isnan(v):
            top = day.sort_values("p_game", ascending=False)
            day_auc = _rank_auc(
                day.loc[day["actual_hit"] == 1, "p_game"].tolist(),
                day.loc[day["actual_hit"] == 0, "p_game"].tolist(),
            )
            days.append({
                "date": str(d.date()), "ndcg": v,
                "day_auc": day_auc,  # amendment #2 primary screen stat input
                "top1": int(top["actual_hit"].iloc[0]),
                "top3": float(top["actual_hit"].head(3).mean()),
            })
    auc = _rank_auc(
        slate.loc[slate["actual_hit"] == 1, "p_game"].tolist(),
        slate.loc[slate["actual_hit"] == 0, "p_game"].tolist(),
    )

    res = {
        "arm": arm, "seed": seed, "family": FAMILY_OF[arm],
        "train_seasons": list(train_seasons), "screen_season": screen_season,
        "train_extra_through": train_extra_through, "screen_start": screen_start,
        "n_swing_cols": len(swing_cols), "n_days": len(days),
        "ndcg_mean": float(np.mean([x["ndcg"] for x in days])) if days else None,
        "top1_hit": float(np.mean([x["top1"] for x in days])) if days else None,
        "top3_hit": float(np.mean([x["top3"] for x in days])) if days else None,
        "auc": auc,
        "per_day": days,
    }
    if out_dir is not None:
        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{arm}_seed{seed}.json").write_text(_json.dumps(res))
    return res


# --- residual-stacking screen (amendment #3, Codex round-4 co-design) ---------
# Removes the covered-only-baseline confound: a full-history production prior
# (production features only) is cross-fit OOF onto covered training rows and
# predicted onto eval rows, then every arm trains on covered rows with the
# prior as a base feature. The delta isolates swing value ON TOP of the
# full-strength production model. See docs/audit/2026-06-13-swing-screen-codex-r4-codesign.md


def _week_fold_id(dates: pd.Series, n_folds: int) -> pd.Series:
    """Group ISO weeks into n_folds buckets (deterministic, no RNG)."""
    iso = pd.to_datetime(dates).dt.isocalendar()
    wk = (iso["year"].astype(int) * 53 + iso["week"].astype(int))
    uniq = {w: i for i, w in enumerate(sorted(wk.unique()))}
    return wk.map(uniq) % n_folds


def build_prod_prior_oof(
    pa: pd.DataFrame,
    prod_features: list[str],
    full_mask: pd.Series,
    cov_mask: pd.Series,
    eval_mask: pd.Series,
    seed: int,
    lgb_overrides: dict | None = None,
    n_folds: int = 5,
) -> pd.Series:
    """Full-history production prior as a feature, leak-free.

    OOF on covered-train rows: for each week-fold of cov_mask, train on
    full_mask MINUS *all rows in the held-out WEEKS* (Codex round-5: group
    exclusion, not just the cov rows — otherwise same-week uncovered rows
    leak date-correlated signal into the held-out prediction), predict the
    fold. Final model (all of full_mask) predicts eval. Returns a Series
    aligned to pa.index; NaN outside cov∪eval.
    """
    import lightgbm as lgb
    from bts.model.predict import LGB_PARAMS

    assert pa.index.is_unique, "build_prod_prior_oof requires a unique index"
    assert not (cov_mask & eval_mask).any(), "cov and eval overlap"
    assert "is_hit" in pa.columns
    assert all(c not in prod_features for c in ("ORACLE_game_hit", "M3LEAK_batter_miss_dist_30g")), \
        "leak columns must not be in prod_features"

    params = {**LGB_PARAMS, **(lgb_overrides or {}), "deterministic": True, "force_row_wise": True}
    out = pd.Series(np.nan, index=pa.index)

    week = pa["date"].dt.to_period("W")
    cov_idx = pa.index[cov_mask]
    folds = _week_fold_id(pa.loc[cov_idx, "date"], n_folds)
    full_idx = pa.index[full_mask]
    for f in range(n_folds):
        holdout = cov_idx[folds.to_numpy() == f]
        if len(holdout) == 0:
            continue
        holdout_weeks = set(week.loc[holdout].unique())
        train_rows = full_idx[~week.loc[full_idx].isin(holdout_weeks).to_numpy()]
        m = lgb.LGBMClassifier(**params, random_state=seed)
        m.fit(pa.loc[train_rows, prod_features], pa.loc[train_rows, "is_hit"])
        out.loc[holdout] = m.predict_proba(pa.loc[holdout, prod_features])[:, 1]

    final = lgb.LGBMClassifier(**params, random_state=seed)
    final.fit(pa.loc[full_idx, prod_features], pa.loc[full_idx, "is_hit"])
    ev_idx = pa.index[eval_mask]
    out.loc[ev_idx] = final.predict_proba(pa.loc[ev_idx, prod_features])[:, 1]
    assert out.loc[cov_mask].notna().all() and out.loc[eval_mask].notna().all()
    return out


def _daily_rank(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("date")[col].rank(pct=True)


def run_residual_arm(
    pa: pd.DataFrame,
    swing_cols: list[str],
    cov_mask: pd.Series,
    eval_mask: pd.Series,
    seed: int,
    prior_col: str = "prod_prior",
    lgb_overrides: dict | None = None,
    extra_cols: dict | None = None,
    out_dir=None,
    arm_name: str | None = None,
) -> dict:
    """Train and score a covered-era residual model at the GAME-BATTER SLATE
    level (Codex round-5: train granularity must match score granularity, and
    prior_rank must be ranked over the scored candidate universe — not PA rows,
    whose multiplicity encodes post-game PA count). base = prior + slate-level
    prior rank + swing_cols. Returns the metric payload."""
    import json as _json
    from pathlib import Path as _Path
    import lightgbm as lgb
    from bts.health.slate_auc import _rank_auc
    from bts.model.predict import LGB_PARAMS
    from bts.validate.slate_rank import daily_ndcg

    frame = pa.copy()
    if extra_cols:
        for k, v in extra_cols.items():
            frame[k] = v
    base = [prior_col, "_prior_rank"]
    cols = base + list(swing_cols)

    def _slate(mask: pd.Series) -> pd.DataFrame:
        # deterministic: sort by stable keys before first(); same-day rolling
        # features are constant per batter-date, lineup is per game.
        sub = frame[mask].sort_values(["date", "game_pk", "batter_id"], kind="mergesort")
        s = sub.groupby(["game_pk", "batter_id"], as_index=False).first()
        outcome = sub.groupby(["game_pk", "batter_id"])["is_hit"].max().rename("actual_hit")
        s = s.drop(columns=["is_hit"]).merge(outcome, on=["game_pk", "batter_id"], how="left")
        s = s.dropna(subset=["actual_hit"])
        s["_prior_rank"] = s.groupby("date")[prior_col].rank(pct=True)
        return s

    cov_slate = _slate(cov_mask)
    eval_slate = _slate(eval_mask)

    params = {**LGB_PARAMS, **(lgb_overrides or {}), "deterministic": True, "force_row_wise": True}
    model = lgb.LGBMClassifier(**params, random_state=seed)
    model.fit(cov_slate[cols], cov_slate["actual_hit"])
    # slate-level model predicts P(game has a hit) directly — no est_pas
    eval_slate["p_game"] = model.predict_proba(eval_slate[cols])[:, 1]
    slate = eval_slate

    days = []
    for d, day in slate.groupby("date"):
        v = daily_ndcg(day, "p_game", k=10)
        if not np.isnan(v):
            top = day.sort_values("p_game", ascending=False)
            day_auc = _rank_auc(day.loc[day.actual_hit == 1, "p_game"].tolist(),
                                day.loc[day.actual_hit == 0, "p_game"].tolist())
            days.append({"date": str(pd.Timestamp(d).date()), "ndcg": v, "day_auc": day_auc,
                         "top1": int(top["actual_hit"].iloc[0]),
                         "top3": float(top["actual_hit"].head(3).mean())})
    auc = _rank_auc(slate.loc[slate.actual_hit == 1, "p_game"].tolist(),
                    slate.loc[slate.actual_hit == 0, "p_game"].tolist())
    res = {"arm": arm_name or "+".join(swing_cols) or "baseline", "seed": seed,
           "n_days": len(days), "auc_mean": float(auc) if auc is not None else float("nan"),
           "ndcg_mean": float(np.mean([x["ndcg"] for x in days])) if days else None,
           "top1_hit": float(np.mean([x["top1"] for x in days])) if days else None,
           "per_day": days}
    if out_dir is not None and arm_name:
        out_dir = _Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{arm_name}_seed{seed}.json").write_text(_json.dumps(res))
    return res
