"""Screen arm registry for the park_drag_delta 2026 backtest.

Two-arm question with the spec's honest competitor and the house control
battery: does adding park-level ball-drag regime state to production
FEATURE_COLS improve within-day candidate ranking on 2026 — and does it beat
a same-recency rolling OUTCOME park factor (no new data dependency)?

Post-selection caveat (spec eval section): 2026 motivated the feature, so
this screen is SUPPORTING evidence, labeled as such — the promotion gate
remains the live-forward shadow. Feature values are drag-physics-derived
(labels never touched), which tempers but does not remove the caveat.

Arms:
  baseline            FEATURE_COLS + availability flags (flags in EVERY arm)
  pd_anchored         + park_drag_delta (the shipped shape)
  pd_expanding        + park_drag_delta_expanding (v1 shape, comparison)
  outcome_pf          + rolling_outcome_pf (rolling-15-venue-date hit rate —
                        the "just use outcomes" competitor)
  pd_plus_outcome     + both primary columns (complementarity probe)
  ctl_mask_only       NaN pattern kept, values destroyed
  ctl_permuted        venue-block permutation WITHIN DATE (game-mates keep a
                        common value; venue->outcome link broken)
  ctl_sentinel_gross  same-day game outcome (harness-leak canary; must explode)
  ctl_sentinel_soft   calibrated ~+0.005 graded leak (power gate)
  ctl_sentinel_leaky  export value at date+1 (includes same-day games) — the
                        M3-style same-day-leak canary specific to this join

Reuses the validated swing_screen slate construction and metric runner shape
(daily NDCG@10, per-day rank AUC, top1/top3), single split with
train_extra_through per the 2026-06-12 amendment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# variant arm columns the DRIVER attaches to the PA frame
_VARIANT_SOURCES = {
    "pd_anchored": ["park_drag_delta"],
    "pd_expanding": ["park_drag_delta_expanding"],
    "outcome_pf": ["rolling_outcome_pf"],
    "pd_plus_outcome": ["park_drag_delta", "rolling_outcome_pf"],
}
_ALL_VARIANT_COLS = ["park_drag_delta", "park_drag_delta_expanding", "rolling_outcome_pf"]

ARMS = (
    ["baseline"] + list(_VARIANT_SOURCES)
    + ["ctl_mask_only", "ctl_permuted",
       "ctl_sentinel_gross", "ctl_sentinel_soft", "ctl_sentinel_leaky"]
)

FAMILY_OF = {"baseline": "baseline",
             "pd_anchored": "park_drag", "pd_expanding": "park_drag",
             "outcome_pf": "competitor", "pd_plus_outcome": "park_drag",
             "ctl_mask_only": "control", "ctl_permuted": "control",
             "ctl_sentinel_gross": "control", "ctl_sentinel_soft": "control",
             "ctl_sentinel_leaky": "control"}


def rolling_outcome_pf(pa: pd.DataFrame) -> pd.DataFrame:
    """Rolling-15-venue-date hit rate, strictly prior — the same-recency
    OUTCOMES competitor the spec requires the drag feature to beat."""
    vd = (pa.groupby(["venue_id", "season", "date"])["is_hit"].mean()
            .rename("vd_hr").reset_index()
            .sort_values(["venue_id", "date"]))
    vd["rolling_outcome_pf"] = (
        vd.groupby(["venue_id", "season"])["vd_hr"]
        .transform(lambda x: x.shift(1).rolling(15, min_periods=5).mean())
    )
    return vd[["venue_id", "date", "rolling_outcome_pf"]]


def _with_avail_flags(pa: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Availability flags for every variant column, in the BASELINE and every
    arm — candidate deltas must measure value beyond coverage information."""
    frame = pa.copy()
    flag_cols = []
    for c in _ALL_VARIANT_COLS:
        if c in frame.columns:
            frame[f"has_{c}"] = frame[c].notna()
            flag_cols.append(f"has_{c}")
    if not flag_cols:
        raise ValueError("no variant source columns present in frame")
    return frame, flag_cols


def _venue_block_permute(frame: pd.DataFrame, col: str, permute_seed: int) -> pd.Series:
    """Shuffle each date's per-venue values among that date's venues.

    Game-mates keep a shared value (the feature is venue-level), the day's
    marginal distribution is preserved, and the venue->outcome link is broken.
    Row-level within-date shuffling would be an easier null (pure noise);
    block permutation is the stricter one for a game-level feature.
    """
    rng = np.random.default_rng(permute_seed)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, day in frame.groupby("date"):
        venue_vals = day.groupby("venue_id")[col].first()
        shuffled = pd.Series(
            rng.permutation(venue_vals.to_numpy()), index=venue_vals.index)
        out.loc[day.index] = day["venue_id"].map(shuffled).to_numpy()
    return out


def build_arm_frame(
    arm: str, pa: pd.DataFrame, permute_seed: int = 13,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, extra_cols); extra_cols are ADDED to FEATURE_COLS."""
    base_frame, flag_cols = _with_avail_flags(pa)

    if arm == "baseline":
        return base_frame, flag_cols

    if arm in _VARIANT_SOURCES:
        cols = [c for c in _VARIANT_SOURCES[arm] if c in base_frame.columns]
        if not cols:
            raise ValueError(f"arm {arm}: no source columns present")
        return base_frame, flag_cols + cols

    if arm == "ctl_mask_only":
        frame = base_frame
        mask_cols = []
        for c in _ALL_VARIANT_COLS:
            if c in frame.columns:
                name = f"mask_{c}"
                frame[name] = np.where(frame[c].notna(), 1.0, np.nan)
                mask_cols.append(name)
        return frame, flag_cols + mask_cols

    if arm == "ctl_permuted":
        frame = base_frame
        perm_cols = []
        for c in _ALL_VARIANT_COLS:
            if c in frame.columns:
                name = f"perm_{c}"
                frame[name] = _venue_block_permute(frame, c, permute_seed)
                perm_cols.append(name)
        return frame, flag_cols + perm_cols

    if arm == "ctl_sentinel_gross":
        if "ORACLE_game_hit" not in pa.columns:
            raise ValueError("ctl_sentinel_gross requires ORACLE_game_hit (driver attaches)")
        return base_frame, flag_cols + ["ORACLE_game_hit"]

    if arm == "ctl_sentinel_soft":
        if "SOFT_ORACLE" not in pa.columns:
            raise ValueError("ctl_sentinel_soft requires SOFT_ORACLE (driver attaches)")
        return base_frame, flag_cols + ["SOFT_ORACLE"]

    if arm == "ctl_sentinel_leaky":
        if "LEAK_pd_next_date" not in pa.columns:
            raise ValueError("ctl_sentinel_leaky requires LEAK_pd_next_date (driver attaches)")
        return base_frame, flag_cols + ["LEAK_pd_next_date"]

    raise KeyError(f"unknown arm: {arm}")


def run_screen_arm(
    arm: str,
    pa: pd.DataFrame,
    train_seasons: tuple,
    screen_season: int,
    seed: int,
    out_dir=None,
    permute_seed: int = 13,
    train_extra_through: str | None = None,
    screen_start: str | None = None,
) -> dict:
    """Train one LightGBM (FEATURE_COLS + arm cols), score screen slates.

    Body mirrors swing_screen.run_screen_arm (validated metric construction);
    payload adds per-day positive/negative counts for pair-weighted report
    aggregation and the post-change-point window split downstream.
    """
    import json as _json
    from pathlib import Path as _Path

    import lightgbm as lgb

    from bts.experiment.swing_screen import PA_EST, _slate_for_season
    from bts.health.slate_auc import _rank_auc
    from bts.model.predict import FEATURE_COLS, LGB_PARAMS
    from bts.validate.slate_rank import daily_ndcg

    frame, extra_cols = build_arm_frame(arm, pa, permute_seed=permute_seed)
    cols = FEATURE_COLS + extra_cols

    train_mask = frame["season"].isin(train_seasons)
    if train_extra_through is not None:
        train_mask |= (frame["season"] == screen_season) & (
            frame["date"] <= pd.Timestamp(train_extra_through)
        )
    train = frame[train_mask]
    params = {**LGB_PARAMS, "deterministic": True, "force_row_wise": True}
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
            n_pos = int((day["actual_hit"] == 1).sum())
            n_neg = int((day["actual_hit"] == 0).sum())
            day_auc = _rank_auc(
                day.loc[day["actual_hit"] == 1, "p_game"].tolist(),
                day.loc[day["actual_hit"] == 0, "p_game"].tolist(),
            )
            days.append({
                "date": str(d.date()), "ndcg": v, "day_auc": day_auc,
                "n_pos": n_pos, "n_neg": n_neg,
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
        "extra_cols": extra_cols, "n_days": len(days),
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
