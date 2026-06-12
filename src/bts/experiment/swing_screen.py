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
    + ["omni_ALL", "ctl_placebo", "ctl_permuted", "ctl_sentinel"]
)

FAMILY_OF = {"baseline": "baseline"}
for fam, members in _FAMILY_MEMBERS.items():
    for m in members:
        FAMILY_OF[m] = fam
    FAMILY_OF[f"omni_{fam}"] = "omnibus"
FAMILY_OF["omni_ALL"] = "omnibus"
for c in ("ctl_placebo", "ctl_permuted", "ctl_sentinel"):
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


def build_arm_frame(
    arm: str, pa: pd.DataFrame, permute_seed: int = 13,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, swing_cols) for an arm. frame = pa + the arm's columns;
    swing_cols are ADDED to production FEATURE_COLS by the runner."""
    if arm == "baseline":
        return pa.copy(), []

    if arm in _SINGLE_VARIANTS:
        return _build_variants(pa, [arm])

    if arm.startswith("omni_") and arm != "omni_ALL":
        fam = arm.split("_", 1)[1]
        return _build_variants(pa, _FAMILY_MEMBERS[fam])

    if arm == "omni_ALL":
        return _build_variants(pa, list(_SINGLE_VARIANTS))

    if arm == "ctl_placebo":
        frame, cols = _build_variants(pa, list(_SINGLE_VARIANTS))
        flag_cols = []
        for c in cols:
            frame[f"has_{c}"] = frame[c].notna()
            flag_cols.append(f"has_{c}")
        return frame[list(pa.columns) + flag_cols], flag_cols

    if arm == "ctl_permuted":
        frame, cols = _build_variants(pa, list(_SINGLE_VARIANTS))
        rng = np.random.default_rng(permute_seed)
        perm_cols = []
        for c in cols:
            name = f"perm_{c}"
            frame[name] = (
                frame.groupby("batter_id")[c]
                .transform(lambda s: s.sample(frac=1, random_state=rng.integers(1 << 30)).to_numpy())
            )
            perm_cols.append(name)
        return frame, perm_cols

    if arm == "ctl_sentinel":
        # driver attaches LEAKY_same_day_miss via bts.features.swing.build_leaky_sentinel
        if "LEAKY_same_day_miss" not in pa.columns:
            raise ValueError("ctl_sentinel requires LEAKY_same_day_miss attached by the driver")
        return pa.copy(), ["LEAKY_same_day_miss"]

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
) -> dict:
    """Train one LightGBM on train_seasons, score screen_season slates,
    return + persist the per-arm metric payload."""
    import json as _json
    from pathlib import Path as _Path

    import lightgbm as lgb

    from bts.health.slate_auc import _rank_auc
    from bts.model.predict import FEATURE_COLS, LGB_PARAMS
    from bts.validate.slate_rank import daily_ndcg

    frame, swing_cols = build_arm_frame(arm, pa, permute_seed=permute_seed)
    cols = (base_cols if base_cols is not None else FEATURE_COLS) + swing_cols

    train = frame[frame["season"].isin(train_seasons)]
    params = {**LGB_PARAMS, **(lgb_overrides or {}),
              "deterministic": True, "force_row_wise": True}
    model = lgb.LGBMClassifier(**params, random_state=seed)
    X = train[cols]
    mask = X.notna().any(axis=1) & train["is_hit"].notna()
    model.fit(X[mask], train["is_hit"][mask])

    slate = _slate_for_season(frame, screen_season)
    p_pa = model.predict_proba(slate[cols])[:, 1]
    est = slate["lineup_position"].map(PA_EST).fillna(4.0)
    slate = slate.assign(p_game=1 - (1 - p_pa) ** est)

    days = []
    for d, day in slate.groupby("date"):
        v = daily_ndcg(day, "p_game", k=10)
        if not np.isnan(v):
            top = day.sort_values("p_game", ascending=False)
            days.append({
                "date": str(d.date()), "ndcg": v,
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
