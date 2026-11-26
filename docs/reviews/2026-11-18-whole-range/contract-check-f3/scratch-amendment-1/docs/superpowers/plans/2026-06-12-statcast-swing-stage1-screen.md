# Statcast Swing Campaign — Stage 1 Screen Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the 2024 screen — every variant of the five swing-feature families plus controls, evaluated as paired daily NDCG@10 deltas vs baseline across 3 seeds — producing the ranked screen report and a frozen-bundle proposal for Stage 2.

**Architecture:** A declarative arm registry (`bts/experiment/swing_screen.py`) maps arm name → feature columns + builder. Each screen run trains ONE LightGBM classifier (not the 12-blend — same form for every arm, paired-fair, ~25× cheaper; production-faithful walk-forward is Stage 2's job on the frozen bundle) on 2019–2023 PA rows with swing features attached, scores every 2024 day's actual starter slate, and writes a per-(arm, seed) result JSON. A resumable driver loops arms × seeds; a report script aggregates into the screen verdicts per the spec's rules (families die only on leakage/coverage/consistent negatives; sentinel MUST inflate; placebo/permuted must not).

**Tech Stack:** existing `bts.features.swing`, `bts.validate.slate_rank`, LightGBM via `bts.model.predict.train_model` params, pandas. Spec: `docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md`. Slate construction reuses the validated pattern from `scripts/replay_m3_serving_parity.py` (modal-starter proxy, lineup 1–9, `actual_hit`).

**Screen arm inventory (28 arms × 3 seeds = 84 runs, ~3–4 min each ≈ 4–5 box-hours):**
- `baseline` — production FEATURE_COLS only
- P (6): `p_miss_7g`, `p_miss_15g`, `p_miss_30g`, `p_miss_60g`, `p_miss_std_30g`, `p_high_share_30g`
- B (5): `b_miss_7g`, `b_miss_15g`, `b_miss_30g`, `b_miss_60g`, `b_miss_std_30g`
- T (2): `t_intercept_drift` (7g−60g contact depth), `t_miss_drift` (7g−60g miss quality)
- S (3): `s_swinglen_drift`, `s_attack_std_30g`, `s_attack_angle_30g`
- M (2): `m_high_alignment` (batter×pitcher high-whiff-share product), `m_high_mismatch` (|diff|)
- Omnibus (6): `omni_P`, `omni_B`, `omni_T`, `omni_S`, `omni_M`, `omni_ALL`
- Controls (3): `ctl_placebo` (boolean has-flags only), `ctl_permuted` (omni_ALL features shuffled within entity), `ctl_sentinel` (same-day leaky miss — MUST inflate)

**Seeds:** 42 (production), 101, 202 — via `BTS_LGBM_RANDOM_STATE`, `BTS_LGBM_DETERMINISTIC=1`.

**Declared scope cut (no silent caps):** the spec's full sweep axes include
exp-decay windows, p90 aggregations, and quality-adjusted-whiff combinations;
this screen ships a pre-registered 28-arm subset (4 windows × mean, std,
share, drift, and interaction forms). If a family looks alive but
window/aggregation-sensitive, a follow-up sweep within that family is a cheap
re-run of the same driver with added registry entries — extending the
registry BEFORE reading 2025 data keeps the selection rule honest. The
"raw-vs-compressed" axis is inherently covered: features feed the model
per-PA raw; aggregation happens only at game-level scoring.

---

### Task 1: Extend swing aggregates/rolling for T and S families

**Files:**
- Modify: `src/bts/features/swing.py`
- Test: `tests/test_swing_features.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_swing_features.py`:

```python
def test_daily_aggregates_include_intercept_and_attack_sq():
    bronze = _bronze([
        {"description": "foul", "miss_distance": None, "attack_angle": 10.0},
        {"description": "swinging_strike", "miss_distance": 2.0, "attack_angle": 14.0},
    ])
    bronze["intercept_ball_minus_batter_pos_y_inches"] = [30.0, 34.0]
    daily = daily_swing_aggregates(bronze, entity="batter")
    row = daily.iloc[0]
    assert row["intercept_y_sum"] == 64.0
    assert row["n_intercept_tracked"] == 2
    assert row["attack_angle_sumsq"] == 10.0**2 + 14.0**2


def test_rolling_includes_intercept_y_and_attack_std():
    daily = pd.DataFrame({
        "batter": [1, 1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
        "n_swings": [10, 10, 10],
        "n_swings_tracked": [10, 10, 10],
        "n_whiffs": [9, 9, 9],
        "n_whiffs_tracked": [9, 9, 9],
        "miss_sum": [27.0, 27.0, 27.0],
        "miss_sumsq": [85.0, 85.0, 85.0],
        "swing_len_sum": [70.0, 70.0, 70.0],
        "attack_angle_sum": [100.0, 120.0, 100.0],
        "attack_angle_sumsq": [1010.0, 1450.0, 1010.0],
        "n_whiff_high": [4, 4, 4],
        "n_whiff_low": [5, 5, 5],
        "intercept_y_sum": [300.0, 320.0, 340.0],
        "n_intercept_tracked": [10, 10, 10],
    })
    feats = rolling_swing_features(daily, entity="batter", windows=[2], min_whiffs=1)
    # day 3 window = days 1+2: intercept mean (300+320)/20 = 31.0
    assert abs(feats.iloc[2]["batter_intercept_y_2g"] - 31.0) < 1e-9
    # attack std from sums: mean=(100+120)/20=11, E[x^2]=(1010+1450)/20=123 -> var=2
    assert abs(feats.iloc[2]["batter_attack_std_2g"] - np.sqrt(2.0)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_swing_features.py -q`
Expected: 2 new FAIL (KeyError `intercept_y_sum` / `batter_intercept_y_2g`)

- [ ] **Step 3: Implement**

In `daily_swing_aggregates`, add before the groupby (after `df["_attack"] = ...`):

```python
    df["_intercept_y"] = pd.to_numeric(
        df.get("intercept_ball_minus_batter_pos_y_inches"), errors="coerce"
    )
```

and add to the `.agg(...)` call:

```python
        attack_angle_sumsq=("_attack", lambda s: float(np.nansum(np.square(s)))),
        intercept_y_sum=("_intercept_y", "sum"),
        n_intercept_tracked=("_intercept_y", "count"),
```

In `rolling_swing_features`, inside the `for w in windows:` loop add after the existing `_roll_sum` lines:

```python
        attack_sumsq = _roll_sum("attack_angle_sumsq", w)
        icpt_sum = _roll_sum("intercept_y_sum", w)
        icpt_n = _roll_sum("n_intercept_tracked", w)
```

and after the existing `out[...] =` assignments:

```python
        mean_attack = (attack / swings_tracked).where(swings_tracked >= min_whiffs)
        var_attack = (attack_sumsq / swings_tracked - mean_attack**2).where(swings_tracked >= min_whiffs)
        out[f"{entity}_attack_std_{w}g"] = np.sqrt(var_attack.clip(lower=0))
        out[f"{entity}_intercept_y_{w}g"] = (icpt_sum / icpt_n).where(icpt_n >= min_whiffs)
```

(Note `mean_attack` duplicates the existing `{entity}_attack_angle_{w}g` numerator — keep both assignments; the local is for the variance formula.)

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_swing_features.py -q`
Expected: all pass (10+)

- [ ] **Step 5: Commit**

```bash
git add src/bts/features/swing.py tests/test_swing_features.py
git commit -m "swing campaign S1: intercept-y + attack-std aggregates for T/S families"
```

---

### Task 2: Arm registry + feature builders (`swing_screen.py`)

**Files:**
- Create: `src/bts/experiment/swing_screen.py`
- Test: `tests/experiment/test_swing_screen.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment/test_swing_screen.py`:

```python
"""Tests for the Stage-1 screen arm registry (swing campaign)."""
import numpy as np
import pandas as pd

from bts.experiment.swing_screen import (
    ARMS,
    FAMILY_OF,
    build_arm_frame,
)


def _pa_frame():
    # PA rows already carrying rolling swing features (as attach would produce)
    return pd.DataFrame({
        "batter_id": [1, 2], "pitcher_id": [9, 9],
        "date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
        "season": [2024, 2024], "is_hit": [1, 0],
        "batter_miss_dist_7g": [2.0, 3.0],
        "batter_miss_dist_60g": [2.5, 2.5],
        "batter_intercept_y_7g": [31.0, 30.0],
        "batter_intercept_y_60g": [30.0, 30.0],
        "batter_swing_len_7g": [7.1, 7.0],
        "batter_swing_len_60g": [7.0, 7.0],
        "batter_whiff_high_share_30g": [0.6, 0.4],
        "pitcher_whiff_high_share_30g": [0.7, 0.7],
    })


def test_registry_arm_names_unique_and_families_mapped():
    assert len(ARMS) == len(set(ARMS))
    assert "baseline" in ARMS
    for arm in ARMS:
        assert arm in FAMILY_OF
    assert {"P", "B", "T", "S", "M", "omnibus", "control", "baseline"} >= set(FAMILY_OF.values())


def test_derived_drift_and_interaction_features():
    pa = _pa_frame()
    frame, cols = build_arm_frame("t_intercept_drift", pa)
    assert cols == ["t_intercept_drift"]
    assert abs(frame["t_intercept_drift"].iloc[0] - 1.0) < 1e-9  # 31-30

    frame, cols = build_arm_frame("m_high_alignment", pa)
    assert abs(frame["m_high_alignment"].iloc[0] - 0.42) < 1e-9  # 0.6*0.7


def test_baseline_arm_adds_no_columns():
    pa = _pa_frame()
    frame, cols = build_arm_frame("baseline", pa)
    assert cols == []


def test_permuted_control_preserves_values_but_breaks_dates():
    rng = np.random.default_rng(0)
    pa = pd.DataFrame({
        "batter_id": [1] * 6, "pitcher_id": [9] * 6,
        "date": pd.to_datetime([f"2024-05-{d:02d}" for d in range(1, 7)]),
        "season": 2024, "is_hit": 1,
        "batter_miss_dist_30g": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    # registry permutes within entity with a fixed seed
    frame, cols = build_arm_frame("ctl_permuted", pa, permute_seed=7)
    col = [c for c in cols if "batter_miss_dist_30g" in c]
    assert col, "permuted control must include permuted copies of omnibus features"
    vals = sorted(frame[col[0]].tolist())
    assert vals == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]      # marginal preserved
    assert frame[col[0]].tolist() != pa["batter_miss_dist_30g"].tolist()  # order broken


def test_placebo_control_is_flags_only():
    pa = _pa_frame()
    frame, cols = build_arm_frame("ctl_placebo", pa)
    assert all(c.startswith("has_") for c in cols)
    assert all(frame[c].dtype == bool for c in cols)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_swing_screen.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.experiment.swing_screen'`

- [ ] **Step 3: Implement**

Create `src/bts/experiment/swing_screen.py`:

```python
"""Stage-1 screen arm registry for the Statcast swing campaign.

Each arm = production FEATURE_COLS + the arm's swing columns. Base rolling
columns are produced by bts.features.swing (attach step in the driver);
derived columns (drifts, interactions, controls) are built here so every
definition is registry-local and testable.

Pre-registered inventory per the spec/plan: baseline, 18 single-variant arms
across families P/B/T/S/M, 6 omnibus arms, 3 controls. Family verdicts and
selection rules live in scripts/swing_screen_report.py.

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# base rolling columns referenced by arms (must exist after attach)
_P = "pitcher"
_B = "batter"

# arm -> list of (output_col, builder) where builder(pa) -> Series, or a
# plain string meaning "use this attached rolling column as-is".
_DERIVED = {
    "t_intercept_drift": lambda pa: pa[f"{_B}_intercept_y_7g"] - pa[f"{_B}_intercept_y_60g"],
    "t_miss_drift": lambda pa: pa[f"{_B}_miss_dist_7g"] - pa[f"{_B}_miss_dist_60g"],
    "s_swinglen_drift": lambda pa: pa[f"{_B}_swing_len_7g"] - pa[f"{_B}_swing_len_60g"],
    "m_high_alignment": lambda pa: pa[f"{_B}_whiff_high_share_30g"] * pa[f"{_P}_whiff_high_share_30g"],
    "m_high_mismatch": lambda pa: (pa[f"{_B}_whiff_high_share_30g"] - pa[f"{_P}_whiff_high_share_30g"]).abs(),
}

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


def _materialize(pa: pd.DataFrame, variant: str) -> tuple[pd.Series, str]:
    """Return (series, column_name) for a single variant name."""
    src = _SINGLE_VARIANTS[variant]
    if src in _DERIVED:
        return _DERIVED[src](pa), variant
    return pa[src], variant


def _omni_all_cols(pa: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = pa.copy()
    cols = []
    for variant in _SINGLE_VARIANTS:
        s, name = _materialize(pa, variant)
        frame[name] = s
        cols.append(name)
    return frame, cols


def build_arm_frame(
    arm: str, pa: pd.DataFrame, permute_seed: int = 13,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame, swing_cols) for an arm. frame = pa + the arm's columns;
    swing_cols are ADDED to production FEATURE_COLS by the runner."""
    if arm == "baseline":
        return pa.copy(), []

    if arm in _SINGLE_VARIANTS:
        frame = pa.copy()
        s, name = _materialize(pa, arm)
        frame[name] = s
        return frame, [name]

    if arm.startswith("omni_") and arm != "omni_ALL":
        fam = arm.split("_", 1)[1]
        frame = pa.copy()
        cols = []
        for variant in _FAMILY_MEMBERS[fam]:
            s, name = _materialize(pa, variant)
            frame[name] = s
            cols.append(name)
        return frame, cols

    if arm == "omni_ALL":
        return _omni_all_cols(pa)

    if arm == "ctl_placebo":
        frame, cols = _omni_all_cols(pa)
        flag_cols = []
        for c in cols:
            frame[f"has_{c}"] = frame[c].notna()
            flag_cols.append(f"has_{c}")
        return frame[list(pa.columns) + flag_cols], flag_cols

    if arm == "ctl_permuted":
        frame, cols = _omni_all_cols(pa)
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
```

- [ ] **Step 4: Run tests** (`mkdir -p tests/experiment` exists already — verify; create `tests/experiment/__init__.py` if missing)

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_swing_screen.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/swing_screen.py tests/experiment/test_swing_screen.py
git commit -m "swing campaign S1: declarative arm registry (28 arms, derived features, controls)"
```

---

### Task 3: Screen runner — train one arm, score 2024 slates

**Files:**
- Modify: `src/bts/experiment/swing_screen.py` (append runner functions)
- Test: `tests/experiment/test_swing_screen.py` (append)

- [ ] **Step 1: Write the failing test** (synthetic end-to-end, tiny LightGBM)

Append to `tests/experiment/test_swing_screen.py`:

```python
def test_run_screen_arm_end_to_end(tmp_path):
    from bts.experiment.swing_screen import run_screen_arm

    rng = np.random.default_rng(3)
    n_days, per_day = 30, 18
    rows = []
    for season, year_days in ((2023, n_days), (2024, n_days)):
        for i in range(year_days):
            for b in range(per_day):
                hit_p = 0.55 + 0.2 * (b % 2)        # feature-correlated outcome
                rows.append({
                    "batter_id": b, "pitcher_id": 100 + (i % 3),
                    "game_pk": season * 1000 + i, "lineup_position": (b % 9) + 1,
                    "is_home": b < 9,
                    "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                    "season": season,
                    "is_hit": int(rng.random() < hit_p),
                    "weather_temp": 70.0,
                    "batter_miss_dist_30g": 3.0 - 0.5 * (b % 2),
                })
    pa = pd.DataFrame(rows)

    res = run_screen_arm(
        arm="b_miss_30g", pa=pa, train_seasons=(2023,), screen_season=2024,
        seed=42, base_cols=["weather_temp"],
        lgb_overrides={"n_estimators": 10, "min_child_samples": 5},
        out_dir=tmp_path,
    )
    assert (tmp_path / "b_miss_30g_seed42.json").exists()
    assert res["arm"] == "b_miss_30g"
    assert res["n_days"] > 0
    assert "ndcg_mean" in res and "top1_hit" in res and "auc" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_swing_screen.py::test_run_screen_arm_end_to_end -q`
Expected: FAIL — `ImportError: cannot import name 'run_screen_arm'`

- [ ] **Step 3: Implement** (append to `src/bts/experiment/swing_screen.py`)

```python
# --- screen runner -----------------------------------------------------------

PA_EST = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0, 6: 3.9, 7: 3.8, 8: 3.7, 9: 3.6}


def _slate_for_season(pa: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (batter, game) for actual starters (lineup 1-9); outcome =
    any hit in that game. Mirrors the validated replay_m3_serving_parity
    construction (modal-starter detail lives upstream: slate rows carry the
    pa frame's pitcher_id)."""
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
    base_cols: list[str] | None = None,
    lgb_overrides: dict | None = None,
    out_dir=None,
    permute_seed: int = 13,
) -> dict:
    """Train one LightGBM on train_seasons, score screen_season slates,
    return + persist the per-arm metric payload."""
    import json as _json
    import lightgbm as lgb
    from pathlib import Path as _Path

    from bts.model.predict import LGB_PARAMS, FEATURE_COLS
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
    # rank AUC without sklearn (reuse health-check implementation)
    from bts.health.slate_auc import _rank_auc
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
```

- [ ] **Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_swing_screen.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/swing_screen.py tests/experiment/test_swing_screen.py
git commit -m "swing campaign S1: screen runner (train-once, score-2024, NDCG payload)"
```

---

### Task 4: Driver + report scripts

**Files:**
- Create: `scripts/swing_screen_driver.py`
- Create: `scripts/swing_screen_report.py`

- [ ] **Step 1: Write the driver**

Create `scripts/swing_screen_driver.py`:

```python
#!/usr/bin/env python3
"""Stage-1 screen driver: all arms x seeds, resumable. Run on bts-hetzner
overnight (nice'd; scheduler contention acceptable — runs are independent).

Usage:
  UV_CACHE_DIR=/tmp/uv-cache nice -n 15 .venv/bin/python \
      scripts/swing_screen_driver.py --out data/validation/swing_screen_2024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bts.experiment.swing_screen import ARMS, run_screen_arm  # noqa: E402
from bts.features.swing import (  # noqa: E402
    attach_swing_features, build_leaky_sentinel, daily_swing_aggregates,
    rolling_swing_features,
)
from bts.features.compute import compute_all_features  # noqa: E402

SEEDS = [42, 101, 202]
TRAIN_SEASONS = (2019, 2020, 2021, 2022, 2023)
SCREEN_SEASON = 2024


def build_pa_frame() -> pd.DataFrame:
    proc = Path("data/processed")
    pa = pd.concat(
        [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))],
        ignore_index=True,
    )
    pa = compute_all_features(pa)
    pa["date"] = pd.to_datetime(pa["date"])

    bronze = pd.concat(
        [pd.read_parquet(p) for p in sorted(proc.glob("swing_*.parquet"))],
        ignore_index=True,
    )
    feats = {}
    for entity in ("batter", "pitcher"):
        daily = daily_swing_aggregates(bronze, entity=entity)
        feats[entity] = rolling_swing_features(daily, entity=entity)
    pa = attach_swing_features(pa, batter_feats=feats["batter"], pitcher_feats=feats["pitcher"])
    # sentinel column for ctl_sentinel only (registry guards its use)
    daily_b = daily_swing_aggregates(bronze, entity="batter")
    pa = build_leaky_sentinel(pa, daily_b, entity="batter")
    return pa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    print("building PA + swing frame...", flush=True)
    t0 = time.time()
    pa = build_pa_frame()
    print(f"  frame ready: {len(pa)} rows in {time.time()-t0:.0f}s", flush=True)

    arms = args.arms or ARMS
    seeds = args.seeds or SEEDS
    total = len(arms) * len(seeds)
    done = 0
    for arm in arms:
        for seed in seeds:
            done += 1
            target = args.out / f"{arm}_seed{seed}.json"
            if target.exists():
                print(f"[{done}/{total}] skip {target.name}", flush=True)
                continue
            t1 = time.time()
            res = run_screen_arm(
                arm=arm, pa=pa, train_seasons=TRAIN_SEASONS,
                screen_season=SCREEN_SEASON, seed=seed, out_dir=args.out,
            )
            print(f"[{done}/{total}] {arm} seed={seed} ndcg={res['ndcg_mean']:.4f} "
                  f"auc={res['auc']:.4f} ({time.time()-t1:.0f}s)", flush=True)
    print("DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the report script**

Create `scripts/swing_screen_report.py`:

```python
#!/usr/bin/env python3
"""Aggregate screen results -> verdicts + frozen-bundle proposal (markdown).

Verdict rules (pre-registered, spec 2026-06-12):
- Controls: ctl_sentinel ndcg/auc MUST exceed baseline conspicuously (else
  the harness can't detect leakage -> STOP). ctl_placebo and ctl_permuted
  must be ~indistinguishable from baseline (else era-marker confounding).
- Variants ranked within family by paired daily NDCG delta vs baseline
  (same-seed pairing, mean across seeds); best variant per family proposed
  for the bundle.
- Families: alive unless coverage failure or consistently negative across
  ALL variants, seeds, and metrics (kill requires unanimity, not p-values).

Usage: .venv/bin/python scripts/swing_screen_report.py \
           --results data/validation/swing_screen_2024 --out docs/audit/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        out[(r["arm"], r["seed"])] = r
    return out


def paired_ndcg_delta(arm_res: dict, base_res: dict) -> float:
    base_by_date = {d["date"]: d["ndcg"] for d in base_res["per_day"]}
    ds = [d["ndcg"] - base_by_date[d["date"]]
          for d in arm_res["per_day"] if d["date"] in base_by_date]
    return float(np.mean(ds)) if ds else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/audit"))
    args = ap.parse_args()

    res = load_results(args.results)
    seeds = sorted({s for (_, s) in res})
    arms = sorted({a for (a, _) in res})

    rows = []
    for arm in arms:
        if arm == "baseline":
            continue
        deltas, top1s, aucs = [], [], []
        for s in seeds:
            if (arm, s) in res and ("baseline", s) in res:
                deltas.append(paired_ndcg_delta(res[(arm, s)], res[("baseline", s)]))
                top1s.append(res[(arm, s)]["top1_hit"] - res[("baseline", s)]["top1_hit"])
                aucs.append((res[(arm, s)]["auc"] or 0) - (res[("baseline", s)]["auc"] or 0))
        rows.append({
            "arm": arm, "family": res[(arm, seeds[0])]["family"],
            "ndcg_delta": float(np.mean(deltas)),
            "ndcg_delta_per_seed": [round(d, 5) for d in deltas],
            "top1_delta": float(np.mean(top1s)),
            "auc_delta": float(np.mean(aucs)),
        })

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    lines = [f"# Swing campaign Stage-1 screen report — {date.today()}", ""]
    lines.append("## Controls")
    for r in rows:
        if r["family"] == "control":
            lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f}, "
                         f"auc Δ {r['auc_delta']:+.5f}, per-seed {r['ndcg_delta_per_seed']}")
    lines.append("")
    lines.append("## Families (variants ranked by paired NDCG delta)")
    bundle = []
    for fam in ("P", "B", "T", "S", "M"):
        lines.append(f"### {fam}")
        fam_rows = sorted(by_family.get(fam, []), key=lambda r: -r["ndcg_delta"])
        for r in fam_rows:
            lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f} "
                         f"(seeds {r['ndcg_delta_per_seed']}), top1 Δ {r['top1_delta']:+.4f}, "
                         f"auc Δ {r['auc_delta']:+.5f}")
        if fam_rows:
            best = fam_rows[0]
            all_negative = all(
                d < 0 for r in fam_rows for d in r["ndcg_delta_per_seed"]
            ) and all(r["top1_delta"] < 0 and r["auc_delta"] < 0 for r in fam_rows)
            verdict = "DEAD (consistently negative everywhere)" if all_negative else "alive"
            lines.append(f"- **family verdict: {verdict}; best variant `{best['arm']}`**")
            if not all_negative:
                bundle.append(best["arm"])
        lines.append("")
    lines.append("## Omnibus arms")
    for r in sorted(by_family.get("omnibus", []), key=lambda r: -r["ndcg_delta"]):
        lines.append(f"- `{r['arm']}`: ndcg Δ {r['ndcg_delta']:+.5f}, "
                     f"top1 Δ {r['top1_delta']:+.4f}, auc Δ {r['auc_delta']:+.5f}")
    lines.append("")
    lines.append(f"## PROPOSED FROZEN BUNDLE (pending human review): {bundle}")
    out_path = args.out / f"{date.today()}-swing-screen-report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax check both**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/swing_screen_driver.py','scripts/swing_screen_report.py']]; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add scripts/swing_screen_driver.py scripts/swing_screen_report.py
git commit -m "swing campaign S1: screen driver (resumable, 84 runs) + report/verdict script"
```

---

### Task 5: Smoke locally, run on box, analyze

- [ ] **Step 1: Full not-slow suite; READ the output before any push**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 2: Local smoke of the driver on 2 arms × 1 seed** (local parquets suffice; swing parquets must first be scp'd from the box: `scp bts-hetzner:~/projects/bts/data/processed/swing_*.parquet data/processed/`)

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/swing_screen_driver.py --out /tmp/swing_smoke --arms baseline b_miss_30g --seeds 42`
Expected: 2 result JSONs; ndcg/auc plausible (auc ~0.58-0.60); note runtime per arm.

- [ ] **Step 3: Push, deploy, launch the full screen on the box overnight**

```bash
git push origin main && git push origin main:deploy
# after canary passes:
ssh bts-hetzner 'cd ~/projects/bts && git log --oneline -1'
ssh bts-hetzner 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache BTS_LGBM_DETERMINISTIC=1 \
  nohup nice -n 15 .venv/bin/python scripts/swing_screen_driver.py \
    --out data/validation/swing_screen_2024 > ~/logs/swing_screen.log 2>&1 & echo started'
```

Monitor: `ssh bts-hetzner 'tail -3 ~/logs/swing_screen.log'` — expect `[N/84] ...` lines, DRIVER DONE after ~4–6h.

- [ ] **Step 4: Generate the report; review controls FIRST**

```bash
ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python scripts/swing_screen_report.py \
  --results data/validation/swing_screen_2024 --out docs/audit/'
```

Gate (pre-registered): if `ctl_sentinel` does NOT show a conspicuous positive delta → the harness can't detect leakage → STOP, debug before reading any family result. If `ctl_placebo`/`ctl_permuted` show material positive deltas → era-marker confounding → STOP.

- [ ] **Step 5: Human review** — bring the report to Eric: family verdicts + proposed frozen bundle. The bundle freeze is HIS call (selection rule was pre-registered, but freeze-before-confirmation is the integrity hinge of the whole campaign). Commit the report doc + update memory after.
