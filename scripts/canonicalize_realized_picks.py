"""Step 0 of SOTA #12 phase 2 — canonicalize the production realized-picks stream.

Per Codex agent-bus #154: produce a single audit-ready parquet that captures
every production pick (primary + double_down) with explicit lineage and
regime labels, so downstream calibration analyses cannot silently repool
contaminated data.

α P0 (Codex bus #203, 2026-05-05) added per-game environment attribution: the
five `pick_*` / `is_park_driven` columns at the end of the schema. Environment
is joined by `pick.game_pk` against a per-game env table built from the PA
frame; this is independent of whether the picked batter actually had a PA in
the game and unambiguous on doubleheader days. `actual_hit` attribution
remains on (batter_id, date) per Codex's "do not broaden scope" guidance.

Output schema columns:
- source_file       — name of the pick JSON (e.g. "2026-04-22.json")
- date              — game date the pick was for (ISO date)
- run_time          — UTC ISO timestamp the prediction was generated
- slot              — "primary" or "double_down"
- batter_id         — int
- batter_name       — str
- pitcher_id        — int (may be missing on early picks)
- game_pk           — int
- p_game_hit        — float (the model's prediction)
- actual_hit        — bool/NaN, from PA frame ground truth (NaN if unresolved)
- result_status     — "resolved" | "pending" (resolved iff actual_hit is bool)
- projected_lineup  — bool, set True if the pick was generated from projected
                      lineup rather than confirmed
- regime            — "post_bpm" | "post_pooled_mdp_pre_bpm" | "pre_pooled_mdp"
- model_cutoff_label — str, label of the regime cutoff that placed this row
- cutoff_iso        — str, the run_time threshold (UTC ISO) for the regime
- attribution_source — "pa_frame" (only value currently emitted; field is
                       reserved for a future fallback path)
- pick_file_result    — the pick file's `result` field ("hit" | "miss" | None);
                        retained for audit purposes so future readers can
                        compare PA-frame attribution against the streak-level
                        result that the pick JSON carries (NOT used as the
                        attribution source for actual_hit; see Codex bus #156)
- pick_venue_id       — int (nullable Int64), MLB venue id of the pick's game
- pick_roof_type      — str (object), raw roof_type from the PA frame
                        ("Open" / "Dome" / "Retractable" / "Closed", typically
                        capitalized; None if game_pk not in PA frame)
- pick_weather_temp   — float, weather temperature reported in the PA frame
                        for the pick's game (NaN if game_pk not in PA frame)
- pick_is_indoor      — bool (nullable BooleanDtype), True iff
                        roof_type.lower() in {"dome","closed","retractable"}
                        (matches src/bts/features/compute.py:557-559); NA when
                        game_pk is not in the PA frame
- is_park_driven      — bool (nullable BooleanDtype), env-leverage proxy:
                        (pick_venue_id == COORS_VENUE_ID)
                        OR (pick_weather_temp >= 85.0 AND NOT pick_is_indoor)
                        NA when game_pk is not in the PA frame. NOT a feature-
                        attribution measure; do not use as deploy authorization.

Pending rows are included in the output (so the canonical view is complete)
but downstream analysis MUST exclude them from metric denominators.

When invoked with `--summary`, the script also prints the headline regime
metrics + fixed-bin reliability tables that the memo cites. The intent is
that the canonical artifact + memo + this script's `--summary` output are
mutually-reproducible from a single command.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RegimeCutoff:
    label: str
    cutoff_iso_utc: str  # run_time strings >= this go to this regime (or earlier)
    description: str


# Cutoffs ordered most-recent first. A pick belongs to the FIRST regime whose
# cutoff_iso_utc <= its run_time. Per Codex #154, use the FINAL commit in the
# pooled-MDP change group (e1ebde9 2026-04-15 23:21 ET) and the
# production-affecting bpm wiring commit (ee4190f 2026-04-30 12:27 ET).
CURRENT_MODEL_CUTOFF = RegimeCutoff(
    label="post_bpm",
    cutoff_iso_utc="2026-04-30T16:27:00+00:00",
    description=(
        "Post-bpm-wiring (commit ee4190f, 2026-04-30 12:27 ET / "
        "2026-04-30T16:27 UTC): batter_pitcher_shrunk_hr promoted to "
        "FEATURE_COLS AND wired in predict path. THE strict 'current model' "
        "regime."
    ),
)
ARCHITECTURE_REGIME_CUTOFF = RegimeCutoff(
    label="post_pooled_mdp_pre_bpm",
    cutoff_iso_utc="2026-04-16T03:21:00+00:00",
    description=(
        "Post-pooled-MDP, pre-bpm (commits 0528bfd → e1ebde9, "
        "2026-04-15 18:14 → 23:21 ET; using the FINAL commit "
        "2026-04-15T23:21 ET = 2026-04-16T03:21 UTC per Codex #154). "
        "Same MDP policy as current production but no bpm feature in "
        "the prediction path."
    ),
)
PRE_POOLED_MDP_CUTOFF = RegimeCutoff(
    label="pre_pooled_mdp",
    cutoff_iso_utc="1970-01-01T00:00:00+00:00",  # catch-all
    description="Pre-pooled-MDP. Different policy table than current production.",
)

ALL_REGIMES = [CURRENT_MODEL_CUTOFF, ARCHITECTURE_REGIME_CUTOFF, PRE_POOLED_MDP_CUTOFF]


# α P0 (Codex bus #203): per-game environment attribution.
# Coors Field is structurally the highest park-factor venue in MLB (mile-high
# air, no day-to-day humidor variance comparable to a coastal park). MLB API
# venue_id 19 = Coors Field; treated as a constant per Codex's #203 D answer
# ("do not add a mandatory raw-JSON dependency to canonicalization").
COORS_VENUE_ID = 19

# Production indoor convention from src/bts/features/compute.py:557-559: the
# raw roof_type values arrive capitalized ("Dome" / "Retractable" / "Open" /
# "Closed") but production lowercases before isin. Match that convention here.
INDOOR_ROOF_TYPES = frozenset({"dome", "closed", "retractable"})

# Statcast-era convention: ball-flight aerodynamics shift measurably above ~85F.
HOT_WEATHER_F = 85.0

# α P1 (Codex bus #215/#216): batter skill quartile attribution.
# Threshold for league-pool eligibility (also threshold below which the
# pick batter's quartile is NA). Justified by 2026-05-05 preflight: at the
# canonical artifact's snapshot, threshold=50 gives 30/30 support in the
# strategic-question target stratum (post_pooled_mdp_pre_bpm) AND preserves
# the n=1 DD park_driven cell (prior_pa=83); threshold=100 zeros that cell
# and halves the stratum to 16/30. 100 documented in the memo as the future-
# /career-data alternative once prior-season PA history is available.
MIN_PRIOR_PA = 50


def assign_regime(run_time_iso: str) -> RegimeCutoff:
    """First regime whose cutoff_iso_utc <= run_time_iso wins."""
    for regime in ALL_REGIMES:
        if run_time_iso >= regime.cutoff_iso_utc:
            return regime
    return PRE_POOLED_MDP_CUTOFF  # unreachable; here for completeness


def build_day_hit_lookup(pa_path: Path) -> dict[tuple[int, str], bool]:
    """(batter_id, date_iso_str) -> day_had_any_hit bool from PA frame.

    Mirrors `bts.health.realized_calibration._build_day_hit_lookup` but loads
    the entire frame (not lookback-bounded) so the canonical artifact captures
    every resolvable pick.
    """
    df = pd.read_parquet(pa_path, columns=["batter_id", "date", "is_hit"])
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    daily = df.groupby(["batter_id", "date"])["is_hit"].max().reset_index()
    return {(int(r.batter_id), r.date): bool(r.is_hit) for r in daily.itertuples(index=False)}


def build_game_env_lookup(pa_path: Path) -> dict[int, dict]:
    """game_pk -> {venue_id, roof_type, weather_temp, is_indoor} from PA frame.

    Env fields are game-level constants in the PA frame (verified on 2026 PA
    frame: 0 of 443 game_pks had multi-valued venue/roof/temp). Group-by-first
    is therefore safe.

    Each value is None/NaN-clean: pandas NaN, pd.NA, NaN-like string scalars,
    and Python None are all coerced to Python None so downstream NA-derivation
    is unambiguous. We use `pd.isna` (handles None / float NaN / pd.NA / NaT)
    rather than narrower isinstance(float) checks.
    """
    df = pd.read_parquet(
        pa_path,
        columns=["game_pk", "venue_id", "roof_type", "weather_temp"],
    )
    first = df.groupby("game_pk").first().reset_index()
    out: dict[int, dict] = {}
    for r in first.itertuples(index=False):
        rt = r.roof_type
        rt_clean: str | None
        is_indoor: bool | None
        if rt is None or pd.isna(rt):
            rt_clean = None
            is_indoor = None
        else:
            rt_clean = str(rt)
            is_indoor = rt_clean.lower() in INDOOR_ROOF_TYPES
        venue_id = None if pd.isna(r.venue_id) else int(r.venue_id)
        weather = None if pd.isna(r.weather_temp) else float(r.weather_temp)
        out[int(r.game_pk)] = {
            "venue_id": venue_id,
            "roof_type": rt_clean,
            "weather_temp": weather,
            "is_indoor": is_indoor,
        }
    return out


def build_skill_pool_lookup(
    pa_path: Path,
    unique_pick_dates,
    *,
    min_prior_pa: int = MIN_PRIOR_PA,
) -> dict:
    """For each unique pick.date, build (per-batter prior info, quartile bounds).

    Per Codex bus #215/#216:
    - As-of-pick-date: only PA rows with date < pick.date strictly
      (no same-day, no future leakage).
    - League pool: ALL PA-frame batters with prior_pa >= min_prior_pa
      as-of pick.date (NOT restricted to picked batters; that would
      blur skill with selection per #215 A).
    - Quartile bounds: q25/q50/q75 of the eligible pool's prior_hit_rate
      distribution, computed via pandas linear-interp quantile.

    Returns: {pick_date_str: {
        "per_batter": {batter_id: {"prior_pa": int, "prior_hit_rate": float}},
        "bounds": (q25, q50, q75) | (None, None, None),
    }}

    Per-batter dict includes ALL batters (eligible AND ineligible) so the
    pick row can populate prior_pa + prior_hit_rate for audit even when
    the quartile is NA (below threshold).
    """
    df = pd.read_parquet(pa_path, columns=["batter_id", "date", "is_hit"])
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    out: dict = {}
    for pick_date in unique_pick_dates:
        prior = df[df["date"] < pick_date]
        if prior.empty:
            out[pick_date] = {"per_batter": {}, "bounds": (None, None, None)}
            continue
        per_batter = prior.groupby("batter_id").agg(
            prior_pa=("is_hit", "size"),
            prior_hit_rate=("is_hit", "mean"),
        )
        eligible = per_batter[per_batter["prior_pa"] >= min_prior_pa]
        if len(eligible) > 0:
            q25 = float(eligible["prior_hit_rate"].quantile(0.25))
            q50 = float(eligible["prior_hit_rate"].quantile(0.50))
            q75 = float(eligible["prior_hit_rate"].quantile(0.75))
            bounds = (q25, q50, q75)
        else:
            bounds = (None, None, None)
        per_batter_dict = {
            int(idx): {
                "prior_pa": int(r["prior_pa"]),
                "prior_hit_rate": float(r["prior_hit_rate"]),
            }
            for idx, r in per_batter.iterrows()
        }
        out[pick_date] = {"per_batter": per_batter_dict, "bounds": bounds}
    return out


def assign_quartile(
    prior_hit_rate: float | None,
    q25: float | None,
    q50: float | None,
    q75: float | None,
) -> int | None:
    """Deterministic quartile assignment with <= comparisons (ties go low).

    Returns None if prior_hit_rate is None or any boundary is None. Otherwise:
        prior_hit_rate <= q25  -> 1
        prior_hit_rate <= q50  -> 2
        prior_hit_rate <= q75  -> 3
        else                   -> 4

    The lower-quartile-bias on ties is documented and tested explicitly. The
    alternative (pd.qcut / fractional ranking) hides tie behavior; <= keeps
    the rule transparent and the column auditable from primary data.
    """
    if prior_hit_rate is None or q25 is None or q50 is None or q75 is None:
        return None
    if prior_hit_rate <= q25:
        return 1
    if prior_hit_rate <= q50:
        return 2
    if prior_hit_rate <= q75:
        return 3
    return 4


def derive_is_park_driven(env: dict | None) -> bool | None:
    """Env-leverage proxy: Coors OR (hot AND outdoor). NA when env is missing.

    Decision tree (per Codex bus #205):
      env is None                                    -> None
      venue_id is None                               -> None  (partial env)
      venue_id == COORS_VENUE_ID                     -> True
      weather_temp is None                           -> None  (partial env)
      weather_temp < HOT_WEATHER_F                   -> False (rule says no)
      weather_temp >= HOT_WEATHER_F, is_indoor None  -> None  (partial env)
      weather_temp >= HOT_WEATHER_F, is_indoor False -> True
      weather_temp >= HOT_WEATHER_F, is_indoor True  -> False (rule says no)

    The contract: False means "observed environment, rule says no." None means
    "we cannot evaluate the rule because some required env field is missing."
    Future analysts can therefore distinguish "park-leverage absent" from
    "park-leverage status unknown" without re-reading the source data.

    NOT a feature-attribution measure. The strategic-question hypothesis is
    that low-skill park-driven picks at predicted 0.65-0.80 realize HIGHER
    than predicted; this boolean flags candidate environments for that cut,
    leaving the downstream memo to draw the inference.
    """
    if env is None:
        return None
    venue_id = env.get("venue_id")
    if venue_id is None:
        return None
    if venue_id == COORS_VENUE_ID:
        return True
    temp = env.get("weather_temp")
    if temp is None:
        return None
    if temp < HOT_WEATHER_F:
        return False
    indoor = env.get("is_indoor")
    if indoor is None:
        return None
    return indoor is False


def extract_picks(pick_json: dict, source_file: str) -> list[dict]:
    """Pull primary and double_down picks from one JSON into row dicts.

    Returns 0, 1, or 2 rows depending on which slots are populated.
    """
    rows: list[dict] = []
    pick_date = pick_json.get("date")
    run_time = pick_json.get("run_time")
    pick_file_result = pick_json.get("result")  # streak-level result, audit-only
    if not pick_date or not run_time:
        return []
    for slot, key in [("primary", "pick"), ("double_down", "double_down")]:
        body = pick_json.get(key)
        if not body or not isinstance(body, dict):
            continue
        p_game_hit = body.get("p_game_hit")
        if p_game_hit is None:
            continue
        rows.append({
            "source_file": source_file,
            "date": pick_date,
            "run_time": run_time,
            "slot": slot,
            "batter_id": body.get("batter_id"),
            "batter_name": body.get("batter_name"),
            "pitcher_id": body.get("pitcher_id"),
            "game_pk": body.get("game_pk"),
            "p_game_hit": float(p_game_hit),
            "projected_lineup": bool(body.get("projected_lineup", False)),
            "pick_file_result": pick_file_result,
        })
    return rows


def canonicalize(picks_dir: Path, pa_path: Path) -> pd.DataFrame:
    day_hit = build_day_hit_lookup(pa_path)
    game_env = build_game_env_lookup(pa_path)

    rows: list[dict] = []
    for f in sorted(picks_dir.glob("2026-*.json")):
        name = f.name
        if ".shadow." in name or "scheduler" in name or name.startswith("._"):
            continue
        try:
            body = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.extend(extract_picks(body, source_file=name))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Attribution: PA frame is authoritative for resolved hits. Per Codex bus
    # #203, this stays on (batter_id, date); env lookup below uses game_pk.
    actual_hit_list: list[bool | None] = []
    for r in df.itertuples(index=False):
        bid = r.batter_id
        if bid is None or pd.isna(bid):
            actual_hit_list.append(None)
            continue
        key = (int(bid), r.date)
        actual_hit_list.append(day_hit.get(key))
    df["actual_hit"] = actual_hit_list
    df["result_status"] = df["actual_hit"].apply(
        lambda x: "resolved" if isinstance(x, bool) else "pending"
    )
    df["attribution_source"] = "pa_frame"

    # Regime assignment.
    regime_objs = [assign_regime(rt) for rt in df["run_time"]]
    df["regime"] = [r.label for r in regime_objs]
    df["model_cutoff_label"] = [r.label for r in regime_objs]
    df["cutoff_iso"] = [r.cutoff_iso_utc for r in regime_objs]

    # α P0 env attachment: lookup by pick.game_pk against the per-game env
    # table. Missing game_pk in the PA frame -> NA across all 5 derived cols.
    venue_ids: list[int | None] = []
    roof_types: list[str | None] = []
    weather_temps: list[float | None] = []
    is_indoors: list[bool | None] = []
    park_drivens: list[bool | None] = []
    for r in df.itertuples(index=False):
        gp = r.game_pk
        env = None
        if gp is not None and not pd.isna(gp):
            env = game_env.get(int(gp))
        if env is None:
            venue_ids.append(None)
            roof_types.append(None)
            weather_temps.append(None)
            is_indoors.append(None)
            park_drivens.append(None)
        else:
            venue_ids.append(env["venue_id"])
            roof_types.append(env["roof_type"])
            weather_temps.append(env["weather_temp"])
            is_indoors.append(env["is_indoor"])
            park_drivens.append(derive_is_park_driven(env))

    df["pick_venue_id"] = pd.array(venue_ids, dtype="Int64")
    df["pick_roof_type"] = roof_types
    df["pick_weather_temp"] = weather_temps
    df["pick_is_indoor"] = pd.array(is_indoors, dtype="boolean")
    df["is_park_driven"] = pd.array(park_drivens, dtype="boolean")

    # α P1 (Codex bus #215/#216): season-to-date skill attribution. Quartile
    # snapshots built per unique pick date so the as-of-pick-date semantics
    # are preserved without scanning the PA frame N times for N picks.
    unique_dates = sorted(df["date"].unique().tolist())
    skill_pool = build_skill_pool_lookup(pa_path, unique_dates)
    prior_pa_list: list[int | None] = []
    prior_rate_list: list[float | None] = []
    quartile_list: list[int | None] = []
    for r in df.itertuples(index=False):
        snapshot = skill_pool.get(r.date)
        bid = r.batter_id
        # Pick batter must have a usable batter_id; otherwise skill is NA.
        if bid is None or pd.isna(bid) or snapshot is None:
            prior_pa_list.append(None)
            prior_rate_list.append(None)
            quartile_list.append(None)
            continue
        info = snapshot["per_batter"].get(int(bid))
        if info is None:
            # No prior PAs at all on this pick.date.
            prior_pa_list.append(0)
            prior_rate_list.append(None)
            quartile_list.append(None)
            continue
        prior_pa_list.append(info["prior_pa"])
        prior_rate_list.append(info["prior_hit_rate"])
        if info["prior_pa"] >= MIN_PRIOR_PA:
            q25, q50, q75 = snapshot["bounds"]
            quartile_list.append(
                assign_quartile(info["prior_hit_rate"], q25, q50, q75)
            )
        else:
            quartile_list.append(None)

    df["batter_skill_prior_pa"] = pd.array(prior_pa_list, dtype="Int64")
    df["batter_skill_prior_hit_rate"] = pd.array(prior_rate_list, dtype="Float64")
    df["batter_skill_quartile"] = pd.array(quartile_list, dtype="Int64")

    column_order = [
        "source_file", "date", "run_time", "slot",
        "batter_id", "batter_name", "pitcher_id", "game_pk",
        "p_game_hit", "actual_hit", "result_status",
        "projected_lineup", "pick_file_result",
        "regime", "model_cutoff_label", "cutoff_iso", "attribution_source",
        "pick_venue_id", "pick_roof_type", "pick_weather_temp",
        "pick_is_indoor", "is_park_driven",
        "batter_skill_prior_pa", "batter_skill_prior_hit_rate",
        "batter_skill_quartile",
    ]
    df = df[column_order]
    return df


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy import stats
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * (((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5) / den
    return (center - half, center + half)


def print_summary(df: pd.DataFrame) -> None:
    """Print the headline metrics + fixed-bin reliability tables cited in the memo."""
    import numpy as np

    resolved = df[df["result_status"] == "resolved"].copy()
    print()
    print("=" * 78)
    print("HEADLINE METRICS (resolved-only)")
    print("=" * 78)
    print(f"{'regime':<28} {'n':>3} {'hits':>4} {'rate':>6} {'mean_p':>7} {'gap':>7} {'Brier':>7} {'BSS':>7}")
    print("-" * 78)
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        p = sub["p_game_hit"].astype(float).values
        y = sub["actual_hit"].astype(int).values
        n = len(sub); n_hit = int(y.sum()); rate = n_hit / n; mp = float(p.mean())
        gap = mp - rate
        brier = float(((p - y) ** 2).mean())
        ref = rate * (1 - rate)
        skill = (1 - brier / ref) if ref > 0 else float("nan")
        gap_s = f"{gap:+.3f}"
        skill_s = f"{skill:+.3f}" if not np.isnan(skill) else "  nan"
        print(f"{regime:<28} {n:>3} {n_hit:>4} {rate:>6.3f} {mp:>7.3f} {gap_s:>7} {brier:>7.4f} {skill_s:>7}")

    print()
    print("=" * 78)
    print("FIXED-BIN RELIABILITY (Wilson 95%)")
    print("=" * 78)
    bins = [(0.55, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.95)]
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        p = sub["p_game_hit"].astype(float).values
        y = sub["actual_hit"].astype(int).values
        print(f"\n=== {regime} (n={len(sub)}) ===")
        print(f"{'bin':<14} {'n':>4} {'mean_p':>7} {'mean_y':>7} {'gap':>8} {'wilson_lo':>10} {'wilson_hi':>10}")
        print("-" * 65)
        for lo, hi in bins:
            mask = (p >= lo) & (p < hi)
            n_b = int(mask.sum())
            if n_b == 0:
                print(f"[{lo:.2f},{hi:.2f})    {n_b:>4} {'-':>7} {'-':>7} {'-':>8} {'-':>10} {'-':>10}")
                continue
            mp = float(p[mask].mean())
            my = float(y[mask].mean())
            k = int(y[mask].sum())
            lo_w, hi_w = _wilson_ci(k, n_b)
            gap_str = f"{(mp - my):+.3f}"
            print(f"[{lo:.2f},{hi:.2f})    {n_b:>4} {mp:>7.3f} {my:>7.3f} {gap_str:>8} {lo_w:>10.3f} {hi_w:>10.3f}")

    print()
    print("=" * 78)
    print("SLOT BREAKDOWN BY REGIME (resolved-only)")
    print("=" * 78)
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        print(f"\n{regime}:")
        for slot in ["primary", "double_down"]:
            s = sub[sub["slot"] == slot]
            if len(s) == 0:
                continue
            n = len(s)
            n_hit = int(s["actual_hit"].sum())
            rate = n_hit / n
            mp = float(s["p_game_hit"].mean())
            gap = mp - rate
            print(f"  {slot:<14} n={n:>3} hits={n_hit}/{n} ({rate:.3f}) mean_p={mp:.3f} gap={gap:+.3f}")

    # α P0 (Codex bus #203): env-cut tables. Cells with NA is_park_driven (no
    # game_pk env match) are excluded from the cut so cell counts denote
    # only rows where the rule could be evaluated.
    print()
    print("=" * 78)
    print("ENV-CUT A — regime × is_park_driven (resolved-only, env-attributed)")
    print("=" * 78)
    print(f"{'regime':<28} {'env':<18} {'n':>3} {'hits':>4} {'rate':>6} {'mean_p':>7} {'gap':>7} {'wilson_lo':>10} {'wilson_hi':>10}")
    print("-" * 100)
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        for pd_val, label in [(True, "park_driven"), (False, "not_park_driven")]:
            s = sub[sub["is_park_driven"] == pd_val]
            n = len(s)
            if n == 0:
                print(f"{regime:<28} {label:<18} {n:>3} {'-':>4} {'-':>6} {'-':>7} {'-':>7} {'-':>10} {'-':>10}")
                continue
            n_hit = int(s["actual_hit"].sum())
            rate = n_hit / n
            mp = float(s["p_game_hit"].mean())
            gap = mp - rate
            lo_w, hi_w = _wilson_ci(n_hit, n)
            print(f"{regime:<28} {label:<18} {n:>3} {n_hit:>4} {rate:>6.3f} {mp:>7.3f} {gap:>+7.3f} {lo_w:>10.3f} {hi_w:>10.3f}")
        # NA count for transparency
        na_n = int(sub["is_park_driven"].isna().sum())
        if na_n > 0:
            print(f"{regime:<28} {'(env-NA, excluded)':<18} {na_n:>3}")

    print()
    print("=" * 78)
    print("ENV-CUT B — regime × slot × is_park_driven (resolved-only)")
    print("Cells with n<5 are exploratory; do not interpret directionally.")
    print("=" * 78)
    print(f"{'regime':<28} {'slot':<13} {'env':<18} {'n':>3} {'hits':>4} {'rate':>6} {'mean_p':>7} {'gap':>7} {'note':<14}")
    print("-" * 110)
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        for slot in ["primary", "double_down"]:
            for pd_val, label in [(True, "park_driven"), (False, "not_park_driven")]:
                s = sub[(sub["slot"] == slot) & (sub["is_park_driven"] == pd_val)]
                n = len(s)
                note = "exploratory" if 0 < n < 5 else ""
                if n == 0:
                    print(f"{regime:<28} {slot:<13} {label:<18} {n:>3} {'-':>4} {'-':>6} {'-':>7} {'-':>7} {note:<14}")
                    continue
                n_hit = int(s["actual_hit"].sum())
                rate = n_hit / n
                mp = float(s["p_game_hit"].mean())
                gap = mp - rate
                print(f"{regime:<28} {slot:<13} {label:<18} {n:>3} {n_hit:>4} {rate:>6.3f} {mp:>7.3f} {gap:>+7.3f} {note:<14}")

    # α P1 (Codex bus #215/#216): skill cut. Reports all four quartiles plus an
    # explicit (skill-NA, excluded) line analogous to env-NA so support loss
    # is visible. n<5 cells marked exploratory.
    print()
    print("=" * 78)
    print("ENV-CUT C — regime × slot × is_park_driven × skill_quartile")
    print("(resolved-only. n<5 exploratory; interpret only Q1 vs Q4 contrasts where n>=5.)")
    print("=" * 78)
    print(f"{'regime':<28} {'slot':<13} {'env':<18} {'Q':<3} {'n':>3} {'hits':>4} {'rate':>6} {'mean_p':>7} {'gap':>7} {'note':<14}")
    print("-" * 115)
    for regime in ["post_bpm", "post_pooled_mdp_pre_bpm", "pre_pooled_mdp"]:
        sub = resolved[resolved["regime"] == regime]
        if len(sub) == 0:
            continue
        for slot in ["primary", "double_down"]:
            for pd_val, label in [(True, "park_driven"), (False, "not_park_driven")]:
                cell = sub[(sub["slot"] == slot) & (sub["is_park_driven"] == pd_val)]
                if len(cell) == 0:
                    continue
                for q in [1, 2, 3, 4]:
                    s = cell[cell["batter_skill_quartile"] == q]
                    n = len(s)
                    note = "exploratory" if 0 < n < 5 else ""
                    if n == 0:
                        print(f"{regime:<28} {slot:<13} {label:<18} {f'Q{q}':<3} {n:>3} {'-':>4} {'-':>6} {'-':>7} {'-':>7} {note:<14}")
                        continue
                    n_hit = int(s["actual_hit"].sum())
                    rate = n_hit / n
                    mp = float(s["p_game_hit"].mean())
                    gap = mp - rate
                    print(f"{regime:<28} {slot:<13} {label:<18} {f'Q{q}':<3} {n:>3} {n_hit:>4} {rate:>6.3f} {mp:>7.3f} {gap:>+7.3f} {note:<14}")
                # Explicit skill-NA line for support transparency (Codex #215/#216).
                na_n = int(cell["batter_skill_quartile"].isna().sum())
                if na_n > 0:
                    print(f"{regime:<28} {slot:<13} {label:<18} {'NA':<3} {na_n:>3} {'(skill-NA, excluded from Q cells)':<60}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks-dir", type=Path, required=True, help="dir containing 2026-*.json picks")
    p.add_argument("--pa-path", type=Path, required=True, help="data/processed/pa_2026.parquet")
    p.add_argument("--output", type=Path, required=True, help="canonical parquet output path")
    p.add_argument("--summary", action="store_true",
                   help="after writing the parquet, print headline metrics + fixed-bin "
                        "reliability + slot breakdown (the tables cited in the memo)")
    args = p.parse_args()

    if not args.picks_dir.exists():
        print(f"picks-dir does not exist: {args.picks_dir}", file=sys.stderr)
        return 2
    if not args.pa_path.exists():
        print(f"pa-path does not exist: {args.pa_path}", file=sys.stderr)
        return 2

    df = canonicalize(args.picks_dir, args.pa_path)
    if df.empty:
        print("no picks found", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"wrote {len(df)} rows to {args.output}")
    print(f"  resolved: {(df['result_status'] == 'resolved').sum()}")
    print(f"  pending:  {(df['result_status'] == 'pending').sum()}")
    print()
    print("regime breakdown (resolved only):")
    resolved = df[df["result_status"] == "resolved"]
    for regime, group in resolved.groupby("regime"):
        n = len(group)
        n_hit = int(group["actual_hit"].sum())
        slot_breakdown = group.groupby("slot").size().to_dict()
        print(
            f"  {regime:<28} n={n:>3} hits={n_hit:>3}/{n} ({n_hit/n:.1%}) "
            f"slots={slot_breakdown}"
        )
    print()
    print("snapshot timestamp (UTC):", datetime.now(timezone.utc).isoformat())

    if args.summary:
        print_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
