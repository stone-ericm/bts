"""Step 0 of SOTA #12 phase 2 — canonicalize the production realized-picks stream.

Per Codex agent-bus #154: produce a single audit-ready parquet that captures
every production pick (primary + double_down) with explicit lineage and
regime labels, so downstream calibration analyses cannot silently repool
contaminated data.

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

    # Attribution: PA frame is authoritative for resolved hits.
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

    column_order = [
        "source_file", "date", "run_time", "slot",
        "batter_id", "batter_name", "pitcher_id", "game_pk",
        "p_game_hit", "actual_hit", "result_status",
        "projected_lineup", "pick_file_result",
        "regime", "model_cutoff_label", "cutoff_iso", "attribution_source",
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
