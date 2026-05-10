"""Read-only clue audit for captured BTS leaderboard data.

This script intentionally does not write by default. It summarizes the parquet
store created by `bts leaderboard scrape` and compares tracked-user consensus
against our canonical realized picks when those artifacts are present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


VALID_RESULTS = {"hit", "not_hit"}


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def _load_user_picks(leaderboard_dir: Path) -> tuple[pd.DataFrame, int, int]:
    files = sorted((leaderboard_dir / "user_picks").glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    empty_files = 0
    for path in files:
        frame = pq.read_table(path).to_pandas()
        if frame.empty:
            empty_files += 1
            continue
        frame["username"] = path.stem
        parts.append(frame)
    if not parts:
        return pd.DataFrame(), len(files), empty_files
    raw = pd.concat(parts, ignore_index=True)
    raw["pick_date"] = pd.to_datetime(raw["pick_date"]).dt.date
    return raw, len(files), empty_files


def _dedupe_user_picks(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    return (
        raw.sort_values("captured_at")
        .drop_duplicates(["username", "pick_date", "pick_number"], keep="last")
        .copy()
    )


def _latest_active_users(leaderboard_dir: Path) -> tuple[set[str], str | None, int]:
    snaps = sorted((leaderboard_dir / "leaderboard_snapshots").glob("*.parquet"))
    if not snaps:
        return set(), None, 0
    latest = pq.read_table(snaps[-1]).to_pandas()
    active = latest[latest["tab"] == "active_streak"]
    return set(active["username"]), snaps[-1].name, len(snaps)


def _consensus_by_date_slot(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame(rows)
    for (pick_date, slot), group in frame.groupby(["pick_date", "pick_number"]):
        group = group[group["batter_id"].notna()]
        if group.empty:
            continue
        counts = (
            group.groupby(["batter_id", "batter_name"])
            .size()
            .sort_values(ascending=False)
        )
        (batter_id, batter_name), count = counts.index[0], int(counts.iloc[0])
        top = group[group["batter_id"] == batter_id]
        mode = top["result"].mode()
        result = mode.iloc[0] if not mode.empty else None
        rows.append(
            {
                "pick_date": pick_date,
                "pick_number": int(slot),
                "batter_id": int(batter_id),
                "batter_name": batter_name,
                "n": int(len(group)),
                "count": count,
                "share": count / len(group),
                "result": result,
            }
        )
    return pd.DataFrame(rows).sort_values(["pick_date", "pick_number"])


def _print_inventory(
    raw: pd.DataFrame,
    dedup: pd.DataFrame,
    file_count: int,
    empty_files: int,
    snapshot_count: int,
    latest_snapshot: str | None,
    active_users: set[str],
) -> None:
    valid = dedup[dedup["result"].isin(VALID_RESULTS)] if not dedup.empty else dedup
    print("=== INVENTORY ===")
    print(f"user_pick_files={file_count} empty_files={empty_files}")
    print(f"raw_rows={len(raw)} dedup_rows={len(dedup)} valid_rows={len(valid)}")
    if not dedup.empty:
        print(
            "users="
            f"{dedup['username'].nunique()} "
            f"pick_date_min={dedup['pick_date'].min()} "
            f"pick_date_max={dedup['pick_date'].max()} "
            f"n_dates={dedup['pick_date'].nunique()}"
        )
    print(
        f"snapshots={snapshot_count} latest_snapshot={latest_snapshot} "
        f"latest_active_top_users={len(active_users)}"
    )
    if not dedup.empty:
        print("\nresult_counts_dedup")
        print(dedup["result"].value_counts(dropna=False).to_string())
        print("\npick_number_counts_dedup")
        print(dedup["pick_number"].value_counts(dropna=False).sort_index().to_string())


def _print_hit_rates(dedup: pd.DataFrame, active_users: set[str]) -> None:
    print("\n=== HIT RATES ===")
    cohorts = [("all_tracked", dedup)]
    if active_users:
        cohorts.append(("latest_active_top100", dedup[dedup["username"].isin(active_users)]))
    for label, frame in cohorts:
        valid = frame[frame["result"].isin(VALID_RESULTS)]
        hit_rate = (valid["result"] == "hit").mean() if len(valid) else None
        print(
            f"{label}: picks={len(valid)} users={valid['username'].nunique()} "
            f"hit_rate={_pct(hit_rate)}"
        )
        if len(valid):
            by_slot = valid.groupby("pick_number")["result"].apply(
                lambda s: (s == "hit").mean()
            )
            print(by_slot.round(3).to_string())


def _print_double_pick_summary(dedup: pd.DataFrame) -> None:
    if dedup.empty:
        return
    user_day = dedup.pivot_table(
        index=["username", "pick_date"],
        columns="pick_number",
        values="result",
        aggfunc="first",
    ).reset_index()
    if 2 not in user_day.columns:
        return
    dd = user_day[user_day[2].notna()].copy()
    dd_valid = dd[dd[1].isin(VALID_RESULTS) & dd[2].isin(VALID_RESULTS)]
    both_hit = (
        ((dd_valid[1] == "hit") & (dd_valid[2] == "hit")).mean()
        if len(dd_valid)
        else None
    )
    print("\n=== DOUBLE PICK USER-DAYS ===")
    print(
        f"dd_user_days={len(dd)} dd_valid_days={len(dd_valid)} "
        f"share_user_days_with_pick2={_pct(len(dd) / len(user_day) if len(user_day) else None)}"
    )
    print(
        f"both_hit_rate={_pct(both_hit)} "
        f"pick1_hit_when_dd={_pct((dd_valid[1] == 'hit').mean() if len(dd_valid) else None)} "
        f"pick2_hit_when_dd={_pct((dd_valid[2] == 'hit').mean() if len(dd_valid) else None)}"
    )


def _print_top_batters(dedup: pd.DataFrame, limit: int) -> None:
    valid = dedup[dedup["result"].isin(VALID_RESULTS)]
    if valid.empty:
        return
    by_batter = (
        valid.groupby(["batter_id", "batter_name"])
        .agg(
            picks=("result", "size"),
            hit_rate=("result", lambda s: (s == "hit").mean()),
            users=("username", "nunique"),
        )
        .reset_index()
        .sort_values(["picks", "hit_rate"], ascending=[False, False])
        .head(limit)
    )
    print("\n=== TOP BATTERS BY TRACKED PICKS ===")
    for row in by_batter.itertuples(index=False):
        print(
            f"{int(row.batter_id)}\t{row.batter_name}\t"
            f"picks={int(row.picks)}\thit_rate={row.hit_rate:.3f}\tusers={int(row.users)}"
        )


def _print_consensus(label: str, consensus: pd.DataFrame, recent: int) -> None:
    print(f"\n=== {label} CONSENSUS BY DATE/SLOT ===")
    if consensus.empty:
        print("no consensus rows")
        return
    valid = consensus[consensus["result"].isin(VALID_RESULTS)]
    hit_rate = (valid["result"] == "hit").mean() if len(valid) else None
    print(
        f"units={len(consensus)} valid_units={len(valid)} "
        f"hit_rate_unweighted={_pct(hit_rate)} "
        f"avg_top_share={_pct(consensus['share'].mean())} "
        f"median_top_share={_pct(consensus['share'].median())} "
        f"max_top_share={_pct(consensus['share'].max())}"
    )
    print("recent")
    for row in consensus.tail(recent).itertuples(index=False):
        print(
            f"{row.pick_date} slot{row.pick_number} {row.batter_name} "
            f"n={row.n} share={row.share:.3f} result={row.result}"
        )


def _print_production_overlap(consensus: pd.DataFrame, validation_dir: Path) -> None:
    print("\n=== PRODUCTION VS TRACKED-CONSENSUS OVERLAP ===")
    files = sorted(validation_dir.glob("realized_picks_canonical_*.parquet"))
    if not files or consensus.empty:
        print("no overlap inputs")
        return
    ours = pq.read_table(files[-1]).to_pandas()
    ours["pick_date"] = pd.to_datetime(ours["date"]).dt.date
    slot_map = {"primary": 1, "double_down": 2, "dd": 2}
    ours["pick_number"] = ours["slot"].map(slot_map)
    ours = ours[ours["pick_number"].notna()].copy()
    ours["pick_number"] = ours["pick_number"].astype(int)
    merged = ours.merge(
        consensus,
        on=["pick_date", "pick_number"],
        how="inner",
        suffixes=("_ours", "_consensus"),
    )
    merged = merged[merged["result"].isin(VALID_RESULTS)]
    raw_overlap = len(merged)
    merged = merged[merged["actual_hit"].notna()].copy()
    print(
        f"canonical_file={files[-1].name} raw_overlap_units={raw_overlap} "
        f"resolved_overlap_units={len(merged)}"
    )
    if merged.empty:
        return
    merged["agree"] = (
        merged["batter_id_ours"].astype("Int64")
        == merged["batter_id_consensus"].astype("Int64")
    )
    disagreements = merged[~merged["agree"]]
    print(
        f"agreement_rate={_pct(merged['agree'].mean())} "
        f"our_hit_rate_overlap={_pct(merged['actual_hit'].astype(bool).mean())} "
        f"consensus_hit_rate_overlap={_pct((merged['result'] == 'hit').mean())}"
    )
    print(
        f"disagreements={len(disagreements)} "
        f"our_hit_rate_disagree={_pct(disagreements['actual_hit'].astype(bool).mean() if len(disagreements) else None)} "
        f"consensus_hit_rate_disagree={_pct((disagreements['result'] == 'hit').mean() if len(disagreements) else None)}"
    )
    print("recent_overlap")
    for row in merged.sort_values(["pick_date", "pick_number"]).tail(20).itertuples(index=False):
        print(
            f"{row.pick_date} slot{row.pick_number} "
            f"ours={row.batter_name_ours} p={row.p_game_hit:.3f} actual={row.actual_hit} "
            f"consensus={row.batter_name_consensus} share={row.share:.3f} "
            f"cons_result={row.result} agree={row.agree}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-dir", default="data/leaderboard")
    parser.add_argument("--validation-dir", default="data/validation")
    parser.add_argument("--recent", type=int, default=16)
    parser.add_argument("--top-batters", type=int, default=25)
    args = parser.parse_args()

    leaderboard_dir = Path(args.leaderboard_dir)
    raw, file_count, empty_files = _load_user_picks(leaderboard_dir)
    dedup = _dedupe_user_picks(raw)
    active_users, latest_snapshot, snapshot_count = _latest_active_users(leaderboard_dir)

    _print_inventory(
        raw,
        dedup,
        file_count,
        empty_files,
        snapshot_count,
        latest_snapshot,
        active_users,
    )
    _print_hit_rates(dedup, active_users)
    _print_double_pick_summary(dedup)
    _print_top_batters(dedup, args.top_batters)

    consensus = _consensus_by_date_slot(dedup)
    _print_consensus("ALL TRACKED", consensus, args.recent)

    if active_users:
        active_consensus = _consensus_by_date_slot(
            dedup[dedup["username"].isin(active_users)]
        )
        _print_consensus("LATEST ACTIVE TOP100", active_consensus, args.recent)

    _print_production_overlap(consensus, Path(args.validation_dir))


if __name__ == "__main__":
    main()
