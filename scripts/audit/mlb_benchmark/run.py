"""W2.3 MLB forecast benchmark driver (design docs/superpowers/specs/2026-10-04-mlb-forecast-benchmark-design.md rev 3,
FROZEN). Reads the accepted W1.2 run's table (served D, eligibility pools, selection state, outcomes) and the BTS
static captures; links MLB's probabilityStarter to the served slate by inference only; scores both forecasters on
the frozen shared pool. Refuses to run before exposure row X-23 is in HEAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit.benchmark_bridge import core as bridge
from scripts.audit.mlb_benchmark import core, metrics as m
from scripts.audit.season_ledger.ids import load_json_bytes, stamp_to_utc

REPO = Path(__file__).resolve().parents[3]
X23_COMMIT = "b68098d"               # the register commit that publishes X-23
WINDOW_START = "2026-07-04"
COHORTS = ("selection_consistent", "inconsistent", "no_selection")
POOLS = ("pool_verified", "pool_surrogate", "pool_all")
FEEDS = {"most_selected_players": "mostSelectedPlayers", "rounds": "rounds", "players": "players", "units": "units"}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def x23_gate() -> str:
    head = git("rev-parse", "HEAD")
    if X23_COMMIT is None:
        raise SystemExit("X-23 gate: X23_COMMIT is unset (publish X-23 first)")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", X23_COMMIT, head]).returncode != 0:
        raise SystemExit(f"X-23 gate: {X23_COMMIT} is not an ancestor of HEAD {head[:7]}")
    if "| X-23 |" not in (REPO / "docs/audit/2026-09-22-exposure-register.md").read_text():
        raise SystemExit("X-23 gate: the register in this checkout has no X-23 row")
    return head


def latest_at(sheets: dict, stamp) -> tuple:
    """(stamp, items) of the latest stored sheet at or before ``stamp`` from {stamp: items}; (None, None) if none."""
    cut = pd.Timestamp(stamp)
    eligible = [s for s in sheets if pd.Timestamp(s) <= cut]
    if not eligible:
        return None, None
    best = max(eligible, key=pd.Timestamp)
    return best, sheets[best]


def _valid_p(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0


def prepare_date(date: str, written_at: pd.Timestamp, slate: pd.DataFrame, msp: list[tuple], sheet_at, sched_pks: set):
    """One date's as-of join (gate 1). The forecast sheet is the latest stored whole sheet at or before written_at;
    its run-start stamp is the boundary for the rounds/players/units lookups (A-E2); the round's units must cover
    the MLB schedule for a unique-game inference. Returns (joined rows, coverage)."""
    cov: dict = {"date": date, "written_at": str(written_at), "excluded": None}
    stamp, _ = core.latest_sheet(msp, {}, date, written_at)
    cov["forecast_stamp"] = stamp
    if stamp is None:
        cov["excluded"] = "no_forecast_sheet"
        return pd.DataFrame(), cov
    r_stamp, rounds = sheet_at("rounds", stamp)
    round_id, why = core.round_for_date(rounds or [], date)
    cov.update(rounds_stamp=r_stamp, round_id=round_id)
    if round_id is None:
        cov["excluded"] = why
        return pd.DataFrame(), cov
    sheet_rows = [r for r in dict(msp)[stamp] if r.get("roundId") == round_id]
    cov["invalid_probability"] = sum(1 for r in sheet_rows if not _valid_p(r.get("probabilityStarter")))
    if not sheet_rows:
        cov["excluded"] = "no_target_round_rows"
        return pd.DataFrame(), cov
    p_stamp, players = sheet_at("players", stamp)
    feed, squad, conflicts = core.player_lookup(players or [])
    cov.update(players_stamp=p_stamp, player_conflicts=len(conflicts))
    valid_msp = [(s, [r for r in rows if _valid_p(r.get("probabilityStarter"))]) for s, rows in msp]
    fc = core.forecasts_asof(valid_msp, {round_id: date}, feed, date, written_at)
    u_stamp, units = sheet_at("units", stamp)
    complete = core.units_complete(units or [], round_id, sched_pks)
    cov.update(units_stamp=u_stamp, units_complete=complete, n_forecasts=len(fc))
    games = core.games_by_batter(fc, squad, units or [], round_id) if complete else {b: None for b in fc}
    joined, link_cov = core.join_to_slate(slate, fc, games)
    cov.update(link_cov)
    joined["mlb_n_sel"] = [fc[b]["n_sel"] if s == "inferred_unique_game" else np.nan
                           for b, s in zip(joined["batter_id"], joined["link_status"])]
    joined["mlb_unchanged_since"] = [core.unchanged_since(valid_msp, stamp, round_id, fc[b]["player_id"])
                                     if s == "inferred_unique_game" else None
                                     for b, s in zip(joined["batter_id"], joined["link_status"])]
    joined["forecast_stamp"] = stamp
    return joined, cov


def _sheets(directory: Path) -> dict:
    """{stamp: path} for one feed's captures (plain and gzipped)."""
    return {stamp_to_utc(p.name): p for p in bridge.capture_files(directory) if stamp_to_utc(p.name)}


def _load(path: Path, key: str):
    doc = load_json_bytes(path.read_bytes())
    return doc.get(key, []) if isinstance(doc, dict) else []


def _auc(g: pd.DataFrame, score: str) -> float:
    a = bridge._rank_auc(g.loc[g["y"] == 1, score], g.loc[g["y"] == 0, score])
    return float("nan") if a is None else float(a)


def _nanmean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else float("nan")


def score_stratum(pool: pd.DataFrame, t: str, n_resamples: int) -> dict:
    """Equal-date proper scores, residual and AUC for both arms under target ``t`` (gate 5), with joint whole-date
    intervals. Each statistic is a mean over date copies of a per-date value, so the per-date summary is resampled
    (``m.summary_bootstrap``, identical to the copy-block bootstrap). Single-class dates are omitted from AUC."""
    tdf = m.target(pool, t)
    if tdf.empty:
        return {"available": False, "reason": "empty"}
    rows = []
    for d, g in tdf.groupby("date", sort=True):
        r = {"date": d, "n": len(g)}
        for arm in ("ours", "mlb"):
            r.update({f"{arm}_brier": m.brier_rows(g[arm], g["y"]), f"{arm}_log_loss": m.log_loss_rows(g[arm], g["y"]),
                      f"{arm}_residual": m.residual_rows(g[arm], g["y"]), f"{arm}_auc": _auc(g, arm)})
        rows.append(r)
    per_date = pd.DataFrame(rows).set_index("date")
    stats = {}
    for k in ("brier", "log_loss", "residual", "auc"):
        for arm in ("ours", "mlb"):
            stats[f"{arm}_{k}"] = lambda s, c=f"{arm}_{k}": _nanmean(s[c])
        stats[f"diff_{k}"] = lambda s, k=k: _nanmean(s[f"mlb_{k}"]) - _nanmean(s[f"ours_{k}"])
    return {"available": True, "n_dates": int(len(per_date)), "n_rows": int(len(tdf)),
            "auc_dates_omitted": {a: int(per_date[f"{a}_auc"].isna().sum()) for a in ("ours", "mlb")},
            "estimates": {k: f(per_date) for k, f in stats.items()},
            "intervals": m.summary_bootstrap(per_date, stats, n_resamples=n_resamples)}


def score_top1(pool: pd.DataFrame, t: str, n_resamples: int) -> dict:
    """Shared-set top-1 (gate 5): winners ranked on the frozen pool before labels; paired rates on dates where both
    winners are known under ``t``; disagreement dates by differing (batter_id, game_pk)."""
    point = pool.assign(_block=pool["date"])
    pair, dis = m.paired_top1(point, "ours", "mlb", t), m.disagreement(point, "ours", "mlb", t)
    wa, wb = m.top1(point, "ours"), m.top1(point, "mlb")
    per = wa[["date", "batter_id", "game_pk", "outcome"]].merge(
        wb[["date", "batter_id", "game_pk", "outcome"]], on="date", suffixes=("_o", "_m")).set_index("date")
    known = m.TARGET_KNOWN[t]
    per["both"] = per["outcome_o"].isin(known) & per["outcome_m"].isin(known)
    per["hit_o"], per["hit_m"] = (per["outcome_o"] == "hit"), (per["outcome_m"] == "hit")
    per["disagree"] = (per["batter_id_o"] != per["batter_id_m"]) | (per["game_pk_o"] != per["game_pk_m"])

    def diff(s, only_disagree=False):
        x = s[s["both"] & (s["disagree"] if only_disagree else True)]
        return float(x["hit_m"].mean() - x["hit_o"].mean()) if len(x) else float("nan")
    boot = m.summary_bootstrap(per, {"top1_diff": diff, "disagreement_diff": lambda s: diff(s, True)},
                               n_resamples=n_resamples)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("dates_common", "dates")}
    return {"paired_top1": strip(pair), "disagreement": {**strip(dis), "n_dates": len(dis["dates"])},
            "intervals": boot}


def score_encompassing(pool: pd.DataFrame, n_resamples: int) -> dict:
    tdf = m.target(pool, "T2")
    point = m.encompassing(tdf.assign(_block=tdf["date"]))
    if not point["available"]:
        return {"point": point}
    keys = ("coef_mlb", "delta_log_loss")
    stats = {k: (lambda x, k=k: (lambda r: r[k] if r["available"] else float("nan"))(m.encompassing(x)))
             for k in keys}
    return {"point": point, "intervals": m.block_bootstrap_many(tdf, stats, n_resamples=n_resamples)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w12-run", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--schedules", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    args = ap.parse_args(argv)
    head = x23_gate()
    from scripts.audit.season_ledger.sources.static import parse_schedule

    w12 = args.w12_run.expanduser().resolve()
    table = pd.read_parquet(w12 / "table.parquet")
    summary = json.loads((w12 / "summary.json").read_text())
    written = {d["date"]: pd.Timestamp(d["written_at"]) for d in summary["day_meta"]}
    snaps = args.data_root.expanduser().resolve() / "leaderboard" / "static_snapshots"
    paths = {f: _sheets(snaps / f) for f in FEEDS}
    msp = sorted(((s, _load(p, FEEDS["most_selected_players"])) for s, p in paths["most_selected_players"].items()),
                 key=lambda c: pd.Timestamp(c[0]))
    cache: dict = {}
    used: dict = {f: set() for f in FEEDS}

    def sheet_at(feed, stamp):
        s, p = latest_at(paths[feed], stamp)
        if s is None:
            return None, None
        used[feed].add(p.name)
        if p not in cache:
            cache[p] = _load(p, FEEDS[feed])
        return s, cache[p]

    stamp_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp_now}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"run dir {run_dir}; code {head[:7]}; W1.2 run {w12.name}")

    parts, covs, sched_used = [], [], {}
    for d in sorted(written):
        if d < WINDOW_START:
            covs.append({"date": d, "excluded": "before_capture_window"})
            continue
        sp = args.schedules / f"{d}.json"
        sched = set()
        if sp.exists():
            parsed = parse_schedule(f"schedules/{d}.json", sp.read_bytes())
            sched = set() if parsed.quarantined else {int(r["game_pk"]) for r in parsed.rows}
            sched_used[sp.name] = hashlib.sha256(sp.read_bytes()).hexdigest()
        slate = table[table["date"] == d]
        joined, cov = prepare_date(d, written[d], slate, msp, sheet_at, sched)
        covs.append(cov)
        if not joined.empty:
            parts.append(joined)
    joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    log(f"joined rows {len(joined):,}; dates with a forecast join {len(parts)}")

    ok = joined["link_status"] == "inferred_unique_game"
    valid = joined["D"].apply(_valid_p) & joined["mlb_p"].apply(_valid_p)
    joined = joined.assign(ours=joined["D"], mlb=joined["mlb_p"], in_pool_base=ok & valid)
    rec = joined[joined["in_pool_base"]]
    recency = {
        "sheet_to_written_at_minutes": (
            (pd.to_datetime(rec["written_at"] if "written_at" in rec else rec["date"].map(written))
             - pd.to_datetime(rec["forecast_stamp"], utc=True)).dt.total_seconds() / 60).describe().to_dict()
        if len(rec) else {},
        "observed_unchanged_minutes": ((pd.to_datetime(rec["forecast_stamp"], utc=True)
                                        - pd.to_datetime(rec["mlb_unchanged_since"], utc=True))
                                       .dt.total_seconds() / 60).describe().to_dict() if len(rec) else {},
        "n_sel_quantiles": rec["mlb_n_sel"].quantile([0, .25, .5, .75, 1]).to_dict() if len(rec) else {},
    }
    results = {"schema": "w23_mlb_benchmark_v1", "code": head, "x23_commit": X23_COMMIT, "w12_run": str(w12),
               "coverage": {"per_date": covs, "invalid_shared_scores": int((ok & ~valid).sum()),
                            "link_status": joined["link_status"].value_counts().to_dict()},
               "recency": recency, "strata": {}}
    for cohort in COHORTS:
        for pool_col in POOLS:
            pool = joined[joined["in_pool_base"] & (joined["sel_state"] == cohort)
                          & joined[pool_col].fillna(False).astype(bool)]
            key = f"{cohort}/{pool_col}"
            if pool.empty:
                results["strata"][key] = {"available": False, "reason": "empty"}
                continue
            log(f"scoring {key}: {len(pool):,} rows, {pool['date'].nunique()} dates")
            results["strata"][key] = {
                "T1": score_stratum(pool, "T1", args.n_resamples), "T2": score_stratum(pool, "T2", args.n_resamples),
                "top1_T1": score_top1(pool, "T1", args.n_resamples), "top1_T2": score_top1(pool, "T2", args.n_resamples),
                "encompassing_T2": score_encompassing(pool, args.n_resamples)}
    joined.to_parquet(run_dir / "joined.parquet", index=False)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"w12_table": sha(w12 / "table.parquet"), "w12_summary": sha(w12 / "summary.json"),
                "msp_files": {p.name: sha(p) for p in paths["most_selected_players"].values()},
                "lookup_files_used": {f: sorted(v) for f, v in used.items() if f != "most_selected_players"},
                "schedules": sched_used}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
