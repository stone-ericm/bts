"""W2.3 MLB forecast benchmark driver (design docs/superpowers/specs/2026-10-04-mlb-forecast-benchmark-design.md rev 3,
FROZEN; code review docs/audit/2026-10-04-w13-w23-code-codex-r1.md). Reads the ACCEPTED W1.2 run (verified bytes:
served D, eligibility pools, selection state, outcomes) and the BTS static captures; links MLB's probabilityStarter to
the served slate by inference only; scores both forecasters on the frozen shared pool. Every consumed input is read
once and hashed from the bytes used. Refuses to run before exposure row X-23 is in HEAD."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
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
IDENTITY_FIELDS = {"units": ("id", "roundId", "feedId", "homeSquadId", "awaySquadId"),
                   "players": ("id", "feedId", "squadId"), "rounds": ("id",),
                   "mostSelectedPlayers": ("roundId", "playerId")}
LIMITS = {"target_semantics": "unresolved: association / target sensitivity only (gate 2)",
          "timing": "capture stamps are run-start stamps, not receipt times; written_at is the slate's last write",
          "linkage": "every accepted link is inferred_unique_game, never witnessed"}


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


def validate_sheet(doc, key: str):
    """The sheet's item list if the document is schema-valid (a dict whose ``key`` is a list of dicts); None when it
    is not. A valid empty list is valid empty support; an invalid document is never selected as the latest sheet."""
    if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
        return None
    items = doc[key]
    if not all(isinstance(i, dict) for i in items):
        return None
    for item in items:                       # typed identities (code review r2 N1): exact int or null, never coerced
        for f in IDENTITY_FIELDS.get(key, ()):
            v = item.get(f)
            if v is not None and not core._int(v):
                return None
    return items


def latest_at(sheets: dict, stamp) -> tuple:
    """(stamp, items) of the latest stored sheet at or before ``stamp`` from {stamp: items}; (None, None) if none."""
    cut = pd.Timestamp(stamp)
    eligible = [s for s in sheets if pd.Timestamp(s) <= cut]
    if not eligible:
        return None, None
    best = max(eligible, key=pd.Timestamp)
    return best, sheets[best]


class Feed:
    """One static-capture feed: each file read once, hashed from the bytes read and schema-validated."""

    def __init__(self, directory: Path, key: str):
        self.key = key
        self.paths = {stamp_to_utc(p.name): p for p in bridge.capture_files(directory) if stamp_to_utc(p.name)}
        self.loaded: dict = {}        # stamp -> (items or None, sha256, file name)

    def load(self, stamp):
        if stamp not in self.loaded:
            p = self.paths[stamp]
            raw = p.read_bytes()
            try:
                items = validate_sheet(load_json_bytes(raw), self.key)
            except ValueError:
                items = None
            self.loaded[stamp] = (items, hashlib.sha256(raw).hexdigest(), p.name)
        return self.loaded[stamp]

    def latest_valid_at(self, stamp):
        """(stamp, items, sha) of the latest schema-valid capture at or before ``stamp``; invalid ones are skipped."""
        cut = pd.Timestamp(stamp)
        for s in sorted((s for s in self.paths if pd.Timestamp(s) <= cut), key=pd.Timestamp, reverse=True):
            items, sha, _ = self.load(s)
            if items is not None:
                return s, items, sha
        return None, None, None

    def used(self) -> dict:
        return {name: {"sha256": sha, "schema_valid": items is not None} for items, sha, name in self.loaded.values()}


def prepare_date(date: str, written_at: pd.Timestamp, slate: pd.DataFrame, msp: list[tuple], sheet_at, sched_pks: set,
                 msp_sha: dict | None = None):
    """One date's as-of join (gate 1). The forecast sheet is the latest schema-valid whole sheet at or before
    written_at; its run-start stamp is the boundary for the rounds/players/units lookups (A-E2, each the latest
    schema-valid capture at or before it); the round's units must be contradiction-free and cover the MLB schedule
    before any unique-game inference. Every joined row carries its MLB identities and source hashes."""
    msp_sha = msp_sha or {}
    cov: dict = {"date": date, "written_at": str(written_at), "excluded": None}
    stamp, _ = core.latest_sheet(msp, {}, date, written_at)
    cov["forecast_stamp"] = stamp
    if stamp is None:
        cov["excluded"] = "no_forecast_sheet"
        return pd.DataFrame(), cov
    r_stamp, rounds, r_sha = sheet_at("rounds", stamp)
    round_id, why = core.round_for_date(rounds or [], date)
    cov.update(rounds_stamp=r_stamp, round_id=round_id)
    if round_id is None:
        cov["excluded"] = why
        return pd.DataFrame(), cov
    if not [r for r in dict(msp)[stamp] if r.get("roundId") == round_id]:
        cov["excluded"] = "no_target_round_rows"
        return pd.DataFrame(), cov
    p_stamp, players, p_sha = sheet_at("players", stamp)
    feed, squad, conflicts = core.player_lookup(players or [])
    cov.update(players_stamp=p_stamp, player_conflicts=len(conflicts))
    fc, fcounts = core.forecasts_counted(msp, {round_id: date}, feed, date, written_at)
    cov["forecast_counts"] = fcounts
    u_stamp, units, u_sha = sheet_at("units", stamp)
    units = units or []
    n_conf = core.unit_conflicts(units, round_id)
    complete = core.units_complete(units, round_id, sched_pks)
    cov.update(units_stamp=u_stamp, units_complete=complete, unit_conflicts=n_conf, n_forecasts=len(fc))
    games = (core.games_by_batter(fc, squad, units, round_id) if complete and n_conf == 0
             else {b: None for b in fc})
    joined, link_cov = core.join_to_slate(slate, fc, games)
    cov.update(link_cov)
    linked = joined["link_status"] == "inferred_unique_game"
    pid = {b: f["player_id"] for b, f in fc.items()}
    joined["mlb_player_id"] = [pid.get(b) if ok else None for b, ok in zip(joined["batter_id"], linked)]
    joined["mlb_squad_id"] = [squad.get(pid.get(b)) if ok else None for b, ok in zip(joined["batter_id"], linked)]
    joined["mlb_unit_ids"] = [sorted(int(u["id"]) for u in units if u.get("roundId") == round_id
                                     and squad.get(pid.get(b)) in (u.get("homeSquadId"), u.get("awaySquadId"))
                                     and core._int(u.get("id"))) if ok else None
                              for b, ok in zip(joined["batter_id"], linked)]
    joined["mlb_round_id"] = round_id
    joined["mlb_n_sel"] = [fc[b]["n_sel"] if ok else np.nan for b, ok in zip(joined["batter_id"], linked)]
    joined["mlb_unchanged_since"] = [core.unchanged_since(msp, stamp, round_id, fc[b]["player_id"]) if ok else None
                                     for b, ok in zip(joined["batter_id"], linked)]
    joined["forecast_stamp"] = stamp
    joined["src_forecast_sha"] = msp_sha.get(stamp)
    joined["src_rounds"] = f"{r_stamp}:{r_sha}"
    joined["src_players"] = f"{p_stamp}:{p_sha}"
    joined["src_units"] = f"{u_stamp}:{u_sha}"
    return joined, cov


def _auc(g: pd.DataFrame, score: str) -> float:
    a = bridge._rank_auc(g.loc[g["y"] == 1, score], g.loc[g["y"] == 0, score])
    return float("nan") if a is None else float(a)


def _nanmean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else float("nan")


def score_stratum(pool: pd.DataFrame, t: str, n_resamples: int) -> dict:
    """Equal-date proper scores, residual and AUC for both arms under target ``t`` (gate 5), with joint whole-date
    intervals over per-date summaries (``m.summary_bootstrap``, identical to the copy-block bootstrap). Single-class
    dates are omitted from AUC and counted. Native and target-known populations are both reported."""
    tdf = m.target(pool, t)
    native = {"rows": int(len(pool)), "dates": int(pool["date"].nunique()),
              "outcomes": pool["outcome"].value_counts().to_dict()}
    if tdf.empty:
        return {"available": False, "reason": "no_target_known_rows", "native": native}
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
    return {"available": True, "native": native, "n_dates": int(len(per_date)), "n_rows": int(len(tdf)),
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


def score_encompassing(pool: pd.DataFrame, n_resamples: int, seed: int = 20261004) -> dict:
    """Gate 6 on the T2 population: one encompassing fit pair per bootstrap draw (copy blocks), both fields recorded
    from that one result; failed draws are counted with reasons and make the interval unavailable."""
    tdf = m.target(pool, "T2")
    if tdf.empty:
        return {"point": {"available": False, "reason": "no_T2_rows"}}
    point = m.encompassing(tdf.assign(_block=tdf["date"]))
    if not point["available"]:
        return {"point": point}
    dates = np.array(sorted(tdf["date"].unique()))
    groups = {d: g for d, g in tdf.groupby("date", sort=False)}
    rng = np.random.default_rng(seed)
    coef, dll, reasons = np.full(n_resamples, np.nan), np.full(n_resamples, np.nan), {}
    for i in range(n_resamples):
        pick = rng.choice(dates, size=len(dates), replace=True)
        r = m.encompassing(pd.concat([groups[d].assign(_block=k) for k, d in enumerate(pick)], ignore_index=True))
        if r["available"]:
            coef[i], dll[i] = r["coef_mlb"], r["delta_log_loss"]
        else:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"point": point, "intervals": {"coef_mlb": m.collect(coef, n_resamples, seed, reasons),
                                          "delta_log_loss": m.collect(dll, n_resamples, seed, reasons)}}


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
    w12_bytes, w12_acc = bridge.read_accepted_run(w12)
    table = pd.read_parquet(io.BytesIO(w12_bytes["table.parquet"]))
    summary = json.loads(w12_bytes["summary.json"])
    written = {d["date"]: pd.Timestamp(d["written_at"]) for d in summary["day_meta"]}
    snaps = args.data_root.expanduser().resolve() / "leaderboard" / "static_snapshots"
    feeds = {f: Feed(snaps / f, key) for f, key in FEEDS.items()}
    mspf = feeds["most_selected_players"]
    msp, msp_sha, msp_invalid = [], {}, []
    for s in sorted(mspf.paths, key=pd.Timestamp):
        items, sha, name = mspf.load(s)
        if items is None:
            msp_invalid.append(name)
            continue
        msp.append((s, items))
        msp_sha[s] = sha

    def sheet_at(feed, stamp):
        return feeds[feed].latest_valid_at(stamp)

    stamp_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp_now}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"run dir {run_dir}; code {head[:7]}; W1.2 run {w12.name}; invalid forecast sheets {len(msp_invalid)}")

    parts, covs, sched_used = [], [], {}
    for d in sorted(written):
        if d < WINDOW_START:
            covs.append({"date": d, "excluded": "before_capture_window"})
            continue
        sp = args.schedules / f"{d}.json"
        sched = set()
        if sp.exists():
            raw = sp.read_bytes()
            sched_used[sp.name] = hashlib.sha256(raw).hexdigest()
            parsed = parse_schedule(f"schedules/{d}.json", raw)
            sched = set() if parsed.quarantined else {int(r["game_pk"]) for r in parsed.rows}
        joined, cov = prepare_date(d, written[d], table[table["date"] == d], msp, sheet_at, sched, msp_sha)
        covs.append(cov)
        if not joined.empty:
            parts.append(joined)
    joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=list(table.columns) + ["link_status", "mlb_p", "forecast_stamp", "mlb_unchanged_since", "mlb_n_sel"])
    log(f"joined rows {len(joined):,}; dates with a forecast join {len(parts)}")

    ok = joined["link_status"] == "inferred_unique_game"
    valid = joined["D"].apply(core.valid_p) & joined["mlb_p"].apply(core.valid_p)
    joined = joined.assign(ours=joined["D"], mlb=joined["mlb_p"], in_pool_base=ok & valid)
    rec = joined[joined["in_pool_base"]]
    recency = {"limits": LIMITS["timing"]}
    if len(rec):
        recency.update({
            "sheet_to_written_at_minutes": ((pd.to_datetime(rec["date"].map(written), utc=True)
                                             - pd.to_datetime(rec["forecast_stamp"], utc=True))
                                            .dt.total_seconds() / 60).describe().to_dict(),
            "observed_unchanged_minutes": ((pd.to_datetime(rec["forecast_stamp"], utc=True)
                                            - pd.to_datetime(rec["mlb_unchanged_since"], utc=True))
                                           .dt.total_seconds() / 60).describe().to_dict(),
            "n_sel_quantiles": rec["mlb_n_sel"].astype(float).quantile([0, .25, .5, .75, 1]).to_dict()})
    results = {"schema": "w23_mlb_benchmark_v2", "code": head, "x23_commit": X23_COMMIT, "w12_run": str(w12),
               "w12_accepted": {k: v for k, v in w12_acc.items() if k != "files"}, "limits": LIMITS,
               "coverage": {"per_date": covs, "invalid_forecast_sheets": msp_invalid,
                            "invalid_shared_scores": int((ok & ~valid).sum()),
                            "link_status": joined["link_status"].value_counts().to_dict()},
               "recency": recency, "strata": {}}
    for cohort in COHORTS:
        for pool_col in POOLS:
            key = f"{cohort}/{pool_col}"
            if joined.empty:
                results["strata"][key] = {"available": False, "reason": "no_joined_rows"}
                continue
            pool = joined[joined["in_pool_base"] & (joined["sel_state"] == cohort)
                          & joined[pool_col].fillna(False).astype(bool)]
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
    manifest = {"w12_accepted_files": {k: hashlib.sha256(v).hexdigest() for k, v in w12_bytes.items()},
                "feeds": {f: fd.used() for f, fd in feeds.items()}, "schedules": sched_used}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
