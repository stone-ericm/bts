"""W1.2 benchmark bridge driver (design docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md, frozen rev 3).

Exposure row X-21 (docs/audit/2026-09-22-exposure-register.md) must be published before this runs; the driver
refuses otherwise. Run from a checkout of main that contains X-21 and this package, pointing --data-root at the
production data directory (read only; outputs go under --out):

    BTS_LGBM_DETERMINISTIC=1 python -m scripts.audit.benchmark_bridge.run \\
        --data-root ~/projects/bts/data --out ~/projects/bts/data/validation/w12_bridge [--dates D1,D2]

Outputs (registered slate dates and candidates only): table.parquet, summary.json, manifest.json.
"""
from __future__ import annotations

import os

os.environ["BTS_LGBM_DETERMINISTIC"] = "1"          # before any training configuration is imported (Codex r1 F3)

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
X21_COMMIT = "e09a7b7"
UNIT_PREGAME = {"scheduled"}
UNIT_NOT_PREGAME = {"playing", "complete", "postponed"}
SURFACES = ["A26", "A26_count", "B26", "C_frozen", "C_served", "D"]
CHAIN = [("A26", "A26_count"), ("A26_count", "B26"), ("B26", "C_frozen"), ("C_frozen", "C_served"), ("C_served", "D")]
POOLS = ["pool_verified", "pool_surrogate", "pool_all"]
REPRO_TOL = 1e-9


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def x21_gate() -> str:
    """Refuse unless the X-21 commit is in HEAD and the register in this checkout carries the X-21 row."""
    head = git("rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", X21_COMMIT, head]).returncode != 0:
        raise SystemExit(f"X-21 gate: {X21_COMMIT} is not an ancestor of HEAD {head[:7]}")
    reg = (REPO / "docs/audit/2026-09-22-exposure-register.md").read_text()
    if "| X-21 |" not in reg:
        raise SystemExit("X-21 gate: the register in this checkout has no X-21 row")
    return head


def _json(p: Path):
    raw = Path(p).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load_inputs(data: Path, dates: list[str] | None) -> dict:
    slates_dir = data / "picks" / "slates"
    slate_files = sorted(slates_dir.glob("2026-*.json"))
    days = []
    for f in slate_files:
        d = f.stem
        if dates and d not in dates:
            continue
        doc = _json(f)
        pick_path, dec_path = data / "picks" / f"{d}.json", data / "picks" / d / "decision.json"
        pick = _json(pick_path) if pick_path.exists() else None
        dec = _json(dec_path) if dec_path.exists() else None
        days.append({"date": d, "slate_path": f, "slate": doc, "pick_path": pick_path if pick else None, "pick": pick,
                     "decision_path": dec_path if dec else None, "decision": dec})
    return {"days": days, "n_slate_files": len(slate_files)}


def primary_of(pick: dict | None) -> dict | None:
    if not pick or not pick.get("pick"):
        return None
    p = pick["pick"]
    return {"batter_id": p.get("batter_id"), "game_pk": p.get("game_pk"), "p_game_hit": p.get("p_game_hit")}


def decision_primary(dec: dict | None) -> dict | None:
    if not dec or dec.get("action") == "skip" or not dec.get("primary"):
        return None
    return dec["primary"]


def unit_status_by_game(data: Path) -> tuple[dict, dict]:
    from scripts.audit.season_ledger.sources.static import parse_units
    hist: dict[int, list[tuple]] = {}
    vocab: dict[str, int] = {}
    for f in sorted((data / "leaderboard" / "static_snapshots" / "units").glob("*.json")):
        parsed = parse_units(f"units/{f.name}", f.read_bytes())
        for r in parsed.rows:
            if r["feed_id"] is None:
                continue
            hist.setdefault(int(r["feed_id"]), []).append((r["captured_at"], r["status"]))
            vocab[str(r["status"])] = vocab.get(str(r["status"]), 0) + 1
    return hist, vocab


def prepare_workdir(run_dir: Path, data: Path) -> dict:
    """An owned working directory (design §9; Codex r1 F4): bts reads data/raw, data/external and the probable-pitcher
    cache relative to the cwd and WRITES that cache. Raw feeds and external tables are symlinked read-only; the
    cache is a copy, so nothing here can modify production state."""
    import shutil
    work = run_dir / "work"
    (work / "data" / "models").mkdir(parents=True)
    for name in ("raw", "external", "processed"):
        (work / "data" / name).symlink_to(data / name, target_is_directory=True)
    src = data / "models" / "probable_pitcher_lookup.json"
    copied = None
    if src.exists():
        shutil.copy2(src, work / "data" / "models" / src.name)
        copied = sha256_file(work / "data" / "models" / src.name)
    os.chdir(work)
    return {"workdir": str(work), "probable_pitcher_lookup_copy_sha256": copied}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dates", default=None, help="comma-separated subset (smoke runs)")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--skip-ab", action="store_true", help="smoke runs only: skip the A26/B26 walk-forward")
    ap.add_argument("--skip-frozen", action="store_true", help="smoke runs only: skip C-frozen training")
    args = ap.parse_args(argv)

    head = x21_gate()
    from bts.model import predict as pm
    assert pm.LGB_PARAMS.get("deterministic") is True, "BTS_LGBM_DETERMINISTIC did not take effect"
    from bts.data.build import read_pa_for_bts_scoring
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from scripts.audit.benchmark_bridge import core, report

    data = args.data_root.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    work = prepare_workdir(run_dir, data)
    log(f"run dir {run_dir}; code {head[:7]}; workdir {work['workdir']}")

    dates = args.dates.split(",") if args.dates else None
    inp = load_inputs(data, dates)
    days = inp["days"]
    registered = [d["date"] for d in days]
    log(f"slate files {inp['n_slate_files']}; registered dates in this run {len(registered)}")

    # --- PA, features (serving computes features over the whole loaded frame) ---
    proc = data / "processed"
    pq = sorted(proc.glob("pa_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in pq], ignore_index=True)
    log(f"PA rows {len(df):,}; computing features")
    df_feat = compute_all_features(df)
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    season_dates = sorted(df_feat.loc[df_feat["season"] == 2026, "date"].dt.strftime("%Y-%m-%d").unique())

    # --- per-date slate frame ---
    hist, vocab = unit_status_by_game(data)
    rows, day_meta, feeds = [], [], {}
    raw_dir = data / "raw" / "2026"
    for d in days:
        slate = d["slate"]
        written_at = pd.Timestamp(slate["written_at"])
        sel = core.selection_consistency(slate["rows"], decision_primary(d["decision"]), primary_of(d["pick"]))
        meta = {"date": d["date"], "written_at": slate["written_at"], "n_rows": len(slate["rows"]), **{f"sel_{k}": v for k, v in sel.items()},
                "action": (d["decision"] or {}).get("action"), "model_pickle_sha256": (d["pick"] or {}).get("model_pickle_sha256"),
                "feature_env": (d["pick"] or {}).get("feature_env"), "model_git_sha": (d["pick"] or {}).get("model_git_sha")}
        for i, r in enumerate(slate["rows"]):
            pk = int(r["game_pk"])
            if pk not in feeds:
                fp = raw_dir / f"{pk}.json"
                feeds[pk] = _json(fp) if fp.exists() else None
            feed = feeds[pk]
            start = None
            if feed:
                st = feed.get("gameData", {}).get("datetime", {}).get("dateTime")
                start = pd.Timestamp(st) if st else None
            surrogate = core.eligible_surrogate(start, written_at)
            verified = core.verified_eligibility(hist.get(pk, []), written_at, UNIT_PREGAME, UNIT_NOT_PREGAME)
            rows.append({"date": d["date"], "row_order": i, "batter_id": int(r["batter_id"]), "game_pk": pk,
                         "lineup": r.get("lineup"), "projected": bool(r.get("projected")), "D": r.get("p_game_hit"),
                         "scheduled_start": start, "written_at": written_at,
                         "eligible_surrogate": surrogate, "eligible_verified": verified,
                         "sel_state": sel["state"]})
        day_meta.append(meta)
    tab = pd.DataFrame(rows)
    tab["pool_all"] = True
    tab["pool_surrogate"] = tab["eligible_surrogate"] == True  # noqa: E712
    tab["pool_verified"] = (tab["eligible_verified"] == True) & tab["pool_surrogate"]  # noqa: E712
    log(f"candidate rows {len(tab):,}; verified-eligible {int(tab['pool_verified'].sum()):,}")

    # --- C-served and C-frozen ---
    frozen = None
    if not args.skip_frozen:
        log("training C-frozen on 2019-2025")
        train_df = df_feat[df_feat["season"] <= 2025]
        frozen = {**pm.train_blend(train_df), "_model": pm.train_model(train_df)}
    c_served_parts, c_frozen_parts, served_status = [], [], {}
    for d, meta in zip(days, day_meta):
        g = tab[tab["date"] == d["date"]]
        slots = []
        for r in d["slate"]["rows"]:
            feed = feeds.get(int(r["game_pk"]))
            slot = core.reconstruct_slot(r, feed)[0] if feed else None
            if slot is not None:
                slots.append(slot)
        if slots and frozen is not None:
            cf = core.score_c(d["date"], slots, df_feat, frozen).rename(columns={"p_game_hit": "C_frozen"})
            cf["date"] = d["date"]
            c_frozen_parts.append(cf)
        pkl = data / "models" / f"blend_{d['date']}.pkl"
        want = meta["model_pickle_sha256"]
        if not pkl.exists():
            served_status[d["date"]] = "artifact_missing"
        elif not want or sha256_file(pkl) != want:
            served_status[d["date"]] = "sha_unbound"
        elif slots:
            cs = core.score_c(d["date"], slots, df_feat, pm.load_blend(pkl)).rename(columns={"p_game_hit": "C_served"})
            cs["date"] = d["date"]
            c_served_parts.append(cs)
            served_status[d["date"]] = "scored"
        log(f"  {d['date']}: C-served {served_status.get(d['date'])}")
    key = ["date", "batter_id", "game_pk"]
    for parts in (c_frozen_parts, c_served_parts):
        if parts:
            tab = tab.merge(pd.concat(parts, ignore_index=True), on=key, how="left")
    for col in ("C_frozen", "C_served"):
        if col not in tab:
            tab[col] = float("nan")

    # --- A26 / B26 on the original 2026 calendar, identical fitted models via the cache ---
    from bts.model.predict import BLEND_CONFIGS
    names = [c[0] for c in BLEND_CONFIGS]
    cache = run_dir / "model_cache"
    profiles = []
    if not args.skip_ab:
        log("A26 walk-forward (actual_pa)")
        a26 = blend_walk_forward(df_feat, 2026, top_n=10 ** 7, cache_dir=cache, cache_seed=42,
                                 cache_reuse_configs=names, game_probability_mode="actual_pa")
        log("B26 walk-forward (estimated_pa, cached models)")
        b26 = blend_walk_forward(df_feat, 2026, top_n=10 ** 7, cache_dir=cache, cache_seed=42,
                                 cache_reuse_configs=names, game_probability_mode="estimated_pa")
        profiles = [(a26, "A26"), (b26, "B26")]
    else:
        tab["A26"], tab["B26"], tab["n_pas_actual"] = float("nan"), float("nan"), float("nan")
    for prof, col in profiles:
        prof = prof.assign(date=pd.to_datetime(prof["date"]).dt.strftime("%Y-%m-%d"))
        keep = prof[["date", "batter_id", "game_pk", "p_game_hit"] + (["n_pas"] if col == "A26" else [])]
        tab = tab.merge(keep.rename(columns={"p_game_hit": col, "n_pas": "n_pas_actual"}), on=key, how="left")
    pa26 = df[df["season"] == 2026].assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
    slot_mode = (pa26.groupby(["date", "batter_id", "game_pk"])["lineup_position"]
                 .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None).rename("lineup_realized").reset_index())
    tab = tab.merge(slot_mode, on=key, how="left")
    tab["A26_count"] = [core.count_normalized(a, n, int(s) if pd.notna(s) else None) if pd.notna(a) else float("nan")
                        for a, n, s in zip(tab["A26"], tab["n_pas_actual"].fillna(0), tab["lineup_realized"])]

    # --- outcomes (after every score is fixed) ---
    pa_s = read_pa_for_bts_scoring(proc / "pa_2026.parquet", ["date", "game_pk", "batter_id", "is_hit"])
    pa_s = pa_s.assign(date=pd.to_datetime(pa_s["date"]).dt.strftime("%Y-%m-%d"))
    final_games = {pk for pk, f in feeds.items()
                   if f and f.get("gameData", {}).get("status", {}).get("abstractGameState") == "Final"}
    tab = core.outcome_table(tab, pa_s, final_games)

    # --- reproduction (C-served vs D) ---
    both = tab[tab["C_served"].notna() & tab["D"].notna() & (tab["sel_state"] == "selection_consistent")]
    resid = (both["C_served"].map(core.serialized_probability) - both["D"]).abs()
    repro = {"rows": int(len(both)), "dates": int(both["date"].nunique()),
             "within_tol": int((resid <= REPRO_TOL).sum()),
             "resid_quantiles": {q: float(resid.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(both) else {},
             "served_status": served_status}

    # --- metrics ---
    cohorts = {"selection_consistent": tab[tab["sel_state"] == "selection_consistent"], "all_dates": tab}
    metrics = {}
    for cname, ct in cohorts.items():
        metrics[cname] = {pool: {"surfaces": [report.surface_metrics(ct, s, pool) for s in SURFACES],
                                 "pairs": [report.pair_metrics(ct, a, b, pool, n_resamples=args.n_resamples)
                                           for a, b in CHAIN]}
                          for pool in POOLS}
    # archived D primary on selection-consistent dates
    actions = pd.Series([m["action"] for m in day_meta]).value_counts(dropna=False).to_dict()

    summary = {"schema": "w12_bridge_v1", "code": head, "smoke_flags": {"skip_ab": args.skip_ab, "skip_frozen": args.skip_frozen},
               "workdir": work, "x21_commit": X21_COMMIT, "run_dir": str(run_dir),
               "registered_dates": registered, "season_game_dates": len(season_dates),
               "missing_fraction": 1 - len(registered) / len(season_dates) if season_dates else None,
               "unit_status_vocabulary": vocab, "day_meta": day_meta, "actions": {str(k): int(v) for k, v in actions.items()},
               "reproduction": repro, "metrics": metrics,
               "pools": {p: int(tab[p].sum()) for p in POOLS}}
    tab.drop(columns=["scheduled_start", "written_at"]).to_parquet(run_dir / "table.parquet", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    manifest = {"slates": {str(d["slate_path"].name): sha256_file(d["slate_path"]) for d in days},
                "picks": {d["date"]: sha256_file(d["pick_path"]) for d in days if d["pick_path"]},
                "decisions": {d["date"]: sha256_file(d["decision_path"]) for d in days if d["decision_path"]},
                "pa_parquets": {p.name: sha256_file(p) for p in pq}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
