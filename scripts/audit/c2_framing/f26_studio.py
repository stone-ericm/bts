"""The 2026 framing test, run off the box (exposure row X-37, 2026-11-12).

The science is f26.py's, unchanged: the starter-proxy table, the as-of catcher value, the arm transform, the
walk-forward hook and the §6 dispositions. This module replaces only f26.py's C1 launcher path (admission gate,
ledger, reservation, guarded unit, launch window), which needs the deleted box and is superseded by the work
policies of 2026-11-12. What it keeps from that path: every input is read from bytes whose sha256 is recorded in
PREPARED.json; a seed runs once; the aggregate refuses anything but the ten registered seeds, each complete.

    prepare    --data-dir D --raw-dir R --lookup L --inputs I     once: table, 2017-2026 lookup, sources, pa_2026 copy
    run        --seed N --data-dir D --inputs I --out O           one seed, three arms
    aggregate  --data-dir D --inputs I --out O                    after all ten seeds: dispositions
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import screen as S

PREPARED = "PREPARED.json"
SCHEMA = "c2_framing_2026_studio_v1"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    """Never overwrite: a second prepare or a second run of a seed is a refusal, not a replacement."""
    with open(path, "xb") as f:
        f.write(data)


def prepared(inputs: Path) -> dict:
    return json.loads((inputs / PREPARED).read_bytes())


# ---------------------------------------------------------------- prepare

def prepare(data_dir: Path, raw_dir: Path, lookup_path: Path, inputs: Path) -> dict:
    """Design §3: the frozen pa_2026 copy, the 2017-2026 lookup, the 2025-2026 starter-proxy table and the raw-source
    manifest, each written once with its sha256 recorded. Only identity and schedule fields are read from the feeds
    (f26.starter_proxy, f26.lookup_entry)."""
    import pandas as pd
    screen = json.loads((Path(__file__).parent / "admission.json").read_bytes())["input_pins"]
    inputs.mkdir(parents=True, exist_ok=False)
    pins = {f"pa_{s}.parquet": screen[f"pa_{s}.parquet"] for s in F.SEASONS_IN if s != F.TEST_SEASON}
    raw26 = (data_dir / F.FROZEN_PA).read_bytes()
    _write_new(inputs / F.FROZEN_PA, raw26)
    pins[F.FROZEN_PA] = _sha(raw26)
    games = {}
    for s in F.PROXY_SEASONS:
        raw = raw26 if s == F.TEST_SEASON else (data_dir / f"pa_{s}.parquet").read_bytes()
        if s != F.TEST_SEASON and _sha(raw) != pins[f"pa_{s}.parquet"]:
            raise F.InputRefused(f"pa_{s}.parquet is not the screen's pinned file")
        pa = pd.read_parquet(io.BytesIO(raw), columns=["game_pk", "season"])
        if not bool((pa["season"] == s).all()):
            raise F.InputRefused(f"pa_{s} holds rows of another season")
        games[s] = F.pa_game_ids(pa)
    lookup_raw = lookup_path.read_bytes()
    if _sha(lookup_raw) != screen[S.LOOKUP_NAME]:
        raise F.InputRefused("the 2017-2025 lookup is not the screen's pinned file")
    lookup = json.loads(lookup_raw)
    records, sources = [], []
    for s in F.PROXY_SEASONS:
        for pk in games[s]:
            rel = f"{s}/{pk}.json"
            path = raw_dir / rel
            if not path.is_file():
                raise F.InputRefused(f"no raw feed {rel} for a {s} game in the PA file")
            b = path.read_bytes()
            sources.append({"path": rel, "sha256": _sha(b), "game_pk": pk, "season": s})
            feed = json.loads(b)
            records += F.starter_proxy(feed, pk, s)
            if s == F.TEST_SEASON:
                if str(pk) in lookup:
                    raise F.InputRefused(f"2026 game {pk} is already in the 2017-2025 lookup")
                lookup[str(pk)] = F.lookup_entry(feed, pk)
    F.table_frame(records)                          # validates every record's shape
    records.sort(key=lambda r: (r["season"], r["official_date"], r["game_number"], r["game_pk"], r["fielding_side"]))
    manifest = {"sources": sources, "counts": {str(s): len(games[s]) for s in F.PROXY_SEASONS},
                "digest": _sha(F.canonical(sources))}
    files = {F.LOOKUP_NAME: F.canonical(dict(sorted(lookup.items(), key=lambda kv: int(kv[0])))),
             F.TABLE_NAME: F.canonical(records), F.SOURCES_NAME: F.canonical(manifest)}
    for name, b in files.items():
        _write_new(inputs / name, b)
        pins[name] = _sha(b)
    reasons: dict = {}
    for r in records:
        by = reasons.setdefault(str(r["season"]), {})
        by[r["reason"]] = by.get(r["reason"], 0) + 1
    rec = {"schema": SCHEMA, "exposure_row": F.EXPOSURE_ROW, "pins": dict(sorted(pins.items())),
           "games": manifest["counts"], "proxy_reasons": reasons,
           "game_number_fallbacks": sum(r["game_number_fallback"] for r in records),
           "lookup_2026_missing_probable_sides": sum(1 for pk in games[F.TEST_SEASON] for k in ("away", "home")
                                                     if lookup[str(pk)][k] is None),
           "machine": platform.platform(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _write_new(inputs / PREPARED, F.canonical(rec))
    return rec


# ---------------------------------------------------------------- one seed

def calendar_2026(inputs: Path, pins: dict) -> list[str]:
    """The evaluation calendar from the frozen pa_2026 file ALONE. The concatenated ten-season frame cannot be used:
    2017-2025 have no `is_resumed_portion` column, so after the concat the flag is an object column with missing
    values and f26.expected_calendar rightly refuses it (found on the first real seed-1 start, 2026-11-12)."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    raw, _ = A.read_pinned(inputs / F.FROZEN_PA, pins[F.FROZEN_PA])
    return F.expected_calendar(pd.read_parquet(io.BytesIO(raw)))


def run(seed: int, data_dir: Path, inputs: Path, out: Path, *, walk_forward=None) -> int:
    import pandas as pd
    if os.environ.get("BTS_LGBM_DETERMINISTIC") != "1":
        raise SystemExit("refusing: BTS_LGBM_DETERMINISTIC=1 is pre-registered (set before bts is imported)")
    if seed not in F.SEEDS:
        raise SystemExit(f"refusing: {seed} is not a registered seed")
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
    from bts.model.predict import LGB_PARAMS
    if not (LGB_PARAMS.get("deterministic") is True and LGB_PARAMS.get("force_row_wise") is True):
        raise SystemExit("refusing: LightGBM's params were built without the deterministic flags")
    settings = S.check_settings()
    pins = prepared(inputs)["pins"]
    run_dir = out / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)     # a seed runs once (design §5: no replacement run)
    from scripts.audit.c1 import admission as A
    S.install_closed_inputs(S.frozen_lookup(A.read_pinned(inputs / F.LOOKUP_NAME, pins[F.LOOKUP_NAME])[0]))
    table = F.table_frame(F.read_pinned_json(inputs / F.TABLE_NAME, pins[F.TABLE_NAME]))
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    walk_forward = walk_forward or blend_walk_forward
    t0 = S.cpu_seconds()
    raw_df = F.load_inputs(data_dir, inputs, pins)
    try:
        F.history_start_problem(raw_df)
        games = {y: F.pa_game_ids(raw_df[raw_df["season"] == y]) for y in F.PROXY_SEASONS}
        problem = (F.sources_problem(F.read_pinned_json(inputs / F.SOURCES_NAME, pins[F.SOURCES_NAME]), games)
                   or F.table_games_problem(table, games))
        if problem:
            raise F.InputRefused(problem)
        calendar = calendar_2026(inputs, pins)
    except F.InputRefused as e:
        S._json(run_dir / "STOPPED.json", {"reason": "inputs", "detail": str(e)})
        raise
    df = compute_all_features(raw_df)
    check = S.self_check(df)
    df = S.framing_by(df, "fielding_catcher_id", F.NEW_COL)
    labels = S.original_portion_labels(df)
    asof = F.AsOfFraming(df)
    S._json(run_dir / "manifest.json", {
        "schema": SCHEMA, "seed": seed, "input_pins": pins, "test_season": F.TEST_SEASON, "arms": list(F.ARMS),
        "basis": S.BASIS, "retrain_every": S.RETRAIN_EVERY, "lgb_params": dict(LGB_PARAMS),
        "feature_settings": settings, "scoring": dict(S.SCORING),
        "env": {k: os.environ.get(k) for k in ("BTS_LGBM_DETERMINISTIC", "BTS_LGBM_RANDOM_STATE", "TZ")},
        "self_check": check, "calendar": calendar, "resumed_flag_2026": F.resumed_totals(raw_df),
        "machine": platform.platform(), "features_cpu_s": S.cpu_seconds() - t0})
    if not check["identical"]:
        S._json(run_dir / "STOPPED.json", {"reason": "self_check", "self_check": check})
        return 4
    featureless = F.featureless_rows(df)
    if featureless:
        S._json(run_dir / "STOPPED.json", {"reason": "all_features_missing", "rows": featureless})
        return 5
    units, profiles = [], {}
    for arm in F.ARMS:
        transform = None if arm == "baseline" else F.ArmTransform(arm, table, asof)
        configs = S.blend_configs("baseline" if arm == "baseline" else "A")
        c0, w0 = S.cpu_seconds(), time.monotonic()
        p = walk_forward(df, F.TEST_SEASON, retrain_every=S.RETRAIN_EVERY, blend_configs=configs,
                         game_probability_mode=S.BASIS, predict_day_transform=transform, strict_predict=True)
        cpu = S.cpu_seconds() - c0
        p, counts = S.relabel(p, labels)
        p["season"] = F.TEST_SEASON
        path = run_dir / f"profiles_{arm}_{F.TEST_SEASON}.parquet"
        p.to_parquet(path, index=False)
        unit = {"arm": arm, "season": F.TEST_SEASON, "cpu_s": cpu, "wall_s": time.monotonic() - w0, "labels": counts}
        if transform is not None:
            unit["catcher"] = {"counts": transform.counts, "identified_ids": sorted(transform.identified_ids)}
            pd.DataFrame(transform.evidence, columns=list(F.EVIDENCE_COLS)).astype(
                {"catcher_id": "Int64", "value": "float64"}).to_parquet(
                run_dir / f"catcher_{arm}_{F.TEST_SEASON}.parquet", index=False)
        units.append(unit)
        S._json(run_dir / "units.json", units)
        profiles[arm] = pd.read_parquet(path)       # score exactly the retained bytes
    cards = {a: compute_full_scorecard(p, **S.SCORING) for a, p in profiles.items()}
    for a, c in cards.items():
        S._json(run_dir / f"scorecard_{a}.json", c)
    results = {"schema": SCHEMA, "seed": seed, "units": units, "total_cpu_s": S.cpu_seconds() - t0,
               "calendar_sha256": _sha(F.canonical(calendar)),
               "rank1": {a: F.rank1(p, calendar) for a, p in profiles.items()}, "arms": {}}
    for a in F.ARMS[1:]:
        diff = diff_scorecards(cards["baseline"], cards[a])
        S._json(run_dir / f"diff_{a}.json", diff)
        results["arms"][a] = F.arm_summary(diff)
    S._json(run_dir / "results.json", results)      # written last: its presence means the seed is complete
    return 0


# ---------------------------------------------------------------- the aggregate

REGISTERED = {"test_season": F.TEST_SEASON, "arms": list(F.ARMS), "basis": S.BASIS, "retrain_every": S.RETRAIN_EVERY}


def trusted(data_dir: Path, inputs: Path, pins: dict) -> "F.Trusted":
    """What every seed is checked against, derived here from the pinned inputs and never from a run (design §6,
    review r1 B1-B3): the calendar, each scoreable batter-game's date and original-portion label, each predicted
    day's side-games, and each A arm's catcher and as-of value per side-game, recomputed from the pinned PA files."""
    truth = F.load_trusted(inputs, pins)
    asof, _ = F.recompute_asof(data_dir, inputs, pins)
    truth.expected = F.expected_values(F.expected_evidence(truth, asof))
    return truth


def _same_json(a, b) -> bool:
    drop = lambda o: {k: v for k, v in o.items() if k != "timestamp"} if isinstance(o, dict) else o   # noqa: E731
    return S._canon(drop(json.loads(S._canon(a)))) == S._canon(drop(json.loads(S._canon(b))))


def validate_seed(out: Path, seed: int, pins: dict, truth: "F.Trusted") -> dict:
    """One finished seed as complete, valid evidence (design §6 "Completeness first"), or RunInvalid. Nothing a run
    recorded about itself is trusted: the calendar and labels come from the pinned inputs, the rank-1 hits, scorecards,
    diffs and summaries are recomputed from the retained profiles, and the catcher evidence is re-derived."""
    import pandas as pd
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    d = out / f"seed_{seed}"
    if (d / "STOPPED.json").exists() or not (d / "results.json").exists():
        raise F.RunInvalid(f"seed {seed} is stopped or unfinished")
    man, res = json.loads((d / "manifest.json").read_bytes()), json.loads((d / "results.json").read_bytes())
    if man.get("seed") != seed or res.get("seed") != seed or man.get("input_pins") != pins:
        raise F.RunInvalid(f"seed {seed}: the run's seed or input pins are not this test's")
    env, lgb = man.get("env") or {}, man.get("lgb_params") or {}
    if (any(man.get(k) != v for k, v in REGISTERED.items()) or man.get("scoring") != dict(S.SCORING)
            or man.get("feature_settings") != S.SETTINGS or not (man.get("self_check") or {}).get("identical")
            or env.get("BTS_LGBM_DETERMINISTIC") != "1" or env.get("BTS_LGBM_RANDOM_STATE") != str(seed)
            or lgb.get("deterministic") is not True or lgb.get("force_row_wise") is not True):
        raise F.RunInvalid(f"seed {seed}: the run was not made under the registered settings")
    calendar = truth.calendar
    if not calendar or man.get("calendar") != calendar:
        raise F.RunInvalid(f"seed {seed}: its calendar is not the pinned pa_2026's complete calendar")
    units = {u.get("arm"): u for u in res.get("units") or []}
    if list(units) != list(F.ARMS):
        raise F.RunInvalid(f"seed {seed}: its units are not exactly the three arms in order")
    hits, ties, cards = {}, {}, {}
    for arm in F.ARMS:
        name = f"profiles_{arm}_{F.TEST_SEASON}.parquet"
        part = pd.read_parquet(d / name)
        problem = S.profile_problem(part, F.TEST_SEASON) or F.identity_problem(part, truth)
        if problem is None:
            problem, ties[arm] = F.order_problem(part)
        if problem is not None:
            raise F.RunInvalid(f"seed {seed} {name}: {problem}")
        try:
            hits[arm] = F.rank1(part, calendar)
        except F.RunInvalid as e:
            raise F.RunInvalid(f"seed {seed} {name}: its dates are not the complete calendar ({e})")
        if hits[arm] != (res.get("rank1") or {}).get(arm):
            raise F.RunInvalid(f"seed {seed} {arm}: retained profiles do not reproduce the recorded rank-1 hits")
        cards[arm] = compute_full_scorecard(part, **S.SCORING)
        if not _same_json(cards[arm], json.loads((d / f"scorecard_{arm}.json").read_bytes())):
            raise F.RunInvalid(f"seed {seed} {arm}: its scorecard is not what its retained profiles give")
    summaries = {}
    for arm in F.ARMS[1:]:
        ev = pd.read_parquet(d / f"catcher_{arm}_{F.TEST_SEASON}.parquet")
        problem = F.evidence_problem(ev, arm, truth, units[arm].get("catcher") or {})
        if problem is not None:
            raise F.RunInvalid(f"seed {seed}: {problem}")
        diff = json.loads(S._canon(diff_scorecards(cards["baseline"], cards[arm])))
        summaries[arm] = F.arm_summary(diff)
        if not _same_json(summaries[arm], (res.get("arms") or {}).get(arm)):
            raise F.RunInvalid(f"seed {seed} {arm}: its stored summary is not what its scorecards give")
    return {"rank1": hits, "ties": ties, "summaries": summaries,
            "coverage": {a: units[a]["catcher"]["counts"] for a in F.ARMS[1:]},
            "walk_forward_cpu_s": sum(float(u["cpu_s"]) for u in units.values()),
            "total_cpu_s": float(res["total_cpu_s"])}


def deltas(rank1s: list[dict], arm: str):
    """x[s, t]: the arm's rank-1 hit minus the baseline's, seeds in the given order, dates ascending (design §6)."""
    import numpy as np
    return np.array([np.array(r[arm]) - np.array(r["baseline"]) for r in rank1s], dtype=float)


def aggregate(data_dir: Path, inputs: Path, out: Path) -> dict:
    """Design §6. Refuses unless all ten registered seeds are complete, valid evidence on the pinned calendar;
    A-posted decides, A-projected is described on the same calendar."""
    import numpy as np
    pins = prepared(inputs)["pins"]
    truth = trusted(data_dir, inputs, pins)
    seeds = [validate_seed(out, s, pins, truth) for s in F.SEEDS]            # §5's literal order
    rank1s = [s["rank1"] for s in seeds]
    calendar = truth.calendar
    agg = {"schema": SCHEMA, "seeds": list(F.SEEDS), "n_days": len(calendar),
           "calendar": [calendar[0], calendar[-1]], "calendar_sha256": _sha(F.canonical(calendar)),
           "A_posted": F.dispose(deltas(rank1s, "A_posted")),
           "A_projected_descriptive": {k: v for k, v in F.dispose(deltas(rank1s, "A_projected")).items()
                                       if k != "disposition"},
           "rank1_hit_rate_by_seed": {a: [float(np.mean(r[a])) for r in rank1s] for a in F.ARMS},
           "catcher_coverage_by_seed": [s["coverage"] for s in seeds],
           "rank_ties_by_seed": [s["ties"] for s in seeds],
           "streak_summaries_by_seed": [s["summaries"] for s in seeds],
           "walk_forward_cpu_h": sum(s["walk_forward_cpu_s"] for s in seeds) / 3600,
           "total_cpu_h": sum(s["total_cpu_s"] for s in seeds) / 3600}
    _write_new(out / "aggregate.json", (json.dumps(agg, indent=1, sort_keys=True) + "\n").encode())
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--lookup", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--inputs", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        rec = prepare(args.data_dir, args.raw_dir, args.lookup, args.inputs)
        print(json.dumps({k: rec[k] for k in ("games", "proxy_reasons", "game_number_fallbacks",
                                              "lookup_2026_missing_probable_sides")}, indent=1))
        return 0
    if args.cmd == "run":
        return run(args.seed, args.data_dir, args.inputs, args.out)
    agg = aggregate(args.data_dir, args.inputs, args.out)
    print(json.dumps(agg["A_posted"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
