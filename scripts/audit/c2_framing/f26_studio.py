"""The 2026 framing test, run off the box (exposure row X-37, 2026-11-12).

The science is f26.py's, unchanged: the starter-proxy table, the as-of catcher value, the arm transform, the
walk-forward hook and the §6 dispositions. This module replaces only f26.py's C1 launcher path (admission gate,
ledger, reservation, guarded unit, launch window), which needs the deleted box and is superseded by the work
policies of 2026-11-12. What it keeps from that path: every input is read from bytes whose sha256 is recorded in
PREPARED.json; a seed runs once; the aggregate refuses anything but the ten registered seeds, each complete.

    prepare    --data-dir D --raw-dir R --lookup L --inputs I     once: table, 2017-2026 lookup, sources, pa_2026 copy
    run        --seed N --data-dir D --inputs I --out O           one seed, three arms
    aggregate  --inputs I --out O                                 after all ten seeds: dispositions
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

def seed_rank1(out: Path, seed: int, pins: dict) -> dict:
    """One finished seed's rank-1 hits per arm, re-derived from its retained profiles (not trusted from results.json)
    and required to equal what the run recorded."""
    import pandas as pd
    d = out / f"seed_{seed}"
    if (d / "STOPPED.json").exists() or not (d / "results.json").exists():
        raise F.RunInvalid(f"seed {seed} is stopped or unfinished")
    man, res = json.loads((d / "manifest.json").read_bytes()), json.loads((d / "results.json").read_bytes())
    if man["seed"] != seed or res["seed"] != seed or man["input_pins"] != pins:
        raise F.RunInvalid(f"seed {seed}: the run's seed or input pins are not this test's")
    calendar = man["calendar"]
    got = {}
    for arm in F.ARMS:
        hits = F.rank1(pd.read_parquet(d / f"profiles_{arm}_{F.TEST_SEASON}.parquet"), calendar)
        if hits != res["rank1"][arm]:
            raise F.RunInvalid(f"seed {seed} {arm}: retained profiles do not reproduce the recorded rank-1 hits")
        got[arm] = hits
    return {"calendar": calendar, "rank1": got, "units": res["units"], "arms": res["arms"]}


def aggregate(inputs: Path, out: Path) -> dict:
    """Design §6. Refuses unless all ten registered seeds are complete on one identical calendar; A-posted decides,
    A-projected is described on the same calendar."""
    import numpy as np
    pins = prepared(inputs)["pins"]
    seeds = [seed_rank1(out, s, pins) for s in F.SEEDS]             # §5's literal order
    calendar = seeds[0]["calendar"]
    if not calendar or any(s["calendar"] != calendar for s in seeds):
        raise F.RunInvalid("the seeds do not share one nonempty evaluation calendar")

    def deltas(arm):
        return np.array([np.array(s["rank1"][arm]) - np.array(s["rank1"]["baseline"]) for s in seeds], dtype=float)
    agg = {"schema": SCHEMA, "seeds": list(F.SEEDS), "n_days": len(calendar),
           "calendar": [calendar[0], calendar[-1]], "calendar_sha256": _sha(F.canonical(calendar)),
           "A_posted": F.dispose(deltas("A_posted")),
           "A_projected_descriptive": {k: v for k, v in F.dispose(deltas("A_projected")).items() if k != "disposition"},
           "baseline_rank1_hit_rate_by_seed": [float(np.mean(s["rank1"]["baseline"])) for s in seeds],
           "catcher_coverage": {a: [next(u["catcher"]["counts"] for u in s["units"] if u["arm"] == a) for s in seeds]
                                for a in F.ARMS[1:]},
           "streak_metrics_by_seed": [s["arms"] for s in seeds],
           "total_cpu_h": sum(u["cpu_s"] for s in seeds for u in s["units"]) / 3600}
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
    agg = aggregate(args.inputs, args.out)
    print(json.dumps(agg["A_posted"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
