"""C2 side item (e): the catcher-grouped framing screen, stage one (design note
`docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md`; register row C2-framing-side-item, Eric 2026-10-06).

**What it measures.** Production's framing proxy `pitcher_catcher_framing` is the borderline called-strike rate grouped
by pitcher (`bts.features.compute`: shift(1), expanding, min_periods 5, per (pitcher, date)). This screen regroups the
same PA-level measure (`pa_borderline_csr`, untouched) by the game's catcher (`fielding_catcher_id`):
- **variant A** replaces `pitcher_catcher_framing` with `catcher_framing` in the base feature set;
- **variant B** adds `catcher_framing` alongside it;
- each 12-model blend config keeps its Statcast extras, as `bts.experiment.runner` rewrites them.

**How.** One claimed run per seed (`run --seed S`): the pinned 2017–2025 PA parquets (2026 is never read), the
production features, a self-check that the regrouping code reproduces production's pitcher feature exactly, then
`blend_walk_forward` for baseline, A and B on 2024 and 2025 on the estimated-PA basis, each season's profiles saved,
CPU time recorded per walk-forward. The first walk-forward stops the run if it costs more than
`FIRST_UNIT_STOP_CPU_H` (Eric: stop and report if the measured cost is well above the estimate). `aggregate` applies
the pre-registered dispositions across the stage-one seeds.

Research code: it changes nothing in production.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SEASONS_IN = tuple(range(2017, 2026))            # the inputs: 2017–2025 PA parquets (2026 excluded)
TEST_SEASONS = (2024, 2025)
STAGE_ONE_SEEDS = (2273360, 260991262, 1746737973)   # canonical-n10 positions 0, 1, 2 (data/seed_sets)
RETRAIN_EVERY = 7
BASIS = "estimated_pa"
FIRST_UNIT_STOP_CPU_H = 7.5                      # 1.5 x the lead's unmeasured ~5 CPU-h per season walk-forward
PRACTICAL_MIN = 0.003                            # +0.3pp P@1, the June screens' practical threshold
T_MIN = 1.5                                      # the repo's multi-seed keep rule (scripts/aggregate_seed_corpora.py)
NEW_COL = "catcher_framing"
OLD_COL = "pitcher_catcher_framing"
ADMISSION_REL = "scripts/audit/c2_framing/admission.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
DESIGN = "docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md"
EXPOSURE_ROW = "X-35"
SCOPE = "catcher framing screen stage one"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c2_framing",
           "src/bts", "pyproject.toml", "uv.lock", DESIGN)


# ---------------------------------------------------------------- the feature

def framing_by(df, key: str, out_col: str):
    """The production framing computation (`bts.features.compute`, the pitcher block) with the grouping key as a
    parameter: per (key, date) mean of `pa_borderline_csr`, then per key shift(1).expanding(min_periods=5).mean(),
    merged back on (key, date). Rows whose key is missing get NaN (groupby drops missing keys)."""
    date_framing = df.groupby([key, "date"])["pa_borderline_csr"].mean().reset_index()
    date_framing.columns = [key, "date", "date_csr"]
    date_framing = date_framing.sort_values([key, "date"])
    date_framing[out_col] = date_framing.groupby(key)["date_csr"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )
    return df.merge(
        date_framing[[key, "date", out_col]].drop_duplicates(subset=[key, "date"]),
        on=[key, "date"], how="left",
    )


def self_check(df) -> dict:
    """The regrouping code must reproduce production's pitcher feature exactly (NaN equal, same rows and order)."""
    import numpy as np
    chk = framing_by(df, "pitcher_id", "_self_check")
    a, b = chk["_self_check"].to_numpy(dtype=float), df[OLD_COL].to_numpy(dtype=float)
    same = len(chk) == len(df) and bool(np.array_equal(a, b, equal_nan=True))
    return {"rows": int(len(df)), "identical": same,
            "max_abs_diff": None if same or len(a) != len(b) else float(np.nanmax(np.abs(a - b)))}


def coverage(df, col: str) -> dict:
    out = {}
    for s in TEST_SEASONS:
        part = df.loc[df["season"] == s, col]
        out[str(s)] = {"rows": int(len(part)), "non_null": int(part.notna().sum())}
    return out


# ---------------------------------------------------------------- the variants

def base_cols(variant: str) -> list[str]:
    from bts.features.compute import FEATURE_COLS
    if variant == "baseline":
        return list(FEATURE_COLS)
    if variant == "A":
        assert FEATURE_COLS.count(OLD_COL) == 1
        return [NEW_COL if c == OLD_COL else c for c in FEATURE_COLS]
    if variant == "B":
        return list(FEATURE_COLS) + [NEW_COL]
    raise ValueError(variant)


def blend_configs(variant: str) -> list:
    """BLEND_CONFIGS with each config's base replaced and its extras kept (`bts.experiment.runner`'s rule)."""
    from bts.features.compute import FEATURE_COLS
    from bts.model.predict import BLEND_CONFIGS
    base = base_cols(variant)
    out = []
    for config in BLEND_CONFIGS:
        name, cols = config[0], config[1]
        extras = [c for c in cols if c not in FEATURE_COLS]
        out.append((name, base + extras, config[2]) if len(config) == 3 else (name, base + extras))
    return out


# ---------------------------------------------------------------- the dispositions (pre-registered)

def seed_summary(diff: dict) -> dict:
    """One seed's per-season P@1 deltas and the repo's screening rule on that seed."""
    from bts.experiment.runner import evaluate_pass_fail
    by = diff.get("p_at_1_by_season", {})
    deltas = {str(k): float(v["delta"]) for k, v in by.items()}
    passed, reason = evaluate_pass_fail(diff)
    return {"p_at_1_delta": deltas, "passed": bool(passed), "reason": reason}


def _t(values: list[float]) -> float:
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    if sd == 0.0:
        return math.inf if m > 0 else (-math.inf if m < 0 else 0.0)
    return m / (sd / math.sqrt(len(values)))


def disposition(per_seed: list[dict]) -> dict:
    """Stage one for one variant, over its seeds' summaries (see the design note §5)."""
    seasons = [str(s) for s in TEST_SEASONS]
    if not per_seed or any(set(p["p_at_1_delta"]) != set(seasons) for p in per_seed):
        return {"disposition": "incomplete"}
    mean_by_season = {s: statistics.fmean(p["p_at_1_delta"][s] for p in per_seed) for s in seasons}
    seed_level = [statistics.fmean(p["p_at_1_delta"][s] for s in seasons) for p in per_seed]
    m, t = statistics.fmean(seed_level), _t(seed_level)
    passes = sum(p["passed"] for p in per_seed)
    majority = len(per_seed) // 2 + 1
    if all(v > 0 for v in mean_by_season.values()) and m >= PRACTICAL_MIN and t >= T_MIN and passes >= majority:
        verdict = "positive"
    elif m <= 0 and passes < majority:
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {"disposition": verdict, "mean_p_at_1_delta": mean_by_season, "seed_level_mean": m,
            "seed_level": seed_level, "t": t if math.isfinite(t) else str(t), "per_seed_passes": passes,
            "n_seeds": len(per_seed)}


# ---------------------------------------------------------------- CPU accounting

def cpu_seconds() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def first_unit_stop(cpu_s: float) -> bool:
    return cpu_s > FIRST_UNIT_STOP_CPU_H * 3600


# ---------------------------------------------------------------- the admitted run

def _json(path: Path, obj) -> None:
    from scripts.audit.c1 import admission as A
    A.durable_write(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def pins_digest(pins: dict) -> str:
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def admission_gate():
    from scripts.audit.c1 import admission as A
    path = A.REPO / ADMISSION_REL
    if not path.is_file():
        raise SystemExit(f"refusing: no admission record at {ADMISSION_REL}")
    raw = path.read_bytes()
    adm = json.loads(raw)
    pins = adm.get("input_pins")
    want = {f"pa_{s}.parquet" for s in SEASONS_IN}
    if not (isinstance(pins, dict) and set(pins) == want
            and all(isinstance(v, str) and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)
                    for v in pins.values())):
        raise SystemExit(f"refusing: admission input_pins must give a sha256 for exactly {sorted(want)}")
    head, reasons = A.admission_check(A.REPO, adm, closure=CLOSURE, admission_rel=ADMISSION_REL,
                                      register_rel=REGISTER_REL, exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE,
                                      inputs_digest=pins_digest(pins))
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**A.accepted_identity(A.REPO, adm), "admission_sha256": hashlib.sha256(raw).hexdigest()}


def load_inputs(data_dir: Path, pins: dict):
    """Read each pinned parquet once from the hashed bytes; nothing else in the directory is read."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    frames = []
    for s in SEASONS_IN:
        raw, _ = A.read_pinned(data_dir / f"pa_{s}.parquet", pins[f"pa_{s}.parquet"])
        frames.append(pd.read_parquet(io.BytesIO(raw)))
    return pd.concat(frames, ignore_index=True)


def run(seed: int, out_root: Path, data_dir: Path, *, walk_forward=None, now=None) -> int:
    import pandas as pd
    from scripts.audit.c1 import admission as A
    if os.environ.get("BTS_LGBM_DETERMINISTIC") != "1":
        raise SystemExit("refusing: BTS_LGBM_DETERMINISTIC=1 is pre-registered (set before bts is imported)")
    if seed not in STAGE_ONE_SEEDS:
        raise SystemExit(f"refusing: {seed} is not a stage-one seed {STAGE_ONE_SEEDS}")
    head, adm, identity = admission_gate()
    foreign = A.foreign_imports()
    if foreign:
        raise SystemExit(f"refusing: modules from outside this checkout: {foreign}")
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
    root = out_root / f"seed_{seed}"
    with A.admission_lock(root):
        register = (A.REPO / REGISTER_REL).read_text()
        if A.claimed_runs(root, register):
            raise SystemExit(f"refusing: a claimed run already exists under {root}")
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        run_dir = A.make_run_dir(root, f"{head[:7]}-{stamp}")
        claim_sha = A.write_claim(run_dir, head)
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    walk_forward = walk_forward or blend_walk_forward
    t0 = cpu_seconds()
    df = compute_all_features(load_inputs(data_dir, adm["input_pins"]))
    check = self_check(df)
    df = framing_by(df, "fielding_catcher_id", NEW_COL)
    manifest = {"schema": "c2_framing_screen_run_v1", "head": head, "claim_sha256": claim_sha, "seed": seed,
                "identity": identity, "input_pins": adm["input_pins"], "test_seasons": list(TEST_SEASONS),
                "basis": BASIS, "retrain_every": RETRAIN_EVERY, "env": {
                    k: os.environ.get(k) for k in ("BTS_LGBM_DETERMINISTIC", "BTS_LGBM_RANDOM_STATE", "TZ")},
                "self_check": check, "coverage": {NEW_COL: coverage(df, NEW_COL), OLD_COL: coverage(df, OLD_COL)},
                "features_cpu_s": cpu_seconds() - t0}
    _json(run_dir / "manifest.json", manifest)
    if not check["identical"]:
        _json(run_dir / "STOPPED.json", {"reason": "self_check", "self_check": check})
        return 4
    units, profiles = [], {}
    for variant in ("baseline", "A", "B"):
        configs = blend_configs(variant)
        parts = []
        for season in TEST_SEASONS:
            c0, w0 = cpu_seconds(), time.monotonic()
            p = walk_forward(df, season, retrain_every=RETRAIN_EVERY, blend_configs=configs,
                             game_probability_mode=BASIS)
            p["season"] = season
            cpu = cpu_seconds() - c0
            p.to_parquet(run_dir / f"profiles_{variant}_{season}.parquet", index=False)
            units.append({"variant": variant, "season": season, "cpu_s": cpu, "wall_s": time.monotonic() - w0})
            _json(run_dir / "units.json", units)
            parts.append(p)
            if len(units) == 1 and first_unit_stop(cpu):
                _json(run_dir / "STOPPED.json", {"reason": "first_unit_cpu", "cpu_s": cpu,
                                                 "limit_cpu_h": FIRST_UNIT_STOP_CPU_H})
                return 3
        profiles[variant] = pd.concat(parts, ignore_index=True)
    cards = {v: compute_full_scorecard(p) for v, p in profiles.items()}
    results = {"seed": seed, "units": units, "total_cpu_s": cpu_seconds() - t0,
               "p_at_1_by_season": {v: {str(k): x for k, x in c["p_at_1_by_season"].items()} for v, c in cards.items()},
               "variants": {}}
    for v in ("A", "B"):
        diff = diff_scorecards(cards["baseline"], cards[v])
        results["variants"][v] = {**seed_summary(diff), "secondary": {
            "p_57_exact": diff.get("p_57_exact"), "mean_max_streak": diff.get("streak_metrics", {}).get(
                "mean_max_streak")}}
    _json(run_dir / "results.json", results)
    return 0


def aggregate(run_dirs: list[Path]) -> dict:
    per_variant = {"A": [], "B": []}
    seeds, cpu = [], 0.0
    for d in run_dirs:
        r = json.loads((d / "results.json").read_text())
        seeds.append(r["seed"])
        cpu += r["total_cpu_s"]
        for v in per_variant:
            per_variant[v].append(r["variants"][v])
    return {"seeds": seeds, "total_cpu_h": cpu / 3600,
            "variants": {v: {"per_seed": rows, **disposition(rows)} for v, rows in per_variant.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.audit.c2_framing.screen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out-root", type=Path, required=True)
    r.add_argument("--data-dir", type=Path, required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.seed, a.out_root, a.data_dir)
    print(json.dumps(aggregate(a.run_dirs), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
