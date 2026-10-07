"""C2 side item (e): the catcher-grouped framing screen, stage one (design note
`docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md`; register rows C2-framing-side-item and C2-framing-seed-gate,
Eric 2026-10-06; review r1 BLOCK B1-B7 answered in revision 2).

**What it measures.** Production's framing proxy `pitcher_catcher_framing` is the borderline called-strike rate grouped
by pitcher (`bts.features.compute`: per (pitcher, date) mean, then shift(1).expanding(min_periods=5).mean()). This
screen regroups the same PA-level measure (`pa_borderline_csr`, untouched) by the game's catcher (`fielding_catcher_id`):
variant A replaces `pitcher_catcher_framing` with `catcher_framing`; variant B adds it alongside; each 12-model blend
config keeps its Statcast extras, as `bts.experiment.runner` rewrites them.

**Closed inputs (r1 B1, B6).** The run reads only the pinned files: the nine 2017-2025 PA parquets and a frozen copy of
production's probable-pitcher lookup restricted to their games (`freeze-lookup`, run once before admission). Inside the
run, `compute_all_features`' live lookup builder is replaced by that frozen lookup (no scan of `data/raw`, no cache
write) and the park-drag attach is replaced by its table-absent result (the column is in no registered blend). The
import-time feature settings must equal the registered values. 2026 is never read.

**Labels (r1 B4).** Each profile's `actual_hit` is rebuilt from the pinned PA rows with the resumed portion excluded
(`bts.data.build.filter_out_resumed_portion`, the BTS scoring rule); a row whose batter/game has no original-portion PA
is void: dropped, and that day re-ranked. Training and features keep every PA, as production does.

**One run per seed, in a canonical namespace (r1 B3).** Claims live only under `OUT_ROOT/seed_<seed>`; the command line
has no output-root option. Seed 1 runs first; seeds 2-3 run only after Eric's recorded release (register row
`C2-framing-release-seeds-2-3`) and seed 1's completed run. `launch` is the reviewed wrapper that builds the launcher
command (name, budget, deterministic flag) under the same checks.

**Aggregation (r1 B2).** `aggregate` gives a stage-one disposition only for exactly the three registered seeds, each a
distinct, completed, claimed run of the same reviewed code and inputs; each seed's summary is recomputed from its
retained diff, never taken from the caller.

Research code: it changes nothing in production.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import resource
import statistics
import subprocess
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
SEED1_BUDGET_CPU_H = 45.0                        # seed 1's declared launcher budget (Eric's stage-one approval)
MAX_WALL_H = 16
SETTINGS = {"ROOKIE_GATE_K": 20, "PITCHER_HR_30G_MIN_PERIODS": 7}   # production's effective values (box .env, 10/06)
SCORING = {"mc_trials": 10_000, "season_length": 180}   # compute_full_scorecard's registered settings (its defaults)
NEW_COL = "catcher_framing"
OLD_COL = "pitcher_catcher_framing"
LOOKUP_NAME = "probable_pitcher_lookup.2017-2025.json"
OUT_ROOT = Path.home() / "projects" / "bts" / "data" / "hetzner_results" / "c2" / "framing_screen"
ADMISSION_REL = "scripts/audit/c2_framing/admission.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
DESIGN = "docs/sota_audit/2026-10-06-prereg-c2-framing-screen.md"
EXPOSURE_ROW = "X-35"
RELEASE_ROW = "C2-framing-release-seeds-2-3"
SCOPE = "catcher framing screen stage one"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c2_framing",
           "src/bts", "pyproject.toml", "uv.lock", DESIGN)
INPUT_NAMES = tuple(f"pa_{s}.parquet" for s in SEASONS_IN) + (LOOKUP_NAME,)


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


def resumed_counts(df) -> dict:
    """PA rows in the resumed portion of a suspended game, per season. They carry the original game's official date,
    so date-level shifting admits them to history from that date (the bounded availability limit, note §8)."""
    if "is_resumed_portion" not in df.columns:
        return {}
    r = df[df["is_resumed_portion"].fillna(False).astype(bool)]
    return {str(int(s)): int(n) for s, n in r.groupby("season").size().items()}


# ---------------------------------------------------------------- closed inputs

def freeze_lookup(cache_bytes: bytes, game_pks) -> bytes:
    """Production's probable-pitcher cache restricted to the given games, as canonical JSON bytes."""
    cache = json.loads(cache_bytes)
    keep = {str(k): cache[str(k)] for k in sorted({int(g) for g in game_pks}) if str(k) in cache}
    return (json.dumps(keep, sort_keys=True, separators=(",", ":")) + "\n").encode()


def frozen_lookup(raw: bytes) -> dict:
    """The run's lookup, keyed as `_build_probable_pitcher_lookup` keys its own ({int game_pk: entry})."""
    return {int(k): v for k, v in json.loads(raw).items()}


def check_settings() -> dict:
    import bts.features.compute as C
    got = {k: getattr(C, k) for k in SETTINGS}
    if got != SETTINGS:
        raise SystemExit(f"refusing: feature settings {got} are not the registered {SETTINGS}")
    return got


def install_closed_inputs(lookup: dict) -> None:
    """Replace the two ancillary reads of `compute_all_features` for this process (r1 B1)."""
    import numpy as np
    import bts.features.compute as C
    import bts.features.park_drag as PD

    def _lookup(raw_dir: str = "data/raw") -> dict:
        return dict(lookup)

    def _no_park_drag(df, table=None):
        out = df.copy()
        out["park_drag_delta"] = np.nan
        return out
    C._build_probable_pitcher_lookup = _lookup
    PD.attach_park_drag = _no_park_drag


# ---------------------------------------------------------------- labels (BTS scoring rule)

def original_portion_labels(df):
    """(batter_id, game_pk) -> 1 if any original-portion PA was a hit, else 0."""
    from bts.data.build import filter_out_resumed_portion
    orig = filter_out_resumed_portion(df)
    return orig.groupby(["batter_id", "game_pk"])["is_hit"].max().astype(int)


def relabel(profiles, labels):
    """Rebuild `actual_hit` from original-portion PAs; void rows (no original-portion PA) are dropped and each day's
    ranks renumbered in their original order. Returns (profiles, counts)."""
    keys = list(zip(profiles["batter_id"].astype(int), profiles["game_pk"].astype(int)))
    new = [labels.get(k) for k in keys]
    p = profiles.copy()
    changed = sum(1 for old, n in zip(p["actual_hit"], new) if n is not None and int(old) != int(n))
    p["actual_hit"] = new
    void = int(p["actual_hit"].isna().sum())
    p = p[p["actual_hit"].notna()].copy()
    p["actual_hit"] = p["actual_hit"].astype(int)
    p = p.sort_values(["date", "rank"], kind="stable")
    p["rank"] = p.groupby("date").cumcount() + 1
    return p.reset_index(drop=True), {"changed": int(changed), "void_dropped": void}


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
    """m / (sd / sqrt(n)); with sd = 0: +inf if m > 0, -inf if m < 0, and 0 if m = 0."""
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    if sd == 0.0:
        return math.inf if m > 0 else (-math.inf if m < 0 else 0.0)
    return m / (sd / math.sqrt(len(values)))


def disposition(per_seed: list[dict]) -> dict:
    """Stage one for one variant, over exactly the three registered seeds' summaries (note §5)."""
    seasons = [str(s) for s in TEST_SEASONS]
    if len(per_seed) != len(STAGE_ONE_SEEDS) or any(set(p["p_at_1_delta"]) != set(seasons) for p in per_seed):
        return {"disposition": "incomplete", "n_seeds": len(per_seed)}
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


# ---------------------------------------------------------------- admission, seed order and release

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    if not (isinstance(pins, dict) and set(pins) == set(INPUT_NAMES)
            and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values())):
        raise SystemExit(f"refusing: admission input_pins must give a sha256 for exactly {sorted(INPUT_NAMES)}")
    head, reasons = A.admission_check(A.REPO, adm, closure=CLOSURE, admission_rel=ADMISSION_REL,
                                      register_rel=REGISTER_REL, exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE,
                                      inputs_digest=pins_digest(pins))
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**A.accepted_identity(A.REPO, adm), "admission_sha256": _sha(raw)}


RELEASE_RE = re.compile(r"^\*\*RULED (\d{4}-\d{2}-\d{2}) \(Eric\): RELEASE seeds 2–3 of the framing screen after "
                        r"seed-1 run `([0-9a-f]{7}-\d{8}T\d{6}Z)`; declared budget (\d+(?:\.\d+)?) CPU-hours per seed\*\*$")


def release(register_text: str) -> tuple[str, float] | None:
    """Eric's release of seeds 2-3 (register row RELEASE_ROW): the ruling cell must match exactly, naming the seed-1 run
    it follows; the source cell's first token must be exactly `Eric` (the shared gate's rule); the budget must be a
    finite positive number. Returns (seed-1 run name, per-seed declared budget), or None (r2 R2-2)."""
    from scripts.audit.c1 import admission as A
    cells = A.row_cells(register_text, RELEASE_ROW)
    if not cells or len(cells) < 4:
        return None
    m = RELEASE_RE.match(cells[2].strip())
    if not m or cells[3].split()[:1] != ["Eric"]:
        return None
    budget = float(m.group(3))
    if not (math.isfinite(budget) and budget > 0):
        return None
    return m.group(2), budget


def seed_allowed(seed: int, register_text: str, out_root: Path, *, identity: dict,
                 pins: dict) -> tuple[bool, str, float | None]:
    """Seed order (row C2-framing-seed-gate): seed 1 first; seeds 2-3 only after Eric's release, which must name seed
    1's run, and only when that run validates as a complete run of the same admitted code and inputs (r2 R2-2).
    Returns (allowed, reason, declared budget)."""
    if seed not in STAGE_ONE_SEEDS:
        return False, f"{seed} is not a stage-one seed {STAGE_ONE_SEEDS}", None
    if seed == STAGE_ONE_SEEDS[0]:
        return True, "seed 1", SEED1_BUDGET_CPU_H
    rel = release(register_text)
    if rel is None:
        return False, f"seeds 2-3 need Eric's release (register row {RELEASE_ROW})", None
    run_name, budget = rel
    root = out_root / f"seed_{STAGE_ONE_SEEDS[0]}"
    runs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if [d.name for d in runs] != [run_name]:
        return False, f"seeds 2-3 need exactly seed 1's released run {run_name} under {root}", None
    try:
        validate_run(runs[0], STAGE_ONE_SEEDS[0], out_root=out_root, identity=identity, pins=pins)
    except RunInvalid as e:
        return False, f"seed 1's run is not a complete admitted run: {e}", None
    return True, "released", budget


def load_inputs(data_dir: Path, pins: dict):
    """Read each pinned parquet once from the hashed bytes; nothing else in the directory is read."""
    import pandas as pd
    from scripts.audit.c1 import admission as A
    frames = []
    for s in SEASONS_IN:
        raw, _ = A.read_pinned(data_dir / f"pa_{s}.parquet", pins[f"pa_{s}.parquet"])
        frames.append(pd.read_parquet(io.BytesIO(raw)))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- the admitted run

def run(seed: int, data_dir: Path, inputs_dir: Path, *, walk_forward=None, now=None, _test_out_root=None) -> int:
    import pandas as pd
    from scripts.audit.c1 import admission as A
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    if os.environ.get("BTS_LGBM_DETERMINISTIC") != "1":
        raise SystemExit("refusing: BTS_LGBM_DETERMINISTIC=1 is pre-registered (set before bts is imported)")
    if seed not in STAGE_ONE_SEEDS:
        raise SystemExit(f"refusing: {seed} is not a stage-one seed {STAGE_ONE_SEEDS}")
    head, adm, identity = admission_gate()
    foreign = A.foreign_imports()
    if foreign:
        raise SystemExit(f"refusing: modules from outside this checkout: {foreign}")
    register = (A.REPO / REGISTER_REL).read_text()
    ok, why, _ = seed_allowed(seed, register, out_root, identity=identity, pins=adm["input_pins"])
    if not ok:
        raise SystemExit(f"refusing: {why}")
    from bts.model.predict import LGB_PARAMS
    if not (LGB_PARAMS.get("deterministic") is True and LGB_PARAMS.get("force_row_wise") is True):
        raise SystemExit("refusing: LightGBM's params were built without the deterministic flags")
    settings = check_settings()
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
    root = out_root / f"seed_{seed}"
    with A.admission_lock(root):
        if A.claimed_runs(root, register):
            raise SystemExit(f"refusing: a claimed run already exists under {root}")
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        run_dir = A.make_run_dir(root, f"{head[:7]}-{stamp}")
        claim_sha = A.write_claim(run_dir, head)
    lookup_raw, _ = A.read_pinned(inputs_dir / LOOKUP_NAME, adm["input_pins"][LOOKUP_NAME])
    install_closed_inputs(frozen_lookup(lookup_raw))
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    walk_forward = walk_forward or blend_walk_forward
    t0 = cpu_seconds()
    df = compute_all_features(load_inputs(data_dir, adm["input_pins"]))
    check = self_check(df)
    df = framing_by(df, "fielding_catcher_id", NEW_COL)
    labels = original_portion_labels(df)
    manifest = {"schema": "c2_framing_screen_run_v2", "head": head, "claim_sha256": claim_sha, "seed": seed,
                "identity": identity, "input_pins": adm["input_pins"], "inputs_digest": pins_digest(adm["input_pins"]),
                "test_seasons": list(TEST_SEASONS), "basis": BASIS, "retrain_every": RETRAIN_EVERY,
                "lgb_params": dict(LGB_PARAMS), "feature_settings": settings, "scoring": dict(SCORING),
                "env": {k: os.environ.get(k) for k in ("BTS_LGBM_DETERMINISTIC", "BTS_LGBM_RANDOM_STATE", "TZ")},
                "self_check": check, "coverage": {NEW_COL: coverage(df, NEW_COL), OLD_COL: coverage(df, OLD_COL)},
                "resumed_portion_rows": resumed_counts(df), "lookup_games": len(frozen_lookup(lookup_raw)),
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
            cpu = cpu_seconds() - c0
            p, counts = relabel(p, labels)
            p["season"] = season
            path = run_dir / f"profiles_{variant}_{season}.parquet"
            p.to_parquet(path, index=False)
            units.append({"variant": variant, "season": season, "cpu_s": cpu, "wall_s": time.monotonic() - w0,
                          "labels": counts})
            _json(run_dir / "units.json", units)
            parts.append(pd.read_parquet(path))    # score exactly the retained bytes validate_run rescores (r3 R3-1)
            if len(units) == 1 and first_unit_stop(cpu):
                _json(run_dir / "STOPPED.json", {"reason": "first_unit_cpu", "cpu_s": cpu,
                                                 "limit_cpu_h": FIRST_UNIT_STOP_CPU_H})
                return 3
        profiles[variant] = pd.concat(parts, ignore_index=True)
    cards = {v: compute_full_scorecard(p, **SCORING) for v, p in profiles.items()}
    for v, c in cards.items():
        _json(run_dir / f"scorecard_{v}.json", c)
    results = {"seed": seed, "head": head, "units": units, "total_cpu_s": cpu_seconds() - t0,
               "p_at_1_by_season": {v: {str(k): x for k, x in c["p_at_1_by_season"].items()} for v, c in cards.items()},
               "variants": {}}
    for v in ("A", "B"):
        diff = diff_scorecards(cards["baseline"], cards[v])
        _json(run_dir / f"diff_{v}.json", diff)
        results["variants"][v] = {**seed_summary(diff), "secondary": {
            "p_57_exact": diff.get("p_57_exact"), "mean_max_streak": diff.get("streak_metrics", {}).get(
                "mean_max_streak")}}
    _json(run_dir / "results.json", results)
    return 0


# ---------------------------------------------------------------- the aggregate (exactly the three registered seeds)

class RunInvalid(RuntimeError):
    pass


AggregateError = RunInvalid                     # the aggregate's refusals are run-validation refusals
UNIT_ORDER = [(v, s) for v in ("baseline", "A", "B") for s in TEST_SEASONS]
IDENTITY_KEYS = ("review_report", "review_report_sha256", "reviewed_commit", "exposure_commit", "admission_sha256")
HEX = re.compile(r"[0-9a-f]{64}")
HEAD = re.compile(r"[0-9a-f]{40}")


def _read(d: Path, name: str) -> bytes:
    p = d / name
    if not p.is_file():
        raise RunInvalid(f"{d}: missing {name}")
    return p.read_bytes()


def _json_of(d: Path, name: str):
    try:
        return json.loads(_read(d, name))
    except RunInvalid:
        raise
    except Exception as e:
        raise RunInvalid(f"{d}: {name} is not valid JSON ({type(e).__name__})")


def head_admitted(repo: Path, identity: dict, head: str) -> list[str]:
    """Problems with a run's retained HEAD as admitted code (r3 R3-3): it must be a commit in `repo` that descends from
    the admitted exposure commit, and no executable-closure file other than the admission record may differ from the
    reviewed commit. That is the shared admission gate's rule, applied to the commit the run actually recorded."""
    from scripts.audit.c1 import admission as A
    rc, xc = identity.get("reviewed_commit"), identity.get("exposure_commit")
    if not all(isinstance(c, str) and HEAD.fullmatch(c) for c in (rc, xc, head)):
        return [f"run HEAD {str(head)[:7]} is not admitted: the head and the admitted commits must be full commit ids"]
    if A._git(repo, "cat-file", "-e", f"{head}^{{commit}}", check=False).returncode != 0:
        return [f"run HEAD {head[:7]} is not admitted: not a commit in this repository"]
    if not A._ancestor(repo, xc, head):
        return [f"run HEAD {head[:7]} is not admitted: the exposure commit is not its ancestor"]
    changed = [f for f in A._git(repo, "diff", "--name-only", rc, head, "--", *CLOSURE).stdout.split()
               if f != ADMISSION_REL]
    return [f"run HEAD {head[:7]} is not admitted: executable files differ from the reviewed commit: {changed[:5]}"] \
        if changed else []


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def validate_run(d: Path, seed: int, *, out_root: Path, identity: dict, pins: dict, repo: Path | None = None) -> dict:
    """One completed, claimed stage-one run, checked semantically against trusted admission evidence (r2 R2-2, R2-3;
    r3 R3-1 to R3-3). `identity` and `pins` come from the admission gate, never from the run. Raises RunInvalid; returns
    {manifest, results, summaries}. It checks:
    - the canonical claim namespace (`out_root/seed_<seed>/<run>`), no stop, and a JSON claim naming this run and the
      manifest's code;
    - the manifest: schema, seed, claim binding, a 40-hex head, the registered basis, retrain interval, test seasons,
      feature settings, scoring settings, deterministic LightGBM params and environment, the ten pins and their digest,
      an identical self-check, and exactly the admitted identity and pins;
    - the run's HEAD: an admitted descendant of the exposure commit with the reviewed executable closure
      (`head_admitted`);
    - the results: seed, head, and the six units in their registered order;
    - every retained profile is non-empty evidence of its own unit's season (every row's season and date year);
    - each variant's full scorecard recomputed from its retained profiles with the registered scoring settings equals
      the stored scorecard (every decision-bearing value: P@1, exact P(57), the streak metrics), with P@1 for exactly
      the registered test seasons; the results' P@1 equals it; each diff recomputed from the retained scorecards equals
      the stored diff; and each stored summary, with its secondary values, recomputed from its diff matches."""
    import pandas as pd
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
    from scripts.audit.c1 import admission as A
    repo = A.REPO if repo is None else repo
    d = Path(d)
    if Path(d).resolve().parent != (out_root / f"seed_{seed}").resolve():
        raise RunInvalid(f"{d}: not in the canonical claim namespace {out_root / f'seed_{seed}'}")
    if (d / "STOPPED.json").exists():
        raise RunInvalid(f"{d}: a stopped run is not a complete seed")
    claim_bytes = _read(d, "CLAIM.json")
    claim = _json_of(d, "CLAIM.json")
    man = _json_of(d, "manifest.json")
    res = _json_of(d, "results.json")
    if not (isinstance(claim, dict) and claim.get("run") == d.name and isinstance(man, dict)
            and claim.get("code") == man.get("head")):
        raise RunInvalid(f"{d}: the claim does not name this run and its code")
    pins_m = man.get("input_pins")
    lgb = man.get("lgb_params") or {}
    ident = man.get("identity")
    env = man.get("env") or {}
    checks = {
        "schema": man.get("schema") == "c2_framing_screen_run_v2",
        "seed": man.get("seed") == seed,
        "claim binding": man.get("claim_sha256") == _sha(claim_bytes),
        "head": isinstance(man.get("head"), str) and bool(HEAD.fullmatch(man["head"])),
        "basis": man.get("basis") == BASIS,
        "retrain": man.get("retrain_every") == RETRAIN_EVERY,
        "seasons": man.get("test_seasons") == list(TEST_SEASONS),
        "settings": man.get("feature_settings") == SETTINGS,
        "scoring": man.get("scoring") == SCORING,
        "lgb determinism": lgb.get("deterministic") is True and lgb.get("force_row_wise") is True,
        "env": env.get("BTS_LGBM_DETERMINISTIC") == "1" and env.get("BTS_LGBM_RANDOM_STATE") == str(seed),
        "pins": isinstance(pins_m, dict) and set(pins_m) == set(INPUT_NAMES)
                and all(isinstance(v, str) and HEX.fullmatch(v) for v in pins_m.values()),
        "pins digest": isinstance(pins_m, dict) and man.get("inputs_digest") == pins_digest(pins_m),
        "identity": isinstance(ident, dict) and all(isinstance(ident.get(k), str) and ident.get(k)
                                                    for k in IDENTITY_KEYS),
        "self-check": (man.get("self_check") or {}).get("identical") is True,
        "admitted identity": isinstance(ident, dict) and {k: ident.get(k) for k in IDENTITY_KEYS}
                             == {k: identity.get(k) for k in IDENTITY_KEYS},
        "admitted pins": pins_m == pins,
        "results": isinstance(res, dict) and res.get("seed") == seed and res.get("head") == man.get("head"),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise RunInvalid(f"{d}: invalid manifest/results: {bad}")
    problems = head_admitted(repo, identity, man["head"])
    if problems:
        raise RunInvalid(f"{d}: " + "; ".join(problems))
    units = res.get("units")
    if not (isinstance(units, list) and [(u.get("variant"), u.get("season")) for u in units] == UNIT_ORDER):
        raise RunInvalid(f"{d}: the six registered units are not all complete")
    cards = {v: _json_of(d, f"scorecard_{v}.json") for v in ("baseline", "A", "B")}
    seasons = [str(s) for s in TEST_SEASONS]
    for v in ("baseline", "A", "B"):
        parts = []
        for s in TEST_SEASONS:
            name = f"profiles_{v}_{s}.parquet"
            _read(d, name)
            try:
                part = pd.read_parquet(d / name)
                years = pd.to_datetime(part["date"]).dt.year
                own = len(part) > 0 and bool((part["season"] == s).all()) and bool((years == s).all())
            except Exception as e:
                raise RunInvalid(f"{d}: {name} is unreadable ({type(e).__name__})")
            if not own:
                raise RunInvalid(f"{d}: {name} is not complete {s} evidence")
            parts.append(part)
        card = json.loads(_canon(compute_full_scorecard(pd.concat(parts, ignore_index=True), **SCORING)))
        if _canon({k: x for k, x in card.items() if k != "timestamp"}) != _canon(
                {k: x for k, x in cards[v].items() if k != "timestamp"}):
            raise RunInvalid(f"{d}: variant {v}'s scorecard does not match a recomputation from its retained profiles")
        if sorted(card.get("p_at_1_by_season") or {}) != seasons:
            raise RunInvalid(f"{d}: variant {v}'s P@1 does not cover exactly the test seasons {seasons}")
        if (res.get("p_at_1_by_season") or {}).get(v) != card["p_at_1_by_season"]:
            raise RunInvalid(f"{d}: variant {v}'s P@1 does not reconcile across profiles, scorecard and results")
    summaries = {}
    for v in ("A", "B"):
        stored = _json_of(d, f"diff_{v}.json")
        recomputed = json.loads(_canon(diff_scorecards(cards["baseline"], cards[v])))
        if _canon(stored) != _canon(recomputed):
            raise RunInvalid(f"{d}: variant {v}'s diff does not match its retained scorecards")
        summary = seed_summary(stored)
        secondary = {"p_57_exact": stored.get("p_57_exact"),
                     "mean_max_streak": stored.get("streak_metrics", {}).get("mean_max_streak")}
        held = (res.get("variants") or {}).get(v) or {}
        if _canon({k: held.get(k) for k in ("p_at_1_delta", "passed", "secondary")}) != _canon(
                {"p_at_1_delta": summary["p_at_1_delta"], "passed": summary["passed"], "secondary": secondary}):
            raise RunInvalid(f"{d}: variant {v}'s stored summary does not match its diff")
        summaries[v] = summary
    return {"manifest": man, "results": res, "summaries": summaries}


def aggregate(run_dirs: list[Path], *, _test_out_root=None) -> dict:
    """Stage one's dispositions, only from exactly three distinct runs of the three registered seeds, each validated by
    `validate_run` in the canonical namespace, all of the same admitted code (accepted identity), inputs and settings.
    Each run keeps its own commit: a metadata-only descendant such as Eric's committed release is admitted by the
    shared gate, so equal HEADs are not required (r2 R2-1)."""
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    dirs = [Path(d).resolve() for d in run_dirs]
    if len(dirs) != len(STAGE_ONE_SEEDS) or len(set(dirs)) != len(dirs):
        raise RunInvalid(f"need exactly {len(STAGE_ONE_SEEDS)} distinct run directories, got {len(run_dirs)}")
    seeds = []
    for d in dirs:
        m = re.fullmatch(r"seed_(\d+)", d.parent.name)
        seeds.append(int(m.group(1)) if m else None)
    if sorted(s for s in seeds if s is not None) != sorted(STAGE_ONE_SEEDS) or None in seeds:
        raise RunInvalid(f"seeds {seeds} are not exactly the registered {list(STAGE_ONE_SEEDS)}")
    _, adm, identity = admission_gate()               # trusted evidence: never the runs' own declarations (r3 R3-3)
    valid = [validate_run(d, s, out_root=out_root, identity=identity, pins=adm["input_pins"]) for d, s in zip(dirs, seeds)]
    first = valid[0]["manifest"]
    for key in ("identity", "input_pins", "inputs_digest", "lgb_params", "feature_settings", "basis", "retrain_every",
                "test_seasons"):
        if any(v["manifest"].get(key) != first.get(key) for v in valid[1:]):
            raise RunInvalid(f"runs disagree on {key}")
    order = [seeds.index(s) for s in STAGE_ONE_SEEDS]
    total = sum(v["results"]["total_cpu_s"] for v in valid)
    return {"seeds": list(STAGE_ONE_SEEDS), "heads": [valid[i]["manifest"]["head"] for i in order],
            "identity": first["identity"], "total_cpu_h": total / 3600,
            "variants": {var: {"per_seed": [valid[i]["summaries"][var] for i in order],
                               **disposition([valid[i]["summaries"][var] for i in order])} for var in ("A", "B")}}


# ---------------------------------------------------------------- the launch wrapper

def launch_command(seed: int, budget: float, data_dir: Path, inputs_dir: Path) -> list[str]:
    k = STAGE_ONE_SEEDS.index(seed) + 1
    return [".venv/bin/python", "-m", "scripts.audit.c1.launch", "run", "--name", f"c2-framing-seed{k}",
            "--cpu-hours", f"{budget:g}", "--max-hours", str(MAX_WALL_H), "--",
            "env", "BTS_LGBM_DETERMINISTIC=1", "TZ=America/New_York", ".venv/bin/python", "-m",
            "scripts.audit.c2_framing.screen", "run", "--seed", str(seed), "--data-dir", str(data_dir),
            "--inputs-dir", str(inputs_dir)]


def launch(seed: int, data_dir: Path, inputs_dir: Path, *, execute=subprocess.run, _test_out_root=None) -> int:
    """The reviewed launch: the same admission and seed-order checks as `run`, then the C1 launcher with the seed's
    declared budget (seed 1: 45; seeds 2-3: the budget in Eric's release row)."""
    from scripts.audit.c1 import admission as A
    out_root = OUT_ROOT if _test_out_root is None else _test_out_root
    _, adm, identity = admission_gate()
    ok, why, budget = seed_allowed(seed, (A.REPO / REGISTER_REL).read_text(), out_root, identity=identity,
                                   pins=adm.get("input_pins"))
    if not ok:
        raise SystemExit(f"refusing: {why}")
    return execute(launch_command(seed, budget, data_dir, inputs_dir), cwd=A.REPO).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.audit.c2_framing.screen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--inputs-dir", type=Path, default=OUT_ROOT / "inputs")
    f = sub.add_parser("freeze-lookup", help="once, before admission: the frozen lookup for the pinned games")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--data-dir", type=Path, required=True)
    f.add_argument("--pins", type=Path, required=True, help="JSON {pa_<season>.parquet: sha256} for the nine inputs")
    f.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.seed, a.data_dir, a.inputs_dir)
    if a.cmd == "launch":
        return launch(a.seed, a.data_dir, a.inputs_dir)
    if a.cmd == "freeze-lookup":
        import pandas as pd
        from scripts.audit.c1 import admission as A
        pins = json.loads(a.pins.read_text())
        pks = set()
        for s in SEASONS_IN:
            raw, _ = A.read_pinned(a.data_dir / f"pa_{s}.parquet", pins[f"pa_{s}.parquet"])
            pks |= set(pd.read_parquet(io.BytesIO(raw), columns=["game_pk"])["game_pk"].astype(int))
        out = freeze_lookup(a.cache.read_bytes(), pks)
        A.durable_write(a.out, out)
        print(json.dumps({"games": len(pks), "covered": len(json.loads(out)), "sha256": _sha(out)}))
        return 0
    print(json.dumps(aggregate(a.run_dirs), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
