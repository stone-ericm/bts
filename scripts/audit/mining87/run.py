"""#87 leaderboard mechanism-mining driver (season wrap W2.4).

Refuses to run until exposure row X-22 is published and ``registration.X22_COMMIT`` names its commit. Registered
mode uses only the frozen values in ``registration.REGISTRATION``; any override needs ``--exploratory``, which labels
the run and disables nomination. Inputs are hashed (and checked against ``--expect-inputs`` when given) before any
outcome-bearing loader runs. Nothing about outcomes is printed; outputs are written into ``<run>.partial`` and only
a complete run is renamed into place with ``COMPLETE.json`` (sha256 of every output and of the input manifest).

    python -m scripts.audit.mining87.run --data-root ~/projects/bts/data \\
        --ledger-dir ~/projects/bts/data/validation/season_2026_ledger/<accepted build> \\
        --out ~/projects/bts/data/validation/w24_mining87 [--surface-witness W.json] [--mechanism-records M.json] \\
        [--production-context C.parquet] [--expect-inputs <earlier run>/input_manifest.json]

Reads (read only): ``<data-root>/leaderboard/user_picks/*.parquet``, ``<data-root>/leaderboard/leaderboard_snapshots/
2026-07-04.parquet``, ``<data-root>/picks/slates/<date>.json`` in the window, the ledger build, and — only when a
witness file is supplied — ``<data-root>/leaderboard/static_snapshots/units`` for the unit→game binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.audit.mining87 import inference, registration, report, surfaces
from scripts.audit.mining87.consensus import cohort_stems, consensus_table, load_cohort, load_public_picks
from scripts.audit.mining87.production import (ACCEPTED_FILE, BUILD_FILE, CONTEST_FILE, CONTEXT_COLUMNS, LEDGER_FILE,
                                               attach_production_context, load_accepted_ledger, project_locked_slots)
from scripts.audit.mining87.registration import (REGISTRATION, REGISTRATION_FINGERPRINT, RegistrationError,
                                                 check_registration, document_hashes, resolve_params, x22_gate)
from scripts.audit.mining87.units import build_units
from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES

REPO = registration.REPO
CODE_FILES = [*sorted(str(p.relative_to(REPO)) for p in (REPO / "scripts/audit/mining87").glob("*.py")),
              "scripts/leaderboard_mechanism_mining.py", "scripts/leaderboard_candidate_join_audit.py",
              "scripts/leaderboard_backfilled_model_audit.py", "scripts/canonicalize_realized_picks.py",
              "scripts/audit/benchmark_bridge/core.py", "scripts/audit/season_ledger/compile.py",
              "scripts/audit/season_ledger/contest.py", "scripts/audit/season_ledger/sources/static.py",
              "src/bts/leaderboard/storage.py", "src/bts/validate/fdr.py"]
_DATE_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
OUTPUTS = ("report.json", "units.parquet", "cells_primary.parquet", "cells_tie_excluded_sensitivity.parquet",
           "consensus_votes.parquet", "input_manifest.json")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ledger-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--surface-witness", type=Path, default=None)
    ap.add_argument("--mechanism-records", type=Path, default=None)
    ap.add_argument("--production-context", type=Path, default=None)
    ap.add_argument("--expect-inputs", type=Path, default=None,
                    help="an earlier run's input_manifest.json: refuse unless every input is byte-identical")
    ex = ap.add_argument_group("exploratory overrides (labelled; cannot nominate)")
    ex.add_argument("--exploratory", action="store_true")
    ex.add_argument("--window-start", default=None)
    ex.add_argument("--window-end", default=None)
    ex.add_argument("--seed", type=int, default=None)
    ex.add_argument("--n-bootstrap", type=int, default=None)
    ex.add_argument("--expected-block-length", type=int, default=None)
    ex.add_argument("--cohort-snapshot-file", dest="snapshot_file", default=None)
    return ap.parse_args(argv)


def input_paths(args, params) -> dict:
    data = args.data_root.expanduser()
    ledger = args.ledger_dir.expanduser()
    lb = data / "leaderboard"
    snapshot = lb / "leaderboard_snapshots" / params.snapshot_file
    paths = {"ledger": {n: ledger / n for n in (ACCEPTED_FILE, LEDGER_FILE, CONTEST_FILE)},
             "cohort_snapshot": snapshot, "user_picks_dir": lb / "user_picks",
             "slates_dir": data / "picks" / "slates", "units_dir": lb / "static_snapshots" / "units"}
    for name, p in paths["ledger"].items():
        if not p.exists():
            raise SystemExit(f"ledger build file missing: {p}")
    if (ledger / BUILD_FILE).exists():                       # upstream manifest identity, hashed when present
        paths["ledger"][BUILD_FILE] = ledger / BUILD_FILE
    if not snapshot.exists():
        raise SystemExit(f"pinned cohort snapshot missing: {snapshot} (no fallback to another file)")
    if not paths["user_picks_dir"].is_dir():
        raise SystemExit(f"user_picks directory missing: {paths['user_picks_dir']}")
    for opt in ("surface_witness", "mechanism_records", "production_context", "expect_inputs"):
        p = getattr(args, opt)
        if p is not None and not p.exists():
            raise SystemExit(f"--{opt.replace('_', '-')} file missing: {p}")
    slates = {}
    if paths["slates_dir"].is_dir():
        slates = {p.stem: p for p in sorted(paths["slates_dir"].glob("*.json")) if _DATE_FILE.match(p.name)
                  and params.window_start <= p.stem <= params.window_end}
    paths["slates"] = slates
    return paths


def hash_inputs(paths: dict, args) -> dict:
    picks = {p.name: sha256_file(p) for p in sorted(paths["user_picks_dir"].glob("*.parquet"))}
    unit_files = None
    if args.surface_witness is not None:
        from scripts.audit.benchmark_bridge.core import capture_files
        unit_files = {p.name: sha256_file(p) for p in capture_files(paths["units_dir"])} \
            if paths["units_dir"].is_dir() else {}
    opt = lambda p: sha256_file(p) if p is not None else None  # noqa: E731
    return {"ledger": {n: sha256_file(p) for n, p in paths["ledger"].items()},
            "cohort_snapshot": {paths["cohort_snapshot"].name: sha256_file(paths["cohort_snapshot"])},
            "user_picks": picks,
            "user_picks_inventory_sha256": hashlib.sha256(json.dumps(picks, sort_keys=True).encode()).hexdigest(),
            "slates": {p.name: sha256_file(p) for p in paths["slates"].values()},
            "surface_witness": opt(args.surface_witness), "mechanism_records": opt(args.mechanism_records),
            "production_context": opt(args.production_context), "unit_captures": unit_files}


def code_identity() -> dict:
    files = {rel: sha256_file(REPO / rel) for rel in CODE_FILES if (REPO / rel).exists()}
    try:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *CODE_FILES],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    return {"files": files, "worktree_dirty": dirty}


def load_unit_games(units_dir: Path) -> dict[int, int]:
    """unit_id → game_pk where every capture names exactly one feedId (else the unit is unbound)."""
    from scripts.audit.benchmark_bridge.core import capture_files
    from scripts.audit.season_ledger.contest import units_lookup
    from scripts.audit.season_ledger.sources.static import parse_units
    rows = []
    for f in capture_files(units_dir):
        rows += parse_units(f"units/{f.name}", f.read_bytes()).rows
    return {uid: next(iter(e["feed_ids"])) for uid, e in units_lookup(rows).items() if len(e["feed_ids"]) == 1}


def admit_dates(locked: pd.DataFrame, slate_paths: dict, witnesses: dict) -> tuple[list[dict], dict, dict]:
    prim = locked[locked["pick_number"] == 1]
    nullable = lambda value, cast: None if pd.isna(value) else cast(value)  # noqa: E731  (R2 finding 3)
    primaries = {r["date"]: {"batter_id": nullable(r["production_batter_id"], int),
                             "game_pk": nullable(r["production_game_pk"], int),
                             "p_game_hit": nullable(r["production_p_game_hit"], float),
                             "locked_at": nullable(r["production_locked_at"], str)}
                 for r in prim.to_dict("records")}
    admission, surf, tables = [], {}, {}
    for d in sorted(set(slate_paths) | set(locked["date"])):
        parsed, sha = None, None
        if d in slate_paths:
            raw = slate_paths[d].read_bytes()
            sha = surfaces.sha256(raw)
            try:
                parsed = surfaces.parse_slate(raw, expected_date=d)
            except surfaces.SlateFormatError as exc:
                parsed = exc
        rec = surfaces.admit_surface(date=d, slate=parsed, slate_sha256=sha, witnesses=witnesses.get(d, []),
                                     production_primary=primaries.get(d))
        admission.append(rec)
        batters = surfaces.batter_table(parsed["rows"]) if isinstance(parsed, dict) else None
        if batters is not None:
            tables[d] = batters
        surf[d] = {"admitted": rec["admitted"], "reason": rec["reason"], "batters": batters}
    return admission, surf, tables


def main(argv=None) -> int:
    args = parse_args(argv)
    head = x22_gate()                                                   # before anything else is read
    overrides = {k: getattr(args, k) for k in registration.OVERRIDABLE}
    try:
        check_registration()
        params = resolve_params(overrides, exploratory=args.exploratory)
    except RegistrationError as exc:
        raise SystemExit(f"registration: {exc}") from exc
    log(f"X-22 gate passed; code {head[:7]}; mode {params.mode}")

    paths = input_paths(args, params)
    inputs = hash_inputs(paths, args)
    log(f"hashed inputs: {len(inputs['user_picks'])} user-pick files, {len(inputs['slates'])} slate files")
    manifest = {"schema": "mining87_input_manifest_v1", "code": head, "x22_commit": registration.X22_COMMIT,
                "registration_fingerprint": REGISTRATION_FINGERPRINT, "run_mode": params.mode,
                "documents": document_hashes(), "code_identity": code_identity(), "inputs": inputs}
    if args.expect_inputs is not None:
        prior = json.loads(args.expect_inputs.read_text())
        pinned = prior.get("inputs", {})
        differ = sorted(k for k in set(pinned) | set(inputs) if pinned.get(k) != inputs.get(k))
        if differ:
            raise SystemExit(f"inputs differ from the pinned manifest in {differ}; a retry must use identical inputs")
        if prior.get("registration_fingerprint") != REGISTRATION_FINGERPRINT or prior.get("run_mode") != params.mode:
            raise SystemExit("the pinned manifest has another registration or run mode; a retry keeps the methods")
        manifest["retry_of"] = {"pinned_manifest_sha256": sha256_file(args.expect_inputs),
                                "code_identical": prior.get("code_identity", {}).get("files")
                                == manifest["code_identity"]["files"]}
        log("inputs and registration identical to the pinned manifest")

    run_id = (f"{head[:7]}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
              + ("-exploratory" if params.mode == "exploratory" else ""))
    out_root = args.out.expanduser()
    partial, final = out_root / f"{run_id}.partial", out_root / run_id
    partial.mkdir(parents=True, exist_ok=False)

    # --- production (outcome-bearing loaders start here, after the gate, the registration and the pins) ---
    ledger, contest_slots, ledger_meta = load_accepted_ledger(args.ledger_dir.expanduser())
    locked, prod_inv = project_locked_slots(ledger, contest_slots, window_start=params.window_start,
                                            window_end=params.window_end)
    context = None
    if args.production_context is not None:
        names = set(pq.read_schema(args.production_context).names)
        cols = ["date", "slot", "batter_id", "game_pk", *[c for c in CONTEXT_COLUMNS if c in names]]
        context = pq.read_table(args.production_context, columns=cols).to_pandas()
    locked, ctx_meta = attach_production_context(locked, context)
    log(f"ledger build accepted ({ledger_meta['run']}); locked production units {len(locked)}")

    # --- public picks and legal consensus ---
    usernames, cohort_meta = load_cohort(paths["cohort_snapshot"], tab=params.cohort_tab)
    latest, pub_inv = load_public_picks(paths["user_picks_dir"], window_start=params.window_start,
                                        window_end=params.window_end,
                                        capture_end_exclusive=params.capture_end_exclusive)
    stems, stem_meta = cohort_stems(usernames, available_stems={p.stem for p in
                                                                paths["user_picks_dir"].glob("*.parquet")})
    fixed, fixed_votes = consensus_table(latest, users=stems, cohort="fixed_cohort")
    allt, all_votes = consensus_table(latest, users=None, cohort="all_tracked")
    consensus = pd.concat([fixed, allt], ignore_index=True)
    votes = pd.concat([fixed_votes, all_votes], ignore_index=True)
    log(f"cohort {cohort_meta['n_usernames']} usernames; latest public observations {len(latest)}")

    # --- surfaces ---
    witnesses, witness_meta = surfaces.load_witnesses(args.surface_witness)
    admission, surf, slate_tables = admit_dates(locked, paths["slates"], witnesses)
    n_admitted = sum(a["admitted"] for a in admission)
    unit_games = load_unit_games(paths["units_dir"]) if n_admitted and paths["units_dir"].is_dir() else None
    log(f"surface admission: {len(admission)} dates considered, {n_admitted} admitted")

    # --- units and streams ---
    units = build_units(locked, consensus, surfaces=surf, unit_games=unit_games)
    records, mech_meta = inference.load_mechanism_records(args.mechanism_records)
    kw = dict(mechanism_records=records, min_n=params.min_n, q_threshold=params.q_threshold,
              mechanism_min_n=params.mechanism_min_n, mechanism_min_lift=params.mechanism_min_lift)
    primary = inference.evaluate_stream(units, stream="primary", can_nominate=params.can_nominate, **kw)
    sens = inference.evaluate_stream(inference.tie_excluded(units), stream="tie_excluded_sensitivity",
                                     can_nominate=False, **kw)
    comparison = inference.compare_streams(primary["cells"], sens["cells"])
    log(f"units {len(units)}; both streams evaluated")

    witness_meta["unused_witness_dates"] = sorted(set(witnesses) - {a["date"] for a in admission})
    rep = {
        "schema_version": report.SCHEMA_VERSION,
        "research_only": True, "production_deploy_claim": False, "no_policy_edit_supported": True,
        "run": {"run_id": run_id, "run_mode": params.mode, "can_nominate": params.can_nominate,
                "overrides": params.overrides, "code": head, "x22_commit": registration.X22_COMMIT,
                "registration": REGISTRATION, "registration_fingerprint": REGISTRATION_FINGERPRINT,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "ledger_build": ledger_meta,
                "parameters": {"window": [params.window_start, params.window_end], "seed": params.seed,
                               "n_bootstrap": params.n_bootstrap,
                               "expected_block_length": params.expected_block_length,
                               "snapshot_file": params.snapshot_file, "top_k": list(params.top_k)},
                "witness_file": witness_meta, "mechanism_records": mech_meta},
        "pre_registration": {"documents": manifest["documents"], "exposure_row": "X-22",
                             "x22_commit": registration.X22_COMMIT,
                             "status": "retrospective post-hoc analysis of a previously exposed 2026 window",
                             "prior_overlapping_exposures": ["X-10"], "untouched_holdout": False,
                             "claim_historical_outcomes_never_inspected": False},
        "coverage_and_denominators": report.coverage(
            units, production_inventory=prod_inv, admission=admission, public_inventory=pub_inv,
            cohort_meta={**cohort_meta, **stem_meta}, context_meta=ctx_meta),
        "primary_estimand_top_n": report.top_n(units, params.top_k),
        "same_slot_agreement_fallback": report.same_slot_agreement(units),
        "secondary_paired_outcomes": report.paired_outcomes(units, params),
        "secondary_conditional_miscalibration": report.miscalibration(units),
        "secondary_consensus_concentration": report.concentration(consensus),
        "streams": {s["stream"]: {"summary": s["summary"], "cells": report.cells_records(s["cells"])}
                    for s in (primary, sens)},
        "sensitivity_comparison": comparison,
        "nomination": report.nomination(primary, comparison, units, params),
        "served_slate_diagnostic": report.served_slate_diagnostic(units, slate_tables, admission, params.top_k),
        "decomposition_variables": DECOMPOSITION_VARIABLES,
        "methodology_constraints": report.METHODOLOGY,
    }

    log("writing outputs")
    (partial / "input_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (partial / "report.json").write_text(json.dumps(report.jsonable(rep), indent=1, allow_nan=False) + "\n")
    units.to_parquet(partial / "units.parquet", index=False)
    primary["cells"].to_parquet(partial / "cells_primary.parquet", index=False)
    sens["cells"].to_parquet(partial / "cells_tie_excluded_sensitivity.parquet", index=False)
    votes.to_parquet(partial / "consensus_votes.parquet", index=False)
    complete = {"schema": "mining87_completion_v1", "run_id": run_id, "run_mode": params.mode, "code": head,
                "x22_commit": registration.X22_COMMIT, "registration_fingerprint": REGISTRATION_FINGERPRINT,
                "input_manifest_sha256": sha256_file(partial / "input_manifest.json"),
                "outputs": {name: sha256_file(partial / name) for name in OUTPUTS},
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    tmp = partial / "COMPLETE.json.tmp"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(complete, indent=1, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, partial / "COMPLETE.json")
    os.rename(partial, final)
    log(f"complete: {final}")
    print(json.dumps({"run_dir": str(final), "complete": True, "run_mode": params.mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
