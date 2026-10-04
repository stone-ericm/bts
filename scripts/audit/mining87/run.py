"""#87 leaderboard mechanism-mining driver (season wrap W2.4).

Refuses to run until exposure row X-22 is published and ``registration.X22_COMMIT`` names its commit. Registered
mode uses only the frozen values in ``registration.REGISTRATION``; any override needs ``--exploratory``, which labels
the run and disables nomination. Every input is read once into a byte snapshot; its hashes (and the code, document
and registration identity) form the execution manifest, which is checked against ``--expect-inputs`` and the output
location's prior attempts and then persisted in the reserved ``<run>.partial`` directory before any outcome-bearing
loader runs. The loaders parse only the snapshot bytes and report the hashes of what they consumed; completion is
refused unless those equal the manifest. Registered mode refuses dirty relevant code, a second registered run after
a completed one, and an unbound retry. Nothing about outcomes is printed; only a complete run is renamed into place
with ``COMPLETE.json`` (sha256 of every output and of the input manifest).

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
import io
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
             "ledger_build": ledger / BUILD_FILE if (ledger / BUILD_FILE).exists() else None,
             "cohort_snapshot": snapshot, "user_picks_dir": lb / "user_picks",
             "slates_dir": data / "picks" / "slates", "units_dir": lb / "static_snapshots" / "units"}
    for name, p in paths["ledger"].items():
        if not p.exists():
            raise SystemExit(f"ledger build file missing: {p}")
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


def snapshot_inputs(paths: dict, args) -> dict:
    """Every input is read exactly once into this byte snapshot (R2 edit 4). The manifest hashes these bytes and the
    analysis parses these bytes and nothing else: no path is reopened and no directory is re-globbed."""
    read = lambda p: None if p is None else Path(p).read_bytes()  # noqa: E731
    snap = {"ledger": {n: read(p) for n, p in paths["ledger"].items()},
            "ledger_build": read(paths["ledger_build"]),
            "cohort_snapshot": (paths["cohort_snapshot"].name, read(paths["cohort_snapshot"])),
            "user_picks": {p.name: read(p) for p in sorted(paths["user_picks_dir"].glob("*.parquet"))},
            "slates": {d: (p.name, read(p)) for d, p in sorted(paths["slates"].items())},
            "surface_witness": read(args.surface_witness), "mechanism_records": read(args.mechanism_records),
            "production_context": read(args.production_context), "unit_captures": None}
    if args.surface_witness is not None:
        from scripts.audit.benchmark_bridge.core import capture_files
        snap["unit_captures"] = ({p.name: read(p) for p in capture_files(paths["units_dir"])}
                                 if paths["units_dir"].is_dir() else {})
    return snap


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def input_identity(snap: dict) -> dict:
    """The frozen input identity. Every optional input carries an explicit status, so absence is pinned too."""
    picks = {n: _sha(b) for n, b in snap["user_picks"].items()}
    opt = lambda b: {"status": "absent"} if b is None else {"status": "supplied", "sha256": _sha(b)}  # noqa: E731
    units = snap["unit_captures"]
    return {"ledger": {n: _sha(b) for n, b in snap["ledger"].items()},
            "cohort_snapshot": {snap["cohort_snapshot"][0]: _sha(snap["cohort_snapshot"][1])},
            "user_picks": picks,
            "user_picks_inventory_sha256": _sha(json.dumps(picks, sort_keys=True).encode()),
            "slates": {name: _sha(raw) for name, raw in snap["slates"].values()},
            "optional_inputs": {
                "ledger_build_manifest": opt(snap["ledger_build"]),
                "surface_witness": opt(snap["surface_witness"]),
                "mechanism_records": opt(snap["mechanism_records"]),
                "production_context": opt(snap["production_context"]),
                "unit_captures": ({"status": "absent", "reason": "no surface witness supplied"} if units is None
                                  else {"status": "supplied", "files": {n: _sha(b) for n, b in units.items()}})}}


def code_identity() -> dict:
    files = {rel: sha256_file(REPO / rel) for rel in CODE_FILES if (REPO / rel).exists()}
    missing = sorted(rel for rel in CODE_FILES if not (REPO / rel).exists())
    try:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *CODE_FILES],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    return {"files": files, "missing_files": missing, "worktree_dirty": None if missing else dirty}


def _prior_attempts(out_root: Path) -> list[dict]:
    attempts = []
    if not out_root.is_dir():
        return attempts
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        manifest, complete = d / "input_manifest.json", d / "COMPLETE.json"
        mode = None
        if complete.exists():
            mode = json.loads(complete.read_text()).get("run_mode")
        elif manifest.exists():
            mode = json.loads(manifest.read_text()).get("run_mode")
        attempts.append({"dir": d, "mode": mode, "complete": complete.exists(),
                         "manifest_sha256": sha256_file(manifest) if manifest.exists() else None})
    return attempts


def enforce_execution_pins(manifest: dict, out_root: Path, expect_inputs: Path | None, mode: str) -> dict | None:
    """R2 edit 5, before the attempt directory is reserved and before any outcome-bearing read. Registered mode:
    relevant code must be clean and verifiable; a completed registered run in this output location ends the
    registered execution (it is not a technical failure eligible for retry); after a failed registered attempt, a
    new attempt must be its technical retry, bound to that attempt's manifest. Any pinned manifest must agree on
    the inputs (inventory and bytes), registration fingerprint, run mode and frozen documents, and in registered
    mode on the relevant code files; an exploratory code change is recorded as diagnostic metadata only."""
    registered = mode == "registered"
    if registered and manifest["code_identity"]["worktree_dirty"] is not False:
        raise SystemExit("registered mode refuses dirty or unverifiable relevant code "
                         f"(worktree_dirty={manifest['code_identity']['worktree_dirty']}, "
                         f"missing={manifest['code_identity'].get('missing_files')}); commit it or run --exploratory")
    prior = [a for a in _prior_attempts(out_root) if a["mode"] == "registered"]
    if registered:
        done = [a["dir"].name for a in prior if a["complete"]]
        if done:
            raise SystemExit(f"a completed registered run exists in {out_root} ({done[0]}); a completed run is not "
                             "a technical failure eligible for retry")
        failed = {a["manifest_sha256"] for a in prior if a["manifest_sha256"]}
        if failed and expect_inputs is None:
            raise SystemExit(f"a prior registered attempt exists in {out_root}; a technical retry must pass "
                             "--expect-inputs <that attempt>/input_manifest.json")
        if failed and sha256_file(expect_inputs) not in failed:
            raise SystemExit("--expect-inputs is not the manifest of a prior registered attempt in this output "
                             "location")
    if expect_inputs is None:
        return None
    pinned = json.loads(expect_inputs.read_text())
    differ = [f"inputs.{k}" for k in sorted(set(pinned.get("inputs", {})) | set(manifest["inputs"]))
              if pinned.get("inputs", {}).get(k) != manifest["inputs"].get(k)]
    differ += [k for k in ("registration_fingerprint", "run_mode", "documents") if pinned.get(k) != manifest[k]]
    code_identical = pinned.get("code_identity", {}).get("files") == manifest["code_identity"]["files"]
    if registered and not code_identical:
        differ.append("code_identity")
    if differ:
        raise SystemExit(f"execution pins differ from the pinned manifest in {differ}; a technical retry uses "
                         "identical inputs, registration, mode, documents and code")
    return {"pinned_manifest_sha256": sha256_file(expect_inputs), "code_identical": code_identical}


def write_synced(path: Path, text: str) -> str:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def verify_consumed(inputs: dict, consumed: dict) -> None:
    """Refuse completion unless every byte string the loaders parsed hashes to the frozen manifest (R2 edit 4)."""
    optional = inputs["optional_inputs"]
    sha_of = lambda key: optional[key].get("sha256")  # noqa: E731
    expected = {"ledger": {**inputs["ledger"], **({BUILD_FILE: sha_of("ledger_build_manifest")}
                                                  if optional["ledger_build_manifest"]["status"] == "supplied"
                                                  else {})},
                "cohort_snapshot": inputs["cohort_snapshot"], "user_picks": inputs["user_picks"],
                "slates": inputs["slates"], "surface_witness": sha_of("surface_witness"),
                "mechanism_records": sha_of("mechanism_records"), "production_context": sha_of("production_context")}
    if "unit_captures" in consumed:
        expected["unit_captures"] = optional["unit_captures"].get("files")
    differ = sorted(k for k in expected if consumed.get(k) != expected[k])
    if differ:
        raise SystemExit(f"consumed inputs differ from the frozen manifest in {differ}; no COMPLETE is written")


def load_unit_games(files: dict[str, bytes]) -> tuple[dict[int, int], dict[str, str]]:
    """unit_id → game_pk where every capture names exactly one feedId (else the unit is unbound), from the pinned
    capture bytes; returns the map and the hashes of the bytes parsed."""
    from scripts.audit.season_ledger.contest import units_lookup
    from scripts.audit.season_ledger.sources.static import parse_units
    rows = []
    for name, raw in sorted(files.items()):
        rows += parse_units(f"units/{name}", raw).rows
    games = {uid: next(iter(e["feed_ids"])) for uid, e in units_lookup(rows).items() if len(e["feed_ids"]) == 1}
    return games, {name: _sha(raw) for name, raw in files.items()}


def admit_dates(locked: pd.DataFrame, slates: dict, witnesses: dict) -> tuple[list[dict], dict, dict, dict]:
    """``slates`` is {date: (file name, pinned bytes)}; returns admission records, per-date surfaces, slate tables
    and the hashes of the slate bytes parsed."""
    prim = locked[locked["pick_number"] == 1]
    nullable = lambda value, cast: None if pd.isna(value) else cast(value)  # noqa: E731  (R2 finding 3)
    primaries = {r["date"]: {"batter_id": nullable(r["production_batter_id"], int),
                             "game_pk": nullable(r["production_game_pk"], int),
                             "p_game_hit": nullable(r["production_p_game_hit"], float),
                             "locked_at": nullable(r["production_locked_at"], str)}
                 for r in prim.to_dict("records")}
    admission, surf, tables, consumed = [], {}, {}, {}
    for d in sorted(set(slates) | set(locked["date"])):
        parsed, sha = None, None
        if d in slates:
            name, raw = slates[d]
            sha = consumed[name] = surfaces.sha256(raw)
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
    return admission, surf, tables, consumed


def read_context(raw: bytes | None) -> pd.DataFrame | None:
    if raw is None:
        return None
    pf = pq.ParquetFile(io.BytesIO(raw))
    names = set(pf.schema_arrow.names)
    return pf.read(columns=["date", "slot", "batter_id", "game_pk",
                            *[c for c in CONTEXT_COLUMNS if c in names]]).to_pandas()


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
    snap = snapshot_inputs(paths, args)
    inputs = input_identity(snap)
    log(f"pinned inputs: {len(inputs['user_picks'])} user-pick files, {len(inputs['slates'])} slate files")
    manifest = {"schema": "mining87_input_manifest_v2", "code": head, "x22_commit": registration.X22_COMMIT,
                "registration_fingerprint": REGISTRATION_FINGERPRINT, "run_mode": params.mode,
                "documents": document_hashes(), "code_identity": code_identity(), "inputs": inputs}
    out_root = args.out.expanduser()
    retry_of = enforce_execution_pins(manifest, out_root, args.expect_inputs, params.mode)
    if retry_of is not None:
        manifest["retry_of"] = retry_of

    run_id = (f"{head[:7]}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
              + ("-exploratory" if params.mode == "exploratory" else ""))
    partial, final = out_root / f"{run_id}.partial", out_root / run_id
    partial.mkdir(parents=True, exist_ok=False)
    manifest_sha = write_synced(partial / "input_manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    log(f"execution pins persisted in {partial.name}")

    # --- outcome-bearing loaders start here: after the gate, the registration and the persisted pins ---
    consumed: dict = {}
    ledger_files = {**snap["ledger"], **({BUILD_FILE: snap["ledger_build"]} if snap["ledger_build"] else {})}
    ledger, contest_slots, ledger_meta = load_accepted_ledger(ledger_files)
    consumed["ledger"] = ledger_meta.pop("consumed_sha256")
    locked, prod_inv = project_locked_slots(ledger, contest_slots, window_start=params.window_start,
                                            window_end=params.window_end)
    consumed["production_context"] = (None if snap["production_context"] is None
                                      else _sha(snap["production_context"]))
    locked, ctx_meta = attach_production_context(locked, read_context(snap["production_context"]))
    log(f"ledger build accepted ({ledger_meta['run']}); locked production units {len(locked)}")

    # --- public picks and legal consensus (the pinned inventory, never a new glob) ---
    cohort_name, cohort_raw = snap["cohort_snapshot"]
    usernames, cohort_meta = load_cohort(cohort_raw, tab=params.cohort_tab, name=cohort_name)
    consumed["cohort_snapshot"] = cohort_meta.pop("consumed_sha256")
    latest, pub_inv = load_public_picks(snap["user_picks"], window_start=params.window_start,
                                        window_end=params.window_end,
                                        capture_end_exclusive=params.capture_end_exclusive)
    consumed["user_picks"] = pub_inv.pop("consumed_sha256")
    stems, stem_meta = cohort_stems(usernames, available_stems={Path(n).stem for n in inputs["user_picks"]})
    fixed, fixed_votes = consensus_table(latest, users=stems, cohort="fixed_cohort")
    allt, all_votes = consensus_table(latest, users=None, cohort="all_tracked")
    consensus = pd.concat([fixed, allt], ignore_index=True)
    votes = pd.concat([fixed_votes, all_votes], ignore_index=True)
    log(f"cohort {cohort_meta['n_usernames']} usernames; latest public observations {len(latest)}")

    # --- surfaces ---
    witnesses, witness_meta = surfaces.load_witnesses(snap["surface_witness"])
    consumed["surface_witness"] = witness_meta.get("sha256")
    admission, surf, slate_tables, consumed["slates"] = admit_dates(locked, snap["slates"], witnesses)
    n_admitted = sum(a["admitted"] for a in admission)
    unit_games = None
    if n_admitted and snap["unit_captures"]:
        unit_games, consumed["unit_captures"] = load_unit_games(snap["unit_captures"])
    log(f"surface admission: {len(admission)} dates considered, {n_admitted} admitted")

    # --- units and streams ---
    units = build_units(locked, consensus, surfaces=surf, unit_games=unit_games)
    records, mech_meta = inference.load_mechanism_records(snap["mechanism_records"])
    consumed["mechanism_records"] = mech_meta.get("sha256")
    kw = dict(mechanism_records=records, min_n=params.min_n, q_threshold=params.q_threshold,
              mechanism_min_n=params.mechanism_min_n, mechanism_min_lift=params.mechanism_min_lift)
    primary = inference.evaluate_stream(units, stream="primary", can_nominate=params.can_nominate, **kw)
    sens = inference.evaluate_stream(inference.tie_excluded(units), stream="tie_excluded_sensitivity",
                                     can_nominate=False, **kw)
    comparison = inference.compare_streams(primary["cells"], sens["cells"])
    log(f"units {len(units)}; both streams evaluated")
    verify_consumed(inputs, consumed)
    if sha256_file(partial / "input_manifest.json") != manifest_sha:
        raise SystemExit("the persisted execution pins changed during the attempt; no COMPLETE is written")

    witness_meta["unused_witness_dates"] = sorted(set(witnesses) - {a["date"] for a in admission})
    rep = {
        "schema_version": report.SCHEMA_VERSION,
        "research_only": True, "production_deploy_claim": False, "no_policy_edit_supported": True,
        "run": {"run_id": run_id, "run_mode": params.mode, "can_nominate": params.can_nominate,
                "overrides": params.overrides, "code": head, "x22_commit": registration.X22_COMMIT,
                "registration": REGISTRATION, "registration_fingerprint": REGISTRATION_FINGERPRINT,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "ledger_build": ledger_meta,
                "input_manifest_sha256": manifest_sha, "retry_of": manifest.get("retry_of"),
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
    (partial / "report.json").write_text(json.dumps(report.jsonable(rep), indent=1, allow_nan=False) + "\n")
    units.to_parquet(partial / "units.parquet", index=False)
    primary["cells"].to_parquet(partial / "cells_primary.parquet", index=False)
    sens["cells"].to_parquet(partial / "cells_tie_excluded_sensitivity.parquet", index=False)
    votes.to_parquet(partial / "consensus_votes.parquet", index=False)
    complete = {"schema": "mining87_completion_v1", "run_id": run_id, "run_mode": params.mode, "code": head,
                "x22_commit": registration.X22_COMMIT, "registration_fingerprint": REGISTRATION_FINGERPRINT,
                "input_manifest_sha256": manifest_sha, "consumed_inputs_verified": True,
                "outputs": {name: sha256_file(partial / name) for name in OUTPUTS},
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_synced(partial / "COMPLETE.json", json.dumps(complete, indent=1, sort_keys=True) + "\n")
    os.rename(partial, final)
    log(f"complete: {final}")
    print(json.dumps({"run_dir": str(final), "complete": True, "run_mode": params.mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
