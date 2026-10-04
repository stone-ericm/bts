"""W2.1 / W2.2 field-products driver (design docs/superpowers/specs/2026-10-04-field-products-design.md rev 2, FROZEN).

Refuses to run until exposure rows X-24 (W2.1: board distribution, own rank/percentile, survivor-selected Cohort A
case series) and X-25 (W2.2: all E members, May 1–July 3 follow-up, restricted final-backfill extension,
unknown/attrition handling, production comparison) are published: both commit constants must be set, be ancestors
of HEAD, and the register in this checkout must carry both rows.

Order of work: the gate; then, before any pick content, outcome label or ledger row is parsed, the membership (E,
acquisition labels) and the daily-file identity binding are built from the manifest, the grab's cohort.json and the
directory LISTING only, and ``freeze.json`` (code, membership and binding hashes) is written; then the outcome-bearing
steps. Every input file read is hashed into ``manifest.json``.

Usage (on the box, in a transient unit):
  .venv/bin/python -m scripts.audit.field_products.run --data-root data \\
      --ledger-dir data/validation/season_2026_ledger/<accepted build> \\
      --out data/validation/w21_w22_field --our-user-id <our stable BTS user id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit.field_products import census as C
from scripts.audit.field_products import cohort as K
from scripts.audit.field_products import compare as Q
from scripts.audit.field_products import leaders as L
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S

REPO = Path(__file__).resolve().parents[3]
REGISTER = REPO / "docs/audit/2026-09-22-exposure-register.md"
X24_COMMIT: str | None = None        # the register commit that publishes X-24 (unset: the run refuses)
X25_COMMIT: str | None = None        # the register commit that publishes X-25 (unset: the run refuses)
FINAL_GRAB = "final_grab_20260927"
EARLY_MANIFEST = REPO / "docs/audit/2026-09-22-early-cohort-2026-05-01.json"
LEDGER_FILES = ("season_2026_ledger.parquet", "season_2026_ledger_contest_slots.parquet", "ACCEPTED.json")
LIMITS = [
    "Daily pick files carry no user_id: attribution rests on the 5/01 manifest usernames; a later username change or "
    "reuse by another account is undetectable except where it changes a settled slot (ownership quarantine).",
    "The daily scraper fetched one profile per username per run, so same-name accounts could share a file; such "
    "members are quarantined when the manifest or the file stems show the collision.",
    "Pick logs are observations of public behaviour, not proof of pre-lock timing.",
    "Allocation B / the extension's 7/04-9/27 histories come from one end-of-season grab: history depth is whatever "
    "the API returned; slots without a pick (pending/unresolved) made the grab fail closed (no parsed file).",
    "'void' is taken as the settled Pass label (season_ledger CONTEST_NORMALIZATION): it settles a round's slot set "
    "but is never a graded slot.",
    "Run reconstruction trusts the reported seasonal streak: a continuation identity is not produced by an exactly "
    "compensating hidden reset-and-rebuild in unobserved rounds.",
    "Pooled slot ratios weight prolific users and DD days more; intervals assume exchangeable date clusters and do not "
    "cover cross-date dependence, observation selection or missing histories. Descriptive only (D3).",
    "Our arm counts 'evidenced' and 'inferred' contest links (the ledger's grade-transferring matches) as uniquely "
    "confirmed; both are reported separately in included_by_match.",
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def gate() -> str:
    """Refuse unless X-24 and X-25 are published: both commits set and in HEAD, both rows in this register."""
    head = git("rev-parse", "HEAD")
    for name, commit in (("X-24", X24_COMMIT), ("X-25", X25_COMMIT)):
        if commit is None:
            raise SystemExit(f"{name} gate: {name.replace('-', '')}_COMMIT is unset (publish {name} first)")
        if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, head]).returncode != 0:
            raise SystemExit(f"{name} gate: {commit} is not an ancestor of HEAD {head[:7]}")
    reg = REGISTER.read_text()
    for name in ("X-24", "X-25"):
        if f"| {name} |" not in reg:
            raise SystemExit(f"{name} gate: the register in this checkout has no {name} row")
    return head


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def jsonable(o):
    """Plain JSON: numpy/pandas scalars unwrapped, dates as ISO strings, NaN/NA as null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) or math.isinf(float(o)) else float(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def _write_table(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (dict, list))).any():
            df[c] = df[c].map(lambda v: json.dumps(jsonable(v), sort_keys=True) if isinstance(v, (dict, list)) else v)
    df.to_parquet(path, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W2.1/W2.2 field products (gated on X-24/X-25)")
    ap.add_argument("--data-root", type=Path, required=True, help="the repo data/ directory (holds leaderboard/)")
    ap.add_argument("--ledger-dir", type=Path, required=True, help="the accepted W1.1 build (holds ACCEPTED.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--our-user-id", type=int, default=None, help="our stable BTS user id (own board rank)")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--early-manifest", type=Path, default=EARLY_MANIFEST)
    ap.add_argument("--final-grab", default=FINAL_GRAB)
    args = ap.parse_args(argv)
    head = gate()

    data = args.data_root.expanduser().resolve()
    grab = data / "leaderboard" / args.final_grab
    daily_dir = data / "leaderboard" / "user_picks"
    led = args.ledger_dir.expanduser().resolve()
    for f in LEDGER_FILES:
        if not (led / f).exists():
            raise SystemExit(f"ledger dir {led} lacks {f} (use the accepted build)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"run dir {run_dir}; code {head[:7]}")
    inputs: dict[str, str] = {}

    def read(path: Path, key: str) -> bytes:
        b = Path(path).read_bytes()
        inputs[key] = _sha_bytes(b)
        return b

    # ---- membership and identity, before any outcome-bearing content ----
    manifest = K.load_manifest(args.early_manifest)
    inputs["early_manifest"] = manifest["sha256"]
    cohort_json = json.loads(read(grab / "cohort.json", f"leaderboard/{args.final_grab}/cohort.json"))
    labels, alloc_checks = K.allocation(manifest, cohort_json)
    grab_copy = grab / "inputs" / "early_cohort.json"
    alloc_checks["grab_input_copy_matches"] = grab_copy.exists() and _sha_file(grab_copy) == manifest["sha256"]
    stems = sorted(p.stem for p in daily_dir.glob("*.parquet"))
    # a pre-2026-06-09 username containing "/" was written into a subdirectory: listed and counted, never bound
    nested = sorted(str(p.relative_to(daily_dir)) for p in daily_dir.rglob("*.parquet") if p.parent != daily_dir)
    binding, bind_counts = K.bind_daily_files(manifest, stems)
    code_files = {p.name: _sha_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    freeze = {"written_before_outcome_reads": True, "code_head": head, "code_files": code_files,
              "manifest_E_sha256": manifest["sha256"], "allocation_checks": alloc_checks,
              "labels_sha256": _sha_bytes(labels.to_csv(index=False).encode()),
              "binding_sha256": _sha_bytes(binding.to_csv(index=False).encode()),
              "daily_listing_sha256": _sha_bytes("\n".join(stems).encode()), "daily_files_listed": len(stems),
              "daily_nested_files_ignored": nested, "binding_counts": bind_counts,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    (run_dir / "freeze.json").write_text(json.dumps(jsonable(freeze), indent=1, sort_keys=True) + "\n")
    _write_table(binding, run_dir / "w22_binding.parquet")
    _write_table(labels, run_dir / "w22_labels.parquet")
    log(f"froze membership: E={manifest['n']}, binding {bind_counts['members']}")

    # ---- W2.1: census and Cohort A ----
    rec = C.load_board_receipts(grab)
    census_gate = C.census_gate(rec)
    season_best = C.season_best_summary(census_gate["qualified"], census_gate, args.our_user_id)
    log(f"census gate: {census_gate['census']} {census_gate['failures']}")
    identity = json.loads(read(grab / "identity.json", f"leaderboard/{args.final_grab}/identity.json"))
    a = L.case_series(grab, board=rec["board"], cohort_json=cohort_json, identity=identity, status=rec["status"])

    # ---- W2.2: daily corpus for bound members ----
    bound = binding[binding["binding"] == "bound"]
    parts, obs_stats = [], {}
    for r in bound.itertuples():
        paths = [daily_dir / f"{s}.parquet" for s in r.files]
        for p in paths:
            read(p, f"leaderboard/user_picks/{p.name}")
        o = P.read_observations(paths, user_id=int(r.user_id), source="daily")
        obs_stats[int(r.user_id)] = {"rows": int(len(o)), "captures": int(o["captured_at"].nunique()) if len(o) else 0,
                                     "first_capture": o["captured_at"].min() if len(o) else None,
                                     "last_capture": o["captured_at"].max() if len(o) else None}
        if len(o):
            parts.append(o)
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res = P.resolve(obs)
    ownership = P.ownership_conflict_users(res)
    available = {u for u, st in obs_stats.items() if st["rows"] and u not in ownership}
    start, end = Q.PRIMARY
    windows = {u: S.window_summary(res.rounds[res.rounds["user_id"] == u], start, end) for u in sorted(available)}
    av = Q.availability(manifest["members"], labels, binding, ownership_quarantined=ownership, res=res,
                        obs_stats=obs_stats, start=start, end=end, windows=windows)
    e = Q.e_arm(res.slots, available, start, end)

    # ---- our ledger arm ----
    ledger = pd.read_parquet(led / "season_2026_ledger.parquet")
    contest = pd.read_parquet(led / "season_2026_ledger_contest_slots.parquet")
    for f in LEDGER_FILES:
        read(led / f, f"ledger/{f}")
    o, ours_exc = Q.ours_slots(ledger, contest, start, end)
    tables = Q.date_tables(e, o, n_resamples=args.n_resamples, n_members=manifest["n"],
                           calendar_dates=(end - start).days + 1)
    e_excl = Q.e_exclusions(res.slots, available, start, end)
    log(f"primary: E {len(e)} slots, ours {len(o)} slots, {tables['all_observed_dates']['dates']} dates")

    # ---- extension: usable final-grab histories of E∩(A∪B) ----
    fg = K.final_grab_status(labels, identity, grab)
    ext_parts = []
    for r in fg[fg["usable"]].itertuples():
        p = grab / r.parsed_path
        read(p, f"leaderboard/{args.final_grab}/{r.parsed_path}")
        ext_parts.append(P.read_observations([p], user_id=int(r.user_id), source="final_grab"))
    ext_res = P.resolve(pd.concat(ext_parts, ignore_index=True) if ext_parts else pd.DataFrame())
    ext = Q.extension(fg, ext_res.slots, n_members=manifest["n"])

    # ---- outputs ----
    status_counts = pd.Series([w["status"] for w in windows.values()]).value_counts().to_dict() if windows else {}
    per_user = av[av["window_graded_slots"] > 0]["window_graded_slots"]
    results = {
        "schema": "w21_w22_field_products_v1", "code": head, "x24_commit": X24_COMMIT, "x25_commit": X25_COMMIT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "inputs": {"data_root": str(data), "final_grab": args.final_grab, "ledger_dir": str(led),
                   "early_manifest": str(args.early_manifest), "our_user_id": args.our_user_id,
                   "n_resamples": args.n_resamples},
        "w21": {"census_gate": {k: v for k, v in census_gate.items() if k != "qualified"},
                "season_best": season_best, "case_series_A": a["summary"]},
        "w22": {
            "manifest_E": {"n": manifest["n"], "sha256": manifest["sha256"], "definition": manifest["definition"],
                           "frozen_at": manifest["frozen_at"], "source_fixture_sha256": manifest["source_fixture_sha256"],
                           "checks": alloc_checks, "allocation_labels": labels["allocation"].value_counts().to_dict(),
                           "note": "A/B/E_in_A/E_unfetched/B_shortfall are acquisition labels, not cohorts"},
            "identity": {**bind_counts, "ownership_quarantined": sorted(ownership),
                         "daily_nested_files_ignored": len(nested),
                         "rule": "bind only by a unique 5/01 username sanitization whose file stems are the member's "
                                 "own raw or sanitized name; otherwise quarantine"},
            "availability": {"members": int(len(av)), "daily_history": av["daily_history"].value_counts().to_dict(),
                             "window_activity": av["window_activity"].value_counts().to_dict(),
                             "by_allocation": av.groupby("allocation")["daily_history"].value_counts().unstack(
                                 fill_value=0).to_dict("index")},
            "primary": {"window": [start.isoformat(), end.isoformat()], "ours": ours_exc, "E_exclusions": e_excl,
                        "tables": {k: v for k, v in tables.items() if k != "per_date"},
                        "per_user_graded_slots": {"users": int(len(per_user)),
                                                  "min": int(per_user.min()) if len(per_user) else None,
                                                  "median": float(per_user.median()) if len(per_user) else None,
                                                  "max": int(per_user.max()) if len(per_user) else None},
                        "window_streaks": {"status": status_counts,
                                           "rule": "carried-in streak excluded; exact only when every in-window "
                                                   "transition is evidenced; else an evidenced-chain lower bound"}},
            "extension": ext},
        "data_contract": {"daily": res.log, "final_grab_A": a["summary"]["revision_log"],
                          "final_grab_E_extension": ext_res.log},
        "limits": LIMITS,
    }
    (run_dir / "results.json").write_text(json.dumps(jsonable(results), indent=1, sort_keys=True) + "\n")
    _write_table(pd.DataFrame(season_best["distribution"]), run_dir / "w21_distribution.parquet")
    _write_table(a["users"], run_dir / "w21_case_series_A.parquet")
    _write_table(a["runs"], run_dir / "w21_runs_A.parquet")
    _write_table(av, run_dir / "w22_availability.parquet")
    _write_table(e, run_dir / "w22_e_slots.parquet")
    _write_table(o, run_dir / "w22_ours_slots.parquet")
    _write_table(tables["per_date"].reset_index().rename(columns={"index": "date"}), run_dir / "w22_per_date.parquet")
    _write_table(fg, run_dir / "w22_final_grab_status.parquet")
    for name, df in (("w22_daily_slots", res.slots), ("w22_daily_rounds", res.rounds),
                     ("w22_extension_slots", ext_res.slots)):
        _write_table(df if len(df) else pd.DataFrame({"empty": []}), run_dir / f"{name}.parquet")
    manifest_out = {"inputs": inputs, "code": {"head": head, "files": code_files},
                    "membership": {"manifest_E_sha256": manifest["sha256"], "labels_sha256": freeze["labels_sha256"],
                                   "binding_sha256": freeze["binding_sha256"],
                                   "daily_listing_sha256": freeze["daily_listing_sha256"]},
                    "outputs": {p.name: _sha_file(p) for p in sorted(run_dir.iterdir())
                                if p.name != "manifest.json"}}
    (run_dir / "manifest.json").write_text(json.dumps(jsonable(manifest_out), indent=1, sort_keys=True) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
