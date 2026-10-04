"""W2.1 / W2.2 field-products driver (design docs/superpowers/specs/2026-10-04-field-products-design.md rev 2, FROZEN;
code review docs/audit/2026-10-04-field-products-code-codex-r1.md).

Order of work (review r1 F1, F6):
1. ``gate()``: X-24 and X-25 must each be PUBLISHED by their named commit (the row is in the register at that commit
   and not in its parent), the commit must be in HEAD, the row must be unchanged at HEAD, the working register must
   equal HEAD's, and every executing repo source (this package and every repo module it imports, plus the register)
   must be tracked and clean.
2. Receipts and membership, read once into the ``Stage``: the manifest E, the grab's status/cohort/identity and its
   manifest copy. Fail closed on the manifest hash, E count, allocation, manifest copy and the grab's recorded
   artifact hashes for cohort.json/identity.json — before any outcome.
3. Identity/path binding: the daily files from the directory LISTING and the 5/01 names; the final-grab board, raw
   pages, raw profiles and parsed pick files from the receipts; the W1.1 acceptance receipt verified
   (``ledger.verify_receipt``). Every one of these files is staged (read and hashed).
4. ``freeze.json`` (gate, code, source, membership and binding manifests) is written before any outcome is parsed.
5. Outcomes are parsed only from the staged bytes. 6. ``Stage.verify``: any content or listing change since the
   freeze refuses the run. 7. Results, tables and ``manifest.json`` (equal to the freeze's source manifest).

Usage (on the box, in a transient unit, from a clean checkout of the reviewed commit):
  .venv/bin/python -m scripts.audit.field_products.run --data-root data \\
      --ledger-dir data/validation/season_2026_ledger/<accepted build> \\
      --out data/validation/w21_w22_field --our-user-id <our stable BTS user id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from scripts.audit.field_products import census as C
from scripts.audit.field_products import cohort as K
from scripts.audit.field_products import compare as Q
from scripts.audit.field_products import leaders as L
from scripts.audit.field_products import ledger as LG
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S
from scripts.audit.field_products.sources import Stage

REPO = Path(__file__).resolve().parents[3]
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
X24_COMMIT: str | None = None        # the register commit that publishes X-24 (unset: the run refuses)
X25_COMMIT: str | None = None        # the register commit that publishes X-25 (unset: the run refuses)
FINAL_GRAB = "final_grab_20260927"
EARLY_MANIFEST = REPO / "docs/audit/2026-09-22-early-cohort-2026-05-01.json"
PER_BATCH_IDENTITY = ("not established by stored evidence: daily username-keyed pick files carry no user id, and no "
                      "stored record (pick schema, season_stats before 2026-09-22, scrape_status, logs, snapshots "
                      "before 2026-07-03) names the account each appended batch was fetched for; E-arm attribution "
                      "rests on the members' stable 5/01 usernames")
LIMITS = [
    "E-arm attribution: " + PER_BATCH_IDENTITY + ". Members are quarantined only where the manifest, the file stems "
    "or a settled-slot change show a collision; an unseen same-name account is undetectable.",
    "Pick logs are observations of public behaviour, not proof of pre-lock timing.",
    "Allocation B / the extension's 7/04-9/27 histories come from one end-of-season grab: history depth is whatever "
    "the API returned; a profile with any pending/unresolved slot made the grab fail closed (no parsed file).",
    "Round completeness (DD frequency) needs a verified raw response: only final-grab rounds have one; daily-corpus "
    "rounds are incomplete for DD/streak denominators while their exact slot grades stay usable.",
    "No exact within-window streak maximum or run dates: no stored record establishes a complete entered-round "
    "history; observed-segment lower bounds only.",
    "Composition (team, home/away) is unknown: stored context comes from capture-time lookups and no independent "
    "historical pick-time witness exists.",
    "'void' is the settled Pass label (W1.1 HOLD normalisation): it settles a round's slot set but is never graded.",
    "Unobserved dates are calendar dates without an observed pick, including dates without a contest round; neither "
    "skips nor an activity denominator. A round a later capture omits is kept and flagged; that proves neither "
    "deletion nor complete follow-up.",
    "Pooled slot ratios weight prolific users and DD days more; intervals assume exchangeable date clusters and do not "
    "cover cross-date dependence, observation selection or missing histories; any failed draw makes the nominal "
    "interval unavailable. Descriptive only (D3).",
    "Our arm counts 'evidenced' and 'inferred' contest links (the W1.1 grade-transferring matches); inferred game "
    "identity remains inferred; both are reported separately in included_by_match.",
]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def git(*args, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _git_blob(repo: Path, rev: str, rel: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _row(blob: bytes | None, name: str) -> str | None:
    if blob is None:
        return None
    rows = [ln for ln in blob.decode().splitlines() if ln.startswith(f"| {name} |")]
    return rows[0] if len(rows) == 1 else None


def executing_sources(repo: Path = REPO) -> list[str]:
    """Repo-relative paths of every loaded module under scripts/ or src/ (this package and its repo dependencies:
    the grab, the leaderboard parser/storage/models, the shared bootstrap, the ledger compiler schemas, ...) plus
    the exposure register."""
    out = {REGISTER_REL}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(repo.resolve())
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in ("scripts", "src"):
            out.add(str(rel))
    return sorted(out)


def gate(repo: Path = REPO, sources: list[str] | None = None) -> dict:
    """Refuse unless X-24 and X-25 are published by their named commits, unchanged at HEAD, with a clean register
    and clean, tracked executing sources."""
    head = git("rev-parse", "HEAD", repo=repo)
    head_reg = _git_blob(repo, "HEAD", REGISTER_REL)
    if head_reg is None or (repo / REGISTER_REL).read_bytes() != head_reg:
        raise SystemExit("gate: the working register differs from HEAD's committed register")
    rows = {}
    for name, commit in (("X-24", X24_COMMIT), ("X-25", X25_COMMIT)):
        if commit is None:
            raise SystemExit(f"{name} gate: {name.replace('-', '')}_COMMIT is unset (publish {name} first)")
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, head],
                          capture_output=True).returncode != 0:
            raise SystemExit(f"{name} gate: {commit} is not an ancestor of HEAD {head[:7]}")
        at = _row(_git_blob(repo, commit, REGISTER_REL), name)
        if at is None:
            raise SystemExit(f"{name} gate: the register at {commit[:7]} has no {name} row")
        if _row(_git_blob(repo, f"{commit}^", REGISTER_REL), name) is not None:
            raise SystemExit(f"{name} gate: {commit[:7]} did not publish {name} (the row is already in its parent)")
        if _row(head_reg, name) != at:
            raise SystemExit(f"{name} gate: the {name} row changed since its publication in {commit[:7]}")
        rows[name] = {"commit": git("rev-parse", commit, repo=repo), "row_sha256": hashlib.sha256(at.encode()).hexdigest()}
    paths = sources if sources is not None else executing_sources(repo)
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *paths, repo=repo)
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", *paths],
                             capture_output=True).returncode == 0
    if dirty or not tracked:
        raise SystemExit(f"gate: executing sources are not clean tracked files at HEAD: {dirty or 'untracked path'}")
    return {"head": head, "rows": rows, "sources": paths}


def code_manifest(head: str) -> dict:
    files = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in executing_sources()}
    lock = REPO / "uv.lock"
    return {"head": head, "files": files, "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
            if lock.exists() else None, "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "pyarrow": pa.__version__}


def jsonable(o):
    """Plain JSON: numpy/pandas scalars unwrapped, dates as ISO strings, NaN/NA as null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
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
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (dict, list, tuple, np.ndarray))).any():
            df[c] = df[c].map(lambda v: json.dumps(jsonable(v), sort_keys=True)
                              if isinstance(v, (dict, list, tuple, np.ndarray)) else v)
    df.to_parquet(path, index=False)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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
    g = gate()
    head = g["head"]

    data = args.data_root.expanduser().resolve()
    grab = data / "leaderboard" / args.final_grab
    daily_dir = data / "leaderboard" / "user_picks"
    led = args.ledger_dir.expanduser().resolve()
    st = Stage()
    rd = st.read

    # ---- 2. receipts and membership; fail closed before any outcome ----
    st.add(args.early_manifest, "early_manifest")
    manifest = K.load_manifest(args.early_manifest, read=rd)
    for name in ("status.json", "cohort.json", "identity.json", "inputs/early_cohort.json"):
        st.add(grab / name, f"final_grab/{name}")
    status = json.loads(rd(grab / "status.json"))
    cohort_json = json.loads(rd(grab / "cohort.json"))
    identity = json.loads(rd(grab / "identity.json"))
    labels, checks = K.allocation(manifest, cohort_json)
    checks["grab_input_copy_matches"] = st.sha(grab / "inputs/early_cohort.json") == manifest["sha256"]
    arts = status.get("artifacts") or {}
    checks["cohort_artifact_hash"] = (arts.get("cohort") or {}).get("sha256") == st.sha(grab / "cohort.json")
    checks["identity_artifact_hash"] = (arts.get("identity") or {}).get("sha256") == st.sha(grab / "identity.json")
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise SystemExit(f"membership/receipt prerequisites failed (no outcome read): {failed}")

    # ---- 3. identity/path binding and staging of every outcome-bearing file ----
    stems = [Path(n).stem for n in st.listing("daily_listing", daily_dir, "*.parquet")]
    nested = [n for n in st.listing("daily_listing_recursive", daily_dir, "*.parquet", recursive=True) if "/" in n]
    binding, bind_counts = K.bind_daily_files(manifest, stems)
    bound = binding[binding["binding"] == "bound"]
    for r in bound.itertuples():
        for s in r.files:
            st.add(daily_dir / f"{s}.parquet", f"daily/{s}.parquet")
    ledger_listing = st.listing("ledger_listing", led, "*")
    for name in LG.READ_FILES:
        st.add(led / name, f"ledger/{name}")
    ledger_info = LG.verify_receipt(led, ledger_listing, rd)
    board_rel = (arts.get("leaderboard_snapshot") or {}).get("path")
    if board_rel:
        st.add(grab / board_rel, f"final_grab/{board_rel}")
    st.listing("final_grab_raw_board", grab / "raw" / "board", "*.json.gz")
    for e in status.get("requests", []):
        if e.get("class") == "board" or (e.get("class") == "profile" and e.get("raw_path")):
            st.add(grab / e["raw_path"], f"final_grab/{e['raw_path']}")
    for rec in identity.values():
        if rec.get("parsed_path"):
            st.add(grab / rec["parsed_path"], f"final_grab/{rec['parsed_path']}")

    # ---- 4. freeze before any outcome is parsed ----
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    code = code_manifest(head)
    freeze = {"written_before_outcome_reads": True, "gate": g, "code": code, "sources": st.manifest(),
              "manifest_E_sha256": manifest["sha256"], "membership_checks": checks,
              "labels_sha256": _sha(labels.to_csv(index=False).encode()),
              "binding_sha256": _sha(json.dumps(jsonable(binding.to_dict("records")), sort_keys=True).encode()),
              "binding_counts": bind_counts, "daily_nested_files_ignored": nested,
              "ledger_receipt": {k: v for k, v in ledger_info.items() if k != "build"},
              "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    (run_dir / "freeze.json").write_text(json.dumps(jsonable(freeze), indent=1, sort_keys=True) + "\n")
    log(f"froze {len(st.files)} source files in {run_dir}; code {head[:7]}")

    # ---- 5. outcomes, from the staged bytes only ----
    rec = C.load_board_receipts(grab, read=rd)
    census_gate = C.census_gate(rec)
    season_best = C.season_best_summary(census_gate["qualified"], census_gate, args.our_user_id)
    log(f"census gate: {census_gate['census']} {census_gate['failures']}")
    a = L.case_series(grab, board=rec["board"], cohort_json=cohort_json, identity=identity, status=status, read=rd)

    parts, obs_stats = [], {}
    for r in bound.itertuples():
        o = P.read_observations([daily_dir / f"{s}.parquet" for s in r.files], user_id=int(r.user_id),
                                source="daily", read=rd)
        obs_stats[int(r.user_id)] = {"rows": int(len(o)), "captures": int(o["captured_at"].nunique()) if len(o) else 0,
                                     "first_capture": o["captured_at"].min() if len(o) else None,
                                     "last_capture": o["captured_at"].max() if len(o) else None}
        if len(o):
            parts.append(o)
    res = P.resolve(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    ownership = P.ownership_conflict_users(res)
    available = {u for u, s_ in obs_stats.items() if s_["rows"] and u not in ownership}
    start, end = Q.PRIMARY
    windows = {u: S.window_summary(res.rounds[res.rounds["user_id"] == u], start, end) for u in sorted(available)}
    av = Q.availability(manifest["members"], labels, binding, ownership_quarantined=ownership, res=res,
                        obs_stats=obs_stats, start=start, end=end, windows=windows)
    e = Q.e_arm(res.slots, available, start, end)

    ledger, contest, ledger_bound = LG.load_bound(led, rd, ledger_info)
    o, ours_exc = Q.ours_slots(ledger, contest, start, end)
    tables = Q.date_tables(e, o, n_resamples=args.n_resamples, n_members=manifest["n"],
                           calendar_dates=(end - start).days + 1)
    e_excl = Q.e_exclusions(res.slots, available, start, end)
    log(f"primary: E {len(e)} slots, ours {len(o)} slots, {tables['all_observed_dates']['dates']} dates")

    fg = K.final_grab_status(labels, identity, grab, read=rd)
    ext_parts = [P.read_observations([grab / r.parsed_path], user_id=int(r.user_id), source="final_grab", read=rd)
                 for r in fg[fg["usable"]].itertuples()]
    ext_res = P.resolve(pd.concat(ext_parts, ignore_index=True) if ext_parts else pd.DataFrame())
    ext = Q.extension(fg, ext_res.slots, n_members=manifest["n"])

    # ---- 6. the sources must be exactly the frozen ones ----
    problems = st.verify()
    if problems:
        raise SystemExit(f"sources changed since the freeze; nothing published: {problems}")

    # ---- 7. outputs ----
    status_counts = pd.Series([w["status"] for w in windows.values()]).value_counts().to_dict() if windows else {}
    per_user = av[av["window_graded_slots"] > 0]["window_graded_slots"]
    results = {
        "schema": "w21_w22_field_products_v2", "code": head, "x24_commit": X24_COMMIT, "x25_commit": X25_COMMIT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "inputs": {"data_root": str(data), "final_grab": args.final_grab, "ledger_dir": str(led),
                   "early_manifest": str(args.early_manifest), "our_user_id": args.our_user_id,
                   "n_resamples": args.n_resamples},
        "w21": {"census_gate": {k: v for k, v in census_gate.items() if k != "qualified"},
                "season_best": season_best, "case_series_A": a["summary"]},
        "w22": {
            "manifest_E": {"n": manifest["n"], "sha256": manifest["sha256"], "definition": manifest["definition"],
                           "frozen_at": manifest["frozen_at"], "source_fixture_sha256": manifest["source_fixture_sha256"],
                           "checks": checks, "allocation_labels": labels["allocation"].value_counts().to_dict(),
                           "note": "A/B/E_in_A/E_unfetched/B_shortfall are acquisition labels, not cohorts"},
            "identity": {**bind_counts, "ownership_quarantined": sorted(ownership),
                         "daily_nested_files_ignored": len(nested), "attribution_basis": Q.ATTRIBUTION_BASIS,
                         "per_batch_identity_established": False, "per_batch_identity": PER_BATCH_IDENTITY,
                         "rule": "bind only by a unique 5/01 username sanitization whose file stems are the member's "
                                 "own raw or sanitized name; otherwise quarantine"},
            "availability": {"members": int(len(av)), "daily_history": av["daily_history"].value_counts().to_dict(),
                             "window_activity": av["window_activity"].value_counts().to_dict(),
                             "by_allocation": av.groupby("allocation")["daily_history"].value_counts().unstack(
                                 fill_value=0).to_dict("index")},
            "primary": {"window": [start.isoformat(), end.isoformat()], "ours": ours_exc, "ledger": ledger_bound,
                        "E_exclusions": e_excl, "tables": {k: v for k, v in tables.items() if k != "per_date"},
                        "per_user_graded_slots": {"users": int(len(per_user)),
                                                  "min": int(per_user.min()) if len(per_user) else None,
                                                  "median": float(per_user.median()) if len(per_user) else None,
                                                  "max": int(per_user.max()) if len(per_user) else None},
                        "window_streaks": {"status": status_counts,
                                           "rule": "observed-segment lower bounds only (carried-in streak excluded); "
                                                   "exact maxima unavailable without a completeness witness"}},
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
    _write_table(binding, run_dir / "w22_binding.parquet")
    _write_table(labels, run_dir / "w22_labels.parquet")
    _write_table(e, run_dir / "w22_e_slots.parquet")
    _write_table(o, run_dir / "w22_ours_slots.parquet")
    _write_table(tables["per_date"].reset_index().rename(columns={"index": "date"}), run_dir / "w22_per_date.parquet")
    _write_table(fg, run_dir / "w22_final_grab_status.parquet")
    for name, df in (("w22_daily_slots", res.slots), ("w22_daily_rounds", res.rounds),
                     ("w22_extension_slots", ext_res.slots)):
        _write_table(df if len(df) else pd.DataFrame({"empty": []}), run_dir / f"{name}.parquet")
    manifest_out = {"sources": st.manifest(), "agrees_with_freeze": jsonable(st.manifest()) == jsonable(freeze["sources"]),
                    "code": code, "gate": g,
                    "membership": {"manifest_E_sha256": manifest["sha256"], "labels_sha256": freeze["labels_sha256"],
                                   "binding_sha256": freeze["binding_sha256"]},
                    "outputs": {p.name: _sha(p.read_bytes()) for p in sorted(run_dir.iterdir())
                                if p.name != "manifest.json"}}
    (run_dir / "manifest.json").write_text(json.dumps(jsonable(manifest_out), indent=1, sort_keys=True) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
