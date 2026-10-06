"""C1 rank 3, T5: the historical count build (registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4 and
X-E1; plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`; code review r1 F1, F6-F9).

1. **Admission** (`scripts.audit.c1.admission`): the reviewed commit, anchored to an archived exact SIGN; X-34
   published in its own commit, unchanged since, and citing the review, the reference commit and the
   "historical count build" scope; the executable closure (the registration included) as reviewed; modules loaded
   from this checkout; a clean tracked tree; no claimed earlier run without Eric's exact INVALIDATE ruling. The
   roots are fixed (`DATA`), with no CLI override.
2. **Metadata-only pre-manifest, before any outcome-bearing byte is read:**
   - the code, the admission record, the closure's git object ids, and the registration's and review report's
     sha256;
   - the contract: the PA definition, completion states, cap, smoothing, stop rate, eligibility rule, the supported
     parquet schema and feed formats, and the environment;
   - the canonical receipt-bound feed inventory (written in full to `inventory.json` and hashed). Receipts are
     validated first (r1 F7):
     - a canonical `<season>/<gamePk>.json[.gz]` path inside the feed root;
     - exact positive ids and 64-hex digests;
     - no conflicting stored records;
     - no unresolved request intent;
   - each PA parquet's size and mtime.
3. **Claim:** a durable `CLAIM.json` in a run directory whose parent entry is fsynced.
4. **Pins:** each PA parquet is read once, hashed, and the pins written (`pins.json`) before any parse. The parquet is
   then parsed from those bytes.
   - **Schema (r1 F6):** int64 `game_pk` / `batter_id` / `season`, string `date`, bool `is_home`, no
     `is_resumed_portion`.
   - **Rows:** no nulls; positive ids; the file's season.
   - **Games:** one date per game; a game appearing in two seasons refuses.
5. **One pass** over every receipt-bound feed, under the acquisition's writer lock.
   - **Hashes:** the stored and decoded sha256 are checked (gzip or plain JSON by the path's suffix). A mismatch,
     or a receipt-named file that is missing, stops the run with a durable `STOPPED_provenance.txt` naming the
     source and the unprocessed count; the claim remains.
   - **Ineligible games** (no parquet rows) are hash-checked and counted, not parsed.
   - **Eligible games:** T1 extract and T2 certify. A metadata exception becomes a quarantine reason for that
     game (r1 F9).
6. **Census:** more than 1% of eligible games quarantined writes `STOPPED_quarantine.txt` and stops.
7. **Outputs** under `data/hetzner_results/c1/r3/count_build/<sha>-<run>/` (restic-backed), each durable, with their
   sha256 in `results.json`: `count_table.json`, `bf_starts.json`, `census.json` and `provenance.json`. Each start
   carries its source receipt's decoded hash, retrieval time (`ended_utc`) and attempt id.

    .venv/bin/python -m scripts.audit.c1_r3.count_build
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from bts.data.schema import PA_ENDING_EVENTS
from scripts.audit.c1 import admission as A
from scripts.audit.c1_r3 import acquire as aq
from scripts.audit.c1_r3 import count_bf as B
from scripts.audit.c1_r3 import count_meta as M
from scripts.audit.c1_r3 import count_table as T
from scripts.audit.c1_r3 import count_verify as V

REPO = A.REPO
DATA = Path.home() / "projects" / "bts" / "data"
SEASONS = (2021, 2022, 2023, 2024, 2025)
ADMISSION_REL = "scripts/audit/c1_r3/admission.json"
REGISTER_REL = "docs/audit/2026-09-22-exposure-register.md"
REGISTRATION = "docs/sota_audit/2026-10-04-prereg-c1-pa-count.md"
EXPOSURE_ROW = "X-34"
SCOPE = "historical count build"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c1_r3", "src/bts",
           "pyproject.toml", "uv.lock", REGISTRATION)
PA_SCHEMA = {"game_pk": "int64", "batter_id": "int64", "season": "int64", "is_home": "bool",
             "date": ("string", "large_string")}
PATH_RE = re.compile(r"^(?P<season>20\d\d)/(?P<pk>[1-9]\d*)\.json(?P<gz>\.gz)?$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ProvenanceError = A.ProvenanceError


def admission_gate() -> str:
    head, reasons = A.admission_check(REPO, json.loads((REPO / ADMISSION_REL).read_text()), closure=CLOSURE,
                                      admission_rel=ADMISSION_REL, register_rel=REGISTER_REL, exposure_row=EXPOSURE_ROW,
                                      scope_phrase=SCOPE)
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head


def dirty_tree() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def feed_inventory(recs: list[dict], feeds_dir: Path) -> list[dict]:
    """The canonical receipt-bound feed inventory (r1 F7). Refuses unresolved intents, malformed or non-canonical
    stored records, and conflicting records for one game."""
    open_ = aq.unresolved_intents(recs)
    if open_:
        raise ProvenanceError(f"unresolved acquisition intents ({len(open_)}): reconcile before any build")
    by_pk: dict = {}
    root = feeds_dir.resolve()
    for r in recs:
        if not (r.get("kind") == "completion" and r.get("outcome") == "stored" and r.get("kind_of", "feed") == "feed"):
            continue
        pk, path = r.get("gamePk"), r.get("stored_path")
        m = PATH_RE.match(path) if isinstance(path, str) else None
        if not (type(pk) is int and pk > 0 and m and int(m.group("pk")) == pk):
            raise ProvenanceError(f"a stored receipt with a non-canonical path or id: {path!r} / {pk!r}")
        if not (HEX64.match(str(r.get("stored_sha256"))) and HEX64.match(str(r.get("decoded_sha256")))):
            raise ProvenanceError(f"{path}: malformed digests")
        if not (root / path).resolve().is_relative_to(root):
            raise ProvenanceError(f"{path}: outside the feed root")
        entry = {"season": int(m.group("season")), "pk": pk, "path": path, "gz": bool(m.group("gz")),
                 "stored_sha256": r["stored_sha256"], "decoded_sha256": r["decoded_sha256"],
                 "attempt_id": r.get("attempt_id"), "retrieved_utc": r.get("ended_utc") or r.get("at_utc")}
        old = by_pk.get(pk)
        key = ("path", "stored_sha256", "decoded_sha256")
        if old is not None and any(old[k] != entry[k] for k in key):
            raise ProvenanceError(f"conflicting stored receipts for game {pk}")
        by_pk.setdefault(pk, entry)
    return sorted(by_pk.values(), key=lambda e: (e["season"], e["pk"]))


def check_schema(b: bytes, name: str) -> None:
    sch = pq.read_schema(io.BytesIO(b))
    types = {f.name: str(f.type) for f in sch}
    if "is_resumed_portion" in types:
        raise ProvenanceError(f"{name}: an enriched schema (is_resumed_portion) is not the declared legacy input")
    for col, want in PA_SCHEMA.items():
        if types.get(col) not in ((want,) if isinstance(want, str) else want):
            raise ProvenanceError(f"{name}: column {col} has type {types.get(col)!r}, not {want}")


def parquet_games(df: pd.DataFrame, season: int, name: str) -> tuple[dict, dict]:
    """Per game: Counter((is_home, batter_id) -> PAs) and the game's date. Every row must be valid (r1 F6)."""
    cols = list(PA_SCHEMA)
    if df[cols].isna().any().any():
        raise ProvenanceError(f"{name}: null values in {cols}")
    if not ((df["game_pk"] > 0).all() and (df["batter_id"] > 0).all() and (df["season"] == season).all()):
        raise ProvenanceError(f"{name}: non-positive ids or rows outside season {season}")
    dates = df.groupby("game_pk")["date"].nunique()
    if (dates != 1).any():
        raise ProvenanceError(f"{name}: games with more than one date: {list(dates[dates != 1].index[:3])}")
    counts: dict = {}
    for (pk, home, batter), n in df.groupby(["game_pk", "is_home", "batter_id"]).size().items():
        counts.setdefault(int(pk), Counter())[(bool(home), int(batter))] = int(n)
    first = df.drop_duplicates("game_pk").set_index("game_pk")["date"]
    return counts, {int(k): str(v) for k, v in first.items()}


def _write_json(path: Path, obj) -> str:
    data = (json.dumps(obj, indent=1, default=str) + "\n").encode()
    A.durable_write(path, data)
    return hashlib.sha256(data).hexdigest()


def contract() -> dict:
    return {"registration": REGISTRATION, "registration_sha256": hashlib.sha256((REPO / REGISTRATION).read_bytes()).hexdigest(),
            "pa_ending_events_sha256": hashlib.sha256(json.dumps(sorted(PA_ENDING_EVENTS)).encode()).hexdigest(),
            "completed_states": M.COMPLETED, "cap": T.CAP, "smoothing": "add-one over categories 1..8",
            "conditioning": "N >= 1", "stop_rate": V.STOP_RATE, "eligibility": "games in the PA parquets (production's "
            "regular-season set; 7-inning COVID doubleheaders already dropped)", "parquet_schema": PA_SCHEMA,
            "feed_formats": ["json", "json.gz"],
            "recipe": "not applicable: the count build fits no model and serves nothing",
            "environment": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__}}


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)          # no options: the roots are fixed
    log = lambda m: print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    feeds_dir, acq_out, pa_dir = DATA / "raw_c1", DATA / "hetzner_results" / "c1" / "r3", DATA / "processed"
    run_root = acq_out / "count_build"
    head = admission_gate()
    register_text = (REPO / REGISTER_REL).read_text()
    dirty = dirty_tree()
    if dirty:
        raise SystemExit(f"refusing: tracked changes in the executing tree: {dirty[:5]}")
    foreign = A.foreign_imports(REPO)
    if foreign:
        raise SystemExit(f"refusing: modules loaded from outside this checkout: {foreign[:5]}")
    with A.admission_lock(run_root):
        blocked = A.claimed_runs(run_root, register_text)
        if blocked:
            raise SystemExit(f"refusing: earlier claimed runs without Eric's invalidation: {blocked}")
        with aq.writer_lock(acq_out):
            return _run(head, run_root, feeds_dir, acq_out, pa_dir, log)


def _run(head, run_root, feeds_dir, acq_out, pa_dir, log) -> int:
    # 2. metadata-only pre-manifest (receipts and file metadata; no outcome-bearing byte)
    inventory = [e for e in feed_inventory(aq.read_receipts(acq_out), feeds_dir) if e["season"] in SEASONS]
    adm = json.loads((REPO / ADMISSION_REL).read_text())
    run_dir = A.make_run_dir(run_root, f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    inv_sha = _write_json(run_dir / "inventory.json", inventory)
    report = adm.get("review_report")
    _write_json(run_dir / "manifest.json", {
        "code": head, "admission": adm,
        "closure": {c: A._git(REPO, "rev-parse", f"HEAD:{c}", check=False).stdout.strip() for c in CLOSURE},
        "review_report_sha256": hashlib.sha256((REPO / report).read_bytes()).hexdigest() if report else None,
        "contract": contract(), "seasons": SEASONS,
        "inventory": {"count": len(inventory), "sha256": inv_sha},
        "pa_parquets": {f"pa_{s}.parquet": {"bytes": (pa_dir / f"pa_{s}.parquet").stat().st_size,
                                            "mtime": (pa_dir / f"pa_{s}.parquet").stat().st_mtime} for s in SEASONS},
        "schedules": "not used (the 10/04 schedules carry no receipts; see the cycle index)"})

    # 3. claim, then 4. pins before any parse
    A.write_claim(run_dir, head)
    log(f"claimed {run_dir.name}; {len(inventory)} receipt-bound feeds")
    blobs = {s: (pa_dir / f"pa_{s}.parquet").read_bytes() for s in SEASONS}
    _write_json(run_dir / "pins.json", {f"pa_{s}.parquet": hashlib.sha256(b).hexdigest() for s, b in blobs.items()})
    counts, dates = {}, {}
    for s, b in blobs.items():
        check_schema(b, f"pa_{s}.parquet")
        c, d = parquet_games(pd.read_parquet(io.BytesIO(b), columns=list(PA_SCHEMA)), s, f"pa_{s}.parquet")
        dup = set(c) & set(counts)
        if dup:
            raise ProvenanceError(f"games in two seasonal parquets: {sorted(dup)[:3]}")
        counts.update(c)
        dates.update(d)
    eligible = set(counts)

    # 5. one pass
    results, rows, starts, provenance, ineligible = {}, [], [], {}, set()
    for n, e in enumerate(inventory):
        f = feeds_dir / e["path"]
        try:
            stored = f.read_bytes()
        except OSError as err:
            (run_dir / "STOPPED_provenance.txt").write_text(f"{e['path']}: {err}; {len(inventory) - n} unprocessed\n")
            raise ProvenanceError(f"{e['path']}: the receipt-named file cannot be read") from err
        if hashlib.sha256(stored).hexdigest() != e["stored_sha256"]:
            (run_dir / "STOPPED_provenance.txt").write_text(f"{e['path']}: stored bytes off their receipt; "
                                                            f"{len(inventory) - n} unprocessed\n")
            raise ProvenanceError(f"{e['path']}: stored bytes do not match their receipt")
        body = gzip.decompress(stored) if e["gz"] else stored
        if hashlib.sha256(body).hexdigest() != e["decoded_sha256"]:
            (run_dir / "STOPPED_provenance.txt").write_text(f"{e['path']}: decoded bytes off their receipt; "
                                                            f"{len(inventory) - n} unprocessed\n")
            raise ProvenanceError(f"{e['path']}: decoded bytes do not match their receipt")
        pk = e["pk"]
        if pk not in eligible:
            ineligible.add(pk)
            continue
        try:
            meta = M.extract(json.loads(body))
            reasons = V.verify_game(meta, pk=pk, season=e["season"], parquet=counts.get(pk), parquet_date=dates.get(pk))
        except Exception as err:  # noqa: BLE001 - a malformed eligible feed is quarantined, never dropped (r1 F9)
            meta, reasons = None, [f"metadata error: {type(err).__name__}: {err}"[:300]]
        results[pk] = reasons
        provenance[str(pk)] = {"feed_timestamp": meta.feed_timestamp if meta else None,
                               "decoded_sha256": e["decoded_sha256"], "retrieved_utc": e["retrieved_utc"],
                               "attempt_id": e["attempt_id"], "certified": not reasons}
        if not reasons:
            rows += T.starter_counts(meta)
            for st in B.starts(meta):
                starts.append({**st, "box_batters_faced": meta.box_batters_faced[st["side"]],
                               "source_decoded_sha256": e["decoded_sha256"], "retrieved_utc": e["retrieved_utc"]})

    # 6. census
    census = V.census(results, eligible=eligible, feeds_without_parquet=ineligible)
    census_sha = _write_json(run_dir / "census.json",
                             {**census, "quarantined": {str(k): v for k, v in census["quarantined"].items()}})
    log(f"census: {census['certified']}/{census['eligible']} certified, {len(census['quarantined'])} quarantined")
    if census["stop"]:
        (run_dir / "STOPPED_quarantine.txt").write_text(f"{census['rate']:.4f} of eligible games quarantined (> 1%)\n")
        return 2

    # 7. outputs
    outputs = {"count_table.json": _write_json(run_dir / "count_table.json", T.count_table(rows)),
               "bf_starts.json": _write_json(run_dir / "bf_starts.json",
                                             {"league_median_bf": B.league_median(starts), "starts": starts}),
               "census.json": census_sha,
               "provenance.json": _write_json(run_dir / "provenance.json", provenance)}
    _write_json(run_dir / "results.json", {"code": head, "outputs": outputs, "census": {
        k: census[k] for k in ("eligible", "certified", "rate", "ineligible_feeds", "reasons")} | {
        "quarantined": len(census["quarantined"])}})
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
