"""C1 rank 3, T5: the historical count build (registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4;
plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`).

1. **Admission** (`scripts.audit.c1.admission`):
   - the reviewed commit, anchored to an archived Codex SIGN;
   - X-34 published in its own commit and unchanged since;
   - the executable closure as reviewed;
   - modules loaded only from this checkout;
   - a clean tracked tree;
   - no claimed earlier run without Eric's exact INVALIDATE ruling.

   The roots are fixed (`DATA`), with no CLI override.
2. **Outcome-free freeze manifest, before any feed is read:**
   - the code commit, the admission record and the closure's git object ids;
   - the receipt-bound feed inventory (every `stored` feed receipt's path and digests);
   - each PA parquet's digest. Each is read once; its bytes are parsed later from the same buffer.

   The five 10/04 schedules carry no receipts; the build does not use them.
3. **Claim:** a durable `CLAIM.json` precedes the first feed read.
4. **One pass** over every receipt-bound feed, under the acquisition's writer lock:
   - the stored bytes must match the receipt's stored sha256, and the decoded bytes its decoded sha256;
   - then T1 extract, T2 certify, and T3/T4 rows from certified games only;
   - each game's producer version (`metaData.timeStamp`) is kept in `provenance.json`. Chronological order is
     certified by a strictly increasing `atBatIndex` (T1).

   Eligible games are those in the PA parquets (`game_pk`, `batter_id`, `is_home`; outcome-free columns).
5. **Census:** more than 1% of eligible games quarantined writes `STOPPED_quarantine.txt` and stops. The claim
   still blocks a rerun.
6. **Outputs** under `data/hetzner_results/c1/r3/count_build/<sha>-<run>/` (restic-backed), each durable, with
   their sha256 in `results.json`: `count_table.json`, `bf_starts.json` (with the league median), `census.json`.

    .venv/bin/python -m scripts.audit.c1_r3.count_build
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
EXPOSURE_ROW = "X-34"
CLOSURE = ("scripts/__init__.py", "scripts/audit/__init__.py", "scripts/audit/c1", "scripts/audit/c1_r3", "src/bts",
           "pyproject.toml", "uv.lock")
PA_COLS = ["game_pk", "batter_id", "is_home"]
ProvenanceError = A.ProvenanceError


def admission_gate() -> str:
    head, reasons = A.admission_check(REPO, json.loads((REPO / ADMISSION_REL).read_text()), closure=CLOSURE,
                                      admission_rel=ADMISSION_REL, register_rel=REGISTER_REL, exposure_row=EXPOSURE_ROW)
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head


def dirty_tree() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def feed_inventory(acq_out: Path) -> dict:
    """{(season, gamePk): latest stored feed receipt}: the receipt-bound feeds, nothing else."""
    inv = {}
    for r in aq.read_receipts(acq_out):
        if r.get("kind") == "completion" and r.get("outcome") == "stored" and r.get("kind_of", "feed") == "feed":
            season = int(str(r["stored_path"]).split("/")[0])
            inv[(season, int(r["gamePk"]))] = r
    return inv


def parquet_counts(df: pd.DataFrame) -> dict:
    out: dict = {}
    for (pk, home, batter), n in df.groupby(["game_pk", "is_home", "batter_id"]).size().items():
        out.setdefault(int(pk), Counter())[(bool(home), int(batter))] = int(n)
    return out


def _write_json(path: Path, obj) -> str:
    data = (json.dumps(obj, indent=1, default=str) + "\n").encode()
    A.durable_write(path, data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)          # no options: the roots are fixed
    log = lambda m: print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    feeds_dir, acq_out, pa_dir = DATA / "raw_c1", DATA / "hetzner_results" / "c1" / "r3", DATA / "processed"
    run_root = acq_out / "count_build"

    # 1. admission
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
        run_dir = run_root / f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        run_dir.mkdir(parents=True)
        return _run(head, run_dir, feeds_dir, acq_out, pa_dir, log)


def _run(head, run_dir, feeds_dir, acq_out, pa_dir, log) -> int:
    # 2. outcome-free freeze manifest
    with aq.writer_lock(acq_out):
        inventory = {k: v for k, v in feed_inventory(acq_out).items() if k[0] in SEASONS}
        pa_bytes = {}
        for s in SEASONS:
            b = (pa_dir / f"pa_{s}.parquet").read_bytes()
            pa_bytes[s] = (b, hashlib.sha256(b).hexdigest())
        listing = sorted(f"{k[0]}/{k[1]} {v['stored_sha256']} {v['decoded_sha256']}" for k, v in inventory.items())
        manifest = {"code": head, "admission": json.loads((REPO / ADMISSION_REL).read_text()),
                    "closure": {c: A._git(REPO, "rev-parse", f"HEAD:{c}", check=False).stdout.strip() for c in CLOSURE},
                    "seasons": SEASONS,
                    "feeds": {"count": len(inventory),
                              "inventory_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest()},
                    "pa_parquets": {f"pa_{s}.parquet": h for s, (_, h) in pa_bytes.items()},
                    "schedules": "not used (the 10/04 schedules carry no receipts; see the cycle index)"}
        _write_json(run_dir / "manifest.json", manifest)
        log(f"manifest written: {len(inventory)} receipt-bound feeds, {len(pa_bytes)} PA parquets")

        # 3. claim, 4. one pass
        A.write_claim(run_dir, head)
        counts, results, rows, starts, provenance = {}, {}, [], [], {}
        for s, (b, _) in pa_bytes.items():
            counts.update(parquet_counts(pd.read_parquet(io.BytesIO(b), columns=PA_COLS)))
        for (season, pk), r in sorted(inventory.items()):
            stored = (feeds_dir / r["stored_path"]).read_bytes()
            if hashlib.sha256(stored).hexdigest() != r["stored_sha256"]:
                raise ProvenanceError(f"{r['stored_path']}: stored bytes do not match their receipt")
            body = gzip.decompress(stored)
            if hashlib.sha256(body).hexdigest() != r["decoded_sha256"]:
                raise ProvenanceError(f"{r['stored_path']}: decoded bytes do not match their receipt")
            meta = M.extract(json.loads(body))
            results[pk] = V.verify_game(meta, pk=pk, season=season, parquet=counts.get(pk))
            provenance[str(pk)] = {"feed_timestamp": meta.feed_timestamp, "decoded_sha256": r["decoded_sha256"],
                                   "certified": pk in counts and not results[pk]}
            if pk in counts and not results[pk]:
                rows += T.starter_counts(meta)
                starts += B.starts(meta)
    eligible = set(counts)
    census = V.census({pk: rs for pk, rs in results.items() if pk in eligible}, eligible=eligible,
                      feeds_without_parquet={pk for pk in results if pk not in eligible})
    census_sha = _write_json(run_dir / "census.json", {**census, "quarantined": {str(k): v for k, v in census["quarantined"].items()}})
    log(f"census: {census['certified']}/{census['eligible']} certified, {len(census['quarantined'])} quarantined")
    if census["stop"]:
        (run_dir / "STOPPED_quarantine.txt").write_text(f"{census['rate']:.4f} of eligible games quarantined (> 1%)\n")
        return 2

    # 6. outputs
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
