"""C1 rank 3, T5: the historical count build (registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4 and
X-E1; plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`; code review r1 F1, F6-F9).

1. **Admission** (`scripts.audit.c1.admission`): the reviewed commit, anchored to an archived exact SIGN; X-34
   published in its own commit, unchanged since, and citing the review, the reference commit and the
   "historical count build" scope; the executable closure (the registration included) as reviewed; modules loaded
   from this checkout; a clean tracked tree; no claimed earlier run without Eric's exact INVALIDATE ruling. The
   roots are fixed (`DATA`), with no CLI override.
   **One admitted record (r3 R3-2):** `admission.json` is read once, by the gate. The run receives that same record
   and the gate's accepted report identity; nothing reloads it.
2. **Metadata-only pre-manifest, before any outcome-bearing byte is read:**
   - the accepted review report's exact identity (from the exposure commit), and the **expected** parquet pins from
     the admitted record's `input_pins` (a hash-only capture after review, cited by digest in X-34; r2 N6);
   - the code, the admission record, the closure's git object ids, and the registration's and review report's
     sha256;
   - the contract: the PA definition, completion states, cap, smoothing, stop rate, eligibility rule, the supported
     parquet schema and feed formats, and the environment;
   - the canonical receipt-bound feed inventory (written in full to `inventory.json` and hashed). Receipts are
     validated first (r1 F7, r3 R3-5):
     - known receipt kinds and outcomes only;
     - a string attempt id on every intent and completion, with no conflicting duplicate for one attempt;
     - a canonical `<season>/<gamePk>.json[.gz]` path inside the feed root;
     - exact positive ids and 64-hex digests;
     - no conflicting stored records;
     - no unresolved request intent;
     - a linked response (`from_attempt_id`) attests the same decoded bytes as the stored record;
   - each PA parquet's size and mtime.
3. **Claim:** a durable `CLAIM.json` in a run directory whose parent entry is fsynced.
4. **Pins:** each PA parquet is read once and checked against its expected pin before any parse, then parsed from
   those bytes.
   - **Schema (r1 F6):** int64 `game_pk` / `batter_id` / `season`, string `date`, bool `is_home`, no
     `is_resumed_portion`.
   - **Rows:** no nulls; positive ids; the file's season.
   - **Games:** one date per game; a game appearing in two seasons refuses.
5. **One pass** over every receipt-bound feed, under the acquisition's writer lock.
   - **Hashes:** the stored and decoded sha256 are checked (gzip or plain JSON by the path's suffix). A mismatch,
     a decode failure or a receipt-named file that is missing stops the run. So does any parquet pin, schema or row
     failure. Each leaves a durable `STOPPED_incomplete.json` naming the source and the unprocessed games (r2 N8);
     the claim remains.
   - **Availability:** a receipt without a valid retrieval time, or with one before its request started,
     quarantines its eligible game (r2 N5, r3 R3-5).
   - **Ineligible games** (no parquet rows) are hash-checked and counted, not parsed.
   - **Eligible games:** T1 extract and T2 certify. A metadata exception becomes a quarantine reason for that
     game (r1 F9).
6. **Census:** more than 1% of eligible games quarantined writes a durable `STOPPED_quarantine.json` and stops.
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
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ProvenanceError = A.ProvenanceError


def pins_digest(pins: dict) -> str:
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def load_admission() -> tuple[dict, str]:
    """The admission record and the sha256 of the exact bytes it was parsed from."""
    b = (REPO / ADMISSION_REL).read_bytes()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def accepted_identity(adm: dict) -> dict:
    return A.accepted_identity(REPO, adm)


def admission_gate() -> tuple[str, dict, dict]:
    """The shared gate, plus the expected input pins (r2 N6): admission.json's `input_pins` must name exactly the five
    PA parquets with 64-hex digests, and X-34 must cite their digest.

    Returns HEAD, the admitted record and its accepted identity. The run consumes exactly these: the record is read
    once, here (r3 R3-2)."""
    adm, adm_sha = load_admission()
    pins = adm.get("input_pins")
    want = {f"pa_{s}.parquet" for s in SEASONS}
    if not (isinstance(pins, dict) and set(pins) == want and all(HEX64.match(str(v)) for v in pins.values())):
        raise SystemExit(f"refusing: admission input_pins must give a sha256 for exactly {sorted(want)}")
    head, reasons = A.admission_check(REPO, adm, closure=CLOSURE, admission_rel=ADMISSION_REL, register_rel=REGISTER_REL,
                                      exposure_row=EXPOSURE_ROW, scope_phrase=SCOPE, inputs_digest=pins_digest(pins))
    if reasons:
        raise SystemExit("refusing: " + "; ".join(reasons))
    return head, adm, {**accepted_identity(adm), "admission_sha256": adm_sha}


def dirty_tree() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def _aware(v) -> datetime | None:
    if not isinstance(v, str):
        return None
    try:
        t = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _utc(v) -> str | None:
    """A timezone-aware ISO timestamp, or None."""
    return v if _aware(v) is not None else None


KNOWN_RECEIPTS = {"intent": None, "validation": {"invalid"}, "reconciled_orphan": None,
                  "completion": {"response", "stored", "rate_limited", "http_error", "network_error"}}


def _receipt_problems(recs: list) -> None:
    """Known kinds and outcomes only (r3 R3-5); refuse anything else."""
    for r in recs:
        kind = r.get("kind") if isinstance(r, dict) else None
        if not (isinstance(kind, str) and kind in KNOWN_RECEIPTS):
            raise ProvenanceError(f"an unknown receipt kind: {kind!r}")
        outcomes = KNOWN_RECEIPTS[kind]
        if outcomes is not None and not (isinstance(r.get("outcome"), str) and r.get("outcome") in outcomes):
            raise ProvenanceError(f"an unknown {kind} outcome: {r.get('outcome')!r}")


REQUEST_KINDS = ("intent", "completion")


def _pos_int(v) -> bool:
    return type(v) is int and v > 0


def _typed_requests(recs: list) -> None:
    """Every intent and completion's identity is typed before any comparison, lookup or join (r4 R4-1; C2 review r1
    C2R1-1): a string attempt id; `kind_of` absent (the 10/04 legacy writer, a feed) or one of feed/schedule; a feed
    has an exact positive integer gamePk and no season (the writer's season is null or absent); a schedule has an
    exact positive integer season and no game id (null or absent); a string `from_attempt_id` when present. Python
    equality would otherwise let 7.0 or True stand in for 7 in a duplicate or a join, and a schedule record could
    smuggle a game id into a feed's reconciliation."""
    for r in recs:
        if r.get("kind") not in REQUEST_KINDS:
            continue
        aid = r.get("attempt_id")
        if not (isinstance(aid, str) and aid):
            raise ProvenanceError(f"a {r['kind']} receipt without a string attempt id: {aid!r}")
        kind_of = r.get("kind_of", "feed")
        if kind_of == "feed":
            if not _pos_int(r.get("gamePk")):
                raise ProvenanceError(f"attempt {aid}: a feed receipt's game id must be a positive int: {r.get('gamePk')!r}")
            if r.get("season") is not None:
                raise ProvenanceError(f"attempt {aid}: a feed receipt carries a season: {r.get('season')!r}")
        elif kind_of == "schedule":
            if not _pos_int(r.get("season")):
                raise ProvenanceError(f"attempt {aid}: a schedule receipt's season must be a positive int: {r.get('season')!r}")
            if r.get("gamePk") is not None:
                raise ProvenanceError(f"attempt {aid}: a schedule receipt carries a game id: {r.get('gamePk')!r}")
        else:
            raise ProvenanceError(f"attempt {aid}: an unsupported kind_of {kind_of!r}")
        if "from_attempt_id" in r and not (isinstance(r["from_attempt_id"], str) and r["from_attempt_id"]):
            raise ProvenanceError(f"attempt {aid}: from_attempt_id {r['from_attempt_id']!r} is not an attempt id")


def _canon(r: dict) -> str:
    return json.dumps(r, sort_keys=True)


def _request_identity(r: dict) -> tuple:
    """(request kind, primary id) of an already typed intent or completion: a feed's gamePk, a schedule's season."""
    kind_of = r.get("kind_of", "feed")
    return (kind_of, r["gamePk"] if kind_of == "feed" else r["season"])


def _witnesses(recs: list, label: str, pred) -> dict:
    """attempt_id -> record for one receipt class. A non-string attempt id, or two different records for one attempt,
    refuses (r3 R3-5: the last assignment used to win silently). Duplicates are compared as canonical JSON of the parsed
    records, so only a canonically identical repeat is tolerated (r4 R4-1: dict equality treats 7.0 and True as 7
    and 1); key order and whitespace are not compared."""
    out: dict = {}
    for r in recs:
        if not pred(r):
            continue
        aid = r.get("attempt_id")
        if not (isinstance(aid, str) and aid):
            raise ProvenanceError(f"a {label} receipt without a string attempt id: {aid!r}")
        if aid in out and _canon(out[aid]) != _canon(r):
            raise ProvenanceError(f"conflicting {label} receipts for attempt {aid}")
        out[aid] = r
    return out


def unresolved(recs: list[dict]) -> list:
    """Intents with no completion for the same attempt (r2 N5). Run after `_typed_requests`; a completion that shares
    an intent's attempt id but names a different request (kind, game or season) refuses rather than resolving it
    (C2 review r1 C2R1-1)."""
    intents = {r["attempt_id"]: r for r in recs if r.get("kind") == "intent"}
    done = set()
    for r in recs:
        if r.get("kind") != "completion":
            continue
        it = intents.get(r["attempt_id"])
        if it is not None and _request_identity(it) != _request_identity(r):
            raise ProvenanceError(f"the completion for attempt {r['attempt_id']} names a different request "
                                  f"{_request_identity(r)!r} than its intent {_request_identity(it)!r}")
        done.add(r["attempt_id"])
    return [aid for aid in intents if aid not in done]


def feed_inventory(recs: list[dict], feeds_dir: Path) -> list[dict]:
    """The canonical receipt-bound feed inventory (r1 F7, r2 N5).
    - **Refused:** unresolved intents; malformed or non-canonical stored records; conflicting records for one game.
    - **Request binding:** each stored record is bound to its request. The legacy layout shares the intent's
      attempt id; the newer layout reaches it through `from_attempt_id`. The intent must name the same game, and
      the request URL must be exactly that game's feed URL.
    - **Availability:** a missing or invalid retrieval time marks the entry unavailable, and an eligible game is
      then quarantined. It is never certified with a null."""
    _receipt_problems(recs)
    _typed_requests(recs)
    intents = _witnesses(recs, "intent", lambda r: r.get("kind") == "intent")
    # One completion per attempt across ALL terminal outcomes (r4 R4-1 B): a stored record and an http_error for the
    # same attempt conflict. The modern layout's stored record has its own store id, linked by from_attempt_id.
    completions = _witnesses(recs, "completion", lambda r: r.get("kind") == "completion")
    responses = {aid: r for aid, r in completions.items() if r.get("outcome") == "response"}
    open_ = unresolved(recs)
    if open_:
        raise ProvenanceError(f"unresolved acquisition intents ({len(open_)}): reconcile before any build")
    by_pk: dict = {}
    root = feeds_dir.resolve()
    for r in recs:
        if not (r.get("kind") == "completion" and r.get("outcome") == "stored" and r.get("kind_of", "feed") == "feed"):
            continue
        pk, path = r.get("gamePk"), r.get("stored_path")
        m = PATH_RE.fullmatch(path) if isinstance(path, str) else None
        if not (type(pk) is int and pk > 0 and m and int(m.group("pk")) == pk):
            raise ProvenanceError(f"a stored receipt with a non-canonical path or id: {path!r} / {pk!r}")
        if not (HEX64.match(str(r.get("stored_sha256"))) and HEX64.match(str(r.get("decoded_sha256")))):
            raise ProvenanceError(f"{path}: malformed digests")
        if not (root / path).resolve().is_relative_to(root):
            raise ProvenanceError(f"{path}: outside the feed root")
        if "from_attempt_id" in r:                                                          # the newer layout
            link = r["from_attempt_id"]
            if not (isinstance(link, str) and link):
                raise ProvenanceError(f"{path}: from_attempt_id {link!r} is not an attempt id")
            src = responses.get(link)
            if src is not None and src.get("decoded_sha256") != r["decoded_sha256"]:
                raise ProvenanceError(f"{path}: the linked response attests different decoded bytes")
        else:                                                                               # the legacy layout
            src = r
        aid = (src or {}).get("attempt_id")
        intent = intents.get(aid) if isinstance(aid, str) and aid else None
        url = FEED_URL.format(pk=pk)
        if not (src and intent and type(intent.get("gamePk")) is int and intent.get("gamePk") == pk
                and type(src.get("gamePk")) is int and src.get("gamePk") == pk
                and intent.get("url") == url and src.get("url") == url):
            raise ProvenanceError(f"{path}: the stored record is not bound to a request for game {pk}")
        retrieved = _utc(src.get("ended_utc"))
        started = _aware(intent.get("started_utc"))
        if retrieved is None:
            problem = "availability: the receipt has no valid retrieval time"
        elif started is not None and _aware(retrieved) < started:
            problem = "availability: the retrieval time precedes its request"
        else:
            problem = None
        entry = {"season": int(m.group("season")), "pk": pk, "path": path, "gz": bool(m.group("gz")),
                 "stored_sha256": r["stored_sha256"], "decoded_sha256": r["decoded_sha256"],
                 "attempt_id": aid, "url": url, "retrieved_utc": retrieved, "available": problem is None,
                 "availability_problem": problem}
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


def _stop(run_dir: Path, name: str, rec: dict) -> None:
    """A durable stopped/incomplete account (r2 N8)."""
    A.durable_write(run_dir / name, (json.dumps(rec, indent=1, default=str) + "\n").encode())


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
    head, adm, identity = admission_gate()
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
            return _run(head, adm, identity, run_root, feeds_dir, acq_out, pa_dir, log)


def _run(head, adm, identity, run_root, feeds_dir, acq_out, pa_dir, log) -> int:
    """`adm` and `identity` are the gate's admitted record and accepted identity, never a re-read (r3 R3-2)."""
    # 2. metadata-only pre-manifest: receipts, file metadata and the EXPECTED parquet pins (no outcome-bearing byte)
    inventory = [e for e in feed_inventory(aq.read_receipts(acq_out), feeds_dir) if e["season"] in SEASONS]
    run_dir = A.make_run_dir(run_root, f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    inv_sha = _write_json(run_dir / "inventory.json", inventory)
    _write_json(run_dir / "manifest.json", {
        "code": head, "admission": adm, "accepted_review": identity,
        "closure": {c: A._git(REPO, "rev-parse", f"HEAD:{c}", check=False).stdout.strip() for c in CLOSURE},
        "contract": contract(), "seasons": SEASONS,
        "inventory": {"count": len(inventory), "sha256": inv_sha},
        "expected_pa_pins": adm["input_pins"], "expected_pa_pins_digest": pins_digest(adm["input_pins"]),
        "pa_parquets": {f"pa_{s}.parquet": {"bytes": (pa_dir / f"pa_{s}.parquet").stat().st_size,
                                            "mtime": (pa_dir / f"pa_{s}.parquet").stat().st_mtime} for s in SEASONS},
        "schedules": "not used (the 10/04 schedules carry no receipts; see the cycle index)"})

    # 3. claim; 4. each parquet read once, checked against its expected pin before any parse
    A.write_claim(run_dir, head)
    log(f"claimed {run_dir.name}; {len(inventory)} receipt-bound feeds")
    pending_pks = [e["pk"] for e in inventory]

    def fatal(source: str, err: Exception, done: int):
        _stop(run_dir, "STOPPED_incomplete.json", {"source": source, "error": f"{type(err).__name__}: {err}"[:500],
                                                   "processed_feeds": done, "unprocessed_feeds": pending_pks[done:]})
        return ProvenanceError(f"{source}: {err}")

    counts, dates = {}, {}
    for s in SEASONS:
        name = f"pa_{s}.parquet"
        try:
            b, _ = A.read_pinned(pa_dir / name, adm["input_pins"][name])
            check_schema(b, name)
            c, d = parquet_games(pd.read_parquet(io.BytesIO(b), columns=list(PA_SCHEMA)), s, name)
            dup = set(c) & set(counts)
            if dup:
                raise ProvenanceError(f"games in two seasonal parquets: {sorted(dup)[:3]}")
        except Exception as err:  # noqa: BLE001 - any parquet failure is a durable, accountable stop
            raise fatal(name, err, 0) from err
        counts.update(c)
        dates.update(d)
    eligible = set(counts)

    # 5. one pass
    results, rows, starts, provenance, ineligible = {}, [], [], {}, set()
    for n, e in enumerate(inventory):
        try:
            stored = (feeds_dir / e["path"]).read_bytes()
            if hashlib.sha256(stored).hexdigest() != e["stored_sha256"]:
                raise ProvenanceError("stored bytes do not match their receipt")
            body = gzip.decompress(stored) if e["gz"] else stored
            if hashlib.sha256(body).hexdigest() != e["decoded_sha256"]:
                raise ProvenanceError("decoded bytes do not match their receipt")
        except Exception as err:  # noqa: BLE001 - a broken provenance anchor stops the run, never a <=1% omission
            raise fatal(e["path"], err, n) from err
        pk = e["pk"]
        if pk not in eligible:
            ineligible.add(pk)
            continue
        if not e["available"]:
            meta, reasons = None, [e["availability_problem"]]
        else:
            try:
                meta = M.extract(json.loads(body))
                reasons = V.verify_game(meta, pk=pk, season=e["season"], parquet=counts.get(pk),
                                        parquet_date=dates.get(pk))
            except Exception as err:  # noqa: BLE001 - a malformed eligible feed is quarantined, never dropped (r1 F9)
                meta, reasons = None, [f"metadata error: {type(err).__name__}: {err}"[:300]]
        results[pk] = reasons
        provenance[str(pk)] = {"feed_timestamp": meta.feed_timestamp if meta else None,
                               "decoded_sha256": e["decoded_sha256"], "retrieved_utc": e["retrieved_utc"],
                               "attempt_id": e["attempt_id"], "url": e["url"], "certified": not reasons}
        if not reasons:
            rows += T.starter_counts(meta)
            for st in B.starts(meta):
                starts.append({**st, "box_batters_faced": meta.box_batters_faced[st["side"]],
                               "source_decoded_sha256": e["decoded_sha256"], "retrieved_utc": e["retrieved_utc"],
                               "attempt_id": e["attempt_id"]})

    # 6. census
    census = V.census(results, eligible=eligible, feeds_without_parquet=ineligible)
    census_sha = _write_json(run_dir / "census.json",
                             {**census, "quarantined": {str(k): v for k, v in census["quarantined"].items()}})
    log(f"census: {census['certified']}/{census['eligible']} certified, {len(census['quarantined'])} quarantined")
    if census["stop"]:
        _stop(run_dir, "STOPPED_quarantine.json", {"rate": census["rate"], "quarantined": len(census["quarantined"])})
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
