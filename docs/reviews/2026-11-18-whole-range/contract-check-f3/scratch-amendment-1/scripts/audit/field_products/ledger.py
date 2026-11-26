"""The accepted W1.1 season ledger, read only through its acceptance receipt (code review r1 F6).

The published build is ``data/validation/season_2026_ledger/<ACCEPTED_BUILD>/`` (docs/audit/2026-09-28-season-ledger.md
§1): code ``df3347e…``, run ``20260929T022806Z-compile2-dd430abe``, rules fingerprint ``5e9d74f2…`` (X-19). Its
``ACCEPTED.json`` is written by the twin-build accept step (plan 2026-09-28-season-ledger-phase1, Task 13) as
{run, accepted_at_utc, files (the six build outputs), compared_with, rules_fingerprint}; it carries no file hashes.

Before outcomes (``verify_receipt``): the directory is the published build, its listing is exactly the six outputs
plus the receipt, the receipt parses and names that run, those six files and the predeclared fingerprint, the build
manifest names the same code and fingerprint, and both read tables have exactly the compiler's schemas. After parsing
(``load_bound``): the parsed rows reproduce the build's own published marginal counts (row kinds; selection commit and
entry statuses; contest-slot match labels). Any failure refuses the run.

Owner disposition for code review r2 R2-4 (the declaration alternative): these are metadata/schema/count checks, not
an acceptance-byte identity check. The receipt carries no output hashes, no independently retained accepted-output
hash or reproduction is compared, and hashing the current files would only record run provenance; so every report
carries ``acceptance_byte_identity_established = False`` and ``ACCEPTANCE_BYTE_DECLARATION`` (the X-25 wording)."""
from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.audit.season_ledger.compile import CONTEST_SCHEMA, LEDGER_SCHEMA

EXPECTED_RULES_FINGERPRINT = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"
ACCEPTED_CODE_SHA = "df3347eb525570aecb328a306fb4d23f8c4ce82d"
ACCEPTED_RUN = "20260929T022806Z-compile2-dd430abe"
ACCEPTED_BUILD = f"{ACCEPTED_CODE_SHA}-{ACCEPTED_RUN}"
LEDGER_FILE = "season_2026_ledger.parquet"
CONTEST_FILE = "season_2026_ledger_contest_slots.parquet"
BUILD_FILE = "season_2026_ledger_build.json"
RECEIPT_FILE = "ACCEPTED.json"
BUILD_FILES = tuple(sorted((LEDGER_FILE, CONTEST_FILE, BUILD_FILE, "season_2026_ledger_occurrences.parquet",
                            "season_2026_ledger_reconciliation.parquet", "season_2026_ledger_summary.md")))
READ_FILES = (RECEIPT_FILE, BUILD_FILE, LEDGER_FILE, CONTEST_FILE)
ACCEPTANCE_BYTE_DECLARATION = (
    "The production denominator is read from frozen current files in the named W1.1 accepted-build directory. "
    "The reader validates acceptance metadata, published build identity, compiler schemas and selected marginal "
    "counts. It does not independently verify equality of outcome bytes to those accepted earlier. Production rates "
    "and comparisons are therefore conditional on those current files remaining unchanged from the accepted build; "
    "a current input hash is run provenance, not proof of earlier acceptance-byte identity.")


def _refuse(problems: list[str]) -> None:
    if problems:
        raise SystemExit("W1.1 acceptance receipt refused: " + "; ".join(problems))


def verify_receipt(ledger_dir: Path, listing: list[str], read) -> dict:
    """Pre-outcome checks on the frozen bytes (``read``) and the frozen directory listing."""
    d = Path(ledger_dir)
    problems = []
    if d.name != ACCEPTED_BUILD:
        problems.append(f"directory {d.name!r} is not the published build {ACCEPTED_BUILD!r}")
    if sorted(listing) != sorted(BUILD_FILES + (RECEIPT_FILE,)):
        problems.append(f"listing {sorted(listing)} is not the six build outputs plus {RECEIPT_FILE}")
    try:
        receipt = json.loads(read(d / RECEIPT_FILE))
        if not isinstance(receipt, dict):
            raise ValueError("not an object")
    except (FileNotFoundError, ValueError) as exc:
        _refuse(problems + [f"{RECEIPT_FILE} is not a parseable receipt ({exc})"])
    if receipt.get("rules_fingerprint") != EXPECTED_RULES_FINGERPRINT:
        problems.append(f"rules_fingerprint {receipt.get('rules_fingerprint')!r} is not the predeclared one")
    if receipt.get("run") != ACCEPTED_RUN:
        problems.append(f"run {receipt.get('run')!r} is not {ACCEPTED_RUN!r}")
    if sorted(receipt.get("files") or []) != list(BUILD_FILES):
        problems.append(f"files {receipt.get('files')} are not the six build outputs")
    try:
        build = json.loads(read(d / BUILD_FILE))
    except (FileNotFoundError, ValueError) as exc:
        _refuse(problems + [f"{BUILD_FILE} unreadable ({exc})"])
    if build.get("code_sha") != ACCEPTED_CODE_SHA:
        problems.append(f"build code_sha {build.get('code_sha')!r} is not {ACCEPTED_CODE_SHA!r}")
    if build.get("rules_fingerprint") != EXPECTED_RULES_FINGERPRINT:
        problems.append("build rules_fingerprint is not the predeclared one")
    for name, schema in ((LEDGER_FILE, LEDGER_SCHEMA), (CONTEST_FILE, CONTEST_SCHEMA)):
        try:
            got = pq.read_schema(io.BytesIO(read(d / name))).remove_metadata()
        except Exception as exc:  # noqa: BLE001 — any unreadable table is a refusal
            problems.append(f"{name} unreadable ({type(exc).__name__})")
            continue
        if not got.equals(schema):
            problems.append(f"{name} schema differs from the compiler's")
    _refuse(problems)
    return {"receipt": receipt, "build_identity": {k: build.get(k) for k in ("builder_version", "code_sha",
                                                                              "rules_fingerprint")},
            "acceptance_byte_identity_established": False,
            "acceptance_byte_identity_declaration": ACCEPTANCE_BYTE_DECLARATION, "build": build}


def _counts(values) -> dict:
    return dict(sorted(Counter("None" if v is None or v != v else str(v) for v in values).items()))


def load_bound(ledger_dir: Path, read, info: dict):
    """Parse the frozen ledger and contest-slot bytes and check them against the build's published marginal counts
    (a consistency check, not acceptance-byte identity)."""
    d = Path(ledger_dir)
    ledger = pq.read_table(io.BytesIO(read(d / LEDGER_FILE))).to_pandas(integer_object_nulls=True)
    contest = pq.read_table(io.BytesIO(read(d / CONTEST_FILE))).to_pandas(integer_object_nulls=True)
    sels = ledger[ledger["row_kind"] == "selection"]
    got = {"row_kinds": _counts(ledger["row_kind"]), "commit_status": _counts(sels["commit_status"]),
           "entry_status": _counts(sels["entry_status"]), "match": _counts(contest["match"])}
    build = info["build"]
    _refuse([f"{k}: parsed {v} != build {build.get(k)}" for k, v in got.items() if build.get(k) != v])
    return ledger, contest, {**{k: v for k, v in info.items() if k != "build"}, "marginal_counts_checked": got}
