# Season 2026 Ledger — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 canonical ledger of the 2026 season's production decisions: lossless source observations, qualified contest evidence matched to specific games, explicit row kinds, and an honest reconciliation, all compiled from a sealed evidence bundle.

**Architecture:** `acquire` runs on the box. It copies every input from the frozen W0.7 snapshot into a sealed bundle with a sha256 manifest. Its only network access is fetching MLB schedules. `compile` verifies the bundle and runs offline in stages:
1. Per-source parsers.
2. Contest slot history and game matching.
3. Day and row rules.
4. Outcomes.
5. Accounting and reconciliation.
6. Deterministic parquet and markdown outputs.

Everything lives under `scripts/audit/season_ledger/`, and production code is untouched.

**Tech Stack:** Python 3.12, stdlib (`json`, `gzip`, `hashlib`, `re`, `urllib`, `zoneinfo`), `pyarrow` 23 (already a dependency), `pytest`. `bts.daily_decision.ACCEPTED_SCHEMAS` and `decision_objective` are reused read-only.

**Spec:** `docs/superpowers/specs/2026-09-28-season-ledger-design.md` (v4, approved on 2026-09-28 by Codex design r4, Claude and Eric).

## Global Constraints
- **Scope:** Phase 1 covers the production stream only (spec §2). It excludes shadow v1/v2, the skip-policy shadow, D8, cached-feed grading, MLB's current record, recipe epochs and slate binding.
- **No production changes.** No change to production code or production files. New code lives only under `scripts/audit/season_ledger/`, `scripts/audit/build_season_ledger.py` and `tests/scripts/season_ledger/`.
- **Network:** `compile` performs no network access. The only network call in `acquire` is `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=<D>&gameType=R&hydrate=team`, once per season date, paced at 0.5 s.
- **Authoritative outcome:** the only authoritative BTS outcome is a qualified contest slot grade whose match is `evidenced` or `inferred` (spec §6–§7).
- **Never inferred** (spec §5–§6, §9): entry absence, history `complete`, saver availability, and eligibility from an outcome. Phase 1 therefore emits:
  - `entry_status ∈ {confirmed, unknown}`
  - `history_status ∈ {known_incomplete, unknown}`
  - `saver_available_before = null`
  - `game_eligibility = unknown` unless there is evidence from before lock (Interpretation I6)
- **Recipe rules are hypotheses** (spec §8). **Task 9's rule table is fixed before any real count is computed; it must not be edited after the first real run.** `test_candidate_rules_are_the_predeclared_lists` pins it.
- **P-01 is untouched:** the completed P-01 read and `scripts/audit/build_slot_dataset.py` are not modified.
- **Tests use synthetic files only.** No test reads the snapshot, the box or the network.
- **Commands:** every `uv` command uses `UV_CACHE_DIR=/tmp/uv-cache`, and pytest runs with `TZ=America/New_York`.
- **Output locations:** outputs go to `data/validation/` (gitignored). The evidence bundle goes to `data/hetzner_results/season_2026_ledger_evidence/v1/`, inside the `archive` restic set.
- **Git:** work on `main` (audit scripts; `main` does not deploy), with one commit per task. Commit messages end with the session's attribution lines.
- **Box:**
  - Nothing is deployed.
  - The code reaches the box as a `git archive` of a named commit under `/tmp/ledger_code`.
  - No service restarts.
  - Long jobs run as `systemd-run --user` units.

## Verified source facts (structure-only probe of `final-20260928/`, 2026-09-28; key names, types and counts, no values)
- **`data/picks/` files by name pattern:**
  - Production files: 156 `DATE.json` (primary `game_pk`/`game_time` null only on 3/29–3/30), 92 `DATE/decision.json`, 167 `DATE/scheduler_state.json`, 27 `DATE/deferred_fallback_<YYYYmmddTHHMMSS-0400>.json`, 124 `lineup_evolution_DATE.jsonl`.
  - Shadow and slate files: 118 `DATE.shadow.json` (has `pick`), 16 `DATE.policy_shadow.json` (no `pick`), 28 `backup_shadow_DATE/DATE.shadow.json`, 105 `slates/DATE.json` (no `pick`).
  - Archives and repairs: 1 `archive/DATE.json.postponed` (has `pick`; no writer in the code, so it was a manual move); `archive_actual_streak_repair_<stamp>[_missed_N]/` holds 4 `DATE.json` + 4 `DATE.json.before` + `streak.before.json`; `archive_replay_restore_<stamp>_post_contest_state_deploy/{README.txt, streak.before.json}`.
  - State and markers: 14 AppleDouble `._*` files, `streak.json`, `.nrestarts_checkpoint`.
  - `account_state/`: `contest_ledger.jsonl` (411 lines), `saver_transitions.jsonl`, `contest_streak.json`, 3 `contest_streak.manual.json.*` variants, `saver_state.json`.
- **Time formats:**
  - `run_time`: `+00:00`, with and without microseconds.
  - `delivered_at`, `pick_locked_at`, `deferred_at`: ET offset with microseconds.
  - `finalized_at`, `recorded_at`: `Z` with microseconds.
  - `game_time`: `Z` without microseconds.
- **Contest ledger:**
  - Every line has `recorded_at`, `active_streak`, `best_streak`, `source_date` and `predictions`. Each line carries the account's full season of rounds so far (70 rounds on the first line, rising to 136 distinct roundIds 827–995).
  - Round keys: `roundId`, `result`, `streak`, `streakIncrease`, `roundPredictions`. Slot keys: `number`, `unitId`, `playerId`, `result`, `hits`, `atBats`.
  - 5 rounds and 18 slots with null `result` all sit in the latest round of their line.
  - 3 slots have null `playerId` (lines 89, 329 and 353).
- **BTS static captures** (`data/leaderboard/static_snapshots/<feed>/<YYYYmmddTHHMMSSZ>.json[.gz]`, plus a `.last_sha256` marker per feed):
  - rounds: 173 files, 1 MB. `{"rounds": [{id, date, status, contestId}]}`, 188 rounds 823→1010, one per date 3/25→9/28, no id gaps.
  - units: 2,335 files, 270 MB. `{"units": [{id, feedId, roundId, status, startDateTime, lockDateTime, awaySquadId, homeSquadId, lineups, …}]}`. The status vocabulary is `scheduled`/`playing`/`complete`/`postponed`, and the last capture is an empty list.
  - players: 3,773 files, 1.5 GB. `{"players": [{id, feedId, squadId, name, …}]}` (2,930 players).
  - The 9/27 grab's `raw/static/` holds `001_rounds`, `002_players` (143 KB), `003_units` (empty) and `004_squads` `.json.gz`.
- **Snapshot root:** holds `cron.log` and `journal_bts-scheduler_retained.txt`. Snapshot pick files keep their live mtimes.
- **MLB schedule:** the endpoint with `hydrate=team` gives `teams.{away,home}.team.abbreviation`, identical to the live feed's `gameData.teams.*.abbreviation`, which is the source of `Pick.team`. It lists `Cancelled`/`Postponed` games (e.g. 9/27 BAL@NYY `C Cancelled`).

## Interpretations pinned by this plan (each is a reading of the spec; Codex reviews them)
- **I1 Qualification:**
  - Identity fields (`roundId`, `unitId`, `playerId`) must be non-null integers.
  - `recorded_at` must parse with an offset.
  - Round and slot `result` keys must be present and may be null; a null means the round is in progress and reads `matched_ungraded`.
  - A line with any slot lacking identity is quarantined whole (spec: line-level qualification). This affects the 3 playerless lines, whose rounds are repeated in neighbouring lines.
- **I2 `streak_before`** is the reported `streak` of the nearest earlier round present in the same qualified line. That is the account's previous entered round, since skip days have no round and leave the streak unchanged; production's `contest_ledger.parse_latest_ledger` uses the same rule. It is null when no earlier round is present.
- **I3 `conflicted`:** commit evidence exists for a different selection in the same date and slot. Commit evidence means a decision naming the selection, or a confirmed pick-file delivery. `committed_pick_written` names no selection, so it counts only for the surviving pick file when no decision exists.
- **I4 Delivery status:**
  - `private_locked` → `delivery_confirmed = false` (an explicit non-delivery).
  - `locked_unconfirmed` → null.
  - A pick-side delivery signal alongside either sets `delivery_evidence_conflict = true`.
- **I5 History:** `known_incomplete` when any observation for the date names a (slot, batter, game) outside the day's canonical selections. Observations are lineup-evolution entries, any archived or repaired version, and the surviving pick file itself. So a discarded double-down preview marks the day's single.
- **I6 Eligibility:**
  - `postponed_evidenced` comes only from a BTS units capture showing status `postponed` whose file stamp (content-deduped, so the stamp is when that content first appeared) is before `locked_at`, and only for an `evidenced` match.
  - `refused_evidenced` comes from a `refused_delivery` archive naming the selection.
  - The MLB schedule is fetched after the season, so it never evidences eligibility.
- **I7 Evidenced game, no same-game local selection:** a contest slot whose game is evidenced but has no local selection for that game becomes `contest_only` with reason `unit_capture_other_game`. If the batter has a local selection that day on another game, that selection reads `match_ambiguous`.
- **I8 One selection, two contest slots:** when two contest slot identities would link to one selection (an entry changed within a round), both are demoted to `ambiguous` (`multiple_contest_slots_for_selection`) and no grade transfers.
- **I9 Recipe labels:**
  - `hypothesis`: some pre-declared rule reproduces both published totals.
  - `partial`: a rule reproduces only one total.
  - `unrecoverable`: no rule reproduces either.
  - `exact` needs independent per-record evidence, which Phase 1 lacks.
  - Per-record evidence interval: the source mtime (suggestive only) against the recipe's ET date.
- **I10 Players lookup:** built from the 9/27 grab plus the first and last static players capture (the 1.5 GB history is not bundled). An unmapped player reads `player_unknown`, and the memo reports how many.
- **I11 `bts_outcome_status = graded`** for any non-null contest slot result. Unknown labels normalize to `UNKNOWN` and are never compared.
- **I12 Outputs:**
  - `…_reconciliation.parquet` is the recipe-membership table (spec §8 table 3).
  - `…_occurrences.parquet` gives every emitted occurrence a `disposition`. This is the anti-join: a production pick-file slot is `canonical_selection`, `unresolved_pick_file_view`, `not_selected` or `outside_season_window`.

## Review Focus
1. AppleDouble `._*` files, runtime markers and `.before`/`.postponed` versions in the picks tree must be routed or excluded with a reason. They are never parsed as production picks, and no stray file crashes the build. *(Task 9 routing tests)*
2. The 3/29–3/30 pick files have null `game_pk`/`game_time`. Their `selection_id` carries `None`, `game_time` is null, and matching reads ambiguous `selection_game_pk_unrecorded`; nothing crashes. *(Task 2, Task 6 tests)*
3. Contest lines recorded mid-round, with null round `result`/`streak` or a null slot `result`, qualify and read `matched_ungraded`. A playerless slot quarantines its line. *(Task 4 tests)*
4. The end-of-season units capture holds an empty `units` list, and plain `.json` sits beside `.json.gz`. Both parse without rows or false conflicts, and a corrupt gzip is quarantined. *(Task 5 tests)*
5. Mixed time formats (`Z`, `+00:00`, ET `-04:00`, with and without microseconds) must end in one fixed-precision UTC form so that string order is time order. A naive time becomes null and is never read in the host zone. *(Task 1 test)*

---

## File Structure
| File | Responsibility |
|---|---|
| `scripts/audit/season_ledger/__init__.py` | package marker, `BUILDER_VERSION` |
| `scripts/audit/season_ledger/ids.py` | sha256, occurrence ids, `Parsed`, JSON/gzip loading, UTC normalization |
| `scripts/audit/season_ledger/bundle.py` | manifest writing (acquire) and verification (compile) |
| `scripts/audit/season_ledger/io.py` | deterministic parquet writer |
| `scripts/audit/season_ledger/sources/__init__.py` | subpackage marker |
| `scripts/audit/season_ledger/sources/pick_files.py` | O1 pick files, O2 scheduler archives, manual and repair versions |
| `scripts/audit/season_ledger/sources/day_records.py` | O3 decisions, O4 scheduler state, O5 lineup evolution |
| `scripts/audit/season_ledger/sources/contest_ledger.py` | O6 contest ledger (lossless), saver-transition attempts |
| `scripts/audit/season_ledger/sources/static.py` | O7 rounds / players / units captures, MLB schedules |
| `scripts/audit/season_ledger/contest.py` | slot history, streak chain, lookups, game matching |
| `scripts/audit/season_ledger/outcomes.py` | outcome normalization, slot comparison, single-pick derivation |
| `scripts/audit/season_ledger/rows.py` | day status, row kinds, commit / history status, delivery predicate |
| `scripts/audit/season_ledger/reconcile.py` | routing, occurrence accounting, invariants, recipe rules |
| `scripts/audit/season_ledger/compile.py` | offline pipeline and output schemas |
| `scripts/audit/season_ledger/acquire.py` | acquisition into a sealed bundle |
| `scripts/audit/build_season_ledger.py` | CLI (`acquire`, `compile`) |
| `tests/scripts/season_ledger/__init__.py`, `builders.py` | synthetic source builders (production file shapes) |
| `tests/scripts/season_ledger/test_*.py` | one test file per module |

---

### Task 1: Package, ids, bundle, deterministic writer

**Files:**
- Create: `scripts/audit/season_ledger/__init__.py`, `ids.py`, `bundle.py`, `io.py`, `sources/__init__.py`
- Create: `tests/scripts/season_ledger/__init__.py` (empty), `tests/scripts/season_ledger/builders.py`, `tests/scripts/season_ledger/test_bundle_io.py`

**Interfaces:**
- **Produces (`ids.py`):**
  - `ids.UTC_FORMAT`
  - `ids.sha256_hex(data: bytes) -> str`
  - `ids.obs_id(rel_path: str, locator: str, content_sha256: str) -> str`
  - `ids.Parsed(rows: list[dict], quarantined: list[dict])`
  - `ids.quarantine(rel_path, locator, reason) -> dict`
  - `ids.load_json_bytes(data: bytes) -> object`, which raises `ValueError("bad_gzip:…" | "invalid_json:…")`
  - `ids.is_int(value) -> bool`
  - `ids.utc_iso(raw) -> str | None`
  - `ids.stamp_to_utc(name: str) -> str | None`
- **Produces (`bundle.py`):**
  - `bundle.BundleEntry`
  - `bundle.write_manifest(root, entries, *, acquired_at_utc, builder_version, source_root=None) -> Path`
  - `bundle.open_bundle(root) -> tuple[dict, dict[str, bytes | None]]`, with keys sorted by path
  - `bundle.BundleError`
- **Produces (`io.py`):** `io.write_table(rows, schema, path, *, sort_keys) -> Path`
- **Produces (`builders.py`):** `builders.dumps(obj) -> bytes` and `builders.seal_bundle(root, files: dict[str, bytes], missing=(), mtimes=None) -> None`

- [ ] **Step 1: Write the builders module and the failing tests**

`tests/scripts/season_ledger/builders.py`:
```python
"""Synthetic source builders for the season-ledger tests. They write the shapes production writes
(bts.picks.save_pick, bts.daily_decision.write_decision, scheduler save_state, the CLI contest-ledger
append); no test reads real data."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.season_ledger.bundle import BundleEntry, write_manifest
from scripts.audit.season_ledger.ids import sha256_hex


def dumps(obj) -> bytes:
    return json.dumps(obj).encode()


def seal_bundle(root, files: dict[str, bytes], missing=(), mtimes: dict[str, str] | None = None) -> None:
    root = Path(root)
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_mtime_utc=(mtimes or {}).get(rel)))
    entries += [BundleEntry(rel_path=rel, status="missing", note="not found at acquisition") for rel in missing]
    write_manifest(root, entries, acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="test")
```

`tests/scripts/season_ledger/test_bundle_io.py`:
```python
import gzip
import json

import pyarrow as pa
import pytest

from scripts.audit.season_ledger.bundle import BundleEntry, BundleError, open_bundle, write_manifest
from scripts.audit.season_ledger.ids import load_json_bytes, obs_id, sha256_hex, stamp_to_utc, utc_iso
from scripts.audit.season_ledger.io import write_table
from tests.scripts.season_ledger.builders import seal_bundle


def test_declared_missing_input_is_accepted(tmp_path):
    seal_bundle(tmp_path, {"picks/2026-05-01.json": b"{}"}, missing=["schedules/2026-05-01.json"])
    _, files = open_bundle(tmp_path)
    assert files == {"picks/2026-05-01.json": b"{}", "schedules/2026-05-01.json": None}


def test_declared_present_but_absent_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").unlink()
    with pytest.raises(BundleError, match="declared present but absent"):
        open_bundle(tmp_path)


def test_changed_file_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").write_bytes(b'{"x": 1}')
    with pytest.raises(BundleError, match="hash mismatch"):
        open_bundle(tmp_path)


def test_unsafe_path_is_refused(tmp_path):
    with pytest.raises(BundleError, match="unsafe"):
        write_manifest(tmp_path, [BundleEntry(rel_path="../x.json", status="missing")],
                       acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="t")


def test_manifest_order_does_not_change_what_compile_reads(tmp_path):
    seal_bundle(tmp_path, {"b.json": b"2", "a.json": b"1"})
    manifest = tmp_path / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()
    manifest.write_text(json.dumps(doc))
    _, files = open_bundle(tmp_path)
    assert list(files) == ["a.json", "b.json"]


def test_duplicate_bytes_at_two_paths_are_two_occurrences():
    digest = sha256_hex(b"same bytes")
    assert obs_id("picks/a.json", "file", digest) != obs_id("picks/archive/a.json", "file", digest)


def test_gzip_and_plain_json_both_load():
    assert load_json_bytes(b'{"a": 1}') == {"a": 1}
    assert load_json_bytes(gzip.compress(b'{"a": 1}')) == {"a": 1}
    with pytest.raises(ValueError, match="bad_gzip"):
        load_json_bytes(b"\x1f\x8b" + b"not really gzip")
    with pytest.raises(ValueError, match="invalid_json"):
        load_json_bytes(b'{"a": ')


def test_times_normalize_to_fixed_precision_utc():
    # Review Focus 5: every source format in the snapshot, plus a naive value.
    assert utc_iso("2026-08-20T17:00:00.123456-04:00") == "2026-08-20T21:00:00.123456Z"
    assert utc_iso("2026-05-01T23:05:00Z") == "2026-05-01T23:05:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00+00:00") == "2026-05-01T15:00:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00") is None
    assert utc_iso(None) is None and utc_iso("garbage") is None
    assert utc_iso("2026-05-01T15:00:00.5Z") < utc_iso("2026-05-01T15:00:01Z")   # string order == time order


def test_capture_stamp_to_utc():
    assert stamp_to_utc("20260704T030011Z.json.gz") == "2026-07-04T03:00:11.000000Z"
    assert stamp_to_utc("001_rounds.json.gz") is None


SCHEMA = pa.schema([("obs_id", pa.string()), ("n", pa.int64()), ("flag", pa.bool_())])


def test_write_table_is_byte_identical_for_reordered_input(tmp_path):
    rows = [{"obs_id": "b", "n": 2, "flag": None}, {"obs_id": "a", "n": None, "flag": True}]
    write_table(rows, SCHEMA, tmp_path / "x.parquet", sort_keys=["obs_id"])
    write_table(list(reversed(rows)), SCHEMA, tmp_path / "y.parquet", sort_keys=["obs_id"])
    assert (tmp_path / "x.parquet").read_bytes() == (tmp_path / "y.parquet").read_bytes()


def test_write_table_refuses_duplicate_sort_keys_and_unknown_columns(tmp_path):
    with pytest.raises(ValueError, match="duplicate sort keys"):
        write_table([{"obs_id": "a"}, {"obs_id": "a"}], SCHEMA, tmp_path / "d.parquet", sort_keys=["obs_id"])
    with pytest.raises(ValueError, match="not in schema"):
        write_table([{"obs_id": "a", "typo": 1}], SCHEMA, tmp_path / "u.parquet", sort_keys=["obs_id"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_bundle_io.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'scripts.audit.season_ledger'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/__init__.py`:
```python
"""Season 2026 canonical ledger, Phase 1 (spec docs/superpowers/specs/2026-09-28-season-ledger-design.md)."""
BUILDER_VERSION = "season-ledger-phase1/1"
```

`scripts/audit/season_ledger/sources/__init__.py`:
```python
"""Per-source parsers: bytes in, lossless rows out."""
```

`scripts/audit/season_ledger/ids.py`:
```python
"""Occurrence identity and small shared helpers."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_STAMP = re.compile(r"(\d{8}T\d{6})Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def obs_id(rel_path: str, locator: str, content_sha256: str) -> str:
    """Identity of one source occurrence: path + locator + content hash, so byte-identical
    files at two paths remain two occurrences (spec §4)."""
    return hashlib.sha256(f"{rel_path}\x00{locator}\x00{content_sha256}".encode()).hexdigest()[:24]


@dataclass
class Parsed:
    rows: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)


def quarantine(rel_path: str, locator: str, reason: str) -> dict:
    return {"source_path": rel_path, "locator": locator, "reason": reason}


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json_bytes(data: bytes):
    """Parse JSON from plain or gzip bytes; raise ValueError('bad_gzip:…' / 'invalid_json:…')."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"bad_gzip:{exc.__class__.__name__}") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_json:{exc.__class__.__name__}") from exc


def utc_iso(raw) -> str | None:
    """ISO-8601 with an offset → fixed-precision UTC ('YYYY-MM-DDTHH:MM:SS.ffffffZ'), so string
    order is time order. None, non-strings, unparseable and naive values → None: a naive time is
    never read in the host's zone."""
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if t.tzinfo is None or t.utcoffset() is None:
        return None
    return t.astimezone(timezone.utc).strftime(UTC_FORMAT)


def stamp_to_utc(name: str) -> str | None:
    """Static-capture file names carry a UTC stamp 'YYYYmmddTHHMMSSZ'."""
    m = _STAMP.search(name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).strftime(UTC_FORMAT)
```

`scripts/audit/season_ledger/bundle.py`:
```python
"""Sealed evidence bundle: manifest writing (acquire) and verification (compile). Spec §3."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .ids import sha256_hex

MANIFEST_NAME = "manifest.json"
BUNDLE_SCHEMA = "bts_season_ledger_bundle_v1"


class BundleError(Exception):
    pass


@dataclass(frozen=True)
class BundleEntry:
    rel_path: str
    status: str                       # "present" | "missing"
    sha256: str | None = None
    size: int | None = None
    source_path: str | None = None    # relative to the acquisition source root, or the fetch URL
    source_mtime_utc: str | None = None
    note: str | None = None


def check_rel_path(rel: str) -> str:
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or p.as_posix() != rel:
        raise BundleError(f"unsafe bundle path: {rel!r}")
    return rel


def write_manifest(root: Path, entries: list[BundleEntry], *, acquired_at_utc: str, builder_version: str,
                   source_root: str | None = None) -> Path:
    for e in entries:
        check_rel_path(e.rel_path)
        if e.status not in ("present", "missing"):
            raise BundleError(f"bad status {e.status!r} for {e.rel_path}")
    body = {"schema": BUNDLE_SCHEMA, "acquired_at_utc": acquired_at_utc, "builder_version": builder_version,
            "source_root": source_root,
            "entries": sorted((asdict(e) for e in entries), key=lambda d: d["rel_path"])}
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return path


def open_bundle(root: Path) -> tuple[dict, dict[str, bytes | None]]:
    """Verify every declared entry; return (manifest, {rel_path: bytes | None}) sorted by path.
    A declared `missing` entry is valid evidence of absence; a declared-present file that is absent
    or changed is refused. Files the manifest does not declare are ignored."""
    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"unreadable manifest: {exc}") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unexpected bundle schema {manifest.get('schema')!r}")
    files: dict[str, bytes | None] = {}
    for e in sorted(manifest["entries"], key=lambda e: e["rel_path"]):
        rel = check_rel_path(e["rel_path"])
        if rel in files:
            raise BundleError(f"duplicate manifest entry {rel}")
        if e["status"] == "missing":
            files[rel] = None
            continue
        path = root / rel
        if not path.is_file():
            raise BundleError(f"declared present but absent: {rel}")
        data = path.read_bytes()
        if sha256_hex(data) != e["sha256"]:
            raise BundleError(f"hash mismatch: {rel}")
        files[rel] = data
    return manifest, files
```

`scripts/audit/season_ledger/io.py`:
```python
"""Deterministic parquet writing (spec §3: identical inputs → identical bytes)."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _sort_key(row: dict, keys: list[str]) -> tuple:
    return tuple("" if row.get(k) is None else str(row.get(k)) for k in keys)


def write_table(rows: list[dict], schema: pa.Schema, path: Path, *, sort_keys: list[str]) -> Path:
    names = set(schema.names)
    for row in rows:
        extra = set(row) - names
        if extra:
            raise ValueError(f"columns not in schema for {Path(path).name}: {sorted(extra)}")
    ordered = sorted(rows, key=lambda r: _sort_key(r, sort_keys))
    keys = [_sort_key(r, sort_keys) for r in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate sort keys in {Path(path).name}; sort on a unique column")
    table = pa.Table.from_pydict({f.name: [r.get(f.name) for r in ordered] for f in schema}, schema=schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=False, write_statistics=False)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_bundle_io.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger tests/scripts/season_ledger
git commit -m "feat(ledger): sealed bundle, occurrence ids, UTC normalization, deterministic parquet (W1.1 task 1)"
```

---

### Task 2: O1 pick files, O2 archives, manual and repair versions

**Files:**
- Create: `scripts/audit/season_ledger/sources/pick_files.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `pick_json`)
- Test: `tests/scripts/season_ledger/test_pick_files.py`

**Interfaces:**
- **Consumes:** Task 1 `ids.*`.
- **Produces:**
  - `parse_pick_file(rel_path, data, *, kind="pick_file") -> Parsed`, where `kind ∈ {pick_file, manual_archive, repair_archive}`
  - `parse_archive(rel_path, data) -> Parsed`
- **Every row has:**
  - Identity: `obs_id, source_kind, source_path, content_sha256, slot ("primary"|"double_down"), date, run_time, day_result_raw, slot_result_raw, has_double_down, absent_fields`.
  - Pick fields: `batter_id, batter_name, team, game_pk, game_time, lineup_position, projected_lineup, pitcher_id, p_game_hit`.
  - Delivery fields: `bluesky_posted, bluesky_uri, notification_sent, notification_id, notification_channel, delivery_attempted, delivered_at`.
  - Provenance fields: `model_git_sha, model_pickle_sha256, policy_npz_sha256, feature_env_hash, tail_policy_sha256`.
  - Archive metadata: `archive_prefix, archive_reason, archived_at`, which are None except on scheduler archives.
- **Absent keys are None** (never defaulted) and are listed in `absent_fields`.

- [ ] **Step 1: Append the builder** to `tests/scripts/season_ledger/builders.py`:

```python
_PICK = {"batter_name": "Ada Batter", "batter_id": 101, "team": "TB", "lineup_position": 1,
         "pitcher_name": "P", "pitcher_id": 900, "p_game_hit": 0.78, "flags": [],
         "projected_lineup": False, "game_pk": 5001, "game_time": "2026-05-01T23:05:00Z", "pitcher_team": "BOS"}
_DD = {**_PICK, "batter_name": "Dee Leg", "batter_id": 202, "team": "NYY", "game_pk": 5002, "lineup_position": 2}


def pick_json(date: str, *, primary: dict | None = None, dd: dict | None = None, **file_fields) -> bytes:
    """A pick file as save_pick writes it (asdict(DailyPick)) with only the given file-level fields;
    pass dd={} for the default double-down."""
    doc = {"date": date, "run_time": f"{date}T15:00:00Z", "pick": {**_PICK, **(primary or {})},
           "double_down": None if dd is None else {**_DD, **dd}, "runner_up": None}
    doc.update(file_fields)
    return dumps(doc)
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_pick_files.py`:

```python
from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file
from tests.scripts.season_ledger.builders import dumps, pick_json


def test_double_down_file_yields_two_slot_rows_with_raw_slot_results():
    data = pick_json("2026-08-20", dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                     notification_sent=True, notification_id="dm-1")
    parsed = parse_pick_file("picks/2026-08-20.json", data)
    assert parsed.quarantined == []
    rows = {r["slot"]: r for r in parsed.rows}
    assert set(rows) == {"primary", "double_down"}
    assert rows["primary"]["slot_result_raw"] == "miss" and rows["double_down"]["slot_result_raw"] == "hit"
    assert rows["primary"]["day_result_raw"] == "miss" and rows["primary"]["has_double_down"] is True
    assert rows["double_down"]["batter_id"] == 202 and rows["double_down"]["game_pk"] == 5002
    assert rows["primary"]["obs_id"] != rows["double_down"]["obs_id"]


def test_early_file_records_absent_fields_and_never_defaults_delivery():
    # Review Focus 2 (and the §4 losslessness rule): the 3/29–3/30 shape — null game, no later fields.
    data = dumps({"date": "2026-03-29", "run_time": "2026-03-29T15:00:00+00:00", "result": "hit",
                  "bluesky_posted": False, "bluesky_uri": None, "runner_up": None, "double_down": None,
                  "pick": {"batter_id": 101, "batter_name": "Ada Batter", "team": "TB", "game_pk": None,
                           "game_time": None, "lineup_position": 3}})
    (row,) = parse_pick_file("picks/2026-03-29.json", data).rows
    assert (row["game_pk"], row["game_time"], row["bluesky_posted"]) == (None, None, False)
    assert row["notification_sent"] is None and row["slot_result_raw"] is None
    absent = set(row["absent_fields"].split(","))
    assert {"notification_sent", "slot_results", "delivered_at", "model_git_sha"} <= absent
    assert "bluesky_posted" not in absent


def test_non_json_and_truncated_files_are_quarantined():
    for data in (b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X", b'{"date": "2026-05-01", "pick": {'):
        parsed = parse_pick_file("picks/2026-05-01.json", data)
        assert parsed.rows == [] and parsed.quarantined[0]["reason"].startswith("invalid_json")


def test_file_without_pick_object_is_quarantined():
    parsed = parse_pick_file("picks/2026-05-01.json", dumps({"date": "2026-05-01"}))
    assert parsed.quarantined == [{"source_path": "picks/2026-05-01.json", "locator": "file", "reason": "no_pick_object"}]


def test_archive_rows_carry_prefix_reason_and_time():
    data = pick_json("2026-08-30", deferred_fallback={"reason": "gap_blocked", "deferred_at": "2026-08-30T12:01:00-04:00"})
    (row,) = parse_archive("picks/2026-08-30/deferred_fallback_20260830T120100-0400.json", data).rows
    assert (row["source_kind"], row["archive_prefix"], row["archive_reason"], row["archived_at"]) == (
        "archive", "deferred_fallback", "gap_blocked", "2026-08-30T12:01:00-04:00")


def test_repair_and_manual_versions_parse_as_pick_versions():
    before = parse_pick_file("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json.before",
                             pick_json("2026-05-24", result="miss"), kind="repair_archive").rows[0]
    postponed = parse_pick_file("picks/archive/2026-04-11.json.postponed", pick_json("2026-04-11"),
                                kind="manual_archive").rows[0]
    assert (before["source_kind"], before["archive_prefix"], before["day_result_raw"]) == ("repair_archive", None, "miss")
    assert (postponed["source_kind"], postponed["batter_id"]) == ("manual_archive", 101)


def test_unrecognized_archive_name_is_quarantined():
    parsed = parse_archive("picks/2026-08-30/something_else.json", pick_json("2026-08-30"))
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "unrecognized_archive_name"
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_pick_files.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.sources.pick_files'`.

- [ ] **Step 4: Write the implementation** — `scripts/audit/season_ledger/sources/pick_files.py`:

```python
"""O1 pick files (`picks/<date>.json`), O2 scheduler archives (`picks/<date>/<prefix>_<stamp>.json`),
and the manual / streak-repair versions of pick files. Lossless: absent keys stay None and are
listed in `absent_fields` (spec §4)."""
from __future__ import annotations

import re

from ..ids import Parsed, load_json_bytes, obs_id, quarantine, sha256_hex

SLOTS = (("primary", "pick"), ("double_down", "double_down"))   # ledger slot, JSON key (= slot_results key)
PICK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "game_time", "lineup_position",
               "projected_lineup", "pitcher_id", "p_game_hit")
DELIVERY_FIELDS = ("bluesky_posted", "bluesky_uri", "notification_sent", "notification_id",
                   "notification_channel", "delivery_attempted", "delivered_at")
PROVENANCE_FIELDS = ("model_git_sha", "model_pickle_sha256", "policy_npz_sha256", "feature_env_hash",
                     "tail_policy_sha256")
TRACKED_FILE_FIELDS = ("result", "slot_results", *DELIVERY_FIELDS, *PROVENANCE_FIELDS)
ARCHIVE_TIME_KEYS = {"deferred_fallback": "deferred_at", "refused_delivery": "refused_at", "stale_pick": "staled_at"}
_ARCHIVE_NAME = re.compile(r"^(deferred_fallback|refused_delivery|stale_pick)_\d{8}T\d{6}[+-]\d{4}\.json$")


def parse_pick_file(rel_path: str, data: bytes, *, kind: str = "pick_file") -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        out.quarantined.append(quarantine(rel_path, "file", str(exc)))
        return out
    if not isinstance(doc, dict) or not isinstance(doc.get("pick"), dict):
        out.quarantined.append(quarantine(rel_path, "file", "no_pick_object"))
        return out
    base = {"source_kind": kind, "source_path": rel_path, "content_sha256": content,
            "date": doc.get("date"), "run_time": doc.get("run_time"), "day_result_raw": doc.get("result"),
            "has_double_down": doc.get("double_down") is not None,
            "absent_fields": ",".join(sorted(k for k in TRACKED_FILE_FIELDS if k not in doc)),
            "archive_prefix": None, "archive_reason": None, "archived_at": None}
    for f in (*DELIVERY_FIELDS, *PROVENANCE_FIELDS):
        base[f] = doc.get(f)
    slot_results = doc.get("slot_results")
    for slot, key in SLOTS:
        pick = doc.get(key)
        if pick is None:
            continue
        if not isinstance(pick, dict):
            out.quarantined.append(quarantine(rel_path, f"slot={slot}", "slot_not_object"))
            continue
        row = dict(base, slot=slot, obs_id=obs_id(rel_path, f"slot={slot}", content))
        for f in PICK_FIELDS:
            row[f] = pick.get(f)
        row["slot_result_raw"] = slot_results.get(key) if isinstance(slot_results, dict) else None
        out.rows.append(row)
    return out


def parse_archive(rel_path: str, data: bytes) -> Parsed:
    match = _ARCHIVE_NAME.match(rel_path.rsplit("/", 1)[-1])
    if not match:
        return Parsed([], [quarantine(rel_path, "file", "unrecognized_archive_name")])
    parsed = parse_pick_file(rel_path, data, kind="archive")
    prefix = match.group(1)
    if parsed.rows:
        doc = load_json_bytes(data)
        info = doc.get(prefix) if isinstance(doc.get(prefix), dict) else {}
        for row in parsed.rows:
            row.update(archive_prefix=prefix, archive_reason=info.get("reason"),
                       archived_at=info.get(ARCHIVE_TIME_KEYS[prefix]))
    return parsed
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_pick_files.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/pick_files.py tests/scripts/season_ledger
git commit -m "feat(ledger): lossless pick-file, archive and repair-version parsers (task 2)"
```

---

### Task 3: O3 decisions, O4 scheduler state, O5 lineup evolution

**Files:**
- Create: `scripts/audit/season_ledger/sources/day_records.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `cand`, `decision_json`, `state_json`, `evolution_jsonl`)
- Test: `tests/scripts/season_ledger/test_day_records.py`

**Interfaces:**
- **Consumes:** Task 1; `bts.daily_decision.ACCEPTED_SCHEMAS`, `bts.daily_decision.decision_objective`.
- **Produces:**
  - `parse_decision(rel_path, data) -> Parsed`. One row with:
    - `obs_id, source_path, content_sha256`
    - every `DECISION_FIELDS` key
    - `objective_raw`, `objective`, `action_source_raw`, `action_source ∈ {mdp, heuristic, unknown}`
    - `{primary,double_down,second_candidate}_{batter_id,batter_name,team,game_pk,p_game_hit}`
  - `parse_scheduler_state(rel_path, data) -> Parsed`. One row with:
    - `obs_id, source_path, content_sha256, date, schedule_fetched_at, pick_locked, pick_locked_at, committed_pick_written, result_status, skip_notified_at`
    - `final_skip_candidate_present, skip_candidate_batter_id, skip_candidate_game_pk, delivery_refusal_archives, fallback_refreshes_n`
  - `parse_lineup_evolution(rel_path, data) -> Parsed`. One row per (line, slot): `obs_id, source_kind="lineup_evolution", source_path, line_no, captured_at, date, run_time, slot, batter_id, batter_name, team, p_game_hit, projected_lineup, game_pk`.

- [ ] **Step 1: Append builders** to `tests/scripts/season_ledger/builders.py`:

```python
def cand(batter_id: int, game_pk: int | None, *, team: str = "TB", name: str | None = None, p: float = 0.77) -> dict:
    return {"batter_id": batter_id, "batter_name": name or f"B{batter_id}", "team": team, "game_pk": game_pk,
            "p_game_hit": p}


def decision_json(date: str, *, action: str, primary: dict | None, double_down: dict | None = None,
                  schema: str = "bts_daily_decision_v3", **fields) -> bytes:
    """A decision record as bts.daily_decision.write_decision writes it (v1/v2 carry no objective)."""
    rec = {"schema_version": schema, "date": date, "action": action, "source": "mdp", "primary": primary,
           "double_down": double_down, "second_candidate": None, "streak": 0, "saver_available": None,
           "state_source": "contest", "state_status": "fresh", "allow_double": True, "contest_source_date": None,
           "delivery_status": "not_applicable" if action == "skip" else "delivered", "scoreable": action != "skip",
           "best_streak": None, "best_status": None, "effective_best": None, "tail_policy_sha256": None,
           "degraded_reason": None, "finalized_at": f"{date}T22:00:00.000000Z"}
    if schema == "bts_daily_decision_v3":
        rec["objective"] = "reach57"
    rec.update(fields)
    return dumps(rec)


def state_json(date: str, **fields) -> bytes:
    """A scheduler_state.json as scheduler.save_state writes asdict(SchedulerState)."""
    rec = {"date": date, "schedule_fetched_at": f"{date}T14:00:00-04:00", "games": [], "confirmed_game_pks": [],
           "runs_completed": [], "pick_locked": False, "pick_locked_at": None, "result_status": None,
           "next_wakeup": None, "final_skip_candidate": None, "committed_pick_written": False,
           "delivery_refusals": None, "fallback_refreshes": None}
    rec.update(fields)
    return dumps(rec)


def evolution_jsonl(date: str, entries: list[tuple[dict, dict | None]]) -> bytes:
    """entries: [(primary_slot, double_down_slot_or_None)] as append_lineup_evolution writes them."""
    lines = [json.dumps({"captured_at": f"{date}T1{i}:00:00+00:00", "date": date, "run_time": f"{date}T1{i}:00:00+00:00",
                         "primary": p, "double_down": d}) for i, (p, d) in enumerate(entries)]
    return ("\n".join(lines) + "\n").encode()
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_day_records.py`:

```python
from scripts.audit.season_ledger.sources.day_records import (parse_decision, parse_lineup_evolution,
                                                             parse_scheduler_state)
from tests.scripts.season_ledger.builders import cand, decision_json, dumps, evolution_jsonl, state_json


def test_v1_decision_reads_as_reach57_and_keeps_raw_objective():
    row = parse_decision("picks/2026-06-23/decision.json",
                         decision_json("2026-06-23", action="single", primary=cand(101, 5001),
                                       schema="bts_daily_decision_v1")).rows[0]
    assert (row["objective"], row["objective_raw"]) == ("reach57", None)
    assert (row["primary_batter_id"], row["primary_game_pk"], row["double_down_batter_id"]) == (101, 5001, None)


def test_invalid_v3_objective_is_unknown():
    row = parse_decision("picks/2026-09-05/decision.json",
                         decision_json("2026-09-05", action="double", primary=cand(101, 5001),
                                       double_down=cand(202, 5002), objective="bogus")).rows[0]
    assert row["objective"] == "unknown" and row["objective_raw"] == "bogus"


def test_action_source_keeps_raw_and_normalizes_unknown():
    for raw in ("forced", "unknown"):
        row = parse_decision("picks/2026-08-10/decision.json",
                             decision_json("2026-08-10", action="single", primary=cand(101, 5001), source=raw)).rows[0]
        assert (row["action_source_raw"], row["action_source"]) == (raw, "unknown")


def test_invalid_decision_record_is_quarantined():
    parsed = parse_decision("picks/2026-08-10/decision.json",
                            dumps({"schema_version": "bts_daily_decision_v3", "scoreable": True}))
    assert parsed.rows == [] and parsed.quarantined[0]["reason"] == "invalid_decision_record"


def test_scheduler_state_exposes_skip_candidate_and_refusal_archives():
    data = state_json("2026-09-19", final_skip_candidate={"primary": cand(303, 7001), "double": None},
                      delivery_refusals=[{"at": "2026-09-19T18:00:00-04:00",
                                          "archive": "refused_delivery_20260919T180000-0400.json"}])
    row = parse_scheduler_state("picks/2026-09-19/scheduler_state.json", data).rows[0]
    assert row["final_skip_candidate_present"] is True
    assert (row["skip_candidate_batter_id"], row["skip_candidate_game_pk"]) == (303, 7001)
    assert row["delivery_refusal_archives"] == "refused_delivery_20260919T180000-0400.json"


def test_lineup_evolution_rows_per_slot_and_bad_lines_quarantined():
    good = evolution_jsonl("2026-05-01", [({"batter_id": 101, "game_pk": 5001, "team": "TB"}, None),
                                          ({"batter_id": 111, "game_pk": 5011, "team": "TB"},
                                           {"batter_id": 202, "game_pk": 5002, "team": "NYY"})])
    parsed = parse_lineup_evolution("picks/lineup_evolution_2026-05-01.jsonl", good + b"{not json\n")
    assert [(r["line_no"], r["slot"], r["batter_id"], r["source_kind"]) for r in parsed.rows] == [
        (1, "primary", 101, "lineup_evolution"), (2, "primary", 111, "lineup_evolution"),
        (2, "double_down", 202, "lineup_evolution")]
    assert parsed.quarantined[0]["locator"] == "line=3"
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_day_records.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.sources.day_records'`.

- [ ] **Step 4: Write the implementation** — `scripts/audit/season_ledger/sources/day_records.py`:

```python
"""O3 decision files, O4 scheduler state, O5 lineup-evolution logs (spec §4)."""
from __future__ import annotations

import json

from bts.daily_decision import ACCEPTED_SCHEMAS, decision_objective

from ..ids import Parsed, load_json_bytes, obs_id, quarantine, sha256_hex

CANDIDATE_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "p_game_hit")
DECISION_FIELDS = ("schema_version", "date", "action", "streak", "saver_available", "state_source",
                   "state_status", "allow_double", "contest_source_date", "delivery_status", "scoreable",
                   "best_streak", "best_status", "effective_best", "tail_policy_sha256", "degraded_reason",
                   "finalized_at")
ACTION_SOURCES = ("mdp", "heuristic")
STATE_FIELDS = ("date", "schedule_fetched_at", "pick_locked", "pick_locked_at", "committed_pick_written",
                "result_status", "skip_notified_at")
EVOLUTION_FIELDS = ("batter_id", "batter_name", "team", "p_game_hit", "projected_lineup", "game_pk")


def parse_decision(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    # Same acceptance as bts.daily_decision.load_decision.
    if (not isinstance(doc, dict) or doc.get("schema_version") not in ACCEPTED_SCHEMAS
            or doc.get("action") not in {"skip", "single", "double"}
            or not isinstance(doc.get("scoreable"), bool) or "date" not in doc):
        return Parsed([], [quarantine(rel_path, "file", "invalid_decision_record")])
    row = {"obs_id": obs_id(rel_path, "file", content), "source_path": rel_path, "content_sha256": content}
    for f in DECISION_FIELDS:
        row[f] = doc.get(f)
    row["objective_raw"] = doc.get("objective")
    row["objective"] = decision_objective(doc)
    row["action_source_raw"] = doc.get("source")
    row["action_source"] = doc.get("source") if doc.get("source") in ACTION_SOURCES else "unknown"
    for name in ("primary", "double_down", "second_candidate"):
        cand = doc.get(name) if isinstance(doc.get(name), dict) else {}
        for f in CANDIDATE_FIELDS:
            row[f"{name}_{f}"] = cand.get(f)
    return Parsed([row], [])


def parse_scheduler_state(rel_path: str, data: bytes) -> Parsed:
    content = sha256_hex(data)
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or "date" not in doc:
        return Parsed([], [quarantine(rel_path, "file", "invalid_scheduler_state")])
    row = {"obs_id": obs_id(rel_path, "file", content), "source_path": rel_path, "content_sha256": content}
    for f in STATE_FIELDS:
        row[f] = doc.get(f)
    skip = doc.get("final_skip_candidate")
    primary = skip.get("primary") if isinstance(skip, dict) else None
    row["final_skip_candidate_present"] = isinstance(skip, dict)
    row["skip_candidate_batter_id"] = primary.get("batter_id") if isinstance(primary, dict) else None
    row["skip_candidate_game_pk"] = primary.get("game_pk") if isinstance(primary, dict) else None
    refusals = doc.get("delivery_refusals") or []
    row["delivery_refusal_archives"] = ",".join(sorted(r.get("archive") or "" for r in refusals
                                                       if isinstance(r, dict))) or None
    row["fallback_refreshes_n"] = len(doc.get("fallback_refreshes") or [])
    return Parsed([row], [])


def parse_lineup_evolution(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, f"line={line_no}", "invalid_json_line"))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, f"line={line_no}", "line_not_object"))
            continue
        for slot in ("primary", "double_down"):
            s = doc.get(slot)
            if not isinstance(s, dict):
                continue
            out.rows.append({"obs_id": obs_id(rel_path, f"line={line_no}/slot={slot}", content),
                             "source_kind": "lineup_evolution", "source_path": rel_path, "line_no": line_no,
                             "captured_at": doc.get("captured_at"), "date": doc.get("date"),
                             "run_time": doc.get("run_time"), "slot": slot,
                             **{f: s.get(f) for f in EVOLUTION_FIELDS}})
    return out
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_day_records.py -q`
Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/day_records.py tests/scripts/season_ledger
git commit -m "feat(ledger): decision, scheduler-state and lineup-evolution parsers (task 3)"
```

---

### Task 4: O6 contest ledger, slot history, streak chain, saver attempts

**Files:**
- Create: `scripts/audit/season_ledger/sources/contest_ledger.py`, `scripts/audit/season_ledger/contest.py`
- Modify: `tests/scripts/season_ledger/builders.py` (append `contest_line`, `rnd`, `slot`)
- Test: `tests/scripts/season_ledger/test_contest.py`

**Interfaces:**
- **Consumes:** Task 1.
- **Produces:**
  - `parse_contest_ledger(rel_path, data) -> Parsed`. Rows have `row_level ∈ {line, round, slot}`:
    - All rows: `obs_id, source_path, line_no, recorded_at` (fixed UTC), `source_date, active_streak, best_streak`.
    - Round and slot rows add `round_id, round_result, round_streak, round_streak_increase`.
    - Slot rows add `slot_number, unit_id, player_id, slot_result, hits, hits_state, at_bats, at_bats_state`, with `*_state ∈ {value, null, absent}`.
  - `parse_saver_transitions(rel_path, data) -> Parsed`. Rows: `obs_id, source_path, line_no, attempted_at, attempt_source, attempt_outcome`.
  - `contest.slot_history(contest_rows) -> list[dict]`. One row per `(round_id, unit_id, player_id)`: `first_seen, last_seen, last_line_no, n_observations, changed, dropped_later, last_obs_id`, plus the last values of `slot_result, hits, hits_state, at_bats, at_bats_state, slot_number, round_result, round_streak, round_streak_increase`.
  - `contest.line_round_streaks(contest_rows) -> dict[int, dict[int, int | None]]`
  - `contest.streak_before(line_streaks, round_id) -> int | None`

- [ ] **Step 1: Append builders** to `tests/scripts/season_ledger/builders.py`:

```python
def contest_line(recorded_at: str, rounds: list[dict], **fields) -> str:
    """One contest_ledger.jsonl line as the CLI appends it."""
    doc = {"recorded_at": recorded_at, "active_streak": 0, "best_streak": 18, "source_date": "2026-08-20",
           "predictions": rounds}
    doc.update(fields)
    return json.dumps(doc)


def rnd(round_id: int, result, streak, increase, slots: list[dict]) -> dict:
    return {"roundId": round_id, "result": result, "streak": streak, "streakIncrease": increase,
            "roundPredictions": slots}


def slot(unit_id: int, player_id, result, *, number: int = 1, hits=1, at_bats=4, drop: tuple = ()) -> dict:
    s = {"number": number, "unitId": unit_id, "playerId": player_id, "result": result, "hits": hits, "atBats": at_bats}
    for key in drop:
        s.pop(key)
    return s
```

- [ ] **Step 2: Write the failing tests** — `tests/scripts/season_ledger/test_contest.py`:

```python
import json

from scripts.audit.season_ledger.contest import line_round_streaks, slot_history, streak_before
from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from tests.scripts.season_ledger.builders import contest_line, rnd, slot

PATH = "picks/account_state/contest_ledger.jsonl"


def _parse(*lines):
    return parse_contest_ledger(PATH, ("\n".join(lines) + "\n").encode())


def test_null_and_absent_stats_are_kept_distinct():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [
        slot(1928, 2513, "hit", hits=None), slot(1927, 1300, "hit", number=2, drop=("atBats",))])]))
    slots = {r["player_id"]: r for r in parsed.rows if r["row_level"] == "slot"}
    assert (slots[2513]["hits"], slots[2513]["hits_state"]) == (None, "null")
    assert (slots[1300]["at_bats"], slots[1300]["at_bats_state"]) == (None, "absent")
    assert (slots[1300]["hits"], slots[1300]["hits_state"]) == (1, "value")


def test_line_with_a_playerless_slot_is_quarantined_whole():
    # Review Focus 3 / Interpretation I1: identity must be complete.
    playerless = slot(1928, None, None, hits=None, at_bats=None)
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit"), playerless])]))
    assert parsed.rows == []
    assert parsed.quarantined == [{"source_path": PATH, "locator": "line=1", "reason": "slot_missing_identity_or_result"}]


def test_in_progress_round_with_null_result_qualifies():
    parsed = _parse(contest_line("2026-08-22T20:00:00Z", [rnd(972, None, None, None,
                                                              [slot(1930, 99, None, hits=None, at_bats=None)])]))
    (s,) = [r for r in parsed.rows if r["row_level"] == "slot"]
    assert parsed.quarantined == [] and (s["slot_result"], s["round_result"], s["round_streak"]) == (None, None, None)


def test_slotless_round_and_line_rows_are_emitted():
    parsed = _parse(contest_line("2026-05-14T14:30:00Z", [rnd(873, "void", 0, 0, [])]))
    assert sorted(r["row_level"] for r in parsed.rows) == ["line", "round"]


def test_round_growth_and_later_drop_are_tracked():
    parsed = _parse(
        contest_line("2026-08-20T20:00:00Z", [rnd(971, "hit", 7, 1, [slot(1928, 2513, "hit")])]),
        contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit"),
                                                                     slot(1927, 1300, "hit", number=2)])]),
        contest_line("2026-08-22T14:30:00Z", [rnd(972, "not_hit", 0, -8, [slot(1930, 99, "not_hit", hits=0)])]))
    hist = {(h["round_id"], h["player_id"]): h for h in slot_history(parsed.rows)}
    a, b = hist[(971, 2513)], hist[(971, 1300)]
    assert (a["first_seen"], a["last_seen"], a["n_observations"]) == (
        "2026-08-20T20:00:00.000000Z", "2026-08-21T14:30:00.000000Z", 2)
    assert a["changed"] is True and b["changed"] is False
    assert b["first_seen"] == "2026-08-21T14:30:00.000000Z"
    assert a["dropped_later"] is True and b["dropped_later"] is True     # round 971 absent from the 8/22 line
    assert hist[(972, 99)]["dropped_later"] is False
    assert (a["slot_result"], a["round_streak"]) == ("hit", 8)           # the older positive is kept


def test_a_malformed_later_line_does_not_mark_older_slots_dropped():
    parsed = _parse(contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])]),
                    '{"recorded_at": "2026-08-22T14:30:00Z", "predictions": [{"result": "hit"}]}')
    assert parsed.quarantined[0]["locator"] == "line=2"
    (h,) = slot_history(parsed.rows)
    assert h["dropped_later"] is False and h["last_seen"] == "2026-08-21T14:30:00.000000Z"


def test_streak_before_uses_the_previous_round_in_the_same_line():
    # Interpretation I2: 970 was a skip day (no round), so 971's previous round is 969.
    parsed = _parse(contest_line("2026-08-22T14:30:00Z", [
        rnd(969, "hit", 5, 1, [slot(1900, 11, "hit")]),
        rnd(971, "hit", 7, 2, [slot(1928, 2513, "hit")]),
        rnd(972, None, None, None, [slot(1930, 99, None, hits=None, at_bats=None)])]))
    streaks = line_round_streaks(parsed.rows)[1]
    assert streak_before(streaks, 971) == 5
    assert streak_before(streaks, 969) is None
    assert streak_before(streaks, 972) == 7


def test_saver_transitions_are_reported_as_attempts():
    data = (json.dumps({"ts": "2026-08-10T12:00:00+00:00", "source": "auto", "outcome": "rejected",
                        "new_state": "used"}) + "\n").encode()
    (row,) = parse_saver_transitions("picks/account_state/saver_transitions.jsonl", data).rows
    assert (row["attempted_at"], row["attempt_source"], row["attempt_outcome"]) == (
        "2026-08-10T12:00:00.000000Z", "auto", "rejected")
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_contest.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.contest'`.

- [ ] **Step 4: Write the implementation**

`scripts/audit/season_ledger/sources/contest_ledger.py`:
```python
"""O6 contest ledger (`account_state/contest_ledger.jsonl`), lossless, and the saver-transition
attempts log (spec §4, §6)."""
from __future__ import annotations

import json

from ..ids import Parsed, is_int, obs_id, quarantine, sha256_hex, utc_iso


def _stat(slot: dict, key: str) -> tuple[object, str]:
    if key not in slot:
        return None, "absent"
    return (slot[key], "value") if slot[key] is not None else (None, "null")


def _disqualify(doc) -> str | None:
    """Interpretation I1: identity ints required; `result` keys present (null = in progress)."""
    if not isinstance(doc, dict) or utc_iso(doc.get("recorded_at")) is None or not isinstance(doc.get("predictions"), list):
        return "line_missing_recorded_at_or_predictions"
    for rnd in doc["predictions"]:
        if not isinstance(rnd, dict) or not is_int(rnd.get("roundId")) or "result" not in rnd:
            return "round_missing_roundId_or_result"
        slots = rnd.get("roundPredictions")
        if slots is not None and not isinstance(slots, list):
            return "round_predictions_not_list"
        for s in slots or []:
            if (not isinstance(s, dict) or not is_int(s.get("unitId")) or not is_int(s.get("playerId"))
                    or "result" not in s):
                return "slot_missing_identity_or_result"
    return None


def _lines(rel_path: str, data: bytes):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.strip()]


def parse_contest_ledger(rel_path: str, data: bytes) -> Parsed:
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(rel_path, data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        try:
            doc = json.loads(line)
            reason = _disqualify(doc)
        except json.JSONDecodeError:
            reason = "invalid_json_line"
        if reason:
            out.quarantined.append(quarantine(rel_path, f"line={line_no}", reason))
            continue
        base = {"source_path": rel_path, "line_no": line_no, "recorded_at": utc_iso(doc["recorded_at"]),
                "source_date": doc.get("source_date"), "active_streak": doc.get("active_streak"),
                "best_streak": doc.get("best_streak")}
        out.rows.append(dict(base, row_level="line", obs_id=obs_id(rel_path, f"line={line_no}", content)))
        for i, rnd in enumerate(doc["predictions"]):
            rbase = dict(base, round_id=rnd["roundId"], round_result=rnd["result"],
                         round_streak=rnd.get("streak"), round_streak_increase=rnd.get("streakIncrease"))
            slots = rnd.get("roundPredictions") or []
            if not slots:
                out.rows.append(dict(rbase, row_level="round", obs_id=obs_id(rel_path, f"line={line_no}/round={i}", content)))
            for j, s in enumerate(slots):
                hits, hits_state = _stat(s, "hits")
                at_bats, at_bats_state = _stat(s, "atBats")
                out.rows.append(dict(rbase, row_level="slot", obs_id=obs_id(rel_path, f"line={line_no}/round={i}/slot={j}", content),
                                     slot_number=s.get("number"), unit_id=s["unitId"], player_id=s["playerId"],
                                     slot_result=s["result"], hits=hits, hits_state=hits_state,
                                     at_bats=at_bats, at_bats_state=at_bats_state))
    return out


def parse_saver_transitions(rel_path: str, data: bytes) -> Parsed:
    """Rows are attempts (including rejected ones), never consumption times (spec §6)."""
    out = Parsed()
    content = sha256_hex(data)
    lines = _lines(rel_path, data)
    if lines is None:
        return Parsed([], [quarantine(rel_path, "file", "not_utf8")])
    for line_no, line in lines:
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            out.quarantined.append(quarantine(rel_path, f"line={line_no}", "invalid_json_line"))
            continue
        if not isinstance(doc, dict):
            out.quarantined.append(quarantine(rel_path, f"line={line_no}", "line_not_object"))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, f"line={line_no}", content), "source_path": rel_path,
                         "line_no": line_no, "attempted_at": utc_iso(doc.get("ts")),
                         "attempt_source": None if doc.get("source") is None else str(doc.get("source")),
                         "attempt_outcome": None if doc.get("outcome") is None else str(doc.get("outcome"))})
    return out
```

`scripts/audit/season_ledger/contest.py`:
```python
"""Contest slot history, the streak chain, lookups and game matching (spec §6)."""
from __future__ import annotations

SLOT_VALUE_KEYS = ("slot_result", "hits", "hits_state", "at_bats", "at_bats_state", "slot_number",
                   "round_result", "round_streak", "round_streak_increase")


def _order(row: dict) -> tuple:
    return (row["recorded_at"], row["line_no"])     # fixed-precision UTC: string order is time order


def slot_history(contest_rows: list[dict]) -> list[dict]:
    """Per slot identity (round_id, unit_id, player_id): first/last seen, the last qualified values,
    whether values changed, and `dropped_later` when a later qualified line no longer shows it.
    A newer omission never erases the older positive observation."""
    lines = sorted({_order(r) for r in contest_rows if r["row_level"] == "line"})
    groups: dict[tuple, list[dict]] = {}
    for r in sorted((r for r in contest_rows if r["row_level"] == "slot"), key=_order):
        groups.setdefault((r["round_id"], r["unit_id"], r["player_id"]), []).append(r)
    out = []
    for (round_id, unit_id, player_id), obs in sorted(groups.items()):
        last = obs[-1]
        out.append({"round_id": round_id, "unit_id": unit_id, "player_id": player_id,
                    "first_seen": obs[0]["recorded_at"], "last_seen": last["recorded_at"],
                    "last_line_no": last["line_no"], "n_observations": len(obs),
                    "changed": len({tuple(o[k] for k in SLOT_VALUE_KEYS) for o in obs}) > 1,
                    "dropped_later": any(line > _order(last) for line in lines),
                    "last_obs_id": last["obs_id"], **{k: last[k] for k in SLOT_VALUE_KEYS}})
    return out


def line_round_streaks(contest_rows: list[dict]) -> dict[int, dict[int, int | None]]:
    """line_no → {round_id: reported post-round streak} for every round present in that line."""
    out: dict[int, dict[int, int | None]] = {}
    for r in contest_rows:
        if r["row_level"] in ("round", "slot"):
            out.setdefault(r["line_no"], {})[r["round_id"]] = r["round_streak"]
    return out


def streak_before(line_streaks: dict[int, int | None], round_id: int) -> int | None:
    """Interpretation I2: the reported streak of the nearest earlier round present in the same line
    (skip days have no round and leave the streak unchanged); None without one."""
    earlier = [r for r in line_streaks if r < round_id]
    return line_streaks[max(earlier)] if earlier else None
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_contest.py -q`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/sources/contest_ledger.py scripts/audit/season_ledger/contest.py tests/scripts/season_ledger
git commit -m "feat(ledger): lossless contest ledger, slot history, streak chain, saver attempts (task 4)"
```

---

### Task 5: O7 static captures, MLB schedules, lookups

**Files:**
- Create: `scripts/audit/season_ledger/sources/static.py`
- Modify: `scripts/audit/season_ledger/contest.py` (append lookups)
- Test: `tests/scripts/season_ledger/test_static.py`

**Interfaces:**
- **Produces (parsers).** Item locators are positional (`item=<i>`), so duplicate items never collide:
  - `parse_rounds(rel_path, data)` → rows `{obs_id, source_path, round_id, round_date}`
  - `parse_players(...)` → rows `{obs_id, source_path, player_id, feed_id, squad_id, name}`
  - `parse_units(...)` → rows `{obs_id, source_path, unit_id, feed_id, round_id, status, captured_at}`, where `captured_at` comes from a `YYYYmmddTHHMMSSZ` file stamp and is otherwise None
  - `parse_schedule(rel_path, data)` → rows `{obs_id, source_path, query_date, game_pk, away_abbr, home_abbr, coded_state, detailed_state, official_date, game_number}`; `rel_path` must be `schedules/YYYY-MM-DD.json`
- **Produces (lookups in `contest.py`):**
  - `rounds_lookup(rows) -> dict[int, set[str]]`
  - `players_lookup(rows) -> dict[int, set[int]]`
  - `units_lookup(rows) -> dict[int, dict]`, each value `{"feed_ids": set[int], "round_ids": set[int]}`
  - `unit_status_history(rows) -> dict[int, list[tuple[str | None, str | None]]]`
  - `team_games(rows) -> dict[tuple[str, str], set[int]]`, keyed by `(query_date, team_abbr)` over every listed game, any status

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_static.py`:

```python
import gzip

from scripts.audit.season_ledger.contest import (players_lookup, rounds_lookup, team_games, unit_status_history,
                                                 units_lookup)
from scripts.audit.season_ledger.sources.static import parse_players, parse_rounds, parse_schedule, parse_units
from tests.scripts.season_ledger.builders import dumps


def test_gzip_and_plain_captures_both_parse_and_corrupt_gzip_is_quarantined():
    # Review Focus 4
    body = dumps({"rounds": [{"id": 971, "date": "2026-08-20T08:00:00-04:00", "status": "complete"}]})
    assert parse_rounds("static/rounds/20260704T030011Z.json", body).rows[0]["round_date"] == "2026-08-20"
    assert parse_rounds("static/rounds/20260705T030011Z.json.gz", gzip.compress(body)).rows[0]["round_id"] == 971
    bad = parse_rounds("static/rounds/20260706T030011Z.json.gz", b"\x1f\x8bgarbage")
    assert bad.rows == [] and bad.quarantined[0]["reason"].startswith("bad_gzip")


def test_units_carry_status_and_capture_time_and_empty_lists_have_no_rows():
    rows = parse_units("static/units/20260801T150000Z.json.gz", gzip.compress(dumps({"units": [
        {"id": 2449, "feedId": 822679, "roundId": 1009, "status": "postponed"}]}))).rows
    assert (rows[0]["captured_at"], rows[0]["status"]) == ("2026-08-01T15:00:00.000000Z", "postponed")
    empty = parse_units("static/units/20260927T230002Z.json.gz", gzip.compress(dumps({"units": []})))
    assert empty.rows == [] and empty.quarantined == []
    assert unit_status_history(rows) == {2449: [("2026-08-01T15:00:00.000000Z", "postponed")]}
    assert parse_units("static/grab_20260927/003_units.json.gz", gzip.compress(dumps({"units": [
        {"id": 1, "feedId": 2, "roundId": 3, "status": "scheduled"}]}))).rows[0]["captured_at"] is None


def test_lookups_keep_conflicting_captures():
    units = parse_units("static/units/20260801T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 822679, "roundId": 1009}]})).rows \
        + parse_units("static/units/20260802T150000Z.json", dumps({"units": [{"id": 2449, "feedId": 999999, "roundId": 1009}]})).rows
    assert units_lookup(units)[2449] == {"feed_ids": {822679, 999999}, "round_ids": {1009}}
    players = parse_players("static/players/p.json", dumps({"players": [{"id": 1300, "feedId": 680757, "squadId": 5,
                                                                          "name": "Steven Kwan"}]})).rows
    assert players_lookup(players) == {1300: {680757}}
    rounds = parse_rounds("static/rounds/r.json", dumps({"rounds": [{"id": 1, "date": "2026-03-25T08:00:00-04:00"}]})).rows
    assert rounds_lookup(rounds) == {1: {"2026-03-25"}}


def test_schedule_lists_every_game_including_postponed():
    def game(pk, number, state):
        return {"gamePk": pk, "gameNumber": number, "officialDate": "2026-05-10",
                "status": {"codedGameState": state[0], "detailedState": state},
                "teams": {"away": {"team": {"abbreviation": "TB"}}, "home": {"team": {"abbreviation": "BOS"}}}}
    body = dumps({"dates": [{"date": "2026-05-10", "games": [game(824765, 1, "Final"), game(824999, 2, "Postponed")]}]})
    rows = parse_schedule("schedules/2026-05-10.json", body).rows
    assert team_games(rows)[("2026-05-10", "TB")] == {824765, 824999}
    assert {r["detailed_state"] for r in rows} == {"Final", "Postponed"}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_static.py -q`
Expected: collection ERROR — `cannot import name 'players_lookup'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/sources/static.py`:
```python
"""O7 BTS static captures (rounds, players, units) and acquired MLB schedule responses (spec §4, §6).
Item locators are positional so a capture that lists an id twice still yields distinct occurrences."""
from __future__ import annotations

import re

from ..ids import Parsed, is_int, load_json_bytes, obs_id, quarantine, sha256_hex, stamp_to_utc

_SCHEDULE_PATH = re.compile(r"^schedules/(\d{4}-\d{2}-\d{2})\.json$")


def _items(rel_path: str, data: bytes, key: str):
    try:
        doc = load_json_bytes(data)
    except ValueError as exc:
        return None, Parsed([], [quarantine(rel_path, "file", str(exc))])
    if not isinstance(doc, dict) or not isinstance(doc.get(key), list):
        return None, Parsed([], [quarantine(rel_path, "file", f"missing_{key}_list")])
    return doc[key], None


def parse_rounds(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "rounds")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, r in enumerate(items):
        if not isinstance(r, dict) or not is_int(r.get("id")) or not isinstance(r.get("date"), str):
            out.quarantined.append(quarantine(rel_path, f"item={i}", "round_missing_id_or_date"))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, f"item={i}", content), "source_path": rel_path,
                         "round_id": r["id"], "round_date": r["date"][:10]})
    return out


def parse_players(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "players")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, p in enumerate(items):
        if not isinstance(p, dict) or not is_int(p.get("id")):
            out.quarantined.append(quarantine(rel_path, f"item={i}", "player_missing_id"))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, f"item={i}", content), "source_path": rel_path,
                         "player_id": p["id"], "feed_id": p.get("feedId") if is_int(p.get("feedId")) else None,
                         "squad_id": p.get("squadId"), "name": p.get("name")})
    return out


def parse_units(rel_path: str, data: bytes) -> Parsed:
    items, bad = _items(rel_path, data, "units")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    captured_at = stamp_to_utc(rel_path.rsplit("/", 1)[-1])
    for i, u in enumerate(items):
        if not isinstance(u, dict) or not is_int(u.get("id")):
            out.quarantined.append(quarantine(rel_path, f"item={i}", "unit_missing_id"))
            continue
        out.rows.append({"obs_id": obs_id(rel_path, f"item={i}", content), "source_path": rel_path,
                         "unit_id": u["id"], "feed_id": u.get("feedId") if is_int(u.get("feedId")) else None,
                         "round_id": u.get("roundId") if is_int(u.get("roundId")) else None,
                         "status": u.get("status"), "captured_at": captured_at})
    return out


def parse_schedule(rel_path: str, data: bytes) -> Parsed:
    m = _SCHEDULE_PATH.match(rel_path)
    if not m:
        return Parsed([], [quarantine(rel_path, "file", "unexpected_schedule_path")])
    items, bad = _items(rel_path, data, "dates")
    if bad is not None:
        return bad
    content, out = sha256_hex(data), Parsed()
    for i, day in enumerate(items):
        for j, g in enumerate((day or {}).get("games") or [] if isinstance(day, dict) else []):
            if not isinstance(g, dict) or not is_int(g.get("gamePk")):
                out.quarantined.append(quarantine(rel_path, f"date={i}/game={j}", "game_missing_gamePk"))
                continue
            teams, status = g.get("teams") or {}, g.get("status") or {}
            out.rows.append({"obs_id": obs_id(rel_path, f"date={i}/game={j}", content), "source_path": rel_path,
                             "query_date": m.group(1), "game_pk": g["gamePk"],
                             "away_abbr": ((teams.get("away") or {}).get("team") or {}).get("abbreviation"),
                             "home_abbr": ((teams.get("home") or {}).get("team") or {}).get("abbreviation"),
                             "coded_state": status.get("codedGameState"), "detailed_state": status.get("detailedState"),
                             "official_date": g.get("officialDate"), "game_number": g.get("gameNumber")})
    return out
```

Append to `scripts/audit/season_ledger/contest.py`:
```python
def rounds_lookup(round_rows: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for r in round_rows:
        out.setdefault(r["round_id"], set()).add(r["round_date"])
    return out


def players_lookup(player_rows: list[dict]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for p in player_rows:
        if p["feed_id"] is not None:
            out.setdefault(p["player_id"], set()).add(p["feed_id"])
    return out


def units_lookup(unit_rows: list[dict]) -> dict[int, dict]:
    """unit_id → every feedId / roundId any capture recorded; conflicts are kept, never collapsed."""
    out: dict[int, dict] = {}
    for u in unit_rows:
        entry = out.setdefault(u["unit_id"], {"feed_ids": set(), "round_ids": set()})
        if u["feed_id"] is not None:
            entry["feed_ids"].add(u["feed_id"])
        if u["round_id"] is not None:
            entry["round_ids"].add(u["round_id"])
    return out


def unit_status_history(unit_rows: list[dict]) -> dict[int, list[tuple]]:
    """unit_id → sorted [(capture time, status)] across captures (Interpretation I6)."""
    out: dict[int, list[tuple]] = {}
    for u in unit_rows:
        out.setdefault(u["unit_id"], []).append((u["captured_at"], u["status"]))
    return {k: sorted(v, key=lambda x: (x[0] or "", str(x[1]))) for k, v in out.items()}


def team_games(schedule_rows: list[dict]) -> dict[tuple[str, str], set[int]]:
    """(query date, team abbreviation) → every listed gamePk, whatever its status (postponed,
    cancelled and suspended entries included), so 'exactly one game' is never produced by filtering."""
    out: dict[tuple[str, str], set[int]] = {}
    for g in schedule_rows:
        for abbr in (g["away_abbr"], g["home_abbr"]):
            if abbr:
                out.setdefault((g["query_date"], abbr), set()).add(g["game_pk"])
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_static.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger tests/scripts/season_ledger
git commit -m "feat(ledger): static capture and schedule parsers, conflict-keeping lookups (task 5)"
```

---

### Task 6: Contest slot → game → selection matching

**Files:**
- Modify: `scripts/audit/season_ledger/contest.py` (append `match_slot`, `resolve_duplicate_links`)
- Test: `tests/scripts/season_ledger/test_matching.py`

**Interfaces:**
- **Consumes:** Task 4 slot-history rows; Task 5 lookups; a `local_selections` list whose rows have `date, batter_id, game_pk, team_at_pick, selection_id`.
- **Produces:**
  - `match_slot(slot, *, rounds, players, units, team_games, schedule_dates, local_selections) -> dict`. Keys: `round_id, unit_id, player_id, date, batter_id, game_pk, selection_id, match ∈ {evidenced, inferred, ambiguous, unmapped}, match_reason`.
  - `resolve_duplicate_links(matches) -> list[dict]` (Interpretation I8).

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_matching.py`:

```python
from scripts.audit.season_ledger.contest import match_slot, resolve_duplicate_links

ROUNDS = {971: {"2026-08-20"}}
PLAYERS = {2513: {802415}, 1300: {680757}}
SEL = {"date": "2026-08-20", "batter_id": 802415, "game_pk": 822934, "team_at_pick": "TB",
       "selection_id": "2026-08-20|primary|802415|822934"}
TB_ONE = {("2026-08-20", "TB"): {822934}}


def _slot(unit_id=1928, player_id=2513, round_id=971):
    return {"round_id": round_id, "unit_id": unit_id, "player_id": player_id}


def _match(*, units=None, games=None, sels=(SEL,), slot=None, dates=("2026-08-20",)):
    return match_slot(slot or _slot(), rounds=ROUNDS, players=PLAYERS, units=units or {},
                      team_games=TB_ONE if games is None else games, schedule_dates=set(dates),
                      local_selections=list(sels))


def test_unit_capture_gives_evidenced_match_to_the_local_selection():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {971}}}, games={})
    assert (m["match"], m["game_pk"], m["selection_id"], m["match_reason"]) == (
        "evidenced", 822934, SEL["selection_id"], "unit_capture")


def test_conflicting_unit_captures_are_ambiguous_and_never_fall_through_to_inference():
    m = _match(units={1928: {"feed_ids": {822934, 900001}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "conflicting_unit_evidence")


def test_unit_capture_seen_in_another_round_is_ambiguous():
    m = _match(units={1928: {"feed_ids": {822934}, "round_ids": {970, 971}}})
    assert (m["match"], m["match_reason"]) == ("ambiguous", "unit_round_contradiction")


def test_unit_capture_for_another_game_does_not_link():
    m = _match(units={1928: {"feed_ids": {900002}, "round_ids": {971}}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("evidenced", None, "unit_capture_other_game")


def test_inference_uses_the_pick_time_team_not_the_current_one():
    # A traded player: a current-team lookup would say SEA; the pick file recorded TB.
    m = _match(games={("2026-08-20", "TB"): {822934}, ("2026-08-20", "SEA"): {900001}})
    assert (m["match"], m["game_pk"], m["match_reason"]) == ("inferred", 822934, "pick_time_team_single_scheduled_game")


def test_doubleheader_or_postponed_plus_played_game_is_ambiguous():
    m = _match(games={("2026-08-20", "TB"): {822934, 822935}})
    assert (m["match"], m["selection_id"], m["match_reason"]) == ("ambiguous", None, "team_schedule_not_unique")


def test_selection_without_a_recorded_game_is_ambiguous():
    # Review Focus 2: the 3/29–3/30 pick files carry game_pk = null.
    m = _match(sels=(dict(SEL, game_pk=None, selection_id="2026-08-20|primary|802415|None"),))
    assert (m["match"], m["match_reason"]) == ("ambiguous", "selection_game_pk_unrecorded")


def test_missing_schedule_and_absent_team_are_ambiguous_with_distinct_reasons():
    assert _match(dates=())["match_reason"] == "team_schedule_missing"
    assert _match(games={("2026-08-20", "NYY"): {5002}})["match_reason"] == "team_not_on_schedule"


def test_contest_only_slot_without_unit_capture_is_unmapped():
    m = _match(sels=())
    assert (m["match"], m["match_reason"]) == ("unmapped", "no_unit_capture_contest_only")


def test_unknown_round_is_unmapped_and_mapped_and_unmapped_slots_share_a_round():
    assert _match(slot=_slot(round_id=5))["match_reason"] == "round_date_unknown"
    known, unknown = _match(), _match(slot=_slot(unit_id=1927, player_id=42))
    assert (known["match"], unknown["match"], unknown["match_reason"]) == ("inferred", "unmapped", "player_unknown")


def test_two_contest_slots_linking_one_selection_are_both_demoted():
    a = {"round_id": 971, "unit_id": 1928, "player_id": 2513, "selection_id": "s", "match": "inferred",
         "match_reason": "x"}
    out = resolve_duplicate_links([a, dict(a, unit_id=1929), dict(a, unit_id=1930, selection_id="t")])
    assert [(m["unit_id"], m["match"], m["selection_id"]) for m in out] == [
        (1928, "ambiguous", None), (1929, "ambiguous", None), (1930, "inferred", "t")]
    assert out[0]["match_reason"] == "multiple_contest_slots_for_selection"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_matching.py -q`
Expected: collection ERROR — `cannot import name 'match_slot'`.

- [ ] **Step 3: Write the implementation** (append to `scripts/audit/season_ledger/contest.py`; add `from collections import Counter` to its imports):

```python
def _result(slot: dict, *, date=None, batter_id=None, game_pk=None, selection_id=None, match: str, reason: str) -> dict:
    return {"round_id": slot["round_id"], "unit_id": slot["unit_id"], "player_id": slot["player_id"],
            "date": date, "batter_id": batter_id, "game_pk": game_pk, "selection_id": selection_id,
            "match": match, "match_reason": reason}


def match_slot(slot: dict, *, rounds: dict, players: dict, units: dict, team_games: dict,
               schedule_dates: set, local_selections: list[dict]) -> dict:
    """Spec §6. Never matches on the slot `number`. Only a unit with no capture at all may use the
    pick-time-team inference; unit evidence that is not one round-consistent feedId is ambiguous and
    transfers nothing. Only `evidenced` / `inferred` links carry a selection_id."""
    dates = rounds.get(slot["round_id"], set())
    if len(dates) != 1:
        return _result(slot, match="unmapped", reason="round_date_unknown" if not dates else "round_date_conflict")
    date = next(iter(dates))
    feeds = players.get(slot["player_id"], set())
    if len(feeds) != 1:
        return _result(slot, date=date, match="unmapped", reason="player_unknown" if not feeds else "player_conflict")
    batter = next(iter(feeds))
    candidates = [s for s in local_selections if s["date"] == date and s["batter_id"] == batter]
    unit = units.get(slot["unit_id"])
    if unit is not None:
        if len(unit["feed_ids"]) != 1:
            reason = "unit_capture_without_feed_id" if not unit["feed_ids"] else "conflicting_unit_evidence"
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)
        if unit["round_ids"] and unit["round_ids"] != {slot["round_id"]}:
            return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="unit_round_contradiction")
        game = next(iter(unit["feed_ids"]))
        same = [s for s in candidates if s["game_pk"] == game]
        if len(same) == 1:
            reason = "unit_capture"
        elif same:
            reason = "unit_capture_multiple_local"
        else:
            reason = "unit_capture_other_game" if candidates else "unit_capture_no_local_selection"
        return _result(slot, date=date, batter_id=batter, game_pk=game,
                       selection_id=same[0]["selection_id"] if len(same) == 1 else None,
                       match="evidenced", reason=reason)
    if not candidates:
        return _result(slot, date=date, batter_id=batter, match="unmapped", reason="no_unit_capture_contest_only")
    if len(candidates) > 1:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="multiple_local_selections")
    sel = candidates[0]
    if sel["game_pk"] is None:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="selection_game_pk_unrecorded")
    if date not in schedule_dates:
        return _result(slot, date=date, batter_id=batter, match="ambiguous", reason="team_schedule_missing")
    games = team_games.get((date, sel["team_at_pick"]), set())
    if games == {sel["game_pk"]}:
        return _result(slot, date=date, batter_id=batter, game_pk=sel["game_pk"], selection_id=sel["selection_id"],
                       match="inferred", reason="pick_time_team_single_scheduled_game")
    if not games:
        reason = "team_not_on_schedule"
    else:
        reason = "team_schedule_not_unique" if len(games) > 1 else "team_schedule_other_game"
    return _result(slot, date=date, batter_id=batter, match="ambiguous", reason=reason)


def resolve_duplicate_links(matches: list[dict]) -> list[dict]:
    """Interpretation I8: two contest slot identities linking one selection (an entry changed within
    a round) are both demoted to ambiguous; neither transfers a grade."""
    counts = Counter(m["selection_id"] for m in matches if m["selection_id"])
    return [dict(m, match="ambiguous", match_reason="multiple_contest_slots_for_selection", selection_id=None)
            if m["selection_id"] and counts[m["selection_id"]] > 1 else m for m in matches]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_matching.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/contest.py tests/scripts/season_ledger/test_matching.py
git commit -m "feat(ledger): contest slot to game matching (evidenced / inferred / ambiguous / unmapped) (task 6)"
```

---

### Task 7: Outcome normalization and comparison

**Files:**
- Create: `scripts/audit/season_ledger/outcomes.py`
- Test: `tests/scripts/season_ledger/test_outcomes.py`

**Interfaces:**
- **Produces:**
  - `LOCAL_NORMALIZATION` and `CONTEST_NORMALIZATION`, both closed dicts
  - `normalize_local(raw) -> str | None` and `normalize_contest(raw) -> str | None`: None for None input, `"UNKNOWN"` for unlisted labels
  - `slot_disagreement(local_norm, contest_norm) -> bool | None`
  - `derived_single_result(pick_row: dict | None) -> tuple[str | None, str | None]`

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_outcomes.py`:

```python
from scripts.audit.season_ledger.outcomes import (derived_single_result, normalize_contest, normalize_local,
                                                  slot_disagreement)


def test_local_miss_and_contest_not_hit_agree():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("not_hit")) is False


def test_void_maps_to_hold_on_both_sides():
    assert normalize_local("void") == normalize_contest("void") == "HOLD"


def test_c03_pattern_disagrees():
    assert slot_disagreement(normalize_local("miss"), normalize_contest("hit")) is True


def test_unknown_or_missing_labels_are_never_compared():
    assert normalize_local("suspended") == normalize_local("unresolved") == "UNKNOWN"
    assert slot_disagreement("UNKNOWN", "HIT") is None and slot_disagreement(None, "HIT") is None


def test_round_labels_are_not_slot_labels():
    assert normalize_contest("used_mulligan") == "UNKNOWN"


def test_single_pick_day_result_derives_only_for_single_pick_records():
    single = {"obs_id": "o1", "has_double_down": False, "slot_result_raw": None, "day_result_raw": "hit"}
    assert derived_single_result(single) == ("hit", "day_result_of_single_pick:o1")
    assert derived_single_result(dict(single, has_double_down=True, day_result_raw="miss")) == (None, None)
    assert derived_single_result(dict(single, slot_result_raw="miss")) == (None, None)
    assert derived_single_result(None) == (None, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_outcomes.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.outcomes'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/outcomes.py`:

```python
"""Closed outcome vocabulary and slot-level comparison (spec §7). HOLD = the slot neither extended
nor broke the streak (a Pass or a voided slot); round labels (`void`, `used_mulligan`, …) are not
part of this table."""
from __future__ import annotations

LOCAL_NORMALIZATION = {"hit": "HIT", "miss": "NO_HIT", "void": "HOLD"}
CONTEST_NORMALIZATION = {"hit": "HIT", "not_hit": "NO_HIT", "void": "HOLD"}
COMPARABLE = frozenset({"HIT", "NO_HIT", "HOLD"})


def normalize_local(raw: str | None) -> str | None:
    return None if raw is None else LOCAL_NORMALIZATION.get(raw, "UNKNOWN")


def normalize_contest(raw: str | None) -> str | None:
    return None if raw is None else CONTEST_NORMALIZATION.get(raw, "UNKNOWN")


def slot_disagreement(local_norm: str | None, contest_norm: str | None) -> bool | None:
    if local_norm not in COMPARABLE or contest_norm not in COMPARABLE:
        return None
    return local_norm != contest_norm


def derived_single_result(pick_row: dict | None) -> tuple[str | None, str | None]:
    """A per-slot value from the day result only for a record that is itself a single pick and has
    no per-slot result, bound to that record (spec §7)."""
    if pick_row is None or pick_row["has_double_down"] or pick_row["slot_result_raw"] is not None:
        return None, None
    if pick_row["day_result_raw"] in LOCAL_NORMALIZATION:
        return pick_row["day_result_raw"], f"day_result_of_single_pick:{pick_row['obs_id']}"
    return None, None
```

- [ ] **Step 4: Run to verify pass** — same command. Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/outcomes.py tests/scripts/season_ledger/test_outcomes.py
git commit -m "feat(ledger): closed outcome vocabulary, slot comparison, single-pick derivation (task 7)"
```

---

### Task 8: Day status, row kinds, commit / history status, delivery predicate

**Files:**
- Create: `scripts/audit/season_ledger/rows.py`
- Test: `tests/scripts/season_ledger/test_rows.py`

**Interfaces:**
- **Consumes:** parsed rows from Tasks 2–3 for one date:
  - the decision row or None
  - the production pick-file slot rows
  - the scheduler-state row or None
  - `observations`: archive, manual- and repair-version rows and lineup-evolution rows
- **Produces:**
  - `selection_id(date, slot, batter_id, game_pk) -> str`, giving `"<date>|<slot>|<batter_id>|<game_pk>"`
  - `decision_names(decision, slot) -> tuple | None`
  - `pick_delivery(pick) -> tuple[bool | None, str]`
  - `delivery(*, decision_status, pick) -> tuple[bool | None, str, bool]`
  - `commit_status(*, selection, slot, decision, pick_slot, state) -> tuple[str, str]`
  - `history_status(selection_keys, observations) -> str`
  - `day_rows(date, *, decision, pick_rows, state, observations) -> list[dict]`. Returns ledger rows carrying `row_id, row_kind, date, slot, selection_id, reason`, the pick, decision and timeline columns, `game_eligibility = "unknown"` and the `*_obs_id` links. Task 10 fills the contest, outcome and eligibility columns.

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_rows.py`:

```python
from scripts.audit.season_ledger.rows import day_rows
from scripts.audit.season_ledger.sources.day_records import (parse_decision, parse_lineup_evolution,
                                                             parse_scheduler_state)
from scripts.audit.season_ledger.sources.pick_files import parse_archive, parse_pick_file
from tests.scripts.season_ledger.builders import cand, decision_json, evolution_jsonl, pick_json, state_json

D = "2026-08-20"
A = {"batter_id": 101, "game_pk": 5001}                    # the builders' default primary
B = {"batter_id": 202, "game_pk": 5002, "team": "NYY"}     # the builders' default double-down


def dec(**kw):
    return parse_decision(f"picks/{D}/decision.json", decision_json(D, **kw)).rows[0]


def picks(data):
    return parse_pick_file(f"picks/{D}.json", data).rows


def state(**kw):
    return parse_scheduler_state(f"picks/{D}/scheduler_state.json", state_json(D, **kw)).rows[0]


def evo(entries):
    return parse_lineup_evolution(f"picks/lineup_evolution_{D}.jsonl", evolution_jsonl(D, entries)).rows


def rows(decision=None, pick_rows=(), st=None, observations=()):
    return day_rows(D, decision=decision, pick_rows=list(pick_rows), state=st, observations=list(observations))


def test_decision_double_gives_two_committed_selections():
    out = rows(decision=dec(action="double", primary=cand(101, 5001), double_down=cand(202, 5002, team="NYY")),
               pick_rows=picks(pick_json(D, dd={}, notification_sent=True, notification_id="dm-1")))
    assert [(r["slot"], r["finalization"], r["commit_status"], r["delivery_confirmed"]) for r in out] == [
        ("primary", "decision", "committed_evidenced", True), ("double_down", "decision", "committed_evidenced", True)]
    assert out[0]["commit_basis"] == "decision:delivered;delivery:dm_notification"
    assert out[0]["lineup_position"] == 1 and out[0]["predicted_at"] == f"{D}T15:00:00.000000Z"


def test_skip_decision_with_declined_candidate_is_one_skip_day_row():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)))
    assert [(r["row_kind"], r["selection_id"], r["declined_batter_id"]) for r in out] == [("skip_day", None, 303)]


def test_skip_decision_with_commit_evidence_is_not_a_skip():
    out = rows(decision=dec(action="skip", primary=cand(303, 7001)),
               pick_rows=picks(pick_json(D, notification_sent=True, notification_id="dm-9")))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_decision_with_commit_evidence")]


def test_scheduler_skip_candidate_alone_is_unfinalized_intent():
    out = rows(st=state(final_skip_candidate={"primary": cand(303, 7001), "double": None}))
    assert [(r["row_kind"], r["reason"]) for r in out] == [("unfinalized_day", "skip_intent_only")]


def test_skip_candidate_with_commit_flag_is_never_a_skip():
    st = state(final_skip_candidate={"primary": cand(303, 7001), "double": None}, committed_pick_written=True)
    assert [(r["row_kind"], r["reason"]) for r in rows(st=st)] == [("unfinalized_day", "commit_flag_without_record")]
    (sel,) = rows(st=st, pick_rows=picks(pick_json(D)))
    assert (sel["row_kind"], sel["commit_status"], sel["commit_basis"]) == (
        "selection", "committed_evidenced", "scheduler_commit_flag")


def test_archive_only_lineup_only_and_no_evidence_days():
    arch = parse_archive(f"picks/{D}/deferred_fallback_20260820T120000-0400.json",
                         pick_json(D, deferred_fallback={"reason": "r", "deferred_at": f"{D}T12:00:00-04:00"})).rows
    assert rows(observations=arch)[0]["reason"] == "archived_candidates_only"
    assert rows(observations=evo([(A, None)]))[0]["reason"] == "lineup_evolution_only"
    assert [(r["row_kind"], r["reason"]) for r in rows()] == [("unobserved_day", "no_evidence")]


def test_decision_only_selection_has_no_pick_file_fields():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)))
    assert (r["finalization"], r["commit_status"], r["lineup_position"], r["pick_obs_id"]) == (
        "decision", "committed_evidenced", None, None)
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (True, "decision_delivered")


def test_decision_and_pick_file_naming_different_selections_is_unresolved_without_attachment():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick_rows=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099}, result="hit",
                                          slot_results={"pick": "hit"})))
    assert (r["finalization"], r["pick_view_batter_id"], r["pick_view_game_pk"]) == ("unresolved", 999, 5099)
    assert r["pick_obs_id"] is None and r["delivered_at"] is None and r["lineup_position"] is None
    assert r["p_stated"] == 0.77 and r["commit_status"] == "committed_evidenced"   # the decision's own values


def test_delivered_other_selection_makes_the_decision_selection_conflicted():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001)),
                pick_rows=picks(pick_json(D, primary={"batter_id": 999, "game_pk": 5099},
                                          notification_sent=True, notification_id="dm-2")))
    assert (r["commit_status"], r["commit_basis"]) == ("conflicted", "decision:delivered;other:delivery:dm_notification")


def test_missing_decision_with_a_delivered_pick_is_committed_via_delivery():
    (r,) = rows(pick_rows=picks(pick_json(D, notification_sent=True, notification_id="dm-7")))
    assert (r["finalization"], r["commit_status"], r["commit_basis"]) == (
        "pick_file_only", "committed_evidenced", "delivery:dm_notification")


def test_undelivered_preview_is_unconfirmed_and_a_lock_alone_proves_nothing():
    (r,) = rows(pick_rows=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["delivery_confirmed"], r["locked_at"]) == ("unconfirmed", None, None)


def test_private_commit_is_committed_without_a_delivery_claim():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick_rows=picks(pick_json(D)), st=state(pick_locked=True, pick_locked_at=f"{D}T17:00:00-04:00"))
    assert (r["commit_status"], r["commit_basis"]) == ("committed_evidenced", "decision:private_locked")
    assert (r["delivery_confirmed"], r["delivery_basis"]) == (False, "decision_private_locked")
    assert r["locked_at"] == f"{D}T21:00:00.000000Z"


def test_locked_unconfirmed_is_committed_with_unknown_delivery():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="locked_unconfirmed"),
                pick_rows=picks(pick_json(D, delivery_attempted=True)))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == (
        "committed_evidenced", None, "decision_locked_unconfirmed")


def test_legacy_public_post_is_a_delivery_signal():
    (r,) = rows(pick_rows=picks(pick_json(D, bluesky_posted=True, bluesky_uri="at://post/1")))
    assert (r["commit_status"], r["delivery_confirmed"], r["delivery_basis"]) == ("committed_evidenced", True, "public_post")
    (r2,) = rows(pick_rows=picks(pick_json(D, bluesky_posted=True)))
    assert (r2["commit_status"], r2["delivery_basis"]) == ("unconfirmed", "bluesky_posted_without_uri")


def test_private_status_with_a_dm_is_flagged_as_conflicting_evidence():
    (r,) = rows(decision=dec(action="single", primary=cand(101, 5001), delivery_status="private_locked"),
                pick_rows=picks(pick_json(D, notification_sent=True, notification_id="dm-3")))
    assert (r["delivery_confirmed"], r["delivery_basis"], r["delivery_evidence_conflict"]) == (True, "dm_notification", True)


def test_history_is_known_incomplete_only_with_evidence_and_never_complete():
    d = dec(action="single", primary=cand(101, 5001))
    assert rows(decision=d)[0]["history_status"] == "unknown"
    other_first = evo([({"batter_id": 111, "game_pk": 5011}, None), (A, None)])
    assert rows(decision=d, observations=other_first)[0]["history_status"] == "known_incomplete"
    # A failed append for an earlier selection followed by an overwrite leaves only consistent evidence.
    assert rows(decision=d, observations=evo([(A, None)]))[0]["history_status"] == "unknown"


def test_discarded_double_down_preview_marks_the_single_known_incomplete():
    out = rows(decision=dec(action="single", primary=cand(101, 5001)), pick_rows=picks(pick_json(D)),
               observations=evo([(A, B), (A, None)]))
    assert [(r["slot"], r["history_status"]) for r in out] == [("primary", "known_incomplete")]


def test_delivered_at_is_the_first_pick_file_signal_and_is_normalized():
    (r,) = rows(pick_rows=picks(pick_json(D, delivered_at=f"{D}T13:36:00-04:00", notification_sent=True,
                                          notification_id="dm")))
    assert (r["delivery_basis"], r["delivered_at"]) == ("delivered_at", f"{D}T17:36:00.000000Z")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_rows.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.rows'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/rows.py`:

```python
"""Day status, row kinds, commit / history status and the delivery predicate (spec §5, §9)."""
from __future__ import annotations

from .ids import utc_iso

SELECTION_ACTIONS = frozenset({"single", "double"})
ARCHIVE_KINDS = frozenset({"archive", "manual_archive", "repair_archive"})


def selection_id(date: str, slot: str, batter_id, game_pk) -> str:
    return f"{date}|{slot}|{batter_id}|{game_pk}"


def decision_names(decision: dict | None, slot: str) -> tuple | None:
    """The (batter_id, game_pk) a single/double decision names for `slot`, else None."""
    if decision is None or decision["action"] not in SELECTION_ACTIONS:
        return None
    if slot == "double_down" and decision["action"] != "double":
        return None
    return decision[f"{slot}_batter_id"], decision[f"{slot}_game_pk"]


def pick_delivery(pick: dict | None) -> tuple[bool | None, str]:
    """The pick-file branches of the §9 era predicate, in order. An attempt alone is not delivery."""
    if pick is None:
        return None, "no_pick_file"
    if pick["delivered_at"]:
        return True, "delivered_at"
    if pick["notification_sent"] and pick["notification_id"]:
        return True, "dm_notification"
    if pick["bluesky_posted"] and pick["bluesky_uri"]:
        return True, "public_post"
    if pick["bluesky_posted"]:
        return None, "bluesky_posted_without_uri"
    if pick["delivery_attempted"]:
        return None, "attempt_only"
    return None, "no_delivery_evidence"


def delivery(*, decision_status: str | None, pick: dict | None) -> tuple[bool | None, str, bool]:
    """§9 for one selection. `decision_status` is the delivery_status of a decision naming this
    selection (else None); `pick` the pick-file slot naming it (else None). Returns (confirmed,
    basis, private/lock evidence alongside a delivery signal) — Interpretation I4."""
    pick_ok, pick_basis = pick_delivery(pick)
    conflict = decision_status in ("private_locked", "locked_unconfirmed") and pick_ok is True
    if decision_status == "delivered":
        return True, "decision_delivered", conflict
    if pick_ok:
        return True, pick_basis, conflict
    if decision_status == "private_locked":
        return False, "decision_private_locked", conflict
    if decision_status == "locked_unconfirmed":
        return None, "decision_locked_unconfirmed", conflict
    return None, pick_basis, conflict


def commit_status(*, selection: tuple, slot: str, decision: dict | None, pick_slot: dict | None,
                  state: dict | None) -> tuple[str, str]:
    """§5 and Interpretation I3, independent of contest entry: evidence naming this selection versus
    evidence naming a different selection in the same slot. A contest match, a generic pick_locked
    flag or a delivery attempt is never commit evidence."""
    this: list[str] = []
    other: list[str] = []
    named = decision_names(decision, slot)
    if named is not None:
        (this if named == selection else other).append(f"decision:{decision['delivery_status']}")
    if pick_slot is not None:
        same = (pick_slot["batter_id"], pick_slot["game_pk"]) == selection
        ok, basis = pick_delivery(pick_slot)
        if ok:
            (this if same else other).append(f"delivery:{basis}")
        if same and decision is None and state is not None and state["committed_pick_written"]:
            this.append("scheduler_commit_flag")
    if other:
        return "conflicted", ";".join(this + [f"other:{o}" for o in other])
    if this:
        return "committed_evidenced", ";".join(this)
    return "unconfirmed", "no_commit_evidence"


def history_status(selection_keys: set, observations: list[dict]) -> str:
    """§5 and Interpretation I5: known_incomplete when an observation for the date names a (slot,
    batter, game) outside the day's canonical selections; otherwise unknown. Never `complete`."""
    for o in observations:
        if (o["slot"], o["batter_id"], o["game_pk"]) not in selection_keys:
            return "known_incomplete"
    return "unknown"


def _decision_cols(decision: dict | None) -> dict:
    if decision is None:
        return {}
    return {"action": decision["action"], "action_source_raw": decision["action_source_raw"],
            "action_source": decision["action_source"], "objective": decision["objective"],
            "degraded_reason": decision["degraded_reason"], "decision_streak": decision["streak"],
            "decision_state_source": decision["state_source"], "decision_state_status": decision["state_status"],
            "decision_obs_id": decision["obs_id"]}


def _day_row(date: str, kind: str, reason: str, *, decision=None, state=None) -> dict:
    row = {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None, "selection_id": None,
           "reason": reason, "state_obs_id": state["obs_id"] if state else None, **_decision_cols(decision)}
    if decision is not None and decision["action"] == "skip":
        row.update(declined_batter_id=decision["primary_batter_id"], declined_game_pk=decision["primary_game_pk"])
    return row


def _selection_row(date, slot, selection, *, name, team, p, decision, pick, pick_view, state, finalization,
                   history) -> dict:
    commit, basis = commit_status(selection=selection, slot=slot, decision=decision,
                                  pick_slot=pick if pick is not None else pick_view, state=state)
    named = decision_names(decision, slot) == selection
    confirmed, dbasis, conflict = delivery(
        decision_status=decision["delivery_status"] if (decision is not None and named) else None, pick=pick)
    locked = (state is not None and state["pick_locked"] and commit == "committed_evidenced"
              and finalization != "unresolved")
    sid = selection_id(date, slot, *selection)
    return {"row_id": sid, "row_kind": "selection", "date": date, "slot": slot, "selection_id": sid, "reason": None,
            "batter_id": selection[0], "batter_name": name, "team_at_pick": team, "game_pk": selection[1],
            "p_stated": p, "game_time": utc_iso(pick["game_time"]) if pick else None,
            "lineup_position": pick["lineup_position"] if pick else None,
            "projected_lineup": pick["projected_lineup"] if pick else None,
            "pitcher_id": pick["pitcher_id"] if pick else None, "finalization": finalization,
            "pick_view_batter_id": pick_view["batter_id"] if pick_view else None,
            "pick_view_game_pk": pick_view["game_pk"] if pick_view else None,
            "commit_status": commit, "commit_basis": basis, "history_status": history,
            "predicted_at": utc_iso(pick["run_time"]) if pick else None,
            "locked_at": utc_iso(state["pick_locked_at"]) if locked else None,
            "delivery_attempted": pick["delivery_attempted"] if pick else None, "delivery_attempted_at": None,
            "delivery_confirmed": confirmed, "delivery_basis": dbasis, "delivery_evidence_conflict": conflict,
            "delivered_at": utc_iso(pick["delivered_at"]) if pick else None,
            "game_eligibility": "unknown", "game_eligibility_at": None, "game_eligibility_basis": None,
            "pick_obs_id": pick["obs_id"] if pick else None,
            "pick_view_obs_id": pick_view["obs_id"] if pick_view else None,
            "state_obs_id": state["obs_id"] if state else None, **_decision_cols(decision)}


def day_rows(date: str, *, decision: dict | None, pick_rows: list[dict], state: dict | None,
             observations: list[dict]) -> list[dict]:
    """Spec §5 table. Declined skip candidates, scheduler intent, archives and lineup-evolution entries
    never become selections."""
    by_slot = {r["slot"]: r for r in pick_rows}
    evidence = [*observations, *pick_rows]
    if decision is not None and decision["action"] in SELECTION_ACTIONS:
        slots = ("primary", "double_down") if decision["action"] == "double" else ("primary",)
        chosen = [(slot, decision_names(decision, slot)) for slot in slots]
        history = history_status({(slot, *sel) for slot, sel in chosen}, evidence)
        out = []
        for slot, sel in chosen:
            pr = by_slot.get(slot)
            same = pr is not None and (pr["batter_id"], pr["game_pk"]) == sel
            out.append(_selection_row(
                date, slot, sel, name=decision[f"{slot}_batter_name"], team=decision[f"{slot}_team"],
                p=decision[f"{slot}_p_game_hit"], decision=decision, pick=pr if same else None,
                pick_view=None if same else pr, state=state,
                finalization="decision" if (same or not pick_rows) else "unresolved", history=history))
        return out
    if decision is not None:   # action == "skip"
        if decision["scoreable"] is not False:
            return [_day_row(date, "unfinalized_day", "skip_decision_unexpected_shape", decision=decision, state=state)]
        if (state is not None and state["committed_pick_written"]) or any(pick_delivery(p)[0] for p in pick_rows):
            return [_day_row(date, "unfinalized_day", "skip_decision_with_commit_evidence", decision=decision, state=state)]
        return [_day_row(date, "skip_day", "decision_skip", decision=decision, state=state)]
    if pick_rows:
        history = history_status({(p["slot"], p["batter_id"], p["game_pk"]) for p in pick_rows}, evidence)
        return [_selection_row(date, p["slot"], (p["batter_id"], p["game_pk"]), name=p["batter_name"],
                               team=p["team"], p=p["p_game_hit"], decision=None, pick=p, pick_view=None,
                               state=state, finalization="pick_file_only", history=history)
                for p in pick_rows]
    if state is not None and state["committed_pick_written"]:
        reason = "commit_flag_without_record"
    elif state is not None and state["final_skip_candidate_present"]:
        reason = "skip_intent_only"
    elif any(o["source_kind"] in ARCHIVE_KINDS for o in observations):
        reason = "archived_candidates_only"
    elif observations:
        reason = "lineup_evolution_only"
    elif state is not None:
        reason = "scheduler_ran_no_record"
    else:
        return [_day_row(date, "unobserved_day", "no_evidence")]
    return [_day_row(date, "unfinalized_day", reason, state=state)]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_rows.py -q`
Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/rows.py tests/scripts/season_ledger/test_rows.py
git commit -m "feat(ledger): day status, row kinds, commit/history status, delivery predicate (task 8)"
```

---

### Task 9: Routing, occurrence accounting, invariants, recipe rules

**Files:**
- Create: `scripts/audit/season_ledger/reconcile.py`
- Test: `tests/scripts/season_ledger/test_reconcile.py`

**Interfaces:**
- **Consumes:** Tasks 1–5 parsers.
- **Produces:**
  - `route(rel_path) -> str | None`, where `None` means excluded
  - `exclusion_reason(rel_path) -> str`
  - `PARSERS: dict[str, Callable[[str, bytes], Parsed]]`
  - `KIND_DISPOSITION`
  - `account(files, routed, parsed, dispositions) -> list[dict]`. Rows: `{source_path, locator, kind, state ∈ {emitted, excluded, quarantined, declared_missing}, reason, disposition}`.
  - `InvariantError`
  - `check_invariants(files, accounting, matches, ledger_rows, season_dates) -> None`
  - `RULES` (fixed below)
  - `evaluate_rules(files, rules, mtimes) -> tuple[list[dict], list[dict]]`, returning (per-rule summary, membership rows)
  - `recipe_labels(summary) -> dict[str, str]`

**The candidate rules are fixed here, before any real count (spec §8). They are pinned by `test_candidate_rules_are_the_predeclared_lists`.**
- **Windows:** the date in the file name (`YYYY-MM-DD`), falling back to the JSON `date`. A record enters a rule only if its top level has a `pick` object.
- **File sets:**
  - **F1** production pick files `picks/YYYY-MM-DD.json`.
  - **F2** the literal glob the 9/11 prose names, `data/picks/2026-*.json`, as the top-level `picks/2026-*.json`. It adds `.shadow.json`; `.policy_shadow.json` has no `pick` and drops out.
  - **F3** every `picks/**/*.json` with a `pick` object. It adds the scheduler archives, the streak-repair `DATE.json` versions and `backup_shadow_*`.
- **Gradings:**
  - **G1:** a primary counts when the day `result ∈ {hit, miss}`; a leg counts when the file has a double-down and the day `result ∈ {hit, miss}`.
  - **G2:** a primary counts when `slot_results.pick ∈ {hit, miss}`, or, for a file with no `slot_results` and no double-down, when the day `result ∈ {hit, miss}`; a leg counts when `slot_results.double_down ∈ {hit, miss}`.
  - **G3:** as G1, but any non-null label counts.
- **9/11 scorecard** (published 141 primaries / 82 legs; prose: "prod pick files `data/picks/2026-*.json` … primary slot, hit/miss only … DD legs"):
  - Rules S1–S8 = {F1, F2} × primary {G1, G2} × legs {G1, G2}, in that order.
  - Window 2026-03-29 → 2026-09-10; recipe date 2026-09-11.
- **9/14 naive tally** (published 191 primaries / 157 legs; no recorded recipe):
  - Rules T1–T18 = {F1, F2, F3} × grading {G1, G2, G3} (the same for both slots) × window end {2026-09-13, 2026-09-14}, in that order.
  - No lower bound; recipe date 2026-09-14.
- **Matching and labels:** a rule matches when both of its totals equal the published pair. Every matching rule is reported as `matches published totals; historical membership unverified`. Recipe labels follow Interpretation I9.

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_reconcile.py`:

```python
import pytest

from scripts.audit.season_ledger.reconcile import (PARSERS, RULES, InvariantError, account, check_invariants,
                                                   evaluate_rules, exclusion_reason, recipe_labels, route)
from tests.scripts.season_ledger.builders import pick_json


@pytest.mark.parametrize("path,kind", [
    ("picks/2026-05-01.json", "pick_file"), ("picks/2026-05-01/decision.json", "decision"),
    ("picks/2026-05-01/scheduler_state.json", "scheduler_state"),
    ("picks/2026-08-30/deferred_fallback_20260830T120000-0400.json", "archive"),
    ("picks/archive/2026-04-11.json.postponed", "manual_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json", "repair_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z_missed_2/2026-05-24.json.before", "repair_archive"),
    ("picks/lineup_evolution_2026-05-01.jsonl", "lineup_evolution"),
    ("picks/account_state/contest_ledger.jsonl", "contest_ledger"),
    ("picks/account_state/saver_transitions.jsonl", "saver_transitions"),
    ("static/rounds/20260704T030011Z.json", "rounds"), ("static/units/20260801T150000Z.json.gz", "units"),
    ("static/grab_20260927/002_players.json.gz", "players"), ("schedules/2026-05-01.json", "schedule")])
def test_routes(path, kind):
    assert route(path) == kind


@pytest.mark.parametrize("path,reason", [
    ("picks/._2026-05-01.json", "appledouble_resource_fork"), ("static/rounds/._x.json", "appledouble_resource_fork"),
    ("picks/2026-05-01.shadow.json", "shadow_model_out_of_scope"),
    ("picks/backup_shadow_2026-05-09/2026-05-08.shadow.json", "shadow_model_out_of_scope"),
    ("picks/2026-05-01.policy_shadow.json", "skip_policy_shadow_out_of_scope"),
    ("picks/slates/2026-05-01.json", "slate_binding_out_of_scope"),
    ("picks/streak.json", "state_snapshot_not_a_record"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/streak.before.json", "state_snapshot_not_a_record"),
    ("picks/account_state/contest_streak.manual.json.archived_post_auto_20260601T000000Z", "state_snapshot_not_a_record"),
    ("picks/archive_replay_restore_20260601T000000Z_post_contest_state_deploy/README.txt", "documentation"),
    ("picks/.nrestarts_checkpoint", "runtime_marker"), ("logs/cron.log", "corroboration_only"),
    ("static/grab_20260927/004_squads.json.gz", "not_used_phase1"), ("picks/mystery.bin", "unrecognized_path")])
def test_exclusions(path, reason):
    assert route(path) is None and exclusion_reason(path) == reason


def test_every_file_is_accounted_with_a_disposition():
    files = {"picks/2026-05-01.json": pick_json("2026-05-01"), "picks/._2026-05-01.json": b"\x00\x05",
             "picks/2026-05-02.json": b"{", "schedules/2026-05-02.json": None}
    routed = {"picks/2026-05-01.json": "pick_file", "picks/2026-05-02.json": "pick_file"}
    parsed = {rel: PARSERS["pick_file"](rel, files[rel]) for rel in routed}
    obs = parsed["picks/2026-05-01.json"].rows[0]["obs_id"]
    acc = account(files, routed, parsed, {obs: "canonical_selection"})
    assert {(a["source_path"], a["state"], a["reason"] or a["disposition"]) for a in acc} == {
        ("picks/2026-05-01.json", "emitted", "canonical_selection"),
        ("picks/._2026-05-01.json", "excluded", "appledouble_resource_fork"),
        ("picks/2026-05-02.json", "quarantined", "invalid_json:JSONDecodeError"),
        ("schedules/2026-05-02.json", "declared_missing", "declared_missing")}


def _acc(*pairs):
    return [{"source_path": p, "locator": loc, "state": "emitted", "disposition": "lookup"} for p, loc in pairs]


def _day(date, kind="unobserved_day"):
    return {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None}


def test_invariants_catch_omission_duplication_double_match_and_bad_days():
    files, days, good = {"a": b"1", "b": b"2"}, ["2026-05-01"], _acc(("a", "o1"), ("b", "o2"))
    check_invariants(files, good, [], [_day("2026-05-01")], days)
    with pytest.raises(InvariantError, match="not accounted"):
        check_invariants(files, good[:1], [], [_day("2026-05-01")], days)
    with pytest.raises(InvariantError, match="duplicate occurrence"):
        check_invariants(files, good + good[:1], [], [_day("2026-05-01")], days)
    with pytest.raises(InvariantError, match="without a disposition"):
        check_invariants(files, [dict(good[0], disposition=None), good[1]], [], [_day("2026-05-01")], days)
    two = [{"selection_id": "s1", "round_id": 1, "unit_id": 1, "player_id": 1, "match": "inferred"},
           {"selection_id": "s1", "round_id": 1, "unit_id": 2, "player_id": 1, "match": "inferred"}]
    with pytest.raises(InvariantError, match="more than one contest slot"):
        check_invariants(files, good, two, [_day("2026-05-01")], days)
    with pytest.raises(InvariantError, match="no ledger row"):
        check_invariants(files, good, [], [], days)
    sel = {"row_id": "2026-05-01|primary|1|2", "row_kind": "selection", "date": "2026-05-01", "slot": "primary"}
    with pytest.raises(InvariantError, match="mixes"):
        check_invariants(files, good, [], [_day("2026-05-01"), sel], days)


def _rule(files, published):
    return {"recipe": "r", "files": files, "primary": "G1", "legs": "G1", "window": ("2026-03-29", "2026-09-13"),
            "published": published, "recipe_date": "2026-09-14"}


def test_two_rules_with_identical_totals_are_both_reported():
    # F1 and F2 differ only by a shadow file without a result, so both reproduce the totals.
    files = {"picks/2026-05-01.json": pick_json("2026-05-01", result="hit"),
             "picks/2026-05-01.shadow.json": pick_json("2026-05-01"),
             "picks/2026-05-02.json": pick_json("2026-05-02", dd={}, result="miss",
                                                slot_results={"pick": "miss", "double_down": "hit"})}
    summary, membership = evaluate_rules(files, {"A": _rule("F1", (2, 1)), "B": _rule("F2", (2, 1))}, {})
    assert {s["rule_id"]: s["label"] for s in summary} == {
        "A": "matches published totals; historical membership unverified",
        "B": "matches published totals; historical membership unverified"}
    assert {m["rule_id"] for m in membership} == {"A", "B"}


def test_recipe_labels_are_hypothesis_partial_or_unrecoverable():
    summary = [{"recipe": "r1", "matched_both": True, "matched_one": True},
               {"recipe": "r1", "matched_both": False, "matched_one": False},
               {"recipe": "r2", "matched_both": False, "matched_one": True},
               {"recipe": "r3", "matched_both": False, "matched_one": False}]
    assert recipe_labels(summary) == {"r1": "hypothesis", "r2": "partial", "r3": "unrecoverable"}


def test_candidate_rules_are_the_predeclared_lists():
    scorecard = {k: v for k, v in RULES.items() if v["recipe"] == "scorecard_0911"}
    tally = {k: v for k, v in RULES.items() if v["recipe"] == "tally_0914"}
    assert sorted(scorecard) == [f"S{i}" for i in range(1, 9)] and sorted(tally, key=lambda k: int(k[1:])) == [
        f"T{i}" for i in range(1, 19)]
    assert RULES["S1"] == {"recipe": "scorecard_0911", "published": (141, 82), "recipe_date": "2026-09-11",
                           "window": ("2026-03-29", "2026-09-10"), "files": "F1", "primary": "G1", "legs": "G1"}
    assert (RULES["S8"]["files"], RULES["S8"]["primary"], RULES["S8"]["legs"]) == ("F2", "G2", "G2")
    assert RULES["T1"] == {"recipe": "tally_0914", "published": (191, 157), "recipe_date": "2026-09-14",
                           "window": ("0000-00-00", "2026-09-13"), "files": "F1", "primary": "G1", "legs": "G1"}
    assert (RULES["T18"]["files"], RULES["T18"]["primary"], RULES["T18"]["window"][1]) == ("F3", "G3", "2026-09-14")


def test_membership_records_the_evidence_interval():
    files = {f"picks/2026-05-0{i}.json": pick_json(f"2026-05-0{i}", result="hit") for i in (1, 2, 3)}
    mtimes = {"picks/2026-05-01.json": "2026-05-02T03:00:00.000000Z",    # 5/01 ET: before the 9/11 recipe
              "picks/2026-05-02.json": "2026-09-11T16:00:00.000000Z",    # same ET day: order unknown
              "picks/2026-05-03.json": "2026-09-12T16:00:00.000000Z"}    # after
    rule = {"R": {"recipe": "scorecard_0911", "files": "F1", "primary": "G1", "legs": "G1",
                  "window": ("2026-03-29", "2026-09-10"), "published": (3, 0), "recipe_date": "2026-09-11"}}
    _, membership = evaluate_rules(files, rule, mtimes)
    assert {m["source_path"]: m["modified_after_recipe_date"] for m in membership} == {
        "picks/2026-05-01.json": False, "picks/2026-05-02.json": None, "picks/2026-05-03.json": True}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_reconcile.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.reconcile'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/reconcile.py`:

```python
"""Path routing, occurrence accounting, build invariants and pre-declared recipe rules (spec §8)."""
from __future__ import annotations

import itertools
import re
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from .ids import Parsed, load_json_bytes
from .sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from .sources.day_records import parse_decision, parse_lineup_evolution, parse_scheduler_state
from .sources.pick_files import parse_archive, parse_pick_file
from .sources.static import parse_players, parse_rounds, parse_schedule, parse_units

_D = r"\d{4}-\d{2}-\d{2}"
_APPLEDOUBLE = re.compile(r"(^|/)\._[^/]*$")
ROUTES = [(re.compile(p), kind) for p, kind in [
    (rf"^picks/{_D}\.json$", "pick_file"),
    (rf"^picks/{_D}/decision\.json$", "decision"),
    (rf"^picks/{_D}/scheduler_state\.json$", "scheduler_state"),
    (rf"^picks/{_D}/(deferred_fallback|refused_delivery|stale_pick)_[^/]+\.json$", "archive"),
    (rf"^picks/archive/{_D}\.json\.postponed$", "manual_archive"),
    (rf"^picks/archive_actual_streak_repair_[^/]+/{_D}\.json(\.before)?$", "repair_archive"),
    (rf"^picks/lineup_evolution_{_D}\.jsonl$", "lineup_evolution"),
    (r"^picks/account_state/contest_ledger\.jsonl$", "contest_ledger"),
    (r"^picks/account_state/saver_transitions\.jsonl$", "saver_transitions"),
    (r"^static/rounds/[^/]+$", "rounds"),
    (r"^static/units/[^/]+$", "units"),
    (r"^static/players/[^/]+$", "players"),
    (r"^static/grab_20260927/[^/]*rounds[^/]*$", "rounds"),
    (r"^static/grab_20260927/[^/]*units[^/]*$", "units"),
    (r"^static/grab_20260927/[^/]*players[^/]*$", "players"),
    (rf"^schedules/{_D}\.json$", "schedule"),
]]
EXCLUSIONS = [(re.compile(p), reason) for p, reason in [
    (r"(^|/)\._[^/]*$", "appledouble_resource_fork"),
    (rf"^picks/{_D}\.shadow\.json$", "shadow_model_out_of_scope"),
    (r"^picks/backup_shadow_[^/]+/", "shadow_model_out_of_scope"),
    (rf"^picks/{_D}\.policy_shadow\.json$", "skip_policy_shadow_out_of_scope"),
    (r"^picks/slates/", "slate_binding_out_of_scope"),
    (r"(^|/)streak(\.before)?\.json$", "state_snapshot_not_a_record"),
    (r"^picks/account_state/(contest_streak|saver_state)[^/]*$", "state_snapshot_not_a_record"),
    (r"(^|/)README\.txt$", "documentation"),
    (r"^picks/\.nrestarts_checkpoint$", "runtime_marker"),
    (r"^logs/", "corroboration_only"),
    (r"^static/grab_20260927/[^/]*squads[^/]*$", "not_used_phase1"),
]]
PARSERS = {
    "pick_file": parse_pick_file, "archive": parse_archive,
    "manual_archive": lambda rel, data: parse_pick_file(rel, data, kind="manual_archive"),
    "repair_archive": lambda rel, data: parse_pick_file(rel, data, kind="repair_archive"),
    "decision": parse_decision, "scheduler_state": parse_scheduler_state,
    "lineup_evolution": parse_lineup_evolution, "contest_ledger": parse_contest_ledger,
    "saver_transitions": parse_saver_transitions, "rounds": parse_rounds, "players": parse_players,
    "units": parse_units, "schedule": parse_schedule,
}
KIND_DISPOSITION = {"archive": "history_evidence", "manual_archive": "history_evidence",
                    "repair_archive": "history_evidence", "lineup_evolution": "history_evidence",
                    "scheduler_state": "day_evidence", "contest_ledger": "contest_evidence",
                    "saver_transitions": "reported_attempt", "rounds": "lookup", "players": "lookup",
                    "units": "lookup", "schedule": "lookup"}   # pick_file / decision come from the ledger


def route(rel_path: str) -> str | None:
    if _APPLEDOUBLE.search(rel_path):
        return None
    for pattern, kind in ROUTES:
        if pattern.match(rel_path):
            return kind
    return None


def exclusion_reason(rel_path: str) -> str:
    for pattern, reason in EXCLUSIONS:
        if pattern.search(rel_path):
            return reason
    return "unrecognized_path"


def account(files: dict[str, bytes | None], routed: dict[str, str], parsed: dict[str, Parsed],
            dispositions: dict[str, str]) -> list[dict]:
    """Spec §8 table 1: every occurrence ends emitted (with a disposition — the anti-join),
    excluded(reason), quarantined(reason) or declared_missing."""
    out = []
    for rel, data in files.items():
        kind = routed.get(rel)
        if data is None:
            out.append({"source_path": rel, "locator": "file", "kind": None, "state": "declared_missing",
                        "reason": "declared_missing", "disposition": None})
            continue
        if kind is None:
            out.append({"source_path": rel, "locator": "file", "kind": None, "state": "excluded",
                        "reason": exclusion_reason(rel), "disposition": None})
            continue
        result = parsed[rel]
        out += [{"source_path": rel, "locator": q["locator"], "kind": kind, "state": "quarantined",
                 "reason": q["reason"], "disposition": None} for q in result.quarantined]
        out += [{"source_path": rel, "locator": r["obs_id"], "kind": kind, "state": "emitted", "reason": None,
                 "disposition": dispositions.get(r["obs_id"]) or KIND_DISPOSITION.get(kind)} for r in result.rows]
        if not result.rows and not result.quarantined:
            out.append({"source_path": rel, "locator": "file", "kind": kind, "state": "excluded",
                        "reason": "no_records", "disposition": None})
    return out


class InvariantError(Exception):
    pass


def check_invariants(files: dict, accounting: list[dict], matches: list[dict], ledger_rows: list[dict],
                     season_dates: list[str]) -> None:
    """Build failures (spec §8): an omitted, duplicated or undisposed occurrence; a contest slot identity
    twice; a selection linked to two contest slots; duplicate row ids; a malformed season day."""
    missing = sorted(set(files) - {a["source_path"] for a in accounting})
    if missing:
        raise InvariantError(f"files not accounted: {missing[:5]}")
    dupes = [k for k, n in Counter((a["source_path"], a["locator"]) for a in accounting).items() if n > 1]
    if dupes:
        raise InvariantError(f"duplicate occurrence rows: {dupes[:5]}")
    undisposed = [(a["source_path"], a["locator"]) for a in accounting
                  if a["state"] == "emitted" and not a["disposition"]]
    if undisposed:
        raise InvariantError(f"emitted occurrences without a disposition: {undisposed[:5]}")
    if any(n > 1 for n in Counter((m["round_id"], m["unit_id"], m["player_id"]) for m in matches).values()):
        raise InvariantError("a contest slot identity appears twice")
    if any(n > 1 for n in Counter(m["selection_id"] for m in matches if m["selection_id"]).values()):
        raise InvariantError("a selection is linked to more than one contest slot")
    if any(n > 1 for n in Counter(r["row_id"] for r in ledger_rows).values()):
        raise InvariantError("duplicate ledger row ids")
    by_date: dict[str, list[dict]] = {}
    for r in ledger_rows:
        if r["row_kind"] != "contest_only":
            by_date.setdefault(r["date"], []).append(r)
    for d in season_dates:
        rows = by_date.get(d, [])
        kinds = {r["row_kind"] for r in rows}
        if not rows:
            raise InvariantError(f"no ledger row for {d}")
        if "selection" in kinds and len(kinds) > 1:
            raise InvariantError(f"{d} mixes a day row with selections")
        if "selection" not in kinds and len(rows) != 1:
            raise InvariantError(f"{d} has {len(rows)} day rows")
        slots = [r["slot"] for r in rows if r["row_kind"] == "selection"]
        if len(slots) > 2 or len(slots) != len(set(slots)):
            raise InvariantError(f"{d} has malformed selection slots {slots}")


# ---- recipe rules: fixed before any real count (plan Task 9 text; pinned by a test) --------------
GRADED = frozenset({"hit", "miss"})
FILE_SETS = {"F1": re.compile(rf"^picks/{_D}\.json$"),
             "F2": re.compile(r"^picks/2026-[^/]*\.json$"),
             "F3": re.compile(r"^picks/.*\.json$")}
_SCORECARD = {"recipe": "scorecard_0911", "published": (141, 82), "recipe_date": "2026-09-11",
              "window": ("2026-03-29", "2026-09-10")}
_TALLY = {"recipe": "tally_0914", "published": (191, 157), "recipe_date": "2026-09-14"}
RULES: dict[str, dict] = {}
for _i, (_f, _p, _l) in enumerate(itertools.product(("F1", "F2"), ("G1", "G2"), ("G1", "G2")), start=1):
    RULES[f"S{_i}"] = {**_SCORECARD, "files": _f, "primary": _p, "legs": _l}
for _i, (_f, _g, _end) in enumerate(itertools.product(("F1", "F2", "F3"), ("G1", "G2", "G3"),
                                                      ("2026-09-13", "2026-09-14")), start=1):
    RULES[f"T{_i}"] = {**_TALLY, "files": _f, "primary": _g, "legs": _g, "window": ("0000-00-00", _end)}
_ET = ZoneInfo("America/New_York")
_FILE_DATE = re.compile(_D)


def _counts_as(value, grading: str) -> bool:
    return value is not None if grading == "G3" else value in GRADED


def _primary_value(doc: dict, grading: str):
    if grading in ("G1", "G3"):
        return doc.get("result")
    sr = doc.get("slot_results")
    if isinstance(sr, dict):
        return sr.get("pick")
    return doc.get("result") if doc.get("double_down") is None else None


def _leg_value(doc: dict, grading: str):
    if grading in ("G1", "G3"):
        return doc.get("result")
    sr = doc.get("slot_results")
    return sr.get("double_down") if isinstance(sr, dict) else None


def _modified_after(mtime_utc: str | None, recipe_date: str) -> bool | None:
    """Interpretation I9: the filesystem mtime (suggestive, not proof) against the recipe's ET date —
    True after, False before, None on the same ET day (order unknown) or without an mtime."""
    if mtime_utc is None:
        return None
    day = datetime.fromisoformat(mtime_utc).astimezone(_ET).date().isoformat()
    return None if day == recipe_date else day > recipe_date


def _natural(rule_id: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", rule_id)]


def evaluate_rules(files: dict[str, bytes | None], rules: dict, mtimes: dict[str, str | None]
                   ) -> tuple[list[dict], list[dict]]:
    docs: dict[str, tuple[dict, str]] = {}
    for rel, data in files.items():
        if data is None or _APPLEDOUBLE.search(rel) or not rel.startswith("picks/") or not rel.endswith(".json"):
            continue
        try:
            doc = load_json_bytes(data)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("pick"), dict):
            m = _FILE_DATE.search(rel.rsplit("/", 1)[-1])
            docs[rel] = (doc, m.group(0) if m else str(doc.get("date") or ""))
    summary, membership = [], []
    for rule_id in sorted(rules, key=_natural):
        rule = rules[rule_id]
        n = {"primary": 0, "double_down": 0}
        lo, hi = rule["window"]
        for rel in sorted(docs):
            doc, day = docs[rel]
            if not FILE_SETS[rule["files"]].match(rel) or not (lo <= day <= hi):
                continue
            values = [("primary", _primary_value(doc, rule["primary"]), rule["primary"])]
            if doc.get("double_down") is not None:
                values.append(("double_down", _leg_value(doc, rule["legs"]), rule["legs"]))
            for slot, value, grading in values:
                included = _counts_as(value, grading)
                n[slot] += included
                membership.append({"rule_id": rule_id, "recipe": rule["recipe"], "source_path": rel, "slot": slot,
                                   "file_date": day, "recipe_value": None if value is None else str(value),
                                   "included": included, "source_mtime_utc": mtimes.get(rel),
                                   "modified_after_recipe_date": _modified_after(mtimes.get(rel), rule["recipe_date"])})
        pub_p, pub_l = rule["published"]
        both = (n["primary"], n["double_down"]) == (pub_p, pub_l)
        one = n["primary"] == pub_p or n["double_down"] == pub_l
        summary.append({"rule_id": rule_id, "recipe": rule["recipe"], "files": rule["files"],
                        "primary_grading": rule["primary"], "leg_grading": rule["legs"],
                        "window": list(rule["window"]), "primaries": n["primary"], "legs": n["double_down"],
                        "published_primaries": pub_p, "published_legs": pub_l, "matched_both": both,
                        "matched_one": one,
                        "label": ("matches published totals; historical membership unverified" if both
                                  else "matches one published total" if one else "no match")})
    return summary, membership


def recipe_labels(summary: list[dict]) -> dict[str, str]:
    """Interpretation I9: hypothesis > partial > unrecoverable per recipe; `exact` would need
    independent per-record evidence, which Phase 1 does not have."""
    rank = {"unrecoverable": 0, "partial": 1, "hypothesis": 2}
    out: dict[str, str] = {}
    for s in summary:
        new = "hypothesis" if s["matched_both"] else "partial" if s["matched_one"] else "unrecoverable"
        out[s["recipe"]] = max(out.get(s["recipe"], "unrecoverable"), new, key=rank.get)
    return dict(sorted(out.items()))
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_reconcile.py -q`
Expected: `34 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/reconcile.py tests/scripts/season_ledger/test_reconcile.py
git commit -m "feat(ledger): routing, occurrence accounting, invariants, pre-declared recipe rules (task 9)"
```

---

### Task 10: Offline compile pipeline and outputs

**Files:**
- Create: `scripts/audit/season_ledger/compile.py`
- Test: `tests/scripts/season_ledger/test_compile.py`

**Interfaces:**
- **Consumes:** everything above.
- **Produces:** `SEASON_DATES` (2026-03-25 → 2026-09-27, 187 dates) and `compile_bundle(bundle_root, out_dir, *, uv_lock_sha256, code_sha=None) -> dict`. `compile_bundle` writes `season_2026_ledger.parquet`, `season_2026_ledger_occurrences.parquet`, `season_2026_ledger_contest_slots.parquet`, `season_2026_ledger_reconciliation.parquet`, `season_2026_ledger_build.json` and `season_2026_ledger_summary.md`.

- [ ] **Step 1: Write the failing tests** — `tests/scripts/season_ledger/test_compile.py`:

```python
import gzip
import json
import random
import shutil

import pyarrow.parquet as pq

from scripts.audit.season_ledger.compile import compile_bundle
from tests.scripts.season_ledger.builders import (cand, contest_line, decision_json, dumps, pick_json, rnd, seal_bundle,
                                                  slot, state_json)

ROUNDS = {971: "2026-08-20", 972: "2026-08-21", 973: "2026-08-22", 974: "2026-08-23", 975: "2026-08-24",
          976: "2026-08-25", 977: "2026-08-26"}
PLAYERS = {2513: 802415, 1300: 202, 1001: 101, 1777: 777, 1404: 404, 1505: 505, 1606: 606, 1707: 707, 1808: 808}


def _game(pk, away, home):
    return {"gamePk": pk, "status": {"codedGameState": "F", "detailedState": "Final"},
            "teams": {"away": {"team": {"abbreviation": away}}, "home": {"team": {"abbreviation": home}}}}


def _schedule(day, *games):
    return dumps({"dates": [{"date": day, "games": list(games)}]})


def _units(*units):
    return gzip.compress(dumps({"units": [{"id": u, "feedId": f, "roundId": r, "status": s} for u, f, r, s in units]}))


def _state(day):
    return state_json(day, pick_locked=True, pick_locked_at=f"{day}T17:00:00-04:00", committed_pick_written=True)


def _season_files() -> dict[str, bytes]:
    """8/20 C-03 double; 8/21 entered-but-undelivered preview; 8/22 contest-only; 8/23 Pass on a game
    postponed after lock; 8/24 game postponed before lock; 8/25 saver-absorbed miss; 8/26 partial-void DD."""
    ledger = contest_line("2026-08-27T14:30:00Z", [
        rnd(971, "hit", 8, 2, [slot(1927, 1300, "hit", number=1), slot(1928, 2513, "hit", number=2)]),
        rnd(972, "hit", 9, 1, [slot(1929, 1001, "hit")]),
        rnd(973, "hit", 10, 1, [slot(1930, 1777, "hit")]),
        rnd(974, "void", 10, 0, [slot(2404, 1404, "void", hits=0, at_bats=0)]),
        rnd(975, "void", 10, 0, [slot(2505, 1505, "void", hits=0, at_bats=0)]),
        rnd(976, "used_mulligan", 10, 0, [slot(1931, 1606, "not_hit", hits=0)]),
        rnd(977, "hit", 12, 2, [slot(1932, 1707, "void", hits=0, at_bats=0), slot(1933, 1808, "hit", number=2)])])
    return {
        "picks/2026-08-20.json": pick_json("2026-08-20", primary={"batter_id": 802415, "game_pk": 822934, "team": "TB"},
                                           dd={}, result="miss", slot_results={"pick": "miss", "double_down": "hit"},
                                           notification_sent=True, notification_id="dm-1"),
        "picks/2026-08-20/decision.json": decision_json("2026-08-20", action="double", primary=cand(802415, 822934),
                                                        double_down=cand(202, 5002, team="NYY")),
        "picks/2026-08-20/scheduler_state.json": _state("2026-08-20"),
        "picks/2026-08-21.json": pick_json("2026-08-21"),
        "picks/2026-08-23/decision.json": decision_json("2026-08-23", action="single", primary=cand(404, 6004, team="SEA")),
        "picks/2026-08-23/scheduler_state.json": _state("2026-08-23"),
        "picks/2026-08-24/decision.json": decision_json("2026-08-24", action="single", primary=cand(505, 6005, team="SEA")),
        "picks/2026-08-24/scheduler_state.json": _state("2026-08-24"),
        "picks/2026-08-25/decision.json": decision_json("2026-08-25", action="single", primary=cand(606, 6006, team="SEA")),
        "picks/2026-08-26/decision.json": decision_json("2026-08-26", action="double", primary=cand(707, 6007),
                                                        double_down=cand(808, 6008, team="NYY")),
        "picks/._2026-08-20.json": b"\x00\x05\x16\x07",
        "picks/account_state/contest_ledger.jsonl": (ledger + "\n").encode(),
        "static/rounds/20260827T120000Z.json.gz": gzip.compress(dumps({"rounds": [
            {"id": r, "date": f"{d}T08:00:00-04:00"} for r, d in ROUNDS.items()]})),
        "static/players/20260827T120000Z.json.gz": gzip.compress(dumps({"players": [
            {"id": p, "feedId": f} for p, f in PLAYERS.items()]})),
        "static/units/20260823T150000Z.json.gz": _units((2404, 6004, 974, "scheduled")),
        "static/units/20260824T020000Z.json.gz": _units((2404, 6004, 974, "postponed")),   # after the 8/23 lock
        "static/units/20260824T150000Z.json.gz": _units((2505, 6005, 975, "postponed")),   # before the 8/24 lock
        "schedules/2026-08-20.json": _schedule("2026-08-20", _game(822934, "TOR", "TB"), _game(5002, "NYY", "BAL")),
        "schedules/2026-08-21.json": _schedule("2026-08-21", _game(5001, "BOS", "TB")),
        "schedules/2026-08-22.json": _schedule("2026-08-22", _game(5777, "LAD", "SD")),
        "schedules/2026-08-25.json": _schedule("2026-08-25", _game(6006, "SEA", "HOU")),
        "schedules/2026-08-26.json": _schedule("2026-08-26", _game(6007, "TB", "CLE"), _game(6008, "NYY", "KC")),
    }


def _compile(tmp_path, name, files=None):
    bundle = tmp_path / f"bundle_{name}"
    seal_bundle(bundle, files or _season_files(), missing=["schedules/2026-08-23.json"])
    out = tmp_path / f"out_{name}"
    compile_bundle(bundle, out, uv_lock_sha256="test-lock")
    return bundle, out


def _ledger(out):
    return {r["row_id"]: r for r in pq.read_table(out / "season_2026_ledger.parquet").to_pylist()}


def test_c03_disagreement_slot_order_and_streaks(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    p, dd = led["2026-08-20|primary|802415|822934"], led["2026-08-20|double_down|202|5002"]
    assert (p["match"], p["entry_status"], p["bts_outcome"], p["local_slot_result_raw"]) == (
        "inferred", "confirmed", "hit", "miss")
    assert p["local_vs_contest_disagreement"] is True and dd["local_vs_contest_disagreement"] is False
    assert (p["slot_number"], dd["slot_number"]) == (2, 1)            # the contest's order differs from ours
    assert (p["streak_before"], p["streak_after"], p["saver_available_before"]) == (None, 8, None)
    assert p["commit_status"] == "committed_evidenced" and p["locked_at"] == "2026-08-20T21:00:00.000000Z"
    assert (p["scheduled_games"], p["round_id"]) == (2, 971)


def test_entered_preview_contest_only_and_unobserved_days(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    preview = led["2026-08-21|primary|101|5001"]
    assert (preview["commit_status"], preview["entry_status"], preview["match"]) == ("unconfirmed", "confirmed", "inferred")
    only = led["contest|973|1930|1777"]
    assert (only["row_kind"], only["match"], only["bts_outcome_status"], only["slot"]) == (
        "contest_only", "unmapped", "unmapped", None)
    assert led["day|2026-08-22|unobserved_day"]["scheduled_games"] == 1
    assert led["day|2026-03-25|unobserved_day"]["reason"] == "no_evidence"


def test_a_pass_is_an_outcome_and_eligibility_needs_evidence_before_lock(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    after, before = led["2026-08-23|primary|404|6004"], led["2026-08-24|primary|505|6005"]
    assert (after["match"], after["bts_outcome"], after["contest_norm"]) == ("evidenced", "void", "HOLD")
    assert after["game_eligibility"] == "unknown"                     # postponed only after lock
    assert (before["game_eligibility"], before["game_eligibility_at"]) == (
        "postponed_evidenced", "2026-08-24T15:00:00.000000Z")
    assert after["streak_before"] == 10 and after["scheduled_games"] is None   # 8/23 schedule declared missing


def test_round_labels_stay_on_the_round_and_legs_keep_their_own_grades(tmp_path):
    led = _ledger(_compile(tmp_path, "a")[1])
    saver = led["2026-08-25|primary|606|6006"]
    assert (saver["bts_outcome"], saver["contest_round_result"], saver["contest_norm"]) == ("not_hit", "used_mulligan", "NO_HIT")
    assert saver["saver_available_before"] is None
    p, dd = led["2026-08-26|primary|707|6007"], led["2026-08-26|double_down|808|6008"]
    assert (p["bts_outcome"], p["contest_norm"], dd["bts_outcome"]) == ("void", "HOLD", "hit")
    assert p["contest_round_result"] == dd["contest_round_result"] == "hit"


def test_every_bundle_file_is_accounted(tmp_path):
    out = _compile(tmp_path, "a")[1]
    occ = pq.read_table(out / "season_2026_ledger_occurrences.parquet").to_pylist()
    states = {(o["source_path"], o["state"]) for o in occ if o["locator"] == "file"}
    assert ("picks/._2026-08-20.json", "excluded") in states
    assert ("schedules/2026-08-23.json", "declared_missing") in states
    assert all(o["disposition"] for o in occ if o["state"] == "emitted")
    build = json.loads((out / "season_2026_ledger_build.json").read_text())
    assert build["row_kinds"]["contest_only"] == 1


def test_outputs_are_byte_identical_across_runs_roots_and_discovery_order(tmp_path):
    bundle_a, out_a = _compile(tmp_path, "a")
    items = list(_season_files().items())
    random.Random(7).shuffle(items)
    out_b = _compile(tmp_path, "b", dict(items))[1]
    moved = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(bundle_a, moved)
    out_c = tmp_path / "out_c"
    compile_bundle(moved, out_c, uv_lock_sha256="test-lock")
    names = sorted(p.name for p in out_a.iterdir())
    assert len(names) == 6
    for f in names:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes() == (out_c / f).read_bytes(), f
    manifest = moved / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()                       # a hand-reordered manifest: only its own hash may change
    manifest.write_text(json.dumps(doc))
    out_d = tmp_path / "out_d"
    compile_bundle(moved, out_d, uv_lock_sha256="test-lock")
    for f in (n for n in names if n.endswith(".parquet")):
        assert (out_a / f).read_bytes() == (out_d / f).read_bytes(), f
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_compile.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.compile'`.

- [ ] **Step 3: Write the implementation** — `scripts/audit/season_ledger/compile.py`:

```python
"""Offline compilation from a sealed bundle (spec §3–§10). Reads only the bundle; no network."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from . import BUILDER_VERSION
from .bundle import open_bundle
from .contest import (line_round_streaks, match_slot, players_lookup, resolve_duplicate_links, rounds_lookup,
                      slot_history, streak_before, team_games, unit_status_history, units_lookup)
from .ids import sha256_hex, utc_iso
from .io import write_table
from .outcomes import derived_single_result, normalize_contest, normalize_local, slot_disagreement
from .reconcile import PARSERS, RULES, account, check_invariants, evaluate_rules, recipe_labels, route
from .rows import day_rows

SEASON_DATES = [(date(2026, 3, 25) + timedelta(days=i)).isoformat() for i in range(187)]   # 3/25 → 9/27
_SEASON = frozenset(SEASON_DATES)
POSTPONED_UNIT_STATUSES = frozenset({"postponed"})
DAY_KINDS = frozenset({"pick_file", "archive", "manual_archive", "repair_archive", "decision", "scheduler_state",
                       "lineup_evolution"})
OBSERVATION_KINDS = ("archive", "manual_archive", "repair_archive", "lineup_evolution")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
S, I, B, F = pa.string(), pa.int64(), pa.bool_(), pa.float64()

LEDGER_SCHEMA = pa.schema([
    ("row_id", S), ("row_kind", S), ("date", S), ("round_id", I), ("slot", S), ("selection_id", S), ("reason", S),
    ("scheduled_games", I),
    ("batter_id", I), ("batter_name", S), ("team_at_pick", S), ("game_pk", I), ("game_time", S),
    ("lineup_position", I), ("projected_lineup", B), ("pitcher_id", I), ("p_stated", F),
    ("finalization", S), ("pick_view_batter_id", I), ("pick_view_game_pk", I),
    ("commit_status", S), ("commit_basis", S), ("history_status", S),
    ("action", S), ("action_source_raw", S), ("action_source", S), ("objective", S), ("degraded_reason", S),
    ("decision_streak", I), ("decision_state_source", S), ("decision_state_status", S),
    ("declined_batter_id", I), ("declined_game_pk", I),
    ("predicted_at", S), ("locked_at", S), ("delivery_attempted", B), ("delivery_attempted_at", S),
    ("delivery_confirmed", B), ("delivery_basis", S), ("delivery_evidence_conflict", B), ("delivered_at", S),
    ("game_eligibility", S), ("game_eligibility_at", S), ("game_eligibility_basis", S),
    ("entry_status", S), ("match", S), ("match_reason", S), ("unit_id", I), ("player_id", I), ("slot_number", I),
    ("entry_observed_at", S), ("contest_slot_grade_raw", S), ("bts_outcome", S), ("bts_outcome_status", S),
    ("contest_round_result", S), ("streak_before", I), ("streak_after", I), ("saver_available_before", B),
    ("local_slot_result_raw", S), ("local_day_result_raw", S), ("local_slot_result_derived", S),
    ("derivation_source", S), ("local_norm", S), ("contest_norm", S), ("local_vs_contest_disagreement", B),
    ("decision_obs_id", S), ("pick_obs_id", S), ("pick_view_obs_id", S), ("state_obs_id", S), ("contest_obs_id", S),
])
CONTEST_SCHEMA = pa.schema([
    ("round_id", I), ("unit_id", I), ("player_id", I), ("date", S), ("batter_id", I), ("game_pk", I),
    ("selection_id", S), ("match", S), ("match_reason", S), ("first_seen", S), ("last_seen", S),
    ("n_observations", I), ("changed", B), ("dropped_later", B), ("slot_number", I), ("slot_result", S),
    ("hits", I), ("hits_state", S), ("at_bats", I), ("at_bats_state", S), ("round_result", S),
    ("round_streak", I), ("round_streak_increase", I), ("last_obs_id", S)])
OCCURRENCE_SCHEMA = pa.schema([("source_path", S), ("locator", S), ("kind", S), ("state", S), ("reason", S),
                               ("disposition", S)])
RECONCILIATION_SCHEMA = pa.schema([("rule_id", S), ("recipe", S), ("source_path", S), ("slot", S), ("file_date", S),
                                   ("recipe_value", S), ("included", B), ("source_mtime_utc", S),
                                   ("modified_after_recipe_date", B)])


def _group(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _counts(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def _eligibility(row: dict, match: dict | None, unit_status: dict, refused: dict) -> tuple:
    """§9 and Interpretation I6: evidence dated before lock only."""
    locked = row["locked_at"]
    if match is not None and match["match"] == "evidenced" and locked:
        before = sorted(t for t, status in unit_status.get(match["unit_id"], [])
                        if status in POSTPONED_UNIT_STATUSES and t is not None and t < locked)
        if before:
            return "postponed_evidenced", before[0], "unit_capture_status_postponed"
    ref = refused.get((row["date"], row["slot"], row["batter_id"], row["game_pk"]))
    if ref is not None:
        return "refused_evidenced", utc_iso(ref["archived_at"]), ref["archive_reason"]
    return "unknown", None, None


def compile_bundle(bundle_root, out_dir, *, uv_lock_sha256: str | None, code_sha: str | None = None) -> dict:
    manifest, files = open_bundle(bundle_root)
    mtimes = {e["rel_path"]: e.get("source_mtime_utc") for e in manifest["entries"]}
    routed = {rel: kind for rel, data in files.items() if data is not None and (kind := route(rel))}
    parsed = {rel: PARSERS[kind](rel, files[rel]) for rel, kind in routed.items()}

    def rows_of(*kinds: str) -> list[dict]:
        return [r for rel, k in routed.items() if k in kinds for r in parsed[rel].rows]

    for rel, kind in routed.items():
        if kind in DAY_KINDS:
            m = _DATE.search(rel)
            for r in parsed[rel].rows:
                r["file_date"] = m.group(0) if m else None

    # §5 day rows
    decisions = {r["file_date"]: r for r in rows_of("decision")}
    states = {r["file_date"]: r for r in rows_of("scheduler_state")}
    picks_by_date = _group(rows_of("pick_file"), "file_date")
    obs_by_date = _group(rows_of(*OBSERVATION_KINDS), "file_date")
    ledger: list[dict] = []
    for d in SEASON_DATES:
        ledger += day_rows(d, decision=decisions.get(d), pick_rows=picks_by_date.get(d, []),
                           state=states.get(d), observations=obs_by_date.get(d, []))
    selections = [r for r in ledger if r["row_kind"] == "selection"]

    # §6 contest evidence and matching
    contest_rows = rows_of("contest_ledger")
    streaks_by_line = line_round_streaks(contest_rows)
    schedule_rows = rows_of("schedule")
    schedule_dates = {rel[len("schedules/"):-len(".json")] for rel, k in routed.items()
                      if k == "schedule" and not parsed[rel].quarantined}
    games_by_date: dict[str, set] = {d: set() for d in schedule_dates}
    for g in schedule_rows:
        games_by_date.setdefault(g["query_date"], set()).add(g["game_pk"])
    rounds = rounds_lookup(rows_of("rounds"))
    lookups = {"rounds": rounds, "players": players_lookup(rows_of("players")),
               "units": units_lookup(rows_of("units")), "team_games": team_games(schedule_rows),
               "schedule_dates": schedule_dates}
    matches = resolve_duplicate_links([{**h, **match_slot(h, local_selections=selections, **lookups)}
                                       for h in slot_history(contest_rows)])
    round_of_date: dict[str, list[int]] = {}
    for rid, dates in rounds.items():
        if len(dates) == 1:
            round_of_date.setdefault(next(iter(dates)), []).append(rid)

    def day_context(d: str | None) -> dict:
        ids = round_of_date.get(d, [])
        return {"round_id": ids[0] if len(ids) == 1 else None,
                "scheduled_games": len(games_by_date[d]) if d in games_by_date else None}

    for row in ledger:
        row.update(day_context(row["date"]))

    # §7 outcomes onto selections
    unit_status = unit_status_history(rows_of("units"))
    refused = {(r["file_date"], r["slot"], r["batter_id"], r["game_pk"]): r for r in rows_of("archive")
               if r["archive_prefix"] == "refused_delivery"}
    picks_by_obs = {r["obs_id"]: r for r in rows_of("pick_file")}
    linked = {m["selection_id"]: m for m in matches if m["selection_id"]}
    unlinked_keys = {(m["date"], m["batter_id"]) for m in matches
                     if not m["selection_id"] and m["match"] in ("ambiguous", "evidenced")}
    for row in selections:
        pick = picks_by_obs.get(row["pick_obs_id"])
        derived, source = derived_single_result(pick)
        row.update(local_slot_result_raw=pick["slot_result_raw"] if pick else None,
                   local_day_result_raw=pick["day_result_raw"] if pick else None,
                   local_slot_result_derived=derived, derivation_source=source, saver_available_before=None)
        row["local_norm"] = normalize_local(row["local_slot_result_raw"] or derived)
        m = linked.get(row["selection_id"])
        eligibility, at, basis = _eligibility(row, m, unit_status, refused)
        row.update(game_eligibility=eligibility, game_eligibility_at=at, game_eligibility_basis=basis)
        if m is None:
            row["entry_status"] = "unknown"
            row["bts_outcome_status"] = ("match_ambiguous" if (row["date"], row["batter_id"]) in unlinked_keys
                                         else "unknown")
            continue
        row.update(entry_status="confirmed", match=m["match"], match_reason=m["match_reason"], round_id=m["round_id"],
                   unit_id=m["unit_id"], player_id=m["player_id"], slot_number=m["slot_number"],
                   entry_observed_at=m["first_seen"], contest_slot_grade_raw=m["slot_result"],
                   bts_outcome=m["slot_result"],
                   bts_outcome_status="graded" if m["slot_result"] is not None else "matched_ungraded",
                   contest_round_result=m["round_result"], streak_after=m["round_streak"],
                   streak_before=streak_before(streaks_by_line.get(m["last_line_no"], {}), m["round_id"]),
                   contest_norm=normalize_contest(m["slot_result"]), contest_obs_id=m["last_obs_id"])
        row["local_vs_contest_disagreement"] = slot_disagreement(row["local_norm"], row["contest_norm"])

    # §5 contest-only rows
    for m in matches:
        if m["selection_id"]:
            continue
        status = {"unmapped": "unmapped", "ambiguous": "match_ambiguous"}.get(
            m["match"], "graded" if m["slot_result"] is not None else "matched_ungraded")
        ledger.append({"row_id": f"contest|{m['round_id']}|{m['unit_id']}|{m['player_id']}",
                       "row_kind": "contest_only", "date": m["date"], "slot": None, "selection_id": None,
                       **day_context(m["date"]), "round_id": m["round_id"],
                       "batter_id": m["batter_id"], "game_pk": m["game_pk"], "entry_status": "confirmed",
                       "match": m["match"], "match_reason": m["match_reason"], "unit_id": m["unit_id"],
                       "player_id": m["player_id"], "slot_number": m["slot_number"],
                       "entry_observed_at": m["first_seen"], "contest_slot_grade_raw": m["slot_result"],
                       "bts_outcome": m["slot_result"] if status == "graded" else None, "bts_outcome_status": status,
                       "contest_round_result": m["round_result"], "streak_after": m["round_streak"],
                       "contest_norm": normalize_contest(m["slot_result"]), "contest_obs_id": m["last_obs_id"]})

    # §8 accounting (with dispositions: the anti-join), invariants, recipes
    dispositions: dict[str, str] = {}
    for r in rows_of("pick_file"):
        dispositions[r["obs_id"]] = "not_selected" if r["file_date"] in _SEASON else "outside_season_window"
    for r in rows_of("decision"):
        dispositions[r["obs_id"]] = "outside_season_window"
    for r in ledger:
        if r.get("pick_obs_id"):
            dispositions[r["pick_obs_id"]] = "canonical_selection"
        if r.get("pick_view_obs_id"):
            dispositions[r["pick_view_obs_id"]] = "unresolved_pick_file_view"
        if r.get("decision_obs_id"):
            dispositions[r["decision_obs_id"]] = "canonical_decision"
    accounting = account(files, routed, parsed, dispositions)
    check_invariants(files, accounting, matches, ledger, SEASON_DATES)
    summary, membership = evaluate_rules(files, RULES, mtimes)

    out = Path(out_dir)
    write_table(ledger, LEDGER_SCHEMA, out / "season_2026_ledger.parquet", sort_keys=["row_id"])
    write_table(accounting, OCCURRENCE_SCHEMA, out / "season_2026_ledger_occurrences.parquet",
                sort_keys=["source_path", "locator"])
    write_table([{k: m.get(k) for k in CONTEST_SCHEMA.names} for m in matches], CONTEST_SCHEMA,
                out / "season_2026_ledger_contest_slots.parquet", sort_keys=["round_id", "unit_id", "player_id"])
    write_table(membership, RECONCILIATION_SCHEMA, out / "season_2026_ledger_reconciliation.parquet",
                sort_keys=["rule_id", "source_path", "slot"])
    sels = [r for r in ledger if r["row_kind"] == "selection"]
    saver = rows_of("saver_transitions")
    build = {"builder_version": BUILDER_VERSION, "code_sha": code_sha,
             "bundle_manifest_sha256": sha256_hex((Path(bundle_root) / "manifest.json").read_bytes()),
             "uv_lock_sha256": uv_lock_sha256, "season_dates": [SEASON_DATES[0], SEASON_DATES[-1]],
             "row_kinds": _counts(ledger, "row_kind"), "finalization": _counts(sels, "finalization"),
             "commit_status": _counts(sels, "commit_status"), "history_status": _counts(sels, "history_status"),
             "entry_status": _counts(sels, "entry_status"),
             "bts_outcome_status": _counts([r for r in ledger if r["row_kind"] in ("selection", "contest_only")],
                                           "bts_outcome_status"),
             "match": _counts(matches, "match"), "match_reason": _counts(matches, "match_reason"),
             "game_eligibility": _counts(sels, "game_eligibility"),
             "local_vs_contest_disagreement": _counts(sels, "local_vs_contest_disagreement"),
             "occurrence_states": _counts(accounting, "state"),
             "exclusion_reasons": _counts([a for a in accounting if a["state"] == "excluded"], "reason"),
             "quarantine_reasons": _counts([a for a in accounting if a["state"] == "quarantined"], "reason"),
             "dispositions": _counts([a for a in accounting if a["state"] == "emitted"], "disposition"),
             "saver_transition_attempts": {"n": len(saver), "by_outcome": _counts(saver, "attempt_outcome")},
             "recipes": summary, "recipe_labels": recipe_labels(summary)}
    (out / "season_2026_ledger_build.json").write_text(json.dumps(build, indent=1, sort_keys=True) + "\n")
    (out / "season_2026_ledger_summary.md").write_text(_summary_md(build))
    return build


def _summary_md(build: dict) -> str:
    lines = ["# Season 2026 ledger — Phase 1 build summary", "",
             f"Builder `{build['builder_version']}` · code `{build['code_sha']}` · bundle manifest "
             f"`{build['bundle_manifest_sha256']}` · uv.lock `{build['uv_lock_sha256']}`", ""]
    for key in ("row_kinds", "finalization", "commit_status", "history_status", "entry_status", "bts_outcome_status",
                "match", "match_reason", "game_eligibility", "local_vs_contest_disagreement", "occurrence_states",
                "exclusion_reasons", "quarantine_reasons", "dispositions"):
        lines += [f"## {key}", *(f"- {k}: {v}" for k, v in build[key].items()), ""]
    s = build["saver_transition_attempts"]
    lines += ["## saver_transitions.jsonl (attempts, not consumption times)", f"- rows: {s['n']}",
              *(f"- {k}: {v}" for k, v in s["by_outcome"].items()), "",
              "## Recipe hypotheses (historical membership unverified)", "",
              "| rule | recipe | files | primary | legs | window | primaries | legs | published | label |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule_id']} | {r['recipe']} | {r['files']} | {r['primary_grading']} | {r['leg_grading']} | "
              f"{r['window'][0]}→{r['window'][1]} | {r['primaries']} | {r['legs']} | "
              f"{r['published_primaries']}/{r['published_legs']} | {r['label']} |" for r in build["recipes"]]
    lines += ["", *(f"- **{k}: {v}**" for k, v in build["recipe_labels"].items()), ""]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_compile.py -q`
Expected: `6 passed`. If the determinism test fails, find the unordered value (a set iteration or dict order reaching an output) and sort it; do not weaken the test.

- [ ] **Step 5: Run the whole ledger suite**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger -q`
Expected: `111 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/season_ledger/compile.py tests/scripts/season_ledger/test_compile.py
git commit -m "feat(ledger): offline compile pipeline with deterministic outputs (task 10)"
```

---

### Task 11: Acquisition and CLI

**Files:**
- Create: `scripts/audit/season_ledger/acquire.py`, `scripts/audit/build_season_ledger.py`
- Test: `tests/scripts/season_ledger/test_acquire.py`

**Interfaces:**
- **Produces:** `acquire(*, snapshot_root, out_root, dates, fetch, now_utc) -> Path`, returning the manifest path. The CLI is `python scripts/audit/build_season_ledger.py acquire|compile …`.
- **Acquisition sources**, all read from the snapshot root:
  - the whole `data/picks/` tree
  - every `static_snapshots/{rounds,units}` capture
  - the first and last `static_snapshots/players` capture (Interpretation I10)
  - the whole `final_grab_20260927/raw/static/`
  - `cron.log` and `journal_bts-scheduler_retained.txt`
  - one schedule response per date
- **Missing inputs:** an expected input that is absent becomes a declared `missing` entry. Dotfile capture markers are not inputs.

- [ ] **Step 1: Write the failing test** — `tests/scripts/season_ledger/test_acquire.py`:

```python
import gzip

import pytest

from scripts.audit.season_ledger.acquire import acquire
from scripts.audit.season_ledger.bundle import open_bundle


def test_acquire_copies_sources_declares_missing_inputs_and_seals(tmp_path):
    snap = tmp_path / "final-20260928"
    picks = snap / "data" / "picks"
    (picks / "2026-05-01").mkdir(parents=True)
    (picks / "2026-05-01.json").write_bytes(b'{"pick": {}}')
    (picks / "2026-05-01" / "decision.json").write_bytes(b"{}")
    static = snap / "data" / "leaderboard" / "static_snapshots"
    for feed in ("rounds", "units", "players"):
        (static / feed).mkdir(parents=True)
        (static / feed / ".last_sha256").write_text("x\n")
    (static / "rounds" / "20260704T030011Z.json.gz").write_bytes(gzip.compress(b'{"rounds": []}'))
    for stamp in ("20260704T030011Z", "20260801T030011Z", "20260928T123001Z"):
        (static / "players" / f"{stamp}.json.gz").write_bytes(gzip.compress(b'{"players": []}'))
    grab = snap / "data" / "leaderboard" / "final_grab_20260927" / "raw" / "static"
    grab.mkdir(parents=True)
    (grab / "002_players.json.gz").write_bytes(gzip.compress(b'{"players": []}'))
    (snap / "cron.log").write_text("log\n")

    def fetch(day):
        if day == "2026-05-02":
            raise OSError("timeout")
        return b'{"dates": []}'

    out = tmp_path / "bundle"
    acquire(snapshot_root=snap, out_root=out, dates=["2026-05-01", "2026-05-02"], fetch=fetch,
            now_utc=lambda: "2026-09-28T16:00:00.000000Z")
    manifest, files = open_bundle(out)
    assert files["picks/2026-05-01.json"] == b'{"pick": {}}'
    assert sorted(k for k in files if k.startswith("static/players/")) == [
        "static/players/20260704T030011Z.json.gz", "static/players/20260928T123001Z.json.gz"]
    assert files["static/units/NO_CAPTURES"] is None and "static/rounds/.last_sha256" not in files
    assert files["static/grab_20260927/002_players.json.gz"].startswith(b"\x1f\x8b")
    assert files["logs/cron.log"] == b"log\n" and files["logs/journal_bts-scheduler_retained.txt"] is None
    assert files["schedules/2026-05-01.json"] == b'{"dates": []}' and files["schedules/2026-05-02.json"] is None
    entries = {e["rel_path"]: e for e in manifest["entries"]}
    assert entries["schedules/2026-05-02.json"]["note"] == "fetch_failed:OSError"
    assert entries["picks/2026-05-01.json"]["source_path"] == "data/picks/2026-05-01.json"
    assert entries["picks/2026-05-01.json"]["source_mtime_utc"].endswith("Z")
    with pytest.raises(FileExistsError):
        acquire(snapshot_root=snap, out_root=out, dates=[], fetch=fetch, now_utc=lambda: "x")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger/test_acquire.py -q`
Expected: collection ERROR — `No module named 'scripts.audit.season_ledger.acquire'`.

- [ ] **Step 3: Write the implementation**

`scripts/audit/season_ledger/acquire.py`:
```python
"""Acquisition into a sealed evidence bundle (spec §3). Reads the frozen W0.7 snapshot; the only
network call is the MLB schedule fetch, whose failures become declared `missing` entries."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import BUILDER_VERSION
from .bundle import BundleEntry, write_manifest
from .ids import UTC_FORMAT, sha256_hex

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&gameType=R&hydrate=team"
STATIC = Path("data/leaderboard/static_snapshots")
GRAB_STATIC = Path("data/leaderboard/final_grab_20260927/raw/static")
LOGS = ("cron.log", "journal_bts-scheduler_retained.txt")


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(UTC_FORMAT)


def _captures(directory: Path) -> list[Path]:
    """Files in a capture directory; dotfiles (.last_sha256 markers) are not inputs."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def acquire(*, snapshot_root, out_root, dates: list[str], fetch: Callable[[str], bytes],
            now_utc: Callable[[], str]) -> Path:
    snap, out_root = Path(snapshot_root), Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"bundle directory not empty: {out_root} (a new acquisition is a new version)")
    picks = snap / "data" / "picks"
    if not picks.is_dir():
        raise FileNotFoundError(f"no picks tree under {snap}")
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[BundleEntry] = []

    def copy(src: Path, rel: str) -> None:
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        data = dest.read_bytes()
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=src.relative_to(snap).as_posix(), source_mtime_utc=_mtime_utc(src)))

    def missing(rel: str, note: str) -> None:
        entries.append(BundleEntry(rel_path=rel, status="missing", note=note))

    for src in sorted(p for p in picks.rglob("*") if p.is_file()):
        copy(src, "picks/" + src.relative_to(picks).as_posix())
    for feed in ("rounds", "units"):
        found = _captures(snap / STATIC / feed)
        for src in found:
            copy(src, f"static/{feed}/{src.name}")
        if not found:
            missing(f"static/{feed}/NO_CAPTURES", "no captures found")
    players = _captures(snap / STATIC / "players")
    for src in sorted({players[0], players[-1]}) if players else []:
        copy(src, f"static/players/{src.name}")
    if not players:
        missing("static/players/NO_CAPTURES", "no captures found")
    grab = _captures(snap / GRAB_STATIC)
    for src in grab:
        copy(src, f"static/grab_20260927/{src.name}")
    if not grab:
        missing("static/grab_20260927/NO_FILES", "grab static directory empty or absent")
    for name in LOGS:
        if (snap / name).is_file():
            copy(snap / name, f"logs/{name}")
        else:
            missing(f"logs/{name}", "not in snapshot")
    for day in dates:
        rel = f"schedules/{day}.json"
        try:
            data = fetch(day)
        except Exception as exc:   # recorded, never fatal: the bundle declares the gap
            missing(rel, f"fetch_failed:{exc.__class__.__name__}")
            continue
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=SCHEDULE_URL.format(date=day), source_mtime_utc=now_utc()))
    return write_manifest(out_root, entries, acquired_at_utc=now_utc(), builder_version=BUILDER_VERSION,
                          source_root=str(snap))
```

`scripts/audit/build_season_ledger.py`:
```python
"""Season 2026 ledger, Phase 1 — `acquire` (box: reads the frozen snapshot, fetches MLB schedules)
and `compile` (anywhere, offline, from the sealed bundle).

  .venv/bin/python scripts/audit/build_season_ledger.py acquire \
      --snapshot data/hetzner_results/season_2026_snapshot/final-20260928 \
      --out data/hetzner_results/season_2026_ledger_evidence/v1
  .venv/bin/python scripts/audit/build_season_ledger.py compile \
      --bundle data/hetzner_results/season_2026_ledger_evidence/v1 --out data/validation --code-sha <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audit.season_ledger.acquire import SCHEDULE_URL, acquire  # noqa: E402
from scripts.audit.season_ledger.compile import SEASON_DATES, compile_bundle  # noqa: E402
from scripts.audit.season_ledger.ids import UTC_FORMAT  # noqa: E402

USER_AGENT = "bts-season-ledger/1 (one-pass audit acquisition)"


def _fetch(day: str) -> bytes:
    req = urllib.request.Request(SCHEDULE_URL.format(date=day), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    time.sleep(0.5)   # courteous pacing to a public API
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Season 2026 ledger, Phase 1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compile")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--code-sha", default=None)
    c.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    args = ap.parse_args(argv)
    if args.cmd == "acquire":
        path = acquire(snapshot_root=args.snapshot, out_root=args.out, dates=SEASON_DATES, fetch=_fetch,
                       now_utc=lambda: datetime.now(timezone.utc).strftime(UTC_FORMAT))
        print(f"sealed {path}")
        return 0
    lock_sha = hashlib.sha256(args.uv_lock.read_bytes()).hexdigest() if args.uv_lock.is_file() else None
    build = compile_bundle(args.bundle, args.out, uv_lock_sha256=lock_sha, code_sha=args.code_sha)
    print(f"row kinds {build['row_kinds']} | matches {build['match']} | recipes {build['recipe_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass, then the suites**

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/scripts/season_ledger -q`
Expected: `112 passed`.

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q`
Expected: the previous fast-suite count + 112, all passing. Report any failure by name.

Run: `cd /Users/eric/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit/build_season_ledger.py compile --help`
Expected: usage text listing `--bundle`, `--out`, `--code-sha`, `--uv-lock`.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/season_ledger/acquire.py scripts/audit/build_season_ledger.py tests/scripts/season_ledger/test_acquire.py
git commit -m "feat(ledger): acquisition into a sealed bundle and the build CLI (task 11)"
```

---

### Task 12: Codex code review (no data access)

**Files:** `.codex-review/season-ledger/prompt-code-rN.md` (gitignored); each round is archived to `docs/audit/<date>-season-ledger-codex-code-rN.md`.

- [ ] **Step 1:** Write the review prompt:
  - **Stance:** adversarial ("assume defects exist").
  - **Scope:** the diff of Tasks 1–11 against spec v4, this plan's Interpretations I1–I12 and Review Focus.
  - **Hard rule, no data access:** nothing under `data/`, no snapshots, no ssh, no network.
  - **Focus:**
    - false greens: tests that hand-supply fields the real parsers never produce, and assertions that can only pass
    - determinism
    - the §5 table
    - matching
    - accounting and dispositions
    - the pre-declared recipe rules
  - **Deliverable:** a file with findings (BLOCKER/SHOULD/NIT) plus a SIGN/BLOCK verdict. Codex's final output is exactly `DONE`.
- [ ] **Step 2:** Send it through the herdr round-trip (consulting-codex skill §Transport; reuse the bound `bts-codex` pane after verifying its recipient tuple). Monitor the wait and validate the deliverable.
- [ ] **Step 3:** Triage each finding as real, false flag or over-engineered, with evidence. Fix real ones test-first, re-run `tests/scripts/season_ledger`, and commit each fix with its finding number in the message.
- [ ] **Step 4:** Re-review until SIGN, capped at 3 rounds, then surface any unresolved disagreement to Eric. Archive every round under `docs/audit/` and commit.

---

### Task 13: Real run on the box, records

- [ ] **Step 1: Ship the reviewed code to the box without deploying:**
```bash
cd /Users/eric/projects/bts && SHA=$(git rev-parse HEAD) && \
git archive "$SHA" scripts/__init__.py scripts/audit/__init__.py scripts/audit/season_ledger scripts/audit/build_season_ledger.py \
  | ssh bts-hetzner "rm -rf /tmp/ledger_code && mkdir -p /tmp/ledger_code && tar -x -C /tmp/ledger_code && echo $SHA > /tmp/ledger_code/CODE_SHA" && echo "shipped $SHA"
```
- [ ] **Step 2: Acquire** as a transient user unit, which survives an SSH drop:
```bash
ssh bts-hetzner 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemd-run --user --unit=bts-ledger-acquire --collect \
  --working-directory=/home/bts/projects/bts \
  -p StandardOutput=append:/home/bts/logs/ledger_acquire.log -p StandardError=append:/home/bts/logs/ledger_acquire.log \
  /home/bts/projects/bts/.venv/bin/python /tmp/ledger_code/scripts/audit/build_season_ledger.py acquire \
  --snapshot data/hetzner_results/season_2026_snapshot/final-20260928 \
  --out data/hetzner_results/season_2026_ledger_evidence/v1'
```
Wait with a Monitor on the log until `sealed` or a traceback appears. Then print the manifest's entry counts and every `missing` entry (path and note only):
```bash
ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python -c "
import json, collections
m = json.load(open(\"data/hetzner_results/season_2026_ledger_evidence/v1/manifest.json\"))
print(len(m[\"entries\"]), collections.Counter(e[\"rel_path\"].split(\"/\")[0] for e in m[\"entries\"]))
print([(e[\"rel_path\"], e[\"note\"]) for e in m[\"entries\"] if e[\"status\"] == \"missing\"])"'
```
Expected: about 3,600 entries (picks tree, 173 rounds, 2,335 units, 2 players, 4 grab files, 2 logs, 187 schedules); `missing` should list at most schedule fetch failures.
- [ ] **Step 3: Compile twice** and compare bytes:
```bash
ssh bts-hetzner 'cd ~/projects/bts && SHA=$(cat /tmp/ledger_code/CODE_SHA) && for OUT in data/validation /tmp/ledger_check; do \
  .venv/bin/python /tmp/ledger_code/scripts/audit/build_season_ledger.py compile \
  --bundle data/hetzner_results/season_2026_ledger_evidence/v1 --out $OUT --code-sha $SHA || exit 1; done && \
  for f in /tmp/ledger_check/season_2026_ledger*; do cmp "$f" "data/validation/$(basename $f)" || echo "DIFF $f"; done; echo compared'
```
Expected: the CLI summary line printed twice, no `DIFF` lines, then `compared`. An `InvariantError` is a stop: diagnose it with the systematic-debugging skill, fix test-first on the Mac, and re-run from Step 1.
- [ ] **Step 4: Back up** the bundle now, rather than waiting for the 04:50 cron:
```bash
ssh bts-hetzner 'cd ~/projects/bts && set -a && . ./.env && set +a && ~/.local/bin/uv run bts backup run --set archive 2>&1 | tail -5'
```
Record the snapshot id.
- [ ] **Step 5: Record.** Copy `season_2026_ledger_build.json` and `season_2026_ledger_summary.md` to the Mac. Then:
  - Write `docs/audit/<date>-season-ledger.md`, covering:
    - the code sha, bundle manifest sha, `uv.lock` sha and restic snapshot
    - input coverage and `missing` entries
    - row kinds, finalization, commit, history, entry and outcome-status counts
    - match counts and reasons (including `player_unknown` and `selection_game_pk_unrecorded`)
    - occurrence states, exclusion and quarantine reasons, and dispositions
    - the `local_vs_contest_disagreement` count (the C-03 check)
    - the recipe table and labels
    - known limits: Phase 1 scope, saver unknown, eligibility only from July unit captures, the §12 local-grader finding
  - Add exposure-register row **X-19**: the ledger build computed coverage and accounting counts, recipe totals and the disagreement count; no rates. Any analysis of the ledger is a new read.
  - Update the W1.1 row in `docs/audit/2026-season-wrap-index.md`.
  - Commit.
- [ ] **Step 6: Optional result review.** If anything in Step 5 looks surprising, run a Codex round on the produced outputs; data access is allowed for this one.

---

## Self-Review (done while writing)
- **Spec coverage:**
  - §3 evidence → Tasks 1 and 11.
  - §4 O1–O7 → Tasks 2–5.
  - §5 row kinds, commit and history → Task 8.
  - §6 qualification, history, matching, entry, round semantics and saver null → Tasks 4, 6 and 10.
  - §7 outcomes → Tasks 7 and 10.
  - §8 accounting and recipes → Tasks 9 and 10.
  - §9 field contract (`LEDGER_SCHEMA`) → Tasks 8 and 10.
  - §10 outputs → Task 10.
  - §11 fixtures → Tasks 2–10: every listed fixture has a test, with the synthetic files run through the real parsers.
  - §12 → not built.
  - Determinism → Tasks 1, 10 and 13.
- **Placeholders:** none; every code step carries its code.
- **Type consistency:**
  - `selection_id` is built only by `rows.selection_id` and reaches `match_slot` through `local_selections`.
  - The parsed-row keys used in `rows.py` and `compile.py` match their producers in Tasks 2–5.
  - Every key a ledger row can carry is in `LEDGER_SCHEMA` (`write_table` refuses unknown columns).
- **Review Focus:** each of the five lines has its test in the named task.
