# Audit teardown safety — design

**Date**: 2026-04-24
**Author**: Eric + Claude (brainstorm)
**Scope**: `scripts/audit_driver.py`, `scripts/audit_attach.py`, `tests/test_audit_driver.py`

## Problem

The audit drivers currently tear down their provisioned cloud boxes unconditionally in a `finally:` block. `retrieve_one` (the rsync helper) logs per-box errors but does not raise, so a partial retrieve silently precedes teardown. Result: data loss whenever rsync fails for any reason.

The unconditional-teardown pattern was added on 2026-04-14 to fix the reverse bug (silent retrieve failure → no teardown → zombie boxes costing money). That fix overcorrected. We now need to thread the needle: clean boxes tear down, dirty boxes are preserved.

Memory: `project_bts_2026_04_23_phase_b_heartbeat.md#open-follow-up-audit_attach-safety`.

## Decisions (from brainstorming)

| # | Question | Choice |
|---|---|---|
| 1 | Default policy | **A** — teardown by default, gated on retrieve success (not explicit opt-in) |
| 2 | Gate granularity | **A** — per-box skip (not fleet-level all-or-nothing) |
| 3 | Scope | **A** — fix both `audit_driver.py` and `audit_attach.py` in this change |
| 4 | Implementation structure | **Approach 1** — extract a `teardown_retrieved` helper; keep `teardown_all` unchanged |
| 5 | Exit code on partial | Non-zero (exit 1) on any box skipped; `raise SystemExit(1)` after `finally:` |

## Architecture

One new pure helper in `scripts/audit_driver.py`, imported by `audit_attach.py`. Both main loops stay structurally identical to today; just the `finally:` block body changes.

```
audit_driver.py
├── retrieve_one(box, out_root, seeds) -> (name, status, errs)       # unchanged
├── teardown_all(provider, boxes) -> int                             # changed — now returns count of successful deletes
├── teardown_retrieved(provider, boxes, retrieve_results)            # NEW
│     -> tuple[int, int]  (selected, deleted)
│     ├── filters boxes where retrieve_results[box.name] == "ok"
│     ├── logs unrecognized keys in retrieve_results (caller-bug signal)
│     ├── logs preserved boxes with name + ip + status
│     ├── calls teardown_all on the clean subset
│     └── returns (len(candidates), count_of_successful_deletes)
└── main() — finally: calls teardown_retrieved                       # changed

audit_attach.py
├── imports teardown_retrieved from audit_driver
└── main() — finally: calls teardown_retrieved                       # changed
```

## Component contract

```python
def teardown_retrieved(
    provider: Provider,
    boxes: list[Box],
    retrieve_results: dict[str, str],
) -> tuple[int, int]:
    """Tear down only the boxes whose retrieve_results[box.name] == "ok".

    Preserved boxes are logged with name + ipv4 + status for manual recovery.
    Unrecognized keys in retrieve_results (names that don't match any box in
    `boxes`) are also logged — signals a caller bug, not a safety concern.

    Returns (selected, deleted):
      - selected: count of boxes where retrieve_results[name] == "ok"
                  (i.e., passed the data-integrity gate; handed to teardown_all)
      - deleted:  count of boxes whose provider.delete() call didn't raise
                  (inherited from teardown_all, which logs and continues on
                  per-box API failures; `selected - deleted` = API-failed count)

    Callers use `skipped_for_safety = len(boxes) - selected` to know how many
    boxes were preserved for data-integrity reasons. This is the signal that
    drives the non-zero exit code.

    Any box name missing from retrieve_results is treated as "not-attempted"
    and preserved. Default-to-preserve makes the helper safe to call even if
    the retrieve loop was interrupted partway through.
    """
```

**Behavior matrix:**

| `retrieve_results[box.name]` | Action |
|---|---|
| `"ok"` | Tear down |
| `"partial"` | Preserve; log with `PRESERVED` prefix with name + ip + status |
| missing | Preserve; log with `PRESERVED` prefix with status=`"not-attempted"` |
| any other value (`None`, `True`, unknown string) | Preserve (future-proof: unknown = don't destroy) |

**Additional log lines** (not per-box):
- Any keys in `retrieve_results` that don't match a box name in `boxes` → single log line `unrecognized key in retrieve_results: <name> — caller bug?` (drains once at helper entry, not per-box)
- Final teardown summary (from call site, after helper returns): `=== TEARDOWN: selected=N/M deleted=D preserved=P api_failed=F ===`

**Call-site pattern (identical in both main functions):**

```python
retrieve_results: dict[str, str] = {}  # before try:
exit_code = 0
try:
    # ... poll loop ...
    # ... retrieve loop populates retrieve_results[nm] = status ...
finally:
    if args.no_teardown:
        log("--no-teardown set; leaving all boxes alive")
    elif not boxes:
        pass
    else:
        selected, deleted = teardown_retrieved(provider, boxes, retrieve_results)
        preserved = len(boxes) - selected      # held back for data-integrity
        api_failed = selected - deleted         # picked for teardown but delete() raised
        log(f"=== TEARDOWN: selected={selected}/{len(boxes)} "
            f"deleted={deleted} preserved={preserved} api_failed={api_failed} ===")
        if preserved > 0:
            exit_code = 1   # data-integrity signal — the primary concern

if exit_code:
    raise SystemExit(exit_code)
```

`api_failed` is surfaced in the log but does NOT trigger exit 1 — it's a
provider/transient concern (Hetzner 500, rate limit), not data-integrity.
Operator sees it in the log line and can manually retry `provider.delete()`.

## Data flow — the four cases

**Happy path (all boxes retrieve cleanly):**
`retrieve_results = {b1: ok, b2: ok, ...}` → all filtered in → `teardown_all` deletes all → exit 0. Identical to today's behavior.

**Partial failure (one box rsync fails):**
`retrieve_results[b7] = "partial"`, rest `"ok"` → 25 boxes torn down, b7 preserved with log:
`PRESERVED b7 ip=80.240.23.160 retrieve_status=partial — manually re-retrieve + teardown` → exit 1. Cost: ~$0.08/hr for one zombie.

**Systemic failure (2026-04-14 replay — all retrieves return partial):**
Every box → `"partial"` → zero deletions → 26 `PRESERVED` log lines → exit 1. Cost: ~$3.40/hr Vultr burn until operator notices. Accepted tradeoff vs destroying 26 seeds of compute work.

**Pre-retrieve exception (deadline hit, then OOM/KeyboardInterrupt before retrieve starts):**
`retrieve_results = {}` → all boxes "not-attempted" → zero deletions → 26 log lines → exit 1.

**Key invariant**: a box is only torn down if its retrieve _completed successfully_. Any path that doesn't produce `retrieve_results[box.name] == "ok"` — exception, crash, partial, skipped — defaults to preservation. Destruction requires explicit evidence of success.

## Error handling

- **`provider.delete()` failures**: inherited from current `teardown_all` (try/except per box, log and continue). No change.
- **Malformed `retrieve_results` values**: anything `!= "ok"` preserves. `None`, booleans, unknown strings all resolve to preserve.
- **`retrieve_results=None` passed to helper**: explicit `if retrieve_results is None: raise TypeError(...)` guard at top of helper — gives a clear error message vs the cryptic `AttributeError: 'NoneType' object has no attribute 'get'` you'd otherwise get.
- **KeyboardInterrupt during retrieve loop**: partially populated dict → finally runs → completed-"ok" boxes torn down, rest preserved. Same invariant as systemic-failure case.
- **KeyboardInterrupt during `teardown_retrieved`**: some boxes already torn down, rest preserved. Python's default propagation takes us to SystemExit. No new handling.

## Testing

**TDD flow**: tests first (red), implementation (green), then wire call sites.

**Test file**: extend `tests/test_audit_driver.py` with a new `TestTeardownRetrieved` class. Existing `TestKeychainFallback` untouched.

**Fixtures:**
```python
class FakeProvider:
    """Captures delete() calls instead of hitting a real API."""
    name = "fake"
    def __init__(self):
        self.deleted = []
    def delete(self, box_id: str) -> None:
        self.deleted.append(box_id)

@pytest.fixture
def boxes():
    return [
        Box(id="1", name="b1", ipv4="10.0.0.1", region=""),
        Box(id="2", name="b2", ipv4="10.0.0.2", region=""),
        Box(id="3", name="b3", ipv4="10.0.0.3", region=""),
    ]
```

**Test cases (11):**

| # | Scenario | Assert |
|---|---|---|
| 1 | All `"ok"` | `fake.deleted == ["1","2","3"]`; returns `(3, 3)` |
| 2 | One `"partial"` | `fake.deleted == ["1","3"]`; returns `(2, 2)`; partial box logged with ip + status |
| 3 | All `"partial"` | `fake.deleted == []`; returns `(0, 0)`; 3 log lines with `PRESERVED` prefix |
| 4 | Missing key | Missing box preserved; logged as `"not-attempted"`; returns `(N-1, N-1)` |
| 5 | Empty `retrieve_results` | All preserved; returns `(0, 0)` |
| 6 | Empty `boxes` list | no-op; returns `(0, 0)` |
| 7 | Malformed value (`None`, `True`, unknown string) | All preserved; returns `(0, 0)` |
| 8 | `provider.delete()` raises on one of 3 selected boxes | Other 2 still torn down (teardown_all swallows per-box); returns `(3, 2)` — `selected=3, deleted=2` |
| 9 | `retrieve_results=None` | `TypeError` raised (explicit guard in helper) |
| 10 | Stray key in `retrieve_results` (name doesn't match any box) | Logged with `"unrecognized key"` prefix; no crash; returns reflect boxes only |
| 11 | `teardown_all` invoked standalone (not via helper) | Returns an int = count of successful deletes (previously returned None) |

Test #11 verifies the small contract change to `teardown_all` itself.

**Log-content tests**: use `monkeypatch.setattr` to replace `audit_driver.log` with a list-append fake, then assert the captured lines contain the box name + ip + status.

**Integration testing**: not adding. The call-site diff is ~3 lines per main (init dict, populate during loop, swap finally body). Unit tests cover the behavior; next real audit is the integration signal.

**Full suite sanity**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest` expects 601 + 11 = **612 passing** after the change.

## Out of scope (explicitly deferred)

- Retry budget / exponential backoff for `retrieve_one` (option C from brainstorm) — ship if production flakes warrant it.
- Migrating `audit_driver.log()` from print to the `logging` module — separate refactor.
- `--auto-teardown` explicit-opt-in flag (option B from brainstorm) — rejected as overcautious.
- Fleet-level threshold escalation — rejected; systemic failure should preserve, not escalate to destroy.
- Integration test via subprocess — too much ceremony for a 3-line call-site diff.

## Rollback plan

If the new gate produces unexpected behavior in production (e.g., spurious "partial" statuses causing mass preservation + cost burn), rollback is a one-line revert: change the `finally:` body back to `teardown_all(provider, boxes)`. No data-shape change to `boxes.json` or any other on-disk state.
