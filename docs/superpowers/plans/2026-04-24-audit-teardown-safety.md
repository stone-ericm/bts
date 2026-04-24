# Audit Teardown Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate audit-driver teardown on retrieve success (per-box skip) so a partial rsync never destroys data.

**Architecture:** Add a pure helper `teardown_retrieved` to `scripts/audit_driver.py` that filters boxes by `retrieve_results[box.name] == "ok"` and returns `(selected, deleted)`. Promote `teardown_all` to return the successful-delete count. Rewire `finally:` blocks in both `audit_driver.main` and `audit_attach.main` to use the helper and `raise SystemExit(1)` when any box is preserved.

**Tech Stack:** Python 3.12, pytest 9.x, pytest monkeypatch fixture.

**Spec:** `docs/superpowers/specs/2026-04-24-audit-teardown-safety-design.md` (commits `d3e5082` + `720e8d0`).

**Branch:** Continue on `main`. No worktree — changes are additive, unit-test-covered, and don't affect the in-flight Vultr audit which uses `--no-teardown`.

---

## File Structure

- **Modify** `tests/test_audit_driver.py` — add 2 new test classes (`TestTeardownRetrieved`, `TestTeardownAllReturn`). Existing `TestKeychainFallback` untouched.
- **Modify** `scripts/audit_driver.py` — change `teardown_all` return type to `int`; add new `teardown_retrieved` function; update `main()`'s retrieve-loop + `finally:` block.
- **Modify** `scripts/audit_attach.py` — same main/finally changes as audit_driver.

No new files. No Python imports added beyond what the two scripts already use.

---

### Task 1: Test-infrastructure fixtures

**Files:**
- Modify: `tests/test_audit_driver.py` — append to end of file

- [ ] **Step 1: Append FakeProvider + fixtures to end of test file**

Append this after the existing `TestKeychainFallback` class:

```python


# ---------------------------------------------------------------------------
# teardown_retrieved + teardown_all tests
# ---------------------------------------------------------------------------

class FakeProvider:
    """Captures delete() calls instead of hitting a real API."""

    name = "fake"

    def __init__(self, raise_on_ids: set[str] | None = None) -> None:
        self.deleted: list[str] = []
        self._raise_on = raise_on_ids or set()

    def delete(self, box_id: str) -> None:
        if box_id in self._raise_on:
            raise RuntimeError(f"fake API failure for {box_id}")
        self.deleted.append(box_id)


@pytest.fixture
def captured_log(monkeypatch):
    """Replace audit_driver.log with a list-appender; returns the list."""
    from audit_driver import log as _original_log  # noqa: F401 — force import first
    import audit_driver
    captured: list[str] = []
    monkeypatch.setattr(audit_driver, "log", captured.append)
    return captured


@pytest.fixture
def boxes():
    from audit_driver import Box
    return [
        Box(id="1", name="b1", ipv4="10.0.0.1", region=""),
        Box(id="2", name="b2", ipv4="10.0.0.2", region=""),
        Box(id="3", name="b3", ipv4="10.0.0.3", region=""),
    ]
```

- [ ] **Step 2: Verify file still parses (no tests added yet)**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_audit_driver.py -v`

Expected: `2 passed` (the existing TestKeychainFallback tests — collection didn't break).

- [ ] **Step 3: Commit**

```bash
git add tests/test_audit_driver.py
git commit -m "test(audit): fixtures for teardown_retrieved tests

FakeProvider captures delete() calls; captured_log monkeypatches
audit_driver.log to a list-appender; boxes fixture yields three
Box instances. No new tests yet — infrastructure only."
```

---

### Task 2: Happy-path + single-partial + all-partial tests (cases 1-3)

**Files:**
- Modify: `tests/test_audit_driver.py`

- [ ] **Step 1: Append TestTeardownRetrieved with first three tests**

Append to end of `tests/test_audit_driver.py`:

```python


class TestTeardownRetrieved:
    def test_all_ok_tears_down_everything(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "ok", "b2": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "2", "3"]
        assert selected == 3
        assert deleted == 3

    def test_one_partial_preserves_only_that_box(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "ok", "b2": "partial", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "3"]
        assert selected == 2
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "PRESERVED b2" in joined
        assert "ip=10.0.0.2" in joined
        assert "retrieve_status=partial" in joined

    def test_all_partial_preserves_all(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        results = {"b1": "partial", "b2": "partial", "b3": "partial"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0
        preserved_lines = [l for l in captured_log if "PRESERVED" in l]
        assert len(preserved_lines) == 3
```

- [ ] **Step 2: Run these three tests, verify RED with ImportError**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_audit_driver.py::TestTeardownRetrieved -v`

Expected:
```
ImportError: cannot import name 'teardown_retrieved' from 'audit_driver'
```
or `3 failed` with ImportError at the top of each test body.

- [ ] **Step 3: Commit (red state)**

```bash
git add tests/test_audit_driver.py
git commit -m "test(audit): teardown_retrieved cases 1-3 (happy / partial / all-partial)

Expect RED until audit_driver.teardown_retrieved exists."
```

---

### Task 3: Edge-case tests (cases 4-7: missing key, empty results, empty boxes, malformed)

**Files:**
- Modify: `tests/test_audit_driver.py`

- [ ] **Step 1: Append four more tests to TestTeardownRetrieved**

Append INSIDE the `TestTeardownRetrieved` class (after `test_all_partial_preserves_all`):

```python
    def test_missing_key_defaults_to_preserve(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        # b2 is missing from the dict
        results = {"b1": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "3"]
        assert selected == 2
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "PRESERVED b2" in joined
        assert "retrieve_status=not-attempted" in joined

    def test_empty_results_preserves_all(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        selected, deleted = teardown_retrieved(provider, boxes, {})

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0
        # All 3 boxes marked not-attempted
        not_attempted = [l for l in captured_log if "not-attempted" in l]
        assert len(not_attempted) == 3

    def test_empty_boxes_list_noop(self, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        selected, deleted = teardown_retrieved(provider, [], {"b1": "ok"})

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0

    def test_malformed_values_preserve(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        # None, bool, unknown string — anything != "ok" preserves
        results = {"b1": None, "b2": True, "b3": "weird"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == []
        assert selected == 0
        assert deleted == 0
```

- [ ] **Step 2: Run and verify still RED**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_audit_driver.py::TestTeardownRetrieved -v`

Expected: 7 failed (import errors on each test body).

- [ ] **Step 3: Commit**

```bash
git add tests/test_audit_driver.py
git commit -m "test(audit): teardown_retrieved cases 4-7 (missing / empty / malformed)"
```

---

### Task 4: Error + integration tests (cases 8-11: raises, None arg, stray key, teardown_all returns int)

**Files:**
- Modify: `tests/test_audit_driver.py`

- [ ] **Step 1: Append remaining four tests**

Two tests go INSIDE `TestTeardownRetrieved`, then a new class `TestTeardownAllReturn` for the standalone-teardown_all return-value test.

Append inside `TestTeardownRetrieved`:

```python
    def test_provider_delete_raises_on_one_box(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        # teardown_all's try/except will swallow this and keep going
        provider = FakeProvider(raise_on_ids={"2"})
        results = {"b1": "ok", "b2": "ok", "b3": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        # selected = 3 (picked all three), deleted = 2 (b2's API call failed)
        assert provider.deleted == ["1", "3"]
        assert selected == 3
        assert deleted == 2
        joined = "\n".join(captured_log)
        assert "FAILED to delete b2" in joined

    def test_retrieve_results_none_raises_typeerror(self, boxes):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()

        with pytest.raises(TypeError):
            teardown_retrieved(provider, boxes, None)

    def test_stray_key_logged_no_crash(self, boxes, captured_log):
        from audit_driver import teardown_retrieved
        provider = FakeProvider()
        # "b99" isn't in boxes — stray key
        results = {"b1": "ok", "b2": "ok", "b3": "ok", "b99": "ok"}

        selected, deleted = teardown_retrieved(provider, boxes, results)

        assert provider.deleted == ["1", "2", "3"]
        assert selected == 3
        assert deleted == 3
        joined = "\n".join(captured_log)
        assert "unrecognized key" in joined
        assert "b99" in joined
```

Append as a NEW class (outside `TestTeardownRetrieved`):

```python


class TestTeardownAllReturn:
    def test_teardown_all_returns_count_of_successful_deletes(self, boxes, captured_log):
        from audit_driver import teardown_all
        provider = FakeProvider(raise_on_ids={"2"})

        deleted = teardown_all(provider, boxes)

        assert deleted == 2  # b1 and b3 succeeded; b2 raised
        assert provider.deleted == ["1", "3"]
```

- [ ] **Step 2: Run and verify all 11 RED**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_audit_driver.py::TestTeardownRetrieved tests/test_audit_driver.py::TestTeardownAllReturn -v`

Expected: 11 failed.

- [ ] **Step 3: Commit (full red state)**

```bash
git add tests/test_audit_driver.py
git commit -m "test(audit): teardown_retrieved cases 8-11 + TestTeardownAllReturn

All 11 new tests red; teardown_retrieved not yet implemented and
teardown_all still returns None."
```

---

### Task 5: Implement `teardown_all` return + `teardown_retrieved` helper

**Files:**
- Modify: `scripts/audit_driver.py` — lines 745-755 (teardown_all); insert new function after teardown_all.

- [ ] **Step 1: Update `teardown_all` to count + return int**

Replace the entire current body of `teardown_all` (audit_driver.py:745-755) with:

```python
def teardown_all(provider: Provider, boxes: list[Box]) -> int:
    """Unconditional teardown. Called from finally blocks to guarantee cleanup
    even if the main flow fails. Fixes the 2026-04-14 Hetzner retrieve-cascade
    bug where partial rsync → no teardown → zombie boxes.

    Returns the count of boxes whose provider.delete() call did not raise.
    """
    log("=== TEARDOWN ===")
    deleted = 0
    for box in boxes:
        try:
            provider.delete(box.id)
            log(f"  deleted {box.name}")
            deleted += 1
        except Exception as e:
            log(f"  FAILED to delete {box.name}: {e}")
    return deleted
```

- [ ] **Step 2: Insert `teardown_retrieved` immediately after `teardown_all`**

Insert after the closing line of `teardown_all` (and before the `# ---` Main divider at audit_driver.py:757):

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
                  (inherited from teardown_all; `selected - deleted` = API-failed)

    Callers use `preserved = len(boxes) - selected` to know how many boxes
    were held back for data-integrity. That's the signal that drives the
    non-zero exit code.

    Any box name missing from retrieve_results is treated as "not-attempted"
    and preserved. Default-to-preserve makes the helper safe to call even if
    the retrieve loop was interrupted partway through.
    """
    if retrieve_results is None:
        raise TypeError("retrieve_results must be a dict, got None")

    box_names = {b.name for b in boxes}
    for key in retrieve_results:
        if key not in box_names:
            log(f"  unrecognized key in retrieve_results: {key} — caller bug?")

    candidates: list[Box] = []
    for box in boxes:
        status = retrieve_results.get(box.name)
        if status == "ok":
            candidates.append(box)
        else:
            status_str = status if status is not None else "not-attempted"
            log(f"  PRESERVED {box.name} ip={box.ipv4} retrieve_status={status_str}")

    deleted = teardown_all(provider, candidates) if candidates else 0
    return len(candidates), deleted
```

- [ ] **Step 3: Run all 11 new tests — expect GREEN**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_audit_driver.py::TestTeardownRetrieved tests/test_audit_driver.py::TestTeardownAllReturn -v`

Expected: `11 passed`.

- [ ] **Step 4: Run full suite sanity**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

Expected: `612 passed` (was 601; +11 new).

- [ ] **Step 5: Commit (green state)**

```bash
git add scripts/audit_driver.py
git commit -m "feat(audit): teardown_retrieved helper + teardown_all returns int

Gates teardown on per-box retrieve success. Preserves any box whose
retrieve_results[name] != 'ok' (including missing keys and malformed
values). Logs PRESERVED lines for each held-back box with name + ip
+ status so operator can investigate / re-retrieve manually.

teardown_all now returns the count of successful delete() calls so
callers can distinguish data-integrity preservation from transient
provider API failures.

All 11 new unit tests pass; full suite 612 passing.

Call-site wiring comes in the next two commits (audit_driver.main
and audit_attach.main)."
```

---

### Task 6: Wire `audit_driver.main` to use `teardown_retrieved`

**Files:**
- Modify: `scripts/audit_driver.py` — main()'s retrieve loop + finally block (around lines 888-902).

- [ ] **Step 1: Read current context to confirm line numbers**

Run: `grep -n "retrieve_results\|for fut in concurrent.futures.as_completed\|# Retrieve\|finally:" /Users/stone/projects/bts/scripts/audit_driver.py | head -20`

Expect a hit at `finally:` around line 897 and `for fut in ...` around line 891 inside `main()`.

- [ ] **Step 2: Add `retrieve_results` init + `exit_code` before the main try block**

Find the line in `main()` that reads `try:` BEFORE the poll-and-retrieve loop (around line 860-875). Immediately before it, insert:

```python
    retrieve_results: dict[str, str] = {}
    exit_code = 0
```

- [ ] **Step 3: Populate `retrieve_results` inside the retrieve futures loop**

Find the block:

```python
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
```

Add ONE line after `nm, status, errs = fut.result()`:

```python
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                retrieve_results[nm] = status
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
```

- [ ] **Step 4: Replace the finally block body**

Find the current block:

```python
    finally:
        # Unconditional teardown, always. This is the key fix for the
        # 2026-04-14 Hetzner retrieve-cascade bug.
        if boxes:
            teardown_all(provider, boxes)
```

Replace with:

```python
    finally:
        if not boxes:
            pass
        else:
            selected, deleted = teardown_retrieved(provider, boxes, retrieve_results)
            preserved = len(boxes) - selected      # held back for data-integrity
            api_failed = selected - deleted         # provider API transient failures
            log(f"=== TEARDOWN: selected={selected}/{len(boxes)} "
                f"deleted={deleted} preserved={preserved} api_failed={api_failed} ===")
            if preserved > 0:
                exit_code = 1  # data-integrity signal for external monitoring
```

- [ ] **Step 5: Add exit-code escalation after the try/finally**

Find the line `log("=== AUDIT DRIVER DONE ===")` at the end of `main()`. Replace with:

```python
    log("=== AUDIT DRIVER DONE ===")
    if exit_code:
        raise SystemExit(exit_code)
```

- [ ] **Step 6: Run full suite — expect 612 still green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

Expected: `612 passed`. The unit tests for `teardown_retrieved` still cover the helper; main()'s wiring doesn't have new unit tests (integration-only per spec).

- [ ] **Step 7: Commit**

```bash
git add scripts/audit_driver.py
git commit -m "feat(audit): wire audit_driver.main to teardown_retrieved

main() now populates retrieve_results during the retrieve futures
loop and uses teardown_retrieved in its finally block. Summary log
line reports selected/deleted/preserved/api_failed; raise
SystemExit(1) when any box is preserved (data-integrity signal).

audit_attach.main wiring comes in the next commit."
```

---

### Task 7: Wire `audit_attach.main` — same change in the second entry point

**Files:**
- Modify: `scripts/audit_attach.py` — main()'s retrieve loop + finally block (lines 101-134).

- [ ] **Step 1: Update imports**

Current `scripts/audit_attach.py:31-40`:

```python
from audit_driver import (
    Box,
    DEFAULT_SEEDS,
    distribute_seeds,
    log,
    make_provider,
    poll,
    retrieve_one,
    teardown_all,
)
```

Add `teardown_retrieved` to the import list (append at the end — alphabetical with the existing names):

```python
from audit_driver import (
    Box,
    DEFAULT_SEEDS,
    distribute_seeds,
    log,
    make_provider,
    poll,
    retrieve_one,
    teardown_all,
    teardown_retrieved,
)
```

- [ ] **Step 2: Add `retrieve_results` init + `exit_code` before the main try block**

Find `scripts/audit_attach.py:101` which is:

```python
    try:
        start = time.time()
```

Insert TWO lines immediately before `try:`:

```python
    retrieve_results: dict[str, str] = {}
    exit_code = 0

    try:
        start = time.time()
```

- [ ] **Step 3: Populate `retrieve_results` inside retrieve futures loop**

Find `scripts/audit_attach.py:122-127`:

```python
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
```

Add one line after `nm, status, errs = fut.result()`:

```python
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                retrieve_results[nm] = status
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
```

- [ ] **Step 4: Replace finally block body**

Find current `scripts/audit_attach.py:128-134`:

```python
    finally:
        if args.no_teardown:
            log("--no-teardown set; leaving boxes alive")
        elif not boxes:
            pass
        else:
            teardown_all(provider, boxes)
```

Replace with:

```python
    finally:
        if args.no_teardown:
            log("--no-teardown set; leaving boxes alive")
        elif not boxes:
            pass
        else:
            selected, deleted = teardown_retrieved(provider, boxes, retrieve_results)
            preserved = len(boxes) - selected
            api_failed = selected - deleted
            log(f"=== TEARDOWN: selected={selected}/{len(boxes)} "
                f"deleted={deleted} preserved={preserved} api_failed={api_failed} ===")
            if preserved > 0:
                exit_code = 1
```

- [ ] **Step 5: Add exit-code escalation after the try/finally**

Find `scripts/audit_attach.py:136` which is:

```python
    log("=== AUDIT ATTACH DONE ===")
```

Replace with:

```python
    log("=== AUDIT ATTACH DONE ===")
    if exit_code:
        raise SystemExit(exit_code)
```

- [ ] **Step 6: Run full suite — expect 612 still green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`

Expected: `612 passed`.

- [ ] **Step 7: Quick syntax/import sanity on the two modified scripts**

Run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import sys; sys.path.insert(0, 'scripts'); import audit_driver; import audit_attach; print('imports OK'); print('teardown_retrieved:', audit_driver.teardown_retrieved)"
```

Expected:
```
imports OK
teardown_retrieved: <function teardown_retrieved at 0x...>
```

- [ ] **Step 8: Commit**

```bash
git add scripts/audit_attach.py
git commit -m "feat(audit): wire audit_attach.main to teardown_retrieved

Symmetric with audit_driver.main. Imports teardown_retrieved,
populates retrieve_results during retrieve loop, uses it in the
finally block, raises SystemExit(1) on preserved > 0.

Completes the partial-retrieve data-loss fix. Both entry points
now enforce the same data-integrity invariant: a box is only
torn down if retrieve_one returned status='ok' for it."
```

---

## Self-Review Checklist (for the implementing engineer)

After Task 7 commits, run through these checks:

- [ ] **Spec coverage**: Read spec sections in order; verify each decision is reflected in the code.
  - Q1 (teardown-by-default gated on success): Yes — `if preserved > 0: exit_code = 1` at both call sites
  - Q2 (per-box skip): Yes — `teardown_retrieved` iterates boxes and filters per-name
  - Q3 (fix both scripts): Yes — Tasks 6 + 7 hit both
  - Q4 (extract helper): Yes — `teardown_retrieved` in audit_driver.py
  - Q5 (exit 1 on skip): Yes — `raise SystemExit(exit_code)` added to both mains
- [ ] **Behavior matrix**: `"ok"` → tear down; `"partial"` → preserve + log; missing → preserve + log `"not-attempted"`; other values → preserve.
- [ ] **Key invariant**: destruction requires explicit `retrieve_results[box.name] == "ok"`. Grep the helper to confirm no code path destroys on any other condition.
- [ ] **All 11 tests in `tests/test_audit_driver.py::TestTeardownRetrieved` + `TestTeardownAllReturn` pass.**
- [ ] **Full suite: 612 passing.**
- [ ] **No stray placeholder strings** in the code (`TODO`, `FIXME`, `XXX`).

## Post-merge: README note for the operator

Not required for this plan but nice-to-have: append a paragraph to `scripts/audit_attach.py`'s module docstring noting "Exits 1 when any box is preserved for data-integrity reasons; check log for PRESERVED lines and manually re-retrieve + teardown." Skipped for now to keep scope tight.

## Rollback

One-line revert per entry point: change the `finally:` body back to `teardown_all(provider, boxes)`. `teardown_all` is backward-compatible with the `-> int` return (return value can be ignored).

No data shape changes on disk. Any in-flight audit (currently running with `--no-teardown`) is unaffected by the code change until audit_attach is restarted, and restart is an operator-controlled action.
