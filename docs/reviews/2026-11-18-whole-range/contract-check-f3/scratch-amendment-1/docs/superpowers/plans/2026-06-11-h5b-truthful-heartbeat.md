# H5(b) Truthful Heartbeat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the scheduler heartbeat reflect cascade *progress*, not pulse-thread aliveness, so a wedged-but-pinging cascade surfaces as a phone alert within ~20 min (alert-only; sd_notify never goes quiet).

**Architecture:** A module-level run-token progress beacon (`bts/progress.py`, pure in-memory) gets stage-entry marks from `run_and_pick`/`run_pipeline`. The `heartbeat_watchdog` pulse reads the beacon each tick: fresh progress → `RUNNING` heartbeat with stage extras; stale progress (default 900s) → new `stalled` state — while ALWAYS sending sd_notify (WatchdogSec=1800 means silence = systemd kill, which is forbidden in Phase 1). The pulse is the single file writer: it drains stage transitions to a durations jsonl (Phase-2 threshold dataset). Existing consumers (`check_heartbeat.py` cron → healthchecks /fail; dashboard `is_heartbeat_fresh`) learn the `stalled` state.

**Tech Stack:** stdlib only (threading, json, time, uuid). Spec: `docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md`.

**Spec deviation (approved in plan review):** the spec says exit = "`end_run(run_id)` and a final synchronous drain"; since `end_run` retires the run_id (making a later drain return `[]`), `end_run` instead closes the in-progress stage and *returns* the remaining transition rows. Same behavior, single call.

**File map:**
- Create: `src/bts/progress.py` — beacon (begin_run/mark/end_run/snapshot/drain_transitions)
- Create: `tests/test_progress.py`
- Modify: `src/bts/heartbeat.py` — `HeartbeatState.STALLED`, `is_heartbeat_fresh` stalled branch, progress-aware `heartbeat_watchdog`
- Modify: `tests/test_heartbeat.py` — stalled-freshness + watchdog tests
- Modify: `scripts/check_heartbeat.py` — `stalled` → stale
- Modify: `tests/test_heartbeat_staleness.py` — stalled test
- Modify: `src/bts/orchestrator.py` — marks in `run_and_pick`
- Modify: `src/bts/model/predict.py` — marks in `run_pipeline` + `train_blend`
- Modify: `src/bts/scheduler.py` — both `heartbeat_watchdog` callsites pass kind/date/stall_after/durations_path
- Modify: `tests/test_orchestrator.py` — marks integration test
- Modify: `ARCHITECTURE.md` — heartbeat section

---

### Task 1: Progress beacon (`bts/progress.py`)

**Files:**
- Create: `src/bts/progress.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress.py`:

```python
"""Tests for the in-process progress beacon (H5b truthful heartbeat)."""
import logging

from bts import progress


def setup_function(_fn):
    # beacon is module-global; isolate tests by retiring any active run
    if progress._run_id is not None:
        progress.end_run(progress._run_id)


def test_begin_run_seeds_cascade_starting():
    rid = progress.begin_run("primary")
    snap = progress.snapshot(rid)
    assert snap is not None
    assert snap["stage"] == "cascade_starting"
    assert snap["kind"] == "primary"
    assert snap["stage_age_s"] >= 0
    assert snap["generation"] == 0


def test_mark_is_stage_entry_and_closes_previous():
    rid = progress.begin_run("primary")
    progress.mark("computing_features")
    snap = progress.snapshot(rid)
    assert snap["stage"] == "computing_features"
    assert snap["generation"] == 1
    rows = progress.drain_transitions(rid)
    assert [r["stage"] for r in rows] == ["cascade_starting"]
    assert rows[0]["duration_s"] >= 0
    assert rows[0]["generation"] == 0
    # drained -> gone
    assert progress.drain_transitions(rid) == []


def test_mark_without_active_run_is_noop():
    progress.mark("orphan_stage")  # must not raise
    assert progress.snapshot("anything") is None


def test_snapshot_rejects_foreign_and_retired_run_ids():
    rid = progress.begin_run("primary")
    assert progress.snapshot("not-" + rid) is None
    progress.end_run(rid)
    assert progress.snapshot(rid) is None


def test_begin_run_supersedes_prior_run():
    rid1 = progress.begin_run("primary")
    rid2 = progress.begin_run("shadow")
    assert progress.snapshot(rid1) is None
    assert progress.snapshot(rid2)["kind"] == "shadow"


def test_end_run_returns_final_transitions():
    rid = progress.begin_run("primary")
    progress.mark("selecting_pick")
    final = progress.end_run(rid)
    assert [r["stage"] for r in final] == ["cascade_starting", "selecting_pick"]
    # idempotent / foreign-safe
    assert progress.end_run(rid) == []


def test_drain_with_mismatched_run_id_returns_empty_without_clearing():
    rid = progress.begin_run("primary")
    progress.mark("stage_b")
    assert progress.drain_transitions("wrong") == []
    assert len(progress.drain_transitions(rid)) == 1


def test_history_bound_drops_oldest(caplog):
    rid = progress.begin_run("primary")
    with caplog.at_level(logging.WARNING):
        for i in range(progress.HISTORY_BOUND + 10):
            progress.mark(f"stage_{i}")
    rows = progress.drain_transitions(rid)
    assert len(rows) == progress.HISTORY_BOUND
    # oldest (cascade_starting + stage_0..stage_9) dropped; newest retained
    assert rows[-1]["stage"] == f"stage_{progress.HISTORY_BOUND + 8}"
    assert any("overflow" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_progress.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bts.progress'`

- [ ] **Step 3: Write the implementation**

Create `src/bts/progress.py`:

```python
"""In-process progress beacon for the prediction cascade (H5b truthful heartbeat).

Pure in-memory, module-level, thread-safe. The cascade pipeline calls
``mark(stage)`` at stage ENTRY (gerund names: "computing_features", not
"features_computed" — the watched age is "how long has the current stage been
running"). ``heartbeat_watchdog`` owns a run token from ``begin_run`` and is
the only reader; without it, marks are inert (the `bts run` CLI hits this
module harmlessly, no file is ever touched here).

Ownership contract: overlapping in-process runs are unsupported (primary and
shadow cascades run sequentially in the scheduler thread). ``begin_run``
supersedes any prior run; marks always attach to the current run and are
no-ops when none is active. A leaked pulse self-retires because
``snapshot(old_run_id)`` returns None.

Spec: docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md
"""

from __future__ import annotations

import logging
import time
import uuid
from threading import Lock

log = logging.getLogger(__name__)

HISTORY_BOUND = 256
START_STAGE = "cascade_starting"

_lock = Lock()
_run_id: str | None = None
_kind: str | None = None
_stage: str | None = None
_stage_started_mono: float = 0.0
_stage_started_wall: float = 0.0
_generation: int = 0
_transitions: list[dict] = []


def begin_run(kind: str) -> str:
    """Start a new run, superseding any prior one. Returns the run token."""
    global _run_id, _kind, _stage, _stage_started_mono, _stage_started_wall
    global _generation, _transitions
    with _lock:
        _run_id = uuid.uuid4().hex[:12]
        _kind = kind
        _stage = START_STAGE
        _stage_started_mono = time.monotonic()
        _stage_started_wall = time.time()
        _generation = 0
        _transitions = []
        return _run_id


def _close_current_stage_locked() -> None:
    if _stage is None:
        return
    if len(_transitions) >= HISTORY_BOUND:
        _transitions.pop(0)
        log.warning("progress transition history overflow; dropping oldest record")
    _transitions.append({
        "stage": _stage,
        "generation": _generation,
        "started_at": _stage_started_wall,
        "ended_at": time.time(),
        "duration_s": time.monotonic() - _stage_started_mono,
    })


def mark(stage: str) -> None:
    """Declare the stage now beginning (stage-ENTRY semantics).

    Completes the previous stage into the transition history. No-op when no
    run is active.
    """
    global _stage, _stage_started_mono, _stage_started_wall, _generation
    with _lock:
        if _run_id is None:
            return
        _close_current_stage_locked()
        _stage = stage
        _stage_started_mono = time.monotonic()
        _stage_started_wall = time.time()
        _generation += 1


def end_run(run_id: str) -> list[dict]:
    """Retire the run; returns the remaining transitions (incl. final stage).

    After this, ``snapshot(run_id)`` returns None. Foreign/already-retired
    run_ids return [].
    """
    global _run_id, _stage, _transitions
    with _lock:
        if run_id != _run_id:
            return []
        _close_current_stage_locked()
        out = _transitions
        _run_id = None
        _stage = None
        _transitions = []
        return out


def snapshot(run_id: str) -> dict | None:
    """Current stage view for the owning run; None for foreign/retired runs."""
    with _lock:
        if _run_id is None or run_id != _run_id or _stage is None:
            return None
        return {
            "stage": _stage,
            "stage_age_s": time.monotonic() - _stage_started_mono,
            "stage_started_wall": _stage_started_wall,
            "generation": _generation,
            "kind": _kind,
        }


def drain_transitions(run_id: str) -> list[dict]:
    """Return and clear completed-stage records; [] (no clearing) on mismatch."""
    global _transitions
    with _lock:
        if _run_id is None or run_id != _run_id:
            return []
        out = _transitions
        _transitions = []
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_progress.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/progress.py tests/test_progress.py
git commit -m "H5b: in-process progress beacon (run-token, stage-entry marks)"
```

---

### Task 2: `stalled` state in heartbeat consumers

**Files:**
- Modify: `src/bts/heartbeat.py` (HeartbeatState + is_heartbeat_fresh)
- Modify: `scripts/check_heartbeat.py`
- Test: `tests/test_heartbeat.py`, `tests/test_heartbeat_staleness.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_heartbeat.py`:

```python
def test_stalled_state_is_never_fresh(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 6, 11, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="stalled", now_utc=now,
                    extra={"stage": "computing_features", "stalled_for_s": 1000})

    # timestamp is brand new — age check alone would call this fresh
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=now) is False
```

Append to `tests/test_heartbeat_staleness.py` (match the file's existing helper style — it writes heartbeat JSON dicts and calls `is_stale`; mirror `test_unknown_state_is_stale`'s structure):

```python
def test_stalled_state_is_stale_with_stage_in_reason(tmp_path):
    path = tmp_path / ".heartbeat"
    path.write_text(json.dumps({
        "state": "stalled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": "computing_features",
        "stalled_for_s": 1234,
    }))

    stale, reason = is_stale(path)

    assert stale is True
    assert "computing_features" in reason
    assert "1234" in reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py::test_stalled_state_is_never_fresh tests/test_heartbeat_staleness.py::test_stalled_state_is_stale_with_stage_in_reason -q`
Expected: 2 failed (`is_heartbeat_fresh` returns True via age check; `is_stale` returns the generic "unknown state" reason without the stage)

- [ ] **Step 3: Implement**

In `src/bts/heartbeat.py`, add to `HeartbeatState`:

```python
class HeartbeatState:
    """Constants for well-known heartbeat state values."""
    RUNNING = "running"
    SLEEPING = "sleeping"
    WAITING_FOR_GAMES = "waiting_for_games"
    IDLE_END_OF_DAY = "idle_end_of_day"
    STALLED = "stalled"  # process alive but cascade progress stopped (H5b)
```

In `is_heartbeat_fresh`, after the `hb is None` check and BEFORE the sleeping branch:

```python
    # Stalled = process alive but no cascade progress; timestamps stay fresh
    # because the pulse keeps writing, so the age check below must not see it.
    if hb.get("state") == HeartbeatState.STALLED:
        return False
```

In `scripts/check_heartbeat.py`, before the `return True, f"unknown state: {state}"` line:

```python
    if state == "stalled":
        stage = raw.get("stage", "?")
        stalled_for = raw.get("stalled_for_s", "?")
        return True, f"cascade stalled in stage '{stage}' for {stalled_for}s (process alive, progress stopped)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py tests/test_heartbeat_staleness.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/heartbeat.py scripts/check_heartbeat.py tests/test_heartbeat.py tests/test_heartbeat_staleness.py
git commit -m "H5b: stalled heartbeat state — fails closed in both consumers"
```

---

### Task 3: Progress-aware `heartbeat_watchdog`

**Files:**
- Modify: `src/bts/heartbeat.py` (replace `heartbeat_watchdog`)
- Test: `tests/test_heartbeat.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_heartbeat.py`:

```python
import time as _time

from bts import progress
from bts.heartbeat import heartbeat_watchdog


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_watchdog_running_heartbeat_carries_stage(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=999):
        progress.mark("computing_features")
        _time.sleep(0.08)
        payload = read_heartbeat(hb)
        assert payload["state"] == "running"
        assert payload["stage"] == "computing_features"
        assert "run_id" in payload


def test_watchdog_flips_to_stalled_and_keeps_sd_notify(tmp_path: Path, monkeypatch):
    pings = []
    monkeypatch.setattr("bts.sd_notify.notify_watchdog", lambda: pings.append(1))
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.0,
                            durations_path=dur):
        progress.mark("computing_features")
        _time.sleep(0.08)
        payload = read_heartbeat(hb)
        assert payload["state"] == "stalled"
        assert payload["stage"] == "computing_features"
        assert payload["stalled_for_s"] >= 0
    assert len(pings) >= 2  # sd_notify NEVER stops during a stall (no systemd kill)
    rows = _read_jsonl(dur)
    incomplete = [r for r in rows if r["status"] == "stalled_incomplete"]
    assert len(incomplete) == 1  # once per stage instance, despite many ticks
    assert incomplete[0]["stage"] == "computing_features"
    assert incomplete[0]["kind"] == "primary"
    assert incomplete[0]["date"] == "2026-06-11"
    assert incomplete[0]["threshold_used_s"] == 0.0
    assert incomplete[0]["pid"] > 0


def test_watchdog_recovery_flips_back_to_running_and_marks_ok_after_stall(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.05,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.12)            # stage_a crosses 0.05s -> stalled
        assert read_heartbeat(hb)["state"] == "stalled"
        progress.mark("stage_b")     # recovery: stage_a completes late
        _time.sleep(0.04)            # stage_b still under threshold
        assert read_heartbeat(hb)["state"] == "running"
    rows = _read_jsonl(dur)
    a_rows = [r for r in rows if r["stage"] == "stage_a"]
    assert {r["status"] for r in a_rows} == {"stalled_incomplete", "ok_after_stall"}


def test_watchdog_second_stall_in_later_stage_gets_own_row(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.05,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.1)
        progress.mark("stage_b")
        _time.sleep(0.1)
    incomplete = [r for r in _read_jsonl(dur) if r["status"] == "stalled_incomplete"]
    assert [r["stage"] for r in incomplete] == ["stage_a", "stage_b"]


def test_watchdog_exit_retires_run_and_drains_final_stage(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="shadow",
                            date="2026-06-11", stall_after_sec=999,
                            durations_path=dur) as run_id:
        progress.mark("selecting_pick")
    assert progress.snapshot(run_id) is None  # retired: leaked pulse writes nothing
    stages = [r["stage"] for r in _read_jsonl(dur)]
    assert "selecting_pick" in stages  # final drain captured the last stage
    assert all(r["kind"] == "shadow" for r in _read_jsonl(dur))
    # post-exit scheduler heartbeats are not overwritten
    write_heartbeat(hb, state="sleeping")
    _time.sleep(0.05)
    assert read_heartbeat(hb)["state"] == "sleeping"


def test_watchdog_stall_latch_not_flipped_on_failed_append(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    blocked = tmp_path / "blocked" 
    blocked.write_text("")  # durations_path parent mkdir will fail under a file
    dur = blocked / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.0,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.06)  # several ticks; every append fails; must not raise
    assert not dur.exists()


def test_watchdog_backward_compatible_defaults(tmp_path: Path):
    # existing callers pass only (path, interval_sec) — must still work
    hb = tmp_path / ".heartbeat"
    with heartbeat_watchdog(hb, interval_sec=0.01):
        _time.sleep(0.03)
        assert read_heartbeat(hb)["state"] == "running"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py -q`
Expected: new tests FAIL (`heartbeat_watchdog` got unexpected keyword `kind` / context manager yields None not run_id)

- [ ] **Step 3: Implement**

In `src/bts/heartbeat.py`: add imports `import logging`, `import os`, `import time` at top plus `log = logging.getLogger(__name__)`, and replace `heartbeat_watchdog` entirely:

```python
@contextmanager
def heartbeat_watchdog(
    path: Path,
    interval_sec: float = 60,
    *,
    kind: str = "primary",
    date: Optional[str] = None,
    stall_after_sec: float = 900,
    durations_path: Optional[Path] = None,
) -> Iterator[str]:
    """Refresh the heartbeat while executing the body — truthfully (H5b).

    Owns a progress-beacon run token. Each tick: ALWAYS sd_notify (the unit
    has WatchdogSec=1800 — silence means systemd kills the scheduler, which
    Phase 1 forbids), then heartbeat state by progress age: fresh marks →
    RUNNING with {stage, stage_age_s, run_id}; older than ``stall_after_sec``
    → STALLED with {stage, stalled_for_s, run_id} (re-snapshotted just before
    writing to shrink the stall-vs-recovery race to one tick). The pulse is
    the single writer of ``durations_path`` (stage-transition jsonl, the
    Phase-2 threshold dataset): completed stages get status ok /
    ok_after_stall; the first tick of each stalled stage instance appends one
    stalled_incomplete row (latch flips only on successful append).

    Yields the run_id (tests use it; callers may ignore). On exit the run is
    retired — a leaked pulse's snapshot returns None, so it can never
    overwrite the scheduler's post-cascade state heartbeats.
    """
    from bts import progress
    from bts import sd_notify

    run_id = progress.begin_run(kind)
    stop = Event()
    stalled_gens: set = set()      # stage generations observed stalled
    persisted_gens: set = set()    # generations whose stalled_incomplete row is on disk

    def _append_rows(rows: list) -> bool:
        if durations_path is None or not rows:
            return durations_path is not None
        try:
            durations_path.parent.mkdir(parents=True, exist_ok=True)
            with durations_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            return True
        except Exception as e:
            log.warning(f"stage-durations append failed (non-fatal): {e}")
            return False

    def _enrich(rec: dict, status: str) -> dict:
        return {
            "run_id": run_id, "pid": os.getpid(), "kind": kind, "date": date,
            "threshold_used_s": stall_after_sec, "status": status, **rec,
        }

    def _drain_completed() -> None:
        rows = [
            _enrich(r, "ok_after_stall" if r["generation"] in stalled_gens else "ok")
            for r in progress.drain_transitions(run_id)
        ]
        _append_rows(rows)

    def _tick() -> None:
        sd_notify.notify_watchdog()  # ALWAYS — never let systemd kill on a stall
        snap = progress.snapshot(run_id)
        if snap is None:
            return  # run retired/superseded: write nothing, ever
        if snap["stage_age_s"] >= stall_after_sec:
            snap = progress.snapshot(run_id)  # revalidate: recovery may have marked
            if snap is None:
                return
        if snap["stage_age_s"] < stall_after_sec:
            write_heartbeat(path, state=HeartbeatState.RUNNING, extra={
                "stage": snap["stage"],
                "stage_age_s": round(snap["stage_age_s"], 1),
                "run_id": run_id,
            })
        else:
            write_heartbeat(path, state=HeartbeatState.STALLED, extra={
                "stage": snap["stage"],
                "stalled_for_s": round(snap["stage_age_s"], 1),
                "run_id": run_id,
            })
            gen = snap["generation"]
            stalled_gens.add(gen)
            if gen not in persisted_gens:
                row = _enrich({
                    "stage": snap["stage"],
                    "generation": gen,
                    "started_at": snap["stage_started_wall"],
                    "ended_at": None,
                    "duration_s": round(snap["stage_age_s"], 1),
                }, "stalled_incomplete")
                if _append_rows([row]):
                    persisted_gens.add(gen)
        _drain_completed()

    def _pulse() -> None:
        while not stop.is_set():
            try:
                _tick()
            except Exception:
                pass
            stop.wait(interval_sec)

    thread = Thread(target=_pulse, daemon=True)
    thread.start()
    try:
        yield run_id
    finally:
        stop.set()
        thread.join(timeout=2)
        final = [
            _enrich(r, "ok_after_stall" if r["generation"] in stalled_gens else "ok")
            for r in progress.end_run(run_id)
        ]
        _append_rows(final)
```

Note: `_append_rows([])` returning `durations_path is not None` is irrelevant in practice (callers pass non-empty rows for the latch path); the latch call always has one row.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_heartbeat.py tests/test_progress.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/bts/heartbeat.py tests/test_heartbeat.py
git commit -m "H5b: progress-aware heartbeat_watchdog — stalled state, sd_notify invariant, durations jsonl"
```

---

### Task 4: Pipeline marks + scheduler callsites

**Files:**
- Modify: `src/bts/orchestrator.py` (`run_and_pick`)
- Modify: `src/bts/model/predict.py` (`run_pipeline`, `train_blend`)
- Modify: `src/bts/scheduler.py` (both `heartbeat_watchdog` callsites)
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py` inside `class TestRunAndPick` (same patch stack as `test_persists_full_slate`):

```python
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    @patch("bts.strategy._mdp_action", return_value="single")
    def test_emits_progress_marks(
        self, _mdp, _statuses, _detailed_statuses, mock_cascade, tmp_path
    ):
        import pandas as pd
        from bts import progress
        from bts.orchestrator import run_and_pick

        mock_cascade.return_value = (
            pd.DataFrame(json.loads(SAMPLE_PREDICTIONS)),
            "mac",
        )
        config = {
            "orchestrator": {"picks_dir": str(tmp_path)},
            "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
        }
        rid = progress.begin_run("primary")
        run_and_pick(config, "2026-04-01")
        stages = [r["stage"] for r in progress.end_run(rid)]

        assert "running_cascade" in stages
        assert "selecting_pick" in stages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest "tests/test_orchestrator.py::TestRunAndPick::test_emits_progress_marks" -q`
Expected: FAIL (`"running_cascade" in stages` — no marks emitted)

- [ ] **Step 3: Implement the marks**

`src/bts/orchestrator.py`, in `run_and_pick` — add `from bts import progress` to the function's local imports, then marks (stage-ENTRY names):

```python
    from bts import progress

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    progress.mark("running_cascade")
    predictions, tier_name = run_cascade(config["tiers"], date)
    if predictions is None or predictions.empty:
        return predictions, None, tier_name

    # Persist the full ranked slate (observability only — save_slate never
    # raises). Enables realized slate-level metrics; see bts/slate.py.
    from bts.slate import save_slate
    progress.mark("persisting_slate")
    save_slate(predictions, date, picks_dir, tier_name)

    progress.mark("loading_decision_state")
    decision_state = load_decision_streak_state(
        picks_dir,
        require_contest_state=_contest_state_required(config),
    )
    try:
        game_statuses_detailed = get_game_statuses_detailed(date)
    except Exception:
        game_statuses_detailed = None
    progress.mark("selecting_pick")
    result = select_pick(
```

`src/bts/model/predict.py`, in `run_pipeline` — add `from bts import progress` to the module imports (top, after the bts.features import), then:

```python
    if refresh_data:
        progress.mark("refreshing_data")
        _refresh_season_data(date, processed_dir=data_dir)

    proc = Path(data_dir)
    progress.mark("loading_parquets")
    dfs = []
```

after the concat:

```python
    df = pd.concat(dfs, ignore_index=True)
    progress.mark("computing_features")
    df = compute_all_features(df)
```

around training:

```python
    if cached_blend:
        model = cached_blend.pop("_model")
        blend = cached_blend
    else:
        progress.mark("training_single_model")
        model = train_model(df, feature_cols=feature_cols_override)
        blend = train_blend(
```

before lookups and predict:

```python
    progress.mark("building_lookups")
    lookups = _build_feature_lookups(df)

    progress.mark("predicting")
    return predict(
```

In `train_blend`, inside the `for config in configs:` loop, first line of the loop body:

```python
    for config in configs:
        progress.mark(f"training_blend_{config[0]}")
```

(`config[0]` is the blend name for both 2- and 3-tuples.)

`src/bts/scheduler.py` — both callsites. Primary (in the lineup-check pick path, currently `with heartbeat_watchdog(heartbeat_path, interval_sec=60):` wrapping `run_and_pick`):

```python
    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))
    stall_after = float(config.get("scheduler", {}).get("heartbeat_stall_after_sec", 900))
    durations_path = Path("data/health_state/cascade_stage_durations.jsonl")
    try:
        with heartbeat_watchdog(
            heartbeat_path, interval_sec=60,
            kind="primary", date=date,
            stall_after_sec=stall_after, durations_path=durations_path,
        ):
            predictions, pick_result, tier = run_and_pick(
```

Shadow (the `with heartbeat_watchdog(heartbeat_path, interval_sec=60):` wrapping `predict_local_shadow`): same pattern with `kind="shadow"` (reuse the same `stall_after`/`durations_path` expressions locally — that function has its own `config` in scope).

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py tests/test_scheduler.py tests/model -q -m "not slow"`
Expected: all pass (scheduler tests exercise the new kwargs via defaults)

- [ ] **Step 5: Commit**

```bash
git add src/bts/orchestrator.py src/bts/model/predict.py src/bts/scheduler.py tests/test_orchestrator.py
git commit -m "H5b: stage-entry marks through cascade + scheduler watchdog wiring"
```

---

### Task 5: Docs, full suite, deploy

**Files:**
- Modify: `ARCHITECTURE.md` (Health Monitoring section)
- Modify: `docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md` (status line)

- [ ] **Step 1: Update ARCHITECTURE.md**

In the Health Monitoring section, after the State files paragraph, add:

```markdown
**Truthful heartbeat (H5b, 2026-06-11):** during cascades, `heartbeat_watchdog`
ticks read an in-process progress beacon (`bts.progress`, stage-entry marks
through `run_and_pick`/`run_pipeline`). Progress fresh → `state=running` with
`{stage, stage_age_s, run_id}`; no progress for `heartbeat_stall_after_sec`
(default 900, `[scheduler]` toml) → `state=stalled` → `check_heartbeat.py`
cron POSTs healthchecks /fail within ≤5 min (dashboard `is_heartbeat_fresh`
also fails closed). **sd_notify pings continue during a stall** — the unit has
`WatchdogSec=1800`, and Phase 1 is alert-only (no auto-kill; Codex-reviewed).
Stage durations append to `data/health_state/cascade_stage_durations.jsonl`
(`status ∈ {ok, ok_after_stall, stalled_incomplete}`) — the dataset for any
Phase-2 data-derived thresholds. Spec:
`docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md`.
```

Update the spec's Status line to: `**Status: IMPLEMENTED + DEPLOYED 2026-06-11 (Phase 1, alert-only). Codex (gpt-5.5) adversarial review incorporated (2 rounds).**`

- [ ] **Step 2: Run the full not-slow suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m "not slow"`
Expected: all pass (1880+; zero failures)

- [ ] **Step 3: Commit docs**

```bash
git add ARCHITECTURE.md docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md
git commit -m "H5b: ARCHITECTURE + spec status"
```

- [ ] **Step 4: Deploy and verify**

```bash
git push origin main
git push origin main:deploy
# wait for workflow; then verify:
gh run list --repo stone-ericm/bts --limit 2          # both success
ssh bts-hetzner 'cd ~/projects/bts && git log --oneline -1 && systemctl --user is-active bts-scheduler bts-dashboard'
```

After the next cascade (or trigger a lineup-check cycle), verify live:

```bash
ssh bts-hetzner 'cat ~/projects/bts/data/.heartbeat; echo; tail -5 ~/projects/bts/data/health_state/cascade_stage_durations.jsonl'
```

Expected: heartbeat carries `stage`/`run_id` during cascades; jsonl rows with `status=ok` accumulate.
```
