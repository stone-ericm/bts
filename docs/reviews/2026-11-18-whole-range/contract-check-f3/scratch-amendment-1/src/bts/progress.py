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
