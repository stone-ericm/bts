"""The reconcile receipt (C1 rank-2 watchdog plan P2; registration I-207 and its producer receipt contract).

`bts reconcile` publishes one receipt per run (schema `bts_reconcile_receipt_v1`; paths, states and consumer rules in
`docs/ops/reconcile-receipt-v1.md`). For every target date and every slot it records what happened:
- **Per date:** skipped past the cutoff, no pick file, not graded, attempted then observed / pending / failed / late.
- **Per write:** correction applied, slot results updated, unchanged, refused after the cutoff, or skipped under the
  lock.
- **Per slot:** the selected batter and game, the terminal or void basis, the sha256 and completion time of every
  source payload the grader parsed, and whether it is covered (observed before the 08:00 ET next-day cutoff).

An empty corrections list establishes nothing; this receipt is the coverage evidence.

**It only records.** Every recording method is guarded: a failure inside one marks the receipt `degraded` and never
changes the reconcile run's requests, results, writes or streak. Publication goes through `bts.receipt_io`: a failed
publication leaves no discoverable receipt, is reported on stderr, and is unavailable evidence.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from bts import receipt_io

SCHEMA = "bts_reconcile_receipt_v1"
REPO = Path(__file__).resolve().parents[2]


def receipts_root(picks_dir: Path) -> Path:
    return Path(picks_dir).parent / "health_state" / "reconcile_receipts"


def discover(picks_dir: Path, run_date: str) -> list[dict]:
    """Published receipts for runs started on one ET date (a failed publication is never returned)."""
    return [json.loads(p.read_text()) for p in receipt_io.discover(receipts_root(picks_dir) / run_date)]


def _guarded(fn):
    """A recording failure degrades the receipt; it never reaches the reconcile run."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as exc:  # noqa: BLE001 - the receipt must never change the run
            self.degraded.append(type(exc).__name__)
            return None
    return wrapper


class ReconcileReceipt:
    def __init__(self, *, clock, lookback_days: int):
        self.clock = clock                      # the reconcile run's own tz-aware clock
        self.run_id = uuid.uuid4().hex
        self.lookback_days = lookback_days
        self.started_at = clock()
        self.finished_at: datetime | None = None
        self.days: list[dict] = []
        self.corrections: list[dict] | None = None
        self.replay: dict = {"state": "not_reached"}
        self.outcome = "incomplete"
        self.error: str | None = None
        self.degraded: list[str] = []
        self._day: dict | None = None
        self._slot: dict | None = None

    # ---- per date -------------------------------------------------------------------------------------------------
    @_guarded
    def day(self, date: str, cutoff: datetime, state: str, **detail) -> None:
        self._day = {"date": date, "cutoff_at": cutoff.isoformat(), "state": state, "detail": detail or None,
                     "selection": None, "status_source": None, "slots": [], "write": None}
        self._slot = None
        self.days.append(self._day)

    @_guarded
    def attempt(self, daily) -> None:
        from bts.picks import iter_daily_pick_slots
        self._day["state"] = "attempted"
        self._day["selection"] = {"result_before": daily.result, "slot_results_before": daily.slot_results,
                                  "slots": [{"slot": k, "batter_id": p.batter_id, "batter_name": p.batter_name,
                                             "game_pk": p.game_pk} for k, p in iter_daily_pick_slots(daily)]}

    @_guarded
    def day_state(self, state: str) -> None:
        self._day["state"] = state

    @_guarded
    def write(self, date: str, state: str, **detail) -> None:
        day = next(d for d in self.days if d["date"] == date)
        day["write"] = {"state": state, **detail}

    # ---- sources (called by bts.picks._fetch_json through the context observer) ----------------------------------
    @_guarded
    def source(self, url: str, raw) -> None:
        body = raw.encode() if isinstance(raw, str) else bytes(raw)
        entry = {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "completed_at": self.clock().isoformat()}
        if self._slot is not None:
            self._slot["sources"].append(entry)
            self._slot["response_completed_at"] = entry["completed_at"]
        elif self._day is not None:
            self._day["status_source"] = {"ok": True, **entry}

    @_guarded
    def status_failed(self, exc: BaseException) -> None:
        self._day["status_source"] = {"ok": False, "error": type(exc).__name__}

    # ---- per slot -------------------------------------------------------------------------------------------------
    @_guarded
    def slot_start(self, slot: str, pick) -> None:
        self._slot = {"slot": slot, "batter_id": pick.batter_id, "game_pk": pick.game_pk, "state": "in_progress",
                      "result": None, "basis": None, "sources": [], "response_completed_at": None, "covered": False,
                      "error": None}
        self._day["slots"].append(self._slot)

    @_guarded
    def slot_done(self, result, schedule_void_state: str | None) -> None:
        try:
            self._slot_done(result, schedule_void_state)
        finally:
            self._slot = None

    def _slot_done(self, result, schedule_void_state: str | None) -> None:
        s = self._slot
        s["result"] = result
        if result is None:
            s["state"] = "pending"
        else:
            s["state"] = "observed"
            if schedule_void_state is not None:
                s["basis"] = f"schedule_void_state:{schedule_void_state}"
            elif result == "void":
                s["basis"] = "suspended_no_evaluable_pa"
            else:
                s["basis"] = "final_feed" if len(s["sources"]) <= 1 else "final_feed_fallback_search"
            done = s["response_completed_at"]
            if done is None:                    # a schedule void needs no feed: observed with the status fetch
                done = (self._day.get("status_source") or {}).get("completed_at")
                s["response_completed_at"] = done
            s["covered"] = done is not None and datetime.fromisoformat(done) < datetime.fromisoformat(
                self._day["cutoff_at"])

    @_guarded
    def slot_failed(self, exc: BaseException) -> None:
        self._slot["state"] = "failed"
        self._slot["error"] = type(exc).__name__
        self._day["state"] = "failed"
        self._slot = None

    @_guarded
    def not_attempted(self, slots) -> None:
        for slot, pick in slots:
            self._day["slots"].append({"slot": slot, "batter_id": pick.batter_id, "game_pk": pick.game_pk,
                                       "state": "not_attempted", "result": None, "basis": None, "sources": [],
                                       "response_completed_at": None, "covered": False, "error": None})

    @_guarded
    def late_answer(self) -> None:
        """The day's answer arrived at or after its cutoff: no slot of it is coverage."""
        self._day["state"] = "late_answer"
        for s in self._day["slots"]:
            s["covered"] = False

    # ---- the run --------------------------------------------------------------------------------------------------
    @_guarded
    def replayed(self, replay) -> None:
        self.replay = ({"state": "unavailable"} if replay is None
                       else {"state": "saved", "streak": replay[0], "saver_available": replay[1]})

    @_guarded
    def finish(self, corrections) -> None:
        self.corrections = corrections
        self.outcome = "completed"
        self.finished_at = self.clock()

    @_guarded
    def raised(self, exc: BaseException) -> None:
        self.outcome = "raised"
        self.error = type(exc).__name__
        self.finished_at = self.clock()

    def record(self) -> dict:
        from bts.picks import _git_head_sha
        iso = lambda t: t.isoformat() if t is not None else None  # noqa: E731
        return {"schema": SCHEMA, "run_id": self.run_id,
                "producer": {"command": "reconcile", "revision": _git_head_sha(REPO)},
                "lookback_days": self.lookback_days, "started_at": iso(self.started_at),
                "finished_at": iso(self.finished_at), "outcome": self.outcome, "error": self.error,
                "days": self.days, "corrections": self.corrections, "replay": self.replay,
                "degraded": self.degraded, "published_at": iso(self.clock())}

    def publish(self, picks_dir: Path) -> Path | None:
        try:
            rec = self.record()
            stamp = self.started_at.strftime("%Y%m%dT%H%M%S%f")
            path = receipts_root(picks_dir) / self.started_at.date().isoformat() / f"{stamp}-{self.run_id}.json"
            receipt_io.publish(path, (json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n").encode())
            return path
        except Exception as exc:  # noqa: BLE001 - publication must never change the run's outcome
            print(f"reconcile: reconcile receipt unavailable ({type(exc).__name__})", file=sys.stderr)
            return None
