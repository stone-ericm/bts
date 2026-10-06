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


WRITE_HOOKS = ("write", "write_intent", "write_done")


def _guarded(fn):
    """A recording failure degrades the receipt and never reaches the reconcile run. It is attributed to the date
    being resolved, or, for a write hook, to its explicit date argument (producer review r3 D6)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as exc:  # noqa: BLE001 - the receipt must never change the run
            day, slot = self._day_ix, self._slot_ix
            if fn.__name__ in WRITE_HOOKS:
                day = next((i for i, d in enumerate(self.days) if a and d["date"] == a[0]), None)
                slot = None
            self.degraded.append({"hook": fn.__name__, "error": type(exc).__name__, "day": day, "slot": slot})
            return None
    return wrapper


def _parse(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class ReconcileReceipt:
    """Live hooks only record events (producer review r2 D5): references to the held response bytes and the instant
    each response returned, tagged with the date and slot being resolved. Hashing, parsing and every coverage
    judgement happen in record(), at publication, after the run's own decisions and writes."""

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
        self.degraded: list[dict] = []
        self._events: list[tuple] = []          # (day_ix, slot_ix, url, raw, returned_at)
        self._day_ix: int | None = None
        self._slot_ix: int | None = None

    # ---- live hooks: events only -----------------------------------------------------------------------------------
    @_guarded
    def day(self, date: str, cutoff: datetime, state: str, **detail) -> None:
        self._day_ix, self._slot_ix = len(self.days), None
        self.days.append({"date": date, "cutoff": cutoff, "state": state, "detail": detail or None,
                          "selection": None, "status_failed": None, "slots": [], "write": None})

    @_guarded
    def attempt(self, daily) -> None:
        from bts.picks import iter_daily_pick_slots
        day = self.days[self._day_ix]
        day["state"] = "attempted"
        day["selection"] = {"result_before": daily.result, "slot_results_before": daily.slot_results,
                            "slots": [{"slot": k, "batter_id": p.batter_id, "batter_name": p.batter_name,
                                       "game_pk": p.game_pk} for k, p in iter_daily_pick_slots(daily)]}

    @_guarded
    def day_state(self, state: str) -> None:
        self.days[self._day_ix]["state"] = state

    @_guarded
    def source(self, url: str, raw) -> None:
        """Called by bts.picks._fetch_json the moment a response's bytes are read: the return instant and the held
        bytes, nothing else."""
        self._events.append((self._day_ix, self._slot_ix, url, raw, self.clock()))

    @_guarded
    def status_failed(self, exc: BaseException) -> None:
        self.days[self._day_ix]["status_failed"] = type(exc).__name__

    @_guarded
    def slot_start(self, slot: str, pick) -> None:
        day = self.days[self._day_ix]
        self._slot_ix = len(day["slots"])
        day["slots"].append({"slot": slot, "batter_id": pick.batter_id, "batter_name": pick.batter_name,
                             "game_pk": pick.game_pk, "state": "in_progress", "result": None,
                             "schedule_void_state": None, "error": None})

    @_guarded
    def slot_done(self, result, schedule_void_state: str | None) -> None:
        s = self.days[self._day_ix]["slots"][self._slot_ix]
        s["result"], s["schedule_void_state"] = result, schedule_void_state
        s["state"] = "pending" if result is None else "observed"
        self._slot_ix = None

    @_guarded
    def slot_failed(self, exc: BaseException) -> None:
        day = self.days[self._day_ix]
        day["slots"][self._slot_ix].update(state="failed", error=type(exc).__name__)
        day["state"] = "failed"
        self._slot_ix = None

    @_guarded
    def not_attempted(self, slots) -> None:
        for slot, pick in slots:
            self.days[self._day_ix]["slots"].append(
                {"slot": slot, "batter_id": pick.batter_id, "batter_name": pick.batter_name, "game_pk": pick.game_pk,
                 "state": "not_attempted", "result": None, "schedule_void_state": None, "error": None})

    @_guarded
    def late_answer(self) -> None:
        self.days[self._day_ix]["state"] = "late_answer"

    # ---- writes (D4): an intent before the save, completion only after it ----------------------------------------
    def _day_of(self, date: str) -> dict:
        return next(d for d in self.days if d["date"] == date)

    @_guarded
    def write(self, date: str, state: str) -> None:
        """A write decision that saves nothing (refused_after_cutoff, skipped_under_lock)."""
        self._day_of(date)["write"] = {"intended": state, "completed": True, "written_selection": None}

    @_guarded
    def write_intent(self, date: str, state: str, daily, **detail) -> None:
        """Before the save: what will be written, to which selection as re-read under the scoring lock."""
        from bts.picks import iter_daily_pick_slots
        self._day_of(date)["write"] = {
            "intended": state, "completed": False, **detail,
            "written_selection": [{"slot": k, "batter_id": p.batter_id, "game_pk": p.game_pk}
                                  for k, p in iter_daily_pick_slots(daily)]}

    @_guarded
    def write_done(self, date: str) -> None:
        self._day_of(date)["write"]["completed"] = True

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

    # ---- the record (publication time) ----------------------------------------------------------------------------
    @staticmethod
    def _source(url, raw, at) -> dict:
        body = raw.encode() if isinstance(raw, str) else bytes(raw)
        return {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "returned_at": at.isoformat()}

    @staticmethod
    def _has_batter_id(resp: dict, batter_id) -> bool:
        """The batter by bound id, never by name (producer review r3 D3): a boxscore entry or a play's matchup."""
        teams = ((resp.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        if any(f"ID{batter_id}" in ((teams.get(side) or {}).get("players") or {}) for side in ("away", "home")):
            return True
        plays = ((resp.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
        return any(((p.get("matchup") or {}).get("batter") or {}).get("id") == batter_id for p in plays)

    @staticmethod
    def _qualify(slot: dict, feeds: list, status) -> tuple[str, int | None]:
        """The basis the consumed payloads themselves establish, and the actual decisive game (D3).
        - **Non-void:** a Final feed that holds the batter **by id**, grades to that result, and names its own
          game (`gameData.game.pk`) equal to the selected game.
        - **Suspended void:** the same, including the game agreement.
        - **Schedule void:** the consumed schedule must list the slot's game in that void state.

        No identity is taken from a request URL or a name match."""
        from bts.picks import _is_void_detailed_state, grade_pick_in_feed
        if slot["schedule_void_state"] is not None:
            games = [g for d in ((status or {}).get("dates") or []) for g in (d.get("games") or [])
                     if g.get("gamePk") == slot["game_pk"]]
            ok = len(games) == 1 and _is_void_detailed_state((games[0].get("status") or {}).get("detailedState"))
            return (f"schedule_void_state:{slot['schedule_void_state']}" if ok else "unqualified"), slot["game_pk"]
        for url, resp in feeds:
            if not isinstance(resp, dict):
                continue
            try:
                final = resp["gameData"]["status"]["abstractGameCode"] == "F"
                actual = (resp["gameData"].get("game") or {}).get("pk")
                grade = grade_pick_in_feed(resp, slot["batter_id"], slot["batter_name"])
                by_id = ReconcileReceipt._has_batter_id(resp, slot["batter_id"])
            except (KeyError, TypeError, AttributeError):
                continue
            if grade is None:
                continue                         # the batter is not in this feed
            if type(actual) is not int or not by_id or not final or grade != slot["result"]:
                return "unqualified", actual if type(actual) is int else None
            if actual != slot["game_pk"]:
                return "fallback_other_game", actual
            return ("suspended_no_evaluable_pa" if slot["result"] == "void" else "final_feed"), actual
        return "unqualified", None

    def _day_record(self, ix: int, day: dict) -> dict:
        cutoff = day["cutoff"]
        events = [e for e in self._events if e[0] == ix]
        degraded = [g for g in self.degraded if g["day"] == ix]
        # D1 (r3): any recording failure on this date may hide an observation in some slot's prefix, so it
        # uncovers every slot of the date
        observation_degraded = [g for g in degraded if g["hook"] not in WRITE_HOOKS]
        status_ev = [e for e in events if e[1] is None]
        status = _parse(status_ev[-1][3]) if status_ev else None
        out = {"date": day["date"], "cutoff_at": cutoff.isoformat(), "state": day["state"], "detail": day["detail"],
               "selection": day["selection"],
               "status_source": ({"ok": False, "error": day["status_failed"]} if day["status_failed"]
                                 else {"ok": True, **self._source(*status_ev[-1][2:])} if status_ev else None),
               "write": None, "slots": []}
        # D2: the day's response instants in order; a step back anywhere, or any instant at/after the cutoff, is
        # irreversible for every slot resolved from that point on.
        times = [e[4] for e in events]
        regression = any(b < a for a, b in zip(times, times[1:]))
        for sx, slot in enumerate(day["slots"]):
            evs = [e for e in events if e[1] == sx]
            sources = [self._source(*e[2:]) for e in evs]
            feeds = [(e[2], _parse(e[3])) for e in evs if "/feed/live" in e[2]]
            basis, actual = (None, None)
            if slot["state"] == "observed":
                basis, actual = self._qualify(slot, feeds, status)
            # every response the slot's resolution could have depended on: the day's status fetch and every
            # source of this and the earlier slots
            relevant = [e[4] for e in events if e[1] is None or e[1] <= sx]
            high_water = max(relevant) if relevant else None
            last = evs[-1][4] if evs else (status_ev[-1][4] if (status_ev and slot["schedule_void_state"]) else None)
            slot_degraded = observation_degraded
            qualified = basis in ("final_feed", "suspended_no_evaluable_pa") or (
                basis is not None and basis.startswith("schedule_void_state:"))
            covered = bool(slot["state"] == "observed" and day["state"] == "observed" and qualified
                           and high_water is not None and high_water < cutoff and not regression
                           and not slot_degraded)
            out["slots"].append({
                "slot": slot["slot"], "batter_id": slot["batter_id"], "game_pk": slot["game_pk"],
                "state": slot["state"], "result": slot["result"], "basis": basis, "actual_game_pk": actual,
                "sources": sources, "response_completed_at": last.isoformat() if last else None,
                "high_water_at": high_water.isoformat() if high_water else None, "clock_regression": regression,
                "degraded": slot_degraded, "covered": covered, "error": slot["error"]})
        # D6 (r3): a save that raised is write_not_completed; missing completion *evidence* (a failed write hook,
        # or a run that did not raise) is write_evidence_unavailable, never a claim about the save
        w = day["write"]
        write_degraded = [g for g in degraded if g["hook"] in WRITE_HOOKS]
        if w is not None:
            w = dict(w)
            observed = [{"slot": s["slot"], "batter_id": s["batter_id"], "game_pk": s["game_pk"]}
                        for s in (day["selection"] or {}).get("slots", [])]
            if w["completed"] and not write_degraded:
                w["state"] = w["intended"]
            elif write_degraded or self.outcome != "raised":
                w["state"] = "write_evidence_unavailable"
            else:
                w["state"] = "write_not_completed"
            if w.get("written_selection") is not None:
                w["selection_changed"] = w["written_selection"] != observed
            out["write"] = w
        elif write_degraded:
            out["write"] = {"state": "write_evidence_unavailable"}
        return out

    def record(self) -> dict:
        from bts.picks import _git_head_sha
        iso = lambda t: t.isoformat() if t is not None else None  # noqa: E731
        return {"schema": SCHEMA, "run_id": self.run_id,
                "producer": {"command": "reconcile", "revision": _git_head_sha(REPO)},
                "lookback_days": self.lookback_days, "started_at": iso(self.started_at),
                "finished_at": iso(self.finished_at), "outcome": self.outcome, "error": self.error,
                "days": [self._day_record(i, d) for i, d in enumerate(self.days)],
                "corrections": self.corrections, "replay": self.replay,
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
