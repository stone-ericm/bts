"""The end-of-day outcome check: every day must end in a known state (2026-11-13).

A narrow replacement for the deferred C1 rank-2 watchdog. It reads the day's decision record
(`data/picks/<date>/decision.json`) and the entry checker's sealed receipts
(`docs/ops/pick-entry-receipt-v1.md`), names the day's end state, and reports anything but a clean one. It keeps no
state of its own and never writes pick, decision, streak or health state: an alert is repeated only as often as
cron runs the check.

Clean states: `no_games`, `deliberate_skip`, `entered` or `entered_game_unverified` (intent "enter"),
`research_commit` (intent "research"). The game-unverified state is a batter-only match, not a game confirmation.
Everything else is a problem. Missing or failed evidence is reported as unverifiable, never as a quiet pass, and
"not entered" is claimed only from a successful observation of the account (the receipt's consumer rules).

Limits: an entry made on a day the policy skipped is not seen (the entry checker observes the account only for a
committed pick); a confirmation proves the entry at its observation time, not through the cutoff; grading and
streak agreement are other checks' business (`result_resolution`, `streak_validation`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as calendar_date, datetime
from pathlib import Path

from bts.contest_state import DATED_MANUAL_SCHEMA, ContestStateError, validate_dated_observation
from bts.saver_state import SaverState, policy_saver_available
from bts.season_contract import ET, validate_contract
from bts.util import read_regular_file_text

OK_STATES = frozenset({"no_games", "deliberate_skip", "entered", "entered_game_unverified", "research_commit"})
ABSENT_REASONS = frozenset({"no_pick", "mismatch"})
EVIDENCE_MAX_BYTES = 64 * 1024


@dataclass(frozen=True)
class DayOutcome:
    date: str
    state: str
    detail: str
    pick: str | None = None

    @property
    def ok(self) -> bool:
        return self.state in OK_STATES

    def message(self) -> str:
        head = f"BTS day check {self.date}: {self.state.replace('_', ' ').upper()}"
        pick = f" Pick: {self.pick}." if self.pick else ""
        return f"{head}. {self.detail}{pick}"

    def as_dict(self) -> dict:
        return {"date": self.date, "state": self.state, "ok": self.ok, "detail": self.detail, "pick": self.pick}


def _pick_text(decision: dict) -> str | None:
    legs = ("primary", "double_down") if decision.get("action") == "double" else ("primary",)
    names = []
    for key in legs:
        summary = decision.get(key)
        name = summary.get("batter_name") if isinstance(summary, dict) else None
        if isinstance(name, str) and name.strip():
            names.append(name)
    return " + ".join(names) if names else None


def _binds(rec: dict, *, date: str, decision_sha256: str | None, expected_username: str | None) -> bool:
    """An `observed` receipt that is about THIS day's committed selection on the expected account, with no failed
    hooks (consumer rules: account, date and season, selection, empty `degraded`)."""
    commit = ((rec.get("selection") or {}).get("commit") or {})
    return (rec.get("outcome") == "observed"
            and rec.get("et_date") == date and rec.get("season") == int(date[:4])
            and decision_sha256 is not None and commit.get("decision_sha256") == decision_sha256
            and expected_username is not None
            and (rec.get("account") or {}).get("username") == expected_username
            and not rec.get("degraded"))


def _entry_state(receipts: list[dict], *, date: str, decision_sha256: str | None,
                 expected_username: str | None) -> tuple[str, str]:
    bound = [r for r in receipts
             if _binds(r, date=date, decision_sha256=decision_sha256, expected_username=expected_username)]
    confirmed = [r for r in bound if (r.get("qualification") or {}).get("all_confirmed") is True]
    in_time = [r for r in confirmed if (r.get("observation") or {}).get("before_cutoff") is True]
    if in_time:
        return "entered", f"The entry was confirmed in {len(in_time)} of {len(receipts)} checks before the cutoff."
    batter_only = [r for r in bound
                   if (r.get("observation") or {}).get("before_cutoff") is True
                   and (r.get("verifier") or {}).get("reason") == "match"
                   and (r.get("qualification") or {}).get("slots")
                   and all(s.get("state") in ("confirmed", "unit_unverified")
                           for s in r["qualification"]["slots"])]
    if batter_only:
        return "entered_game_unverified", "The batter entry matched before the cutoff; the game was not verified."
    if confirmed:
        at = min(str((r.get("observation") or {}).get("response_completed_at")) for r in confirmed)
        return "entered_late", f"The entry was first confirmed at {at}, after the cutoff."
    absent = [r for r in bound if (r.get("verifier") or {}).get("reason") in ABSENT_REASONS]
    if absent:
        return "not_entered", (f"Not entered: {len(absent)} successful account checks saw no matching entry and "
                               f"none confirmed one.")
    kinds: dict[str, int] = {}
    for r in receipts:
        kinds[str(r.get("outcome"))] = kinds.get(str(r.get("outcome")), 0) + 1
    seen = ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())) or "no receipts at all"
    return "entry_unverifiable", (f"No check both observed the account and confirmed this pick, and none observed "
                                  f"its absence ({seen}). Whether it was entered is unknown.")


def classify_day(*, date: str, games: int | None, decision: dict | None, decision_sha256: str | None,
                 receipts: list[dict], intent: str | None, expected_username: str | None,
                 void_games: int = 0) -> DayOutcome:
    """Name the day's end state. `games` is the number of regular-season games scheduled that date, or None when the
    schedule could not be read; `void_games` counts postponed/cancelled games. `receipts` are the day's SEALED
    entry receipts (`entry_receipt.discover`)."""
    def out(state, detail, pick=None):
        return DayOutcome(date=date, state=state, detail=detail, pick=pick)

    if intent not in ("enter", "research"):
        return out("intent_unknown", "scheduler.entry_intent is missing or invalid, so what the day should look "
                                     "like is undefined.")
    if games is None:
        return out("schedule_unverifiable", "The schedule could not be read, so whether a pick was due is unknown.")
    if games > 0 and void_games == games:
        return out("no_games", f"All {games} scheduled games were postponed or cancelled.")
    committed = decision is not None and decision.get("action") in ("single", "double")
    pick = _pick_text(decision) if committed else None
    if games == 0:
        if committed:
            return out("contradiction", "A pick was committed on a date with no regular-season games.", pick)
        return out("no_games", "No regular-season games.")
    if decision is None:
        return out("no_decision", f"{games} games were scheduled and the day has no decision record: no pick was "
                                  f"committed and no skip was decided.")
    if decision.get("action") == "skip":
        if decision.get("source") == "mdp":
            return out("deliberate_skip", "The policy skipped the day.")
        return out("skip_not_deliberate", f"The day ended as a skip decided by {decision.get('source')!r}, not by "
                                          f"the policy.")
    if not committed:
        return out("no_decision", f"The decision record's action is {decision.get('action')!r}.")
    if not decision.get("scoreable"):
        return out("commit_not_scoreable", f"A pick was chosen but not delivered (delivery status "
                                           f"{decision.get('delivery_status')!r}).", pick)
    if intent == "research":
        if decision.get("delivery_status") == "private_locked":
            return out("research_commit", "Research mode: the pick was saved privately.", pick)
        return out("delivered_under_research", f"Research mode, but the pick's delivery status is "
                                               f"{decision.get('delivery_status')!r}.", pick)
    state, detail = _entry_state(receipts, date=date, decision_sha256=decision_sha256,
                                 expected_username=expected_username)
    return out(state, detail, pick)


def _read_evidence(path):
    text = read_regular_file_text(path)
    if len(text.encode("utf-8")) > EVIDENCE_MAX_BYTES:
        raise ValueError("oversize evidence")
    return json.loads(text)


def _dated_evidence(account, day):
    paths = [account / "contest_streak.manual.json"]
    try:
        paths.extend(sorted(account.glob("contest_streak.manual.revoked-*.json")))
    except OSError:
        pass
    for path in paths:
        try:
            raw = _read_evidence(path)
            if not isinstance(raw, dict) or raw.get("schema_version") != DATED_MANUAL_SCHEMA:
                continue
            item = validate_dated_observation(raw, path)
            if item.effective_date != day:
                continue
            yield (
                datetime.fromisoformat(item.recorded_at),
                (item.streak, policy_saver_available(SaverState(item.saver_state, day.year), item.streak)),
            )
        except (OSError, ContestStateError, ValueError, RecursionError, OverflowError):
            continue


def _revoke_on_day(account, day):
    try:
        switch = validate_contract(_read_evidence(account / "season_contract.json"))
        if switch["season"] != day.year or switch["revoked_at"] is None:
            return None
        stamp = datetime.fromisoformat(switch["revoked_at"]).astimezone(ET)
        return stamp if stamp.date() == day else None
    except (OSError, ValueError, RecursionError, OverflowError):
        return None


def _availability(available):
    return "available" if available else "unavailable"


def _report_skip_evidence(outcome, decision, picks_dir, intent):
    streak, available = decision.get("streak"), decision.get("saver_available")
    if type(streak) is not int or streak < 0 or type(available) is not bool:
        return outcome
    try:
        day = calendar_date.fromisoformat(outcome.date)
        if day.isoformat() != outcome.date:
            return outcome
    except ValueError:
        return outcome
    account = Path(picks_dir) / "account_state"
    dated = list(_dated_evidence(account, day))
    skip = (streak, available)
    reasons = []
    if dated:
        _, latest = max(dated, key=lambda item: item[0])
        if latest != skip:
            reasons.append(
                f"A dated observation now on file for {day} gives streak {latest[0]} "
                f"and policy saver availability {_availability(latest[1])}."
            )
    if intent == "enter" and all(values != skip for _, values in dated):
        revoke = _revoke_on_day(account, day)
        if revoke is not None:
            reasons.append(f"The season switch records a revocation on {day} at {revoke:%H:%M} ET.")
    if not reasons:
        return outcome
    opening = f"The policy recorded a skip for {day} using streak {streak} and policy saver availability {_availability(available)}."
    closing = (
        "Check MLB for unsettled entries and verify the current streak, season best and saver. "
        "Follow checklist row B8a for the current ET date and year before changing any state."
    )
    return DayOutcome(outcome.date, "skip_state_changed", " ".join([opening, *reasons, closing]))


def gather(date: str, picks_dir, config: dict, *, expected_username: str | None, fetch_schedule=None) -> DayOutcome:
    """Read the day's evidence and classify it. Read-only: the schedule (one public request), the decision record and
    the sealed entry receipts. A schedule failure is reported as unverifiable, not raised."""
    from bts import entry_receipt
    from bts.daily_decision import load_decision_bytes
    from bts.entry_intent import check_entry_intent
    from bts.picks import _is_void_detailed_state
    if fetch_schedule is None:
        from bts.scheduler import fetch_schedule
    void_games = 0
    try:
        schedule = fetch_schedule(date)
        games = len(schedule)
        void_games = sum(_is_void_detailed_state((g.get("status") or {}).get("detailedState")) for g in schedule)
    except Exception:
        games = None
    decision, raw = load_decision_bytes(date, picks_dir)
    intent = check_entry_intent(config, include_live_gate=False)
    outcome = classify_day(date=date, games=games, void_games=void_games,
                           decision=decision, decision_sha256=entry_receipt._sha(raw),
                           receipts=entry_receipt.discover(picks_dir, date),
                           intent=intent.intent if intent.ok else None, expected_username=expected_username)
    if outcome.state == "deliberate_skip":
        return _report_skip_evidence(outcome, decision, picks_dir, intent.intent)
    return outcome
