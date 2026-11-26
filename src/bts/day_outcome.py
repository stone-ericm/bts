"""The end-of-day outcome check: every day must end in a known state (2026-11-13).

A narrow replacement for the deferred C1 rank-2 watchdog. It reads the day's decision record
(`data/picks/<date>/decision.json`) and the entry checker's sealed receipts
(`docs/ops/pick-entry-receipt-v1.md`), names the day's end state, and reports anything but a clean one. It keeps no
state of its own and never writes pick, decision, streak or health state: an alert is repeated only as often as
cron runs the check.

Clean states: `no_games`, `deliberate_skip`, `entered` or `entered_game_unverified` (intent "enter"),
`research_commit` (intent "research"). The game-unverified state is a batter-only match, not a game confirmation.
Everything else is a problem. "not entered" is claimed only from a successful observation of the account
(the receipt's consumer rules).

A deliberate MDP skip can prompt `skip_state_review` if the selected dated observation for that day records
a different streak or policy saver availability. The report uses a valid installed observation when present
and otherwise checks retained archives. In enter mode it also prompts review if the switch remains revoked
for that season, the revocation was on the skip date in ET, and no valid retained dated observation matches
both inputs.

Missing or unusable optional contest evidence can leave a deliberate skip clean. The base outcome checks
report missing schedule, decision or entry evidence where required.

Limits: an entry made on a day the policy skipped is not seen (the entry checker observes the account only for a
committed pick); a confirmation proves the entry at its observation time, not through the cutoff; grading and
streak agreement are other checks' business (`result_resolution`, `streak_validation`).
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, replace
from datetime import date as Date, datetime
from pathlib import Path

from bts.contest_state import DATED_MANUAL_SCHEMA, ContestStateError, validate_dated_observation
from bts.saver_state import SaverState, policy_saver_available
from bts.season_contract import ET, InvalidContract, validate_contract
from bts.util import FileReadLimitError, NotRegularFileError, read_regular_file_text

OK_STATES = frozenset({"no_games", "deliberate_skip", "entered", "entered_game_unverified", "research_commit"})
ABSENT_REASONS = frozenset({"no_pick", "mismatch"})
EVIDENCE_MAX_BYTES = 64 * 1024
STREAK_MAX_BYTES = 1024 * 1024
DM_MAX_CODE_POINTS = 1000


@dataclass(frozen=True)
class DayOutcome:
    date: str
    state: str
    detail: str
    pick: str | None = None
    streak_problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.state in OK_STATES and self.streak_problem is None

    def message(self) -> str:
        title = self.state.replace("_", " ").upper()
        if self.streak_problem is not None and self.state in OK_STATES:
            title = "MODEL STREAK FILE NEEDS REVIEW"
        head = f"BTS day check {self.date}: {title}"
        pick = f" Pick: {self.pick}." if self.pick else ""
        body = f"{head}. {self.detail}{pick}"
        if self.streak_problem is None:
            return body
        body += (
            f" Also: The local model streak file (streak.json) needs review ({self.streak_problem}). "
            "Grading or replay may be affected. Preserve the files in place during triage. Follow B8b item 2 "
            "in docs/ops/2027-season-start.md before the next affected job. Give the BTS lead this alert, "
            "the host and ET run time, and any grading or reconcile errors, receipt paths and health alerts. "
            "If the lead session is unavailable, escalate through Eric. The 08:00 ET correction cutoff still applies."
        )
        if len(body) > DM_MAX_CODE_POINTS:
            return (
                f"{head}. Local model streak file needs review ({self.streak_problem}). Preserve the files. "
                "See day-outcome JSON in cron.log and B8b item 2 in docs/ops/2027-season-start.md before the "
                "next affected job. Escalate through Eric if needed."
            )
        return body

    def as_dict(self) -> dict:
        return {"date": self.date, "state": self.state, "ok": self.ok, "detail": self.detail, "pick": self.pick,
                "streak_problem": self.streak_problem}


def _streak_problem(path: Path) -> str | None:
    try:
        text = read_regular_file_text(path, max_bytes=STREAK_MAX_BYTES)
    except FileNotFoundError:
        return None
    except NotRegularFileError:
        return "not a regular file"
    except FileReadLimitError:
        return "exceeds 1 MiB"
    except (OSError, UnicodeDecodeError) as exc:
        return f"cannot be read ({type(exc).__name__})"

    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return "not valid JSON"
    except (ValueError, RecursionError) as exc:
        return f"JSON exceeds decoder limits ({type(exc).__name__})"

    if not isinstance(record, dict):
        return "not a JSON object"
    streak = record.get("streak", 0)
    if type(streak) is not int or streak < 0:
        return "streak is not a non-negative integer"
    return None


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


def _read_evidence(path: Path) -> object:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            raw = bytearray()
            while len(raw) <= EVIDENCE_MAX_BYTES:
                chunk = os.read(fd, EVIDENCE_MAX_BYTES + 1 - len(raw))
                if not chunk:
                    return json.loads(raw.decode("utf-8"))
                raw.extend(chunk)
        finally:
            os.close(fd)
    except (OSError, ValueError, RecursionError):
        pass
    return None


def _dated_evidence(path: Path, day: Date) -> tuple[datetime, tuple[int, bool]] | None:
    data = _read_evidence(path)
    if not isinstance(data, dict) or data.get("schema_version") != DATED_MANUAL_SCHEMA:
        return None
    try:
        observation = validate_dated_observation(data, path)
    except ContestStateError:
        return None
    if observation.effective_date != day:
        return None
    recorded_at = datetime.fromisoformat(observation.recorded_at)
    try:
        recorded_at.astimezone(ET)
    except OverflowError:
        return None
    available = policy_saver_available(SaverState(state=observation.saver_state, season=day.year),
                                       observation.streak)
    return recorded_at, (observation.streak, available)


def _archives(account: Path):
    try:
        with os.scandir(account) as entries:
            for entry in entries:
                if entry.name.startswith("contest_streak.manual.revoked-") and entry.name.endswith(".json"):
                    yield Path(entry.path)
    except OSError:
        return


def _revocation_time(account: Path, day: Date) -> datetime | None:
    try:
        switch = validate_contract(_read_evidence(account / "season_contract.json"))
        if switch["season"] == day.year and switch["status"] == "revoked":
            revoked = datetime.fromisoformat(switch["revoked_at"]).astimezone(ET)
            if revoked.date() == day:
                return revoked
    except (InvalidContract, OverflowError):
        pass
    return None


def _availability(available: bool) -> str:
    return "available" if available else "unavailable"


def _skip_review(date: str, picks_dir, decision: dict, intent: str) -> DayOutcome | None:
    streak = decision.get("streak")
    saver = decision.get("saver_available")
    if (decision.get("date") != date or decision.get("scoreable") is not False
            or type(streak) is not int or streak < 0 or type(saver) is not bool):
        return None
    try:
        day = Date.fromisoformat(date)
    except ValueError:
        return None

    account = Path(picks_dir) / "account_state"
    skip_inputs = (streak, saver)
    current = _dated_evidence(account / "contest_streak.manual.json", day)
    revoked = _revocation_time(account, day) if intent == "enter" else None
    matching_dated = current is not None and current[1] == skip_inputs
    newest = None
    newest_inputs: set[tuple[int, bool]] = set()
    if current is None or (revoked is not None and not matching_dated):
        for path in _archives(account):
            archive = _dated_evidence(path, day)
            if archive is None:
                continue
            recorded_at, inputs = archive
            matching_dated |= inputs == skip_inputs
            if newest is None or recorded_at > newest:
                newest = recorded_at
                newest_inputs = {inputs}
            elif recorded_at == newest:
                newest_inputs.add(inputs)

    reasons = []
    if current is not None:
        dated_inputs = current[1]
        label = "The installed dated observation"
    else:
        dated_inputs = next(iter(newest_inputs)) if len(newest_inputs) == 1 else None
        label = "An archived dated observation"
    if dated_inputs is not None and dated_inputs != skip_inputs:
        observed_streak, observed_saver = dated_inputs
        reasons.append(f"{label} for {date} records streak {observed_streak} and policy saver "
                       f"availability {_availability(observed_saver)}.")
    if revoked is not None and not matching_dated:
        reasons.append(f"The season switch records a revocation on {date} at {revoked:%H:%M} ET.")
    if not reasons:
        return None
    detail = " ".join([
        f"The policy recorded a skip for {date} using streak {streak} and policy saver availability "
        f"{_availability(saver)}.",
        *reasons,
        "Check MLB for unsettled entries and verify the current streak, season best and saver.",
    ])
    return DayOutcome(date=date, state="skip_state_review", detail=detail)


def gather(date: str, picks_dir, config: dict, *, expected_username: str | None, fetch_schedule=None) -> DayOutcome:
    """Read the day's evidence and classify it. Read-only: the schedule (one public request), the decision record and
    the sealed entry receipts. A deliberate skip may also use contest state on file. Check the local model streak file
    after classifying the day.
    A schedule failure is reported as unverifiable, not raised."""
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
        outcome = _skip_review(date, picks_dir, decision, intent.intent) or outcome
    return replace(outcome, streak_problem=_streak_problem(Path(picks_dir) / "streak.json"))
