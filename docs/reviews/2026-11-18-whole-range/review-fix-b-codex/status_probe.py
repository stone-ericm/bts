"""Try to disprove the status counterfactual with unchanged on-disk input."""
import datetime as dt
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from probe_support import REVIEW, REV, NOW, ET, REAL, auto, bounded, dated, dump, fetch, invoke, offline, put, snapshot, switch
from bts.contest_state import automatic_reading_record, load_decision_streak_state


def verdict(picks, now):
    try:
        with bounded():
            state = load_decision_streak_state(picks, now=now, require_contest_state=True)
            reading = automatic_reading_record(state, now)
            return ["accepted", state.streak, state.status, state.saver_flag_state,
                    reading and reading["reason"]]
    except Exception as exc:
        return [type(exc).__name__, str(exc).splitlines()[0].replace(str(picks), "PICKS")]


rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="status-fixtures-") as directory:
    root = Path(directory)

    def evaluate(label, now=NOW, switch_body=True, auto_body=True, manual_body=None,
                 special=None, another=False, partial_fetch=False, settlement=False):
        picks = root / label / "picks"
        account = picks / "account_state"
        account.mkdir(parents=True)
        for name, body, default in [
            ("season_contract.json", switch_body, switch(now)),
            ("contest_streak.json", auto_body, auto(now)),
            ("contest_streak.manual.json", manual_body, dated(now)),
        ]:
            if body is not None:
                put(picks, name, default if body is True else body)
        if settlement:
            for offset, outcome in ((0, "hit"), (1, "miss")):
                path = picks / ((now.date()+dt.timedelta(days=offset)).isoformat()+".json")
                path.write_text('{"result": "' + outcome + '"}')
        if special:
            path, kind = special
            path = picks / path
            if path.exists():
                path.unlink()
            path.mkdir() if kind == "directory" else os.mkfifo(path)
        if partial_fetch:
            status = picks.parent / "health_state/contest_streak_fetch_status.json"
            status.mkdir(parents=True)
            with offline(now):
                failed = fetch(picks)
            assert failed.exit_code == (2 if REV == "head" else 1)
        before = snapshot(picks.parent)
        with offline(now), bounded():
            result = invoke(picks, "season-contract", "status",
                            *(("--season", str(now.year-1)) if another else ()))
        assert before == snapshot(picks.parent), label
        lines = result.output.replace(str(picks), "PICKS").splitlines()
        independent = [line for line in lines if line.startswith("automatic path alone:")]
        if REV == "head":
            assert len(independent) == 1, (label, lines, result.exception)
        live_now = verdict(picks, now)
        fallback = verdict(picks, now+dt.timedelta(days=1))
        rows[label] = dict(exit=result.exit_code, lines=lines, live=live_now, next_day=fallback)
        if REV == "head" and independent and not special and now.month != 12:
            headline = independent[0]
            assert ("alone: accepted" in headline) == (fallback[0] == "accepted"), rows[label]

    for manual in (None, True):
        for automatic in (True, auto(schema_version="bts_contest_streak_auto_v1")):
            evaluate(f"matrix-{manual}-{automatic is True}", auto_body=automatic, manual_body=manual)
    evaluate("no-auto-dated", auto_body=None, manual_body=True)
    evaluate("expired-dated", manual_body=dated(NOW-dt.timedelta(days=1)))
    evaluate("winning-legacy", manual_body={"active_streak": 3, "override_expires_at":
                                            (NOW+dt.timedelta(days=2)).isoformat()})
    evaluate("corrupt-manual", manual_body=b"{")
    evaluate("corrupt-auto", auto_body=b"[]", manual_body=True)
    evaluate("corrupt-switch", switch_body=b"\xff", manual_body=True)
    evaluate("wrong-id", auto_body=auto(activation_id="b"*32), manual_body=True)
    evaluate("old", auto_body=auto(recorded_at=(NOW-dt.timedelta(hours=13)).isoformat()))
    evaluate("future", auto_body=auto(recorded_at=(NOW+dt.timedelta(hours=2)).isoformat()))
    evaluate("missing-stamp", auto_body=auto(recorded_at=None))
    evaluate("naive-stamp", auto_body=auto(recorded_at="2027-07-15T11:30:00"))
    evaluate("incomplete-fetch", partial_fetch=True, manual_body=True)
    evaluate("settlement-gap", settlement=True)
    december = REAL(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    for crossing in (False, True):
        now = december + dt.timedelta(seconds=2*crossing)
        evaluate(f"new-year-{crossing}", now=now, auto_body=auto(december), switch_body=switch(december),
                 manual_body=dated(december), another=True)
    for path in ("account_state/contest_streak.manual.json", "account_state/contest_streak.json",
                 "account_state/season_contract.json", "account_state/saver_state.json",
                 "account_state/contest_ledger.jsonl", "streak.json", "2027-07-15.json"):
        for kind in ("fifo", "directory"):
            evaluate(f"{Path(path).name}-{kind}", special=(path, kind))

dump("status", rows)
