"""Compare the CLI's counterfactual to the ordinary required loader."""
from datetime import timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from probe_support import (
    ET, NOW, REAL, REVIEW, REV, Timeout, auto, bounded, dated, dump, fetch, invoke, offline,
    put, snapshot, switch,
)
from bts.contest_state import automatic_reading_record, load_decision_streak_state


def verdict(picks, now):
    try:
        with bounded():
            state = load_decision_streak_state(picks, now=now, require_contest_state=True)
    except (Exception, Timeout) as exc:
        return dict(outcome=type(exc).__name__, detail=str(exc).splitlines()[0])
    record = automatic_reading_record(state, now)
    return dict(outcome="accepted", kind=state.observation_kind, streak=state.streak,
                status=state.status, saver=state.saver_flag_state, model=state.model_state_error,
                stamp=record["recorded_at"] if record and record["reason"] != "unreadable" else "unavailable")


def decision_line(live):
    if live["outcome"] == "accepted":
        return f'automatic path alone: accepted (streak {live["streak"]}, recorded {live["stamp"]})'
    return f'automatic path alone: refused: {live["detail"]}'


rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="status-fixtures-") as directory:
    root = Path(directory)

    def evaluate(label, now=NOW, *, switch_body=True, auto_body=True, manual_body=None,
                 saver_body=None, special=None, inspection_year=None, partial_fetch=False,
                 settlement=False, omit=False):
        picks = root / label / "picks"
        (picks / "account_state").mkdir(parents=True)
        for name, body, default in [
            ("season_contract.json", switch_body, switch(now)),
            ("contest_streak.json", auto_body, auto(now)),
            ("contest_streak.manual.json", manual_body, dated(now)),
            ("saver_state.json", saver_body, {"season": now.year, "state": "used"}),
        ]:
            if body is not None:
                put(picks, name, default if body is True else body)
        if settlement:
            for offset, outcome in ((0, "hit"), (1, "miss")):
                day = now.date() + timedelta(days=offset)
                (picks / f"{day}.json").write_text('{"result": "' + outcome + '"}')
        if special:
            relative, kind = special
            path = picks / relative
            path.unlink(missing_ok=True)
            path.mkdir() if kind == "directory" else os.mkfifo(path)
        if partial_fetch:
            status = picks.parent / "health_state/contest_streak_fetch_status.json"
            status.mkdir(parents=True)
            with offline(now):
                failed = fetch(picks)
            assert failed.exit_code == (1 if REV == "base" else 2), failed.output
        before = snapshot(picks.parent)
        args = () if inspection_year is None else ("--season", str(inspection_year))
        with offline(now), bounded():
            result = invoke(picks, "season-contract", "status", *args)
        assert before == snapshot(picks.parent), label
        lines = result.output.splitlines()
        independent = [line for line in lines if line.startswith("automatic path alone:")]
        live_now = verdict(picks, now)
        manual = picks / "account_state/contest_streak.manual.json"
        if omit:
            manual.unlink()
        alone = verdict(picks, now) if omit else live_now
        bounded_out = isinstance(result.exception, Timeout)
        rows[label] = dict(exit=result.exit_code, blocked=bounded_out,
                           lines=result.output.replace(str(picks), "PICKS").splitlines(),
                           live_now=live_now, live_without_selected_dated=alone)
        if REV == "head" and not special:
            assert not bounded_out and independent == [decision_line(alone)], (label, rows[label])
            assert result.exit_code == int(switch_body == b"\xff"), (label, lines)
            assert any("expires at midnight ET" in line for line in lines) == (
                live_now.get("kind") == "dated" and alone["outcome"] != "accepted"), (label, lines)
        if REV == "head" and special:
            relative, kind = special
            if kind == "fifo":
                assert bounded_out and not independent, (label, rows[label])
                expected = "accepted" if relative.endswith("ledger.jsonl") else "Timeout"
                assert alone["outcome"] == expected, (label, rows[label])
            else:
                assert not bounded_out and independent == [decision_line(alone)], (label, rows[label])

    if REV == "head":
        for manual in (None, True):
            for automatic in (True, auto(schema_version="bts_contest_streak_auto_v1")):
                evaluate(f"matrix-{manual}-{automatic is True}", auto_body=automatic,
                         manual_body=manual, omit=bool(manual))
        evaluate("missing-auto-dated", auto_body=None, manual_body=True, omit=True)
        evaluate("expired-dated", manual_body=dated(NOW - timedelta(days=1)))
        evaluate("winning-legacy", manual_body={"active_streak": 3, "override_expires_at":
                                                (NOW + timedelta(days=2)).isoformat()})
        evaluate("corrupt-manual", manual_body=b"{")
        evaluate("corrupt-auto-unmasked", auto_body=b"[]")
        evaluate("corrupt-auto-masked", auto_body=b"[]", manual_body=True, omit=True)
        evaluate("corrupt-switch-unmasked", switch_body=b"\xff")
        evaluate("corrupt-switch-masked", switch_body=b"\xff", manual_body=True, omit=True)
        evaluate("wrong-id", auto_body=auto(activation_id="b"*32), manual_body=True, omit=True)
        for tag, stamp in {
            "old": (NOW - timedelta(hours=15)).isoformat(),
            "future": (NOW + timedelta(hours=2)).isoformat(), "missing": None,
            "naive": "2027-07-15T11:30:00",
        }.items():
            evaluate(tag, auto_body=auto(recorded_at=stamp))
        evaluate("uninitialized", saver_body={})
        evaluate("saver-read-raises", saver_body=b"[" * 3000 + b"]" * 3000,
                 manual_body=True, omit=True)
        evaluate("settlement-gap", settlement=True, saver_body=True)
        evaluate("incomplete-fetch", partial_fetch=True, manual_body=True, omit=True)
        evaluate("other-inspection-year", inspection_year=2026, saver_body=True)
        december = REAL(2027, 12, 31, 23, 59, 59, tzinfo=ET)
        for crossing in (False, True):
            now = december + timedelta(seconds=2*crossing)
            evaluate(f"new-year-{crossing}", now, auto_body=auto(december), switch_body=switch(december),
                     manual_body=dated(december), inspection_year=now.year-1, omit=not crossing)

    for relative in ("account_state/contest_streak.manual.json", "account_state/contest_streak.json",
                     "account_state/season_contract.json", "account_state/saver_state.json",
                     "account_state/contest_ledger.jsonl", "streak.json", "2027-07-15.json"):
        for kind in ("fifo", "directory"):
            evaluate(f"{Path(relative).name}-{kind}", special=(relative, kind))
    for relative in ("account_state/saver_state.json", "streak.json", "2027-07-15.json",
                     "account_state/contest_ledger.jsonl"):
        evaluate(f"masked-{Path(relative).name}-fifo", special=(relative, "fifo"),
                 manual_body=True, omit=True)

dump("status", rows)
