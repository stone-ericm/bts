"""Check ET rollover, interrupted scoring/reading, and a restarted report."""

import datetime as dt
import errno
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

from bts import cli, picks as scoring, util
from report_harness import Chat, DATE, ET, _box, _write_decision, observed, report, snapshot


def et_boundaries(root):
    picks, config = _box(root)
    path = picks / "streak.json"
    prior = b'{"season":2027,"streak":17}'
    path.write_bytes(prior)
    before = snapshot(root)
    times = [(dt.datetime(2027, 12, 31, 23, 59, tzinfo=ET), None, "2027-12-31"),
             (dt.datetime(2028, 1, 1, 0, 1, tzinfo=ET), None, "2028-01-01"),
             (dt.datetime(2028, 1, 1, 8, 30, tzinfo=ET), "2027-12-31", "2027-12-31"),
             (dt.datetime(2028, 3, 24, 8, 30, tzinfo=ET), "2028-03-23", "2028-03-23")]
    for wall, date, reported in times:
        result, chat = report(picks, config, 0, date=date, wall=wall)
        payload = json.loads(result.stdout)
        assert (payload["date"], payload["streak_problem"], result.exit_code, chat.accepted) == (reported, None, 0, [])
        assert snapshot(root) == before
        print(f"{wall.isoformat()} reported {reported} with prior-season file quiet")
    assert scoring.load_streak(picks, season=2028) == 0
    path.write_bytes(b'{"season":2027,"streak":"17"}')
    result, chat = report(picks, config, 0, date=times[-1][1], wall=times[-1][0])
    assert scoring.load_streak(picks, season=2028) == 0
    assert result.exit_code == 1 and len(chat.accepted) == 1
    assert json.loads(result.stdout)["streak_problem"] == "streak is not a non-negative integer"
    print("Opening morning: bad count in readable prior-season record still gives a file advisory; scorer starts at zero")


def scorer_interrupts(root):
    picks, config = _box(root)
    _write_decision(picks)
    observed(picks)
    path = picks / "streak.json"
    before = b'{"season":2027,"streak":3}'
    path.write_bytes(before)
    with patch.object(util.os, "replace", side_effect=InterruptedError):
        try:
            scoring.update_streak([True], picks, date=DATE)
        except InterruptedError:
            assert path.read_bytes() == before
        else:
            raise AssertionError("scorer reached replacement")

    primary = scoring.Pick(batter_id=1, batter_name="Kwan", team="CLE", lineup_position=1,
                           pitcher_name="Offline", pitcher_id=2, p_game_hit=0.78, flags=[],
                           projected_lineup=False, game_pk=900, game_time="2027-04-02T23:00:00Z")
    committed = scoring.DailyPick(DATE, "2027-04-02T20:00:00+00:00", primary, None, None, notification_sent=True)
    scoring.save_pick(committed, picks)
    args = ["check-results", "--date", DATE, "--picks-dir", str(picks)]
    grade_time = dt.datetime(2027, 4, 3, 1, tzinfo=ET)
    with patch.object(cli, "_now_et", return_value=grade_time), \
         patch.object(scoring, "resolve_daily_slot_results", return_value={"pick": "hit"}):
        with patch.object(scoring, "save_pick", side_effect=KeyboardInterrupt):
            cut = CliRunner().invoke(cli.cli, args)
        assert cut.exit_code == 1 and "Aborted" in cut.output, (cut.output, cut.exception)
        state = json.loads(path.read_bytes())
        assert state["streak"] == 4 and DATE in state["applied_dates"]
        assert scoring.load_pick(DATE, picks).result is None
        before_report = snapshot(root)
        outcome, chat = report(picks, config, 1)
        assert json.loads(outcome.stdout)["state"] == "entered" and outcome.exit_code == 0
        assert snapshot(root) == before_report and not chat.accepted
        retry = CliRunner().invoke(cli.cli, args)
    assert retry.exit_code == 0 and scoring.load_pick(DATE, picks).result == "hit", (retry.output, retry.exception)
    assert json.loads(path.read_bytes()) == state
    print("Interrupt before replace preserved old streak; interrupt after replace left unresolved pick with valid marker. "
          "Report stayed quiet and read-only; scorer retry settled the pick without reapplying the streak.")


def report_interrupt_and_retry(root):
    picks, config = _box(root)
    path = picks / "streak.json"
    path.write_bytes(b"{")
    before = snapshot(root)
    old_open, old_read = os.open, os.read
    streak_fds = []

    def open_streak(candidate, flags, *args):
        fd = old_open(candidate, flags, *args)
        if Path(candidate) == path:
            streak_fds.append(fd)
        return fd

    def interrupt(fd, size):
        if streak_fds and fd == streak_fds[-1]:
            assert old_read(fd, size) == b"{"
            raise KeyboardInterrupt
        return old_read(fd, size)

    with patch.object(util.os, "open", open_streak), patch.object(util.os, "read", interrupt):
        interrupted, chat = report(picks, config, 1)
    assert interrupted.exit_code == 1 and "Aborted" in interrupted.stderr and not chat.attempted
    assert len(streak_fds) == 1 and snapshot(root) == before
    try:
        os.fstat(streak_fds[0])
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        raise AssertionError("interrupted streak descriptor is still open")

    results = {}
    for failure, expected in (("timeout", "TimeoutError"), ("no-id", "KeyError"), (None, None)):
        result, chat = report(picks, config, 1, chat=Chat(failure=failure))
        assert result.exit_code == 1 and len(chat.attempted) == 1
        assert result.stderr == (f"day-outcome: DM failed: {expected}\n" if expected else "")
        assert snapshot(root) == before
        results[failure] = (result.stdout, chat.attempted)
    assert len({(stdout, tuple(sends)) for stdout, sends in results.values()}) == 1
    result, chat = report(picks, config, 1, recipient=False)
    assert result.exit_code == 1 and result.stdout == results[None][0] and not chat.attempted
    assert snapshot(root) == before
    print("Interrupt in the read closed the FD; timeout/no-ID each attempted one send; restart resent the identical "
          "report; no-recipient mode gave the same stdout; no report wrote state.")


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    et_boundaries(root / "et")
    scorer_interrupts(root / "scorer")
    report_interrupt_and_retry(root / "report")
