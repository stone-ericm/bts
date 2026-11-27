"""Replay the K1 scheduler and operator scenarios from the round-one review."""

import importlib
from datetime import datetime
from pathlib import Path

import pytest

from scripts.audit.season_walk.walk import ET
from tests.test_pick_chosen_on_cs5a import DAY


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-k1-codex"
BOUNDARIES = (
    (DAY, datetime.fromisoformat(DAY).replace(tzinfo=ET, hour=18), False),
    ("2027-06-14", datetime(2027, 6, 15, tzinfo=ET), True),
    ("2027-12-31", datetime(2028, 1, 1, tzinfo=ET), True),
)
SOURCES = {
    "check": ["contest_state"],
    "loader": ["contest_state"],
    "saved": ["contest_state"],
    "in_loop": ["contest_state", "contest_state", "missed_pick"],
    "final": ["contest_state", "missed_pick"],
    "missed_live": ["missed_pick"],
    "missed_preview": ["missed_pick"],
    "void": ["postponed_pick"],
    "late_initial": ["late_delivery"],
    "late_dm": ["late_delivery"],
    "late_public": ["late_delivery"],
    "delivered": [],
}


@pytest.fixture
def review_replays(monkeypatch):
    monkeypatch.syspath_prepend(str(REVIEW))
    return importlib.import_module("replay_callers"), importlib.import_module("replay_operator")


@pytest.mark.parametrize("slate,clock,overnight", BOUNDARIES, ids=["same-day", "midnight", "new-year"])
@pytest.mark.parametrize("case", SOURCES)
def test_scheduler_alert_dates(review_replays, case, slate, clock, overnight, monkeypatch):
    callers, _ = review_replays
    if slate == "2027-12-31":
        # These replays test the dedup date across the year boundary, not the season boundary.
        # Keep the synthetic December slate inside a season (positive stored days) so change
        # set K2's last-day floor for a zero-day slate does not alter the fixture's decision path.
        from bts import strategy
        monkeypatch.setitem(strategy.SEASON_END_DATES, 2027, "2028-01-02")
    result = callers.replay(case, slate, clock, overnight)
    assert [row[0] for row in result["trace"]] == SOURCES[case]
    assert all(row[4] == slate for row in result["trace"])
    if SOURCES[case]:
        assert result["receipt"]["sent_day"] == slate


@pytest.mark.parametrize(
    "repair,last_only",
    [(False, False), (True, False), (False, True)],
    ids=["absent", "filed-between-checks", "after-last-check"],
)
def test_operator_late_start(review_replays, repair, last_only):
    _, operator = review_replays
    operator.probe_operator.late_start(repair, last_only=last_only)


def test_operator_delivery_refusals(review_replays):
    _, operator = review_replays
    operator.probe_operator.guarded_delivery()


def test_operator_overnight_fallback(review_replays):
    callers, operator = review_replays
    results = {}
    for slate, clock, overnight in BOUNDARIES[1:]:
        result = callers.replay("final", slate, clock, overnight)
        result["wire"] = [
            body for body in result["wire"]
            if "- [contest_state] Morning check" not in body
        ]
        results[f"{slate}/final"] = result
    operator.probe_operator.overnight(results)


def test_b8a_filing_after_first_check_refusal():
    checklist = (ROOT / "docs/ops/2027-season-start.md").read_text(encoding="utf-8")
    row, = (line for line in checklist.splitlines() if line.startswith("| B8a |"))
    old = "**Enter it in the morning, before the scheduler's first lineup check.**"
    new = (
        "**Enter it in the morning, before the scheduler's first lineup check, when possible.** "
        "If that check refused before choosing a pick, confirm this is still today's slate in ET "
        "and that MLB has settled every prior entry or confirmed its absence after its window closed. "
        "File the verified dated observation between checks while another lineup check or fallback remains."
    )
    assert "**Enter it in the morning, before the scheduler's first lineup check, when possible.** " + K1B_LATE_NEW in row
    assert old not in row
    assert "after their entry windows closed" in row
    assert "if a later lineup check or the fallback refresh still runs" in row


K1B_DAILY_OLD = ("`bts set-contest-streak --date <today> --streak N --best-streak B --saver S` "
                 "with N, B and S read from MLB.")
K1B_DAILY_NEW = 'Resolve a relative `picks_dir` against the scheduler service\'s working directory. In the shell where you will run both commands, run `IFS= read -r DIR`, paste the scheduler\'s absolute picks directory as it appears on disk, and press Enter. Run `bts set-contest-streak --date <today> --streak N --best-streak B --saver S --picks-dir "${DIR:?Set DIR in this shell}"` with today\'s ET date and N, B and S read from MLB. Check the same directory with `bts season-contract status --picks-dir "${DIR:?Set DIR in this shell}"` and confirm the dated row and `live decision now: accepted`.'
K1B_LATE_OLD = ("If that check refused before choosing a pick, confirm this is still today's slate in ET "
                "and that MLB has settled every prior entry or confirmed its absence after its window closed. "
                "File the verified dated observation between checks while another lineup check or fallback remains.")
K1B_LATE_NEW = ("If a lineup check or fallback is blocked by unusable contest state, confirm this is still today's "
                "slate in ET and that MLB has settled every prior entry or confirmed its absence after its window "
                "closed. If a later lineup check or fallback refresh still has time to produce an enterable pick, "
                "file the verified dated observation between runs.")


def _b8a_row():
    checklist = (ROOT / "docs/ops/2027-season-start.md").read_text(encoding="utf-8")
    row, = (line for line in checklist.splitlines() if line.startswith("| B8a |"))
    return row


def test_k1b_daily_command_names_the_scheduler_picks_dir():
    row = _b8a_row()
    assert K1B_DAILY_NEW in row and K1B_DAILY_OLD not in row


def test_k1b_late_filing_covers_every_check_and_fallback():
    row = _b8a_row()
    assert K1B_LATE_NEW in row and K1B_LATE_OLD not in row
    assert "**Enter it in the morning, before the scheduler's first lineup check, when possible.**" in row


def test_k1b_daily_command_lands_in_the_scheduler_picks_dir(tmp_path, monkeypatch):
    """B2: the B8a command, filled in with the live directory, writes the observation where the live read looks."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    from click.testing import CliRunner
    from bts import cli, contest_state

    et = ZoneInfo("America/New_York")
    now = dt.datetime(2027, 7, 15, 18, 40, tzinfo=et)
    monkeypatch.chdir(tmp_path)
    import shlex
    import subprocess
    live = tmp_path / "live picks"
    live.mkdir()
    command = next(part for part in K1B_DAILY_NEW.split("`") if part.startswith("bts set-contest-streak"))
    command = command.replace("<today>", "2027-07-15").replace(" N ", " 57 ")
    command = command.replace(" B ", " 60 ").replace(" S ", " used ")
    # The shell expands "$DIR" as ONE argument even when the path holds a space (the K1b check's case).
    expanded = subprocess.run(["/bin/sh", "-c", 'printf "%s\\0" ' + command[len("bts "):]],
                              env={"DIR": str(live)}, capture_output=True, timeout=10, check=True).stdout
    filled = expanded.decode().split("\0")[:-1]
    real = dt.datetime

    class Clock(real):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with monkeypatch.context() as patch:
        patch.setattr(dt, "datetime", Clock)
        patch.setattr(cli, "datetime", Clock)
        result = CliRunner().invoke(cli.cli, filled)
    assert result.exit_code == 0, result.output
    assert (live / "account_state" / "contest_streak.manual.json").exists()
    assert not (tmp_path / "data" / "picks" / "account_state" / "contest_streak.manual.json").exists()
    state = contest_state.load_decision_streak_state(live, require_contest_state=True, now=now)
    assert state.observation_kind == "dated" and state.streak == 57
