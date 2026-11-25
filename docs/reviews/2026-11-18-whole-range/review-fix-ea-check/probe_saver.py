"""Follow the stale warnings' saver instructions in the displayed ET year."""

from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, orchestrator, picks, saver_state, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk

from probe_contest import at, command, config, fetch, observation, send_stale, stale, state


WORK = Path(__file__).resolve().parent


def morning(path, now, expected_stamp):
    walk._set_clock(now)
    day = now.date().isoformat()
    sent = []
    with patch.object(alert, "send_dm", side_effect=lambda _, body: sent.append(body)):
        scheduler._morning_state_check(config(path), path, day, now, now + timedelta(hours=1))
    assert len(sent) == 1
    expected = (f"In {expected_stamp} ET away from midnight, if MLB shows this season's saver used, "
                "check bts saver-state --picks-dir DIR.")
    assert expected in sent[0]
    print("CAPTURED", sent[0], sep="\n")
    return sent[0]


def mark_if_shown_used(path, warning_year, mlb_season):
    code, reported = walk._invoke("saver-state", "--picks-dir", path)
    assert code == 0
    print(reported.strip())
    current = int(reported.rsplit("season ", 1)[1].rstrip().rstrip(")"))
    if warning_year != current or mlb_season != current:
        return
    if "saver_state: uninitialized" in reported:
        command(path, "saver-state", "--init", "used", "--season", current)
    elif "saver_state: active" in reported or "saver_state: not_earned" in reported:
        command(path, "saver-state", "--use")
    else:
        assert "saver_state: used" in reported


def after_expiry(path, route, read_date):
    day14 = at("2027-07-14T21:25") if route == "delivery" else at("2027-07-14T10:00")
    stale(path, day14, streak=12, best=13, saver="active")
    warning = send_stale(path, day14) if route == "delivery" else morning(path, day14, 2027)
    assert "In 2027 ET away from midnight," in warning
    for day, value in (("2027-07-15", 13), ("2027-07-16", 14)):
        walk._set_clock(at(day + "T01:10"))
        fetch(path, value, value, (at(day + "T01:10") - timedelta(days=1)).date())
        assert saver_state.load_saver_state(path, season=2027).state == ("used" if read_date < day else "active")
        if day == read_date:
            walk._set_clock(at(day + "T08:30"))
            print(route, "warning read", day, "at 08:30. MLB confirmed its 2027 saver used July 13.")
            assert "saver: active; season: 2027" in command(path, "season-contract", "status")
            if day == "2027-07-15":
                observation(path, day, value, value, "used")
                assert state(path).observation_kind == "dated"
                assert saver_state.load_saver_state(path, season=2027).state == "active"
            mark_if_shown_used(path, 2027, 2027)

    now = at("2027-07-16T18:15")
    walk._set_clock(now)
    assert state(path).observation_kind == "automatic"
    assert saver_state.load_saver_state(path, season=2027).state == "used"
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in range(70000, 70003)}
    with patch.object(picks, "get_game_statuses_detailed", return_value=statuses), \
            patch.object(orchestrator, "run_cascade", return_value=(walk._predictions(now.date(), .79), "pinned")):
        _, used, _ = orchestrator.run_and_pick(config(path), now.date().isoformat())
        command(path, "saver-state", "--undo")
        _, active, _ = orchestrator.run_and_pick(config(path), now.date().isoformat())
    assert (used.action, used.saver_available, active.action, active.saver_available) == ("skip", False, "single", True)
    print("July 16 automatic: used -> skip; uncorrected active -> single")


def flags_and_year_boundary(root):
    dec = at("2027-12-31T10:00")
    jan = at("2028-01-01T10:00")
    path = root / "dec-jan" / "picks"
    stale(path, dec, saver="active")
    dec_body = morning(path, dec, 2027)
    assert "Morning check for 2027-12-31" in dec_body
    # Positive date control from a fresh receipt. The old warning is inapplicable on Jan 1.
    walk._set_clock(jan)
    before = (path / "account_state/saver_state.json").read_bytes()
    assert "uninitialized (season 2028)" in command(path, "saver-state")
    mark_if_shown_used(path, 2027, 2027)
    assert (path / "account_state/saver_state.json").read_bytes() == before
    assert saver_state.load_saver_state(path, season=2027).state == "active"
    jan_path = root / "jan" / "picks"
    stale(jan_path, jan, saver="active")
    assert "Morning check for 2028-01-01" in morning(jan_path, jan, 2028)

    walk._set_clock(at("2027-12-31T23:59:55"))
    crossing = root / "crossing" / "picks"
    before = at("2027-12-31T23:59:55")
    after = at("2028-01-01T00:00:20")
    stale(crossing, before)
    assert "In 2028 ET" in send_stale(crossing, before, after)
    assert "uninitialized (season 2028)" in command(crossing, "saver-state")
    walk._set_clock(jan)
    mark_if_shown_used(crossing, 2028, 2027)
    assert saver_state.load_saver_state(crossing, season=2028).state == "uninitialized"
    assert saver_state.load_saver_state(crossing, season=2027).state == "active"
    print("Year-crossing synthetic send uses 2028. A 2027 MLB saver usage does not authorize a 2028 flag write.")

    walk._set_clock(at("2027-07-15T08:30"))
    for flag in ("active", "not_earned", "uninitialized", "used"):
        path = root / flag / "picks"
        if flag != "uninitialized":
            command(path, "saver-state", "--init", flag, "--season", 2027)
        mark_if_shown_used(path, 2027, 2027)
        assert saver_state.load_saver_state(path, season=2027).state == "used"
        print("Flag", flag, "-> used")
    race = root / "race" / "picks"
    assert "uninitialized" in command(race, "saver-state")
    command(race, "saver-state", "--init", "active", "--season", 2027)
    assert "already initialized" in command(race, "saver-state", "--init", "used", "--season", 2027, expected=1)
    mark_if_shown_used(race, 2027, 2027)
    assert saver_state.load_saver_state(race, season=2027).state == "used"
    disk = json.loads((race / "account_state/saver_state.json").read_text())
    print("Reread after init race ->", disk["state"])


if __name__ == "__main__":
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(at("2027-07-14T21:25")):
        root = Path(temp)
        after_expiry(root / "delivery" / "picks", "delivery", "2027-07-15")
        after_expiry(root / "morning" / "picks", "morning", "2027-07-16")
        flags_and_year_boundary(root)
