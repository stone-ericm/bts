"""W1.5 incident register: expected-failure fixtures for unfixed defects whose contract is fixed.

Design: docs/superpowers/specs/2026-09-29-incident-register-design.md §9.7, §10.

Every strict expected failure raises its OWN exception class (never an AssertionError), and only
from the module's ``_oracle`` — and only when the WHOLE observed outcome equals the declared bad
outcome. Any other outcome is an ordinary assertion failure, so a defect that changes shape (e.g.
``None`` instead of ``miss``) fails loudly instead of inheriting the expected failure. Ordinary
assertions before the oracle prove the fixture actually executed the declared path (HTTP calls
made, the scheduler's check ran at the declared time, the classifier returned the declared
state). HTTP is the only thing mocked for L01 (``grade_pick_in_feed`` and the real
``bts check-results`` command) and L02 (``reconcile_results`` itself); E77 is component-level and
lists its mocks.
"""
from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import cli as cli_mod
from bts import picks as picks_mod
from bts.cli import cli
from bts.picks import (
    DailyPick,
    Pick,
    grade_pick_in_feed,
    load_pick,
    load_saver_available,
    load_streak,
    reconcile_results,
    save_pick,
    save_streak,
)

ET = ZoneInfo("America/New_York")


class IncidentExpectedFailure(Exception):
    """Base of the register's dedicated expected-failure classes (deliberately not AssertionError)."""


class PassGradedAsMiss(IncidentExpectedFailure):
    """L01: a BTS Pass (no official AB and no SF, did not play, or suspended before a hit) graded "miss"."""


class PassScoredAsNoHit(IncidentExpectedFailure):
    """L01 downstream: a Pass leg reset the streak or consumed the saver."""


class SameDayReplayRollback(IncidentExpectedFailure):
    """L02: reconcile's season replay drops today's already-applied terminal result."""


class SingletonSlateUndelivered(IncidentExpectedFailure):
    """E77: a one-game slate whose start moved up after the morning fetch ends with an enterable pick never delivered."""


class ResultAppliedTwice(IncidentExpectedFailure):
    """L04: a death between the streak write and the terminal pick save lets the restarted scorer apply the result again."""


class PostponedPickDelivered(IncidentExpectedFailure):
    """L03: the fallback delivered a cached pick whose game was evidenced postponed before the send."""


def _emit_oracle_record(what: str, actual, required, bad) -> None:
    """With ``$W15_ORACLE_OUT`` set (evidence runs only), append the oracle's structured inputs so the
    acceptance check can compare the declared bad value with what production returned or persisted
    (Codex phase-1 r3 #4). Values here are this fixture's own primitives/tuples/dicts."""
    path = os.environ.get("W15_ORACLE_OUT")
    if path:
        node = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"node": node, "what": what, "actual": actual, "required": required,
                                 "bad": bad}) + "\n")


def _oracle(actual, required, bad, exc: type[IncidentExpectedFailure], what: str) -> None:
    """Pass on ``required``; raise ``exc`` ONLY for the declared bad value; else AssertionError."""
    _emit_oracle_record(what, actual, required, bad)
    if actual == required:
        return
    if actual == bad:
        raise exc(f"{what}: got the declared bad value {actual!r}; required {required!r}")
    raise AssertionError(
        f"{what}: {actual!r} is neither the required {required!r} nor the declared bad {bad!r}"
    )


# ---------------------------------------------------------------------------
# Synthetic MLB responses — complete inputs: final game, explicit stats, timed plays
# ---------------------------------------------------------------------------
DATE = "2026-06-10"
AFTER_NORMAL_GAMES = datetime(2026, 6, 11, 1, 0, tzinfo=ET)       # 01:00 ET the next day
AFTER_RESUMPTION = datetime(2026, 6, 11, 20, 0, tzinfo=ET)        # after the resumed portion ended
GAME_A, GAME_B = 824001, 824002
BATTER_A, BATTER_B = 660001, 660002
PRE = "2026-06-10T23:30:00Z"      # before the suspension
RESUME = "2026-06-11T17:00:00Z"   # resumeDateTime of a suspended game (13:00 ET)
POST = "2026-06-11T17:30:00Z"     # resumed portion — never evaluated for BTS


def _batting(*, hits=0, at_bats=0, sac_flies=0, walks=0, hbp=0, sac_bunts=0):
    pa = at_bats + sac_flies + walks + hbp + sac_bunts
    return {"hits": hits, "atBats": at_bats, "sacFlies": sac_flies, "baseOnBalls": walks,
            "hitByPitch": hbp, "sacBunts": sac_bunts, "plateAppearances": pa}


def _play(batter: int, event: str, start: str) -> dict:
    return {"result": {"eventType": event},
            "matchup": {"batter": {"id": batter, "fullName": f"Batter {batter}"}},
            "about": {"startTime": start}}


def _player(batter: int, batting: dict, *, bench: bool = False) -> dict:
    entry = {"person": {"id": batter, "fullName": f"Batter {batter}"},
             "stats": {"batting": batting},
             "gameStatus": {"isCurrentBatter": False, "isOnBench": bench, "isSubstitute": False}}
    if not bench:
        entry["battingOrder"] = "100"
    return entry


def _feed(batter: int, *, batting: dict, plays=(), resume: str | None = None,
          bench: bool = False, on_roster: bool = True) -> dict:
    players = {f"ID{batter}": _player(batter, batting, bench=bench)} if on_roster else {}
    return {
        "gameData": {"status": {"abstractGameCode": "F", "detailedState": "Final"},
                     "datetime": {"resumeDateTime": resume} if resume else {}},
        "liveData": {"plays": {"allPlays": list(plays)},
                     "boxscore": {"teams": {"away": {"players": players},
                                            "home": {"players": {}}}}},
    }


# name: (feed builder, required grade, declared bad grade or None for a control, as-of clock)
CASES = {
    "walks_only": (lambda: _feed(BATTER_A, batting=_batting(walks=2, hbp=1),
                                 plays=[_play(BATTER_A, "walk", PRE),
                                        _play(BATTER_A, "hit_by_pitch", PRE),
                                        _play(BATTER_A, "walk", PRE)]), "void", "miss", AFTER_NORMAL_GAMES),
    # did not play: on the roster, explicit zero stats, on the bench, no batting order, no plays
    "did_not_play": (lambda: _feed(BATTER_A, batting=_batting(), bench=True), "void", "miss",
                     AFTER_NORMAL_GAMES),
    "sac_bunt_only": (lambda: _feed(BATTER_A, batting=_batting(sac_bunts=1, walks=1),
                                    plays=[_play(BATTER_A, "sac_bunt", PRE),
                                           _play(BATTER_A, "walk", PRE)]), "void", "miss",
                      AFTER_NORMAL_GAMES),
    "suspended_walks_only": (lambda: _feed(BATTER_A, batting=_batting(walks=1, hits=1, at_bats=1),
                                           plays=[_play(BATTER_A, "walk", PRE),
                                                  _play(BATTER_A, "single", POST)],
                                           resume=RESUME), "void", "miss", AFTER_RESUMPTION),
    "suspended_ab_no_hit": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=2),
                                          plays=[_play(BATTER_A, "field_out", PRE),
                                                 _play(BATTER_A, "home_run", POST)],
                                          resume=RESUME), "void", "miss", AFTER_RESUMPTION),
    # controls — the required value is what HEAD already returns
    "sac_fly_no_hit": (lambda: _feed(BATTER_A, batting=_batting(sac_flies=1),
                                     plays=[_play(BATTER_A, "sac_fly", PRE)]), "miss", None,
                       AFTER_NORMAL_GAMES),
    "official_ab_no_hit": (lambda: _feed(BATTER_A, batting=_batting(at_bats=4),
                                         plays=[_play(BATTER_A, "field_out", PRE)] * 4), "miss", None,
                           AFTER_NORMAL_GAMES),
    "resumed_only": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=1),
                                   plays=[_play(BATTER_A, "single", POST)], resume=RESUME), "void", None,
                     AFTER_RESUMPTION),
    "pre_suspension_hit": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=2),
                                         plays=[_play(BATTER_A, "single", PRE),
                                                _play(BATTER_A, "field_out", POST)],
                                         resume=RESUME), "hit", None, AFTER_RESUMPTION),
    "absent_from_rosters": (lambda: _feed(BATTER_A, batting=_batting(), on_roster=False), None, None,
                            AFTER_NORMAL_GAMES),
}
XFAIL_CASES = [k for k, (_f, _r, bad, _c) in CASES.items() if bad is not None]


def _params(names, exc):
    return [pytest.param(n, marks=pytest.mark.xfail(strict=True, raises=exc,
                                                    reason="L01 unfixed: BTS Pass graded as a miss"))
            if CASES[n][2] is not None else n for n in names]


def _route_http(monkeypatch, feeds: dict[int, dict]) -> list[str]:
    """Answer MLB statsapi URLs from ``feeds`` ({game_pk: feed}); record every URL."""
    calls: list[str] = []

    def fake_urlopen(url, timeout=15, **_kw):
        calls.append(url)
        if "/feed/live" in url:
            pk = int(url.split("/game/")[1].split("/")[0])
            return io.BytesIO(json.dumps(feeds[pk]).encode())
        if "/api/v1/schedule" in url:
            games = [{"gamePk": pk, "status": {"abstractGameCode": "F", "detailedState": "Final",
                                                "statusCode": "F"}} for pk in feeds]
            return io.BytesIO(json.dumps({"dates": [{"date": DATE, "games": games}]}).encode())
        raise AssertionError(f"unexpected URL in fixture: {url}")

    monkeypatch.setattr(picks_mod, "retry_urlopen", fake_urlopen)
    return calls


def _feed_calls(calls: list[str], game_pk: int) -> int:
    return sum(1 for u in calls if f"/game/{game_pk}/feed/live" in u)


def _pick(batter: int, game_pk: int, **over) -> Pick:
    fields = dict(batter_name=f"Batter {batter}", batter_id=batter, team="NYM", lineup_position=1,
                  pitcher_name="P", pitcher_id=1, p_game_hit=0.75, flags=[], projected_lineup=False,
                  game_pk=game_pk, game_time="2026-06-10T23:10:00Z")
    fields.update(over)
    return Pick(**fields)


def _delivered_daily(pick: Pick, double_down: Pick | None = None, date: str = DATE) -> DailyPick:
    return DailyPick(date=date, run_time="2026-06-10T21:00:00+00:00", pick=pick,
                     double_down=double_down, runner_up=None, notification_sent=True,
                     notification_channel="bluesky_dm", notification_id="dm-1", delivery_attempted=True,
                     delivered_at="2026-06-10T21:01:00+00:00")


def _check_results(monkeypatch, picks_dir: Path, tmp_path: Path, as_of: datetime):
    monkeypatch.setattr(cli_mod, "_now_et", lambda: as_of)
    return CliRunner().invoke(cli, ["check-results", "--date", DATE, "--picks-dir", str(picks_dir),
                                    "--shadow-status-output", str(tmp_path / "shadow_status.json")])


def _outcome(picks_dir: Path) -> tuple:
    daily = load_pick(DATE, picks_dir)
    return (daily.slot_results, daily.result, load_streak(picks_dir), load_saver_available(picks_dir))


# ---------------------------------------------------------------------------
# L01 — BTS Pass grading (official rules §6 A/B/C; pinned excerpt in the evidence dir)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", _params(list(CASES), PassGradedAsMiss))
def test_l01_grade_pick_in_feed(case):
    feed, required, bad, _as_of = CASES[case]
    grade = grade_pick_in_feed(feed(), BATTER_A, f"Batter {BATTER_A}")
    if bad is None:
        assert grade == required
        return
    _oracle(grade, required, bad, PassGradedAsMiss, f"grade_pick_in_feed[{case}]")


SINGLE_CASES = XFAIL_CASES + ["sac_fly_no_hit", "official_ab_no_hit", "resumed_only", "pre_suspension_hit"]


@pytest.mark.parametrize("case", _params(SINGLE_CASES, PassScoredAsNoHit))
def test_l01_check_results_single_pick(case, monkeypatch, tmp_path):
    """Single pick through the real `bts check-results` (HTTP stubbed): a Pass holds (5, saver)."""
    feed, required_grade, bad, as_of = CASES[case]
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    calls = _route_http(monkeypatch, {GAME_A: feed()})

    result = _check_results(monkeypatch, picks_dir, tmp_path, as_of)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1, "the grader never fetched the pick's game feed"
    after = {"void": 5, "miss": 0, "hit": 6}
    required = ({"pick": required_grade}, required_grade, after[required_grade], True)
    if bad is None:
        assert _outcome(picks_dir) == required
        return
    _oracle(_outcome(picks_dir), required, ({"pick": bad}, bad, after[bad], True),
            PassScoredAsNoHit, f"check-results single[{case}]")


@pytest.mark.parametrize("case", _params(["walks_only", "did_not_play", "suspended_ab_no_hit"],
                                         PassScoredAsNoHit))
def test_l01_check_results_hit_plus_pass(case, monkeypatch, tmp_path):
    """Double down: primary Hit + a Pass leg = +1 (5 -> 6), never a reset."""
    feed, _required, _bad, as_of = CASES[case]
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_B, GAME_B), double_down=_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    hit_feed = _feed(BATTER_B, batting=_batting(hits=1, at_bats=4), plays=[_play(BATTER_B, "single", PRE)])
    calls = _route_http(monkeypatch, {GAME_B: hit_feed, GAME_A: feed()})

    result = _check_results(monkeypatch, picks_dir, tmp_path, as_of)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1 and _feed_calls(calls, GAME_B) >= 1
    required = ({"pick": "hit", "double_down": "void"}, "hit", 6, True)
    bad = ({"pick": "hit", "double_down": "miss"}, "miss", 0, True)
    _oracle(_outcome(picks_dir), required, bad, PassScoredAsNoHit, f"hit+pass[{case}]")


@pytest.mark.xfail(strict=True, raises=PassScoredAsNoHit, reason="L01 unfixed: Pass+Pass reset the streak")
def test_l01_check_results_pass_plus_pass_preserves(monkeypatch, tmp_path):
    """Double down: Pass + Pass = streak preserved (5 stays 5)."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_B, GAME_B), double_down=_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    walks_b = _feed(BATTER_B, batting=_batting(walks=3), plays=[_play(BATTER_B, "walk", PRE)] * 3)
    calls = _route_http(monkeypatch, {GAME_B: walks_b, GAME_A: CASES["did_not_play"][0]()})

    result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1 and _feed_calls(calls, GAME_B) >= 1
    required = ({"pick": "void", "double_down": "void"}, "void", 5, True)
    bad = ({"pick": "miss", "double_down": "miss"}, "miss", 0, True)
    _oracle(_outcome(picks_dir), required, bad, PassScoredAsNoHit, "pass+pass")


@pytest.mark.xfail(strict=True, raises=PassScoredAsNoHit,
                   reason="L01 unfixed: a Pass at streak 12 consumes the saver")
def test_l01_pass_at_saver_streak_keeps_the_saver(monkeypatch, tmp_path):
    """At streak 12 with the saver available, a Pass must leave (12, saver available)."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(12, picks_dir, saver_available=True)
    calls = _route_http(monkeypatch, {GAME_A: CASES["walks_only"][0]()})

    result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1
    required = ({"pick": "void"}, "void", 12, True)
    bad = ({"pick": "miss"}, "miss", 12, False)          # the saver absorbed a "miss" that was a Pass
    _oracle(_outcome(picks_dir), required, bad, PassScoredAsNoHit, "pass at saver streak")


def test_l01_absent_player_stays_pending(monkeypatch, tmp_path):
    """Control: a batter on neither roster (and no other final game) is pending, never a Pass or a miss."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    calls = _route_http(monkeypatch, {GAME_A: CASES["absent_from_rosters"][0]()})

    result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1, "the resolver was never reached"
    assert _outcome(picks_dir) == (None, None, 5, True)


# ---------------------------------------------------------------------------
# L02 — same-day replay rollback in reconcile_results (Codex reconcile-cutoff r1 #3)
# ---------------------------------------------------------------------------
def _resolved(date: str, result: str, batter: int, game_pk: int) -> DailyPick:
    daily = _delivered_daily(_pick(batter, game_pk), date=date)
    daily.result = result
    daily.slot_results = {"pick": result}
    return daily


def _no_http(monkeypatch) -> list:
    calls: list = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("reconcile fetched a feed for a date past its BTS cutoff")

    monkeypatch.setattr(picks_mod, "retry_urlopen", refuse)
    return calls


def _snapshot(picks_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(picks_dir.glob("2026-*.json"))}


def _two_late_reconciles(picks_dir: Path, now: datetime, monkeypatch) -> list[tuple[int, bool]]:
    """Run reconcile twice at ``now``; ordinary assertions pin everything except (streak, saver)."""
    calls = _no_http(monkeypatch)
    before = _snapshot(picks_dir)
    states = []
    for run in (1, 2):
        corrections = reconcile_results(picks_dir, clock=lambda: now)
        assert corrections == [], f"run {run}: unexpected corrections {corrections}"
        assert _snapshot(picks_dir) == before, f"run {run}: a pick file changed"
        assert calls == [], f"run {run}: fetched {len(calls)} feed(s)"
        states.append((load_streak(picks_dir), load_saver_available(picks_dir)))
    return states


@pytest.mark.xfail(strict=True, raises=SameDayReplayRollback,
                   reason="L02 unfixed: replay excludes today's already-applied hit")
def test_l02_same_day_hit_survives_two_late_reconciles(tmp_path, monkeypatch):
    picks_dir = tmp_path / "picks"
    save_pick(_resolved("2026-06-10", "hit", BATTER_A, GAME_A), picks_dir)
    save_pick(_resolved("2026-06-11", "hit", BATTER_B, GAME_B), picks_dir)
    save_streak(2, picks_dir, saver_available=True)
    states = _two_late_reconciles(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET), monkeypatch)
    _oracle(states, [(2, True), (2, True)], [(1, True), (1, True)], SameDayReplayRollback,
            "two 23:00 reconciles (streak, saver)")


@pytest.mark.xfail(strict=True, raises=SameDayReplayRollback,
                   reason="L02 unfixed: replay restores a saver today's miss consumed")
def test_l02_same_day_saver_consumption_survives_two_late_reconciles(tmp_path, monkeypatch):
    picks_dir = tmp_path / "picks"
    for day in range(1, 11):                       # 06-01 .. 06-10: ten single hits -> streak 10
        save_pick(_resolved(f"2026-06-{day:02d}", "hit", BATTER_A, GAME_A + day), picks_dir)
    save_pick(_resolved("2026-06-11", "miss", BATTER_B, GAME_B), picks_dir)
    save_streak(10, picks_dir, saver_available=False)   # today's miss at 10 consumed the saver
    states = _two_late_reconciles(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET), monkeypatch)
    _oracle(states, [(10, False), (10, False)], [(10, True), (10, True)], SameDayReplayRollback,
            "two 23:00 reconciles (streak, saver)")


def test_l02_unplayed_undelivered_preview_is_excluded(tmp_path, monkeypatch):
    """Control: today's unplayed, undelivered preview is not replayed; yesterday's hit stands."""
    picks_dir = tmp_path / "picks"
    save_pick(_resolved("2026-06-10", "hit", BATTER_A, GAME_A), picks_dir)
    save_pick(DailyPick(date="2026-06-11", run_time="2026-06-11T07:00:00+00:00",
                        pick=_pick(BATTER_B, GAME_B), double_down=None, runner_up=None), picks_dir)
    save_streak(1, picks_dir, saver_available=True)
    states = _two_late_reconciles(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET), monkeypatch)
    assert states == [(1, True), (1, True)]


# ---------------------------------------------------------------------------
# L04 — scoring crash window (design §10.3; fable5 audit P3, deferred in 49f2538)
#
# `bts check-results` writes streak.json (update_streak) BEFORE the terminal pick save, so a death
# between the two leaves an applied result on a pick that still reads unresolved, and the restarted
# scorer applies it again. HTTP is stubbed as for L01. The death is the pick save raising a
# BaseException, which nothing in check-results catches (as nothing catches SIGKILL, OOM or a
# restart); it fires before that save writes anything, i.e. strictly between the two writes.
# ---------------------------------------------------------------------------
class _ProcessDeath(BaseException):
    """Stands in for the process dying between two writes."""


def _crash_then_restart(monkeypatch, picks_dir: Path, tmp_path: Path, feed: dict) -> tuple:
    """Score once with the terminal pick save dying, then again as the restarted process; return the
    outcome after the restart. Only the fault point is asserted here (the save was reached and had
    not persisted the result): the streak at the death is what the defect is about, and a repair
    (a journal, a reordering, one atomic file) may leave it different, so it is never presupposed."""
    calls = _route_http(monkeypatch, {GAME_A: feed})
    real_save = picks_mod.save_pick
    deaths: list[str] = []

    def dies_before_persisting(daily, picks_dir_arg):
        deaths.append(daily.date)
        raise _ProcessDeath("killed between the streak write and the terminal pick save")

    monkeypatch.setattr(picks_mod, "save_pick", dies_before_persisting)
    with pytest.raises(_ProcessDeath):
        _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
    assert deaths == [DATE], "the fault never reached the terminal pick save"
    assert load_pick(DATE, picks_dir).result is None, "the result was persisted before the death"
    monkeypatch.setattr(picks_mod, "save_pick", real_save)              # the restarted process
    result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 2, "the restarted scorer never re-resolved the pick"
    return _outcome(picks_dir)


HIT_FEED = lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=4),  # noqa: E731
                         plays=[_play(BATTER_A, "single", PRE)] + [_play(BATTER_A, "field_out", PRE)] * 3)
MISS_FEED = lambda: _feed(BATTER_A, batting=_batting(at_bats=4),  # noqa: E731
                          plays=[_play(BATTER_A, "field_out", PRE)] * 4)


@pytest.mark.xfail(strict=True, raises=ResultAppliedTwice,
                   reason="L04 unfixed: a death between the streak write and the pick save double-applies a hit")
def test_l04_hit_applied_once_across_a_crash_and_restart(monkeypatch, tmp_path):
    """Streak 5, a hit, death between the writes, restart: the hit counts once (6), never twice (7)."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    after = _crash_then_restart(monkeypatch, picks_dir, tmp_path, HIT_FEED())
    _oracle(after, ({"pick": "hit"}, "hit", 6, True), ({"pick": "hit"}, "hit", 7, True),
            ResultAppliedTwice, "hit across a crash and restart")


@pytest.mark.xfail(strict=True, raises=ResultAppliedTwice,
                   reason="L04 unfixed: the re-applied saver-absorbed miss resets the streak")
def test_l04_saver_miss_applied_once_across_a_crash_and_restart(monkeypatch, tmp_path):
    """Streak 12 with the saver, a miss (the saver absorbs it), death, restart: (12, saver used),
    never a second application that finds the saver gone and resets to 0."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(12, picks_dir, saver_available=True)
    after = _crash_then_restart(monkeypatch, picks_dir, tmp_path, MISS_FEED())
    _oracle(after, ({"pick": "miss"}, "miss", 12, False), ({"pick": "miss"}, "miss", 0, False),
            ResultAppliedTwice, "saver miss across a crash and restart")


@pytest.mark.xfail(strict=True, raises=ResultAppliedTwice,
                   reason="L04 unfixed: the daemon's polling dies between the writes; the 01:00 scorer re-applies")
def test_l04_polling_death_then_the_cron_scorer_applies_once(monkeypatch, tmp_path):
    """The two production scorers in sequence: the daemon's result polling dies between its writes
    (e.g. a deploy restart at midnight), then the 01:00 `bts check-results` scores the same date."""
    from bts import scheduler as sched_mod
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    calls = _route_http(monkeypatch, {GAME_A: HIT_FEED()})
    monkeypatch.setattr(sched_mod, "retry_urlopen", picks_mod.retry_urlopen)     # the same HTTP stub
    monkeypatch.setattr(sched_mod, "_now_et", lambda: datetime(2026, 6, 11, 0, 0, tzinfo=ET))
    real_save = picks_mod.save_pick
    deaths: list[str] = []

    def dies_before_persisting(daily, picks_dir_arg):
        deaths.append(daily.date)
        raise _ProcessDeath("daemon killed between the streak write and the terminal pick save")

    monkeypatch.setattr(picks_mod, "save_pick", dies_before_persisting)
    with pytest.raises(_ProcessDeath):
        sched_mod.run_result_polling(GAME_A, DATE, picks_dir)
    assert deaths == [DATE], "the fault never reached the polling path's terminal pick save"
    assert load_pick(DATE, picks_dir).result is None, "the result was persisted before the death"
    monkeypatch.setattr(picks_mod, "save_pick", real_save)
    result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)     # the 01:00 cron
    assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 2, "the cron scorer never re-resolved the pick"
    _oracle(_outcome(picks_dir), ({"pick": "hit"}, "hit", 6, True), ({"pick": "hit"}, "hit", 7, True),
            ResultAppliedTwice, "hit across a polling death and the cron scorer")


def test_l04_control_without_the_fault_a_rerun_is_a_no_op(monkeypatch, tmp_path):
    """Control: with the pick save completing, the same two runs apply the hit exactly once."""
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    calls = _route_http(monkeypatch, {GAME_A: HIT_FEED()})
    for _ in range(2):
        result = _check_results(monkeypatch, picks_dir, tmp_path, AFTER_NORMAL_GAMES)
        assert result.exit_code == 0, result.output
    assert _feed_calls(calls, GAME_A) >= 1
    assert _outcome(picks_dir) == ({"pick": "hit"}, "hit", 6, True)


# ---------------------------------------------------------------------------
# E77 — 7/16 singleton slate (characterization; the repair design is open, the contract is not)
#
# Component level. Mocked: fetch_schedule (MLB schedule), bts.picks.get_game_statuses_detailed (MLB
# status), count_new_confirmations (boxscore lineups), the model cascade (bts.orchestrator.run_and_pick
# and the fallback refresh's import-time bts.scheduler.run_and_pick: a canned confirmed selection of
# the same batter), run_result_polling, the live-forward
# capture trigger, bts.dm.send_dm (transport), load_decision_streak_state (contest state),
# _idle_until_next_wakeup (post-observation idle; it reads the real wall clock), and the clock
# (_now_et + time.sleep). run_day, run_single_check, the lock classifier and the delivery
# chokepoint run for real; the two spies below only observe them.
#
# The schedule mock is TRUTHFUL at every instant (Codex phase-1 r2 #7): it answers 19:10 before
# ``move_at`` and 18:10 from then on (declared assumption: MLB moved the game at 12:00 ET, after the
# 10:00 morning fetch; the true move time is not in the repo). Any fetch the scheduler makes gets
# the answer a real fetch would have got then, so a repair is not presupposed and not blocked.
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta) -> None:
        self.now = self.now + delta


def _singleton_config(picks_dir: Path) -> dict:
    return {
        "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(picks_dir / ".hb")},
        "tiers": [],
        "bluesky": {"dm_recipient": "eric"},
        "scheduler": {"pick_delivery": "dm", "early_lock_gap": 0.03, "lineup_check_offset_min": 60,
                      "cluster_min": 10, "doubleheader_recheck_min": 15, "fallback_deadline_min": 35,
                      "fallback_deadline_min_morning": 25, "results_poll_interval_min": 15,
                      "results_cap_hour_et": 5, "cascade_budget_min": 12, "operator_reserve_min": 10},
        "health_checks": {"enabled": False},
    }


def _dm_kind(text: str) -> str:
    if text.startswith("BTS health"):
        return "alert"
    if "pick" in text.lower():
        return "pick"
    return "other"


RECIPIENT = "eric"


def _dm_recorder(obs: dict, clock):
    """The transport mock: every send is recorded with its recipient, text and its OWN id."""
    def send_dm(recipient, text):
        msg_id = f"dm-{len(obs['dms']) + 1}"
        obs["dms"].append({"at": clock(), "kind": _dm_kind(text), "text": text, "recipient": recipient, "id": msg_id})
        return msg_id
    return send_dm


def _pick_text_matches(text: str, pick: Pick) -> bool:
    """AFFIRMATIVE pick-recommendation content for the declared selection, specified here and never
    produced by the formatter under test (Codex phase-1 r5 #5: any text naming the batter, e.g. 'No pick
    today for Trea Turner' or 'Do not enter Trea Turner', counted as the delivery). The single-pick
    message names the batter, the batter's team and the opposing pitcher, which is the game the
    selection is in. The day comes from the send's and the record's timestamps."""
    pattern = (rf"Today's pick: {re.escape(pick.batter_name)} \({re.escape(pick.team)}\)\n"
               rf"vs {re.escape(pick.pitcher_name)} \| \d{{1,3}}\.\d%\n\nStreak: \d+")
    return re.fullmatch(pattern, text) is not None


def _mentions(obs: dict, batter: str) -> list[dict]:
    """Every non-alert send, to anyone, whose text names the batter."""
    return [d for d in obs["dms"] if d["kind"] != "alert" and batter in d["text"]]


def _pick_sends(obs: dict, pick: Pick, *, recipient: str = RECIPIENT) -> list[dict]:
    """The sends to ``recipient`` whose text is the affirmative pick message for ``pick``."""
    return [d for d in obs["dms"] if d.get("recipient") == recipient and _pick_text_matches(d["text"], pick)]


def _on_day_before(t: datetime, day: str, cutoff: datetime) -> bool:
    return t.astimezone(ET).date().isoformat() == day and t < cutoff


def _record_matches(daily, dm: dict, pick: Pick, *, day: str, cutoff: datetime) -> bool:
    """The saved delivery record describes exactly THIS send (Codex phase-1 r4 #7): the declared selection
    and day, flagged sent, the send's own id, a delivery time at the send's instant. BOTH timestamps fall
    on the declared day and strictly before the cutoff (Codex phase-1 r5 #5: a previous-day send and a
    record persisted at the cutoff itself were accepted)."""
    if not (daily and daily.date == day and daily.notification_sent and daily.notification_id == dm.get("id")
            and daily.delivered_at and daily.pick.batter_name == pick.batter_name
            and daily.pick.game_pk == pick.game_pk):
        return False
    delivered = datetime.fromisoformat(daily.delivered_at)
    return (_on_day_before(dm["at"], day, cutoff) and _on_day_before(delivered, day, cutoff)
            and abs((delivered - dm["at"]).total_seconds()) <= 60)


def _run_singleton_day(picks_dir: Path, preview: DailyPick, *, move_at: datetime,
                       true_first_pitch: datetime, cascade=None) -> dict:
    """Run ``preview.date`` through run_day with the mocks listed above; return observations."""
    from unittest.mock import patch

    from bts import scheduler as sch
    from tests.test_scheduler import _game

    save_pick(preview, picks_dir)
    game = preview.pick.game_pk
    clock = _Clock(datetime.combine(true_first_pitch.date(), datetime.min.time(), ET) + timedelta(hours=10))
    obs = {"checks": [], "classified": [], "dms": [], "idle_at": [], "schedule_answers": [], "completed": False}

    def schedule(date):
        if date != preview.date:
            return []                                   # tomorrow's slate plays no part
        start = "19:10" if clock() < move_at else "18:10"
        obs["schedule_answers"].append((clock(), start))
        return [_game(game, start, "NYM", "PHI", date=date)]

    def statuses(_date):
        started = clock() >= true_first_pitch
        return {game: {"abstract": "L" if started else "P",
                       "detailed": "In Progress" if started else "Scheduled",
                       "code": "I" if started else "S"}}

    real_check, real_classify = sch.run_single_check, picks_mod.classify_pick_lock_state

    def spy_check(**kw):
        at = clock()
        res = real_check(**kw)
        pr = res.get("pick_result")
        obs["checks"].append({"at": at, "locked": bool(pr and pr.locked)})
        return res

    def spy_classify(daily, date):
        state = real_classify(daily, date)
        obs["classified"].append({"at": clock(), "game_pk": state.game_pk, "locked": state.locked,
                                  "reason": state.reason})
        return state

    send_dm = _dm_recorder(obs, clock)

    with patch("bts.scheduler.fetch_schedule", side_effect=schedule), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=lambda s: clock.advance(timedelta(seconds=s))), \
         patch("bts.scheduler.run_single_check", side_effect=spy_check), \
         patch("bts.picks.classify_pick_lock_state", side_effect=spy_classify), \
         patch("bts.picks.get_game_statuses_detailed", side_effect=statuses), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick", side_effect=cascade or _confirmed_cascade), \
         patch("bts.scheduler.run_and_pick", side_effect=cascade or _confirmed_cascade), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.scheduler._idle_until_next_wakeup",
               side_effect=lambda *a, **k: obs["idle_at"].append(clock())), \
         patch("bts.dm.send_dm", side_effect=send_dm), \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        sch.run_day(date=preview.date, config=_singleton_config(picks_dir))
        obs["completed"] = True
    obs["end"] = clock()
    obs["daily"] = load_pick(preview.date, picks_dir)
    return obs


E77_TRUE_FIRST_PITCH = datetime(2026, 7, 16, 18, 10, tzinfo=ET)
E77_TRUE_CUTOFF = E77_TRUE_FIRST_PITCH - timedelta(minutes=5)
E77_MOVE_AT = datetime(2026, 7, 16, 12, 0, tzinfo=ET)          # declared: after the 10:00 fetch


def _turner(**over) -> Pick:
    fields = dict(batter_name="Trea Turner", batter_id=607208, team="PHI", lineup_position=1,
                  pitcher_name="P", pitcher_id=1, p_game_hit=0.675, flags=[], projected_lineup=True,
                  game_pk=824716, game_time="2026-07-16T23:10:00Z")          # the stale 19:10 ET start
    fields.update(over)
    return Pick(**fields)


def _preview(pick: Pick, **delivery) -> DailyPick:
    return DailyPick(date="2026-07-16", run_time="2026-07-16T07:00:00+00:00", pick=pick,
                     double_down=None, runner_up=None, **delivery)


def _confirmed_cascade(*_a, **_k):
    """The cascade's declared response: a confirmed selection of the same batter at the TRUE start."""
    import pandas as pd

    from bts.strategy import PickResult, SelectionResult

    confirmed = _preview(_turner(projected_lineup=False, game_time="2026-07-16T22:10:00Z"))
    predictions = pd.DataFrame([{"batter_name": "Trea Turner", "batter_id": 607208, "team": "PHI",
                                 "game_pk": 824716, "p_game_hit": 0.675, "flags": ""}])
    sel = SelectionResult(pick_result=PickResult(daily=confirmed, locked=False), action="single",
                          source="mdp", primary_candidate=None, double_candidate=None,
                          no_pick_reason=None, streak=0)
    return predictions, sel, "local"


E77_BATTER, E77_GAME, E77_DAY = "Trea Turner", 824716, "2026-07-16"
E77_PICK = _turner()                      # the declared selection: batter, team, opposing pitcher, game
E77_TEXT = "Today's pick: Trea Turner (PHI)\nvs P | 67.5%\n\nStreak: 0"        # written out, never formatted


def _delivery_outcome(obs: dict) -> tuple:
    """Delivery outcome from the IDENTIFIED send and the saved record together (Codex phase-1 r4 #7, r5
    #5). Required: exactly one send to the recipient carrying the AFFIRMATIVE pick message for the
    declared selection, no other send naming the batter, and a saved record that describes that send,
    with both timestamps on the declared day and strictly before the true cutoff. Bad: no send naming
    the batter to anyone, and a record claiming no delivery. Anything else is 'other'."""
    daily = obs["daily"]
    picks = _pick_sends(obs, E77_PICK)
    others = [d for d in _mentions(obs, E77_BATTER) if d not in picks]
    if len(picks) == 1 and not others and _record_matches(daily, picks[0], E77_PICK, day=E77_DAY,
                                                          cutoff=E77_TRUE_CUTOFF):
        return ("delivered_before_cutoff",)
    if not _mentions(obs, E77_BATTER) and daily and not daily.notification_sent and daily.delivered_at is None:
        return ("never_delivered",)
    return ("other", bool(daily and daily.notification_sent), len(picks), len(others))


def _e77_verdict(obs: dict) -> None:
    """E77's oracle over one observed day (Codex phase-1 r2 #7).

    Both branches first prove the day really ran past the cutoff (ordinary assertions). The declared
    BAD outcome must also show the declared mechanism — the lone check at the true first pitch (from
    the stale 19:10 plan), a started-game lock of the undelivered candidate, containment-only DMs —
    before the dedicated exception. A verified pre-cutoff delivery (pick-file flags + identified pick
    DM, both from production) reaches the required branch BY ANY PATH — a re-planned check, a woken
    fallback — so any repair turns the marked node into XPASS(strict); no repair shape is presupposed
    (2026-09-29, as for L04). Anything else is an ordinary failure.
    """
    assert obs["completed"] and obs["end"] >= E77_TRUE_CUTOFF, obs["end"]
    outcome = _delivery_outcome(obs)
    if outcome == ("never_delivered",):
        assert [c["at"].strftime("%H:%M") for c in obs["checks"]] == ["18:10"], obs["checks"]
        assert any(c["game_pk"] == 824716 and c["locked"] and c["reason"] == "game_started_or_final"
                   and c["at"] >= E77_TRUE_FIRST_PITCH for c in obs["classified"]), obs["classified"]
        assert all(d["kind"] == "alert" for d in obs["dms"]), obs["dms"]          # containment only
    _oracle(outcome, ("delivered_before_cutoff",), ("never_delivered",),
            SingletonSlateUndelivered, "7/16 singleton slate")


@pytest.mark.xfail(strict=True, raises=SingletonSlateUndelivered,
                   reason="E77 unfixed: a moved-up singleton slate gets its only check at first pitch")
def test_e77_singleton_slate_moved_up_is_delivered_before_cutoff(tmp_path):
    obs = _run_singleton_day(tmp_path / "picks", _preview(_turner()), move_at=E77_MOVE_AT,
                             true_first_pitch=E77_TRUE_FIRST_PITCH)
    _e77_verdict(obs)


def test_e77_positive_execution_control_correct_schedule_delivers(tmp_path):
    """Positive execution control (component level): the SAME machinery and mocks, with the move
    already in the morning schedule (move_at 09:00, before the 10:00 fetch), runs its 17:10 check
    and DMs the pick before the 18:05 cutoff — and the verdict accepts it."""
    obs = _run_singleton_day(tmp_path / "picks", _preview(_turner(game_time="2026-07-16T22:10:00Z")),
                             move_at=datetime(2026, 7, 16, 9, 0, tzinfo=ET),
                             true_first_pitch=E77_TRUE_FIRST_PITCH)
    assert [c["at"].strftime("%H:%M") for c in obs["checks"]][:1] == ["17:10"], obs["checks"]
    assert _delivery_outcome(obs) == ("delivered_before_cutoff",), (obs["dms"], obs["daily"])
    _e77_verdict(obs)


def _obs(*, checks, classified=(), dms=(), daily, end=E77_TRUE_FIRST_PITCH + timedelta(hours=1)):
    return {"completed": True, "end": end, "checks": [{"at": t, "locked": False} for t in checks],
            "classified": list(classified), "dms": list(dms), "daily": daily}


def _at(hh, mm):
    return datetime(2026, 7, 16, hh, mm, tzinfo=ET)


def _dm(at, text, *, recipient=RECIPIENT, msg_id="dm-1") -> dict:
    return {"at": at, "kind": _dm_kind(text), "text": text, "recipient": recipient, "id": msg_id}


def _flagged(at_utc: str, msg_id: str = "dm-1") -> DailyPick:
    return _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                    notification_id=msg_id, delivery_attempted=True, delivered_at=at_utc)


E77_CASES = {
    # Codex phase-1 r4 #7
    "status_message": ([(_at(17, 11), "No pick today")], "2026-07-16T21:11:00+00:00"),
    "wrong_batter": ([(_at(17, 11), E77_TEXT.replace("Trea Turner", "Kyle Schwarber"))], "2026-07-16T21:11:00+00:00"),
    "wrong_recipient": ([(_at(17, 11), E77_TEXT, "mallory")], "2026-07-16T21:11:00+00:00"),
    "inconsistent_record": ([(_at(17, 11), E77_TEXT, RECIPIENT, "dm-7")], "2026-07-16T21:11:00+00:00"),
    "duplicate_send": ([(_at(17, 11), E77_TEXT), (_at(17, 12), E77_TEXT, RECIPIENT, "dm-2")], "2026-07-16T21:11:00+00:00"),
    # Codex phase-1 r5 #5: a name mention is not delivery; both timestamps on the day, before the cutoff
    "named_status": ([(_at(17, 11), "No pick today for Trea Turner")], "2026-07-16T21:11:00+00:00"),
    "negative_instruction": ([(_at(17, 11), "Do not enter Trea Turner")], "2026-07-16T21:11:00+00:00"),
    "wrong_game_and_date": ([(_at(17, 11), "pick: Trea Turner, game 999 on 2026-07-15")], "2026-07-16T21:11:00+00:00"),
    "another_games_pick": ([(_at(17, 11), E77_TEXT.replace("(PHI)\nvs P", "(PHI)\nvs Q"))], "2026-07-16T21:11:00+00:00"),
    "status_beside_the_pick": ([(_at(17, 11), E77_TEXT), (_at(17, 12), "No pick today for Trea Turner", RECIPIENT, "dm-2")],
                               "2026-07-16T21:11:00+00:00"),
    "previous_day": ([(_at(17, 11) - timedelta(days=1), E77_TEXT)], "2026-07-15T21:11:00+00:00"),
    "persisted_at_the_cutoff": ([(_at(18, 4), E77_TEXT)], "2026-07-16T22:05:00+00:00"),
}


@pytest.mark.parametrize("case", list(E77_CASES))
def test_e77_verdict_a_flagged_record_without_the_identified_send_fails_ordinarily(case):
    """Negative controls: the delivery is the identified AFFIRMATIVE send AND its record, both on the day
    and before the cutoff (Codex phase-1 r4 #7, r5 #5)."""
    sends, delivered_at = E77_CASES[case]
    dms = [_dm(*send[:2], **dict(zip(("recipient", "msg_id"), send[2:]))) for send in sends]
    obs = _obs(checks=[_at(17, 10)], dms=dms, daily=_flagged(delivered_at))
    assert _delivery_outcome(obs)[0] == "other"
    with pytest.raises(AssertionError):
        _e77_verdict(obs)


def test_e77_verdict_a_record_of_another_day_fails_ordinarily():
    """The saved record must be the declared day's own, even when its timestamps fall on that day."""
    daily = _flagged("2026-07-16T21:11:00+00:00")
    daily.date = "2026-07-15"
    obs = _obs(checks=[_at(17, 10)], dms=[_dm(_at(17, 11), E77_TEXT)], daily=daily)
    assert _delivery_outcome(obs)[0] == "other"
    with pytest.raises(AssertionError):
        _e77_verdict(obs)


def test_e77_verdict_fixed_direction_passes():
    """A verified pre-cutoff delivery reaches the required branch (the marked node would XPASS)."""
    daily = _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                     notification_id="dm-1", delivery_attempted=True, delivered_at="2026-07-16T21:11:00+00:00")
    _e77_verdict(_obs(checks=[_at(17, 10)], dms=[_dm(_at(17, 11), E77_TEXT)], daily=daily))


def test_e77_verdict_bad_direction_raises_the_dedicated_exception():
    classified = [{"at": _at(18, 10), "game_pk": 824716, "locked": True, "reason": "game_started_or_final"}]
    with pytest.raises(SingletonSlateUndelivered):
        _e77_verdict(_obs(checks=[_at(18, 10)], classified=classified,
                          dms=[_dm(_at(19, 0), "BTS health CRITICAL")], daily=_preview(_turner())))


def test_e77_verdict_any_pre_cutoff_delivery_path_passes():
    """A repair need not deliver from a lineup check: a verified pre-cutoff delivery with no check at
    all (e.g. a woken fallback) also reaches the required branch."""
    daily = _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                     notification_id="dm-1", delivery_attempted=True, delivered_at="2026-07-16T21:35:00+00:00")
    _e77_verdict(_obs(checks=[], dms=[_dm(_at(17, 35), E77_TEXT)], daily=daily))


@pytest.mark.parametrize("case", ["late_delivery", "no_check", "bad_outcome_other_mechanism"])
def test_e77_verdict_other_observations_fail_ordinarily(case):
    undelivered = _preview(_turner())
    if case == "late_delivery":
        daily = _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                         notification_id="dm-1", delivery_attempted=True, delivered_at="2026-07-16T22:12:00+00:00")
        obs = _obs(checks=[_at(18, 10)], dms=[_dm(_at(18, 12), E77_TEXT)], daily=daily)
    elif case == "no_check":
        obs = _obs(checks=[], daily=undelivered)
    else:   # never delivered, but not through the declared lone-check-at-first-pitch mechanism
        obs = _obs(checks=[_at(17, 10)], daily=undelivered)
    with pytest.raises(AssertionError):
        _e77_verdict(obs)


# ---------------------------------------------------------------------------
# L03 — postponed cached fallback (design §10.3; deferred as F1 in the 7/09 sol audit, 30452eb)
#
# Component level, on the E77 harness. Mocked: fetch_schedule, bts.picks.get_game_statuses_detailed,
# count_new_confirmations, the cascade (bts.orchestrator.run_and_pick, which run_single_check imports at
# call time, AND bts.scheduler.run_and_pick, the fallback refresh's import-time binding), run_result_polling, the
# live-forward trigger, bts.dm.send_dm, contest state, _idle_until_next_wakeup and the clock. run_day,
# run_single_check, the lock decision, the fallback refresh + planner and the delivery chokepoint run
# for real. The status mock is TRUTHFUL at every instant: Scheduled until ``postponed_at``, then
# Postponed (declared: rain at 18:20 ET, after the 18:10 check and before the 18:35 fallback). The
# cascade's first call (the 18:10 check) selects a projected-lineup pick in the game (so the lock waits
# for the fallback); every later call returns what production's run_and_pick returns when every tier
# fails, (None, None, tier): the refresh fails and the fallback falls back to the cached pick.
# ---------------------------------------------------------------------------
L03_DATE = "2026-07-20"
L03_GAME = 824801
L03_BATTER = "Batter 660101"
L03_FIRST_PITCH = datetime(2026, 7, 20, 19, 10, tzinfo=ET)
L03_CUTOFF = L03_FIRST_PITCH - timedelta(minutes=5)
L03_CHECK = datetime(2026, 7, 20, 18, 10, tzinfo=ET)          # first pitch - 60 (box TOML offset)
L03_FALLBACK = datetime(2026, 7, 20, 18, 35, tzinfo=ET)       # first pitch - 35 (fallback deadline)
L03_POSTPONED_AT = datetime(2026, 7, 20, 18, 20, tzinfo=ET)


def _l03_selection():
    import pandas as pd

    from bts.strategy import PickResult, SelectionResult

    pick = Pick(batter_name=L03_BATTER, batter_id=660101, team="NYM", lineup_position=2, pitcher_name="P",
                pitcher_id=1, p_game_hit=0.74, flags=["PROJECTED"], projected_lineup=True, game_pk=L03_GAME,
                game_time="2026-07-20T23:10:00Z")
    daily = DailyPick(date=L03_DATE, run_time="2026-07-20T22:10:00+00:00", pick=pick, double_down=None,
                      runner_up=None)
    predictions = pd.DataFrame([{"batter_name": L03_BATTER, "batter_id": 660101, "team": "NYM",
                                 "game_pk": L03_GAME, "p_game_hit": 0.74, "flags": "PROJECTED"}])
    sel = SelectionResult(pick_result=PickResult(daily=daily, locked=False), action="single",
                          source="mdp", primary_candidate=None, double_candidate=None,
                          no_pick_reason=None, streak=0)
    return predictions, sel, "local"


L03_PICK = _l03_selection()[1].pick_result.daily.pick
L03_TEXT = "Today's pick: Batter 660101 (NYM)\nvs P | 74.0%\n\nStreak: 0"       # written out, never formatted


def _run_l03_day(picks_dir: Path, *, postponed_at: datetime | None) -> dict:
    """Run L03_DATE through run_day with the mocks listed above; return observations."""
    from unittest.mock import patch

    from bts import scheduler as sch
    from tests.test_scheduler import _game

    clock = _Clock(datetime(2026, 7, 20, 10, 0, tzinfo=ET))
    obs = {"cascade": [], "dms": [], "status_answers": [], "completed": False}

    def schedule(date):
        return [_game(L03_GAME, "19:10", "NYM", "PHI", date=date)] if date == L03_DATE else []

    def statuses(_date):
        now = clock()
        if postponed_at is not None and now >= postponed_at:
            st = {"abstract": "F", "detailed": "Postponed", "code": "D"}
        elif now >= L03_FIRST_PITCH:
            st = {"abstract": "L", "detailed": "In Progress", "code": "I"}
        else:
            st = {"abstract": "P", "detailed": "Scheduled", "code": "S"}
        obs["status_answers"].append((now, st["detailed"]))
        return {L03_GAME: st}

    def cascade(*_a, **_k):
        obs["cascade"].append(clock())
        return _l03_selection() if len(obs["cascade"]) == 1 else (None, None, "local")

    send_dm = _dm_recorder(obs, clock)

    with patch("bts.scheduler.fetch_schedule", side_effect=schedule), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=lambda s: clock.advance(timedelta(seconds=s))), \
         patch("bts.picks.get_game_statuses_detailed", side_effect=statuses), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick", side_effect=cascade), \
         patch("bts.scheduler.run_and_pick", side_effect=cascade), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.scheduler._idle_until_next_wakeup"), \
         patch("bts.dm.send_dm", side_effect=send_dm), \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        sch.run_day(date=L03_DATE, config=_singleton_config(picks_dir))
        obs["completed"] = True
    obs["end"] = clock()
    obs["daily"] = load_pick(L03_DATE, picks_dir)
    return obs


def _l03_outcome(obs: dict, postponed_at: datetime) -> tuple:
    """Decided from the COMPLETE send history relative to the postponement, with the saved record
    required to agree (Codex phase-1 r4 #8: a valid delivery BEFORE the postponement was refused).
    A delivery is the AFFIRMATIVE pick message for the declared selection (Codex phase-1 r5 #5: a status
    message naming the batter reached the dedicated exception). Bad: the one such send came at/after the
    postponement and the record describes it. Required: no such send, and the record agrees: it
    describes the one earlier send, or claims no delivery when there was none. Any other send naming the
    batter, to anyone, is 'other'."""
    daily = obs["daily"]
    picks = _pick_sends(obs, L03_PICK)
    if [d for d in _mentions(obs, L03_BATTER) if d not in picks]:
        return ("other", "a send naming the batter that is not the declared pick delivery")
    after = [d for d in picks if d["at"] >= postponed_at]
    if after:
        if len(picks) == 1 and _record_matches(daily, after[0], L03_PICK, day=L03_DATE, cutoff=L03_CUTOFF):
            return ("postponed_pick_delivered",)
        return ("other", len(picks), bool(daily and daily.notification_sent))
    if picks:
        if len(picks) == 1 and _record_matches(daily, picks[0], L03_PICK, day=L03_DATE, cutoff=L03_CUTOFF):
            return ("no_postponed_delivery",)
        return ("other", len(picks), bool(daily and daily.notification_sent))
    if daily is None or (not daily.notification_sent and daily.delivered_at is None):
        return ("no_postponed_delivery",)
    return ("other", "a record claiming a delivery no send supports")


def _l03_verdict(obs: dict) -> None:
    """Both branches prove the day ran through the check and the fallback deadline; the bad branch
    also proves the cached-fallback mechanism (the check at 18:10 selected the pick, the refresh at the
    deadline got no predictions, the send came after the postponement and before the cutoff)."""
    assert obs["completed"] and obs["end"] >= L03_FALLBACK, obs["end"]
    assert obs["cascade"] and obs["cascade"][0] == L03_CHECK, obs["cascade"]
    outcome = _l03_outcome(obs, L03_POSTPONED_AT)
    if outcome == ("postponed_pick_delivered",):
        assert any(t >= L03_FALLBACK for t in obs["cascade"][1:]), obs["cascade"]   # the failed refresh
        sent = [d["at"] for d in _pick_sends(obs, L03_PICK)]
        assert sent and all(L03_POSTPONED_AT <= t < L03_CUTOFF for t in sent), sent
        assert obs["daily"].pick.batter_name == L03_BATTER                            # the cached pick
    _oracle(outcome, ("no_postponed_delivery",), ("postponed_pick_delivered",),
            PostponedPickDelivered, "postponed cached fallback")


@pytest.mark.xfail(strict=True, raises=PostponedPickDelivered,
                   reason="L03 unfixed: the fallback delivers the cached pick of a postponed game when the refresh fails")
def test_l03_postponed_game_is_never_delivered_by_the_cached_fallback(tmp_path):
    _l03_verdict(_run_l03_day(tmp_path / "picks", postponed_at=L03_POSTPONED_AT))


def test_l03_positive_execution_control_cached_fallback_delivers_a_playable_game(tmp_path):
    """Positive execution control: the SAME machinery with the game never postponed — the failed refresh
    falls back to the cached pick and the fallback delivers it at the deadline (correct: the game is
    playable), so the incident node differs only in the postponement."""
    obs = _run_l03_day(tmp_path / "picks", postponed_at=None)
    assert obs["cascade"][0] == L03_CHECK and any(t >= L03_FALLBACK for t in obs["cascade"][1:]), obs["cascade"]
    sent = [d["at"] for d in _pick_sends(obs, L03_PICK)]                     # the affirmative pick message
    assert sent and all(L03_FALLBACK <= t < L03_CUTOFF for t in sent), obs["dms"]
    assert obs["daily"].notification_sent


def _l03_obs(dms, daily) -> dict:
    return {"daily": daily, "dms": dms, "completed": True, "end": L03_FALLBACK + timedelta(minutes=30),
            "cascade": [L03_CHECK, L03_FALLBACK]}


def _l03_daily(at_utc: str | None, msg_id: str = "dm-1") -> DailyPick:
    daily = _l03_selection()[1].pick_result.daily
    if at_utc:
        daily.notification_sent, daily.notification_id, daily.delivered_at = True, msg_id, at_utc
    return daily


def test_l03_verdict_a_valid_delivery_before_the_postponement_passes():
    """Codex phase-1 r4 #8: the pick DM'd at 18:15, before the 18:20 postponement, and nothing after it."""
    at = datetime(2026, 7, 20, 18, 15, tzinfo=ET)
    obs = _l03_obs([_dm(at, L03_TEXT)], _l03_daily("2026-07-20T22:15:00+00:00"))
    assert _l03_outcome(obs, L03_POSTPONED_AT) == ("no_postponed_delivery",)
    _l03_verdict(obs)


@pytest.mark.parametrize("case", ["record_disagrees", "record_without_send", "send_to_another_recipient",
                                  "named_status_after_the_postponement", "previous_day", "persisted_at_the_cutoff"])
def test_l03_verdict_a_history_the_record_does_not_describe_fails_ordinarily(case):
    """Negative controls (Codex phase-1 r4 #8, r5 #5): none of these is the declared postponed delivery,
    so none may reach the dedicated exception."""
    at = datetime(2026, 7, 20, 18, 15, tzinfo=ET)
    late = L03_POSTPONED_AT + timedelta(minutes=15)
    obs = {"record_disagrees": _l03_obs([_dm(at, L03_TEXT)], _l03_daily("2026-07-20T22:15:00+00:00", "dm-9")),
           "record_without_send": _l03_obs([], _l03_daily("2026-07-20T22:15:00+00:00")),
           "send_to_another_recipient": _l03_obs([_dm(at, L03_TEXT, recipient="mallory")], _l03_daily(None)),
           "named_status_after_the_postponement": _l03_obs([_dm(late, f"No pick today for {L03_BATTER}")],
                                                           _l03_daily(late.isoformat())),
           "previous_day": _l03_obs([_dm(late - timedelta(days=1), L03_TEXT)],
                                    _l03_daily((late - timedelta(days=1)).isoformat())),
           "persisted_at_the_cutoff": _l03_obs([_dm(L03_CUTOFF - timedelta(minutes=1), L03_TEXT)],
                                               _l03_daily(L03_CUTOFF.isoformat()))}[case]
    assert _l03_outcome(obs, L03_POSTPONED_AT)[0] == "other"
    with pytest.raises(AssertionError):
        _l03_verdict(obs)
