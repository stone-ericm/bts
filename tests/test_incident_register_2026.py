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


def _oracle(actual, required, bad, exc: type[IncidentExpectedFailure], what: str) -> None:
    """Pass on ``required``; raise ``exc`` ONLY for the declared bad value; else AssertionError."""
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
# E77 — 7/16 singleton slate (characterization; the repair design is open, the contract is not)
#
# Component level. Mocked: fetch_schedule (MLB schedule), bts.picks.get_game_statuses_detailed (MLB
# status), count_new_confirmations (boxscore lineups), bts.orchestrator.run_and_pick (the model
# cascade: a canned confirmed selection of the same batter), run_result_polling, the live-forward
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

    def send_dm(recipient, text):
        obs["dms"].append({"at": clock(), "kind": _dm_kind(text), "text": text})
        return "dm-1"

    with patch("bts.scheduler.fetch_schedule", side_effect=schedule), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=lambda s: clock.advance(timedelta(seconds=s))), \
         patch("bts.scheduler.run_single_check", side_effect=spy_check), \
         patch("bts.picks.classify_pick_lock_state", side_effect=spy_classify), \
         patch("bts.picks.get_game_statuses_detailed", side_effect=statuses), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick", side_effect=cascade or _confirmed_cascade), \
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


def _delivery_outcome(obs: dict) -> tuple:
    """Delivery outcome from the pick file AND the identified pick DM (alerts never count)."""
    daily = obs["daily"]
    pick_dms = [d for d in obs["dms"] if d["kind"] == "pick"]
    before = [d for d in pick_dms if d["at"] < E77_TRUE_CUTOFF]
    delivered_at = (datetime.fromisoformat(daily.delivered_at).astimezone(ET)
                    if daily and daily.delivered_at else None)
    if daily and daily.notification_sent and before and delivered_at and delivered_at < E77_TRUE_CUTOFF:
        return ("delivered_before_cutoff",)
    if daily and not daily.notification_sent and not pick_dms and daily.delivered_at is None:
        return ("never_delivered",)
    return ("other", bool(daily and daily.notification_sent), len(pick_dms), str(delivered_at))


def _e77_verdict(obs: dict) -> None:
    """E77's oracle over one observed day (Codex phase-1 r2 #7).

    Both branches first prove the day really ran (ordinary assertions). The declared BAD outcome must
    also show the declared mechanism — the lone check at the true first pitch (from the stale 19:10
    plan), a started-game lock of the undelivered candidate, containment-only DMs — before the
    dedicated exception; a verified pre-cutoff delivery (pick-file flags + identified pick DM, from a
    check before the cutoff) reaches the required branch, so a repair turns the marked node into
    XPASS(strict). Anything else is an ordinary failure.
    """
    assert obs["completed"] and obs["end"] >= E77_TRUE_CUTOFF, obs["end"]
    assert obs["checks"], "run_day ran no lineup check"
    outcome = _delivery_outcome(obs)
    if outcome == ("never_delivered",):
        assert [c["at"].strftime("%H:%M") for c in obs["checks"]] == ["18:10"], obs["checks"]
        assert any(c["game_pk"] == 824716 and c["locked"] and c["reason"] == "game_started_or_final"
                   and c["at"] >= E77_TRUE_FIRST_PITCH for c in obs["classified"]), obs["classified"]
        assert all(d["kind"] == "alert" for d in obs["dms"]), obs["dms"]          # containment only
    elif outcome == ("delivered_before_cutoff",):
        assert any(c["at"] < E77_TRUE_CUTOFF for c in obs["checks"]), obs["checks"]
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


def test_e77_verdict_fixed_direction_passes():
    """A verified pre-cutoff delivery reaches the required branch (the marked node would XPASS)."""
    daily = _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                     notification_id="dm-1", delivery_attempted=True, delivered_at="2026-07-16T21:11:00+00:00")
    _e77_verdict(_obs(checks=[_at(17, 10)], dms=[{"at": _at(17, 11), "kind": "pick", "text": "pick"}], daily=daily))


def test_e77_verdict_bad_direction_raises_the_dedicated_exception():
    classified = [{"at": _at(18, 10), "game_pk": 824716, "locked": True, "reason": "game_started_or_final"}]
    with pytest.raises(SingletonSlateUndelivered):
        _e77_verdict(_obs(checks=[_at(18, 10)], classified=classified,
                          dms=[{"at": _at(19, 0), "kind": "alert", "text": "BTS health"}], daily=_preview(_turner())))


@pytest.mark.parametrize("case", ["late_delivery", "no_check", "bad_outcome_other_mechanism"])
def test_e77_verdict_other_observations_fail_ordinarily(case):
    undelivered = _preview(_turner())
    if case == "late_delivery":
        daily = _preview(_turner(), notification_sent=True, notification_channel="bluesky_dm",
                         notification_id="dm-1", delivery_attempted=True, delivered_at="2026-07-16T22:12:00+00:00")
        obs = _obs(checks=[_at(18, 10)], dms=[{"at": _at(18, 12), "kind": "pick", "text": "pick"}], daily=daily)
    elif case == "no_check":
        obs = _obs(checks=[], daily=undelivered)
    else:   # never delivered, but not through the declared lone-check-at-first-pitch mechanism
        obs = _obs(checks=[_at(17, 10)], daily=undelivered)
    with pytest.raises(AssertionError):
        _e77_verdict(obs)
