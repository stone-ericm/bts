"""W1.5 incident register: expected-failure fixtures for unfixed defects whose contract is fixed.

Design: docs/superpowers/specs/2026-09-29-incident-register-design.md §9.7, §10.

Every strict expected failure raises its OWN exception class (never an AssertionError), and
only from ``_oracle`` for the declared bad value. Any other mismatch is an ordinary assertion
failure, so a defect that changes shape (e.g. ``None`` instead of ``miss``) fails loudly
instead of inheriting the expected failure. HTTP is the only thing mocked: grading runs
through ``grade_pick_in_feed`` and the ``bts check-results`` command; the replay cases run
``reconcile_results`` itself.
"""
from __future__ import annotations

import io
import json
from datetime import datetime
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
    """L01 downstream: a Pass leg reset (or failed to advance) the persisted streak."""


class SameDayReplayRollback(IncidentExpectedFailure):
    """L02: reconcile's season replay drops today's already-applied terminal result."""


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
# Synthetic MLB responses (complete inputs: final game, stats present, timed plays)
# ---------------------------------------------------------------------------
DATE = "2026-06-10"
NEXT_DAY_0100 = datetime(2026, 6, 11, 1, 0, tzinfo=ET)
GAME_A, GAME_B = 824001, 824002
BATTER_A, BATTER_B = 660001, 660002
PRE = "2026-06-10T23:30:00Z"      # before the suspension
RESUME = "2026-06-11T17:00:00Z"   # resumeDateTime of a suspended game
POST = "2026-06-11T17:30:00Z"     # resumed portion — never evaluated for BTS


def _batting(*, hits=0, at_bats=0, sac_flies=0, walks=0, hbp=0, sac_bunts=0):
    pa = at_bats + sac_flies + walks + hbp + sac_bunts
    return {"hits": hits, "atBats": at_bats, "sacFlies": sac_flies, "baseOnBalls": walks,
            "hitByPitch": hbp, "sacBunts": sac_bunts, "plateAppearances": pa}


def _play(batter: int, event: str, start: str) -> dict:
    return {"result": {"eventType": event},
            "matchup": {"batter": {"id": batter, "fullName": f"Batter {batter}"}},
            "about": {"startTime": start}}


def _feed(batter: int, *, batting: dict | None, plays=(), resume: str | None = None,
          on_roster: bool = True) -> dict:
    """A final game feed. ``batting=None`` with ``on_roster`` = on the roster, no plate appearance."""
    players = {}
    if on_roster:
        players[f"ID{batter}"] = {"person": {"id": batter, "fullName": f"Batter {batter}"},
                                  "stats": {"batting": batting or {}}}
    return {
        "gameData": {"status": {"abstractGameCode": "F", "detailedState": "Final"},
                     "datetime": {"resumeDateTime": resume} if resume else {}},
        "liveData": {"plays": {"allPlays": list(plays)},
                     "boxscore": {"teams": {"away": {"players": players},
                                            "home": {"players": {}}}}},
    }


# Each case: (feed for BATTER_A in GAME_A, required grade, declared bad grade or None for a control)
CASES = {
    "walks_only": (lambda: _feed(BATTER_A, batting=_batting(walks=2, hbp=1),
                                 plays=[_play(BATTER_A, "walk", PRE),
                                        _play(BATTER_A, "hit_by_pitch", PRE),
                                        _play(BATTER_A, "walk", PRE)]), "void", "miss"),
    "did_not_play": (lambda: _feed(BATTER_A, batting=None), "void", "miss"),
    "sac_bunt_only": (lambda: _feed(BATTER_A, batting=_batting(sac_bunts=1, walks=1),
                                    plays=[_play(BATTER_A, "sac_bunt", PRE),
                                           _play(BATTER_A, "walk", PRE)]), "void", "miss"),
    "suspended_walks_only": (lambda: _feed(BATTER_A, batting=_batting(walks=1, hits=1, at_bats=1),
                                           plays=[_play(BATTER_A, "walk", PRE),
                                                  _play(BATTER_A, "single", POST)],
                                           resume=RESUME), "void", "miss"),
    "suspended_ab_no_hit": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=2),
                                          plays=[_play(BATTER_A, "field_out", PRE),
                                                 _play(BATTER_A, "home_run", POST)],
                                          resume=RESUME), "void", "miss"),
    # controls — the required value is what HEAD already returns
    "sac_fly_no_hit": (lambda: _feed(BATTER_A, batting=_batting(sac_flies=1),
                                     plays=[_play(BATTER_A, "sac_fly", PRE)]), "miss", None),
    "official_ab_no_hit": (lambda: _feed(BATTER_A, batting=_batting(at_bats=4),
                                         plays=[_play(BATTER_A, "field_out", PRE)] * 4), "miss", None),
    "resumed_only": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=1),
                                   plays=[_play(BATTER_A, "single", POST)], resume=RESUME), "void", None),
    "pre_suspension_hit": (lambda: _feed(BATTER_A, batting=_batting(hits=1, at_bats=2),
                                         plays=[_play(BATTER_A, "single", PRE),
                                                _play(BATTER_A, "field_out", POST)],
                                         resume=RESUME), "hit", None),
    "absent_from_rosters": (lambda: _feed(BATTER_A, batting=None, on_roster=False), None, None),
}
XFAIL_CASES = [k for k, (_, _, bad) in CASES.items() if bad is not None]
CONTROL_CASES = [k for k, (_, _, bad) in CASES.items() if bad is None]


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


def _pick(batter: int, game_pk: int) -> Pick:
    return Pick(batter_name=f"Batter {batter}", batter_id=batter, team="NYM", lineup_position=1,
                pitcher_name="P", pitcher_id=1, p_game_hit=0.75, flags=[], projected_lineup=False,
                game_pk=game_pk, game_time="2026-06-10T23:10:00Z")


def _delivered_daily(pick: Pick, double_down: Pick | None = None) -> DailyPick:
    return DailyPick(date=DATE, run_time="2026-06-10T21:00:00+00:00", pick=pick,
                     double_down=double_down, runner_up=None, notification_sent=True,
                     notification_channel="dm", notification_id="dm-1", delivery_attempted=True,
                     delivered_at="2026-06-10T21:01:00+00:00")


def _check_results(monkeypatch, picks_dir: Path, tmp_path: Path):
    monkeypatch.setattr(cli_mod, "_now_et", lambda: NEXT_DAY_0100)
    return CliRunner().invoke(cli, ["check-results", "--date", DATE, "--picks-dir", str(picks_dir),
                                    "--shadow-status-output", str(tmp_path / "shadow_status.json")])


# ---------------------------------------------------------------------------
# L01 — BTS Pass grading (official rules §6 A/B/C; pinned excerpt in the evidence dir)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", _params(list(CASES), PassGradedAsMiss))
def test_l01_grade_pick_in_feed(case):
    feed, required, bad = CASES[case]
    grade = grade_pick_in_feed(feed(), BATTER_A, f"Batter {BATTER_A}")
    if bad is None:
        assert grade == required
        return
    _oracle(grade, required, bad, PassGradedAsMiss, f"grade_pick_in_feed[{case}]")


@pytest.mark.parametrize("case", _params(XFAIL_CASES + ["sac_fly_no_hit", "resumed_only"], PassScoredAsNoHit))
def test_l01_check_results_single_pick_streak(case, monkeypatch, tmp_path):
    """A single-pick Pass must hold the streak at 5 (production cron entry, HTTP mocked)."""
    feed, required_grade, bad = CASES[case]
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    _route_http(monkeypatch, {GAME_A: feed()})

    result = _check_results(monkeypatch, picks_dir, tmp_path)
    assert result.exit_code == 0, result.output
    required_streak = {"void": 5, "miss": 0, "hit": 6}[required_grade]
    if bad is None:
        assert load_pick(DATE, picks_dir).result == required_grade
        assert load_streak(picks_dir) == required_streak
        return
    _oracle(load_streak(picks_dir), required_streak, 0, PassScoredAsNoHit,
            f"check-results streak[{case}]")


@pytest.mark.parametrize("case", _params(["walks_only", "did_not_play", "suspended_ab_no_hit"],
                                         PassScoredAsNoHit))
def test_l01_check_results_hit_plus_pass_adds_one(case, monkeypatch, tmp_path):
    """Double down: primary Hit + a Pass leg = streak +1 (5 -> 6), never a reset."""
    feed, _required, _bad = CASES[case]
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_B, GAME_B), double_down=_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    hit_feed = _feed(BATTER_B, batting=_batting(hits=1, at_bats=4),
                     plays=[_play(BATTER_B, "single", PRE)])
    _route_http(monkeypatch, {GAME_B: hit_feed, GAME_A: feed()})

    result = _check_results(monkeypatch, picks_dir, tmp_path)
    assert result.exit_code == 0, result.output
    _oracle(load_streak(picks_dir), 6, 0, PassScoredAsNoHit, f"hit+pass streak[{case}]")


def test_l01_absent_player_stays_pending(monkeypatch, tmp_path):
    """Control: a batter on neither roster (and no other final game) is pending, never a Pass or a miss."""
    feed, _required, _bad = CASES["absent_from_rosters"]
    picks_dir = tmp_path / "picks"
    save_pick(_delivered_daily(_pick(BATTER_A, GAME_A)), picks_dir)
    save_streak(5, picks_dir, saver_available=True)
    _route_http(monkeypatch, {GAME_A: feed()})

    result = _check_results(monkeypatch, picks_dir, tmp_path)
    assert result.exit_code == 0, result.output
    assert load_pick(DATE, picks_dir).result is None
    assert load_streak(picks_dir) == 5


# ---------------------------------------------------------------------------
# L02 — same-day replay rollback in reconcile_results (Codex reconcile-cutoff r1 #3)
# ---------------------------------------------------------------------------
def _resolved(date: str, result: str, batter: int, game_pk: int) -> DailyPick:
    daily = _delivered_daily(_pick(batter, game_pk))
    daily.date = date
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


def _reconcile_twice(picks_dir: Path, now: datetime, required: tuple[int, bool],
                     bad: tuple[int, bool], monkeypatch) -> None:
    calls = _no_http(monkeypatch)
    before = _snapshot(picks_dir)
    for run in (1, 2):
        corrections = reconcile_results(picks_dir, clock=lambda: now)
        assert corrections == [], f"run {run}: unexpected corrections {corrections}"
        assert _snapshot(picks_dir) == before, f"run {run}: a pick file changed"
        assert calls == [], f"run {run}: fetched {len(calls)} feed(s)"
        state = (load_streak(picks_dir), load_saver_available(picks_dir))
        _oracle(state, required, bad, SameDayReplayRollback, f"reconcile run {run} (streak, saver)")


@pytest.mark.xfail(strict=True, raises=SameDayReplayRollback,
                   reason="L02 unfixed: replay excludes today's already-applied hit")
def test_l02_same_day_hit_survives_a_late_reconcile(tmp_path, monkeypatch):
    picks_dir = tmp_path / "picks"
    save_pick(_resolved("2026-06-10", "hit", BATTER_A, GAME_A), picks_dir)
    save_pick(_resolved("2026-06-11", "hit", BATTER_B, GAME_B), picks_dir)
    save_streak(2, picks_dir, saver_available=True)
    _reconcile_twice(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET),
                     required=(2, True), bad=(1, True), monkeypatch=monkeypatch)


@pytest.mark.xfail(strict=True, raises=SameDayReplayRollback,
                   reason="L02 unfixed: replay restores a saver today's miss consumed")
def test_l02_same_day_saver_consumption_survives_a_late_reconcile(tmp_path, monkeypatch):
    picks_dir = tmp_path / "picks"
    for day in range(1, 11):                       # 06-01 .. 06-10: ten single hits -> streak 10
        save_pick(_resolved(f"2026-06-{day:02d}", "hit", BATTER_A, GAME_A + day), picks_dir)
    save_pick(_resolved("2026-06-11", "miss", BATTER_B, GAME_B), picks_dir)
    save_streak(10, picks_dir, saver_available=False)   # today's miss at 10 consumed the saver
    _reconcile_twice(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET),
                     required=(10, False), bad=(10, True), monkeypatch=monkeypatch)


def test_l02_unplayed_preview_is_excluded(tmp_path, monkeypatch):
    """Control: today's unplayed preview is not replayed; yesterday's hit stands."""
    picks_dir = tmp_path / "picks"
    save_pick(_resolved("2026-06-10", "hit", BATTER_A, GAME_A), picks_dir)
    preview = _delivered_daily(_pick(BATTER_B, GAME_B))
    preview.date = "2026-06-11"
    save_pick(preview, picks_dir)
    save_streak(1, picks_dir, saver_available=True)
    _reconcile_twice(picks_dir, datetime(2026, 6, 11, 23, 0, tzinfo=ET),
                     required=(1, True), bad=(0, True), monkeypatch=monkeypatch)


# ---------------------------------------------------------------------------
# E77 — 7/16 singleton slate (characterization; the repair design is open, the contract is not)
# ---------------------------------------------------------------------------
class SingletonSlateUndelivered(IncidentExpectedFailure):
    """E77: a one-game slate whose start moved up after the morning fetch ends with an enterable pick never delivered."""


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


def _run_singleton_day(picks_dir: Path, preview: DailyPick):
    """Run 2026-07-16 through run_day: the 10:00 fetch saw a 19:10 start; MLB moved the game to
    18:10. Only boundaries are mocked (schedule, game status, boxscore confirmations, DM transport,
    contest state, clock, result polling); run_day, run_single_check and the lock classifier are
    real. Returns (final pick, every DM sent with its time)."""
    from datetime import timedelta
    from unittest.mock import patch

    from bts.scheduler import run_day
    from tests.test_scheduler import _game

    save_pick(preview, picks_dir)
    game = preview.pick.game_pk
    true_first_pitch = datetime(2026, 7, 16, 18, 10, tzinfo=ET)
    clock = _Clock(datetime(2026, 7, 16, 10, 0, tzinfo=ET))

    def statuses(_date):
        started = clock() >= true_first_pitch
        return {game: {"abstract": "L" if started else "P",
                       "detailed": "In Progress" if started else "Scheduled",
                       "code": "I" if started else "S"}}

    def not_expected(*_a, **_k):
        raise AssertionError("the prediction cascade is not part of this characterization")

    dm_sends: list[tuple[datetime, str]] = []      # every DM (pick deliveries AND alerts)

    def send_dm(*args, **kwargs):
        text = " ".join(str(a) for a in args[1:]) + " " + " ".join(str(v) for v in kwargs.values())
        dm_sends.append((clock(), text))
        return "dm-1"

    with patch("bts.scheduler.fetch_schedule",
               side_effect=[[_game(game, "19:10", "NYM", "PHI", date=preview.date)], []]), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=lambda s: clock.advance(timedelta(seconds=s))), \
         patch("bts.picks.get_game_statuses_detailed", side_effect=statuses), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.orchestrator.run_and_pick", side_effect=not_expected), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.dm.send_dm", side_effect=send_dm), \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        run_day(date=preview.date, config=_singleton_config(picks_dir))
    return load_pick(preview.date, picks_dir), dm_sends


def _turner_preview(**delivery) -> DailyPick:
    turner = Pick(batter_name="Trea Turner", batter_id=607208, team="PHI", lineup_position=1,
                  pitcher_name="P", pitcher_id=1, p_game_hit=0.675, flags=[], projected_lineup=True,
                  game_pk=824716, game_time="2026-07-16T23:10:00Z")   # the stale 19:10 ET start
    return DailyPick(date="2026-07-16", run_time="2026-07-16T07:00:00+00:00", pick=turner,
                     double_down=None, runner_up=None, **delivery)


E77_TRUE_CUTOFF = datetime(2026, 7, 16, 18, 5, tzinfo=ET)


def _e77_oracle(daily: DailyPick | None, dm_sends) -> None:
    assert daily is not None, "the pick file vanished"
    if not daily.notification_sent and daily.delivered_at is None:
        # Containment evidence for the record: any DM here is the missed-pick ALERT, not a delivery.
        raise SingletonSlateUndelivered(
            f"enterable pick {daily.pick.batter_name} never delivered (true cutoff "
            f"{E77_TRUE_CUTOFF:%H:%M}); DMs sent: {[(t.strftime('%H:%M'), txt[:40]) for t, txt in dm_sends]}")
    assert daily.delivered_at is not None, f"notification_sent without delivered_at: {daily!r}"
    assert datetime.fromisoformat(daily.delivered_at).astimezone(ET) < E77_TRUE_CUTOFF, daily.delivered_at


@pytest.mark.xfail(strict=True, raises=SingletonSlateUndelivered,
                   reason="E77 unfixed: a moved-up singleton slate gets its only check at first pitch")
def test_e77_singleton_slate_moved_up_is_delivered_before_cutoff(tmp_path):
    daily, dm_sends = _run_singleton_day(tmp_path / "picks", _turner_preview())
    _e77_oracle(daily, dm_sends)


def test_e77_oracle_control_delivered_before_cutoff(tmp_path):
    """Oracle control: the same day with the pick already delivered at 17:30 satisfies the contract."""
    daily, dm_sends = _run_singleton_day(
        tmp_path / "picks",
        _turner_preview(notification_sent=True, notification_channel="dm", notification_id="dm-0",
                        delivery_attempted=True, delivered_at="2026-07-16T21:30:00+00:00"))
    _e77_oracle(daily, dm_sends)
