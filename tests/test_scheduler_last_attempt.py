"""S6: real day loop, model cascade, selection and delivery; synthetic HTTP and time."""
import copy
import io
import json
import shutil
from datetime import datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from bts.daily_decision import load_decision
from bts.picks import DailyPick, Pick, load_pick, save_pick
from tests.test_cold_start_integration import (
    DATE, ET, _fake_http, _game, _isolate, _live_feed, _prepare_history,
)


def _at(hhmm):
    return datetime.fromisoformat(f"{DATE}T{hhmm}:00").replace(tzinfo=ET)


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    from tests.conftest import sample_game_feed
    root = tmp_path_factory.mktemp("last-attempt-history")
    _prepare_history(root, sample_game_feed.__wrapped__())
    return root


@pytest.fixture
def drive(tmp_path, monkeypatch, sample_game_feed, history):
    import bts.contest_state as contest
    import bts.health.alert as alerts
    import bts.model.predict as predict
    import bts.picks as picks
    import bts.scheduler as sch

    def run(*, starts=("19:10",), posted_at=None, outage_until=None,
            skip=None, preview=False, now="10:00", scheduler=None, regular=True, streak=0,
            early_sleep_targets=(), refuse_planned_pick=False, projected_index=None,
            cascade_min=0, void_after=None, fail_status_read=False):
        _isolate(monkeypatch)
        shutil.copytree(history / "data", tmp_path / "data")
        monkeypatch.chdir(tmp_path)
        clock = {"now": _at(now), "mono": 0.0}
        calls, sent, early_sleeps = [], [], []
        cascades, status_reads = [], []
        pick_status_reads = 0
        games = []
        for i, start in enumerate(starts):
            game = _game(pk=9001 + i)
            game["gameDate"] = _at(start).astimezone(ET).isoformat()
            game["teams"]["away"]["team"]["id"] += i * 10
            game["teams"]["home"]["team"]["id"] += i * 10
            game["teams"]["away"]["team"]["name"] += str(i)
            game["teams"]["home"]["team"]["name"] += str(i)
            if not regular:
                game["gameType"] = "S"
            games.append(game)

        prior_date = "2027-03-23"
        prior = None
        if projected_index is not None:
            prior = copy.deepcopy(sample_game_feed)
            prior["gameData"]["game"].update(pk=9000, season="2027", type="R")
            prior["gameData"]["datetime"]["officialDate"] = prior_date
            for side, abbr in (("away", "BOS"), ("home", "NYY")):
                prior["gameData"]["teams"][side].update(
                    id=games[projected_index]["teams"][side]["team"]["id"], abbreviation=abbr)
            folder = tmp_path / "data" / "raw" / "2027"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "9000.json").write_text(json.dumps(prior))

        def prior_schedule():
            game = _game(prior_date, pk=9000, final=True)
            game["teams"] = copy.deepcopy(games[projected_index]["teams"])
            return game

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

        def sleep(seconds, *a, **k):
            clock["mono"] += seconds
            target = clock["now"] + timedelta(seconds=seconds)
            if target.strftime("%H:%M") in early_sleep_targets:
                early_sleeps.append(target)
                clock["now"] = target - timedelta(seconds=1)
            else:
                clock["now"] = target

        def network(url, **kw):
            nonlocal pick_status_reads
            url = getattr(url, "full_url", url)
            calls.append((clock["now"], url))
            if "/schedule" in url:
                query = parse_qs(urlparse(url).query)
                obj = {"dates": [{"games": games if query.get("date") == [DATE] else []}]}
                if prior is not None:
                    prior_ids = {prior["gameData"]["teams"][s]["id"] for s in ("away", "home")}
                    if (query.get("date") == [prior_date] or
                            ("teamId" in query and int(query["teamId"][0]) in prior_ids)):
                        obj = {"dates": [{"games": [prior_schedule()]}]}
                if query.get("date") == [DATE]:
                    if "hydrate" not in query:
                        status_reads.append(clock["now"])
                        if fail_status_read:
                            raise OSError("synthetic status-read failure")
                    if void_after:
                        obj = copy.deepcopy(obj)
                        for game in obj["dates"][0]["games"]:
                            change = void_after.get(game["gamePk"])
                            if change and clock["now"] >= _at(change[0]):
                                game["status"] = {"abstractGameCode": "F", "detailedState": change[1]}
                if (refuse_planned_pick and query.get("date") == [DATE]
                        and "hydrate" not in query and (picks_dir / f"{DATE}.json").exists()):
                    pick_status_reads += 1
                    # The first lookup after saving is the confirmation gate;
                    # the next is the delivery guard. Postpone between the two.
                    if pick_status_reads >= 2:
                        obj = copy.deepcopy(obj)
                        obj["dates"][0]["games"][0]["status"]["detailedState"] = "Postponed"
            elif "/9000/feed/live" in url and prior is not None:
                obj = prior
            elif "/feed/live" in url:
                if outage_until and clock["now"] < _at(outage_until):
                    raise OSError("synthetic prediction-feed outage")
                pk = int(url.split("/game/")[1].split("/")[0])
                post_time = posted_at.get(pk) if isinstance(posted_at, dict) else posted_at
                posted = post_time is not None and clock["now"] >= _at(post_time)
                # On a multi-game day the first game's lineup never arrives.
                if (not isinstance(posted_at, dict) and len(games) > 1
                        and starts[pk - 9001] == starts[0]):
                    posted = False
                projected = projected_index is not None and pk == 9001 + projected_index
                obj = _live_feed(prior if projected else sample_game_feed, posted)
                obj["gameData"]["game"]["pk"] = pk
                for side in ("away", "home"):
                    obj["gameData"]["teams"][side]["id"] = games[pk - 9001]["teams"][side]["team"]["id"]
                    if projected:
                        obj["gameData"]["teams"][side]["abbreviation"] = "BOS" if side == "away" else "NYY"
                obj["gameData"]["datetime"]["dateTime"] = games[pk - 9001]["gameDate"]
                obj["gameData"]["status"] = {"abstractGameCode": "P", "detailedState": "Scheduled"}
                if clock["now"] >= _at(starts[pk - 9001]) + timedelta(minutes=10):
                    obj["gameData"]["status"] = {"abstractGameCode": "F", "detailedState": "Final"}
            else:
                raise AssertionError(f"unexpected HTTP request: {url}")
            return io.BytesIO(json.dumps(obj).encode())

        _fake_http(monkeypatch, network)
        for module in (sch, picks, alerts, contest):
            monkeypatch.setattr(module, "datetime", Clock)
        monkeypatch.setattr(sch, "_watchdog_ping_sleep", sleep)
        monkeypatch.setattr(sch.time, "sleep", sleep)
        if cascade_min:
            real_pipeline = predict.run_pipeline

            def timed_pipeline(*a, **kw):
                cascades.append(clock["now"])
                result = real_pipeline(*a, **kw)
                if clock["now"] >= _at("18:30"):
                    sleep(cascade_min * 60)
                return result

            monkeypatch.setattr(predict, "run_pipeline", timed_pipeline)
            monkeypatch.setattr(sch.time, "monotonic", lambda: clock["mono"])
        monkeypatch.setattr("bts.dm.send_dm", lambda *a: sent.append((clock["now"], a)) or "dm-id")
        monkeypatch.setattr(alerts, "send_dm", lambda *a: sent.append((clock["now"], a)) or "dm-id")
        picks_dir = tmp_path / "data" / "picks"
        if streak:
            account_state = picks_dir / "account_state"
            account_state.mkdir(parents=True, exist_ok=True)
            (account_state / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v1", "source": "mlb_bts_profile",
                "active_streak": streak, "best_streak": streak, "saver_available": False,
                "source_date": DATE, "recorded_at": f"{DATE}T12:00:00Z",
            }))
        if skip:
            state = sch.SchedulerState(DATE, "", [], [], [], False, None, None, None)
            if skip == "candidate":
                state.final_skip_candidate = {"primary": {"batter_id": 100001, "batter_name": "Test Batter",
                    "team": "PIT", "game_pk": 9001, "p_game_hit": .70}, "streak": 9}
            else:
                state.skip_summary = {"best_batter": "Test Batter", "best_team": "PIT", "best_p": .70, "streak": 9}
            sch.save_state(state, picks_dir)
        if preview:
            pick = Pick("Test Batter", 100001, "PIT", 3, "Home Pitcher", 300001, .75, [],
                        False, 9001, games[0]["gameDate"])
            save_pick(DailyPick(DATE, _at("09:00").isoformat(), pick, None, None), picks_dir)
        config = {"orchestrator": {"picks_dir": str(picks_dir),
                    "heartbeat_path": str(tmp_path / "data" / ".heartbeat")},
                  "tiers": [{"name": "local", "type": "local"}],
                  "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60,
                    "live_forward_capture_on_lock": False, **(scheduler or {})},
                  "bluesky": {"dm_recipient": "fake"}, "health_checks": {"enabled": False}}
        sch.run_day(DATE, config)
        state = sch.load_state(DATE, picks_dir)
        return SimpleNamespace(state=state, sent=sent, calls=calls, picks_dir=picks_dir,
                               decision=load_decision(DATE, picks_dir), daily=load_pick(DATE, picks_dir),
                               early_sleeps=early_sleeps, cascades=cascades, status_reads=status_reads)
    return run


def _times(day):
    return [r["time"][11:16] for r in day.state.runs_completed]


def test_lineups_never_arrive_extra_attempt_then_one_immediate_alert(drive):
    day = drive()
    assert _times(day) == ["16:10", "17:10", "18:10", "18:35"]
    assert day.daily is None and day.decision is None
    assert len(day.sent) == 1
    assert day.sent[0][0] == _at("18:35")
    assert "no pick is on file for today" in day.sent[0][1][1].lower()


@pytest.mark.parametrize("skip", ["candidate", "summary"])
def test_recorded_skip_has_no_extra_attempt_or_alert(drive, skip):
    day = drive(skip=skip)
    assert _times(day) == ["16:10", "17:10", "18:10"]
    assert day.sent == [] and day.daily is None
    if skip == "candidate":
        assert day.decision["action"] == "skip"


def test_outage_recovers_at_later_distinct_start_deadline(drive):
    day = drive(starts=("19:10", "19:10", "19:20", "19:30"),
                outage_until="18:40", posted_at="18:40")
    assert _times(day) == ["18:10", "18:30", "18:35", "18:45"]
    assert day.state.pick_locked and day.decision["scoreable"]
    assert day.decision["delivery_status"] == "delivered"
    assert day.daily.pick.game_pk in {9003, 9004}
    assert day.sent[0][0] == _at("18:45") and len(day.sent) == 1


def test_existing_undelivered_pick_keeps_its_original_fallback_time(drive):
    day = drive(preview=True)
    assert _times(day) == ["16:10", "17:10", "18:10"]
    assert day.state.pick_locked and day.decision["delivery_status"] == "delivered"
    assert len(day.sent) == 1 and day.sent[0][0] == _at("18:35")


def test_first_extra_attempt_recording_a_skip_stops_later_attempts(drive):
    day = drive(starts=("19:10", "19:20", "19:30"), posted_at="18:34", streak=57)
    assert _times(day) == ["18:10", "18:30", "18:35"]
    assert day.daily is None and day.decision["action"] == "skip"
    assert day.decision["streak"] == 57 and day.decision["state_source"] == "contest"
    assert not (day.picks_dir / "streak.json").exists()
    assert not day.decision["scoreable"]
    assert all("no pick is on file for today" not in args[1].lower() for _, args in day.sent)


def test_no_deadline_ahead_has_no_extra_attempt_but_alerts_now(drive):
    day = drive(now="18:50")
    assert _times(day) == ["18:50"]
    assert len(day.sent) == 1 and day.sent[0][0] == _at("18:50")


def test_nonregular_games_have_no_attempt_or_alert(drive):
    day = drive(regular=False)
    assert day.state is None and day.sent == []


@pytest.mark.parametrize("start,now,settings,expected", [
    ("10:30", "07:00", {}, "10:03"),  # morning 25 is floored to 5 + 12 + 10
    ("19:10", "10:00", {"fallback_deadline_min": 10,
                          "fallback_deadline_min_morning": 10, "cascade_budget_min": 20}, "18:35"),
])
def test_extra_attempt_uses_morning_deadline_and_budget_floor(drive, start, now, settings, expected):
    day = drive(starts=(start,), now=now, scheduler=settings)
    assert _times(day)[-1] == expected
    assert len(day.sent) == 1 and day.sent[0][0] == _at(expected)


def test_early_extra_sleep_after_refused_planned_pick_keeps_next_attempt(drive):
    day = drive(starts=("19:10", "19:20", "19:30"), posted_at={9001: "18:29"},
                early_sleep_targets=("18:35",), refuse_planned_pick=True)
    assert day.state.delivery_refusals[0]["reason"] == "game_not_playable"
    assert day.state.delivery_refusals[0]["at"] == _at("18:30").isoformat()
    archives = list((day.picks_dir / DATE).glob("refused_delivery_*.json"))
    assert any(json.loads(path.read_text())["pick"]["game_pk"] == 9001 for path in archives)
    # The refused pick's existing fallback sleep and the extra attempt's sleep
    # both return just before the same deadline.
    assert day.early_sleeps == [_at("18:35")] * 2
    assert _times(day)[:3] == ["18:10", "18:30", "18:45"]


def test_early_extra_sleep_without_any_previous_check_keeps_next_attempt(drive):
    day = drive(starts=("19:10", "19:20", "19:30"),
                early_sleep_targets=("18:10", "18:30", "18:35"))
    assert day.early_sleeps == [_at("18:10"), _at("18:30"), _at("18:35")]
    assert _times(day) == ["18:45", "18:55"]
    assert day.daily is None


def test_extra_attempt_reuses_its_unlocked_cascade_at_own_deadline(drive):
    day = drive(starts=("19:10", "19:10"), posted_at={9001: "18:25"},
                projected_index=1, outage_until="18:20", cascade_min=16,
                scheduler={"cascade_budget_min": 18})
    assert day.state.pick_locked and day.decision["delivery_status"] == "delivered"
    assert [t.strftime("%H:%M") for t in day.cascades] == ["16:10", "17:10", "18:10", "18:35"]
    assert day.sent[0][0] == _at("18:51")
    assert day.state.fallback_refreshes[-1]["reused_check_cascade"] is True


def test_extra_attempt_waits_and_refreshes_for_its_later_pick_deadline(drive):
    day = drive(starts=("19:10", "20:10"), posted_at={9002: "19:00"},
                projected_index=1, outage_until="18:20", cascade_min=16,
                scheduler={"lineup_check_offset_min": 180, "cascade_budget_min": 18})
    assert day.state.pick_locked and day.decision["delivery_status"] == "delivered"
    assert [t.strftime("%H:%M") for t in day.cascades] == ["16:10", "17:10", "18:35", "19:35"]
    assert day.daily.pick.game_pk == 9002 and day.sent[0][0] == _at("19:51")
    assert day.state.fallback_refreshes[-1]["reused_check_cascade"] is False


def test_planned_overrun_keeps_its_existing_refresh_behavior(drive):
    day = drive(starts=("19:10", "19:10"), posted_at={9001: "18:25"},
                projected_index=1, outage_until="18:20", cascade_min=16,
                scheduler={"lineup_check_offset_min": 30, "cascade_budget_min": 18})
    assert not day.state.pick_locked and day.decision is None
    assert [t.strftime("%H:%M") for t in day.cascades] == ["16:10", "17:10", "18:40", "18:56"]
    assert day.state.delivery_refusals[0]["at"] == _at("19:12").isoformat()
    assert day.state.fallback_refreshes[-1]["reused_check_cascade"] is False


@pytest.mark.parametrize("status", ["Postponed", "Cancelled"])
def test_all_games_void_after_last_planned_check_get_no_attempt_or_alert(drive, status):
    day = drive(void_after={9001: ("18:25", status)})
    assert _times(day) == ["16:10", "17:10", "18:10"]
    assert day.sent == [] and day.daily is None
    assert day.status_reads == [_at("18:35"), _at("18:35")]


@pytest.mark.parametrize("starts,void_pks,checks,reads", [
    (("19:10", "19:20", "19:30"), (9001, 9003), ["18:10", "18:30", "18:45"],
     ["18:35", "18:45", "18:55", "18:55"]),
    (("19:10", "19:10", "19:20"), (9001, 9003), ["18:10", "18:35"],
     ["18:35", "18:45", "18:45"]),
])
def test_only_starts_with_a_playable_game_get_extra_attempts(drive, starts, void_pks, checks, reads):
    day = drive(starts=starts, void_after={pk: ("14:00", "Cancelled") for pk in void_pks})
    assert _times(day) == checks
    assert len(day.sent) == 1 and day.sent[0][0] == _at(reads[-1])
    assert day.status_reads == [_at(t) for t in reads]


def test_failed_status_reads_keep_attempt_and_alert(drive):
    day = drive(fail_status_read=True)
    assert _times(day) == ["16:10", "17:10", "18:10", "18:35"]
    assert len(day.sent) == 1 and day.sent[0][0] == _at("18:35")
    assert day.status_reads == [_at("18:35"), _at("18:35")]


def test_games_become_void_during_last_attempt_suppress_the_alert(drive):
    day = drive(void_after={9001: ("18:36", "Postponed")}, cascade_min=1)
    assert _times(day) == ["16:10", "17:10", "18:10", "18:36"]
    assert day.sent == []
    assert day.status_reads == [_at("18:35"), _at("18:36")]


@pytest.mark.parametrize("preview", [False, True])
def test_alert_dedup_uses_slate_date_across_midnight_and_next_day(tmp_path, monkeypatch,
                                                               sample_game_feed, preview):
    import bts.contest_state as contest
    import bts.health.alert as alerts
    import bts.picks as picks
    import bts.scheduler as sch

    _isolate(monkeypatch)
    monkeypatch.chdir(tmp_path)
    next_date = "2027-03-25"
    clock = {"now": _at("10:00")}
    sent = []
    picks_dir = tmp_path / "data" / "picks"

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

    def network(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            query = parse_qs(urlparse(url).query)
            slate = query.get("date", [""])[0]
            games = [_game(slate)] if slate in (DATE, next_date) else []
            obj = {"dates": [{"games": games}]}
        elif "/9001/feed/live" in url:
            obj = _live_feed(sample_game_feed, posted=False)
        else:
            raise AssertionError(f"unexpected request: {url}")
        return io.BytesIO(json.dumps(obj).encode())

    def sleep(seconds, *a, **k):
        clock["now"] += timedelta(seconds=seconds)

    def failed_pick_dm(*a, **k):
        raise OSError("synthetic pick transport failure")

    _fake_http(monkeypatch, network)
    for module in (sch, picks, alerts, contest):
        monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(sch, "_watchdog_ping_sleep", sleep)
    monkeypatch.setattr(sch.time, "sleep", sleep)
    monkeypatch.setattr("bts.dm.send_dm", failed_pick_dm)
    monkeypatch.setattr(alerts, "send_dm", lambda *a: sent.append((clock["now"], a)) or "alert-id")
    config = {"orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(tmp_path / ".hb")},
              "tiers": [], "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60,
                                          "live_forward_capture_on_lock": False},
              "bluesky": {"dm_recipient": "fake"}, "health_checks": {"enabled": False}}

    def seed(slate):
        if preview:
            pick = Pick("Test Batter", 100001, "PIT", 3, "Pitcher", 300001, .75, [],
                        False, 9001, f"{slate}T23:10:00Z")
            save_pick(DailyPick(slate, f"{slate}T14:00:00Z", pick, None, None), picks_dir)

    status_path = picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
    seed(DATE)
    sch.run_day(DATE, config)
    assert json.loads(status_path.read_text())["sent_day"] == DATE
    clock["now"] = datetime.fromisoformat(f"{next_date}T00:05:00").replace(tzinfo=ET)
    sch.run_day(DATE, config)
    assert len(sent) == 1, "a post-midnight restart must not alert again for the same slate"
    assert json.loads(status_path.read_text())["sent_day"] == DATE
    clock["now"] = datetime.fromisoformat(f"{next_date}T10:00:00").replace(tzinfo=ET)
    seed(next_date)
    sch.run_day(next_date, config)
    expected_time = "19:00" if preview else "18:35"
    assert [t.isoformat() for t, _ in sent] == [
        f"{DATE}T{expected_time}:00-04:00", f"{next_date}T{expected_time}:00-04:00"]
    assert json.loads(status_path.read_text())["sent_day"] == next_date
