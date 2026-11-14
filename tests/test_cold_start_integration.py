"""Real cold-start model/selection/delivery paths with synthetic HTTP, clocks and DMs."""
import copy
import io
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pytest

from bts.data.build import build_season
from bts.daily_decision import load_decision
from tests.conftest import sample_game_feed as _sample_game_feed

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
DATE = "2027-03-24"


def _isolate(monkeypatch):
    # Limit real LightGBM fits to one CPU through its joblib CPU-count configuration.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    monkeypatch.setenv("BTS_USE_CALIBRATION", "0")
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    monkeypatch.setattr("bts.picks._git_head_sha", lambda *a, **k: "0" * 40)
    monkeypatch.setattr("bts.strategy._mdp_cache", None)


def _prepare_history(root, sample):
    for year in (2025, 2026):
        folder = root / "data" / "raw" / str(year)
        folder.mkdir(parents=True)
        for day in range(1, 17):
            feed = copy.deepcopy(sample)
            feed["gameData"]["game"].update(pk=year * 100 + day, season=str(year))
            feed["gameData"]["datetime"]["officialDate"] = f"{year}-06-{day:02}"
            (folder / f"{year * 100 + day}.json").write_text(json.dumps(feed))
        frame = build_season(root / "data" / "raw", root / "data" / "processed" / f"pa_{year}.parquet", year)
        assert len(frame) == 32
    models = root / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        shutil.copyfile(ROOT / "data" / "models" / name, models / name)


def _fake_http(monkeypatch, network):
    import bts.model.predict as predict
    import bts.data.pull as pull
    import bts.scheduler as sch
    import bts.picks as picks
    for module, attr in ((predict, "urlopen"), (pull, "urlopen"), (pull, "retry_urlopen"),
                         (picks, "retry_urlopen"), (sch, "retry_urlopen")):
        monkeypatch.setattr(module, attr, network)


def _game(date=DATE, pk=9001, final=False):
    return {"gamePk": pk, "gameType": "R", "officialDate": date, "gameDate": date + "T23:10:00Z",
            "status": {"detailedState": "Final" if final else "Scheduled", "abstractGameCode": "F" if final else "P"},
            "teams": {"away": {"team": {"id": 134, "name": "Pirates"},
                               "probablePitcher": {"id": 400001, "fullName": "Away Pitcher"}},
                      "home": {"team": {"id": 121, "name": "Mets"},
                               "probablePitcher": {"id": 300001, "fullName": "Home Pitcher"}}}}


def _live_feed(sample, posted):
    feed = copy.deepcopy(sample)
    feed["gameData"]["game"].update(pk=9001, season="2027")
    feed["gameData"]["datetime"]["officialDate"] = DATE
    feed["liveData"]["plays"]["allPlays"] = []
    for side, abbr in (("away", "PIT"), ("home", "NYM")):
        feed["gameData"]["teams"][side]["abbreviation"] = abbr
        if not posted:
            for player in feed["liveData"]["boxscore"]["teams"][side]["players"].values():
                player.pop("battingOrder", None)
    return feed


def test_cold_pipeline_training_selection_delivery(tmp_path, monkeypatch, sample_game_feed):
    import bts.model.predict as predict
    import bts.scheduler as sch
    from bts.orchestrator import run_and_pick
    _isolate(monkeypatch)
    _prepare_history(tmp_path, sample_game_feed)
    monkeypatch.chdir(tmp_path)
    posted = {"value": False}
    spring = tmp_path / "data" / "raw" / "2027"
    spring.mkdir()
    nonregular = copy.deepcopy(sample_game_feed)
    nonregular["gameData"]["game"].update(pk=8888, season="2027", type="S")
    nonregular["gameData"]["datetime"]["officialDate"] = "2027-03-20"
    (spring / "8888.json").write_text(json.dumps(nonregular))

    def network(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            query = parse_qs(urlparse(url).query)
            games = [_game()] if query.get("date") == [DATE] else []
            obj = {"dates": [{"games": games}]}
        elif "/9001/feed/live" in url:
            obj = _live_feed(sample_game_feed, posted["value"])
        else:
            raise AssertionError(f"unexpected HTTP request: {url}")
        return io.BytesIO(json.dumps(obj).encode())

    _fake_http(monkeypatch, network)
    sent = []
    monkeypatch.setattr("bts.dm.send_dm", lambda *args: sent.append(args) or "fake-dm-id")
    monkeypatch.setattr(sch, "_now_et", lambda: datetime(2027, 3, 24, 18, tzinfo=ET))
    models = tmp_path / "data" / "models"
    cache = models / f"blend_{DATE}.pkl"
    # Real refresh, features, all thirteen model fits, lookups and slot parsing.
    assert predict.run_pipeline(DATE, save_blend_path=cache).empty
    blend = predict.load_blend(cache)
    assert len(blend) == 13 and "_model" in blend
    posted["value"] = True
    picks_dir = tmp_path / "data" / "picks"
    config = {"orchestrator": {"picks_dir": str(picks_dir)},
              "tiers": [{"name": "local", "type": "local"}],
              "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False},
              "bluesky": {"dm_recipient": "fake-recipient"}}
    predictions, selection, tier = run_and_pick(config, DATE)
    assert tier == "local" and len(predictions) == 2 and predictions.p_game_hit.notna().all()
    assert selection.streak == 0 and selection.source == "mdp" and selection.action == "single"
    assert selection.pick_result is not None
    state = sch.SchedulerState(DATE, "", [], [], [], False, None, None, None)
    assert sch._deliver_and_lock_pick(selection.pick_result.daily, config, picks_dir, state,
                                      DATE, "cold-start-test", selection=selection)
    decision = load_decision(DATE, picks_dir)
    assert decision["scoreable"] and decision["delivery_status"] == "delivered" and state.pick_locked
    assert len(sent) == 1 and sent[0][0] == "fake-recipient"
    proc = tmp_path / "data" / "processed"
    assert not (proc / "pa_2027.parquet").exists()
    assert not list(proc.glob(".refreshed_*"))


def test_next_day_prior_lineup_projection(tmp_path, monkeypatch, sample_game_feed):
    import bts.model.predict as predict
    historical = copy.deepcopy(sample_game_feed)
    historical["gameData"]["datetime"]["officialDate"] = DATE
    historical["gameData"]["game"].update(pk=9001, season="2027")
    for side, abbr in (("away", "PIT"), ("home", "NYM")):
        historical["gameData"]["teams"][side]["abbreviation"] = abbr
    current = _live_feed(sample_game_feed, posted=False)
    current["gameData"]["game"]["pk"] = 9002
    current["gameData"]["datetime"]["officialDate"] = "2027-03-25"
    # Second pair of teams has no prior 2027 regular-season game.
    newcomers = copy.deepcopy(current)
    newcomers["gameData"]["game"]["pk"] = 9003
    newcomers["gameData"]["teams"]["away"].update(id=200, abbreviation="NEW1")
    newcomers["gameData"]["teams"]["home"].update(id=201, abbreviation="NEW2")

    def network(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            query = parse_qs(urlparse(url).query)
            if "teamId" in query:
                games = [_game(final=True)] if int(query["teamId"][0]) in (134, 121) else []
            else:
                returning = _game("2027-03-25", pk=9002)
                other = _game("2027-03-25", pk=9003)
                other["teams"]["away"]["team"]["id"] = 200
                other["teams"]["home"]["team"]["id"] = 201
                games = [returning, other]
            obj = {"dates": [{"games": games}]}
        elif "/9001/feed/live" in url:
            obj = historical
        elif "/9002/feed/live" in url:
            obj = current
        elif "/9003/feed/live" in url:
            obj = newcomers
        else:
            raise AssertionError(f"unexpected HTTP request: {url}")
        return io.BytesIO(json.dumps(obj).encode())

    monkeypatch.setattr(predict, "urlopen", network)
    slots = predict._fetch_game_slots("2027-03-25")
    assert len(slots) == 2 and all(slot["projected"] for slot in slots)
    assert {slot["team"] for slot in slots} == {"PIT", "NYM"}
    assert {slot["game_pk"] for slot in slots} == {9002}


@pytest.fixture(scope="module")
def late_lineup_day(tmp_path_factory):
    """One real day-loop run feeds both the planned-check control and last-attempt check."""
    import bts.scheduler as sch
    root = tmp_path_factory.mktemp("late-opening-lineup")
    sample = _sample_game_feed.__wrapped__()
    with pytest.MonkeyPatch.context() as monkeypatch:
        _isolate(monkeypatch)
        _prepare_history(root, sample)
        monkeypatch.chdir(root)
        clock = {"now": datetime(2027, 3, 24, 10, tzinfo=ET)}
        calls = []

        def network(url, **kw):
            url = getattr(url, "full_url", url)
            calls.append((clock["now"].isoformat(), url))
            if "/schedule" in url:
                query = parse_qs(urlparse(url).query)
                games = [_game()] if query.get("date") == [DATE] else []
                obj = {"dates": [{"games": games}]}
            elif "/9001/feed/live" in url:
                obj = _live_feed(sample, posted=clock["now"] >= datetime(2027, 3, 24, 18, 25, tzinfo=ET))
                # Delivery now reaches result polling; complete this synthetic
                # game's feed so the real day loop can finish too.
                final = clock["now"] >= datetime(2027, 3, 24, 19, 20, tzinfo=ET)
                obj["gameData"]["status"] = {
                    "abstractGameCode": "F" if final else "P",
                    "detailedState": "Final" if final else "Scheduled"}
            else:
                raise AssertionError(f"unexpected HTTP request: {url}")
            return io.BytesIO(json.dumps(obj).encode())

        _fake_http(monkeypatch, network)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

        monkeypatch.setattr(sch, "datetime", Clock)
        monkeypatch.setattr(sch, "_watchdog_ping_sleep", lambda seconds, *a, **k:
                            clock.update(now=clock["now"] + timedelta(seconds=seconds)))
        sent = []
        monkeypatch.setattr("bts.dm.send_dm", lambda *args: sent.append(args) or "fake-dm-id")
        picks_dir = root / "data" / "picks"
        config = {"orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / "data" / ".heartbeat")},
                  "tiers": [{"name": "local", "type": "local"}],
                  "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60,
                                "live_forward_capture_on_lock": False},
                  "bluesky": {"dm_recipient": "fake"}, "health_checks": {"enabled": False}}
        sch.run_day(DATE, config)
        state = sch.load_state(DATE, picks_dir)
        # A setup failure in the real pipeline must be visible in the non-xfail control too.
        assert (root / "data" / "models" / f"blend_{DATE}.pkl").exists()
    return SimpleNamespace(state=state, calls=calls, sent=sent, picks_dir=picks_dir)


def test_opening_day_late_lineup_checks_today(late_lineup_day):
    assert [run["time"][11:16] for run in late_lineup_day.state.runs_completed][:3] == ["16:10", "17:10", "18:10"]
    feed_times = [time[11:16] for time, url in late_lineup_day.calls if "/feed/live" in url]
    assert {"16:10", "17:10", "18:10", "18:35"} <= set(feed_times)
    assert [run["time"][11:16] for run in late_lineup_day.state.runs_completed][3:] == ["18:35"]


def test_opening_day_late_lineup_gets_a_last_attempt(late_lineup_day):
    decision = load_decision(DATE, late_lineup_day.picks_dir)
    assert decision is not None, "S6: no decision was saved after the 18:10 check; no later pick attempt occurred"
    assert decision["scoreable"] and decision["delivery_status"] == "delivered"
    assert late_lineup_day.state.pick_locked and len(late_lineup_day.sent) == 1
