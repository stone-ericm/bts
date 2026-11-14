"""Real day loop and cascade; fake only HTTP, time, and DM transport."""
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

import bts.contest_state as contest
import bts.health.alert as alerts
import bts.picks as picks
import bts.scheduler as sch
import bts.strategy as strategy
from bts.daily_decision import load_decision
from tests.conftest import sample_game_feed
from tests.test_cold_start_integration import DATE, ET, _fake_http, _game, _isolate, _live_feed, _prepare_history

HERE = Path(__file__).resolve().parent
summary = {}
sample = sample_game_feed.__wrapped__()

with tempfile.TemporaryDirectory(dir=HERE) as directory:
    root = Path(directory)
    history = root / "history"
    _prepare_history(history, sample)

    def run(name, *, starts=("19:10",), posted=None, mode="dm", intent="enter", skip=False,
            restart=None, doubleheader=False, old_streak=None, never_send_pick=False,
            only_game=None, begin="10:00", refuse=False):
        work = root / name
        shutil.copytree(history / "data", work / "data")
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LOKY_MAX_CPU_COUNT", "1")
            mp.setenv("BTS_USE_CALIBRATION", "0")
            mp.delenv("BTS_REFRESH_ALWAYS", raising=False)
            mp.setattr(strategy, "_mdp_cache", None)
            mp.chdir(work)
            def at(value):
                return datetime.fromisoformat(f"{DATE}T{value}:00").replace(tzinfo=ET)
            clock = {"now": at(begin)}
            calls, sent = [], []
            cancellation = {"checks": 0, "active": False}
            games = []
            for i, start in enumerate(starts):
                game = _game(pk=9001 + i)
                game["gameDate"] = at(start).isoformat()
                if not doubleheader:
                    for side in ("away", "home"):
                        game["teams"][side]["team"]["id"] += i * 10
                        game["teams"][side]["team"]["name"] += str(i)
                games.append(game)

            class Clock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

            for module in (sch, picks, alerts, contest, strategy):
                mp.setattr(module, "datetime", Clock)
            def sleep(seconds, *a, **k):
                clock["now"] += timedelta(seconds=seconds)
            mp.setattr(sch, "_watchdog_ping_sleep", sleep)
            mp.setattr(sch.time, "sleep", sleep)

            def network(url, **kwargs):
                url = getattr(url, "full_url", url)
                calls.append((clock["now"].isoformat(), url))
                if "/schedule" in url:
                    today = parse_qs(urlparse(url).query).get("date") == [DATE]
                    answer = copy.deepcopy(games) if today else []
                    # The chosen game is cancelled between selection and delivery.
                    if refuse and (picks_dir / f"{DATE}.json").exists():
                        cancellation["checks"] += 1
                        cancellation["active"] = cancellation["checks"] >= 2
                    if cancellation["active"]:
                        for game in answer:
                            game["status"].update(detailedState="Cancelled")
                    body = {"dates": [{"games": answer}]}
                elif "/feed/live" in url:
                    pk = int(url.split("/game/")[1].split("/")[0])
                    i = pk - 9001
                    available = posted is not None and clock["now"] >= at(posted)
                    if only_game is not None and i != only_game:
                        available = False
                    body = _live_feed(sample, posted=available)
                    body["gameData"]["game"]["pk"] = pk
                    body["gameData"]["datetime"]["dateTime"] = games[i]["gameDate"]
                    for side in ("away", "home"):
                        body["gameData"]["teams"][side]["id"] = games[i]["teams"][side]["team"]["id"]
                    final = clock["now"] >= at(starts[i]) + timedelta(minutes=10)
                    body["gameData"]["status"] = {
                        "abstractGameCode": "F" if final else "P",
                        "detailedState": "Final" if final else "Scheduled"}
                else:
                    raise AssertionError(f"Unfaked HTTP: {url}")
                return io.BytesIO(json.dumps(body).encode())
            _fake_http(mp, network)
            mp.setattr(sch, "retry_urlopen", network)
            def send(recipient, body):
                sent.append((clock["now"].isoformat(), body))
                if never_send_pick and "health CRITICAL" not in body:
                    raise OSError("synthetic pick-transport failure")
                return "dm-id"
            mp.setattr("bts.dm.send_dm", send)
            mp.setattr(alerts, "send_dm", send)
            picks_dir = work / "data" / "picks"
            picks_dir.mkdir(exist_ok=True)
            if skip:
                (picks_dir / "account_state").mkdir()
                (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "active_streak": 57, "source_date": DATE, "recorded_at": f"{DATE}T12:00:00Z"}))
            if old_streak is not None:
                (picks_dir / "streak.json").write_text(json.dumps({"streak": old_streak, "saver_available": True}))
            config = {"orchestrator": {"picks_dir": str(picks_dir),
                                        "heartbeat_path": str(work / "data" / ".heartbeat")},
                      "tiers": [{"name": "local", "type": "local"}],
                      "scheduler": {"pick_delivery": mode, "lineup_check_offset_min": 60,
                                    "entry_intent": intent, "live_forward_capture_on_lock": False},
                      "bluesky": {"dm_recipient": "fake"}, "health_checks": {"enabled": False}}
            cascades = []
            def profile(frame, event, arg):
                if event == "call" and frame.f_code.co_name == "run_cascade" and frame.f_globals.get("__name__") == "bts.orchestrator":
                    cascades.append(clock["now"].isoformat())
            prior_profile = sys.getprofile()
            sys.setprofile(profile)
            try:
                sch.run_day(DATE, config)
            finally:
                sys.setprofile(prior_profile)
            first_state = sch.load_state(DATE, picks_dir)
            first_times = [r["time"][11:16] for r in first_state.runs_completed]
            if restart:
                clock["now"] = at(restart)
                sch.run_day(DATE, config)
            state = sch.load_state(DATE, picks_dir)
            daily = picks.load_pick(DATE, picks_dir)
            decision = load_decision(DATE, picks_dir)
            return {"times": first_times, "restart_times": [r["time"][11:16] for r in state.runs_completed],
                    "sent": sent, "decision": decision, "result_status": first_state.result_status,
                    "cascades": cascades,
                    "model_streak_after": picks.load_streak(picks_dir),
                    "daily_result": daily.result if daily else None,
                    "refusals": state.delivery_refusals,
                    "feed_times": sorted({t[11:16] for t, url in calls if "/feed/live" in url})}

    summary["restart_empty"] = run("restart_empty", restart="18:40")
    assert len(summary["restart_empty"]["sent"]) == 1
    for mode, intent in (("private", "research"), ("dm", "enter")):
        name = f"no_lineups_{mode}"
        summary[name] = run(name, mode=mode, intent=intent)
        assert len(summary[name]["sent"]) == 1
        name = f"late_delivery_{mode}"
        summary[name] = run(name, mode=mode, intent=intent, posted="18:25", restart="20:00")
        assert len(summary[name]["sent"]) == (0 if mode == "private" else 1)
        assert summary[name]["decision"]["scoreable"] and summary[name]["result_status"] == "final"
    summary["actual_extra_skip"] = run("actual_extra_skip", starts=("19:10", "19:20", "19:30"),
                                        posted="18:34", skip=True)
    assert summary["actual_extra_skip"]["decision"]["action"] == "skip"
    assert summary["actual_extra_skip"]["times"] == ["18:10", "18:30", "18:35"]
    summary["equal_starts_no_lineups"] = run("equal_starts", starts=("19:10", "19:10", "19:20"))
    assert summary["equal_starts_no_lineups"]["times"] == ["18:10", "18:35", "18:45"]
    summary["doubleheader"] = run("doubleheader", starts=("19:10", "19:20"), doubleheader=True)
    assert summary["doubleheader"]["times"][-2:] == ["18:35", "18:45"]
    summary["started_game_filtered"] = run("started", starts=("17:00", "19:20", "19:30"),
                                            posted="18:40", only_game=2)
    assert summary["started_game_filtered"]["decision"]["primary"]["game_pk"] == 9003
    summary["old_model_first_void"] = run("old_model", posted="18:25", old_streak=12)
    summary["transport_failure"] = run("transport_failure", posted="18:25", never_send_pick=True)
    summary["max_extra_cascades"] = run("max_extra", starts=("19:10", "19:20", "19:30"),
                                         posted="18:54", only_game=2, never_send_pick=True)
    assert len(summary["max_extra_cascades"]["cascades"]) == 7  # two planned, N+2 extra
    summary["refused_extra"] = run("refused_extra", starts=("19:10", "19:20", "19:30"),
                                    posted="18:34", refuse=True)
    assert summary["refused_extra"]["refusals"] and summary["refused_extra"]["times"][-1] == "18:35"

    # Repeat the same late-lineup/empty days with the exact base scheduler.
    base_scheduler = types.ModuleType("bts._review_base_scheduler")
    sys.modules[base_scheduler.__name__] = base_scheduler
    exec(compile(subprocess.check_output(["git", "show", "12dbaaa:src/bts/scheduler.py"], text=True),
                 "12dbaaa:src/bts/scheduler.py", "exec"), base_scheduler.__dict__)
    sch = base_scheduler
    summary["base_empty"] = run("base_empty")
    summary["base_late_lineup"] = run("base_late_lineup", posted="18:25")
    assert summary["base_empty"]["times"] == ["16:10", "17:10", "18:10"]
    assert summary["base_empty"]["sent"] == [] and summary["base_late_lineup"]["decision"] is None

(HERE / "scheduler-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: {k: value[k] for k in ("times", "restart_times", "result_status", "model_streak_after", "daily_result")}
                  for key, value in summary.items()}, indent=2))
