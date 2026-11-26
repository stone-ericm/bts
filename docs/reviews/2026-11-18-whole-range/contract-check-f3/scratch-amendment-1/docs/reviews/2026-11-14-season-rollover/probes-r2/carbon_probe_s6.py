"""Adversarial S6 probes driving the real run_day (HEAD vs 12dbaaa) with fake HTTP, clock, DMs."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import subprocess
import unittest.mock as mock
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from bts.daily_decision import load_decision
from bts.day_outcome import gather as gather_day_outcome
from bts.picks import load_pick
from tests.conftest import sample_game_feed as _sample_game_feed
from tests.test_cold_start_integration import _fake_http, _game, _isolate, _live_feed, _prepare_history

ET = ZoneInfo("America/New_York")


def _at(date_str: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{date_str}T{hhmm}:00").replace(tzinfo=ET)


def _load_base_scheduler():
    import sys
    src = subprocess.check_output(["git", "show", "12dbaaa:src/bts/scheduler.py"], text=True)
    tmp = Path(".codex-review/sr-review/scratch/_base_scheduler.py")
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("_base_scheduler", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_base_scheduler"] = mod
    spec.loader.exec_module(mod)
    return mod


def probe_1_double_cascade_loses_pick(history_root: Path, sample: dict, out_root: Path) -> None:
    print("=== 1. S6 Extra Attempt w/ should_post=False Runs Back-to-Back Double Cascade & Loses Pick ===")
    import bts.health.alert as alerts
    import bts.model.predict as predict
    import bts.picks as picks
    import bts.contest_state as cstate
    import bts.scheduler as sch

    for variant in ("HEAD_UNPATCHED", "WITH_LOCK_DECISION_FIX"):
        work = out_root / f"p1_{variant}"
        shutil.copytree(history_root / "data", work / "data")
        # Add 1 Final 2027 game on 2027-03-24 for teams 234/221 so game 9002 on 2027-03-25 has a projected lineup!
        prior_game = copy.deepcopy(sample)
        prior_game["gameData"]["game"].update(pk=9000, season="2027", type="R")
        prior_game["gameData"]["datetime"]["officialDate"] = "2027-03-24"
        prior_game["gameData"]["teams"]["away"].update(id=234, abbreviation="BOS", name="Red Sox")
        prior_game["gameData"]["teams"]["home"].update(id=221, abbreviation="NYY", name="Yankees")
        for side in ("away", "home"):
            for p in prior_game["liveData"]["boxscore"]["teams"][side]["players"].values():
                p["person"]["id"] += 500
        raw_2027 = work / "data" / "raw" / "2027"
        raw_2027.mkdir(parents=True, exist_ok=True)
        (raw_2027 / "9000.json").write_text(json.dumps(prior_game))

        date_str = "2027-03-25"
        g1 = _game(date_str, pk=9001)
        g1["gameDate"] = _at(date_str, "19:10").isoformat()
        g2 = _game(date_str, pk=9002)
        g2["gameDate"] = _at(date_str, "19:10").isoformat()
        g2["teams"]["away"]["team"] = {"id": 234, "name": "Red Sox"}
        g2["teams"]["home"]["team"] = {"id": 221, "name": "Yankees"}
        games = [g1, g2]

        clock = {"now": _at(date_str, "10:00"), "mono": 0.0}
        sent = []
        cascade_starts = []

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

        def sleep_fn(seconds, *a, **k):
            clock["now"] += timedelta(seconds=seconds)
            clock["mono"] += seconds

        real_run_pipeline = predict.run_pipeline

        def timed_run_pipeline(*a, **kw):
            cascade_starts.append(clock["now"].strftime("%H:%M"))
            res = real_run_pipeline(*a, **kw)
            # Simulate a 16-minute cascade (well inside an 18-minute cascade_budget_min!)
            if clock["now"] >= _at(date_str, "18:30"):
                sleep_fn(16 * 60)
            return res

        def network(url, **kw):
            url = getattr(url, "full_url", url)
            if "/schedule" in url:
                query = parse_qs(urlparse(url).query)
                if "teamId" in query:
                    tid = int(query["teamId"][0])
                    g_hist = _game("2027-03-24", pk=9000, final=True)
                    g_hist["teams"]["away"]["team"] = {"id": 234, "name": "Red Sox"}
                    g_hist["teams"]["home"]["team"] = {"id": 221, "name": "Yankees"}
                    obj = {"dates": [{"games": [g_hist] if tid in (234, 221) else []}]}
                else:
                    d_arg = query.get("date", [""])[0]
                    if d_arg == date_str:
                        gs = copy.deepcopy(games)
                        if clock["now"] >= _at(date_str, "19:30"):
                            for g in gs:
                                g["status"] = {"abstractGameCode": "F", "detailedState": "Final"}
                        obj = {"dates": [{"games": gs}]}
                    elif d_arg == "2027-03-24":
                        g_hist = _game("2027-03-24", pk=9000, final=True)
                        obj = {"dates": [{"games": [g_hist]}]}
                    else:
                        obj = {"dates": [{"games": []}]}
            elif "/9000/feed/live" in url:
                obj = prior_game
            elif "/9001/feed/live" in url:
                if clock["now"] < _at(date_str, "18:20"):
                    raise OSError("upstream MLB feed outage")
                obj = _live_feed(sample, posted=clock["now"] >= _at(date_str, "18:25"))
                obj["gameData"]["datetime"]["officialDate"] = date_str
                obj["gameData"]["datetime"]["dateTime"] = g1["gameDate"]
                final = clock["now"] >= _at(date_str, "19:30")
                obj["gameData"]["status"] = {
                    "abstractGameCode": "F" if final else "P",
                    "detailedState": "Final" if final else "Scheduled",
                }
            elif "/9002/feed/live" in url:
                if clock["now"] < _at(date_str, "18:20"):
                    raise OSError("upstream MLB feed outage")
                # Game 9002 has NOT posted today's lineup yet at 18:35 (projected from 9000!)
                obj = _live_feed(prior_game, posted=False)
                obj["gameData"]["game"]["pk"] = 9002
                obj["gameData"]["datetime"]["officialDate"] = date_str
                obj["gameData"]["datetime"]["dateTime"] = g2["gameDate"]
                final = clock["now"] >= _at(date_str, "19:30")
                obj["gameData"]["status"] = {
                    "abstractGameCode": "F" if final else "P",
                    "detailedState": "Final" if final else "Scheduled",
                }
            else:
                raise AssertionError(f"unexpected url: {url}")
            return io.BytesIO(json.dumps(obj).encode())

        orig_rsc = sch.run_single_check

        def patched_rsc(*a, **kw):
            r = orig_rsc(*a, **kw)
            if r.get("pick_result") and not r["pick_result"].locked and "lock_decision" not in r:
                # Supply the LockDecision that run_single_check computed on line 1781
                r["lock_decision"] = sch.LockDecision(
                    should_lock=r["should_post"], best_projected=0.95,
                    should_lock_ungated=False, block_reason="projected_leg",
                    contender_game_pk=9002, contender_game_pks=(9002,),
                )
            return r

        import pytest
        with pytest.MonkeyPatch.context() as mp:
            _isolate(mp)
            mp.chdir(work)
            _fake_http(mp, network)
            for m in (sch, picks, alerts, cstate):
                mp.setattr(m, "datetime", Clock)
            mp.setattr(sch, "_watchdog_ping_sleep", sleep_fn)
            mp.setattr(sch.time, "sleep", sleep_fn)
            mp.setattr(sch.time, "monotonic", lambda: clock["mono"])
            mp.setattr(predict, "run_pipeline", timed_run_pipeline)
            mp.setattr("bts.dm.send_dm", lambda *a: sent.append((clock["now"].strftime("%H:%M"), "pick_dm", a[1])) or "dm-id")
            mp.setattr(alerts, "send_dm", lambda *a: sent.append((clock["now"].strftime("%H:%M"), "alert_dm", a[1])) or "dm-id")
            if variant == "WITH_LOCK_DECISION_FIX":
                mp.setattr(sch, "run_single_check", patched_rsc)

            picks_dir = work / "data" / "picks"
            config = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work / "data" / ".heartbeat")},
                "tiers": [{"name": "local", "type": "local"}],
                "scheduler": {
                    "pick_delivery": "dm",
                    "lineup_check_offset_min": 60,
                    "cascade_budget_min": 18,
                    "live_forward_capture_on_lock": False,
                },
                "bluesky": {"dm_recipient": "fake"},
                "health_checks": {"enabled": False},
            }
            sch.run_day(date_str, config)
            st = sch.load_state(date_str, picks_dir)
            dec = load_decision(date_str, picks_dir)
            print(f"  {variant}: cascade_starts={cascade_starts}, pick_locked={st.pick_locked}, "
                  f"fallback_refreshes={[(x['reused_check_cascade'], x['duration_sec']) for x in (st.fallback_refreshes or [])]}, "
                  f"decision={dec and (dec['action'], dec['delivery_status'], dec['scoreable'])}, "
                  f"DMs={[(t, kind, msg[:70]) for t, kind, msg in sent]}")


def probe_2_unbound_local_result(history_root: Path, sample: dict, out_root: Path) -> None:
    print("\n=== 2. S6 UnboundLocalError on `result` When Sleep Wakes Slightly Before Wall-Clock Target ===")
    import bts.scheduler as sch
    import bts.picks as picks
    import bts.health.alert as alerts
    import pytest

    work = out_root / "p2_unbound"
    shutil.copytree(history_root / "data", work / "data")
    date_str = "2027-03-24"
    clock = {"now": _at(date_str, "10:00")}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

    def slightly_early_sleep(seconds, *a, **k):
        # Simulate monotonic sleep waking 50ms before NTP-adjusted wall clock on target sleeps
        clock["now"] += timedelta(seconds=max(0.0, seconds - 0.05))

    def network(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            query = parse_qs(urlparse(url).query)
            games = [_game(date_str)] if query.get("date") == [date_str] else []
            return io.BytesIO(json.dumps({"dates": [{"games": games}]}).encode())
        return io.BytesIO(json.dumps(_live_feed(sample, posted=False)).encode())

    with pytest.MonkeyPatch.context() as mp:
        _isolate(mp)
        mp.chdir(work)
        _fake_http(mp, network)
        for m in (sch, picks, alerts):
            mp.setattr(m, "datetime", Clock)
        mp.setattr(sch, "_watchdog_ping_sleep", slightly_early_sleep)
        mp.setattr(sch.time, "sleep", slightly_early_sleep)
        picks_dir = work / "data" / "picks"
        config = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work / "data" / ".heartbeat")},
            "tiers": [{"name": "local", "type": "local"}],
            "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60, "live_forward_capture_on_lock": False},
            "bluesky": {"dm_recipient": "fake"},
            "health_checks": {"enabled": False},
        }
        try:
            sch.run_day(date_str, config)
            err = "no error"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    print(f"  Outcome on slight early wake: {err}")


def probe_3_postponed_slate_false_critical(history_root: Path, sample: dict, out_root: Path) -> None:
    print("\n=== 3. S6 All-Postponed Slate (12dbaaa vs HEAD vs day_outcome) ===")
    import bts.scheduler as head_sch
    base_sch = _load_base_scheduler()
    import bts.picks as picks
    import bts.health.alert as alerts
    import pytest

    date_str = "2027-03-24"
    for label, mod in (("BASE_12dbaaa", base_sch), ("HEAD", head_sch)):
        work = out_root / f"p3_{label}"
        shutil.copytree(history_root / "data", work / "data")
        clock = {"now": _at(date_str, "10:00")}
        sent = []

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

        def sleep_fn(seconds, *a, **k):
            clock["now"] += timedelta(seconds=seconds)

        def network(url, **kw):
            url = getattr(url, "full_url", url)
            is_ppd = clock["now"] >= _at(date_str, "14:00")
            if "/schedule" in url:
                query = parse_qs(urlparse(url).query)
                if query.get("date") == [date_str]:
                    g = _game(date_str)
                    if is_ppd:
                        g["status"] = {"abstractGameCode": "F", "detailedState": "Postponed"}
                    games = [g]
                else:
                    games = []
                return io.BytesIO(json.dumps({"dates": [{"games": games}]}).encode())
            elif "/9001/feed/live" in url:
                obj = _live_feed(sample, posted=False)
                if is_ppd:
                    obj["gameData"]["status"] = {"abstractGameCode": "F", "detailedState": "Postponed"}
                return io.BytesIO(json.dumps(obj).encode())
            raise AssertionError(url)

        with pytest.MonkeyPatch.context() as mp:
            _isolate(mp)
            mp.chdir(work)
            _fake_http(mp, network)
            mp.setattr(mod, "retry_urlopen", network)
            for m in (mod, picks, alerts):
                mp.setattr(m, "datetime", Clock)
            mp.setattr(mod, "_watchdog_ping_sleep", sleep_fn)
            mp.setattr(mod.time, "sleep", sleep_fn)
            mp.setattr("bts.dm.send_dm", lambda *a: sent.append((clock["now"].strftime("%H:%M"), a[1])) or "dm-id")
            mp.setattr(alerts, "send_dm", lambda *a: sent.append((clock["now"].strftime("%H:%M"), a[1])) or "dm-id")
            picks_dir = work / "data" / "picks"
            config = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work / "data" / ".heartbeat")},
                "tiers": [{"name": "local", "type": "local"}],
                "scheduler": {"pick_delivery": "dm", "entry_intent": "enter",
                              "lineup_check_offset_min": 60, "live_forward_capture_on_lock": False},
                "bluesky": {"dm_recipient": "fake"},
                "health_checks": {"enabled": False},
            }
            mod.run_day(date_str, config)
            st = mod.load_state(date_str, picks_dir)
            do = gather_day_outcome(date_str, picks_dir, config, expected_username="stonehengee",
                                    fetch_schedule=mod.fetch_schedule)
            print(f"  {label:12s}: runs={[r['time'][11:16] for r in st.runs_completed]}, "
                  f"DMs={sent}, day_outcome=(state={do.state}, ok={do.ok})")


def probe_4_post_midnight_and_restart_and_private(history_root: Path, sample: dict, out_root: Path) -> None:
    print("\n=== 4. S6 Post-Midnight Dedup Poisoning, Mid-Day Restart After Alert, and Private Mode ===")
    import bts.scheduler as sch
    import bts.picks as picks
    import bts.health.alert as alerts
    import pytest

    # 4A: Restart at 00:05 ET on D+1 while re-running D -> sends 2nd alert for D AND suppresses D+1's own alert!
    work_a = out_root / "p4_midnight"
    shutil.copytree(history_root / "data", work_a / "data")
    sent_a = []
    clock_a = {"now": _at("2027-03-24", "10:00")}

    class ClockA(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock_a["now"].astimezone(tz) if tz else clock_a["now"].replace(tzinfo=None)

    def sleep_a(seconds, *a, **k):
        clock_a["now"] += timedelta(seconds=seconds)

    def net_a(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            q = parse_qs(urlparse(url).query).get("date", [""])[0]
            if q in ("2027-03-24", "2027-03-25"):
                return io.BytesIO(json.dumps({"dates": [{"games": [_game(q)]}]}).encode())
            return io.BytesIO(json.dumps({"dates": [{"games": []}]}).encode())
        return io.BytesIO(json.dumps(_live_feed(sample, posted=False)).encode())

    with pytest.MonkeyPatch.context() as mp:
        _isolate(mp)
        mp.chdir(work_a)
        _fake_http(mp, net_a)
        for m in (sch, picks, alerts):
            mp.setattr(m, "datetime", ClockA)
        mp.setattr(sch, "_watchdog_ping_sleep", sleep_a)
        mp.setattr(sch.time, "sleep", sleep_a)
        mp.setattr(alerts, "send_dm", lambda *a: sent_a.append((clock_a["now"].strftime("%Y-%m-%d %H:%M"), a[1])) or "dm-id")
        picks_dir = work_a / "data" / "picks"
        cfg = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work_a / "data" / ".heartbeat")},
            "tiers": [{"name": "local", "type": "local"}],
            "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60, "live_forward_capture_on_lock": False},
            "bluesky": {"dm_recipient": "fake"},
            "health_checks": {"enabled": False},
        }
        # Day 1 normal empty day -> sends at 2027-03-24 18:35
        mp.setattr(sch, "_idle_until_next_wakeup", lambda *a, **k: None)
        sch.run_day("2027-03-24", cfg)
        # Suppose a post-midnight call/restart for 2027-03-24 (or a late West Coast 23:55 ET game whose cascade ends 00:05 ET)
        clock_a["now"] = _at("2027-03-25", "00:05")
        sch.run_day("2027-03-24", cfg)
        # Now Day 2 (2027-03-25) runs and ALSO gets no lineups at 18:35!
        clock_a["now"] = _at("2027-03-25", "10:00")
        sch.run_day("2027-03-25", cfg)
    print(f"  4A Post-midnight dedup: DMs sent = {[(t, m.splitlines()[0]) for t, m in sent_a]} "
          f"(Notice: 2nd DM for 03-24 at 00:05 AND 03-25 18:35 DM completely suppressed!)")

    # 4B: Mid-day restart at 18:40 AFTER 18:35 already sent "No pick was selected today... All ... attempts are over"
    work_b = out_root / "p4_restart"
    shutil.copytree(history_root / "data", work_b / "data")
    sent_b = []
    clock_b = {"now": _at("2027-03-24", "10:00")}

    class ClockB(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock_b["now"].astimezone(tz) if tz else clock_b["now"].replace(tzinfo=None)

    def sleep_b(seconds, *a, **k):
        clock_b["now"] += timedelta(seconds=seconds)

    def net_b(url, **kw):
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            q = parse_qs(urlparse(url).query).get("date", [""])[0]
            if q == "2027-03-24":
                g = _game("2027-03-24")
                if clock_b["now"] >= _at("2027-03-24", "19:30"):
                    g["status"] = {"abstractGameCode": "F", "detailedState": "Final"}
                return io.BytesIO(json.dumps({"dates": [{"games": [g]}]}).encode())
            return io.BytesIO(json.dumps({"dates": [{"games": []}]}).encode())
        # Lineups post late at 18:39 (after 18:35 extra attempt, before 18:40 restart!)
        obj = _live_feed(sample, posted=clock_b["now"] >= _at("2027-03-24", "18:39"))
        final_b = clock_b["now"] >= _at("2027-03-24", "19:30")
        obj["gameData"]["status"] = {
            "abstractGameCode": "F" if final_b else "P",
            "detailedState": "Final" if final_b else "Scheduled",
        }
        return io.BytesIO(json.dumps(obj).encode())

    with pytest.MonkeyPatch.context() as mp:
        _isolate(mp)
        mp.chdir(work_b)
        _fake_http(mp, net_b)
        for m in (sch, picks, alerts):
            mp.setattr(m, "datetime", ClockB)
        mp.setattr(sch, "_watchdog_ping_sleep", sleep_b)
        mp.setattr(sch.time, "sleep", sleep_b)
        mp.setattr(sch, "_idle_until_next_wakeup", lambda *a, **k: None)
        mp.setattr("bts.dm.send_dm", lambda *a: sent_b.append((clock_b["now"].strftime("%H:%M"), "pick_dm", a[1])) or "dm-id")
        mp.setattr(alerts, "send_dm", lambda *a: sent_b.append((clock_b["now"].strftime("%H:%M"), "alert_dm", a[1])) or "dm-id")
        picks_dir = work_b / "data" / "picks"
        cfg = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work_b / "data" / ".heartbeat")},
            "tiers": [{"name": "local", "type": "local"}],
            "scheduler": {"pick_delivery": "dm", "lineup_check_offset_min": 60, "live_forward_capture_on_lock": False},
            "bluesky": {"dm_recipient": "fake"},
            "health_checks": {"enabled": False},
        }
        sch.run_day("2027-03-24", cfg)
        clock_b["now"] = _at("2027-03-24", "18:40")
        sch.run_day("2027-03-24", cfg)
        dec_b = load_decision("2027-03-24", picks_dir)
    print(f"  4B Restart at 18:40 after 18:35 empty alert: DMs={[(t, k, m.splitlines()[0][:65]) for t, k, m in sent_b]}, "
          f"final_decision={dec_b and (dec_b['action'], dec_b['delivery_status'])}")

    # 4C: Private / research mode when no pick is selected (still DMs dm_recipient!)
    work_c = out_root / "p4_private"
    shutil.copytree(history_root / "data", work_c / "data")
    sent_c = []
    clock_c = {"now": _at("2027-03-24", "10:00")}

    class ClockC(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock_c["now"].astimezone(tz) if tz else clock_c["now"].replace(tzinfo=None)

    with pytest.MonkeyPatch.context() as mp:
        _isolate(mp)
        mp.chdir(work_c)
        _fake_http(mp, net_a)
        for m in (sch, picks, alerts):
            mp.setattr(m, "datetime", ClockC)
        mp.setattr(sch, "_watchdog_ping_sleep", lambda s, *a, **k: clock_c.update(now=clock_c["now"] + timedelta(seconds=s)))
        mp.setattr(sch.time, "sleep", lambda s, *a, **k: clock_c.update(now=clock_c["now"] + timedelta(seconds=s)))
        mp.setattr(sch, "_idle_until_next_wakeup", lambda *a, **k: None)
        mp.setattr(alerts, "send_dm", lambda *a: sent_c.append((clock_c["now"].strftime("%H:%M"), a[1])) or "dm-id")
        picks_dir = work_c / "data" / "picks"
        cfg = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(work_c / "data" / ".heartbeat")},
            "tiers": [{"name": "local", "type": "local"}],
            "scheduler": {"pick_delivery": "private", "entry_intent": "research",
                          "lineup_check_offset_min": 60, "live_forward_capture_on_lock": False},
            "bluesky": {"dm_recipient": "fake"},
            "health_checks": {"enabled": False},
        }
        sch.run_day("2027-03-24", cfg)
    print(f"  4C Private/research mode w/ no pick selected: DMs={[(t, m.splitlines()[0]) for t, m in sent_c]}")


if __name__ == "__main__":
    import sys
    scratch = Path(".codex-review/sr-review/scratch/tmp_s6")
    shutil.rmtree(scratch, ignore_errors=True)
    hist = scratch / "history"
    sample = _sample_game_feed.__wrapped__()
    _prepare_history(hist, sample)
    only_rest = "--rest" in sys.argv
    if not only_rest:
        probe_1_double_cascade_loses_pick(hist, sample, scratch)
        probe_2_unbound_local_result(hist, sample, scratch)
    probe_3_postponed_slate_false_critical(hist, sample, scratch)
    probe_4_post_midnight_and_restart_and_private(hist, sample, scratch)
