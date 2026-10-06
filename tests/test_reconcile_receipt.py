"""Watchdog plan P2: the reconcile receipt (registration I-207 and the producer receipt contract,
docs/sota_audit/2026-10-04-prereg-c1-watchdog.md; schema docs/ops/reconcile-receipt-v1.md).

The real grading path runs (`resolve_daily_slot_results` -> `get_game_statuses_detailed` / `check_hit` ->
`retry_urlopen`). Only `bts.picks.retry_urlopen` is faked, by URL, so the receipt's source hashes come from the payload
bytes the grader actually parsed. No network is used.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bts.picks import DailyPick, Pick, load_pick, reconcile_results, save_pick, save_streak
from bts.reconcile_receipt import ReconcileReceipt
from bts.reconcile_receipt import discover as rr_discover

ET = ZoneInfo("America/New_York")
BATTER, GAME, DD_BATTER, DD_GAME = 802415, 822934, 700001, 822935
DAY = "2026-08-20"


class SimClock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


def plant(picks_dir, day=DAY, *, result="hit", dd=False):
    def mk(bid, gpk, name):
        return Pick(batter_name=name, batter_id=bid, team="TB", lineup_position=1, pitcher_name="P", pitcher_id=456,
                    p_game_hit=0.79, flags=[], projected_lineup=False, game_pk=gpk, game_time=f"{day}T17:10:00Z")
    slots = {"pick": result} | ({"double_down": result} if dd else {})
    save_pick(DailyPick(date=day, run_time=f"{day}T15:00:00Z", pick=mk(BATTER, GAME, "Chandler Simpson"),
                        double_down=mk(DD_BATTER, DD_GAME, "DD Batter") if dd else None, runner_up=None,
                        result=result, slot_results=slots), picks_dir)
    save_streak(1, picks_dir)


def feed(hits=0, *, code="F", batter=BATTER, name="Chandler Simpson", pk=None):
    return {"gameData": {"status": {"abstractGameCode": code}, "datetime": {}, **({"game": {"pk": pk}} if pk else {})},
            "liveData": {"boxscore": {"teams": {"away": {"players": {f"ID{batter}": {
                "person": {"fullName": name}, "stats": {"batting": {"hits": hits}}}}}, "home": {"players": {}}}},
                "plays": {"allPlays": []}}}


def schedule(states=None):
    states = states or {}
    return {"dates": [{"games": [{"gamePk": pk, "status": {"abstractGameCode": "F", "detailedState": st,
                                                          "statusCode": "F"}}
                                 for pk, st in ({GAME: "Final", DD_GAME: "Final"} | states).items()]}]}


class Net:
    """A fake retry_urlopen keyed by URL. `on` maps a URL substring to a callable run before answering."""

    def __init__(self, routes, on=None):
        self.routes, self.on, self.calls, self.bodies = routes, on or {}, [], {}

    def __call__(self, url, timeout=15, **kw):
        self.calls.append(url)
        for key, hook in self.on.items():
            if key in url:
                hook()
        for key, payload in self.routes.items():
            if key in url:
                if isinstance(payload, Exception):
                    raise payload
                body = json.dumps(payload).encode()
                self.bodies[url] = body

                class R:
                    def read(self_inner):
                        return body
                return R()
        raise AssertionError(f"unexpected url {url}")


def run(picks_dir, net, clock, *, receipt=True, lookback=8):
    rec = ReconcileReceipt(clock=clock, lookback_days=lookback) if receipt else None
    with patch("bts.picks.retry_urlopen", net):
        out = reconcile_results(picks_dir, lookback_days=lookback, clock=clock, receipt=rec)
    if rec:
        rec.finish(out)                                   # as the CLI does
    return out, (rec.record() if rec else None)


def day_of(r, d=DAY):
    (x,) = [x for x in r["days"] if x["date"] == d]
    return x


def sha(b):
    return hashlib.sha256(b).hexdigest()


AT_0200 = datetime(2026, 8, 21, 2, 0, tzinfo=ET)


# ---- per-date and per-slot states ---------------------------------------------------------------------------------

def test_a_pre_cutoff_correction_is_observed_and_applied(tmp_path):
    plant(tmp_path)
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)})
    corrections, r = run(tmp_path, net, SimClock(AT_0200))
    assert [c["new_result"] for c in corrections] == ["miss"]
    d = day_of(r)
    assert d["state"] == "observed" and d["write"]["state"] == "correction_applied"
    assert (d["write"]["old_result"], d["write"]["new_result"]) == ("hit", "miss")
    assert datetime.fromisoformat(d["cutoff_at"]) == datetime(2026, 8, 21, 8, 0, tzinfo=ET)
    assert d["status_source"]["sha256"] == sha(net.bodies[next(u for u in net.bodies if "/schedule" in u)])
    (slot,) = d["slots"]
    assert (slot["slot"], slot["batter_id"], slot["game_pk"], slot["state"], slot["result"], slot["basis"]) == \
        ("pick", BATTER, GAME, "observed", "miss", "final_feed")
    assert slot["sources"][0]["sha256"] == sha(net.bodies[next(u for u in net.bodies if f"/game/{GAME}/" in u)])
    assert slot["covered"] is True and datetime.fromisoformat(slot["response_completed_at"]) == AT_0200
    assert r["corrections"] == corrections and r["replay"]["state"] == "saved"


def test_an_unchanged_result_is_coverage_even_with_no_corrections(tmp_path):
    """I-207: an empty corrections list proves nothing; the receipt says the slot was observed and unchanged."""
    plant(tmp_path)
    corrections, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1)}), SimClock(AT_0200))
    assert corrections == []
    d = day_of(r)
    assert d["write"]["state"] == "unchanged" and d["slots"][0]["covered"] is True


def test_dates_outside_the_window_or_without_a_graded_pick_are_skipped_with_a_reason(tmp_path):
    """At 8/20 02:00 only 8/19 is still before its cutoff (8/20 08:00); every older date is final."""
    at = SimClock(datetime(2026, 8, 20, 2, 0, tzinfo=ET))
    plant(tmp_path, "2026-08-14")
    _, r = run(tmp_path, Net({}), at)
    states = {x["date"]: x["state"] for x in r["days"]}
    assert states.pop("2026-08-19") == "no_pick_file" and set(states.values()) == {"past_cutoff"} and len(r["days"]) == 8
    save_pick(DailyPick(date="2026-08-19", run_time="x", pick=load_pick("2026-08-14", tmp_path).pick, double_down=None,
                        runner_up=None, result=None), tmp_path)
    _, r = run(tmp_path, Net({}), at)
    assert day_of(r, "2026-08-19")["state"] == "not_graded" and day_of(r, "2026-08-19")["detail"] == {"result": None}


def test_a_pending_slot_is_not_coverage(tmp_path):
    plant(tmp_path)
    _, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(code="L")}), SimClock(AT_0200))
    d = day_of(r)
    assert d["state"] == "pending" and d["slots"][0]["state"] == "pending" and d["slots"][0]["covered"] is False
    assert d["write"] is None


def test_a_pending_first_slot_leaves_the_double_down_not_attempted(tmp_path):
    plant(tmp_path, dd=True)
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(code="L"), f"/game/{DD_GAME}/": feed(hits=1)})
    _, r = run(tmp_path, net, SimClock(AT_0200))
    assert [(s["slot"], s["state"]) for s in day_of(r)["slots"]] == [("pick", "pending"),
                                                                     ("double_down", "not_attempted")]
    assert not any(f"/game/{DD_GAME}/" in u for u in net.calls)              # behaviour: it was never fetched


def test_a_schedule_void_slot_records_its_basis_without_a_feed(tmp_path):
    plant(tmp_path)
    net = Net({"/schedule": schedule({GAME: "Postponed"})})
    _, r = run(tmp_path, net, SimClock(AT_0200))
    (slot,) = day_of(r)["slots"]
    assert (slot["result"], slot["basis"], slot["sources"]) == ("void", "schedule_void_state:Postponed", [])


def test_a_status_fetch_failure_is_recorded(tmp_path):
    plant(tmp_path)
    import urllib.error
    _, r = run(tmp_path, Net({"/schedule": urllib.error.URLError("down"), f"/game/{GAME}/": feed(hits=1)}),
               SimClock(AT_0200))
    d = day_of(r)
    assert d["status_source"] == {"ok": False, "error": "URLError"} and d["slots"][0]["state"] == "observed"


def test_a_feed_failure_is_recorded_and_still_raises(tmp_path):
    plant(tmp_path)
    import urllib.error
    rec = ReconcileReceipt(clock=SimClock(AT_0200), lookback_days=8)
    with patch("bts.picks.retry_urlopen", Net({"/schedule": schedule(), f"/game/{GAME}/": urllib.error.URLError("x")})):
        with pytest.raises(urllib.error.URLError):
            reconcile_results(tmp_path, lookback_days=8, clock=SimClock(AT_0200), receipt=rec)
    r = rec.record()
    d = day_of(r)
    assert d["state"] == "failed" and d["slots"][0]["state"] == "failed" and d["slots"][0]["error"] == "URLError"
    assert r["replay"]["state"] == "not_reached"


def test_a_feed_answer_after_the_cutoff_is_a_late_observation(tmp_path):
    plant(tmp_path)
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET))
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)},
              on={f"/game/{GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET))})
    corrections, r = run(tmp_path, net, clock)
    d = day_of(r)
    assert corrections == [] and d["state"] == "late_answer" and d["slots"][0]["covered"] is False
    assert datetime.fromisoformat(d["slots"][0]["response_completed_at"]) == datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)


def test_a_write_landing_after_the_cutoff_is_refused_and_recorded(tmp_path):
    import bts.picks as picks_mod
    plant(tmp_path)
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 58, tzinfo=ET))
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def slow_lock(picks_dir):
        clock.now = datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)
        with real_lock(picks_dir):
            yield
    with patch("bts.picks.scoring_lock", slow_lock):
        corrections, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)}), clock)
    d = day_of(r)
    assert corrections == [] and d["state"] == "observed" and d["write"]["state"] == "refused_after_cutoff"
    assert load_pick(DAY, tmp_path).result == "hit"


# ---- behaviour unchanged ------------------------------------------------------------------------------------------

def test_the_receipt_changes_no_request_or_result(tmp_path):
    for with_receipt in (False, True):
        d = tmp_path / str(with_receipt)
        d.mkdir()
        plant(d, dd=True)
        net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0),
                   f"/game/{DD_GAME}/": feed(hits=1, batter=DD_BATTER, name="DD Batter")})
        out, _ = run(d, net, SimClock(AT_0200), receipt=with_receipt)
        streak = json.loads((d / "streak.json").read_text())
        streak.pop("updated")                                     # streak.json's own write time
        yield_ = (net.calls, out, load_pick(DAY, d).slot_results, streak)
        if with_receipt:
            assert yield_ == baseline
        else:
            baseline = yield_


def test_a_receipt_method_failure_never_changes_the_run(tmp_path, monkeypatch):
    import bts.reconcile_receipt as rr
    plant(tmp_path)
    monkeypatch.setattr(ReconcileReceipt, "slot_done", rr._guarded(lambda self, *a: 1 / 0))
    corrections, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)}), SimClock(AT_0200))
    assert [c["new_result"] for c in corrections] == ["miss"]
    assert [(g["hook"], g["error"], g["day"], g["slot"]) for g in r["degraded"]] == [("<lambda>", "ZeroDivisionError", 0, 0)]
    assert day_of(r)["slots"][0]["covered"] is False


# ---- the CLI ------------------------------------------------------------------------------------------------------

def _cli(tmp_path, net):
    from click.testing import CliRunner
    from bts.cli import cli
    picks = tmp_path / "picks"
    picks.mkdir()
    with patch("bts.picks.retry_urlopen", net):
        res = CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(picks)])
    from bts.reconcile_receipt import receipts_root
    root = receipts_root(picks)
    recs = []
    for day in (sorted(root.iterdir()) if root.exists() else []):
        recs += rr_discover(picks, day.name)
    return res, recs


def test_the_cli_publishes_one_receipt_per_run_and_keeps_its_output(tmp_path):
    res, recs = _cli(tmp_path, Net({}))
    assert res.exit_code == 0 and "No scoring changes detected" in res.output
    (r,) = recs
    assert r["schema"] == "bts_reconcile_receipt_v1" and r["outcome"] == "completed" and len(r["days"]) == 8
    assert r["producer"]["command"] == "reconcile" and len(r["run_id"]) == 32
    for k in ("started_at", "finished_at", "published_at"):
        assert datetime.fromisoformat(r[k]).tzinfo is not None


@pytest.mark.parametrize("step", ["before_rename", "after_rename"])
def test_a_failed_publication_changes_nothing_else_and_is_never_discoverable(tmp_path, monkeypatch, step):
    import os
    import bts.receipt_io as rio
    real_replace, real_dirsync = os.replace, rio._fsync_dir
    state = {"replaced": False}
    if step == "before_rename":
        monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disk full"))
                            if "reconcile_receipts" in str(b) else real_replace(a, b))
    else:
        def replace_then_arm(a, b):
            real_replace(a, b)
            state["replaced"] = state["replaced"] or "reconcile_receipts" in str(b)
        monkeypatch.setattr(os, "replace", replace_then_arm)
        monkeypatch.setattr(rio, "_fsync_dir", lambda p: (_ for _ in ()).throw(OSError("dirsync"))
                            if state["replaced"] else real_dirsync(p))
    res, recs = _cli(tmp_path, Net({}))
    assert res.exit_code == 0 and "No scoring changes detected" in res.output
    assert "reconcile receipt unavailable" in res.output and recs == []


def test_a_response_after_the_cutoff_is_never_coverage_even_if_the_clock_steps_back(tmp_path, monkeypatch):
    """The feed answers at 08:00:01, then the wall clock steps back to 07:59:59 (e.g. an NTP correction) before the
    run's own arrival check. The run sees the day as on time, but the receipt keeps the slot's actual response time
    and never counts it as coverage."""
    plant(tmp_path)
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 58, tzinfo=ET))
    real_source = ReconcileReceipt.source

    def source_then_step_back(self, url, raw):
        real_source(self, url, raw)
        if f"/game/{GAME}/" in url:
            clock.now = datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET)
    monkeypatch.setattr(ReconcileReceipt, "source", source_then_step_back)
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1)},
              on={f"/game/{GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET))})
    _, r = run(tmp_path, net, clock)
    d = day_of(r)
    assert d["state"] == "observed"                                          # the run's view
    slot = d["slots"][0]
    assert datetime.fromisoformat(slot["response_completed_at"]) == datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)
    assert slot["covered"] is False



# ---- producer review r2 D1-D5 ------------------------------------------------------------------------------------
ALT = 822999


def test_d1_a_lost_feed_observation_is_never_coverage(tmp_path, monkeypatch):
    import bts.reconcile_receipt as rr
    plant(tmp_path)
    real = ReconcileReceipt.source

    @rr._guarded
    def lose_feeds(self, url, raw):
        if "/feed/live" in url:
            raise RuntimeError("recording failure")
        return real.__wrapped__(self, url, raw)
    monkeypatch.setattr(ReconcileReceipt, "source", lose_feeds)
    corrections, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)}), SimClock(AT_0200))
    assert [c["new_result"] for c in corrections] == ["miss"]                 # the run is unchanged
    slot = day_of(r)["slots"][0]
    assert slot["state"] == "observed" and slot["sources"] == [] and slot["basis"] == "unqualified"
    assert slot["degraded"] and slot["covered"] is False


def test_d2_a_late_response_is_not_erased_by_a_later_step_back(tmp_path):
    """Double-down: the primary's feed answers at 08:00:01, then the clock steps back and the second leg answers at
    07:59:59. The run sees an on-time day; neither slot is coverage."""
    plant(tmp_path, dd=True)
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 58, tzinfo=ET))
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1),
               f"/game/{DD_GAME}/": feed(hits=1, batter=DD_BATTER, name="DD Batter")},
              on={f"/game/{GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)),
                  f"/game/{DD_GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET))})
    _, r = run(tmp_path, net, clock)
    d = day_of(r)
    assert d["state"] == "observed"                                           # the run's own view
    assert [s["covered"] for s in d["slots"]] == [False, False]
    assert all(s["clock_regression"] for s in d["slots"])


def test_d2_a_late_selected_feed_then_an_on_time_fallback_is_not_coverage(tmp_path):
    plant(tmp_path)
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 57, tzinfo=ET))
    calls = {"n": 0}

    def schedule_hook():
        calls["n"] += 1
        if calls["n"] == 2:                                                    # the fallback search, after a step back
            clock.now = datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET)
    net = Net({"/schedule": schedule({ALT: "Final"}), f"/game/{GAME}/": feed(hits=1, batter=111, name="Other"),
               f"/game/{DD_GAME}/": feed(hits=0, batter=222, name="Nobody"), f"/game/{ALT}/": feed(hits=1)},
              on={"/schedule": schedule_hook,
                  f"/game/{GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET))})
    _, r = run(tmp_path, net, clock)
    slot = day_of(r)["slots"][0]
    assert slot["clock_regression"] is True and slot["covered"] is False
    assert datetime.fromisoformat(slot["high_water_at"]) == datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)


@pytest.mark.parametrize("alt_code, basis, covered", [("L", "unqualified", False), ("F", "fallback_other_game", False)])
def test_d3_the_basis_comes_from_the_consumed_payloads(tmp_path, alt_code, basis, covered):
    """The selected game's feed lacks the batter; another game (Final per the schedule) has him with a hit. The
    legacy grader returns hit either way; the receipt qualifies the decisive feed itself."""
    plant(tmp_path, result="miss")
    net = Net({"/schedule": schedule({ALT: "Final"}), f"/game/{GAME}/": feed(hits=1, batter=111, name="Other"),
               f"/game/{DD_GAME}/": feed(hits=0, batter=222, name="Nobody"), f"/game/{ALT}/": feed(hits=1, code=alt_code)})
    corrections, r = run(tmp_path, net, SimClock(AT_0200))
    assert [c["new_result"] for c in corrections] == ["hit"]                  # legacy grading, unchanged
    slot = day_of(r)["slots"][0]
    assert (slot["basis"], slot["covered"]) == (basis, covered)
    if alt_code == "F":
        assert slot["actual_game_pk"] == ALT


def test_d3_a_feed_naming_another_game_is_not_the_selected_game(tmp_path):
    plant(tmp_path)
    _, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1, pk=999)}), SimClock(AT_0200))
    slot = day_of(r)["slots"][0]
    assert (slot["basis"], slot["actual_game_pk"], slot["covered"]) == ("fallback_other_game", 999, False)


def test_d4_a_failed_save_is_never_recorded_as_applied(tmp_path, monkeypatch):
    import bts.picks as picks_mod
    plant(tmp_path)
    monkeypatch.setattr(picks_mod, "save_pick", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    rec = ReconcileReceipt(clock=SimClock(AT_0200), lookback_days=8)
    with patch("bts.picks.retry_urlopen", Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)})):
        with pytest.raises(OSError):
            reconcile_results(tmp_path, lookback_days=8, clock=SimClock(AT_0200), receipt=rec)
    rec.raised(OSError())
    w = day_of(rec.record())["write"]
    assert (w["state"], w["intended"], w["completed"]) == ("write_not_completed", "correction_applied", False)
    assert load_pick(DAY, tmp_path).result == "hit"


def test_d4_the_written_selection_is_recorded_separately(tmp_path):
    import bts.picks as picks_mod
    plant(tmp_path)
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def swap_then_lock(picks_dir):
        daily = load_pick(DAY, picks_dir)
        daily.pick.batter_id = 99
        save_pick(daily, picks_dir)
        with real_lock(picks_dir):
            yield
    with patch("bts.picks.scoring_lock", swap_then_lock):
        _, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)}), SimClock(AT_0200))
    w = day_of(r)["write"]
    assert w["state"] == "correction_applied" and w["selection_changed"] is True
    assert w["written_selection"][0]["batter_id"] == 99
    assert day_of(r)["selection"]["slots"][0]["batter_id"] == BATTER


def test_d5_receipt_processing_cannot_move_the_run_past_its_cutoff(tmp_path, monkeypatch):
    """Hashing and parsing happen at publication: a slow receipt computation cannot change a 07:59:59 decision."""
    import bts.reconcile_receipt as rr
    results = {}
    for with_receipt in (False, True):
        d = tmp_path / str(with_receipt)
        d.mkdir()
        plant(d)
        clock = SimClock(datetime(2026, 8, 21, 7, 59, 58, tzinfo=ET))
        real_sha = rr.hashlib.sha256

        def slow_sha(*a, _clock=clock, **k):
            _clock.now = datetime(2026, 8, 21, 8, 0, 5, tzinfo=ET)
            return real_sha(*a, **k)
        monkeypatch.setattr(rr.hashlib, "sha256", slow_sha)
        out, _ = run(d, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=0)}), clock, receipt=with_receipt)
        monkeypatch.setattr(rr.hashlib, "sha256", real_sha)
        results[with_receipt] = (out, load_pick(DAY, d).result)
    assert results[True] == results[False] and results[True][1] == "miss"


def test_d1_a_day_level_hook_failure_uncovers_its_slots(tmp_path, monkeypatch):
    import bts.reconcile_receipt as rr
    plant(tmp_path)
    monkeypatch.setattr(ReconcileReceipt, "attempt", rr._guarded(lambda self, daily: 1 / 0))
    _, r = run(tmp_path, Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1)}), SimClock(AT_0200))
    slot = day_of(r)["slots"][0]
    assert slot["basis"] == "final_feed" and slot["sources"] and slot["degraded"] and slot["covered"] is False


def test_d2_an_out_of_order_clock_before_the_cutoff_is_still_not_coverage(tmp_path):
    plant(tmp_path)
    clock = SimClock(datetime(2026, 8, 21, 7, 0, 0, tzinfo=ET))
    net = Net({"/schedule": schedule(), f"/game/{GAME}/": feed(hits=1)},
              on={f"/game/{GAME}/": lambda: setattr(clock, "now", datetime(2026, 8, 21, 6, 0, 0, tzinfo=ET))})
    _, r = run(tmp_path, net, clock)
    slot = day_of(r)["slots"][0]
    assert slot["clock_regression"] is True and slot["covered"] is False
