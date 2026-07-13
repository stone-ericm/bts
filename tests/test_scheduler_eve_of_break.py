"""Eve-of-break regression tests (2026-07-12 incident).

The last day before the All-Star break: games today, ZERO games tomorrow.
run_day step 7 called compute_wakeup_time(fetch_schedule(tomorrow)) directly,
and compute_wakeup_time([]) returns TODAY at the default hour — hours in the
past by end-of-day. _idle_until_next_wakeup treats a past wakeup as a silent
no-op, so run_day returned, the process exited 0, and systemd Restart=always
relaunched it every ~48s: 47 restarts, each re-running the EOD health suite
and re-sending the CRITICAL DM (no dedup — see tests/health/test_alert.py).

The audit-E1 helper _next_day_wakeup already existed for the no-games-TODAY
path and handles exactly this ("on a multi-day break compute_wakeup_time
would return today's hour (already past)") — step 7 just never used it.
"""
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import bts.scheduler as sch
from bts.scheduler import _idle_until_next_wakeup, load_state

from tests.test_scheduler_decision_record_integration import _game, _result, _seq

ET = ZoneInfo("America/New_York")


class TestStep7NextWakeup:
    def test_games_today_empty_tomorrow_stores_future_wakeup(self, tmp_path, monkeypatch):
        """Eve-of-break: the stored next_wakeup must be tz-aware and in the
        FUTURE, or the post-EOD idle no-ops and the daemon thrash-restarts.

        The date is REAL today (not a fixed past date): compute_wakeup_time's
        empty-schedule branch anchors to the real wall clock, so a fixed past
        date would make "today 10:00" look future and hide the bug.
        """
        run_date = date.today().isoformat()
        now = datetime.strptime(f"{run_date} 23:30", "%Y-%m-%d %H:%M").replace(tzinfo=ET)

        # Games today, [] for every later fetch (tomorrow) — eve-of-break.
        monkeypatch.setattr(sch, "fetch_schedule",
                            _seq([[_game(100, "19:05", date=run_date)], []]))
        monkeypatch.setattr(sch, "_now_et", lambda: now)
        monkeypatch.setattr(sch.time, "sleep", lambda *a, **k: None)
        monkeypatch.setattr(sch, "run_result_polling", lambda *a, **k: "final")
        monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
        monkeypatch.setattr(sch, "run_single_check", lambda *a, **k: _result(skipped=True))
        config = {
            "orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")},
            "tiers": [],
            "scheduler": {"pick_delivery": "private", "early_lock_gap": 0.03,
                          "lineup_check_offset_min": 45, "cluster_min": 10,
                          "doubleheader_recheck_min": 15, "missed_pick_alert_min": 10,
                          "results_poll_interval_min": 15, "results_cap_hour_et": 5},
            "health_checks": {"enabled": False},
        }
        sch.run_day(date=run_date, config=config, dry_run=False)

        state = load_state(run_date, tmp_path)
        assert state is not None and state.next_wakeup, "step 7 must store next_wakeup"
        wakeup = datetime.fromisoformat(state.next_wakeup)
        assert wakeup.tzinfo is not None, "next_wakeup must be tz-aware"
        assert wakeup > now, (
            f"next_wakeup {wakeup} is not in the future of {now}: "
            "_idle_until_next_wakeup will no-op and Restart=always will thrash"
        )
        assert wakeup >= now.replace(hour=0, minute=0) + timedelta(days=1), (
            "wakeup must be tomorrow, not later today"
        )


class TestIdleGuardLoud:
    """The idle guard's no-op returns stay (substituting a made-up wakeup could
    sleep through a REAL game day — worse than bounded restart churn), but they
    must be LOUD: tonight's silent return cost ~40 minutes of DM spam before
    the journal showed why."""

    def test_past_wakeup_logs_reason(self, capsys):
        past = datetime(2026, 7, 12, 10, 0, tzinfo=ET)
        _idle_until_next_wakeup(past.isoformat(), None)
        err = capsys.readouterr().err
        assert "not idling" in err and "past" in err

    def test_naive_wakeup_logs_reason(self, capsys):
        _idle_until_next_wakeup("2026-07-13T10:00:00", None)
        err = capsys.readouterr().err
        assert "not idling" in err and "naive" in err

    def test_malformed_wakeup_logs_reason(self, capsys):
        _idle_until_next_wakeup("garbage-not-a-date", None)
        err = capsys.readouterr().err
        assert "not idling" in err

    def test_empty_wakeup_logs_reason(self, capsys):
        _idle_until_next_wakeup(None, None)
        err = capsys.readouterr().err
        assert "not idling" in err


class TestNextDayWakeupEarlySlate:
    """Codex review #1/#2 (round 2): routing every games day through
    _next_day_wakeup must not turn its day-bump into slate skips."""

    def test_nonempty_tomorrow_past_wake_hands_off_soon(self, monkeypatch):
        """Post-midnight EOD (result polling caps 05:00) before an early-start
        day (London Series 06:10 ET, wake 05:10): bumping a calendar day would
        sleep through the entire real slate. Must hand off within minutes."""
        from bts.scheduler import _next_day_wakeup
        # It is 05:20 ET on D+1; run_day is still processing date D.
        now = datetime(2026, 6, 14, 5, 20, tzinfo=ET)
        monkeypatch.setattr(sch, "_now_et", lambda: now)
        london = {"gamePk": 1, "gameType": "R",
                  "gameDate": "2026-06-14T10:10:00Z"}  # 06:10 ET
        monkeypatch.setattr(sch, "fetch_schedule", lambda d: [london])

        wakeup = _next_day_wakeup("2026-06-13", {})

        assert wakeup > now
        assert (wakeup - now).total_seconds() <= 5 * 60, (
            f"wakeup {wakeup} skips the 06:10 slate instead of handing off"
        )

    def test_malformed_schedule_entry_hands_off_not_raises(self, monkeypatch):
        """Round-3 fix of round-2 #1: a SUCCESSFUL fetch with a malformed game
        (no gameDate) must not raise through step 7 — that would recreate a
        bare 30s churn loop with no handoff pacing."""
        from bts.scheduler import _next_day_wakeup
        now = datetime(2026, 6, 13, 22, 0, tzinfo=ET)
        monkeypatch.setattr(sch, "_now_et", lambda: now)
        monkeypatch.setattr(sch, "fetch_schedule",
                            lambda d: [{"gamePk": 1, "gameType": "R"}])
        wakeup = _next_day_wakeup("2026-06-13", {})
        assert wakeup > now
        assert (wakeup - now).total_seconds() <= 30 * 60

    def test_fetch_failure_retries_soon_not_next_morning(self, monkeypatch):
        """A transient schedule-fetch failure must not commit to 'tomorrow
        10:00 is safe' — with a 09:05 game that oversleeps first pitch. A
        short handoff lets the exit→restart cycle re-fetch."""
        from bts.scheduler import _next_day_wakeup

        def _boom(d):
            raise OSError("statsapi down")

        now = datetime(2026, 6, 13, 22, 0, tzinfo=ET)
        monkeypatch.setattr(sch, "_now_et", lambda: now)
        monkeypatch.setattr(sch, "fetch_schedule", _boom)

        wakeup = _next_day_wakeup("2026-06-13", {})

        assert wakeup > now
        assert (wakeup - now).total_seconds() <= 30 * 60, (
            f"wakeup {wakeup} accepts an unverified morning instead of retrying"
        )


class TestFetchScheduleGameType:
    def test_non_regular_season_games_filtered(self, monkeypatch):
        """The All-Star Game (gameType A) must not look like a pickable slate:
        BTS is a regular-season contest, and an unfiltered 7/14 would run the
        full lineup-check/pick pipeline against All-Star rosters."""
        payload = {"dates": [{"games": [
            {"gamePk": 1, "gameType": "R", "gameDate": "2026-07-16T23:05:00Z"},
            {"gamePk": 2, "gameType": "A", "gameDate": "2026-07-14T00:00:00Z"},
            {"gamePk": 3, "gameType": "E", "gameDate": "2026-07-14T22:00:00Z"},
        ]}]}

        class _Resp:
            def read(self):
                return json.dumps(payload).encode()

        monkeypatch.setattr(sch, "retry_urlopen", lambda *a, **k: _Resp())
        games = sch.fetch_schedule("2026-07-14")
        assert [g["gamePk"] for g in games] == [1]

    def test_missing_gametype_included(self, monkeypatch):
        """Lenient on absent gameType (older fixtures/mocks omit it)."""
        payload = {"dates": [{"games": [{"gamePk": 7, "gameDate": "2026-07-16T23:05:00Z"}]}]}

        class _Resp:
            def read(self):
                return json.dumps(payload).encode()

        monkeypatch.setattr(sch, "retry_urlopen", lambda *a, **k: _Resp())
        assert [g["gamePk"] for g in sch.fetch_schedule("2026-07-16")] == [7]
