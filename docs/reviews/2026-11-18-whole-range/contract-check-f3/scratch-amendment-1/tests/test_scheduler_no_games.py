"""No-games days must idle until tomorrow, not thrash-restart (audit E1).

On an off-day run_day returned immediately; systemd Restart=always then relaunched
within ~30s, cycling all day (the All-Star break is ~4 days). That spikes
NRestarts -> false restart_spike CRITICAL. The wake time must always be in the
future, even on a multi-day break where compute_wakeup_time([]) returns today.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def test_next_day_wakeup_is_future_on_multiday_break(monkeypatch):
    from bts.scheduler import _next_day_wakeup

    # Evening of an off-day; tomorrow ALSO has no games (mid-break).
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda d: [])
    fixed = datetime(2026, 7, 15, 22, 0, tzinfo=ET)  # 10pm ET
    monkeypatch.setattr("bts.scheduler._now_et", lambda: fixed)

    wakeup = _next_day_wakeup("2026-07-15", {})

    assert wakeup > fixed, "wakeup must be in the future or _idle no-ops -> thrash"
    assert wakeup.date() > fixed.date(), "must wake tomorrow, not today"
