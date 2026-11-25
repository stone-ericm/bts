"""Exercise the 00:20 morning wording while a prior entry awaits settlement."""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick


now = datetime.fromisoformat("2027-07-16T00:20:00-04:00")
prior, today = "2027-07-15", "2027-07-16"
with TemporaryDirectory(dir=Path(__file__).parent) as tmp, walk.offline(now):
    path = Path(tmp) / "picks"
    path.mkdir()
    assert not walk.install_state(path, (now - timedelta(days=1)).date(),
                                  walk.State("automatic", 12, 14, "active", .99))
    walk._set_clock(now)
    daily = daily_pick()
    daily.delivered_at = "2027-07-15T23:30:00-04:00"
    daily.notification_sent = True
    daily.notification_id = "prior-pick"
    picks.save_pick(daily, path)
    assert picks.load_pick(prior, path).result is None
    config = walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}
    bodies = []
    with patch.object(alert, "send_dm", side_effect=lambda _, body: bodies.append(body) or "warning"):
        scheduler._morning_state_check(config, path, today, now, now + timedelta(hours=12))
    assert "Before the first check, if prior entries are settled or absent" in bodies[0]
    assert f"--date {today}" in bodies[0]
    assert not (path / "account_state/contest_streak.manual.json").exists()
    print("00:20 ET. July 15 pending in MLB fixture. Following the condition skips the July 16 write.")
    print(bodies[0])
