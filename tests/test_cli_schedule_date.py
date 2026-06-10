"""`bts schedule` default-date timezone (audit finding O2).

The scheduler runs on an ET box and restarts at arbitrary times (deploys, OOM,
watchdog). If the default 'today' is derived in UTC, a restart between ~8pm and
midnight ET initializes run_day for *tomorrow* and abandons tonight's result
polling + late-slate pick delivery. The default must be the Eastern date.
"""
from datetime import datetime as _dt
from zoneinfo import ZoneInfo
from unittest.mock import patch

import bts.cli as cli


def test_schedule_defaults_to_eastern_date_not_utc(tmp_path):
    # 2026-04-03 22:30 ET == 2026-04-04 02:30 UTC: UTC date is already tomorrow.
    fixed = _dt(2026, 4, 3, 22, 30, tzinfo=ZoneInfo("America/New_York"))
    captured = {}

    with patch("bts.cli.datetime") as mock_dt, \
            patch("bts.scheduler.run_day",
                  side_effect=lambda date, config, dry_run: captured.update(date=date)), \
            patch("bts.orchestrator.load_config", return_value={}):
        mock_dt.now.return_value = fixed
        cli.schedule.callback(date=None, config_path="ignored", dry_run=True)

    assert captured["date"] == "2026-04-03", (
        "schedule defaulted to the UTC date; an evening ET restart would skip "
        "tonight's games"
    )
