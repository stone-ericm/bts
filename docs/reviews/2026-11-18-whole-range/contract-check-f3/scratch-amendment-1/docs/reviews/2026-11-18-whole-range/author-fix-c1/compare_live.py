"""Render current-mode results for byte comparison across base and C1."""

from dataclasses import asdict
from datetime import timedelta
import json
import logging
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from bts import cli, scheduler
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health import alert


ROOT = Path(__file__).resolve().parents[2]
history = runpy.run_path(str(ROOT / "tests/test_wr_c1_history.py"))
probe = SimpleNamespace(**runpy.run_path(str(
    ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-c1/probe.py"
)))
logging.disable(logging.CRITICAL)

with TemporaryDirectory(prefix="wr-c1-live-") as temp:
    picks = Path(temp) / "picks"
    day = probe.D
    following = day + timedelta(days=1)
    config = {"bluesky": {"dm_recipient": "offline.test"}}
    result = {}
    for label, accepted in (("refusal", False), ("acceptance", True)):
        probe.auto(picks, following, recorded_at=probe.at(following, 9).isoformat())
        probe.switch(picks, following, status="automatic" if accepted else "revoked")
        probe.saver_flag(picks, following, state="active" if accepted else "not_earned")
        now = probe.at(following, 10)
        with patch.object(cli, "_now_et", return_value=now):
            status = CliRunner().invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
        assert status.exit_code == 0, status.exception
        try:
            decision = repr(load_decision_streak_state(picks, require_contest_state=True, now=now))
        except ContestStateError as error:
            decision = str(error)
        with patch.object(alert, "send_dm", return_value="offline-ack") as transport:
            scheduler._morning_state_check(config, picks, str(following), now, probe.at(following, 11))
        alerts, wire = history["run"](picks, following, now, isolate=True)
        result[label] = {
            "status": status.output, "decision": decision,
            "morning": [call.args for call in transport.call_args_list],
            "runner": [asdict(item) for item in alerts], "wire": wire,
        }
    print(json.dumps(result, indent=2, sort_keys=True).replace(temp, "<fixture>"))
