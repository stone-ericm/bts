"""Drive different real refusal causes across midnight and inspect status."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import scheduler
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health import alert
from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline


logging.disable(logging.CRITICAL)
scratch = sys.argv[1] == "scratch"
day = "2027-06-14"
before = datetime(2027, 6, 14, 23, 55, tzinfo=ET)
after = datetime(2027, 6, 15, 0, 2, tzinfo=ET)

with TemporaryDirectory() as temp, offline(before):
    picks = Path(temp) / "picks"
    rc, output = _invoke("season-contract", "enable", "--picks-dir", picks, "--season", 2027)
    assert rc == 0, output
    cfg = {"orchestrator": {"picks_dir": str(picks)}, "bluesky": {"dm_recipient": "op.invalid"}}
    sent = []

    def capture(_, text):
        assert len(text) < 1000
        sent.append(text)
        return "wire"

    with patch.object(alert, "send_dm", capture):
        for at in (before, after):
            _set_clock(at)
            if at == after:
                (picks / "account_state/contest_streak.json").write_text("[]")
            try:
                load_decision_streak_state(picks, require_contest_state=True)
            except ContestStateError as error:
                print("real refusal", at.isoformat(), str(error).splitlines()[0])
                scheduler._alert_contest_state_failure(cfg, error, **({"slate_date": day} if scratch else {}))
    assert len(sent) == (1 if scratch else 2)
    print("accepted alerts", len(sent))
    (picks / "account_state/contest_streak.json").unlink()
    rc, output = _invoke("season-contract", "status", "--picks-dir", picks)
    assert rc == 0
    template = next(line for line in output.splitlines() if "Alternative dated path" in line)
    print("D+1 repair for D alert", template)
    assert "--date 2027-06-15" in template

with TemporaryDirectory() as temp, offline(before):
    picks = Path(temp) / "picks"
    picks.mkdir()
    (picks / "streak.json").write_text(json.dumps({"season": 2027, "streak": True}))
    try:
        load_decision_streak_state(picks, require_contest_state=False)
    except ContestStateError as error:
        print("model-only refuses", str(error))
    rc, output = _invoke("season-contract", "status", "--picks-dir", picks)
    assert rc == 0
    verdict = next(line for line in output.splitlines() if line.startswith("live decision now:"))
    assert "switch is absent" in verdict
    print("model-only status verdict", verdict)
