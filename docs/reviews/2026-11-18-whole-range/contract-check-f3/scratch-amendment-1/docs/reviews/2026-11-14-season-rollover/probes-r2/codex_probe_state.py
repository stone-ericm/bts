"""Read-only source review: compare season loaders with 12dbaaa and exercise CLI."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
from click.testing import CliRunner

import bts.contest_state as current
from bts.cli import cli
from bts.picks import update_streak, load_streak, reconcile_results
from tests.conftest import sample_game_feed
from tests.test_cold_start_integration import (
    DATE, ET, _fake_http, _game, _isolate, _live_feed, _prepare_history,
)

HERE = Path(__file__).resolve().parent
source = subprocess.check_output(["git", "show", "12dbaaa:src/bts/contest_state.py"], text=True)
base = types.ModuleType("bts._review_base_contest")
sys.modules[base.__name__] = base
exec(compile(source, "12dbaaa:src/bts/contest_state.py", "exec"), base.__dict__)
NOW = datetime(2027, 3, 24, 16, tzinfo=timezone.utc)


def capture(fn):
    try:
        state = fn()
        return {"streak": state.streak, "status": state.status}
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


out = {"corrupt_historical_pick": []}
for payload in ([], None, {"result": []}, {"result": {}}):
    for observation in (None, {"active_streak": 0, "recorded_at": "2027-03-24T12:00:00Z"}):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            picks = Path(directory)
            (picks / "2026-09-26.json").write_text(json.dumps(payload))
            if observation is not None:
                (picks / "account_state").mkdir()
                (picks / "account_state" / "contest_streak.json").write_text(json.dumps(observation))
            before = capture(lambda: base.load_decision_streak_state(picks, now=NOW))
            after = capture(lambda: current.load_decision_streak_state(picks, now=NOW))
            assert "error" not in before and after["error"] in ("AttributeError", "TypeError")
            out["corrupt_historical_pick"].append({"payload": payload, "observation": observation,
                                                    "base": before, "head": after})

with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    (picks / "2026-09-26.json").write_text("[]")
    (picks / "account_state").mkdir()
    (picks / "account_state" / "contest_streak.json").write_text(json.dumps({
        "active_streak": 0, "recorded_at": "2027-03-24T12:00:00Z"}))
    out["required_healthy_first_fetch"] = {
        "base": capture(lambda: base.load_decision_streak_state(picks, now=NOW, require_contest_state=True)),
        "head": capture(lambda: current.load_decision_streak_state(picks, now=NOW, require_contest_state=True))}
    assert out["required_healthy_first_fetch"]["base"]["streak"] == 0
    assert out["required_healthy_first_fetch"]["head"]["error"] == "AttributeError"

# First settlement is deliberately probed before the next season replay/reconcile.
with tempfile.TemporaryDirectory(dir=HERE) as directory:
    picks = Path(directory)
    (picks / "streak.json").write_text('{"streak":12,"saver_available":true}')
    first = current.load_decision_streak_state(picks, now=NOW).streak
    scored = update_streak([False], picks, date=DATE)
    (picks / f"{DATE}.json").write_text('{"result":"miss"}')
    second = current.load_decision_streak_state(picks, now=NOW).streak
    out["first_settlement_before_reconcile"] = {"first_decision": first, "first_miss_scored": scored,
                                                  "next_decision": second}
    assert (first, scored, second) == (0, 12, 12)
    reconcile_results(picks, clock=lambda: NOW)
    out["first_settlement_before_reconcile"]["after_reconcile"] = load_streak(picks)
    assert load_streak(picks) == 0

# Full preview command, real build/features/training/selection. HTTP is synthetic.
with tempfile.TemporaryDirectory(dir=HERE) as directory, pytest.MonkeyPatch.context() as mp:
    root = Path(directory)
    mp.setenv("LOKY_MAX_CPU_COUNT", "1")
    mp.setenv("BTS_USE_CALIBRATION", "0")
    mp.delenv("BTS_REFRESH_ALWAYS", raising=False)
    mp.setattr("bts.strategy._mdp_cache", None)
    _prepare_history(root, sample_game_feed.__wrapped__())
    mp.chdir(root)
    picks = root / "data" / "picks"
    picks.mkdir()
    (picks / "2026-09-26.json").write_text("[]")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    mp.setattr(current, "datetime", Clock)
    def network(url, **kwargs):
        from urllib.parse import parse_qs, urlparse
        url = getattr(url, "full_url", url)
        if "/schedule" in url:
            games = [_game()] if parse_qs(urlparse(url).query).get("date") == [DATE] else []
            body = {"dates": [{"games": games}]}
        elif "/9001/feed/live" in url:
            body = _live_feed(sample_game_feed.__wrapped__(), posted=True)
        else:
            raise AssertionError(f"Unfaked HTTP: {url}")
        return io.BytesIO(json.dumps(body).encode())
    _fake_http(mp, network)
    command = ["preview", "--date", DATE, "--picks-dir", str(picks)]
    head_result = CliRunner().invoke(cli, command)
    mp.setattr(current, "load_decision_streak_state", base.load_decision_streak_state)
    base_result = CliRunner().invoke(cli, command)
    out["preview_cli"] = {"head_exit": head_result.exit_code, "head_error": repr(head_result.exception),
                            "base_exit": base_result.exit_code, "base_output": base_result.output}
    assert head_result.exit_code == 1 and isinstance(head_result.exception, AttributeError)
    assert base_result.exit_code == 0 and "Saved to" in base_result.output

(HERE / "state-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
