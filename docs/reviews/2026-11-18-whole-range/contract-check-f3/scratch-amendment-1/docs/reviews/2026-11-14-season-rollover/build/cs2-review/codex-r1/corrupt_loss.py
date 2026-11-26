"""Concrete loss: a current-season 12/used state is silently replaced by 1/available.

Use the same offline PYTHONPATH setup as probe.py. No pytest invocation.
"""
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
from bts import picks as p
from bts.cli import cli
from bts.health.streak_validation import check
from bts.health.runner import _safe_run
from bts.contest_state import load_decision_streak_state

ROOT = Path(__file__).resolve().parent
LABEL = sys.argv[1]
ET = ZoneInfo("America/New_York")
OUT = []
def outcome(fn):
    try:
        value = fn()
        return {"value": dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value}
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def pick(d, day, result):
    pk = p.Pick(batter_name="A", batter_id=1, team="NYY", lineup_position=1,
                pitcher_name="P", pitcher_id=2, p_game_hit=.85, flags=[],
                projected_lineup=False, game_pk=111, game_time=day + "T23:10:00Z")
    daily = p.DailyPick(date=day, run_time=day + "T20:00:00Z", pick=pk, double_down=None,
        runner_up=None, result=result, slot_results={"pick": result} if result else None,
        notification_sent=True, notification_channel="bluesky_dm", notification_id="probe")
    p.save_pick(daily, d)


for bad in (b"{", b"\xff", b"[]"):
    d = Path(tempfile.mkdtemp(prefix=LABEL + "-loss-", dir=ROOT / "runs"))
    for n in range(1, 13):
        pick(d, f"2027-05-{n:02}", "hit")
    pick(d, "2027-06-10", None)
    f = d / "streak.json"
    prior = {"season": 2027, "streak": 12, "saver_available": False, "applied_dates": ["2027-05-12"]}
    f.write_text(json.dumps(prior))
    original = f.read_bytes().hex()
    f.write_bytes(bad)
    # Older pick became unreadable too; nightly cannot restore the formerly tracked state.
    (d / "2027-05-01.json").write_text("{")
    with contextlib.redirect_stderr(io.StringIO()):
        before_health = _safe_run("streak_validation", lambda: check(d))
    with patch("bts.cli._now_et", return_value=datetime(2027, 6, 10, 23, tzinfo=ET)), \
         patch.object(p, "resolve_daily_slot_results", return_value={"pick": "hit"}):
        result = CliRunner().invoke(cli, ["check-results", "--date", "2027-06-10", "--picks-dir", str(d),
            "--shadow-status-output", str(d / "shadow-status.json")])
    after = f.read_bytes().hex()
    with patch.object(p, "resolve_daily_slot_results", side_effect=lambda daily, *a, **kw: daily.slot_results):
        night = outcome(lambda: p.reconcile_results(d, clock=lambda: datetime(2027, 6, 11, 2, tzinfo=ET)))
    with contextlib.redirect_stderr(io.StringIO()):
        after_health = _safe_run("streak_validation", lambda: check(d))
        decision = outcome(lambda: load_decision_streak_state(d, now=datetime(2027, 6, 11, 2, tzinfo=ET)))
    OUT.append({"fixture": str(d), "known_prior": prior, "original_hex": original, "corrupt_hex": bad.hex(),
        "health_before": [dataclasses.asdict(a) for a in before_health],
        "scorer": {"exit": result.exit_code, "error": type(result.exception).__name__ if result.exception else None,
                   "output": result.output}, "bytes_after": after, "bytes_nightly": f.read_bytes().hex(),
        "night": night, "health_after": [dataclasses.asdict(a) for a in after_health], "decision": decision})
(ROOT / (LABEL + "-corrupt-loss.json")).write_text(json.dumps(OUT, indent=2) + "\n")
print(LABEL, "recorded", len(OUT), "corrupt-state loss cases")
